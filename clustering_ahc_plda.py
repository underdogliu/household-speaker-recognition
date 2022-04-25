from collections import defaultdict

import numpy as np
from scipy.special import gammaln, psi
from numpy.random import choice


class CRP:
    def __init__(self, alpha, beta):
        assert alpha >= 0 <= beta <= 1
        self.alpha = alpha
        self.beta = beta

    def sample(self, n):
        alpha, beta = self.alpha, self.beta
        labels = np.zeros(n, int)  # restricted growth string, labels start at 0
        counts = np.zeros(n, int)  # table occupancies (up to n tables)
        p = np.empty(n)
        counts[0] = 1  # seat first customer at table 0
        nt = 1  # number of occupied tables
        for i in range(1, n):  # seat rest of customers
            # i is number of seated customers and index of to-be-seated customer
            pi = p[: nt + 1]
            pi[:nt] = (counts[:nt] - beta) / (i + alpha)  # occupied tables
            pi[nt] = (alpha + nt * beta) / (i + alpha)  # new table
            t = choice(nt + 1, None, True, pi)  # chosen table
            labels[i] = t
            counts[t] += 1
            if t == nt:
                nt += 1  # new table was chosen
        return labels, counts[:nt]

    def samples(self, n, m):
        """
        Sample m independent partitions of size n from this CRP
        Returns a list of m arrays of block sizes.
        (The array sizes are variable, depending on the number of blocks in
        the partition.)
        """
        assert n >= 1 <= m
        counts_list = []
        for i in range(m):
            labels, counts = self.sample(n)
            counts_list.append(counts)
        return counts_list

    def llr_joins(self, counts, i):
        """
        Let logLR(i,j) = log P(join(i,j)|crp) - log P( labels| crp), where
        labels is represented by the given occupancy counts; and where join(i,j)
        joins tables i and j, while leaving other tables as-is. A vector is
        returned with all logLR(i,j), with j > i.

        For use by AHC (agglomerative hierarchical clustering) algorithms that
        seek greedy MAP partitions, where this CRP forms the partition prior.
        """
        alpha, beta = self.alpha, self.beta
        K = len(counts)  # tables
        assert K > 1
        ci = counts[i]
        cj = counts[i + 1 :]
        llr = gammaln(1 - beta) - np.log(beta) - np.log(alpha / beta + K - 1)
        llr += gammaln(cj + (ci - beta)) - gammaln(ci - beta) - gammaln(cj - beta)
        return llr

    def exp_num_tables(self, n):
        """
        n: number of customers
        """
        alpha, beta = self.alpha, self.beta
        if alpha == 0 and beta == 0:
            e = 1
        elif alpha == np.Inf:
            e = n
        elif alpha > 0 and beta > 0:
            A = (
                gammaln(alpha + beta + n)
                + gammaln(alpha + 1)
                - np.log(beta)
                - gammaln(alpha + n)
                - gammaln(alpha + beta)
            )
            B = alpha / beta
            e = B * np.expm1(A - np.log(B))  # exp(A)-B
        elif alpha > 0 and beta == 0:
            e = alpha * (psi(n + alpha) - psi(alpha))
        elif alpha == 0 and beta > 0:
            A = gammaln(beta + n) - np.log(beta) - gammaln(n) - gammaln(beta)
            e = np.exp(A)
        return e

    def __repr__(self):
        return f"CRP(alpha={self.alpha}, beta={self.beta})"

    def ahc(self, labels):
        """
        Returns an AHC object, initialized at the given labels.

        For use by AHC (agglomerative hierarchical clustering) algorithms that
        seek greedy MAP partitions, where this CRP forms the partition prior.

        """
        return AHC(self, labels)


class AHC:
    """
    For use by AHC (agglomerative hierarchical clustering) algorithms that
    seek greedy MAP partitions, where this CRP forms the partition prior.
    """

    def __init__(self, crp, labels):
        self.crp = crp
        tables, counts = np.unique(labels, return_counts=True)
        self.counts = counts

    def llr_joins(self, i):
        """
        Scores in logLR form, the CRP prior's contribution when joining tables
        i with all tables j > i.
        """
        crp, counts = self.crp, self.counts
        return crp.llr_joins(counts, i)

    def join(self, i, j):
        """
        Joins tables i and j in this AHC object.
        """
        counts = self.counts
        counts[i] += counts[j]
        self.counts = np.delete(counts, j)


class SingletonDict(dict):
    def __getitem__(self, key):
        return super().__getitem__(key) if key in self else {key}


class OriginalAgglomerativeClusteringPLDA(object):
    def __init__(self, X, w_inv, alpha, beta, B=None):
        if B is not None:
            assert X.shape == B.shape
        n, d = X.shape

        prior = CRP(alpha, beta)

        self.n = self.N = n
        if B is None:
            self.R = R = np.tile(w_inv, (n, 1))
        else:
            self.R = R = (w_inv * B) / (w_inv + B)
        self.RX = RX = R * X  # (n,d)

        self.LLH = (RX**2 / (1.0 + R) - np.log1p(R)).sum(axis=1) / 2.0  # (n,)
        self.LLRs = []

        labels = np.arange(n, dtype=int)  # full length labels, contains result
        self.ind = labels.copy()  #

        self.prior_ahc = prior.ahc(labels)

        # map every element to a singleton cluster containing that element
        self.clusters = SingletonDict()

    def join(self, i, j):
        clusters = self.clusters
        join = clusters[i] | clusters[j]
        for e in join:
            clusters[e] = join

    def iteration(self, thr=0.0):
        RX, R, n = self.RX, self.R, self.n
        prior_ahc, LLH = self.prior_ahc, self.LLH
        ind = self.ind

        # M = np.full((n,n),-np.Inf)

        maxval = -np.Inf
        for i in range(n - 1):
            r = R[i, :]  # (d,)
            rR = r + R[i + 1 :, :]  # (n-i-1, d)
            rx = RX[i, :]
            rxRX = rx + RX[i + 1 :, :]
            llh = (rxRX**2 / (1.0 + rR) - np.log1p(rR)).sum(axis=1) / 2.0
            score = llh + prior_ahc.llr_joins(i) - LLH[i] - LLH[i + 1 :]
            # M[i,i+1:] = score
            j = score.argmax()
            scj = score[j]
            # print(i,i+j+1,': ',np.around(np.exp(scj),1))
            if scj > maxval:
                maxi = i
                maxj = j + i + 1
                maxval = scj

        # print(np.around(np.exp(M),1),'\n')
        LLRs = self.LLRs
        LLRs.append(maxval)

        if maxval > thr:
            # print('joining: ',maxi,'+',maxj)
            # print('ind = ',ind)
            ii, jj = ind[maxi], ind[maxj]
            # print('joining: ',ii,'+',jj)
            self.join(ii, jj)

            RX[maxi, :] += RX[maxj, :]
            R[maxi, :] += R[maxj, :]
            self.RX = np.delete(RX, maxj, axis=0)
            self.R = np.delete(R, maxj, axis=0)

            self.n = n - 1

            prior_ahc.join(maxi, maxj)

            LLH[maxi] = maxval + LLH[maxi] + LLH[maxj]
            self.LLH = np.delete(LLH, maxj)

            self.ind = np.delete(ind, maxj)

        return maxval

    def cluster(self, thr=0.0):
        while self.n > 1:
            llr = self.iteration(thr)
            if llr <= thr:
                break
        # return clusters2labels(self.clusters,self.N)
        return self.labelclusters()

    def labelclusters(self):
        clusters, n = self.clusters, self.N
        labels = np.full(n, -1)
        label = -1
        for i in range(n):
            s = clusters[i]
            for e in s:
                break  # get first set element
            if labels[e] < 0:
                label += 1
                labels[list(s)] = label
        return labels


class AgglomerativeClusteringPLDA(object):
    def __init__(self, X, w_inv, alpha=1.0, beta=0.1, X_labeled=(), labels_known=()):
        B = None
        n_samples, dim = X.shape
        n_labeled = len(labels_known)

        self.labels_known = labels_known

        prior = CRP(alpha, beta)

        self.n = self.N = n_samples + n_labeled
        if n_labeled > 0:
            X = np.r_[X_labeled, X]

        if B is None:
            self.R = R = np.tile(w_inv, (self.n, 1))
        else:
            self.R = R = (w_inv * B) / (w_inv + B)
        self.RX = RX = R * X  # (n,d)

        self.LLH = (RX**2 / (1.0 + R) - np.log1p(R)).sum(axis=1) / 2.0  # (n,)
        self.LLRs = []

        self.mask = np.ones((R.shape[0],)) > 0

        labels = np.arange(self.n, dtype=int)
        self.ind = labels.copy()  #

        self.prior_ahc = prior.ahc(labels)

        self.clusters = SingletonDict()
        self.cannot_link = defaultdict(list)

        self.merge_labeled(labels_known)

    def merge_labeled(self, labels_known):
        if len(labels_known) == 0:
            return

        n_samples = self.n - len(labels_known)

        # join labeled data into clusters
        # maxj > maxi

        labels_unknown = 1 + np.max(labels_known) + np.arange(n_samples, dtype=int)
        labels = np.r_[labels_known, labels_unknown]
        nn = len(labels)
        while np.max(np.unique(labels, return_counts=True)[1]) > 1:
            classes, counts = np.unique(labels, return_counts=True)
            for idx, count in enumerate(counts):
                if count > 1:
                    c = classes[idx]
                    mask = labels == c
                    idxs = np.arange(len(labels))[mask]
                    maxi = idxs[0]
                    maxj = idxs[1]
                    i = maxi
                    j = maxj - (i + 1)
                    scores = self.scores_llr_joins(i)
                    maxval = scores[j]
                    labels = np.delete(labels, maxj)
                    self.update(maxval, maxi, maxj)
                    break

        idxs = np.arange(len(labels_known), dtype=int)
        for c in np.unique(labels_known):
            mask = labels_known == c
            class_idxs = idxs[mask]
            rest_idxs = idxs[np.logical_not(mask)]
            for i in class_idxs:
                for j in rest_idxs:
                    self.cannot_link[i].append(j)

    def can_join(self, i, j):
        clusters = self.clusters
        allow = True
        for a in clusters[i]:
            for b in clusters[j]:
                if b in self.cannot_link[a] or a in self.cannot_link[b]:
                    allow = False
        return allow

    def join(self, i, j):
        clusters = self.clusters
        join = clusters[i] | clusters[j]
        for e in join:
            clusters[e] = join

    def scores_llr_joins(self, i):

        R, RX, LLH, _ = self.get_params()
        n = self.n
        prior_ahc = self.prior_ahc

        r = R[i, :]  # (d,)
        rR = r + R[i + 1 :, :]  # (n-i-1, d)
        rx = RX[i, :]
        rxRX = rx + RX[i + 1 :, :]
        llh = (rxRX**2 / (1.0 + rR) - np.log1p(rR)).sum(axis=1) / 2.0
        scores = llh + prior_ahc.llr_joins(i) - LLH[i] - LLH[i + 1 :]
        return scores

    def iteration(self, thr=0.0):
        n = self.n
        ind = self.ind
        scores_list = [-np.inf]
        pairs = [(0, 0)]

        for i in range(n - 1):
            scores = self.scores_llr_joins(i)

            scores_list += scores.tolist()
            pairs += [(i, j) for j in range(i + 1, n)]

        scores = np.array(scores_list)
        pairs = np.array(pairs)
        idx_sort = np.argsort(scores)[::-1]  # largest first
        scores = scores[idx_sort]
        pairs = pairs[idx_sort]

        for maxval, (maxi, maxj) in zip(scores, pairs):
            ii, jj = ind[maxi], ind[maxj]
            if self.can_join(ii, jj):

                LLRs = self.LLRs
                LLRs.append(maxval)

                if maxval > thr:
                    self.update(maxval, maxi, maxj)
                break
            else:
                pass

        return maxval

    def iteration_fast(self, thr=0.0):
        prior_ahc = self.prior_ahc

        R, RX, LLH, ind = self.get_params()

        d = R.shape[1]
        n = self.n

        rR_3d = R.reshape(n, 1, d) + R[1:, :].reshape(1, n - 1, d)
        rxRX_3d = RX.reshape(n, 1, d) + RX[1:, :].reshape(1, n - 1, d)
        LLH_2d = LLH.reshape(n, 1) + LLH[1:].reshape(1, n - 1)
        llh = (rxRX_3d**2 / (1.0 + rR_3d) - np.log1p(rR_3d)).sum(axis=-1) / 2.0

        idx_row, idx_col = np.triu_indices(n - 1)
        idx_ravel = np.ravel_multi_index((idx_row, idx_col), dims=(n, n - 1))
        import itertools

        prior_triu = [prior_ahc.llr_joins(i) for i in range(n - 1)]
        prior_prior_ahc = np.array(list(itertools.chain(*prior_triu)))
        scores = llh - LLH_2d
        scores = scores.ravel()[idx_ravel] + prior_prior_ahc
        pairs = np.array(list(zip(idx_row, idx_col + 1)))

        idx_sort = np.argsort(scores)[::-1]  # largest first
        scores = scores[idx_sort]
        pairs = pairs[idx_sort]

        for maxval, (maxi, maxj) in zip(scores, pairs):
            ii, jj = ind[maxi], ind[maxj]
            if self.can_join(ii, jj):

                LLRs = self.LLRs
                LLRs.append(maxval)

                if maxval > thr:
                    self.update(maxval, maxi, maxj)

                break
            else:
                pass
        return maxval

    def update(self, maxval, maxi, maxj):
        R, RX, LLH, ind = self.get_params()
        n = self.n

        ii, jj = ind[maxi], ind[maxj]
        self.join(ii, jj)

        RX[maxi, :] += RX[maxj, :]
        R[maxi, :] += R[maxj, :]

        self.n = n - 1

        self.prior_ahc.join(maxi, maxj)

        LLH[maxi] = maxval + LLH[maxi] + LLH[maxj]

        while not self.mask[maxj]:
            maxj += 1
        self.mask[maxj] = 0

    def get_params(self):
        mask = self.mask
        return self.R[mask, :], self.RX[mask, :], self.LLH[mask], self.ind[mask]

    def cluster(self, thr=0.0):
        while self.n > 1:
            llr = self.iteration_fast(thr)
            if llr <= thr:
                break
        return self.labelclusters()

    def labelclusters(self):
        clusters, n = self.clusters, self.N
        labels = np.full(n, -1)
        if len(self.labels_known) > 0:
            first_label = max(self.labels_known)
        else:
            first_label = -1
        label = first_label
        for i in range(n):
            s = clusters[i]
            e = min(s)
            if labels[e] < first_label + 1:
                if e < len(self.labels_known):
                    labels[list(s)] = self.labels_known[e]
                else:
                    label += 1
                    labels[list(s)] = label
        return labels[len(self.labels_known) :]
