import torch
import math
import clustering
from utils import check_tensor_1d


def entropy(p):
    p = torch.tensor(p)
    eps = 1e-16
    H = -torch.sum(p * torch.log(p + eps))
    return H


class BaseRecognizer(object):
    def __init__(self):
        super().__init__()
        self.representations = {}

    def get_classes(self):
        return list(self.representations.keys())

    def verify(self, class_id, x):
        raise NotImplementedError

    def verify_all(self, x):
        classes = list(self.representations.keys())
        scores = []
        for c in classes:
            s = self.verify(c, x)
            scores += [s.view(-1, 1)]
        scores = torch.cat(scores, dim=1)
        return scores

    def fit(self, X):
        raise NotImplementedError


class RecognizerCentroids(BaseRecognizer):
    def __init__(self, similarity_score):
        super().__init__()
        self.similarity_score = similarity_score

    def init_classes(self, class2emb):
        for (c, embs) in class2emb.items():
            X = torch.cat([e for e in embs if check_tensor_1d(e)])
            centroid = torch.mean(X, dim=0, keepdim=True)
            cardinality = X.shape[0]
            self.representations[c] = (centroid, cardinality)

    def verify(self, class_id, x):
        centroid, n = self.representations[class_id]
        score = self.similarity_score((centroid, n), x).view(-1)
        return score


class RecognizerMemory(BaseRecognizer):
    def __init__(self, similarity_score):
        super().__init__()
        self.similarity_score = similarity_score

    def init_classes(self, class2emb):
        for (c, embs) in class2emb.items():
            X = torch.cat([e.view(1, -1) for e in embs if check_tensor_1d(e)])
            self.representations[c] = X

    def verify(self, class_id, x):
        X = self.representations[class_id]
        score = self.similarity_score(X, x).view(-1)
        return score


class RecognizerCentroidsOnline(RecognizerCentroids):
    def __init__(self, similarity_score, threshold, alpha=None, **params):
        super().__init__(similarity_score)
        self.threshold = threshold  # enrichment threshold
        self.alpha = alpha  # exponential smoothing
        self.n_updates = {}
        self.params = params

    def init_classes(self, class2emb):
        super().init_classes(class2emb)
        self.n_updates = {c: 0 for c in self.representations}

    def verify(self, class_id, x):
        centroid, n = self.representations[class_id]
        score = self.similarity_score((centroid, n), x).view(-1)

        # Scoring model from https://hal.archives-ouvertes.fr/hal-01927584/document
        prior_unk = self.params.get("prior_unk", None)
        if prior_unk is not None:
            classes = list(self.representations.keys())
            idx = classes.index(class_id)
            n_classes = len(classes)
            scores = []
            for c in classes:
                centroid, n = self.representations[c]
                s = self.similarity_score((centroid, n), x).view(-1)
                scores += [s.view(-1, 1)]
            scores = torch.cat(scores, dim=1)  # (N, n_classes)
            scale = self.params["calibration"]["scale"]
            shift = self.params["calibration"]["shift"]
            scores_raw = (scores - shift) / scale
            mult, bias = [], []
            for k in range(len(classes)):
                if k == idx:
                    bias += [math.log(prior_unk)]
                    mult += [0.0]
                else:
                    bias += [math.log(1 - prior_unk) - math.log(n_classes - 1)]
                    mult += [1.0]
            bias = torch.tensor(bias).view(1, -1)
            mult = torch.tensor(mult).view(1, -1)
            denominator = mult * scores_raw + bias
            score = scores_raw[:, idx] - torch.logsumexp(denominator, dim=1)
            score = scale * score + shift
        return score

    def fit(self, X):

        classes = list(self.representations.keys())
        if X.dim() == 1:
            X = X.unsqueeze(0)

        for x in X:
            x = x.unsqueeze(0)
            scores = self.verify_all(x).view(-1)
            idx_max = torch.argmax(scores)
            s_max = scores[idx_max]
            c = classes[idx_max]
            if s_max > self.threshold:  # enrich
                centroid, n = self.representations[c]
                if self.alpha is None:
                    alpha = 1 / (n + 1)  # simple average
                    centroid = (1 - alpha) * centroid + alpha * x
                    self.representations[c] = (centroid, n + 1)
                else:
                    alpha = self.alpha
                    centroid = (1 - alpha) * centroid + alpha * x
                    # nn = n+1 #TODO: how to set counts in ths case?
                    Ha = entropy([alpha, 1 - alpha])
                    H = (
                        Ha * (1 - (1 - alpha) ** (self.n_updates[c] + 4)) / alpha
                    )  # +4 - ad hoc fix, assuming 3 enrolls!
                    nn = torch.exp(H)
                    self.representations[c] = (centroid, nn)
                self.n_updates[c] += 1
            else:  # impostor
                pass


class RecognizerCentroidsOnlineV2(RecognizerCentroidsOnline):
    def __init__(self, similarity_score, threshold, alpha):
        super().__init__(similarity_score, threshold, alpha)

    def verify(self, class_id, x):
        centroid, n = self.representations[class_id]
        score = self.similarity_score((centroid, n), (x, 1)).view(-1)
        return score


class RecognizerMemoryOnline(RecognizerMemory):
    def __init__(self, similarity_score, threshold, **params):
        super().__init__(similarity_score)
        self.threshold = threshold  # enrichment threshold
        self.n_updates = {}
        self.params = params

    def init_classes(self, class2emb):
        super().init_classes(class2emb)
        self.n_updates = {c: 0 for c in self.representations}

    def fit(self, X):

        classes = list(self.representations.keys())
        if X.dim() == 1:
            X = X.unsqueeze(0)

        for x in X:
            x = x.unsqueeze(0)
            scores = self.verify_all(x).view(-1)
            idx_max = torch.argmax(scores)
            s_max = scores[idx_max]
            c = classes[idx_max]
            if s_max > self.threshold:  # enrich
                X = self.representations[c]
                Xx = torch.cat([X, x])
                self.representations[c] = Xx
                self.n_updates[c] += 1
            else:  # impostor
                pass


class RecognizerCentroidsOffline(RecognizerCentroids):
    def __init__(self, similarity_score, algorithm="kmeans", threshold=None, **params):
        super().__init__(similarity_score)
        self.algorithm = algorithm
        self.threshold = threshold
        self.params = params

    def fit(self, X):
        classes, centroids, counts = [], [], []
        for c, (centroid, n) in self.representations.items():
            classes += [c]
            centroids += [centroid]
            counts += [n]
        centroids = torch.cat(centroids)
        labels = torch.arange(len(classes))

        if self.algorithm == "kmeans":
            labels_pred = clustering.kmeans_clustering(
                centroids, labels, X, self.similarity_score, self.threshold
            )
        else:
            raise NotImplementedError

        for i, c in enumerate(classes):
            x_extra = X[labels_pred == i]
            assert x_extra.dim() == 2
            centroid = centroids[i]
            n = counts[i]
            n_upd = n + x_extra.shape[0]  # TODO: check!
            centroid_upd = (
                centroid * n + torch.sum(x_extra, dim=0, keepdim=True)
            ) / n_upd
            # update class representations
            self.representations[c] = (
                centroid_upd,
                n_upd,
            )


class RecognizerMemoryOffline(RecognizerMemory):
    def __init__(self, similarity_score, algorithm="kmeans", threshold=None, **params):
        super().__init__(similarity_score)
        self.algorithm = algorithm
        self.threshold = torch.tensor(threshold).to(torch.get_default_dtype())
        self.params = params

    def fit(self, X):
        classes = []
        X_labeled = []
        labels = []
        for i, (c, x) in enumerate(self.representations.items()):
            classes += [c]
            X_labeled += [x]
            labels += [i * torch.ones(x.shape[0])]

        X_labeled = torch.cat(X_labeled)
        labels = torch.cat(labels)
        n_classes = len(self.representations)

        if self.algorithm == "kmeans":
            labels_pred = clustering.kmeans_clustering(
                X,
                n_classes,
                self.similarity_score,
                self.threshold,
                X_labeled,
                labels,
                n_iter=50,
            )
        elif self.algorithm in ["vb_plda_sph", "vb_plda_diag"]:
            b = self.params["plda"]["b"]
            w = self.params["plda"]["w"]
            Fa = self.params["Fa"]
            prob_outlier = torch.sigmoid(self.threshold)
            labels_pred = clustering.vb_plda_clustering(
                X, b, w, n_classes, X_labeled, labels, prob_outlier, n_iter=50, Fa=Fa
            )
        elif self.algorithm in ["ahc_plda_sph"]:
            b = self.params["plda"]["b"]
            w = self.params["plda"]["w"]
            scale = self.params["calibration"]["scale"]
            shift = self.params["calibration"]["shift"]
            threshold_plda = (self.threshold - shift) / scale
            labels_pred = clustering.ahc_plda_clustering_sph(
                X, b, w, threshold_plda, 1.0, 0.1, X_labeled, labels
            )
        elif self.algorithm in ["ahc_plda_diag"]:
            b = self.params["plda"]["b"]
            w = self.params["plda"]["w"]
            scale = self.params["calibration"]["scale"]
            shift = self.params["calibration"]["shift"]
            threshold_plda = (self.threshold - shift) / scale
            labels_pred = clustering.ahc_plda_clustering_diag(
                X, w, threshold_plda, 1.0, 0.1, X_labeled, labels
            )
        elif self.algorithm == "label_spread":
            scores = self.verify_all(X)
            scores_max, _ = torch.max(scores, dim=1)
            mask_known = scores_max > self.threshold
            labels_pred = -1 * torch.ones(X.shape[0])
            labels = clustering.label_spreading(X[mask_known], X_labeled, labels)
            labels_pred[mask_known] = labels
        else:
            raise NotImplementedError

        for i, c in enumerate(classes):
            x = self.representations[c]
            x_extra = X[labels_pred == i]
            assert x_extra.dim() == 2
            # update class representations
            self.representations[c] = torch.cat([x, x_extra])


class RecognizerCentroidsOfflineUnsupervised(RecognizerCentroids):
    pass


class RecognizerMemoryOfflineUnsupervised(RecognizerMemory):
    def __init__(self, similarity_score, algorithm="kmeans", threshold=None, **params):
        super().__init__(similarity_score)
        self.algorithm = algorithm
        self.threshold = threshold
        self.params = params

    def fit(self, X):

        n_classes = 20

        if self.algorithm == "kmeans":
            labels_pred = clustering.kmeans_clustering(
                X, n_classes, self.similarity_score, self.threshold, n_iter=50
            )
        elif self.algorithm in ["vb_plda_sph", "vb_plda_diag"]:
            b = self.params["plda"]["b"]
            w = self.params["plda"]["w"]
            Fa = self.params["Fa"]
            Fb = self.params["Fb"]
            prob_outlier = 0.0
            labels_pred = clustering.vb_plda_clustering(
                X, b, w, n_classes, prob_outlier, n_iter=50, Fa=Fa, Fb=Fb
            )
        elif self.algorithm in ["ahc_plda_sph"]:
            b = self.params["plda"]["b"]
            w = self.params["plda"]["w"]
            scale = self.params["calibration"]["scale"]
            shift = self.params["calibration"]["shift"]
            threshold_plda = (self.threshold - shift) / scale
            labels_pred = clustering.ahc_plda_clustering_sph(
                X, b, w, threshold_plda, 1.0, 0.1
            )
        elif self.algorithm in ["ahc_plda_diag"]:
            b = self.params["plda"]["b"]
            w = self.params["plda"]["w"]
            scale = self.params["calibration"]["scale"]
            shift = self.params["calibration"]["shift"]
            threshold_plda = (self.threshold - shift) / scale
            labels_pred = clustering.ahc_plda_clustering_diag(
                X, w, threshold_plda, 1.0, 0.1
            )
        elif self.algorithm == "ahc":
            distance_threshold = (-1.0) * self.threshold
            S = (-1.0) * torch.cat(
                [self.similarity_score(x.view(1, -1), X).view(1, -1) for x in X]
            ).t()
            labels_pred = clustering.ahc_clustering(
                S, "precomputed", distance_threshold
            )

        else:
            raise NotImplementedError

        class2emb = {}
        classes = torch.unique(labels_pred)
        for c in classes:
            mask = labels_pred == c
            x_cluster = X[mask]
            class2emb[c] = x_cluster

        self.init_classes(class2emb)
