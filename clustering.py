import math
import numpy as np
import torch

from clustering_kmeans import kmeans_clustering
from clustering_vb_plda import vb_plda_clustering
from clustering_ahc_plda import (
    AgglomerativeClusteringPLDA,
    OriginalAgglomerativeClusteringPLDA,
)

from sklearn.semi_supervised import LabelSpreading
from sklearn.cluster import KMeans, AgglomerativeClustering

empty = torch.tensor([])


# def vb_plda_clustering_sph(X, labels, X_unlabeled, b, w, threshold):
#     return vb_plda_clustering(X, b, w, max_classes, X_labeled=empty, labels=empty, prob_outlier=0, n_iter=20)
#
# def vb_plda_clustering_diag(X, labels, X_unlabeled, w, threshold):
#     return vb_plda_clustering(X, labels, X_unlabeled, w, threshold)
#


def ahc_clustering(X, distance_fn, distance_threshold):
    X = X.cpu().numpy()
    cluster_model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        compute_full_tree=True,
        affinity=distance_fn,
        linkage="complete",
    )
    y_pred = cluster_model.fit_predict(X)
    return torch.tensor(y_pred)


def ahc_plda_clustering_sph(
    X, b, w, threshold, alpha=1.0, beta=0.1, X_labeled=(), labels=()
):

    b, w = float(b), float(w)

    n_labeled = len(labels)

    labels = np.array(labels)

    N, dim = X.shape

    threshold_ahc = threshold  # = 0

    X_tr = X.cpu().numpy() / math.sqrt(b)
    X_labeled_tr = X_labeled.cpu().numpy() / math.sqrt(b) if n_labeled > 0 else ()

    w_inv = 1 / w
    w_inv_tr = w_inv * b

    w_inv_diag = np.full(dim, w_inv_tr)

    np.random.seed(42)
    ahc = AgglomerativeClusteringPLDA(
        X_tr, w_inv_diag, alpha, beta, X_labeled_tr, labels
    )
    y_pred = ahc.cluster(threshold_ahc)

    if n_labeled > 0:
        classes_known = np.unique(labels)
        for c in np.unique(y_pred):
            if not c in classes_known:
                y_pred[y_pred == c] = -1

    return torch.tensor(y_pred)


def ahc_plda_clustering_diag(
    X, w, threshold, alpha=1.0, beta=0.1, X_labeled=(), labels=()
):

    N, dim = X.shape
    assert w.numel() == dim

    w = w.cpu().numpy()

    n_labeled = len(labels)

    labels = np.array(labels)

    threshold_ahc = threshold  # = 0

    X = X.cpu().numpy()
    X_labeled = X_labeled.cpu().numpy() if n_labeled > 0 else ()

    w_inv = 1 / w
    w_inv_diag = w_inv

    np.random.seed(42)
    ahc = AgglomerativeClusteringPLDA(X, w_inv_diag, alpha, beta, X_labeled, labels)
    y_pred = ahc.cluster(threshold_ahc)

    if n_labeled > 0:
        classes_known = np.unique(labels)
        for c in np.unique(y_pred):
            if not c in classes_known:
                y_pred[y_pred == c] = -1

    return torch.tensor(y_pred)


def label_spreading(X, X_labeled, labels):

    N = X.shape[0]
    n_labeled = X_labeled.shape[0]

    y = torch.cat([labels, -1 * torch.ones(N)]).cpu().numpy()
    X = torch.cat([X_labeled, X]).cpu().numpy()

    label_spread = LabelSpreading(kernel="knn")
    # label_spread = LabelSpreading(kernel="rbf", alpha=0.1, gamma=10)
    label_spread.fit(X, y)
    output_labels = label_spread.transduction_
    return torch.tensor(output_labels[n_labeled:])


if __name__ == "__main__":

    from scoring_impl import cosine_similarity

    T = lambda x: torch.tensor(x).to(torch.get_default_dtype())

    seed = 42

    # k-means

    np.random.seed(1)

    N = np.random.randint(100) * 5
    dim = np.random.randint(100)
    X = np.random.randn(N, dim)
    threshold = float(np.random.rand()) * 10

    # X_labeled = ()
    # labels = ()

    X_labeled = np.random.randn(10, dim)
    labels = np.random.randint(2, size=10)

    max_classes = 4

    # UNSUPERVISED

    similarity_score = cosine_similarity
    y_pred = kmeans_clustering(
        T(X),
        max_classes,
        similarity_score,
        threshold=-1e33,
        X_labeled=T(X_labeled),
        labels=T(labels),
        n_iter=20,
    )

    if False:
        L = np.random.randn(dim, dim)
        B = L @ L.T
        L = np.random.randn(dim, dim)
        W = L @ L.T

        from scipy.linalg import eigh

        w_vec, Tr = eigh(W, B)

        B_tr = Tr.T @ B @ Tr
        W_tr = Tr.T @ W @ Tr
        X_tr = X @ Tr

        prob_outlier = 0.5

        # VB-full w/o outliers
        y_pred_full = vb_plda_clustering(
            T(X),
            B,
            W,
            max_classes,
            X_labeled=T(X_labeled),
            labels=T(labels),
            prob_outlier=prob_outlier,
            n_iter=20,
        )

        # VB-full w/o outliers transformed space
        y_pred_full_tr = vb_plda_clustering(
            T(X_tr),
            B_tr,
            W_tr,
            max_classes,
            X_labeled=T(X_labeled),
            labels=T(labels),
            prob_outlier=prob_outlier,
            n_iter=20,
        )
        y_pred_full_tr_2 = vb_plda_clustering(
            T(X_tr),
            1,
            w_vec,
            max_classes,
            X_labeled=T(X_labeled),
            labels=T(labels),
            prob_outlier=prob_outlier,
            n_iter=20,
        )

        diff = torch.mean((y_pred_full != y_pred_full_tr).to(torch.get_default_dtype()))
        print(diff)
        diff = torch.mean(
            (y_pred_full != y_pred_full_tr_2).to(torch.get_default_dtype())
        )
        print(diff)

        exit()

    b = np.random.rand() * 10
    w = np.random.rand() * 0.1
    B = np.eye(dim) * b
    W = np.eye(dim) * w

    # VB-sph w/o outliers
    y_pred_sph = vb_plda_clustering(
        T(X),
        b,
        w,
        max_classes,
        X_labeled=T(X_labeled),
        labels=T(labels),
        prob_outlier=0,
        n_iter=30,
    )
    print("\n---------------------\n")
    y_pred_sph_2 = vb_plda_clustering(
        T(X / np.sqrt(b)),
        1,
        w / b,
        max_classes,
        X_labeled=T(X_labeled / np.sqrt(b)),
        labels=T(labels),
        prob_outlier=0,
        n_iter=30,
    )

    # VB-sph w/ outliers
    y_pred_sph_out = vb_plda_clustering(
        T(X),
        b,
        w,
        max_classes,
        X_labeled=T(X_labeled),
        labels=T(labels),
        prob_outlier=0.5,
        n_iter=20,
    )

    # VB-diag w/o outliers
    from scipy.linalg import eigh

    w_vec, Tr = eigh(W, B)
    B_tr = Tr.T @ B @ Tr
    W_tr = Tr.T @ W @ Tr
    X_tr = X @ Tr
    if len(labels) > 0:
        X_labeled_tr = X_labeled @ Tr

    print(np.mean((X / np.sqrt(b) - X_tr) ** 2))

    # w_diag = np.ones((dim,)) * w / b
    # y_pred_diag = vb_plda_clustering(T(X / np.sqrt(b)), 1, w_diag, max_classes, X_labeled=T(X_labeled), labels=T(labels), prob_outlier=0, n_iter=20)
    #
    # # VB-diag w/ outliers
    # y_pred_diag_out = vb_plda_clustering(T(X / np.sqrt(b)), 1, w_diag, max_classes, X_labeled=T(X_labeled), labels=T(labels), prob_outlier=0.5, n_iter=20)

    y_pred_diag = vb_plda_clustering(
        T(X_tr),
        1,
        w_vec,
        max_classes,
        X_labeled=T(X_labeled_tr),
        labels=T(labels),
        prob_outlier=0,
        n_iter=20,
    )

    # VB-diag w/ outliers
    y_pred_diag_out = vb_plda_clustering(
        T(X_tr),
        1,
        w_vec,
        max_classes,
        X_labeled=T(X_labeled_tr),
        labels=T(labels),
        prob_outlier=0.5,
        n_iter=20,
    )

    # VB-full w/o outliers
    y_pred_full = vb_plda_clustering(
        T(X),
        B,
        W,
        max_classes,
        X_labeled=T(X_labeled),
        labels=T(labels),
        prob_outlier=0,
        n_iter=20,
    )
    y_pred_full_2 = vb_plda_clustering(
        T(X_tr),
        B_tr,
        W_tr,
        max_classes,
        X_labeled=T(X_labeled_tr),
        labels=T(labels),
        prob_outlier=0,
        n_iter=20,
    )

    # VB-full w/ outliers
    y_pred_full_out = vb_plda_clustering(
        T(X),
        B,
        W,
        max_classes,
        X_labeled=T(X_labeled),
        labels=T(labels),
        prob_outlier=0.5,
        n_iter=20,
    )
    y_pred_full_out_2 = vb_plda_clustering(
        T(X_tr),
        B_tr,
        W_tr,
        max_classes,
        X_labeled=T(X_labeled_tr),
        labels=T(labels),
        prob_outlier=0.5,
        n_iter=20,
    )

    diff = torch.mean((y_pred_sph != y_pred_sph_2).to(torch.get_default_dtype()))
    print(diff)

    diff = torch.mean((y_pred_sph != y_pred_diag).to(torch.get_default_dtype()))
    print(diff)
    diff = torch.mean((y_pred_full != y_pred_sph).to(torch.get_default_dtype()))
    print(diff)
    diff = torch.mean((y_pred_full != y_pred_diag).to(torch.get_default_dtype()))
    print(diff)

    diff = torch.mean((y_pred_sph_out != y_pred_diag_out).to(torch.get_default_dtype()))
    print(diff)
    diff = torch.mean((y_pred_full_out != y_pred_sph_out).to(torch.get_default_dtype()))
    print(diff)

    diff = torch.mean((y_pred_full != y_pred_full_2).to(torch.get_default_dtype()))
    print("\n", diff)
    diff = torch.mean(
        (y_pred_full_out != y_pred_full_out_2).to(torch.get_default_dtype())
    )
    print(diff)

    exit()
    print("-------------------------------------")

    # Let's compare AgglomerativeClusteringPLDA, OriginalAgglomerativeClusteringPLDA

    for i in range(10):

        np.random.seed(i)
        N = np.random.randint(100)
        dim = np.random.randint(100)

        X = np.random.randn(N, dim)

        threshold_ahc = float(np.random.rand()) * 10
        alpha = 1.0
        beta = 0.1

        w_inv_diag = np.random.rand(dim) * 10

        np.random.seed(42)
        ahc = OriginalAgglomerativeClusteringPLDA(X, w_inv_diag, alpha, beta)
        y_pred_orig = ahc.cluster(threshold_ahc)

        np.random.seed(42)
        y_pred = (
            ahc_plda_clustering_diag(
                T(X), T(1 / w_inv_diag), threshold_ahc, alpha, beta
            )
            .cpu()
            .numpy()
        )

        diff = np.mean(y_pred_orig != y_pred) * 100
        print(diff)

    print("-------------------------------------")

    # Let's compare AgglomerativeClusteringPLDA with diagonal and spherical covariances

    for i in range(10):

        np.random.seed(i)
        N = np.random.randint(100)
        dim = np.random.randint(100)

        X = np.random.randn(N, dim)

        threshold_ahc = float(np.random.rand()) * 10
        alpha = 1.0
        beta = 0.1

        w_inv = np.random.rand() * 10
        w_inv_diag = np.ones((dim,)) * w_inv

        np.random.seed(42)
        y_pred_sph = (
            ahc_plda_clustering_sph(T(X), 1, 1 / w_inv, threshold_ahc, alpha, beta)
            .cpu()
            .numpy()
        )

        np.random.seed(42)
        y_pred_diag = (
            ahc_plda_clustering_diag(
                T(X), T(1 / w_inv_diag), threshold_ahc, alpha, beta
            )
            .cpu()
            .numpy()
        )

        diff = np.mean(y_pred_diag != y_pred_sph) * 100
        print(diff)
