import torch

empty = torch.tensor([])


def class_means(X, y):
    classes, y = torch.unique(y, return_inverse=True)
    counts = torch.bincount(y)
    means = torch.zeros(len(classes), X.shape[1]).to(X.dtype)
    means.index_add_(0, y, X)
    means /= counts.view(-1, 1)
    return means, classes


def kmeans_clustering(
    X,
    max_classes,
    similarity_score,
    threshold=-1e33,
    X_labeled=empty,
    labels=empty,
    n_iter=20,
):
    n_labeled = len(labels)
    X = torch.cat([X_labeled, X]) if n_labeled > 0 else X

    n_samples, dim = X.shape
    semi_supervised = True if n_labeled > 0 else False
    means = torch.randn(max_classes, dim).to(X.dtype)

    if semi_supervised:
        classes, y = torch.unique(labels, return_inverse=True)
        n_classes = len(classes)
        means_labeled, classes_labeled = class_means(X_labeled, y)
        means[classes_labeled] = means_labeled

    # k-means with known classes and outliers
    for iteration in range(n_iter):

        scores = torch.cat(
            [similarity_score(m.view(1, -1), X).view(1, -1) for m in means]
        ).t()

        scores_max, labels_pred = torch.max(scores, dim=1)
        if semi_supervised:
            labels_pred[:n_labeled] = y
        mask_known = scores_max > threshold
        mask_known[:n_labeled] = 1
        labels_pred[torch.logical_not(mask_known)] = -1

        means_pred, classes_pred = class_means(X[mask_known], labels_pred[mask_known])
        means[classes_pred] = means_pred

    return labels_pred[n_labeled:]
