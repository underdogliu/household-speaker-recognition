import numpy as np
from sklearn.metrics import jaccard_score, roc_curve
from scipy.optimize import brentq, linear_sum_assignment
from scipy.interpolate import interp1d


def eer(scores, labels, pos_label=1):
    far, tpr, thresholds = roc_curve(labels, scores, pos_label=pos_label)
    frr = 1 - tpr
    eer = brentq(lambda x: 1.0 - x - interp1d(far, tpr)(x), 0.0, 1.0)
    thresh = interp1d(far, thresholds)(eer)
    return eer, thresh


def far_at_frr(scores, labels, frr_point=1, pos_label=1):
    far, tpr, thresholds = roc_curve(labels, scores, pos_label=pos_label)
    # frr = 1 - tpr
    idx = np.searchsorted(tpr, 1 - (frr_point / 100))
    thresh = thresholds[idx]
    far_point = far[idx]
    return far_point * 100, thresh


def match_labels(y_hypothesis, y_reference, neg_label=-1, score="jaccard"):
    assert score in ["jaccard"]
    classes_pred = np.array([c for c in np.unique(y_hypothesis) if c != neg_label])
    classes_true = np.array([c for c in np.unique(y_reference) if c != neg_label])
    n_classes_pred = len(classes_pred)
    n_classes_true = len(classes_true)
    cost_matrix = np.zeros((n_classes_true, n_classes_pred))
    for i, c_true in enumerate(classes_true):
        for j, c_pred in enumerate(classes_pred):
            mask_pred = y_hypothesis == c_pred
            mask_true = y_reference == c_true
            if score == "jaccard":
                cost_matrix[i, j] = 1 - jaccard_score(mask_true, mask_pred)
            else:
                raise NotImplementedError
    true_inds, pred_inds = linear_sum_assignment(cost_matrix)
    classes_pred_new = np.ones_like(classes_true) * np.nan
    for i, idx in enumerate(pred_inds):
        classes_pred_new[i] = classes_pred[idx]
    return classes_true, classes_pred_new


def compute_jer(y_hypothesis, y_reference, neg_label=-1, return_individual=False):
    classes_pred = [c for c in np.unique(y_hypothesis) if c != neg_label]
    classes_true = [c for c in np.unique(y_reference) if c != neg_label]
    n_classes_pred = len(classes_pred)
    n_classes_true = len(classes_true)
    cost_matrix = np.zeros((n_classes_true, n_classes_pred))
    for i, c_true in enumerate(classes_true):
        for j, c_pred in enumerate(classes_pred):
            mask_pred = y_hypothesis == c_pred
            mask_true = y_reference == c_true
            cost_matrix[i, j] = 1 - jaccard_score(mask_true, mask_pred)
    true_inds, pred_inds = linear_sum_assignment(cost_matrix)
    jers = np.ones(n_classes_true)
    for (i, j) in zip(true_inds, pred_inds):
        jers[i] = cost_matrix[i, j]
    if return_individual:
        return np.array(jers) * 100
    else:
        return np.mean(jers) * 100


def compute_jer_v2(y_hypothesis, y_reference, neg_label=-1, return_individual=False):
    classes_true, classes_pred = match_labels(
        y_hypothesis, y_reference, neg_label=neg_label, score="jaccard"
    )

    jers = []
    for c_true, c_pred in zip(classes_true, classes_pred):
        mask_pred = y_hypothesis == c_pred
        mask_true = y_reference == c_true
        jer = 1 - jaccard_score(mask_true, mask_pred)
        jers += [jer]
    if return_individual:
        return np.array(jers) * 100
    else:
        return np.mean(jers) * 100
