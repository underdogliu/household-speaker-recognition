import torch
import torch.nn.functional as F


def train_plda_sph_mle(X, y):

    device = X.device

    X_norm = F.normalize(X, dim=1)

    classes, yy = torch.unique(y, return_inverse=True)
    I = torch.eye(len(classes)).to(device)
    Y = I[yy]
    means = (
        torch.mm(Y.t(), X_norm) / torch.sum(Y, dim=0, keepdim=True).t()
    )  # TODO: class_means
    X_means = means[yy]
    X_c = X_norm - X_means

    w = torch.mean(X_c ** 2)
    b = torch.mean(means ** 2)
    return b, w
