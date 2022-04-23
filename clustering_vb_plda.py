import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

empty = torch.tensor([])


def mahalanobis_distance(X, Y, M):
    quad_term1 = torch.sum((X @ M) * X, dim=1, keepdim=True)
    quad_term2 = torch.sum((Y @ M) * Y, dim=1, keepdim=True).t()
    cross_term = 2 * (X @ M) @ Y.t()
    dist = quad_term1 + quad_term2 - cross_term
    return dist


def euclidean_distance(X, Y):
    quad_term1 = torch.sum(X * X, dim=1, keepdim=True)
    quad_term2 = torch.sum(Y * Y, dim=1, keepdim=True).t()
    cross_term = 2 * X @ Y.t()
    dist = quad_term1 + quad_term2 - cross_term
    return dist


def vb_plda_clustering(
    X,
    B,
    W,
    max_classes,
    X_labeled=empty,
    labels=empty,
    prob_outlier=0,
    n_iter=20,
    Fa=1.0,
    Fb=1.0,
    seed=0,
):
    torch.manual_seed(seed)

    device = X.device

    B, W = torch.tensor(B).to(device), torch.tensor(W).to(device)

    n_labeled = len(labels)
    X = torch.cat([X_labeled, X]) if n_labeled > 0 else X

    n_samples, dim = X.shape
    semi_supervised = True if n_labeled > 0 else False
    background_class = True if prob_outlier > 0 else False

    posteriors = torch.rand(n_samples, max_classes).to(device)
    # posteriors = torch.distributions.gamma.Gamma(torch.tensor([1.0]), torch.tensor([1.0])).sample((n_samples, max_classes)).squeeze(-1)
    # rnd_vec = torch.randn(max_classes, dim)
    # from scoring_impl import cosine_similarity
    # posteriors = torch.exp(cosine_similarity(X, rnd_vec))

    if background_class:
        posteriors = torch.cat([posteriors, torch.ones(n_samples, 1).to(device)], dim=1)
    posteriors = posteriors / torch.sum(posteriors, dim=1, keepdim=True)

    if semi_supervised:
        classes, y = torch.unique(labels, return_inverse=True)
        n_classes = len(classes)
        I = torch.eye(n_classes).to(device)
        labels_onehot = I[y]  # one-hot encoding
        posteriors[:n_labeled, :n_classes] = labels_onehot
        posteriors[:n_labeled, n_classes:] *= 0
    else:
        labels_onehot = ()

    if background_class:
        prior = torch.ones(max_classes + 1)
        prior[-1] = prob_outlier
        prior[:max_classes] = (1 - prob_outlier) / max_classes
    else:
        prior = torch.ones(max_classes) / max_classes

    prior = prior.to(device)
    assert torch.isclose(prior.sum(), torch.tensor([1.0]).to(device))

    if W.numel() == 1:
        posteriors = vb_plda_inference_sph(
            X,
            labels_onehot,
            B,
            W,
            posteriors,
            prior,
            background_class,
            n_iter=n_iter,
            Fa=Fa,
            Fb=Fb,
        )
    elif W.dim() == 2 and W.numel() > 1:
        assert W.shape[0] == W.shape[1]
        posteriors = vb_plda_inference_full(
            X,
            labels_onehot,
            B,
            W,
            posteriors,
            prior,
            background_class,
            n_iter=n_iter,
            Fa=Fa,
            Fb=Fb,
        )
    elif W.numel() > 1:
        posteriors = vb_plda_inference_diag(
            X,
            labels_onehot,
            B,
            W,
            posteriors,
            prior,
            background_class,
            n_iter=n_iter,
            Fa=Fa,
            Fb=Fb,
        )

    labels_pred = torch.argmax(posteriors[n_labeled:, :], dim=1)
    if prob_outlier > 0:
        mask = labels_pred >= max_classes
        labels_pred[mask] = -1
    return labels_pred


def vb_plda_inference_full(
    X, Y, B, W, post, prior, background_class, n_iter=20, Fa=1.0, Fb=1.0
):
    semi_supervised = True if len(Y) > 0 else False
    max_classes = post.shape[1]
    if background_class:
        max_classes = max_classes - 1
    if semi_supervised:
        n_labeled, n_classes = Y.shape
    else:
        n_labeled, n_classes = 0, max_classes

    n_samples, dim = X.shape
    # mu = torch.zeros(max_classes, dim)
    # Sigma = torch.zeros(max_classes, dim, dim)

    B_inv = torch.linalg.inv(B)
    W_inv = torch.linalg.inv(W)
    B_plus_W_inv = torch.linalg.inv(B + W)

    const = -0.5 * dim * math.log(2 * math.pi)

    log_prior = torch.log(prior)
    if background_class:
        log_post_outlier = (
            -0.5 * torch.sum((X @ B_plus_W_inv) * X, dim=1, keepdim=True)
            + 0.5 * torch.logdet(B_plus_W_inv)
            + const
        )  # outlier
        # print(log_post_outlier)
    for _ in range(n_iter):

        post[n_labeled:, :] *= Fa / Fb

        # update clusters
        # for k in range(max_classes):
        #     p = post[:, k].unsqueeze(1)
        #     Sigma[k] = torch.linalg.inv(B_inv + torch.sum(p) * W_inv)
        #     mu[k] = Sigma[k] @ W_inv @ torch.sum(p * X, dim=0)
        counts = post[:, :max_classes].sum(0).view(-1, 1, 1)
        Sigma = torch.linalg.inv(B_inv + counts * W_inv)
        mu = (
            torch.matmul(Sigma, W_inv)
            @ torch.mm(post[:, :max_classes].t(), X).unsqueeze(-1)
        ).squeeze()

        # update assignments
        log_post = -0.5 * mahalanobis_distance(X, mu, W_inv) + 0.5 * torch.logdet(W_inv)
        for k in range(max_classes):
            log_post[:, k] += (
                -0.5 * torch.trace(W_inv @ Sigma[k]) + const
            )  # TODO: batch

        if background_class:
            log_post = torch.cat([log_post, log_post_outlier], 1)
        log_post = Fa * log_post + log_prior.view(1, -1)
        post = torch.exp(log_post - torch.logsumexp(log_post, dim=1, keepdim=True))

        # correct assignments for the labeled part
        if semi_supervised:
            post[:n_labeled, :n_classes] = Y
            post[:n_labeled, n_classes:] *= 0

        # update prior
        log_counts = torch.log(post.sum(0))
        log_prior = log_counts - torch.logsumexp(log_counts, dim=0)

    return post


def vb_plda_inference_diag(
    X, Y, eye, w_diag, post, prior, background_class, n_iter=20, Fa=1.0, Fb=1.0
):
    semi_supervised = True if len(Y) > 0 else False
    max_classes = post.shape[1]
    if background_class:
        max_classes = max_classes - 1
    if semi_supervised:
        n_labeled, n_classes = Y.shape
    else:
        n_labeled, n_classes = 0, max_classes

    n_samples, dim = X.shape

    w = w_diag
    w = w.view(1, -1)
    w_inv = 1 / w
    b_plus_w_inv = 1 / (1 + w)

    const = -0.5 * dim * math.log(2 * math.pi)

    log_prior = torch.log(prior)
    if background_class:
        log_post_outlier = (
            -0.5 * torch.sum(X ** 2 * b_plus_w_inv.view(1, -1), dim=1, keepdim=True)
            + 0.5 * torch.sum(torch.log(b_plus_w_inv))
            + const
        )
        # B_plus_W_inv = torch.linalg.inv(torch.eye(dim) + torch.diag(w_diag))
        # log_post_outlier = -0.5 * torch.sum((X @ B_plus_W_inv) * X, dim=1, keepdim=True) \
        #                    + 0.5 * torch.logdet(B_plus_W_inv) + const  # outlier

        # print(log_post_outlier)
    for _ in range(n_iter):

        post[n_labeled:, :] *= Fa / Fb

        # update clusters
        counts = torch.sum(post[:, :max_classes], dim=0, keepdim=True)
        Sigma = 1 / (1 + torch.mm(counts.t(), w_inv))
        mu = Sigma * w_inv.view(1, -1) * torch.mm(post[:, :max_classes].t(), X)

        # update assignments
        log_post = -0.5 * mahalanobis_distance(
            X, mu, torch.diag(w_inv.view(-1))
        ) + 0.5 * torch.sum(torch.log(w_inv))
        log_post += -0.5 * torch.sum(Sigma * w_inv.view(1, -1), dim=1) + const

        if background_class:
            log_post = torch.cat([log_post, log_post_outlier], 1)
        log_post = Fa * log_post + log_prior.view(1, -1)
        post = torch.exp(log_post - torch.logsumexp(log_post, dim=1, keepdim=True))

        # correct assignments for the labeled part
        if semi_supervised:
            post[:n_labeled, :n_classes] = Y
            post[:n_labeled, n_classes:] *= 0

        # update prior
        log_counts = torch.log(post.sum(0))
        log_prior = log_counts - torch.logsumexp(log_counts, dim=0)

    return post


def vb_plda_inference_sph(
    X, Y, b, w, post, prior, background_class, n_iter=20, Fa=1.0, Fb=1.0
):
    semi_supervised = True if len(Y) > 0 else False
    max_classes = post.shape[1]
    if background_class:
        max_classes = max_classes - 1
    if semi_supervised:
        n_labeled, n_classes = Y.shape
    else:
        n_labeled, n_classes = 0, max_classes

    n_samples, dim = X.shape

    b_inv = 1 / b
    w_inv = 1 / w
    b_plus_w_inv = 1 / (b + w)

    const = -0.5 * dim * math.log(2 * math.pi)

    log_prior = torch.log(prior)
    if background_class:
        log_post_outlier = (
            -0.5 * b_plus_w_inv * torch.sum(X ** 2, dim=1, keepdim=True)
            + 0.5 * dim * torch.log(b_plus_w_inv)
            + const
        )  # outlier
        # print(log_post_outlier)
    for _ in range(n_iter):

        post[n_labeled:, :] *= Fa / Fb

        # update clusters
        counts = torch.sum(post[:, :max_classes], dim=0)
        Sigma = 1 / (b_inv + counts * w_inv * 1)
        mu = w_inv * 1 * Sigma.view(-1, 1) * torch.mm(post[:, :max_classes].t(), X)

        # update assignments
        log_post = -0.5 * w_inv * euclidean_distance(X, mu) + 0.5 * dim * torch.log(
            w_inv
        )
        log_post += -0.5 * dim * w_inv * Sigma + const

        if background_class:
            log_post = torch.cat([log_post, log_post_outlier], 1)
        log_post = Fa * log_post + log_prior.view(1, -1)
        post = torch.exp(log_post - torch.logsumexp(log_post, dim=1, keepdim=True))

        # correct assignments for the labeled part
        if semi_supervised:
            post[:n_labeled, :n_classes] = Y
            post[:n_labeled, n_classes:] *= 0

        # update prior
        log_counts = torch.log(post.sum(0))
        log_prior = log_counts - torch.logsumexp(log_counts, dim=0)

    return post
