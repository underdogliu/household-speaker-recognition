import torch
import torch.nn.functional as F


def cosine_similarity(a, b, numpy_input=False):
    if numpy_input:
        a, b = torch.tensor(a), torch.tensor(b)
    a = a / torch.norm(a, dim=-1, keepdim=True)
    b = b / torch.norm(b, dim=-1, keepdim=True)
    if numpy_input:
        return torch.matmul(a, b.t()).cpu().numpy()
    else:
        return torch.matmul(a, b.t())


def score_plda_sph_centroids(
    data_enroll, data_test, b, w, by_the_book=True, len_norm=False
):
    b, w = torch.tensor(b), torch.tensor(w)

    if isinstance(data_enroll, (tuple, list)):
        c_e, n_e = data_enroll
    else:
        n_e = data_enroll.shape[0]
        c_e = torch.mean(data_enroll, dim=0, keepdim=True)

    if isinstance(data_test, (tuple, list)):
        c_t, n_t = data_test
    else:
        # batch (multi-trial) scoring
        c_t = data_test
        n_t = 1

    if not by_the_book:
        n_e = 1
        n_t = 1

    if len_norm:
        c_e = F.normalize(c_e, dim=1)
        c_t = F.normalize(c_t, dim=1)

    dim = c_e.shape[1]

    b_inv = 1 / b
    w_inv = 1 / w

    a_e = n_e * c_e * w_inv
    a_t = n_t * c_t * w_inv

    sigma_e_inv = b_inv + n_e * w_inv
    sigma_t_inv = b_inv + n_t * w_inv
    sigma_e = 1 / sigma_e_inv
    sigma_t = 1 / sigma_t_inv
    sigma_inv_sum = sigma_e_inv + sigma_t_inv - b_inv

    a = a_e + a_t
    mu_quad_term = -0.5 * sigma_e * torch.sum(
        a_e * a_e, dim=1
    ) - 0.5 * sigma_t * torch.sum(a_t * a_t, dim=1)
    const = dim * (
        0.5 * torch.log(b)
        + 0.5 * torch.log(sigma_e_inv)
        + 0.5 * torch.log(sigma_t_inv)
        - 0.5 * torch.log(sigma_inv_sum)
    )
    score = 0.5 / sigma_inv_sum * torch.sum(a * a, dim=1) + mu_quad_term + const
    return score


# TODO: maybe merge plda_sph and plda_diag function into a single one?
def score_plda_diag_centroids(
    data_enroll, data_test, w_diag, by_the_book=True, len_norm=False
):
    if isinstance(data_enroll, (tuple, list)):
        c_e, n_e = data_enroll
    else:
        n_e = data_enroll.shape[0]
        c_e = torch.mean(data_enroll, dim=0, keepdim=True)

    if isinstance(data_test, (tuple, list)):
        c_t, n_t = data_test
    else:
        # batch (multi-trial) scoring
        c_t = data_test
        n_t = 1

    if not by_the_book:
        n_e = 1
        n_t = 1

    if len_norm:
        c_e = F.normalize(c_e, dim=1)
        c_t = F.normalize(c_t, dim=1)

    # dim = c_e.shape[1]

    w_diag_inv = 1 / w_diag.view(1, -1)

    a_e = n_e * c_e * w_diag_inv
    a_t = n_t * c_t * w_diag_inv

    sigma_e_inv = 1 + n_e * w_diag_inv
    sigma_t_inv = 1 + n_t * w_diag_inv
    sigma_e = 1 / sigma_e_inv
    sigma_t = 1 / sigma_t_inv
    lmbda = sigma_e_inv + sigma_t_inv - 1

    a = a_e + a_t
    mu_quad_term = -0.5 * torch.sum(a_e ** 2 * sigma_e) - 0.5 * torch.sum(
        a_t ** 2 * sigma_t, dim=1
    )
    const = (
        0.5 * torch.sum(torch.log(sigma_e_inv))
        + 0.5 * torch.sum(torch.log(sigma_t_inv))
        - 0.5 * torch.sum(torch.log(lmbda))
    )
    return 0.5 * torch.sum(a ** 2 / lmbda, dim=1) + mu_quad_term + const


def score_cosine_embeddings_averaging(X1, X2):
    centroid = torch.mean(X1, dim=0, keepdim=True)
    score = cosine_similarity(centroid, X2)  # .squeeze(0)
    return score


def score_cosine_scores_averaging(X1, X2):
    score = torch.mean(cosine_similarity(X1, X2), dim=0)  # .squeeze(0)
    return score


def score_plda_embeddings_averaging(x1, x2, b, w):
    if isinstance(x1, (tuple, list)):
        enr = x1
    else:
        n = x1.shape[0]
        centroid = torch.mean(x1, dim=0, keepdim=True)
        enr = (centroid, n)
    score = score_plda_sph_centroids(enr, x2, b, w)  # .squeeze(0)
    return score
