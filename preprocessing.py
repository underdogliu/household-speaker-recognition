import numpy as np

import torch
import torch.nn.functional as F
from utils import rpartial


def joint_diagonalization(I, D, return_diagonal=True):
    # T: T @ I @ T.T = eye, T @ D @ T.T = diagonal
    lmbda, L = np.linalg.eig(np.linalg.inv(I))
    Lmbda_sqrt = np.diag(lmbda ** 0.5)
    _, U = np.linalg.eig(Lmbda_sqrt @ L.T @ D @ L @ Lmbda_sqrt)
    T = U.T @ Lmbda_sqrt @ L.T
    if return_diagonal:
        return T, np.diag(T @ D @ T.T)
    else:
        return T


def load_plda(preprocessing_type):
    if len(preprocessing_type) == 0:
        raise ValueError
    elif preprocessing_type == "xvector":
        data = np.load(f"saved/plda_xvector.npz")
        B = data["B"]
        W = data["W"]
        T, w_vec = joint_diagonalization(B, W)
        return T, w_vec


def load_backend(preprocessing_name, diagonalize_plda=True):

    if len(preprocessing_name) == 0:
        return (), {}
    elif preprocessing_name == "xvector":
        data = np.load(f"saved/plda_xvector.npz")
        V = torch.tensor(data["PCA"]).to(torch.get_default_dtype())
        mu = torch.tensor(data["mean"]).to(torch.get_default_dtype())
        L = torch.tensor(data["LDA"]).to(torch.get_default_dtype())
        len_norm = lambda X: F.normalize(X, dim=1)
        B = data["B"].astype(np.float32)
        W = data["W"].astype(np.float32)

        transforms = []
        transforms += [rpartial(torch.matmul, V)]
        transforms += [rpartial(torch.subtract, mu)]
        transforms += [rpartial(torch.matmul, L)]
        transforms += [len_norm]

        params = {}
        if diagonalize_plda:
            T, w_vec = joint_diagonalization(B, W)
            T = torch.tensor(np.transpose(T)).to(torch.get_default_dtype())
            transforms += [rpartial(torch.matmul, T)]
            params["B"] = np.ones(len(w_vec))
            params["W"] = w_vec  # TODO: return torch tensors as well??
        else:
            params["B"] = B
            params["W"] = W
        return transforms, params
    else:
        raise ValueError


def apply_sequence(data, transforms=()):
    if len(transforms) == 0:
        return data
    else:
        if isinstance(data, dict):
            utt2emb = data
            utt2emb_result = {}
            for (utt, x) in utt2emb.items():
                for tr in transforms:
                    x = tr(x)
                utt2emb_result[utt] = x
            return utt2emb_result
        else:
            X = data
            for tr in transforms:
                X = tr(X)
            return X
