import torch
import torch.nn as nn
import torch.nn.functional as F

import parameterization as param
import scoring_impl


class ScoreCosine(nn.Module):
    def __init__(self, averaging):
        assert averaging in ["embeddings", "scores"]
        self.averaging = averaging
        super().__init__()

    def forward(self, enroll, test):
        if isinstance(enroll, (tuple, list)):
            enroll, n = enroll
        if self.averaging == "embeddings":
            scores = scoring_impl.score_cosine_embeddings_averaging(enroll, test)
        else:
            scores = scoring_impl.score_cosine_scores_averaging(enroll, test)
        return scores


class ScoreSphPLDA(nn.Module):
    def __init__(self, b, w, by_the_book=True, len_norm=False):
        super().__init__()
        b, w = torch.tensor([b]), torch.tensor([w])
        self.b_ = nn.Parameter(param.pos2real(b), requires_grad=True)
        self.w_ = nn.Parameter(param.pos2real(w), requires_grad=True)
        self.by_the_book = by_the_book
        self.len_norm = len_norm

    def get_params(self):
        return param.real2pos(self.b_), param.real2pos(self.w_)

    def forward(self, enroll, test):
        b, w = self.get_params()
        scores = scoring_impl.score_plda_sph_centroids(
            enroll, test, b, w, self.by_the_book, self.len_norm
        )
        return scores


class ScoreDiagPLDA(nn.Module):
    def __init__(self, w, by_the_book=True, len_norm=False):
        super().__init__()
        self.w_ = nn.Parameter(param.pos2real(w), requires_grad=True)
        self.by_the_book = by_the_book
        self.len_norm = len_norm

    def get_params(self):
        return param.real2pos(self.w_)

    def forward(self, enroll, test):
        w = self.get_params()
        scores = scoring_impl.score_plda_diag_centroids(
            enroll, test, w, self.by_the_book, self.len_norm
        )
        return scores
