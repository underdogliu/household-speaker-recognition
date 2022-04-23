# WIP
# Implement household scoring model with enrollment and test pairs
# algorithm is from Figure 1 in the ASRU 2021 publication:
#   https://arxiv.org/abs/2109.02576
import torch
import torch.nn as nn

from models import BaseRecognizer


class HouseholdScoring(nn.Module):
    def __init__(self, embed_dim=512, low_dim=256):
        self.cos_similiarity = nn.CosineSimilarity()
        self.input_dropout = nn.Dropout(p=0.5, inplace=True)
        self.low_dim_layer = nn.Sequential(nn.Linear(embed_dim, low_dim), nn.ReLU())
        self.score_pos = nn.Linear(1, 1)
        self.score_neg = nn.Linear(1, 1)
        self.score_out = nn.Sigmoid()

    def euclidean_distance(self, A, B):
        return ((A - B) ** 2).sum(axis=0)

    def forward(self, e1, e2):
        s_g = self.cos_similiarity(e1, e2)

        e1_star = self.input_dropout(e1)
        e2_star = self.input_dropout(e2)

        e1_dash = self.low_dim_layer(e1_star)
        e2_dash = self.low_dim_layer(e2_star)

        s_h = self.euclidean_distance(e1_dash, e2_dash)

        s = self.score_out(self.score_pos(s_g) + self.score_neg(s_h))
        return s


class RecognizerNeuralScoring(BaseRecognizer):
    def __init__(self, similarity_score):
        super().__init__()
        self.similarity_score = similarity_score

    def init_classes(self, class2emb):
        super().init_classes(class2emb)
        self.n_updates = {c: 0 for c in self.representations}

    def verify(self, class_id, x):
        centroid, n = self.representations[class_id]
        score = self.similarity_score((centroid, n), x).view(-1)
        return score

    def verify_all(self, x):
        classes = list(self.representations.keys())
        scores = []
        for c in classes:
            s = self.verify(c, x)
            scores += [s]
        scores = torch.tensor(scores)
        return scores

    def fit(self, X):
        pass
