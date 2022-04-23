import numpy as np
import torch
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression

from utils import generate_trials
import utils_io


def calibrate(similarity_score_raw, X, y, utt_ids):

    utt2idx = {utt: idx for idx, utt in enumerate(utt_ids)}

    trials = []
    utts_enroll = utils_io.read_lines_file("meta/calibration_trials_enroll.txt")
    utts_test = utils_io.read_lines_file("meta/calibration_trials_test.txt")
    labels = utils_io.read_lines_file("meta/calibration_trials_label.txt")

    for utts_e, utts_t in zip(utts_enroll, utts_test):
        idx_enr = torch.tensor([utt2idx[u] for u in utts_e])
        idx_test = torch.tensor([utt2idx[u] for u in utts_t])
        trials += [(idx_enr, idx_test)]

    scores = []
    for (idx_enr, idx_test) in tqdm(trials):

        x_enr = X[idx_enr]
        x_test = X[idx_test : idx_test + 1]

        score = similarity_score_raw(x_enr, x_test)
        scores += [score]

    scores = torch.cat(scores)
    scores = scores.cpu().numpy()

    clf = LogisticRegression(random_state=0, C=1000)
    clf.fit(scores.reshape([-1, 1]), labels)

    scale, shift = float(clf.coef_), float(clf.intercept_)
    return scale, shift
