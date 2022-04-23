from functools import partial

import torch
import numpy as np
import utils_io


def requires_grad(model, flag):
    for param in model.parameters():
        param.requires_grad = flag


class rpartial(partial):
    def __call__(self, *args, **kwargs):
        kw = self.keywords.copy()
        kw.update(kwargs)
        return self.func(*(args + self.args), **kw)


# def concatenate_with_labels(pos, neg):
#     n = len(pos)
#     m = len(neg)
#     return np.r_[pos, neg], np.r_[np.ones((n,)), np.zeros((m,))]


def concatenate_with_labels(*arrays):
    result = np.concatenate(list(arrays))
    labels = []
    for i, a in enumerate(arrays):
        labels += [i * np.ones_like(a)]
    labels = np.concatenate(labels)
    return result, labels


def check_tensor_2d(X, dim=1):
    assert X.dim() == 2
    mask_ok = torch.norm(X, dim=dim) > 1e-6
    return torch.all(mask_ok)


def check_tensor_1d(x):
    return torch.norm(x) > 1e-6


def generate_trials(y, n_tar, n_imp, n_enrolls=1, n_tests=1, seed=0):

    torch.manual_seed(seed)
    np.random.seed(seed)

    classes, _ = torch.unique(y, return_inverse=True)

    trials = []
    labels = []

    for i in range(n_tar):

        perm = torch.randperm(classes.size(0))
        c = classes[perm][0]

        idx = torch.nonzero(y == c).view(-1)
        if len(idx) < n_enrolls + n_tests:
            continue

        perm = torch.randperm(idx.size(0))
        idx = idx[perm]

        idx_enr = idx[:n_enrolls]
        idx_test = idx[n_enrolls : n_enrolls + n_tests]

        trials += [(idx_enr, idx_test)]
        labels += [1]

    for i in range(n_imp):

        perm = torch.randperm(classes.size(0))
        c1 = classes[perm][0]
        c2 = classes[perm][1]

        idx1 = torch.nonzero(y == c1).view(-1)
        idx2 = torch.nonzero(y == c2).view(-1)
        if len(idx1) < n_enrolls or len(idx2) < n_tests:
            continue

        perm = torch.randperm(idx1.size(0))
        idx1 = idx1[perm]
        perm = torch.randperm(idx2.size(0))
        idx2 = idx2[perm]

        idx_enr = idx1[:n_enrolls]
        idx_test = idx2[:n_tests]

        trials += [(idx_enr, idx_test)]
        labels += [0]

    labels = np.array(labels)
    return trials, labels


def get_labels_from_trials_(file_trials):
    speakers_list = {}
    utterances_list = []  # unique utterances
    speaker_index = 1
    utterances = []
    labels = []
    with open(file_trials, "r") as src:
        for line in src:
            speaker, utterance, decision = line.split()[:3]

            if utterance in utterances_list:
                continue
            else:
                utterances_list.append(utterance)

            if decision != "target":
                label = -1
            else:  # if target
                if speaker in speakers_list:
                    label = speakers_list[speaker]
                else:
                    label = speaker_index
                    speakers_list[speaker] = label
                    speaker_index += 1
            utterances.append(utterance)
            labels.append(label)
    return utterances, labels


def get_labels_from_trials(file_trials):
    data = utils_io.read_lines_file(file_trials)
    _, utts_list, trial_labels, spk_labels = list(zip(*data))
    utt2spk_id = {}
    spk2int = {}
    i = 1
    for utt, decision, spk in zip(utts_list, trial_labels, spk_labels):
        if decision == "unknown_nontarget":
            utt2spk_id[utt] = -1
        else:
            if not spk in spk2int:
                spk2int[spk] = i
                i += 1
            utt2spk_id[utt] = spk2int[spk]

    utterances, labels = list(zip(*utt2spk_id.items()))
    # utterances2, labels2 = get_labels_from_trials_(file_trials)
    # utt2spk_id2 = {utt: spk for (utt, spk) in zip(utterances2, labels2)}
    return utterances, labels
