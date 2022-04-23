import numpy as np
import torch


def compute_scores(recognizer, utt2emb, trials, labels):
    classes = recognizer.get_classes()
    speaker_ids, test_utts = zip(*trials)
    speaker_ids, test_utts = np.array(speaker_ids), np.array(test_utts)
    scores = np.zeros((len(trials),))
    for c in classes:
        mask = speaker_ids == c
        x = torch.cat([utt2emb[u].view(1, -1) for u in test_utts[mask]])
        scores[mask] = recognizer.verify(c, x).view(-1).cpu().numpy()
    scores_target = scores[labels == "target"]
    scores_impostor_known = scores[labels == "known_nontarget"]
    scores_impostor_unknown = scores[labels == "unknown_nontarget"]
    return scores_target, scores_impostor_known, scores_impostor_unknown


def compute_scores_passive(recognizer, utt2emb, test_utts):
    X_test = torch.cat([utt2emb[utt].view(1, -1) for utt in test_utts])
    scores = recognizer.verify_all(X_test)
    scores_max, labels_pred = torch.max(scores, dim=1)
    return scores_max, labels_pred
