import os
import numpy as np
import torch
import torch.nn.functional as F
import utils_io


def load_asvspoof(
    embeddings_name, embeddings_type, embeddings_path, length_normalized=False
):
    if len(embeddings_type) > 0:
        embeddings_type_path = f"_{embeddings_type}"
    else:
        embeddings_type_path = ""

    if embeddings_name == "clova":
        get_utt_id = lambda utt: utt
    else:
        get_utt_id = lambda utt: utt.split("-")[1]

    if length_normalized:
        emb_from_npy = lambda x: F.normalize(torch.tensor(x).reshape(1, -1), dim=1).to(
            torch.get_default_dtype()
        )
    else:
        emb_from_npy = (
            lambda x: torch.tensor(x).reshape(1, -1).to(torch.get_default_dtype())
        )

    data = np.load(
        f"{embeddings_path}/emb_asvspoof2019_train_{embeddings_name}{embeddings_type_path}.npz"
    )
    X = data["X"]
    utt_ids = data["ids"]
    utt2emb_train = {get_utt_id(utt): emb_from_npy(x) for (x, utt) in zip(X, utt_ids)}

    data = np.load(
        f"{embeddings_path}/emb_asvspoof2019_eval_enroll_{embeddings_name}{embeddings_type_path}.npz"
    )
    X = data["X"]
    utt_ids = data["ids"]
    utt2emb_enroll = {get_utt_id(utt): emb_from_npy(x) for (x, utt) in zip(X, utt_ids)}

    data = np.load(
        f"{embeddings_path}/emb_asvspoof2019_eval_test_{embeddings_name}{embeddings_type_path}.npz"
    )
    X = data["X"]
    utt_ids = data["ids"]
    utt2emb_test = {get_utt_id(utt): emb_from_npy(x) for (x, utt) in zip(X, utt_ids)}

    utt_ids_train = np.loadtxt("meta/ASVspoof2019PA_adaptation.txt", dtype="str")
    utt2spk = {}
    with open("meta/ASVspoof2019_utt2labels_train.txt", "r") as f:
        for line in f.readlines():
            utt, spk = line.strip().split()
            utt2spk[utt] = spk

    # X = []
    # y = []
    # for utt in utt_ids_train:
    #     x = utt2emb_train[utt]
    #     X += [x]
    #     y += [utt2spk[utt]]
    # X = torch.cat(X)
    # y = torch.tensor(np.unique(y, return_inverse=True)[1])
    #
    # # remove zero embeddings
    # mask = torch.norm(X, dim=1) > 1e-6
    # X = X[mask]
    # y = y[mask]

    utt2emb = {**utt2emb_train, **utt2emb_enroll, **utt2emb_test}

    return utt2emb


def load_voxceleb(
    embeddings_name, embeddings_type, embeddings_path, length_normalized=False
):

    if len(embeddings_type) > 0:
        embeddings_type_path = f"_{embeddings_type}"
    else:
        embeddings_type_path = ""

    get_utt_id = lambda utt: utt
    # get_spk_id = lambda utt: utt.split('-')[0]

    if length_normalized:
        emb_from_npy = lambda x: F.normalize(torch.tensor(x).reshape(1, -1), dim=1).to(
            torch.get_default_dtype()
        )
    else:
        emb_from_npy = (
            lambda x: torch.tensor(x).reshape(1, -1).to(torch.get_default_dtype())
        )

    data = np.load(
        f"{embeddings_path}/emb_vox1_train_{embeddings_name}{embeddings_type_path}.npz"
    )
    X = data["X"]
    utt_ids_train = data["ids"]
    utt2emb_train = {
        get_utt_id(utt): emb_from_npy(x) for (x, utt) in zip(X, utt_ids_train)
    }
    # utt2spk_train = {utt: get_spk_id(utt) for utt in utt_ids_train}

    data = np.load(
        f"{embeddings_path}/emb_vox1_test_{embeddings_name}{embeddings_type_path}.npz"
    )
    X = data["X"]
    utt_ids_test = data["ids"]
    utt2emb_test = {
        get_utt_id(utt): emb_from_npy(x) for (x, utt) in zip(X, utt_ids_test)
    }
    # utt2spk_test = {utt: get_spk_id(utt) for utt in utt_ids_test}

    utt2emb = {**utt2emb_train, **utt2emb_test}
    # utt2spk = {**utt2spk_train, **utt2spk_test}
    return utt2emb


def load_train_data(
    embeddings_name, embeddings_type, embeddings_path, length_normalized=False
):

    if len(embeddings_type) > 0:
        embeddings_type_path = f"_{embeddings_type}"
    else:
        embeddings_type_path = ""

    get_utt_id = lambda utt: utt
    get_spk_id = lambda utt: utt.split("-")[0]

    if length_normalized:
        emb_from_npy = lambda x: F.normalize(torch.tensor(x).reshape(1, -1), dim=1).to(
            torch.get_default_dtype()
        )
    else:
        emb_from_npy = (
            lambda x: torch.tensor(x).reshape(1, -1).to(torch.get_default_dtype())
        )

    data = np.load(
        f"{embeddings_path}/emb_vox1_train_{embeddings_name}{embeddings_type_path}.npz"
    )
    X = data["X"]
    utt_ids_train = data["ids"]
    utt2emb_train = {
        get_utt_id(utt): emb_from_npy(x) for (x, utt) in zip(X, utt_ids_train)
    }
    utt2spk_train = {utt: get_spk_id(utt) for utt in utt_ids_train}

    data = np.load(
        f"{embeddings_path}/emb_vox1_test_{embeddings_name}{embeddings_type_path}.npz"
    )
    X = data["X"]
    utt_ids_test = data["ids"]
    utt2emb_test = {
        get_utt_id(utt): emb_from_npy(x) for (x, utt) in zip(X, utt_ids_test)
    }
    utt2spk_test = {utt: get_spk_id(utt) for utt in utt_ids_test}

    utt2emb = {**utt2emb_train, **utt2emb_test}
    utt2spk = {**utt2spk_train, **utt2spk_test}

    # use a set of speakers that do not intersect with "proto_vox1b_v2"
    speakers = utils_io.read_lines_file("meta/proto_vox1a_v2_speakers.txt", merge=True)

    X = []
    y = []
    utt_ids = []
    for utt in utt2emb:
        if get_spk_id(utt) in speakers:
            x = utt2emb[utt]
            X += [x]
            y += [utt2spk[utt]]
            utt_ids += [utt]
    X = torch.cat(X)
    y = torch.tensor(np.unique(y, return_inverse=True)[1])

    # remove zero embeddings
    mask = torch.norm(X, dim=1) > 1e-6
    X = X[mask]
    y = y[mask]
    utt_ids = [utt for (i, utt) in enumerate(utt_ids) if mask[i]]
    assert len(y) == len(utt_ids)

    return X, y, utt_ids
