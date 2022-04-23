import os
import glob

import numpy as np
from tqdm import tqdm


embeddings_name = "speechbrain"
embeddings_type = "trimmed_2sec"
assert embeddings_type in ["", "trimmed", "trimmed_2sec"]

include_19_impostors = False

embeddings_root = "/home/alexey/Documents/data/embeddings"
embeddings_save_dir = "./embeddings"

os.makedirs(embeddings_save_dir, exist_ok=True)

if len(embeddings_type) > 0:
    embeddings_type_ext = f"_{embeddings_type}"
else:
    embeddings_type_ext = ""

data_path_enroll = f"/home/alexey/Documents/data/embeddings/{embeddings_name}/asvspoof2019_PA/eval_enroll{embeddings_type_ext}/npys"
data_path_test = f"/home/alexey/Documents/data/embeddings/{embeddings_name}/asvspoof2019_PA/eval_test{embeddings_type_ext}/npys"
data_path_train = f"/home/alexey/Documents/data/embeddings/{embeddings_name}/asvspoof2019_PA/train{embeddings_type_ext}/npys"

# create utt2labels
# TODO: get this directly from challenge protocols
file_list = glob.glob(os.path.join(data_path_train, "*.npy"))
with open("meta/ASVspoof2019_utt2labels_train.txt", "w") as f:
    for filepath in file_list:
        utt_id = os.path.basename(filepath).split(".")[0].split("-")[1]
        utt_info = os.path.basename(filepath).split(".")[0].split("-")[0]
        f.write(f"{utt_id} {utt_info}\n")

with open("meta/ASVspoof2019_utt2labels_eval_enroll.txt", "w") as f:
    file_list = glob.glob(os.path.join(data_path_enroll, "*.npy"))
    for filepath in file_list:
        utt_id = os.path.basename(filepath).split(".")[0].split("-")[1]
        utt_info = os.path.basename(filepath).split(".")[0].split("-")[0]
        f.write(f"{utt_id} {utt_info}\n")

if include_19_impostors:
    with open("meta/ASVspoof2019_eval_test.txt", "r") as f:
        utt_ids_test = set([line.strip() for line in f.readlines()])

with open("meta/ASVspoof2019_utt2labels_eval_test.txt", "w") as f:
    file_list = glob.glob(os.path.join(data_path_test, "*.npy"))
    for filepath in file_list:
        utt_id = os.path.basename(filepath).split(".")[0].split("-")[1]
        utt_info = os.path.basename(filepath).split(".")[0].split("-")[0]
        if include_19_impostors:
            if utt_id in utt_ids_test:
                f.write(f"{utt_id} {utt_info}\n")
        else:
            f.write(f"{utt_id} {utt_info}\n")


# TRAIN
X_train_split = []
utt_ids_train = []
for filepath in tqdm(glob.glob(os.path.join(data_path_train, "*.npy"))):
    utt_id = os.path.basename(filepath).split(".")[0]
    utt_ids_train += [utt_id]
    x = np.load(filepath)
    X_train_split += [np.expand_dims(x, 0)]

X_train_split = np.concatenate(X_train_split)
file_path = f"{embeddings_save_dir}/emb_asvspoof2019_train_{embeddings_name}{embeddings_type_ext}"
np.savez(file_path, X=X_train_split, ids=utt_ids_train)


# ENROLL
X_enroll_split = []
utt_ids_enroll = []
for filepath in tqdm(glob.glob(os.path.join(data_path_enroll, "*.npy"))):
    utt_id = os.path.basename(filepath).split(".")[0]
    utt_ids_enroll += [utt_id]
    x = np.load(filepath)
    X_enroll_split += [np.expand_dims(x, 0)]

X_enroll_split = np.concatenate(X_enroll_split)
file_path = f"{embeddings_save_dir}/emb_asvspoof2019_eval_enroll_{embeddings_name}{embeddings_type_ext}"
np.savez(file_path, X=X_enroll_split, ids=utt_ids_enroll)


# TEST
X_test_split = []
utt_ids_test = []

if include_19_impostors:
    with open("meta/ASVspoof2019_eval_test.txt", "r") as f:
        utt_ids = set([line.strip() for line in f.readlines()])

file_list = glob.glob(os.path.join(data_path_test, "*.npy"))
file_list_subset = []
for filepath in file_list:
    utt_id = os.path.basename(filepath).split(".")[0].split("-")[1]
    if include_19_impostors:
        if utt_id in utt_ids:
            file_list_subset += [filepath]
    else:
        file_list_subset += [filepath]

for filepath in tqdm(file_list_subset):
    utt_id = os.path.basename(filepath).split(".")[0]
    utt_ids_test += [utt_id]
    x = np.load(filepath)
    X_test_split += [np.expand_dims(x, 0)]


X_test_split = np.concatenate(X_test_split)
file_path = f"{embeddings_save_dir}/emb_asvspoof2019_eval_test_{embeddings_name}{embeddings_type_ext}"
np.savez(file_path, X=X_test_split, ids=utt_ids_test)
