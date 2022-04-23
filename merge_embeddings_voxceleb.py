import os
import glob

import numpy as np
from tqdm import tqdm

# embeddings_root = "/home/alexey/Documents/data/embeddings"
embeddings_save_dir = "./embeddings"

vox1_part = "test"
assert vox1_part in ["train", "test"]


##############################

# speechbrain
embeddings_name = "speechbrain"
embeddings_type = "trimmed_2sec"
assert embeddings_type in ["", "trimmed", "trimmed_2sec"]

os.makedirs(embeddings_save_dir, exist_ok=True)

if len(embeddings_type) > 0:
    embeddings_type_ext = f"_{embeddings_type}"
else:
    embeddings_type_ext = ""

data_path_train = f"/home/alexey/Documents/data/embeddings/20220210_speechbrain_voxceleb/voxceleb1_{vox1_part}_trimmed_2sec/npys"

X_train_split = []
utt_ids_train = []
for filepath in tqdm(glob.glob(os.path.join(data_path_train, "*.npy"))):
    utt_id = os.path.basename(filepath).split(".")[0]
    utt_ids_train += [utt_id]
    x = np.load(filepath)
    X_train_split += [np.expand_dims(x, 0)]

X_train_split = np.concatenate(X_train_split)
file_path = (
    f"{embeddings_save_dir}/emb_vox1_{vox1_part}_{embeddings_name}{embeddings_type_ext}"
)
np.savez(file_path, X=X_train_split, ids=utt_ids_train)

##############################

# xvector
embeddings_name = "xvector"
embeddings_type = "trimmed_2sec"
assert embeddings_type in ["", "trimmed", "trimmed_2sec"]

os.makedirs(embeddings_save_dir, exist_ok=True)

if len(embeddings_type) > 0:
    embeddings_type_ext = f"_{embeddings_type}"
else:
    embeddings_type_ext = ""

data_path_train = f"/home/alexey/Documents/data/embeddings/20220124_vox1_embeddings_deepasv/vox1_{vox1_part}_trimmed_2sec/npys"

X_train_split = []
utt_ids_train = []
for filepath in tqdm(glob.glob(os.path.join(data_path_train, "*.npy"))):
    utt_id = os.path.basename(filepath).split(".")[0]
    utt_ids_train += [utt_id]
    x = np.load(filepath)
    X_train_split += [np.expand_dims(x, 0)]

X_train_split = np.concatenate(X_train_split)
file_path = (
    f"{embeddings_save_dir}/emb_vox1_{vox1_part}_{embeddings_name}{embeddings_type_ext}"
)
np.savez(file_path, X=X_train_split, ids=utt_ids_train)
