import data_load
from utils import generate_trials
import yaml
import numpy as np
from tqdm import tqdm

config_common = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)
embeddings_path = config_common["embeddings_path"]
X, y, utt_ids = data_load.load_train_data(
    "clova", "2sec", embeddings_path, length_normalized=True
)

n_tar = 25000
n_imp = 25000

trials_1vs1, labels_1vs1 = generate_trials(
    y, n_tar, n_imp, n_enrolls=1, n_tests=1, seed=0
)
trials_10vs1, labels_10vs1 = generate_trials(
    y, n_tar, n_imp, n_enrolls=10, n_tests=1, seed=0
)

trials = trials_1vs1 + trials_10vs1
labels = np.r_[labels_1vs1, labels_10vs1]

print("Save trials")
with open("meta/calibration_trials_enroll.txt", "w") as f_enr, open(
    "meta/calibration_trials_test.txt", "w"
) as f_tst, open("meta/calibration_trials_label.txt", "w") as f_lab:
    for i in tqdm(range(len(trials))):
        enr_idx, tst_idx = trials[i]
        label = labels[i]
        utts_enr = [utt_ids[idx] for idx in enr_idx]
        utts_tst = [utt_ids[idx] for idx in tst_idx]
        f_enr.write(f"{' '.join(utts_enr)}\n")
        f_tst.write(f"{' '.join(utts_tst)}\n")
        f_lab.write(f"{int(label)}\n")
