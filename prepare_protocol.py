from collections import defaultdict

import os
import glob

import numpy as np
from tqdm import tqdm

import utils_io

PROTOCOL_PATH = "protocols/proto_vox1a_v2"
# PROTOCOL_PATH = "protocols/proto_vox1b_v2"

np.random.seed(0)

trl_files = glob.glob(f"{PROTOCOL_PATH}/*.trl.txt")
simulation_ids = [os.path.basename(f).split(".trl")[0] for f in trl_files]

get_spk_vox = lambda utt: utt.split("-")[0]
if True:
    spk2utt = defaultdict(set)

    speakers = set()
    for simulation_id in tqdm(simulation_ids):

        file_enroll = f"{PROTOCOL_PATH}/{simulation_id}.enroll.txt"
        file_adapt = f"{PROTOCOL_PATH}/{simulation_id}.trn.txt"
        file_trials = f"{PROTOCOL_PATH}/{simulation_id}.trl.txt"

        spk2utt_enroll = utils_io.read_dict_file(file_enroll)
        for spk in spk2utt_enroll:
            speakers.add(spk.upper())

        utts_adapt = utils_io.read_lines_file(file_adapt, sep=",", merge=True)
        for utt in utts_adapt:
            spk = get_spk_vox(utt)
            speakers.add(spk)

    with open("meta/proto_vox1a_v2_speakers.txt", "w") as f:
        for spk in speakers:
            f.write(f"{spk}\n")
