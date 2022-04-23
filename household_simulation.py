import warnings

warnings.filterwarnings("ignore")
from datetime import datetime
import argparse
import itertools
import os
import glob
import random
import shutil
import yaml
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import data_load
import metrics
import evaluation
import calibration
import scoring
import utils_io
import utils
import preprocessing as prepr
from utils import get_labels_from_trials

from mappings import _recognizers
from training import train_plda_sph_mle


def main(args):

    # common config
    config_common = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)

    # set the seed to avoid randomness
    seed = config_common["seed"]

    # evaluation protocol
    PROTOCOL_NAME = args.protocol
    PROTOCOL_PATH = f"protocols/{PROTOCOL_NAME}"
    # assert PROTOCOL_NAME in ["proto_v5b", "proto_vox1a_v2", "proto_vox1b_v2"]

    # define embedding path
    embeddings_path = config_common["embeddings_path"]
    embeddings_name = config["embeddings_name"]  # ['clova', 'speechbrain', 'xvector']
    assert embeddings_name in ["clova", "speechbrain", "xvector"]
    embeddings_type = config["embeddings_type"]

    enrollment_type = config.get("enrollment_type", "active")  # ['active', 'passive']
    # enrollment_type = "active" if args.active else "passive"

    n_enrolls = args.n_enrolls
    hh_size = args.size
    subset_adapt = args.subset_adapt

    length_normalized = config["len_norm"]

    score_type = config["score_type"]
    assert score_type in ["cos_emb_avg", "cos_sc_avg", "plda_sph", "plda_diag"]

    plda_by_the_book = config["by_the_book"]
    plda_len_norm = config.get("plda_len_norm", False)

    preprocessing_name = config["preprocessing_name"]

    recognizer = config["recognizer"]
    assert recognizer in _recognizers
    # sequential_updates = config["sequential_updates"]

    frr_point = config["frr_point"]

    calibrate_scores = config["calibrate_scores"]
    calibration_params_path = config["calibration_params_path"]

    print("Protocol:", PROTOCOL_NAME)

    if args.cuda:
        device = "cuda:0"
    else:
        device = "cpu"

    if len(embeddings_type) > 0:
        embeddings_type_path = f"_{embeddings_type}"
    else:
        embeddings_type_path = ""

    # load data:
    # utt2emb - evaluation data
    if PROTOCOL_NAME.split("/")[-1] in ["proto_v5b", "proto_v5b_small"]:
        utt2emb = data_load.load_asvspoof(
            embeddings_name,
            embeddings_type,
            embeddings_path,
            length_normalized=length_normalized,
        )

    elif PROTOCOL_NAME.split("/")[-1] in ["proto_vox1a_v2", "proto_vox1b_v2"]:
        utt2emb = data_load.load_voxceleb(
            embeddings_name,
            embeddings_type,
            embeddings_path,
            length_normalized=length_normalized,
        )

    # Always use Voxceleb1 for training/calibration
    # X, y - training/calibration data (features, labels)
    X, y, utts_train = data_load.load_train_data(
        embeddings_name,
        embeddings_type,
        embeddings_path,
        length_normalized=length_normalized,
    )

    params = config.get("params", None)
    if params is None:
        params = {}

    # data preprocessing
    transforms, model_params = prepr.load_backend(preprocessing_name)
    X = prepr.apply_sequence(X, transforms)  # TODO: remove repeated operation
    utt2emb = prepr.apply_sequence(utt2emb, transforms)

    # transfer data to device
    X = X.to(device)
    y = y.to(device)
    for (utt, emb) in utt2emb.items():
        utt2emb[utt] = emb.to(device)

    if not score_type == "plda_diag":
        b, w = train_plda_sph_mle(X, y)  # TRAIN DATA
        params["plda"] = {"b": b, "w": w}

    if score_type == "cos_emb_avg":
        similarity_score_raw = scoring.ScoreCosine("embeddings").to(device)
    elif score_type == "cos_sc_avg":
        similarity_score_raw = scoring.ScoreCosine("scores").to(device)
    elif score_type == "plda_sph":
        similarity_score_raw = scoring.ScoreSphPLDA(
            b, w, by_the_book=plda_by_the_book, len_norm=plda_len_norm
        ).to(device)
    elif score_type == "plda_diag":
        # load pretrained PLDA
        w_vec = model_params["W"]
        w_vec = torch.tensor(w_vec).to(torch.get_default_dtype())
        similarity_score_raw = scoring.ScoreDiagPLDA(
            w_vec, by_the_book=plda_by_the_book, len_norm=plda_len_norm
        ).to(device)
        params["plda"] = {"b": 1.0, "w": w_vec}

    utils.requires_grad(similarity_score_raw, False)

    # calibrate
    if calibrate_scores:
        print("Calibration")
        if len(calibration_params_path) > 0:
            cal_params_dict = utils_io.read_dict_file(calibration_params_path)
            scale, shift = cal_params_dict["scale"], cal_params_dict["shift"]
        else:
            # TODO: load calibration trials from file
            scale, shift = calibration.calibrate(
                similarity_score_raw, X, y, utts_train
            )  # TRAIN DATA
            # save parameters

            if score_type.startswith("plda"):
                score_config = f"_btb{int(plda_by_the_book)}_ln{int(plda_len_norm)}"
            else:
                score_config = ""

            os.makedirs("./saved", exist_ok=True)
            calibration_params_path = f"saved/{score_type}{score_config}_{embeddings_name}{embeddings_type_path}.txt"
            utils_io.write_dict_file(
                calibration_params_path, {"scale": scale, "shift": shift}
            )

        # global linear calibration
        similarity_score = lambda x, y: scale * similarity_score_raw(x, y) + shift

        params["calibration"] = {"scale": scale, "shift": shift}
    else:
        similarity_score = similarity_score_raw
        params["calibration"] = {"scale": 1.0, "shift": 0.0}

    # delete the data that is no longer useful
    del X, y

    # add one more parameter shared by all algorithms - threshold
    t_min = config["threshold_min"]
    t_max = config["threshold_max"]
    t_num = config["thresholds_num"]
    thresholds = np.linspace(t_min, t_max, t_num).tolist()
    params.update({"threshold": thresholds})

    keys = []
    val_lists = []
    for (key, val) in params.items():
        keys.append(key)
        if not isinstance(val, (tuple, list)):
            val = [val]
        val_lists.append(val)

    params_list = []
    # make all possible combinations of parameters
    for vals in list(itertools.product(*val_lists)):
        params_list += [{k: v for (k, v) in zip(keys, vals)}]

    trl_files = glob.glob(f"{PROTOCOL_PATH}/*.trl.txt")
    simulation_ids = [os.path.basename(f).split(".trl")[0] for f in trl_files]

    if hh_size > 0:
        print("Simulations, total:", len(simulation_ids))
        simulation_ids = [
            sim_id for sim_id in simulation_ids if int(sim_id.split(".")[-2]) == hh_size
        ]
        print(f"Simulations with hh size == {hh_size}:", len(simulation_ids))

    assert len(simulation_ids) > 0

    if not args.debug:
        t_now = datetime.now()
        protocol_name = PROTOCOL_PATH.split(os.sep)[-1]
        n_simulations = int(args.subset * len(simulation_ids))
        if args.subset < 1.0:
            protocol_name = f"{protocol_name}_{n_simulations}"
        results_dir = f"./results/{protocol_name}/{enrollment_type}/{embeddings_name}_{t_now.strftime('%Y-%m-%d')}/{t_now.strftime('%H-%M-%S')}"
        os.makedirs(results_dir, exist_ok=False)
        print("Results location:", results_dir)

    ########
    # Up to this point, passive and active are shared
    ########

    print("Evaluation")
    # loop over parameters configurations
    # TODO: parallel processing?
    time_start = time.time()
    for cur_iter, params in enumerate(params_list):

        if not args.debug:
            results_dir_iter = f"{results_dir}/{cur_iter}"
            os.makedirs(results_dir_iter, exist_ok=False)

            with open(f"{results_dir_iter}/params.pickle", "wb") as params_file:
                pickle.dump(params, params_file)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if enrollment_type == "active":

            (
                scores_target_pooled,
                scores_impostor_known_pooled,
                scores_impostor_unknown_pooled,
            ) = ([], [], [])
            (
                scores_target_pooled_enrich,
                scores_impostor_known_pooled_enrich,
                scores_impostor_unknown_pooled_enrich,
            ) = ([], [], [])

            metrics_dict = {}
            # run several simulations and pool all the scores
            n_simulations = int(args.subset * len(simulation_ids))
            for sim_idx, simulation_id in enumerate(
                tqdm(simulation_ids[:n_simulations])
            ):

                file_enroll = f"{PROTOCOL_PATH}/{simulation_id}.enroll.txt"
                file_adapt = f"{PROTOCOL_PATH}/{simulation_id}.trn.txt"
                file_trials = f"{PROTOCOL_PATH}/{simulation_id}.trl.txt"

                spk2utt_enroll = utils_io.read_dict_file(file_enroll)
                if PROTOCOL_NAME.startswith("original"):
                    spk2utt_adapt = utils_io.read_dict_file(file_adapt)
                    for spk in spk2utt_adapt:
                        if spk in spk2utt_enroll:
                            pass
                            # Oracle
                            spk2utt_enroll[spk].extend(spk2utt_adapt[spk])
                    utts_adapt = sum(list(spk2utt_adapt.values()), [])
                else:
                    utts_adapt = utils_io.read_lines_file(
                        file_adapt, sep=",", merge=True
                    )

                spk2emb_enroll = {
                    spk: [utt2emb[u] for u in utts[:n_enrolls]]
                    for (spk, utts) in spk2utt_enroll.items()
                }

                trials, labels = utils_io.read_trials_file(file_trials)

                clf = _recognizers[recognizer](similarity_score, **params)

                # create profiles
                clf.init_classes(spk2emb_enroll)  # 'active enrollment'

                # evaluation, before adaptation
                (
                    scores_target,
                    scores_impostor_known,
                    scores_impostor_unknown,
                ) = evaluation.compute_scores(clf, utt2emb, trials, labels)
                scores_target_pooled += [scores_target]
                scores_impostor_known_pooled += [scores_impostor_known]
                scores_impostor_unknown_pooled += [scores_impostor_unknown]

                # get the adaptation set
                X_adapt = []
                n_utts_adapt = int(
                    subset_adapt * len(utts_adapt)
                )  # adaptation set size
                for u in utts_adapt[:n_utts_adapt]:
                    X_adapt += [utt2emb[u]]
                X_adapt = torch.cat(X_adapt)

                # update profiles
                clf.fit(X_adapt)

                # evaluation, after adaptation
                (
                    scores_target,
                    scores_impostor_known,
                    scores_impostor_unknown,
                ) = evaluation.compute_scores(clf, utt2emb, trials, labels)
                scores_target_pooled_enrich += [scores_target]
                scores_impostor_known_pooled_enrich += [scores_impostor_known]
                scores_impostor_unknown_pooled_enrich += [scores_impostor_unknown]

            # without adaptation
            scores_target_pooled = np.concatenate(scores_target_pooled)
            scores_impostor_known_pooled = np.concatenate(scores_impostor_known_pooled)
            scores_impostor_unknown_pooled = np.concatenate(
                scores_impostor_unknown_pooled
            )

            scores, labels = utils.concatenate_with_labels(
                scores_impostor_known_pooled, scores_target_pooled
            )
            far_at_frr_known, _ = metrics.far_at_frr(
                scores, labels, frr_point=frr_point
            )
            EER_known = metrics.eer(scores, labels)[0] * 100

            scores, labels = utils.concatenate_with_labels(
                scores_impostor_unknown_pooled, scores_target_pooled
            )
            far_at_frr_unknown, _ = metrics.far_at_frr(
                scores, labels, frr_point=frr_point
            )
            EER_unknown = metrics.eer(scores, labels)[0] * 100

            metrics_dict["FAR@FRR, known"] = far_at_frr_known
            metrics_dict["FAR@FRR, unknown"] = far_at_frr_unknown
            metrics_dict["EER, known"] = EER_known
            metrics_dict["EER, unknown"] = EER_unknown

            if not args.debug:
                np.savetxt(
                    f"{results_dir_iter}/scores_tar.txt",
                    scores_target_pooled,
                    fmt="%.3f",
                )
                np.savetxt(
                    f"{results_dir_iter}/scores_imp.txt",
                    scores_impostor_known_pooled,
                    fmt="%.3f",
                )
                np.savetxt(
                    f"{results_dir_iter}/scores_imp_unk.txt",
                    scores_impostor_unknown_pooled,
                    fmt="%.3f",
                )

            # with adaptation
            scores_target_pooled_enrich = np.concatenate(scores_target_pooled_enrich)
            scores_impostor_known_pooled_enrich = np.concatenate(
                scores_impostor_known_pooled_enrich
            )
            scores_impostor_unknown_pooled_enrich = np.concatenate(
                scores_impostor_unknown_pooled_enrich
            )

            scores, labels = utils.concatenate_with_labels(
                scores_impostor_known_pooled_enrich, scores_target_pooled_enrich
            )
            far_at_frr_known, _ = metrics.far_at_frr(
                scores, labels, frr_point=frr_point
            )
            EER_known = metrics.eer(scores, labels)[0] * 100

            scores, labels = utils.concatenate_with_labels(
                scores_impostor_unknown_pooled_enrich, scores_target_pooled_enrich
            )
            far_at_frr_unknown, _ = metrics.far_at_frr(
                scores, labels, frr_point=frr_point
            )
            EER_unknown = metrics.eer(scores, labels)[0] * 100

            metrics_dict["FAR@FRR, known, adapted"] = far_at_frr_known
            metrics_dict["FAR@FRR, unknown, adapted"] = far_at_frr_unknown
            metrics_dict["EER, known, adapted"] = EER_known
            metrics_dict["EER, unknown, adapted"] = EER_unknown

            if not args.debug:
                np.savetxt(
                    f"{results_dir_iter}/scores_adapted_tar.txt",
                    scores_target_pooled_enrich,
                    fmt="%.3f",
                )
                np.savetxt(
                    f"{results_dir_iter}/scores_adapted_imp.txt",
                    scores_impostor_known_pooled_enrich,
                    fmt="%.3f",
                )
                np.savetxt(
                    f"{results_dir_iter}/scores_adapted_imp_unk.txt",
                    scores_impostor_unknown_pooled_enrich,
                    fmt="%.3f",
                )

        elif enrollment_type == "passive":

            outputs_list = []

            metrics_dict = {}

            # run several simulations
            n_simulations = int(args.subset * len(simulation_ids))
            for sim_idx, simulation_id in enumerate(
                tqdm(simulation_ids[:n_simulations])
            ):

                # file_enroll = f"{PROTOCOL_PATH}/{simulation_id}.enroll.txt"
                file_adapt = f"{PROTOCOL_PATH}/{simulation_id}.trn.txt"
                file_trials = f"{PROTOCOL_PATH}/{simulation_id}.trl.txt"

                spk2emb_enroll = {}  # no enrollment data

                clf = _recognizers[recognizer](similarity_score, **params)

                # create profiles
                clf.init_classes(spk2emb_enroll)  # do nothing

                # get the adaptation set
                utts_adapt = utils_io.read_lines_file(file_adapt, sep=",", merge=True)
                X_adapt = []
                n_utts_adapt = int(
                    subset_adapt * len(utts_adapt)
                )  # adaptation set size
                for u in utts_adapt[:n_utts_adapt]:
                    X_adapt += [utt2emb[u]]
                X_adapt = torch.cat(X_adapt)

                # update profiles
                clf.fit(X_adapt)

                # compute scores and labels
                test_utterances, labels_true = get_labels_from_trials(file_trials)
                scores_max, labels_pred = evaluation.compute_scores_passive(
                    clf, utt2emb, test_utterances
                )
                outputs_list += [(simulation_id, scores_max, labels_true, labels_pred)]

            # evaluate with threshold
            threshold = params["threshold"]
            # THIS CAN BE A SEPARATE THRESHOLD,
            # NOT NECESSARY THE ONE USED AS AN ALGORITHM'S PARAMETER

            AVERAGE_PER_CLUSTER = True  # set to False to average JERs across households

            jers_all = []
            for (simulation_id, scores, labels_true, labels_pred) in outputs_list:

                if not args.debug:
                    file_results = (
                        f"{results_dir_iter}/passive_scores.{simulation_id}.txt"
                    )
                    with open(file_results, "w") as f:
                        for score, label_true, label_pred in zip(
                            scores, labels_true, labels_pred
                        ):
                            f.write(
                                "{0} {1} {2}\n".format(score, label_true, label_pred)
                            )

                    # data = utils_io.read_lines_file(file_results)
                    # scores, labels_true, labels_pred = list(zip(*data))

                labels_pred_threshold = []
                for (score, label) in zip(scores, labels_pred):
                    if float(score) > threshold:
                        labels_pred_threshold += [int(label)]
                    else:
                        labels_pred_threshold += [-1]

                labels_true = np.array([int(idx) for idx in labels_true])  # !
                labels_pred_threshold = np.array(labels_pred_threshold)

                # compute JER
                jers = metrics.compute_jer(
                    labels_pred_threshold,
                    labels_true,
                    return_individual=AVERAGE_PER_CLUSTER,
                )
                if AVERAGE_PER_CLUSTER:
                    jers_all += jers.tolist()
                else:
                    jers_all += [jers]

            JER = np.mean(jers_all)
            # print(f"threshold: {threshold:.2f}, JER: {JER:.1f} %")
            metrics_dict["JER"] = JER

        # print metrics
        if args.verbose:
            for metric, value in metrics_dict.items():
                print(f"{metric} == {np.around(value, 3)}")

            # del some keys for clarity of printing
            params.pop("plda", None)
            params.pop("calibration", None)
            print(params)

    if not args.debug:
        # in the end, copy the config file
        # if you don't see a config file in the results directory, perhaps the experiment was interrupted
        shutil.copy(args.config, results_dir)

    time_end = time.time()
    print(f"Finished after {int((time_end - time_start)/60)} minutes!\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="configs/online/config_clova_2sec_score_plda_sph.yaml",  # configs/online/config_clova_2sec_cos_emb_avg.yaml
    )
    parser.add_argument(
        "--protocol",
        type=str,
        required=False,
        default="proto_vox1b_v2",  # proto_vox1b_v2 # proto_v5b
        help="name of the evaluation protocol. dataset will be chosen accordingly",
    )
    parser.add_argument(
        "--n_enrolls",
        type=int,
        required=False,
        default=777,
        help="maximum number of the enrollment utterances",
    )
    parser.add_argument(
        "--size",
        type=int,
        required=False,
        default=0,
        help="keep househoolds with this size only, ignore the rest",
    )
    parser.add_argument(
        "--subset_adapt",
        type=float,
        required=False,
        default=1.0,
        help="fraction of the adaptation set to use. for example, 0.1 means 10 percent",
    )
    parser.add_argument(
        "--debug",
        type=int,
        required=False,
        default=1,
        help="if true, no data will be saved to the disk",
    )
    parser.add_argument(
        "--double",
        type=int,
        required=False,
        default=1,
        help="use float64 (should be true)",
    )
    parser.add_argument(
        "--subset",
        type=float,
        required=False,
        default=1,
        help="fraction of the full set of simulations (for quick experiments)",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        required=False,
        default=1,
        help="if true, results will be printed",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        required=False,
        default=0,
        help="use cuda (only few algorithms support it)",
    )

    args = parser.parse_args()

    print("\n\nConfig:", args.config)

    if args.double:
        torch.set_default_tensor_type(torch.DoubleTensor)

    main(args)
