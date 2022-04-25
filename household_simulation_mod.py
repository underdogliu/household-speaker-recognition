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
from tqdm import tqdm

import data_load
import metrics
import evaluation
import calibration
import scoring
import utils_io
import utils
import preprocessing
from utils import get_labels_from_trials

from mappings import _recognizers
from training import train_plda_sph_mle


def copy_config(config, results_dir):
    shutil.copy(config, results_dir)


def specify_result_dir(
    protocol_path,
    simulation_ids,
    enrollment_type="active",
    embeddings_name="speechbrain",
):
    # specify result directory
    t_now = datetime.now()
    protocol_name = protocol_path.split(os.sep)[-1]
    n_simulations = int(args.subset * len(simulation_ids))
    if args.subset < 1.0:
        protocol_name = f"{protocol_name}_{n_simulations}"
    results_dir = f"./results/{protocol_name}/{enrollment_type}/{embeddings_name}_{t_now.strftime('%Y-%m-%d')}/{t_now.strftime('%H-%M-%S')}"
    os.makedirs(results_dir, exist_ok=False)
    print("Results location:", results_dir)


def set_seed(config_common):
    # set the seed to avoid randomness
    seed = config_common["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_embeddings(config_common, config, protocol_name="proto_v5b", device="cpu"):
    """Load the speaker embeddings to train the backend.

    Args:
        config_common (str): common config on embeddings
        config (str): detailed config of embeddings and scoring backend.
        protocol_name (str, optional): name of experimental protocol. Defaults to "proto_v5b".
        device (str, optional): running device. Defaults to "cpu" or "gpu:0".
    """
    # load related parameters
    embeddings_path = config_common["embeddings_path"]

    embeddings_type = config["embeddings_type"]
    embeddings_name = config["embeddings_name"]  # ['clova', 'speechbrain', 'xvector']
    assert embeddings_name in ["clova", "speechbrain", "xvector"]

    recognizer = config["recognizer"]
    assert recognizer in _recognizers

    # load data:
    # utt2emb - evaluation data
    if protocol_name.split("/")[-1] in ["proto_v5b", "proto_v5b_small"]:
        utt2emb = data_load.load_asvspoof(
            embeddings_name,
            embeddings_type,
            embeddings_path,
            length_normalized=config["len_norm"],
        )

    elif protocol_name.split("/")[-1] in ["proto_vox1a_v2", "proto_vox1b_v2"]:
        utt2emb = data_load.load_voxceleb(
            embeddings_name,
            embeddings_type,
            embeddings_path,
            length_normalized=config["len_norm"],
        )

    # X, y - training/calibration data (features, labels)
    X, y, utts_train = data_load.load_train_data(
        embeddings_name,
        embeddings_type,
        embeddings_path,
        length_normalized=config["len_norm"],
    )

    # data preprocessing
    preprocessing_name = config["preprocessing_name"]
    transforms, _ = preprocessing.load_backend(preprocessing_name)
    X = preprocessing.apply_sequence(X, transforms)
    utt2emb = preprocessing.apply_sequence(utt2emb, transforms)

    # transfer data to device
    X = X.to(device)
    y = y.to(device)
    for (utt, emb) in utt2emb.items():
        utt2emb[utt] = emb.to(device)

    return X, y, utts_train, utt2emb


def plda_train_infer(config, utts_train, device="cpu"):
    """_summary_

    Args:
        config (_type_): _description_
        utts_train (_type_): _description_
        calibrate (bool, optional): _description_. Defaults to False.
        device (str, optional): _description_. Defaults to "cpu".

    Returns:
        _type_: _description_
    """
    # load related parameters
    embeddings_type = config["embeddings_type"]
    embeddings_name = config["embeddings_name"]  # ['clova', 'speechbrain', 'xvector']
    assert embeddings_name in ["clova", "speechbrain", "xvector"]
    if len(embeddings_type) > 0:
        embeddings_type_path = f"_{embeddings_type}"
    else:
        embeddings_type_path = ""

    score_type = config["score_type"]
    assert score_type in ["cos_emb_avg", "cos_sc_avg", "plda_sph", "plda_diag"]

    plda_by_the_book = config["by_the_book"]
    plda_len_norm = config.get("plda_len_norm", False)

    preprocessing_name = config["preprocessing_name"]
    _, model_params = preprocessing.load_backend(preprocessing_name)

    params = config.get("params", None)
    if params is None:
        params = {}

    # PLDA training
    if not score_type == "plda_diag":
        b, w = train_plda_sph_mle(X, y)
        params["plda"] = {"b": b, "w": w}

    # PLDA scoring
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

    # score calibration (optional)
    calibrate = config["calibrate_scores"]
    calibration_params_path = config["calibration_params_path"]
    if calibrate:
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

    return similarity_score, params


def load_threshold(config, params):
    """_summary_

    Args:
        config (_type_): _description_
        params (_type_): _description_

    Returns:
        _type_: _description_
    """
    t_min = config["threshold_min"]
    t_max = config["threshold_max"]
    t_num = config["thresholds_num"]
    thresholds = np.linspace(t_min, t_max, t_num).tolist()
    params.update({"threshold": thresholds})
    return params


def load_trials_convert_params(params, protocol_name="proto_v5b", household_size=7):
    """_summary_

    Args:
        params (_type_): _description_
        protocol_name (_type_): _description_
        household_size (int, optional): _description_. Defaults to 7.
    """
    # load trial file list and simulation ID lists
    protocol_path = f"protocols/{protocol_name}"
    trl_files = glob.glob(f"{protocol_path}/*.trl.txt")
    simulation_ids = [os.path.basename(f).split(".trl")[0] for f in trl_files]

    keys = []
    val_lists = []
    for (key, val) in params.items():
        keys.append(key)
        if not isinstance(val, (tuple, list)):
            val = [val]
        val_lists.append(val)

    # make all possible combinations of parameters
    params_list = []
    for vals in list(itertools.product(*val_lists)):
        params_list += [{k: v for (k, v) in zip(keys, vals)}]

    if household_size > 0:
        print("Simulations, total:", len(simulation_ids))
        simulation_ids = [
            sim_id
            for sim_id in simulation_ids
            if int(sim_id.split(".")[-2]) == household_size
        ]
        print(
            f"Simulations with household size == {household_size}:", len(simulation_ids)
        )
    assert len(simulation_ids) > 0

    return params_list, simulation_ids


def active_enrollment(
    config,
    params_list,
    simulation_ids,
    protocol_path,
    utt2emb,
    similarity_score,
    results_dir,
    subset_adapt=False,
    n_enrolls=777,
    protocol_name="proto_v5b",
    verbose=False,
):
    # load related params
    recognizer = config["recognizer"]
    assert recognizer in _recognizers
    frr_point = config["frr_point"]

    embeddings_name = config["embeddings_name"]  # ['clova', 'speechbrain', 'xvector']
    assert embeddings_name in ["clova", "speechbrain", "xvector"]

    results_dir = specify_result_dir(
        protocol_path,
        simulation_ids,
        enrollment_type="active",
        embeddings_name=embeddings_name,
    )

    for cur_iter, params in enumerate(params_list):

        if not args.debug:
            results_dir_iter = f"{results_dir}/{cur_iter}"
            os.makedirs(results_dir_iter, exist_ok=False)

            with open(f"{results_dir_iter}/params.pickle", "wb") as params_file:
                pickle.dump(params, params_file)

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
        for sim_idx, simulation_id in enumerate(tqdm(simulation_ids[:n_simulations])):

            file_enroll = f"{protocol_path}/{simulation_id}.enroll.txt"
            file_adapt = f"{protocol_path}/{simulation_id}.trn.txt"
            file_trials = f"{protocol_path}/{simulation_id}.trl.txt"

            spk2utt_enroll = utils_io.read_dict_file(file_enroll)
            if protocol_name.startswith("original"):
                spk2utt_adapt = utils_io.read_dict_file(file_adapt)
                for spk in spk2utt_adapt:
                    if spk in spk2utt_enroll:
                        pass
                        # Oracle
                        spk2utt_enroll[spk].extend(spk2utt_adapt[spk])
                utts_adapt = sum(list(spk2utt_adapt.values()), [])
            else:
                utts_adapt = utils_io.read_lines_file(file_adapt, sep=",", merge=True)

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
            n_utts_adapt = int(subset_adapt * len(utts_adapt))  # adaptation set size
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
        scores_impostor_unknown_pooled = np.concatenate(scores_impostor_unknown_pooled)

        scores, labels = utils.concatenate_with_labels(
            scores_impostor_known_pooled, scores_target_pooled
        )
        far_at_frr_known, _ = metrics.far_at_frr(scores, labels, frr_point=frr_point)
        EER_known = metrics.eer(scores, labels)[0] * 100

        scores, labels = utils.concatenate_with_labels(
            scores_impostor_unknown_pooled, scores_target_pooled
        )
        far_at_frr_unknown, _ = metrics.far_at_frr(scores, labels, frr_point=frr_point)
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
        far_at_frr_known, _ = metrics.far_at_frr(scores, labels, frr_point=frr_point)
        EER_known = metrics.eer(scores, labels)[0] * 100

        scores, labels = utils.concatenate_with_labels(
            scores_impostor_unknown_pooled_enrich, scores_target_pooled_enrich
        )
        far_at_frr_unknown, _ = metrics.far_at_frr(scores, labels, frr_point=frr_point)
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

    # print metrics
    if verbose:
        for metric, value in metrics_dict.items():
            print(f"{metric} == {np.around(value, 3)}")

        # del some keys for clarity of printing
        params.pop("plda", None)
        params.pop("calibration", None)
        print(params)

    return results_dir


def passive_enrollment(
    config,
    params_list,
    simulation_ids,
    protocol_path,
    utt2emb,
    similarity_score,
    results_dir,
    subset_adapt=False,
    n_enrolls=777,
    protocol_name="proto_v5b",
    verbose=False,
):
    # load related params
    recognizer = config["recognizer"]
    assert recognizer in _recognizers

    embeddings_name = config["embeddings_name"]  # ['clova', 'speechbrain', 'xvector']
    assert embeddings_name in ["clova", "speechbrain", "xvector"]

    results_dir = specify_result_dir(
        protocol_path,
        simulation_ids,
        enrollment_type="passive",
        embeddings_name=embeddings_name,
    )

    for cur_iter, params in enumerate(params_list):

        if not args.debug:
            results_dir_iter = f"{results_dir}/{cur_iter}"
            os.makedirs(results_dir_iter, exist_ok=False)

            with open(f"{results_dir_iter}/params.pickle", "wb") as params_file:
                pickle.dump(params, params_file)

        outputs_list = []

        metrics_dict = {}

        # run several simulations
        n_simulations = int(args.subset * len(simulation_ids))
        for sim_idx, simulation_id in enumerate(tqdm(simulation_ids[:n_simulations])):

            # file_enroll = f"{PROTOCOL_PATH}/{simulation_id}.enroll.txt"
            file_adapt = f"{protocol_path}/{simulation_id}.trn.txt"
            file_trials = f"{protocol_path}/{simulation_id}.trl.txt"

            spk2emb_enroll = {}  # no enrollment data

            clf = _recognizers[recognizer](similarity_score, **params)

            # create profiles
            clf.init_classes(spk2emb_enroll)  # do nothing

            # get the adaptation set
            utts_adapt = utils_io.read_lines_file(file_adapt, sep=",", merge=True)
            X_adapt = []
            n_utts_adapt = int(subset_adapt * len(utts_adapt))  # adaptation set size
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
                file_results = f"{results_dir_iter}/passive_scores.{simulation_id}.txt"
                with open(file_results, "w") as f:
                    for score, label_true, label_pred in zip(
                        scores, labels_true, labels_pred
                    ):
                        f.write("{0} {1} {2}\n".format(score, label_true, label_pred))

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
        metrics_dict["JER"] = JER

    # print metrics
    if verbose:
        for metric, value in metrics_dict.items():
            print(f"{metric} == {np.around(value, 3)}")

        # del some keys for clarity of printing
        params.pop("plda", None)
        params.pop("calibration", None)
        print(params)

    return results_dir


def main(args):
    if args.double:
        torch.set_default_tensor_type(torch.DoubleTensor)

    # start timing for the program
    # loop over parameters configurations
    # TODO: parallel processing?
    time_start = time.time()

    # common config
    config_common = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)

    # set the seed to avoid randomness
    set_seed(config_common)

    # load protocol data
    protocol_name = args.protocol
    protocol_path = f"protocols/{protocol_name}"

    # Set device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # load training data
    X, y, utts_train, utt2emb = load_embeddings(
        config_common, config, protocol_name=protocol_name, device=device
    )

    # initialize back-end parameters, train the PLDA and do the scoring
    similarity_score, backend_params = plda_train_infer(
        config, utts_train, device=device
    )

    # additionally, load threshold into the parameters
    backend_params = load_threshold(backend_params)

    # load trials and convert parameters
    backend_params_list, simulation_ids = load_trials_convert_params(
        backend_params, protocol_name=protocol_name, household_size=args.household_size
    )

    enrollment_type = "active" if args.active else "passive"
    enrollment_func = (
        active_enrollment if enrollment_type == "active" else passive_enrollment
    )
    results_dir = enrollment_func(
        config,
        backend_params_list,
        simulation_ids,
        protocol_path,
        utt2emb,
        similarity_score,
        subset_adapt=args.subset_adapt,
        protocol_name=protocol_name,
        verbose=args.verbose,
    )
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
        "--household-size",
        type=int,
        required=False,
        default=0,
        help="keep households with this size only, ignore the rest",
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

    main(args)
