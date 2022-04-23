
# This script will run configs from "./configs/final/*" containing "final" parameter configurations to produce ready-to-publish results
# Usually, these configs include a single configuration of algorithms parameters and a single threshold
# Note: sometimes config filenames may not be aligned with their contents, be careful
# Since each command takes some times (5-20 minutes), it is recommended to split this script into several parts and run them in parallel

PROTO="proto_v5b" # "proto_vox1b_v2"

DEBUG=0 # Set DEBUG to 0 to save the scores
VERBOSE=1

# ONLINE

# speechbrain
python household_simulation.py --config configs/final/online/config_speechbrain_trimmed_2sec_cos_emb_avg.yaml --debug $DEBUG --verbose $VERBOSE --protocol $PROTO;
python household_simulation.py --config configs/final/online/config_speechbrain_trimmed_2sec_score_cos_sc_avg.yaml --debug $DEBUG --verbose $VERBOSE --protocol $PROTO;
python household_simulation.py --config configs/final/online/config_speechbrain_trimmed_2sec_score_plda_sph.yaml --debug $DEBUG --verbose $VERBOSE --protocol $PROTO;

# clova
python household_simulation.py --config configs/final/online/config_clova_2sec_cos_emb_avg.yaml --debug $DEBUG --verbose $VERBOSE --protocol $PROTO;
python household_simulation.py --config configs/final/online/config_clova_2sec_score_cos_sc_avg.yaml --debug $DEBUG --verbose $VERBOSE --protocol $PROTO;
python household_simulation.py --config configs/final/online/config_clova_2sec_score_plda_sph.yaml --debug $DEBUG --verbose $VERBOSE --protocol $PROTO;

# xvector
python household_simulation.py --config configs/final/online/config_xvector_trimmed_2sec_score_plda_diag.yaml --debug $DEBUG --verbose $VERBOSE --protocol $PROTO;
#python household_simulation.py --config configs/final/online/config_xvector_trimmed_2sec_score_plda_diag_by-the-book.yaml --debug $DEBUG --verbose $VERBOSE --protocol $PROTO;


# OFFLINE

# speechbrain
python household_simulation.py --config configs/final/offline/config_speechbrain_trimmed_2sec_cos_emb_avg_offline_kmeans.yaml --debug $DEBUG --verbose $VERBOSE --protocol $PROTO;
python household_simulation.py --config configs/final/offline/config_speechbrain_trimmed_2sec_plda_sph_offline_kmeans.yaml --debug $DEBUG --verbose $VERBOSE --protocol $PROTO;
python household_simulation.py --config configs/final/offline/config_speechbrain_trimmed_2sec_cos_emb_avg_offline_vb_plda_sph.yaml --debug $DEBUG --verbose $VERBOSE --protocol $PROTO;
python household_simulation.py --config configs/final/offline/config_speechbrain_trimmed_2sec_cos_emb_avg_offline_label_spread.yaml --debug $DEBUG --verbose $VERBOSE --protocol $PROTO;

# clova
python household_simulation.py --config configs/final/offline/config_clova_2sec_cos_emb_avg_offline_kmeans.yaml --debug $DEBUG --verbose $VERBOSE --protocol $PROTO;
python household_simulation.py --config configs/final/offline/config_clova_2sec_plda_sph_offline_kmeans.yaml --debug $DEBUG --verbose $VERBOSE --protocol $PROTO;
python household_simulation.py --config configs/final/offline/config_clova_2sec_cos_emb_avg_offline_vb_plda_sph.yaml --debug $DEBUG --verbose $VERBOSE --protocol $PROTO;
python household_simulation.py --config configs/final/offline/config_clova_2sec_cos_emb_avg_offline_label_spread.yaml --debug $DEBUG --verbose $VERBOSE --protocol $PROTO;

# xvector
python household_simulation.py --config configs/final/offline/config_xvector_trimmed_2sec_plda_diag_offline_kmeans.yaml --debug $DEBUG --verbose $VERBOSE --protocol $PROTO;
#python household_simulation.py --config configs/final/offline/config_xvector_trimmed_2sec_plda_diag_offline_vb_plda_sph.yaml --debug $DEBUG --verbose $VERBOSE --protocol $PROTO;

# run AHC separately because it is very very slow
# AHC
#python household_simulation.py --config configs/final/offline/config_speechbrain_trimmed_2sec_cos_emb_avg_offline_ahc_plda_sph.yaml --debug $DEBUG --verbose $VERBOSE --protocol $PROTO;
#python household_simulation.py --config configs/final/offline/config_clova_2sec_cos_emb_avg_offline_ahc_plda_sph.yaml --debug $DEBUG --verbose $VERBOSE --protocol $PROTO;
#python household_simulation.py --config configs/final/offline/config_xvector_trimmed_2sec_plda_diag_offline_ahc_plda_sph.yaml --debug $DEBUG --verbose $VERBOSE --protocol $PROTO;


#####################################


# special experiments for a single algorithm, e.g., "score_plda_sph"

DEBUG=0
VERBOSE=1

# clova
CONFIG=configs/final/online/config_clova_2sec_score_plda_sph.yaml
# speechbrain
#CONFIG=configs/final/online/config_speechbrain_trimmed_2sec_score_plda_sph.yaml

# run for different sizes of the adaptation set (0.1 means that only 10% of the original adaptation set will be used)
python household_simulation.py --config $CONFIG --debug $DEBUG --verbose $VERBOSE --protocol $PROTO
python household_simulation.py --config $CONFIG --debug $DEBUG --verbose $VERBOSE --protocol $PROTO --subset_adapt 1.0
python household_simulation.py --config $CONFIG --debug $DEBUG --verbose $VERBOSE --protocol $PROTO --subset_adapt 0.5
python household_simulation.py --config $CONFIG --debug $DEBUG --verbose $VERBOSE --protocol $PROTO --subset_adapt 0.1

# run for different number of enrollment utterances: 1, 2 or 3 (or 4, for Voxceleb protocol)
python household_simulation.py --config $CONFIG --debug $DEBUG --verbose $VERBOSE --protocol $PROTO --n_enrolls 3
python household_simulation.py --config $CONFIG --debug $DEBUG --verbose $VERBOSE --protocol $PROTO --n_enrolls 2
python household_simulation.py --config $CONFIG --debug $DEBUG --verbose $VERBOSE --protocol $PROTO --n_enrolls 1

# run for a fixed household size. we have household of size 4, 6, 8, or 10
python household_simulation.py --config $CONFIG --debug $DEBUG --verbose $VERBOSE --protocol $PROTO --size 4
python household_simulation.py --config $CONFIG --debug $DEBUG --verbose $VERBOSE --protocol $PROTO --size 6
python household_simulation.py --config $CONFIG --debug $DEBUG --verbose $VERBOSE --protocol $PROTO --size 8
python household_simulation.py --config $CONFIG --debug $DEBUG --verbose $VERBOSE --protocol $PROTO --size 10




#####################################

# Produce curves for scoring comparison

# speechbrain

PROTO="proto_v5b"
DEBUG=0
VERBOSE=1
python household_simulation.py --config configs/online/config_speechbrain_trimmed_2sec_cos_emb_avg.yaml --debug $DEBUG --verbose $VERBOSE --protocol $PROTO;
PROTO="proto_vox1b_v2"
DEBUG=0
VERBOSE=1
python household_simulation.py --config configs/online/config_speechbrain_trimmed_2sec_cos_emb_avg.yaml --debug $DEBUG --verbose $VERBOSE --protocol $PROTO;


PROTO="proto_v5b"
DEBUG=0
VERBOSE=1
python household_simulation.py --config configs/online/config_speechbrain_trimmed_2sec_score_cos_sc_avg.yaml --debug $DEBUG --verbose $VERBOSE --protocol $PROTO;
PROTO="proto_vox1b_v2"
DEBUG=0
VERBOSE=1
python household_simulation.py --config configs/online/config_speechbrain_trimmed_2sec_score_cos_sc_avg.yaml --debug $DEBUG --verbose $VERBOSE --protocol $PROTO;


PROTO="proto_v5b"
DEBUG=0
VERBOSE=1
python household_simulation.py --config configs/online/config_speechbrain_trimmed_2sec_score_plda_sph.yaml --debug $DEBUG --verbose $VERBOSE --protocol $PROTO;
PROTO="proto_vox1b_v2"
DEBUG=0
VERBOSE=1
python household_simulation.py --config configs/online/config_speechbrain_trimmed_2sec_score_plda_sph.yaml --debug $DEBUG --verbose $VERBOSE --protocol $PROTO;




# clova

PROTO="proto_v5b"
DEBUG=0
VERBOSE=1
python household_simulation.py --config configs/online/config_clova_2sec_cos_emb_avg.yaml --debug $DEBUG --verbose $VERBOSE --protocol $PROTO;
PROTO="proto_vox1b_v2"
DEBUG=0
VERBOSE=1
python household_simulation.py --config configs/online/config_clova_2sec_cos_emb_avg.yaml --debug $DEBUG --verbose $VERBOSE --protocol $PROTO;


PROTO="proto_v5b"
DEBUG=0
VERBOSE=1
python household_simulation.py --config configs/online/config_clova_2sec_score_cos_sc_avg.yaml --debug $DEBUG --verbose $VERBOSE --protocol $PROTO;
PROTO="proto_vox1b_v2"
DEBUG=0
VERBOSE=1
python household_simulation.py --config configs/online/config_clova_2sec_score_cos_sc_avg.yaml --debug $DEBUG --verbose $VERBOSE --protocol $PROTO;


PROTO="proto_v5b"
DEBUG=0
VERBOSE=1
python household_simulation.py --config configs/online/config_clova_2sec_score_plda_sph.yaml --debug $DEBUG --verbose $VERBOSE --protocol $PROTO;
PROTO="proto_vox1b_v2"
DEBUG=0
VERBOSE=1
python household_simulation.py --config configs/online/config_clova_2sec_score_plda_sph.yaml --debug $DEBUG --verbose $VERBOSE --protocol $PROTO;


#gnome-terminal -- bash -c "echo test; CMD ;exec bash"


