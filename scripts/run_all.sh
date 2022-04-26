
# Set DEBUG to 0 to save the scores
DEBUG=0
VERBOSE=0

#./run_online.sh
#./run_offline.sh
#./run_passive.sh
#./run_ahc_plda_active.sh
#./run_ahc_plda_passive.sh

# ONLINE

# speechbrain
python3 household_simulation.py --config configs/config_speechbrain_trimmed_2sec_cos_emb_avg.yaml --debug $DEBUG --verbose $VERBOSE
python3 household_simulation.py --config configs/config_speechbrain_trimmed_2sec_score_cos_sc_avg.yaml --debug $DEBUG --verbose $VERBOSE
python3 household_simulation.py --config configs/config_speechbrain_trimmed_2sec_score_plda_sph.yaml --debug $DEBUG --verbose $VERBOSE

## clova
python3 household_simulation.py --config configs/config_clova_2sec_cos_emb_avg.yaml --debug $DEBUG --verbose $VERBOSE
python3 household_simulation.py --config configs/config_clova_2sec_score_cos_sc_avg.yaml --debug $DEBUG --verbose $VERBOSE
python3 household_simulation.py --config configs/config_clova_2sec_score_plda_sph.yaml --debug $DEBUG --verbose $VERBOSE

# xvector
python3 household_simulation.py --config configs/config_xvector_trimmed_2sec_score_plda_diag.yaml --debug $DEBUG --verbose $VERBOSE
python3 household_simulation.py --config configs/config_xvector_trimmed_2sec_score_plda_diag_by-the-book.yaml --debug $DEBUG --verbose $VERBOSE


#OFFLINE

# speechbrain
python3 household_simulation.py --config configs/offline/config_speechbrain_trimmed_2sec_cos_emb_avg_offline_kmeans.yaml --debug $DEBUG --verbose $VERBOSE
python3 household_simulation.py --config configs/offline/config_speechbrain_trimmed_2sec_cos_emb_avg_offline_vb_plda_sph.yaml --debug $DEBUG --verbose $VERBOSE
python3 household_simulation.py --config configs/offline/config_speechbrain_trimmed_2sec_cos_emb_avg_offline_label_spread.yaml --debug $DEBUG --verbose $VERBOSE


# clova
python3 household_simulation.py --config configs/offline/config_clova_2sec_cos_emb_avg_offline_kmeans.yaml --debug $DEBUG --verbose $VERBOSE
python3 household_simulation.py --config configs/offline/config_clova_2sec_cos_emb_avg_offline_vb_plda_sph.yaml --debug $DEBUG --verbose $VERBOSE
python3 household_simulation.py --config configs/offline/config_clova_2sec_cos_emb_avg_offline_label_spread.yaml --debug $DEBUG --verbose $VERBOSE


# xvector
python3 household_simulation.py --config configs/offline/config_xvector_trimmed_2sec_plda_diag_offline_kmeans.yaml --debug $DEBUG --verbose $VERBOSE
python3 household_simulation.py --config configs/offline/config_xvector_trimmed_2sec_plda_diag_offline_vb_plda_sph.yaml --debug $DEBUG --verbose $VERBOSE

# -----
# AHC
python3 household_simulation.py --config configs/offline/config_speechbrain_trimmed_2sec_cos_emb_avg_offline_ahc_plda_sph.yaml --debug $DEBUG --verbose $VERBOSE
python3 household_simulation.py --config configs/offline/config_clova_2sec_cos_emb_avg_offline_ahc_plda_sph.yaml --debug $DEBUG --verbose $VERBOSE

python3 household_simulation.py --config configs/offline/config_xvector_trimmed_2sec_plda_diag_offline_ahc_plda_sph.yaml --debug $DEBUG --verbose $VERBOSE


# PASSIVE

## clova
python3 household_simulation.py --config configs/passive/config_clova_2sec_ahc.yaml --debug $DEBUG --verbose $VERBOSE
python3 household_simulation.py --config configs/passive/config_clova_2sec_ahc_plda.yaml --debug $DEBUG --verbose $VERBOSE
python3 household_simulation.py --config configs/passive/config_clova_2sec_vb_plda_sph.yaml --debug $DEBUG --verbose $VERBOSE

## speechbrain
python3 household_simulation.py --config configs/passive/config_speechbrain_trimmed_2sec_ahc.yaml --debug $DEBUG --verbose $VERBOSE
python3 household_simulation.py --config configs/passive/config_speechbrain_trimmed_2sec_ahc_plda.yaml --debug $DEBUG --verbose $VERBOSE
python3 household_simulation.py --config configs/passive/config_speechbrain_trimmed_2sec_vb_plda.yaml --debug $DEBUG --verbose $VERBOSE

## xvector
python3 household_simulation.py --config configs/passive/config_xvector_trimmed_2sec_ahc.yaml --debug $DEBUG --verbose $VERBOSE
python3 household_simulation.py --config configs/passive/config_xvector_trimmed_2sec_ahc_plda.yaml --debug $DEBUG --verbose $VERBOSE
python3 household_simulation.py --config configs/passive/config_xvector_trimmed_2sec_vb_plda.yaml --debug $DEBUG --verbose $VERBOSE