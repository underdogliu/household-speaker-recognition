
# Set DEBUG to 0 to save the scores
DEBUG=0
VERBOSE=0

# OFFLINE

# speechbrain
python3 household_simulation.py --config configs/offline/config_speechbrain_trimmed_2sec_cos_emb_avg_offline_kmeans.yaml --debug $DEBUG --verbose $VERBOSE
python3 household_simulation.py --config configs/offline/config_speechbrain_trimmed_2sec_cos_emb_avg_offline_vb_plda_sph.yaml --debug $DEBUG --verbose $VERBOSE
python3 household_simulation.py --config configs/offline/config_speechbrain_trimmed_2sec_cos_emb_avg_offline_label_spread.yaml --debug $DEBUG --verbose $VERBOSE

# clova
python3 household_simulation.py --config configs/offline/config_clova_2sec_cos_emb_avg_offline_kmeans.yaml --debug $DEBUG --verbose $VERBOSE
python3 household_simulation.py --config configs/offline/config_clova_2sec_cos_emb_avg_offline_vb_plda_sph.yaml --debug $DEBUG --verbose $VERBOSE
python3 household_simulation.py --config configs/offline/config_clova_2sec_cos_emb_avg_offline_label_spread.yaml --debug $DEBUG --verbose $VERBOSE

# xvector (no label propagation)
python3 household_simulation.py --config configs/offline/config_xvector_trimmed_2sec_plda_diag_offline_kmeans.yaml --debug $DEBUG --verbose $VERBOSE
python3 household_simulation.py --config configs/offline/config_xvector_trimmed_2sec_plda_diag_offline_vb_plda_sph.yaml --debug $DEBUG --verbose $VERBOSE
