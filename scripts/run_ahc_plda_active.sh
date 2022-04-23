
# Set DEBUG to 0 to save the scores
DEBUG=0
VERBOSE=0

# OFFLINE
# AHC-PLDA

python household_simulation.py --config configs/offline/config_speechbrain_trimmed_2sec_cos_emb_avg_offline_ahc_plda_sph.yaml --debug $DEBUG --verbose $VERBOSE
python household_simulation.py --config configs/offline/config_clova_2sec_cos_emb_avg_offline_ahc_plda_sph.yaml --debug $DEBUG --verbose $VERBOSE
python household_simulation.py --config configs/offline/config_xvector_trimmed_2sec_plda_diag_offline_ahc_plda_sph.yaml --debug $DEBUG --verbose $VERBOSE


#cmd1="python household_simulation.py --config configs/offline/config_speechbrain_trimmed_2sec_cos_emb_avg_offline_ahc_plda_sph.yaml --debug ${DEBUG} --verbose ${VERBOSE}"
#cmd2="python household_simulation.py --config configs/offline/config_clova_2sec_cos_emb_avg_offline_ahc_plda_sph.yaml --debug ${DEBUG} --verbose ${VERBOSE}"
#cmd3="python household_simulation.py --config configs/offline/config_xvector_trimmed_2sec_plda_diag_offline_ahc_plda_sph.yaml --debug ${DEBUG} --verbose ${VERBOSE}"
#
#(echo $cmd1; echo $cmd2; echo $cmd3) | parallel
