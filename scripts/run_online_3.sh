
# Set DEBUG to 0 to save the scores
DEBUG=0
VERBOSE=0

# ONLINE

## speechbrain
#python household_simulation.py --config configs/online/config_speechbrain_trimmed_2sec_cos_emb_avg.yaml --debug $DEBUG --verbose $VERBOSE
#python household_simulation.py --config configs/online/config_speechbrain_trimmed_2sec_score_cos_sc_avg.yaml --debug $DEBUG --verbose $VERBOSE
#python household_simulation.py --config configs/online/config_speechbrain_trimmed_2sec_score_plda_sph.yaml --debug $DEBUG --verbose $VERBOSE

# clova
python household_simulation.py --config configs/online/config_clova_2sec_cos_emb_avg.yaml --debug $DEBUG --verbose $VERBOSE
python household_simulation.py --config configs/online/config_clova_2sec_score_cos_sc_avg.yaml --debug $DEBUG --verbose $VERBOSE
python household_simulation.py --config configs/online/config_clova_2sec_score_plda_sph.yaml --debug $DEBUG --verbose $VERBOSE

## xvector
#python household_simulation.py --config configs/online/config_xvector_trimmed_2sec_score_plda_diag.yaml --debug $DEBUG --verbose $VERBOSE
#python household_simulation.py --config configs/online/config_xvector_trimmed_2sec_score_plda_diag_by-the-book.yaml --debug $DEBUG --verbose $VERBOSE
## no cosine for xvector
