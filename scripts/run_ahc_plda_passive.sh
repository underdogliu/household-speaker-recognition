
# Set DEBUG to 0 to save the scores
DEBUG=0
VERBOSE=0

# PASSIVE
# AHC-PLDA

python3 household_simulation.py --config configs/passive/config_clova_2sec_ahc_plda.yaml --debug $DEBUG --verbose $VERBOSE
python3 household_simulation.py --config configs/passive/config_speechbrain_trimmed_2sec_ahc_plda.yaml --debug $DEBUG --verbose $VERBOSE
python3 household_simulation.py --config configs/passive/config_xvector_trimmed_2sec_ahc_plda.yaml --debug $DEBUG --verbose $VERBOSE