
# Set DEBUG to 0 to save the scores
DEBUG=0
VERBOSE=0

# PASSIVE

## speechbrain
#python household_simulation.py --config configs/passive/config_speechbrain_trimmed_2sec_ahc.yaml --debug $DEBUG --verbose $VERBOSE
#python household_simulation.py --config configs/passive/config_speechbrain_trimmed_2sec_vb_plda.yaml --debug $DEBUG --verbose $VERBOSE
python household_simulation_mod.py --config configs/passive/config_speechbrain_trimmed_2sec_vb_plda.yaml --debug $DEBUG --verbose $VERBOSE
#
## clova
#python household_simulation.py --config configs/passive/config_clova_2sec_ahc.yaml --debug $DEBUG --verbose $VERBOSE
#python household_simulation.py --config configs/passive/config_clova_2sec_vb_plda_sph.yaml --debug $DEBUG --verbose $VERBOSE

# xvector
#python household_simulation.py --config configs/passive/config_xvector_trimmed_2sec_ahc.yaml --debug $DEBUG --verbose $VERBOSE
#python household_simulation.py --config configs/passive/config_xvector_trimmed_2sec_vb_plda.yaml --debug $DEBUG --verbose $VERBOSE
