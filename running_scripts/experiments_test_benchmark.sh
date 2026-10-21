#!/usr/bin/env bash
# External benchmarks (CASMI 2016, CASMI 2017, EMBL-MCF) with the Q-TOF models,
# alone and with BUDDY and SIRIUS. Run from the repository root with the raw
# data in data/benchmark/{casmi2016,casmi2017,embl}/. MODEL_TAG selects local
# checkpoints (see compare.sh).
set -e
source running_scripts/compare.sh

python scripts/casmi2mgf.py --data_config_path ./msfiddle/config/fiddle_tcn_casmi.yml
python scripts/embl2mgf.py \
    --raw_path ./data/benchmark/embl/MoNA-export-EMBL-MCF_2.0_HRMS_Library.sdf \
    --mgf_path ./data/embl.mgf \
    --data_config_path ./msfiddle/config/fiddle_tcn_embl.yml

compare ./data/casmi2016.mgf qtof casmi2016 --config_path ./msfiddle/config/fiddle_tcn_casmi.yml
compare ./data/casmi2017.mgf qtof casmi2017 --config_path ./msfiddle/config/fiddle_tcn_casmi.yml
compare ./data/embl.mgf qtof embl --config_path ./msfiddle/config/fiddle_tcn_embl.yml
