#!/usr/bin/env bash
# External benchmarks (CASMI 2016, CASMI 2017, EMBL-MCF 2.0) with the released
# Q-TOF models, alone and combined with BUDDY and SIRIUS. Run from the
# repository root after `msfiddle download`; MODEL_TAG=102026_muon tests the
# v2.3 models from train_release_v2.3.sh instead (see compare.sh).
#
# Raw data:
#   data/benchmark/casmi2016/MoNA-export-CASMI_2016.sdf
#   data/benchmark/casmi2017/CASMI-solutions.csv, Chal1to45Summary.csv and
#       challenges-001-045-msms-mgf-20170908/ (unzipped)
#   data/benchmark/embl/MoNA-export-EMBL-MCF_2.0_HRMS_Library.sdf
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
