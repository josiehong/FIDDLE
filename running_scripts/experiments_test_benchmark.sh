#!/usr/bin/env bash
# External benchmarks (CASMI 2016, CASMI 2017, EMBL-MCF 2.0) with the released
# Q-TOF models. Run from the repository root after `msfiddle download`.
#
# Raw data:
#   data/benchmark/casmi2016/MoNA-export-CASMI_2016.sdf
#   data/benchmark/casmi2017/CASMI-solutions.csv, Chal1to45Summary.csv and
#       challenges-001-045-msms-mgf-20170908/ (unzipped)
#   data/benchmark/embl/MoNA-export-EMBL-MCF_2.0_HRMS_Library.sdf
#
# FIDDLE + BUDDY runs read BUDDY output from ./run_buddy/<name>/ (see the docs
# on BUDDY and SIRIUS); they are skipped when it is missing.
set -e

python scripts/casmi2mgf.py --data_config_path ./msfiddle/config/fiddle_tcn_casmi.yml
python scripts/embl2mgf.py \
    --raw_path ./data/benchmark/embl/MoNA-export-EMBL-MCF_2.0_HRMS_Library.sdf \
    --mgf_path ./data/embl.mgf \
    --data_config_path ./msfiddle/config/fiddle_tcn_embl.yml

for name in casmi2016 casmi2017 embl; do
    config=./msfiddle/config/fiddle_tcn_casmi.yml
    [ "$name" = embl ] && config=./msfiddle/config/fiddle_tcn_embl.yml
    predict="python -m msfiddle predict --test_data ./data/$name.mgf \
        --instrument_type qtof --config_path $config --num_workers 8 --eval"

    $predict --result_path ./result/fiddle_$name.csv
    if [ -e ./run_buddy/$name ]; then
        $predict --buddy_path ./run_buddy/$name \
            --result_path ./result/fiddle_buddy_$name.csv
    fi
done
