#!/usr/bin/env bash
# External benchmarks (CASMI 2016, CASMI 2017, EMBL-MCF) with the Q-TOF models,
# alone and with BUDDY and SIRIUS, on all compounds and on those absent from
# the models' training data (*_unseen). Run from the repository root with the
# raw data in data/benchmark/{casmi2016,casmi2017,embl}/. DATA_TAG names the
# data build the models were trained on; MODEL_TAG selects local checkpoints
# (see compare.sh).
set -e
source running_scripts/compare.sh

DATA=./data/cl_pkl_${DATA_TAG:-082426}

python scripts/casmi2mgf.py --data_config_path ./msfiddle/config/fiddle_tcn_casmi.yml
python scripts/embl2mgf.py \
    --raw_path ./data/benchmark/embl/MoNA-export-EMBL-MCF_2.0_HRMS_Library.sdf \
    --mgf_path ./data/embl.mgf \
    --data_config_path ./msfiddle/config/fiddle_tcn_embl.yml
python scripts/remove_training_compounds.py \
    --train_data $(ls $DATA/qtof_maxmin_{train,val}.pkl 2>/dev/null) \
    --mgf ./data/casmi2016.mgf ./data/casmi2017.mgf ./data/embl.mgf

for BENCH in casmi2016 casmi2017 embl; do
    CONFIG=./msfiddle/config/fiddle_tcn_casmi.yml
    [ $BENCH = embl ] && CONFIG=./msfiddle/config/fiddle_tcn_embl.yml
    compare ./data/$BENCH.mgf qtof $BENCH --config_path $CONFIG
    compare ./data/${BENCH}_unseen.mgf qtof ${BENCH}_unseen --config_path $CONFIG
done
