#!/usr/bin/env bash
# External benchmarks with FIDDLE alone and with BUDDY and SIRIUS, on all
# compounds and on those absent from the models' training data (*_unseen).
# CASMI 2016 (Q Exactive) and EMBL-MCF 2.0 (Orbitrap Exploris) use the
# Orbitrap models, CASMI 2017 the Q-TOF models. Run from the repository root
# with the raw data in data/benchmark/{casmi2016,casmi2017,embl}/. DATA_TAG
# names the data build the models were trained on; MODEL_TAG selects local
# checkpoints (see compare.sh). Ends with a summary table and figure for the
# docs.
set -e
source running_scripts/compare.sh

DATA=./data/cl_pkl_${DATA_TAG:-102026}

# Training and validation data of an instrument's models: the MGF exports,
# scanned for their SMILES, or the much larger pickles when they are missing
train_data() {
    local split
    for split in train val; do
        if [ -f "$DATA/$1_maxmin_$split.mgf" ]; then
            echo "$DATA/$1_maxmin_$split.mgf"
        else
            echo "$DATA/$1_maxmin_$split.pkl"
        fi
    done
}

python scripts/casmi2mgf.py --data_config_path ./msfiddle/config/fiddle_tcn_casmi.yml
python scripts/embl2mgf.py \
    --raw_path ./data/benchmark/embl/MoNA-export-EMBL-MCF_2.0_HRMS_Library.sdf \
    --mgf_path ./data/embl.mgf \
    --data_config_path ./msfiddle/config/fiddle_tcn_embl.yml
python scripts/remove_training_compounds.py \
    --train_data $(train_data orbitrap) \
    --mgf ./data/casmi2016.mgf ./data/embl.mgf
python scripts/remove_training_compounds.py \
    --train_data $(train_data qtof) \
    --mgf ./data/casmi2017.mgf

for MGF in casmi2016 casmi2016_unseen; do
    compare ./data/$MGF.mgf orbitrap $MGF
done
for MGF in casmi2017 casmi2017_unseen; do
    compare ./data/$MGF.mgf qtof $MGF --config_path ./msfiddle/config/fiddle_tcn_casmi.yml
done
for MGF in embl embl_unseen; do
    compare ./data/$MGF.mgf orbitrap $MGF --config_path ./msfiddle/config/fiddle_tcn_embl.yml
done

python scripts/plot_benchmarks.py --csv ./result/benchmarks.csv \
    --figure ./docs/_static/benchmarks.png
