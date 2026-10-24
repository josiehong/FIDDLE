#!/usr/bin/env bash
# Robustness to noise: add random noise peaks to held-out test spectra and
# predict, alone and with BUDDY and SIRIUS. Run from the repository root.
set -e
source running_scripts/compare.sh

# DATA_TAG must name the data build the models were trained on, so the test
# compounds stay unseen; MODEL_TAG selects local checkpoints (see compare.sh).
DATA=./data/cl_pkl_${DATA_TAG:-102026}

for INST in qtof orbitrap; do
    python scripts/prepare_msms_noised.py \
        --mgf_data $DATA/${INST}_maxmin_test.mgf \
        --config_path ./msfiddle/config/fiddle_tcn_$INST.yml \
        --out_path ./data/noised/noised_${INST}_test.mgf \
        --noise_times 1 2 3 4 5 --num_spectra 1000

    compare ./data/noised/noised_${INST}_test.mgf $INST ${INST}_noised
done
