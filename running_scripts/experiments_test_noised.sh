#!/usr/bin/env bash
# Robustness to noise: add 1-5x random noise peaks to held-out test spectra
# and predict with the released models, alone and combined with BUDDY and
# SIRIUS. Run from the repository root after `msfiddle download` and the data
# step of train_release_v2.2.sh.
set -e
source running_scripts/compare.sh

DATA=./data/cl_pkl_082426

for INST in qtof orbitrap; do
    python scripts/prepare_msms_noised.py \
        --mgf_data $DATA/${INST}_maxmin_test.mgf \
        --config_path ./msfiddle/config/fiddle_tcn_$INST.yml \
        --out_path ./data/noised/noised_${INST}_test.mgf \
        --noise_times 1 2 3 4 5 --num_spectra 1000

    compare ./data/noised/noised_${INST}_test.mgf $INST ${INST}_noised
done
