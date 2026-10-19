#!/usr/bin/env bash
# Chimeric spectra: mix pairs of held-out test spectra at 90/70/50% and
# predict with the released models, alone and combined with BUDDY and SIRIUS.
# Run from the repository root after `msfiddle download` and the data step of
# train_release_v2.2.sh.
set -e
source running_scripts/compare.sh

DATA=./data/cl_pkl_082426

for INST in qtof orbitrap; do
    python scripts/prepare_msms_chimeric.py \
        --mgf_data $DATA/${INST}_maxmin_test.mgf \
        --config_path ./msfiddle/config/fiddle_tcn_$INST.yml \
        --out_dir ./data/chimeric/ --mix_ratios 90 70 50 --num_pairs 1000 --mz_tol 0.005

    for RATIO in 90 70 50; do
        compare ./data/chimeric/${INST}_maxmin_test_chimeric$RATIO.mgf $INST ${INST}_chimeric$RATIO
    done
done
