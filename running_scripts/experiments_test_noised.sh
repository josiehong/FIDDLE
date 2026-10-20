#!/usr/bin/env bash
# Robustness to noise: add 1-5x random noise peaks to held-out test spectra
# and predict with the released models, alone and combined with BUDDY and
# SIRIUS. Run from the repository root after `msfiddle download` and the data
# step of the release training script.
set -e
source running_scripts/compare.sh

# The test spectra must come from the data build the models were trained on,
# or test compounds may be training compounds: DATA_TAG 082426 for the v2.2
# models, 102026 for v2.3. Test v2.3 before its release with
#   DATA_TAG=102026 MODEL_TAG=102026_muon bash <this script>
DATA=./data/cl_pkl_${DATA_TAG:-082426}

for INST in qtof orbitrap; do
    python scripts/prepare_msms_noised.py \
        --mgf_data $DATA/${INST}_maxmin_test.mgf \
        --config_path ./msfiddle/config/fiddle_tcn_$INST.yml \
        --out_path ./data/noised/noised_${INST}_test.mgf \
        --noise_times 1 2 3 4 5 --num_spectra 1000

    compare ./data/noised/noised_${INST}_test.mgf $INST ${INST}_noised
done
