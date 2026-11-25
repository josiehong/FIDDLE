#!/usr/bin/env bash
# External benchmarks with FIDDLE alone and with BUDDY and SIRIUS, on all
# compounds. CASMI 2016 (Q Exactive) and EMBL-MCF 2.0 (Orbitrap Exploris) use
# the Orbitrap models, CASMI 2017 the Q-TOF models. Run from the repository
# root; the raw data are downloaded to data/benchmark/ from MoNA and
# casmi-contest.org (files already there are kept). No training data is read.
# MODEL_TAG selects local checkpoints (see compare.sh). Ends with a summary
# table and figure in result/; experiments_test_benchmark_unseen.sh draws the
# documentation's figure, with unseen compounds as a second row.
# --fiddle_only (the only argument) runs FIDDLE alone and needs nothing but
# msfiddle; otherwise
# BUDDY and SIRIUS run as well (install_buddy.sh, install_sirius.sh).
set -e
source running_scripts/compare.sh

for arg in "$@"; do
    case $arg in
        --fiddle_only) export FIDDLE_ONLY=1 ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done
methods=""
if [ -n "$FIDDLE_ONLY" ]; then
    methods="--methods fiddle"
fi

python scripts/download_benchmarks.py --out_dir ./data/benchmark
python scripts/casmi2mgf.py --data_config_path ./msfiddle/config/fiddle_tcn_casmi.yml
python scripts/embl2mgf.py \
    --raw_path ./data/benchmark/embl/MoNA-export-EMBL-MCF_2.0_HRMS_Library.sdf \
    --mgf_path ./data/embl.mgf \
    --data_config_path ./msfiddle/config/fiddle_tcn_embl.yml

compare ./data/casmi2016.mgf orbitrap casmi2016
compare ./data/casmi2017.mgf qtof casmi2017 --config_path ./msfiddle/config/fiddle_tcn_casmi.yml
compare ./data/embl.mgf orbitrap embl

python scripts/plot_benchmarks.py $methods --csv ./result/benchmarks_all.csv \
    --figure ./result/benchmarks_all.png
