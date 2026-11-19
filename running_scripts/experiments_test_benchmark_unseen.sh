#!/usr/bin/env bash
# The external benchmarks on unseen compounds only: those absent from the
# models' training and validation data. Needs those data as MGF files in
# data/cl_pkl_<DATA_TAG>/, so it runs only where the models were trained; the
# all-compounds benchmark (experiments_test_benchmark.sh) needs no training
# data. Run that script first: this one reuses its benchmark MGF files. Run
# from the repository root. MODEL_TAG selects local checkpoints (see
# compare.sh). Ends with the documentation's figure: all compounds above,
# unseen compounds below.
set -e
source running_scripts/compare.sh

DATA=./data/cl_pkl_${DATA_TAG:-102026}
for f in ./data/casmi2016.mgf ./data/casmi2017.mgf ./data/embl.mgf; do
    [ -f "$f" ] || { echo "$f is missing; run experiments_test_benchmark.sh first"; exit 1; }
done

python scripts/remove_training_compounds.py \
    --train_data $DATA/orbitrap_maxmin_{train,val}.mgf \
    --mgf ./data/casmi2016.mgf ./data/embl.mgf
python scripts/remove_training_compounds.py \
    --train_data $DATA/qtof_maxmin_{train,val}.mgf \
    --mgf ./data/casmi2017.mgf

compare ./data/casmi2016_unseen.mgf orbitrap casmi2016_unseen
compare ./data/casmi2017_unseen.mgf qtof casmi2017_unseen --config_path ./msfiddle/config/fiddle_tcn_casmi.yml
compare ./data/embl_unseen.mgf orbitrap embl_unseen --config_path ./msfiddle/config/fiddle_tcn_embl.yml

python scripts/plot_benchmarks.py --subsets all unseen \
    --csv ./result/benchmarks.csv --figure ./docs/_static/benchmarks.png
