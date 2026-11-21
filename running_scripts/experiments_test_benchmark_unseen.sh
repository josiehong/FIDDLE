#!/usr/bin/env bash
# The external benchmarks on unseen compounds only: those absent from the
# models' training and validation data. Run experiments_test_benchmark.sh
# first: this script reuses its benchmark MGF files. Run from the repository
# root. MODEL_TAG selects local checkpoints (see compare.sh). Ends with the
# documentation's figure: all compounds above, unseen compounds below.
#
# Which compounds are unseen comes from MANIFEST (default the release's,
# benchmarks/unseen_v2.4.0.csv), so the script runs anywhere. Where the
# training data are (MGF files in data/cl_pkl_<DATA_TAG>/, default 112026),
# it splits by them instead and rewrites the manifest; do that for each new
# release, with DATA_TAG and MANIFEST set to its data and version.
set -e
source running_scripts/compare.sh

DATA=./data/cl_pkl_${DATA_TAG:-112026}
MANIFEST=${MANIFEST:-./benchmarks/unseen_v2.4.0.csv}
for f in ./data/casmi2016.mgf ./data/casmi2017.mgf ./data/embl.mgf; do
    [ -f "$f" ] || { echo "$f is missing; run experiments_test_benchmark.sh first"; exit 1; }
done

training=($DATA/{orbitrap,qtof}_maxmin_{train,val}.mgf)
if ls "${training[@]}" > /dev/null 2>&1; then
    echo "Splitting by the training data in $DATA; saving the split to $MANIFEST"
    python scripts/remove_training_compounds.py \
        --train_data $DATA/orbitrap_maxmin_{train,val}.mgf \
        --mgf ./data/casmi2016.mgf ./data/embl.mgf --manifest_out "$MANIFEST"
    python scripts/remove_training_compounds.py \
        --train_data $DATA/qtof_maxmin_{train,val}.mgf \
        --mgf ./data/casmi2017.mgf --manifest_out "$MANIFEST"
elif [ -f "$MANIFEST" ]; then
    echo "Splitting by $MANIFEST"
    python scripts/remove_training_compounds.py --manifest "$MANIFEST" \
        --mgf ./data/casmi2016.mgf ./data/casmi2017.mgf ./data/embl.mgf
else
    echo "Neither the training data ($DATA) nor $MANIFEST found"; exit 1
fi

compare ./data/casmi2016_unseen.mgf orbitrap casmi2016_unseen
compare ./data/casmi2017_unseen.mgf qtof casmi2017_unseen --config_path ./msfiddle/config/fiddle_tcn_casmi.yml
compare ./data/embl_unseen.mgf orbitrap embl_unseen

python scripts/plot_benchmarks.py --subsets all unseen \
    --csv ./result/benchmarks.csv --figure ./docs/_static/benchmarks.png
