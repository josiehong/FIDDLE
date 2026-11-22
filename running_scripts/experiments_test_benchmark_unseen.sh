#!/usr/bin/env bash
# The external benchmarks on unseen compounds only: those absent from the
# models' training and validation data. Run experiments_test_benchmark.sh
# first: this script reuses its benchmark MGF files. Run from the repository
# root. MODEL_TAG selects local checkpoints (see compare.sh). Ends with the
# figure of the documentation, result/benchmarks.png: all compounds above,
# unseen compounds below (for a release, copy it to docs/_static/).
#
# Which compounds are unseen comes from MANIFEST (default the release's,
# data/benchmark/unseen_v2.5.0.csv), so the script runs anywhere and gives the
# release's results. For a new release, run it with --rebuild_manifest where
# the training data are (MGF files in data/cl_pkl_<DATA_TAG>/, default
# v2.5): it splits by them and writes the split to MANIFEST; set DATA_TAG
# and MANIFEST to the release's data and version.
# --fiddle_only runs FIDDLE alone, as for experiments_test_benchmark.sh.
set -e
source running_scripts/compare.sh

rebuild=""
for arg in "$@"; do
    case $arg in
        --fiddle_only) export FIDDLE_ONLY=1 ;;
        --rebuild_manifest) rebuild=1 ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done
methods=""
if [ -n "$FIDDLE_ONLY" ]; then
    methods="--methods fiddle"
fi

DATA=./data/cl_pkl_${DATA_TAG:-v2.5}
MANIFEST=${MANIFEST:-./data/benchmark/unseen_v2.5.0.csv}
for f in ./data/casmi2016.mgf ./data/casmi2017.mgf ./data/embl.mgf; do
    [ -f "$f" ] || { echo "$f is missing; run experiments_test_benchmark.sh first"; exit 1; }
done

if [ -n "$rebuild" ]; then
    for f in $DATA/{orbitrap,qtof}_maxmin_{train,val}.mgf; do
        [ -f "$f" ] || { echo "$f is missing; --rebuild_manifest needs the training data"; exit 1; }
    done
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
    echo "$MANIFEST is missing; build it with --rebuild_manifest"; exit 1
fi

compare ./data/casmi2016_unseen.mgf orbitrap casmi2016_unseen
compare ./data/casmi2017_unseen.mgf qtof casmi2017_unseen --config_path ./msfiddle/config/fiddle_tcn_casmi.yml
compare ./data/embl_unseen.mgf orbitrap embl_unseen

python scripts/plot_benchmarks.py $methods --subsets all unseen \
    --csv ./result/benchmarks.csv --figure ./result/benchmarks.png
