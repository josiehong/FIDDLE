#!/usr/bin/env bash
# Train the v2.3 models and evaluate them on the held-out test sets.
#
#   Q-TOF:    rebuild the data (it includes Waters again), train the TCN, then
#             the rescorer. The MaxMin test split changes with the data, so
#             Q-TOF results do not compare directly with v2.2.
#   Orbitrap: data and TCN unchanged from v2.2 (cl_pkl_082426); only the
#             rescorer is retrained. It runs in the background from the start.
#   Rescorers: pairs come from the TCN training set, with a share of training
#             compounds held out as validation. SEEDS rescorers are trained and
#             the one with the best validation top-1 is kept, so the test set
#             is used only for the final evaluation.
#
# Each step skips work whose output exists, so a rerun resumes. Run with
#   nohup bash running_scripts/train_release_v2.3.sh > train_v2.3.log 2>&1 &
# Env overrides: QTOF_DATA_TAG, SEEDS, NUM_WORKERS, DEVICE_QTOF (space-separated
# GPU list for the Q-TOF TCN), DEVICE_ORBITRAP (one GPU).
# Memory: the Q-TOF TCN holds about 100 GB of training data in RAM and the
# Orbitrap pair preparation about 50 GB, and they run at the same time.
# Package the release checkpoints with scripts/package_release_checkpoints.py.
set -e

QTOF_DATA_TAG=${QTOF_DATA_TAG:-102026}
SEEDS=${SEEDS:-"1 2 3"}
NUM_WORKERS=${NUM_WORKERS:-8}
DEVICE_QTOF=${DEVICE_QTOF:-"0 1"}
DEVICE_ORBITRAP=${DEVICE_ORBITRAP:-2}

QTOF_PKL=./data/cl_pkl_$QTOF_DATA_TAG
ORBITRAP_PKL=./data/cl_pkl_082426
TCN_QTOF=./check_point/fiddle_tcn_qtof_${QTOF_DATA_TAG}_muon.pt
TCN_ORBITRAP=./check_point/fiddle_tcn_orbitrap_082426_muon.pt
RESCORE_ORBITRAP_V22=./check_point/fiddle_rescore_orbitrap_082426_muon.pt

# rescore INST PKL_DIR TCN GPU: build pairs, train SEEDS rescorers, keep the
# best by validation top-1 as fiddle_rescore_<INST>_<tag>_v2.3.pt
rescore() {
    local inst=$1 pkl=$2 tcn=$3 gpu=$4
    local tag
    tag=$(basename "$tcn" .pt)
    tag=${tag#fiddle_tcn_${inst}_}
    local dir=./data/rescore_${tag}_v2.3
    local config=./msfiddle/config/fiddle_tcn_${inst}.yml
    local final=./check_point/fiddle_rescore_${inst}_${tag}_v2.3.pt
    [ -s "$final" ] && return

    if [ ! -s "$dir/${inst}_maxmin_rescore_val.pkl" ]; then
        CUDA_VISIBLE_DEVICES=$gpu python -u scripts/prepare_augment_rescore.py \
            --train_data "$pkl/${inst}_maxmin_train.pkl" --config_path "$config" \
            --resume_path "$tcn" --rescore_dir "$dir" \
            --pos_cap 10 --neg_per_pos 8 --tolerance 50 \
            --num_workers "$NUM_WORKERS" --device 0
    fi
    for seed in $SEEDS; do
        [ -s "${final%.pt}_seed$seed.pt" ] ||
            CUDA_VISIBLE_DEVICES=$gpu python -u scripts/train_rescore.py \
                --train_data "$dir/${inst}_maxmin_rescore_train.pkl" \
                --val_data "$dir/${inst}_maxmin_rescore_val.pkl" \
                --config_path "$config" --resume_path "$tcn" \
                --checkpoint_path "${final%.pt}_seed$seed.pt" --seed "$seed" --device 0
    done
    python - "${final%.pt}"_seed*.pt "$final" <<'PY'
import shutil, sys, torch
*seeds, final = sys.argv[1:]
top1 = {p: torch.load(p, map_location="cpu", weights_only=False)["best_val_top1"] for p in seeds}
for p, t in top1.items():
    print(f"{p}: validation top-1 {t:.4f}")
best = max(top1, key=top1.get)
shutil.copy(best, final)
print(f"Kept {best} as {final}")
PY
}

# predict_eval INST PKL_DIR TCN RESCORE NAME GPU
predict_eval() {
    CUDA_VISIBLE_DEVICES=$6 python -m msfiddle predict \
        --test_data "$2/$1_maxmin_test.mgf" \
        --config_path "./msfiddle/config/fiddle_tcn_$1.yml" \
        --resume_path "$3" --rescore_resume_path "$4" \
        --result_path "./result/fiddle_$1_$5.csv" \
        --device 0 --batch_size 64 --num_workers "$NUM_WORKERS" --eval
}

# Orbitrap: rescorer only, in the background
(
    rescore orbitrap $ORBITRAP_PKL $TCN_ORBITRAP "$DEVICE_ORBITRAP"
    predict_eval orbitrap $ORBITRAP_PKL $TCN_ORBITRAP $RESCORE_ORBITRAP_V22 \
        v2.2_rescore "$DEVICE_ORBITRAP"
    predict_eval orbitrap $ORBITRAP_PKL $TCN_ORBITRAP \
        ./check_point/fiddle_rescore_orbitrap_082426_muon_v2.3.pt v2.3 "$DEVICE_ORBITRAP"
) > orbitrap_v2.3.log 2>&1 &
orbitrap_job=$!

# Q-TOF: data, TCN, rescorer, evaluation
if [ ! -s "$QTOF_PKL/qtof_maxmin_test.pkl" ]; then
    mkdir -p "$QTOF_PKL"
    python -u scripts/prepare_msms.py \
        --dataset agilent nist20 nist23 mona waters gnps --instrument_type qtof \
        --config_path ./msfiddle/config/fiddle_tcn_qtof.yml \
        --num_workers "$NUM_WORKERS" --pkl_dir "$QTOF_PKL/" --maxmin_pick \
        > "prep_qtof_$QTOF_DATA_TAG.log" 2>&1
fi
if [ ! -s "$TCN_QTOF" ]; then
    n_gpus=$(echo $DEVICE_QTOF | wc -w)
    CUDA_VISIBLE_DEVICES=${DEVICE_QTOF// /,} python -u scripts/train_tcn_gpus_cl.py \
        --train_data "$QTOF_PKL/qtof_maxmin_train.pkl" \
        --test_data "$QTOF_PKL/qtof_maxmin_test.pkl" \
        --config_path ./msfiddle/config/fiddle_tcn_qtof.yml \
        --checkpoint_path "$TCN_QTOF" \
        --result_path "./result/fiddle_tcn_qtof_${QTOF_DATA_TAG}_muon.csv" \
        --device $(seq -s' ' 0 $((n_gpus - 1))) > "tcn_qtof_${QTOF_DATA_TAG}_muon.log" 2>&1
fi
(
    rescore qtof "$QTOF_PKL" "$TCN_QTOF" "${DEVICE_QTOF%% *}"
    predict_eval qtof "$QTOF_PKL" "$TCN_QTOF" \
        "./check_point/fiddle_rescore_qtof_${QTOF_DATA_TAG}_muon_v2.3.pt" v2.3 "${DEVICE_QTOF%% *}"
) > qtof_v2.3.log 2>&1

wait $orbitrap_job
echo "Done. Results: grep -A4 '^Eval against' qtof_v2.3.log orbitrap_v2.3.log"
