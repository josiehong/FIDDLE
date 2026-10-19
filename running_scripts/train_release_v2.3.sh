#!/usr/bin/env bash
# Train the v2.3 models from scratch and evaluate them.
#
# Per instrument (Q-TOF and Orbitrap run in parallel):
#   1. Build the data: train, validation and test sets split by compound. The
#      MaxMin test set is the share of compounds MaxMin leaves out; the
#      validation set is a random share of the rest (prepare_msms.py --val_ratio).
#   2. Train the TCN, with early stopping and checkpoint choice on validation.
#   3. Build rescore pairs (training pairs from the training set, validation
#      pairs from the validation set), train SEEDS rescorers and keep the one
#      with the best validation top-1.
#   4. Evaluate on the test set; this is the only use of the test set.
#
# Each step skips work whose output is complete, and an interrupted TCN resumes
# from its last-epoch checkpoint, so a rerun picks up where it stopped. Run with
#   nohup bash running_scripts/train_release_v2.3.sh > train_v2.3.log 2>&1 &
# Env overrides: DATA_TAG, SEEDS, NUM_WORKERS, DEVICE_QTOF / DEVICE_ORBITRAP
# (space-separated GPU lists; the rescore and evaluation steps use the first).
# Memory: both TCNs hold their training data in RAM (Q-TOF about 100 GB,
# Orbitrap about 80 GB) and train at the same time.
# Package the release checkpoints with scripts/package_release_checkpoints.py.
set -e

DATA_TAG=${DATA_TAG:-102026}
SEEDS=${SEEDS:-"1 2 3"}
NUM_WORKERS=${NUM_WORKERS:-8}
DEVICE_QTOF=${DEVICE_QTOF:-"0 1"}
DEVICE_ORBITRAP=${DEVICE_ORBITRAP:-"2 3"}
PKL_DIR=./data/cl_pkl_$DATA_TAG
TAG=${DATA_TAG}_muon

# pipeline INST DEVICES TCN_EXTRA_ARGS DATASET...
pipeline() {
    local inst=$1 devices=$2 tcn_extra=$3
    shift 3
    local config=./msfiddle/config/fiddle_tcn_$inst.yml
    local data=$PKL_DIR/${inst}_maxmin
    local tcn=./check_point/fiddle_tcn_${inst}_$TAG.pt
    local rescore=./check_point/fiddle_rescore_${inst}_$TAG.pt
    local rescore_dir=./data/rescore_$TAG
    local gpu=${devices%% *}

    # 1. Data (train_pairs.pkl is written last)
    if [ ! -s "${data}_train_pairs.pkl" ]; then
        python -u scripts/prepare_msms.py --dataset "$@" --instrument_type "$inst" \
            --config_path "$config" --num_workers "$NUM_WORKERS" \
            --pkl_dir "$PKL_DIR/" --maxmin_pick
    fi

    # 2. TCN. The trainer saves its best checkpoint as it goes and writes the
    # result CSV when training ends, so the CSV marks a finished run.
    local tcn_done=./result/fiddle_tcn_${inst}_$TAG.csv
    if [ ! -s "$tcn_done" ]; then
        local resume=()
        [ -s "${tcn%.pt}_last.pt" ] && resume=(--resume_path "${tcn%.pt}_last.pt")
        local n_gpus
        n_gpus=$(echo $devices | wc -w)
        CUDA_VISIBLE_DEVICES=${devices// /,} python -u scripts/train_tcn_gpus_cl.py \
            --train_data "${data}_train.pkl" --val_data "${data}_val.pkl" \
            --config_path "$config" --checkpoint_path "$tcn" --result_path "$tcn_done" \
            "${resume[@]}" $tcn_extra --device $(seq -s' ' 0 $((n_gpus - 1)))
        # Rescore pairs come from TCN predictions: drop any built from an
        # earlier TCN, with the rescorers trained on them
        rm -f "$rescore_dir/${inst}_maxmin_rescore_"*.pkl "${rescore%.pt}"*.pt
    fi

    # 3. Rescorer: pairs (train pkl is written last), then SEEDS runs
    if [ ! -s "$rescore" ]; then
        if [ ! -s "$rescore_dir/${inst}_maxmin_rescore_train.pkl" ]; then
            CUDA_VISIBLE_DEVICES=$gpu python -u scripts/prepare_augment_rescore.py \
                --train_data "${data}_train.pkl" --val_data "${data}_val.pkl" \
                --config_path "$config" --resume_path "$tcn" \
                --rescore_dir "$rescore_dir" \
                --pos_cap 10 --neg_per_pos 8 --tolerance 50 \
                --num_workers "$NUM_WORKERS" --device 0
        fi
        for seed in $SEEDS; do
            local ckpt=${rescore%.pt}_seed$seed.pt
            [ -s "$ckpt" ] && continue
            # Train to a temporary file so an interrupted seed is retrained
            CUDA_VISIBLE_DEVICES=$gpu python -u scripts/train_rescore.py \
                --train_data "$rescore_dir/${inst}_maxmin_rescore_train.pkl" \
                --val_data "$rescore_dir/${inst}_maxmin_rescore_val.pkl" \
                --config_path "$config" --resume_path "$tcn" \
                --checkpoint_path "$ckpt.part" --seed "$seed" --device 0
            mv "$ckpt.part" "$ckpt"
        done
        python - "${rescore%.pt}"_seed*.pt "$rescore" <<'PY'
import shutil, sys, torch
*seeds, final = sys.argv[1:]
top1 = {p: torch.load(p, map_location="cpu", weights_only=False)["best_val_top1"] for p in seeds}
for p, t in top1.items():
    print(f"{p}: validation top-1 {t:.4f}")
best = max(top1, key=top1.get)
shutil.copy(best, final)
print(f"Kept {best} as {final}")
PY
    fi

    # 4. Test set evaluation
    CUDA_VISIBLE_DEVICES=$gpu python -m msfiddle predict \
        --test_data "${data}_test.mgf" --config_path "$config" \
        --resume_path "$tcn" --rescore_resume_path "$rescore" \
        --result_path "./result/fiddle_${inst}_$TAG.csv" \
        --device 0 --batch_size 64 --num_workers "$NUM_WORKERS" --eval
}

mkdir -p "$PKL_DIR" ./result
pipeline qtof "$DEVICE_QTOF" "" agilent nist20 nist23 mona waters gnps \
    > "qtof_$TAG.log" 2>&1 &
qtof_job=$!
pipeline orbitrap "$DEVICE_ORBITRAP" "--amp" nist20 nist23 mona gnps \
    > "orbitrap_$TAG.log" 2>&1 &
orbitrap_job=$!

status=0
wait $qtof_job || { echo "Q-TOF pipeline failed; see qtof_$TAG.log"; status=1; }
wait $orbitrap_job || { echo "Orbitrap pipeline failed; see orbitrap_$TAG.log"; status=1; }
echo "Results: grep -A4 '^Eval against' qtof_$TAG.log orbitrap_$TAG.log"
exit $status
