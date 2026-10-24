#!/usr/bin/env bash
# Train and evaluate the v2.3 models; the instruments run in parallel:
#   1. Build train, validation and test sets, split by compound.
#   2. Train the TCN, choosing the checkpoint on validation.
#   3. Build rescore pairs, train SEEDS rescorers and keep the best on validation.
#   4. Evaluate on the test set, its only use.
# A rerun skips finished steps and resumes an interrupted TCN:
#   nohup bash running_scripts/train_release_v2.3.sh > train_v2.3.log 2>&1 &
# Env: INSTRUMENTS, DATA_TAG, SEEDS, NUM_WORKERS, DEVICE_QTOF / DEVICE_ORBITRAP /
# DEVICE_TIMSTOF (space-separated GPUs; rescore and evaluation use the first),
# TIMSTOF_TRANSFER_FROM (a Q-TOF TCN to fine-tune from) and
# CASMI2026_MAX_PER_COMPOUND (fewer timsTOF spectra, to save RAM).
# Each TCN keeps its training data in RAM. Update this file only through git
# while it runs: bash reads it as it goes.
set -e

DATA_TAG=${DATA_TAG:-102026}
SEEDS=${SEEDS:-"1 2 3"}
NUM_WORKERS=${NUM_WORKERS:-8}
DEVICE_QTOF=${DEVICE_QTOF:-"0 1"}
DEVICE_ORBITRAP=${DEVICE_ORBITRAP:-"2 3"}
DEVICE_TIMSTOF=${DEVICE_TIMSTOF:-"4 5"}
INSTRUMENTS=${INSTRUMENTS:-"qtof orbitrap timstof"}
CASMI2026_PARQUET=./data/origin/casmi2026/train.parquet
CASMI2026_MAX_PER_COMPOUND=${CASMI2026_MAX_PER_COMPOUND:-0}
TIMSTOF_TRANSFER_FROM=${TIMSTOF_TRANSFER_FROM:-""}  # empty: from scratch
PKL_DIR=./data/cl_pkl_$DATA_TAG
TAG=${DATA_TAG}_muon

# pipeline INST DEVICES TCN_EXTRA_ARGS DATASET...
# Env: SPLIT (default maxmin), PREP_EXTRA (more prepare_msms.py arguments),
# TRANSFER_FROM (a TCN checkpoint to fine-tune from)
pipeline() {
    local inst=$1 devices=$2 tcn_extra=$3
    shift 3
    local split=${SPLIT:-maxmin}
    local config=./msfiddle/config/fiddle_tcn_$inst.yml
    local data=$PKL_DIR/${inst}_$split
    local tcn=./check_point/fiddle_tcn_${inst}_$TAG.pt
    local rescore=./check_point/fiddle_rescore_${inst}_$TAG.pt
    local rescore_dir=./data/rescore_$TAG
    local gpu=${devices%% *}

    # 1. Data (train_pairs.pkl is written last)
    if [ ! -s "${data}_train_pairs.pkl" ]; then
        python -u scripts/prepare_msms.py --dataset "$@" --instrument_type "$inst" \
            --config_path "$config" --num_workers "$NUM_WORKERS" \
            --pkl_dir "$PKL_DIR/" --split "$split" $PREP_EXTRA
    fi

    # 2. TCN; the result CSV marks a finished run
    local tcn_done=./result/fiddle_tcn_${inst}_$TAG.csv
    if [ ! -s "$tcn_done" ]; then
        local resume=()
        if [ -s "${tcn%.pt}_last.pt" ]; then
            resume=(--resume_path "${tcn%.pt}_last.pt")
        elif [ -n "$TRANSFER_FROM" ]; then
            resume=(--resume_path "$TRANSFER_FROM" --transfer)
        fi
        local n_gpus
        n_gpus=$(echo $devices | wc -w)
        CUDA_VISIBLE_DEVICES=${devices// /,} python -u scripts/train_tcn_gpus_cl.py \
            --train_data "${data}_train.pkl" --val_data "${data}_val.pkl" \
            --config_path "$config" --checkpoint_path "$tcn" --result_path "$tcn_done" \
            "${resume[@]}" $tcn_extra --device $(seq -s' ' 0 $((n_gpus - 1)))
        # Drop rescore pairs and rescorers built on an earlier TCN
        rm -f "$rescore_dir/${inst}_${split}_rescore_"*.pkl "${rescore%.pt}"*.pt
    fi

    # 3. Rescorer: pairs (train pkl is written last), then SEEDS runs
    if [ ! -s "$rescore" ]; then
        if [ ! -s "$rescore_dir/${inst}_${split}_rescore_train.pkl" ]; then
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
                --train_data "$rescore_dir/${inst}_${split}_rescore_train.pkl" \
                --val_data "$rescore_dir/${inst}_${split}_rescore_val.pkl" \
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
        --test_data "${data}_test.mgf" --instrument_type "$inst" --config_path "$config" \
        --resume_path "$tcn" --rescore_resume_path "$rescore" \
        --result_path "./result/fiddle_${inst}_$TAG.csv" \
        --device 0 --batch_size 64 --num_workers "$NUM_WORKERS" --eval
}

mkdir -p "$PKL_DIR" ./result
case " $INSTRUMENTS " in
    *" timstof "*) [ -s "$CASMI2026_PARQUET" ] || { echo "Missing $CASMI2026_PARQUET"; exit 1; } ;;
esac
jobs=()
for inst in $INSTRUMENTS; do
    case $inst in
        qtof)
            pipeline qtof "$DEVICE_QTOF" "" agilent nist20 nist23 mona waters gnps \
                > "qtof_$TAG.log" 2>&1 & ;;
        orbitrap)
            pipeline orbitrap "$DEVICE_ORBITRAP" "--amp" nist20 nist23 mona gnps \
                > "orbitrap_$TAG.log" 2>&1 & ;;
        timstof)
            SPLIT=scaffold TRANSFER_FROM=$TIMSTOF_TRANSFER_FROM \
            PREP_EXTRA="--casmi_parquet $CASMI2026_PARQUET --holdout_lib enveda-np-examples --max_spectra_per_compound $CASMI2026_MAX_PER_COMPOUND" \
            pipeline timstof "$DEVICE_TIMSTOF" "--amp" casmi2026 \
                > "timstof_$TAG.log" 2>&1 & ;;
        *) echo "Unknown instrument $inst"; exit 1 ;;
    esac
    jobs+=("$inst:$!")
done

status=0
for job in "${jobs[@]}"; do
    inst=${job%%:*}
    wait "${job#*:}" || { echo "$inst pipeline failed; see ${inst}_$TAG.log"; status=1; }
done
echo "Results: grep -A4 '^Eval against' *_$TAG.log"
exit $status
