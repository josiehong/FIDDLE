#!/usr/bin/env bash
# Train and evaluate the v2.5 models, one instrument after another on all
# GPUs (PARALLEL=1 runs them at once; then give each instrument its own
# DEVICE_* GPUs, since all default to "0 1"):
#   1. Build train, validation and test sets, split by compound with MaxMin
#      picking for every instrument (SPLIT). The split is recorded next to
#      the TCN checkpoint; another split needs another DATA_TAG. The models train on the measured precursor m/z; as in the
#      paper, NIST20's and NIST23's, which are computed from the structure,
#      are replaced by the theoretical m/z plus a seeded Gaussian error with
#      a standard deviation of a third of the instrument's tolerance
#      (simulate_precursor_ppm in the configs). The v2.4 data (DATA_TAG
#      112026) encode the theoretical m/z instead: do not reuse them.
#   2. Train the TCN, choosing the checkpoint on validation.
#   3. Build rescore pairs, train SEEDS rescorers and keep the best on validation.
#   4. Evaluate on the test set, its only use.
# Needs the raw libraries in data/origin/ (file names in prepare_msms.py)
# and, for timsTOF, data/origin/casmi2026/train.parquet.
# A rerun skips finished steps 1-3, resumes an interrupted TCN and repeats
# the evaluation:
#   nohup bash running_scripts/train_release_v2.5.sh > train_v2.5.log 2>&1 &
# Env: INSTRUMENTS, DATA_TAG, SEEDS, NUM_WORKERS, PARALLEL, DEVICE_QTOF /
# DEVICE_ORBITRAP / DEVICE_TIMSTOF (space-separated GPUs, default "0 1", for
# two 11 GB GPUs; the TCN uses all of them, rescore and evaluation the first),
# TIMSTOF_TRANSFER_FROM (a Q-TOF TCN to fine-tune from) and
# CASMI2026_MAX_PER_COMPOUND (fewer timsTOF spectra, to save RAM),
# SIMULATION_SEED (of the precursor m/z simulation), and
# TCN_BATCH_SIZE / TCN_GRAD_ACCUM (the TCN's batch, default the config's).
# The TCNs train in fp32 with TF32 matrix multiplies: on an H100, fp16 AMP
# was 6x slower for this model.
# Each TCN keeps its training data in RAM. Update this file only through git
# while it runs: bash reads it as it goes.
set -e

DATA_TAG=${DATA_TAG:-v2.5}
SEEDS=${SEEDS:-"1 2 3"}
NUM_WORKERS=${NUM_WORKERS:-8}
PARALLEL=${PARALLEL:-0}
DEVICE_QTOF=${DEVICE_QTOF:-"0 1"}
DEVICE_ORBITRAP=${DEVICE_ORBITRAP:-"0 1"}
DEVICE_TIMSTOF=${DEVICE_TIMSTOF:-"0 1"}
INSTRUMENTS=${INSTRUMENTS:-"qtof orbitrap timstof"}
CASMI2026_PARQUET=./data/origin/casmi2026/train.parquet
CASMI2026_MAX_PER_COMPOUND=${CASMI2026_MAX_PER_COMPOUND:-0}
TIMSTOF_TRANSFER_FROM=${TIMSTOF_TRANSFER_FROM:-""}  # empty: from scratch
SIMULATION_SEED=${SIMULATION_SEED:-0}
# TCN batch: empty keeps the config's (256 pairs x 2 accumulated, for 11 GB
# GPUs); 512 x 1 is the same effective batch in one step on a large GPU
TCN_BATCH_SIZE=${TCN_BATCH_SIZE:-}
TCN_GRAD_ACCUM=${TCN_GRAD_ACCUM:-}
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

    # The checkpoint and result names leave out the split: record it, and
    # refuse to reuse a run trained on another split
    local split_mark=${tcn%.pt}.split
    if [ -s "$split_mark" ] && [ "$(cat "$split_mark")" != "$split" ]; then
        echo "$tcn was trained on the $(cat "$split_mark") split, not $split; set another DATA_TAG"
        return 1
    fi
    # A run with no mark has an unknown split: never stamp it as this one
    if [ ! -s "$split_mark" ] && { [ -s "$tcn" ] || [ -s "${tcn%.pt}_last.pt" ]; }; then
        echo "$tcn has no split record ($split_mark); set another DATA_TAG"
        return 1
    fi
    echo "$split" > "$split_mark"

    # 1. Data (train_pairs.pkl is written last)
    if [ ! -s "${data}_train_pairs.pkl" ]; then
        python -u scripts/prepare_msms.py --dataset "$@" --instrument_type "$inst" \
            --config_path "$config" --num_workers "$NUM_WORKERS" \
            --pkl_dir "$PKL_DIR/" --split "$split" \
            --simulation_seed "$SIMULATION_SEED" $PREP_EXTRA
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
        local batch=()
        [ -n "$TCN_BATCH_SIZE" ] && batch+=(--batch_size "$TCN_BATCH_SIZE")
        [ -n "$TCN_GRAD_ACCUM" ] && batch+=(--grad_accum_steps "$TCN_GRAD_ACCUM")
        local n_gpus
        n_gpus=$(echo $devices | wc -w)
        CUDA_VISIBLE_DEVICES=${devices// /,} python -u scripts/train_tcn_gpus_cl.py \
            --train_data "${data}_train.pkl" --val_data "${data}_val.pkl" \
            --config_path "$config" --checkpoint_path "$tcn" --result_path "$tcn_done" \
            "${resume[@]}" "${batch[@]}" $tcn_extra --device $(seq -s' ' 0 $((n_gpus - 1)))
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

mkdir -p "$PKL_DIR" ./result ./check_point
case " $INSTRUMENTS " in
    *" timstof "*) [ -s "$CASMI2026_PARQUET" ] || { echo "Missing $CASMI2026_PARQUET"; exit 1; } ;;
esac
run_instrument() {
    case $1 in
        qtof)
            pipeline qtof "$DEVICE_QTOF" "" agilent nist20 nist23 mona waters gnps ;;
        orbitrap)
            pipeline orbitrap "$DEVICE_ORBITRAP" "" nist20 nist23 mona gnps ;;
        timstof)
            TRANSFER_FROM=$TIMSTOF_TRANSFER_FROM \
            PREP_EXTRA="--casmi_parquet $CASMI2026_PARQUET --holdout_lib enveda-np-examples --max_spectra_per_compound $CASMI2026_MAX_PER_COMPOUND" \
            pipeline timstof "$DEVICE_TIMSTOF" "" casmi2026 ;;
        *) echo "Unknown instrument $1"; return 1 ;;
    esac
}

status=0
jobs=()
for inst in $INSTRUMENTS; do
    if [ "$PARALLEL" = 1 ]; then
        run_instrument "$inst" >> "${inst}_$TAG.log" 2>&1 &
        jobs+=("$inst:$!")
    else
        echo "$(date) $inst: started, see ${inst}_$TAG.log"
        # As a job, so set -e still stops the pipeline at a failed step (it
        # is ignored inside a function called before ||); a failure does not
        # stop the other instruments
        run_instrument "$inst" >> "${inst}_$TAG.log" 2>&1 &
        wait $! || { echo "$inst pipeline failed; see ${inst}_$TAG.log"; status=1; }
        echo "$(date) $inst: done"
    fi
done
for job in "${jobs[@]}"; do
    inst=${job%%:*}
    wait "${job#*:}" || { echo "$inst pipeline failed; see ${inst}_$TAG.log"; status=1; }
done
echo "Results: grep -A4 '^Eval against' *_$TAG.log"
exit $status
