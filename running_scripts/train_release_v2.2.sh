#!/usr/bin/env bash
# Train the v2.2 release models from scratch: build the data (PART 1), train
# the TCNs (PART 2) and rescore models (PART 3), and evaluate on the held-out
# test sets (PART 4). Q-TOF trains in fp32, Orbitrap with --amp. Each part
# skips work whose outputs already exist, so a rerun resumes where it stopped.
#
# Env overrides: DATA_TAG (pkl generation), TAG (model tag), NUM_WORKERS,
#   DEVICE_QTOF / DEVICE_ORBITRAP (space-separated GPU lists, e.g. "6 7").
#
# The release checkpoints are fiddle_{tcn,rescore}_{qtof,orbitrap}_082426_muon.pt,
# shipped as fiddle_{tcn,rescore}_{qtof,orbitrap}.pt
# (scripts/package_release_checkpoints.py).

DATA_TAG=${DATA_TAG:-082426}
TAG=${TAG:-${DATA_TAG}_muon}
DEVICE_QTOF=${DEVICE_QTOF:-"0 1"}
DEVICE_ORBITRAP=${DEVICE_ORBITRAP:-"3 4"}
PKL_DIR=./data/cl_pkl_$DATA_TAG

# PART 2 skip, per instrument: an instrument whose TCN checkpoint already
# exists is not retrained (delete the checkpoint to force retraining), so a
# rerun after a finished training goes straight to the rescore/eval parts.
qtof_ckpt_done=false; [ -s ./check_point/fiddle_tcn_qtof_$TAG.pt ] && qtof_ckpt_done=true
orbitrap_ckpt_done=false; [ -s ./check_point/fiddle_tcn_orbitrap_$TAG.pt ] && orbitrap_ckpt_done=true

if [ "$qtof_ckpt_done" = false ] || [ "$orbitrap_ckpt_done" = false ]; then
    # Refuse to stack on an existing run: repeated launches (e.g. retried
    # commands from a frozen terminal) would train duplicate processes into the
    # same checkpoint paths and exhaust RAM.
    if pgrep -f "train_tcn_gpus_cl.py" > /dev/null; then
        echo "A train_tcn_gpus_cl.py process is already running — refusing to launch."
        echo "Inspect with: ps aux | grep train_tcn   (kill it first if intended)"
        exit 1
    fi

    # Each CL training holds its dataset in RAM (Orbitrap: ~54GB train + ~7GB
    # test as float32). Refuse to start without headroom rather than push the
    # node into swap.
    need_ram_gb=0
    [ "$qtof_ckpt_done" = false ] && need_ram_gb=$((need_ram_gb + 100))
    [ "$orbitrap_ckpt_done" = false ] && need_ram_gb=$((need_ram_gb + 80))
    avail_ram_gb=$(free -g | awk '/^Mem:/ {print $7}')
    if [ "$avail_ram_gb" -lt "$need_ram_gb" ]; then
        echo "Only ${avail_ram_gb}GB RAM available — need ~${need_ram_gb}GB for the pending trainings."
        echo "Free memory (or run one instrument at a time) before launching."
        exit 1
    fi
fi

# Each training job is fenced to its own GPUs via CUDA_VISIBLE_DEVICES, so the
# in-process device indices are always 0..n-1 regardless of the physical
# indices in DEVICE_*.
DEVICE_QTOF_CSV=${DEVICE_QTOF// /,}
DEVICE_ORBITRAP_CSV=${DEVICE_ORBITRAP// /,}
DEVICE_QTOF_LOCAL=$(seq -s' ' 0 $(($(echo $DEVICE_QTOF | wc -w) - 1)))
DEVICE_ORBITRAP_LOCAL=$(seq -s' ' 0 $(($(echo $DEVICE_ORBITRAP | wc -w) - 1)))

# ============================================================================
# PART 1 — Rebuild the pkl data (CPU-only; skipped if $PKL_DIR is complete)
# ============================================================================

NUM_WORKERS=${NUM_WORKERS:-8}

QTOF_OUTPUTS="$PKL_DIR/qtof_maxmin_train.pkl $PKL_DIR/qtof_maxmin_train_pairs.pkl \
$PKL_DIR/qtof_maxmin_test.pkl"
ORBITRAP_OUTPUTS="$PKL_DIR/orbitrap_maxmin_train.pkl \
$PKL_DIR/orbitrap_maxmin_train_pairs.pkl $PKL_DIR/orbitrap_maxmin_test.pkl"

# Per-instrument skip: an instrument with all its outputs present is not
# rebuilt (delete an instrument's files to force its rebuild — partial files
# count as missing only if you remove them, so clean up interrupted runs).
qtof_done=true
for f in $QTOF_OUTPUTS; do [ -s "$f" ] || qtof_done=false; done
orbitrap_done=true
for f in $ORBITRAP_OUTPUTS; do [ -s "$f" ] || orbitrap_done=false; done

if [ "$qtof_done" = true ] && [ "$orbitrap_done" = true ]; then
    echo "Rebuilt data already present at $PKL_DIR — skipping PART 1"
else
    avail_gb=$(df -Pk ./data | awk 'NR==2 {print int($4/1048576)}')
    if [ "$avail_gb" -lt 80 ]; then
        echo "Only ${avail_gb}GB free under ./data — need ~80GB for a full rebuild."
        exit 1
    fi

    mkdir -p $PKL_DIR

    if [ "$qtof_done" = true ]; then
        echo "Q-TOF pkls already present — skipping"
    else
        python -u scripts/prepare_msms.py \
        --dataset agilent nist20 nist23 mona waters gnps \
        --instrument_type qtof \
        --config_path ./msfiddle/config/fiddle_tcn_qtof.yml \
        --num_workers $NUM_WORKERS \
        --pkl_dir $PKL_DIR/ --maxmin_pick > prep_qtof_$DATA_TAG.log 2>&1 &
    fi

    if [ "$orbitrap_done" = true ]; then
        echo "Orbitrap pkls already present — skipping"
    else
        python -u scripts/prepare_msms.py \
        --dataset nist20 nist23 mona gnps \
        --instrument_type orbitrap \
        --config_path ./msfiddle/config/fiddle_tcn_orbitrap.yml \
        --num_workers $NUM_WORKERS \
        --pkl_dir $PKL_DIR/ --maxmin_pick > prep_orbitrap_$DATA_TAG.log 2>&1 &
    fi

    wait  # remaining preprocessing runs done

    for log in prep_qtof_$DATA_TAG.log prep_orbitrap_$DATA_TAG.log; do
        [ -f "$log" ] && { echo "=== tail of $log ==="; tail -8 "$log"; }
    done
    for f in $QTOF_OUTPUTS $ORBITRAP_OUTPUTS; do
        [ -s "$f" ] || { echo "MISSING: $f — check the logs above"; exit 1; }
    done
    echo "rebuild complete: $PKL_DIR"
fi

# ============================================================================
# PART 2 — TCN training from scratch (both instruments in parallel)
# ============================================================================

if [ "$qtof_ckpt_done" = true ]; then
    echo "Q-TOF TCN checkpoint already present for tag $TAG — skipping"
else
    CUDA_VISIBLE_DEVICES=$DEVICE_QTOF_CSV python -u scripts/train_tcn_gpus_cl.py \
    --train_data $PKL_DIR/qtof_maxmin_train.pkl \
    --test_data $PKL_DIR/qtof_maxmin_test.pkl \
    --config_path ./msfiddle/config/fiddle_tcn_qtof.yml \
    --checkpoint_path ./check_point/fiddle_tcn_qtof_$TAG.pt \
    --result_path ./result/fiddle_tcn_qtof_$TAG.csv \
    --device $DEVICE_QTOF_LOCAL > tcn_qtof_$TAG.log 2>&1 &
fi

if [ "$orbitrap_ckpt_done" = true ]; then
    echo "Orbitrap TCN checkpoint already present for tag $TAG — skipping"
else
    CUDA_VISIBLE_DEVICES=$DEVICE_ORBITRAP_CSV python -u scripts/train_tcn_gpus_cl.py \
    --train_data $PKL_DIR/orbitrap_maxmin_train.pkl \
    --test_data $PKL_DIR/orbitrap_maxmin_test.pkl \
    --config_path ./msfiddle/config/fiddle_tcn_orbitrap.yml \
    --checkpoint_path ./check_point/fiddle_tcn_orbitrap_$TAG.pt \
    --result_path ./result/fiddle_tcn_orbitrap_$TAG.csv \
    --device $DEVICE_ORBITRAP_LOCAL --amp > tcn_orbitrap_$TAG.log 2>&1 &
fi

wait

if [ "$qtof_ckpt_done" = false ] || [ "$orbitrap_ckpt_done" = false ]; then
    for log in tcn_qtof_$TAG.log tcn_orbitrap_$TAG.log; do
        [ -f "$log" ] && { echo "=== tail of $log ==="; tail -5 "$log"; }
    done
fi
for f in ./check_point/fiddle_tcn_qtof_$TAG.pt ./check_point/fiddle_tcn_orbitrap_$TAG.pt; do
    [ -s "$f" ] || { echo "MISSING: $f — check the training log above"; exit 1; }
done

# ============================================================================
# PART 3 — Rescore retrain on enumeration-generated pairs, against the NEW
# TCNs and data (each instrument's prepare precedes its own training, so the
# pipeline is sequential inside each subshell and parallel across GPUs; the
# rescore/eval steps are single-GPU, so use the first index of each list)
# ============================================================================

TCN_QTOF=./check_point/fiddle_tcn_qtof_$TAG.pt
TCN_ORBITRAP=./check_point/fiddle_tcn_orbitrap_$TAG.pt
RESCORE_QTOF=./check_point/fiddle_rescore_qtof_$TAG.pt
RESCORE_ORBITRAP=./check_point/fiddle_rescore_orbitrap_$TAG.pt
# Rescore pairs depend on the TCN, so they live per model tag (the pkl dir is
# shared by every model trained on this data generation).
RESCORE_DIR=./data/rescore_$TAG
mkdir -p $RESCORE_DIR
DEV1_QTOF=${DEVICE_QTOF%% *}
DEV1_ORBITRAP=${DEVICE_ORBITRAP%% *}

# Per-instrument skip: an existing rescore checkpoint is kept (delete it to
# force pair regeneration + retraining). Only valid while the TCN it was
# trained against is unchanged — retraining a TCN means deleting its
# rescore checkpoint too.
if [ -s "$RESCORE_QTOF" ]; then
    echo "Q-TOF rescore checkpoint already present for tag $TAG — skipping"
else
(
python scripts/prepare_augment_rescore.py \
--train_data $PKL_DIR/qtof_maxmin_train.pkl \
--test_data $PKL_DIR/qtof_maxmin_test.pkl \
--config_path ./msfiddle/config/fiddle_tcn_qtof.yml \
--resume_path $TCN_QTOF \
--rescore_dir $RESCORE_DIR \
--pos_cap 10 --neg_per_pos 8 --tolerance 50 --num_workers 8 --device $DEV1_QTOF

python scripts/train_rescore.py \
--train_data $RESCORE_DIR/qtof_maxmin_rescore_train.pkl \
--test_data $RESCORE_DIR/qtof_maxmin_rescore_test.pkl \
--config_path ./msfiddle/config/fiddle_tcn_qtof.yml \
--resume_path $TCN_QTOF \
--checkpoint_path $RESCORE_QTOF --device $DEV1_QTOF
) > rescore_qtof_$TAG.log 2>&1 &
fi

if [ -s "$RESCORE_ORBITRAP" ]; then
    echo "Orbitrap rescore checkpoint already present for tag $TAG — skipping"
else
(
python scripts/prepare_augment_rescore.py \
--train_data $PKL_DIR/orbitrap_maxmin_train.pkl \
--test_data $PKL_DIR/orbitrap_maxmin_test.pkl \
--config_path ./msfiddle/config/fiddle_tcn_orbitrap.yml \
--resume_path $TCN_ORBITRAP \
--rescore_dir $RESCORE_DIR \
--pos_cap 10 --neg_per_pos 8 --tolerance 50 --num_workers 8 --device $DEV1_ORBITRAP

python scripts/train_rescore.py \
--train_data $RESCORE_DIR/orbitrap_maxmin_rescore_train.pkl \
--test_data $RESCORE_DIR/orbitrap_maxmin_rescore_test.pkl \
--config_path ./msfiddle/config/fiddle_tcn_orbitrap.yml \
--resume_path $TCN_ORBITRAP \
--checkpoint_path $RESCORE_ORBITRAP --device $DEV1_ORBITRAP
) > rescore_orbitrap_$TAG.log 2>&1 &
fi

wait  # rescore models trained

for log in rescore_qtof_$TAG.log rescore_orbitrap_$TAG.log; do
    [ -f "$log" ] && { echo "=== tail of $log ==="; tail -5 "$log"; }
done
for f in $RESCORE_QTOF $RESCORE_ORBITRAP; do
    [ -s "$f" ] || { echo "MISSING: $f — check the log above"; exit 1; }
done

# ============================================================================
# PART 4 — Held-out evaluation: new TCN + new rescorer (parallel)
# ============================================================================

(
python -m msfiddle predict --num_workers 8 --test_data $PKL_DIR/qtof_maxmin_test.mgf \
--config_path ./msfiddle/config/fiddle_tcn_qtof.yml \
--resume_path $TCN_QTOF \
--rescore_resume_path ./check_point/fiddle_rescore_qtof_$TAG.pt \
--result_path ./result/fiddle_qtof_$TAG.csv --device $DEV1_QTOF \
--eval
) > eval_qtof_$TAG.log 2>&1 &

(
python -m msfiddle predict --num_workers 8 --test_data $PKL_DIR/orbitrap_maxmin_test.mgf \
--config_path ./msfiddle/config/fiddle_tcn_orbitrap.yml \
--resume_path $TCN_ORBITRAP \
--rescore_resume_path ./check_point/fiddle_rescore_orbitrap_$TAG.pt \
--result_path ./result/fiddle_orbitrap_$TAG.csv --device $DEV1_ORBITRAP \
--eval
) > eval_orbitrap_$TAG.log 2>&1 &

wait
echo "v2.2 release cycle complete; see eval_qtof_$TAG.log / eval_orbitrap_$TAG.log"

