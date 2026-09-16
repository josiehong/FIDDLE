# ============================================================================
# MSE ARM — poisson-vs-mse ablation on the v2.2 data (Q-TOF only).
#
# Trains the incumbent MSE loss from scratch on cl_pkl_082426 with everything
# else identical to the poisson run (same data, config, script, devices), then
# rescores and evaluates. Compare eval_qtof_$TAG.log against the poisson run's
# eval_qtof_082426_v2.log: the two tables differ by exactly one variable.
#
# Context (09/2026): poisson from-scratch lands at 53.7% top-1 / 91.2% top-20
# (spectra-level) on the rebuilt test set; v2.1 (mse, old data) reported 70.2%
# top-1 on the old test set. This arm decides whether the loss is the cause.
#
# Env overrides: TAG, DATA_TAG, DEVICE (space-separated GPU list, e.g. "0 1").
# ============================================================================

DATA_TAG=${DATA_TAG:-082426}
TAG=${TAG:-${DATA_TAG}_mse}
DEVICE=${DEVICE:-"0 1"}
PKL_DIR=./data/cl_pkl_$DATA_TAG

TCN_CKPT=./check_point/fiddle_tcn_qtof_$TAG.pt
RESCORE_CKPT=./check_point/fiddle_rescore_qtof_$TAG.pt
CONFIG=./msfiddle/config/fiddle_tcn_qtof.yml

# Workaround for the degraded GPU driver on carnot (since 08/14); safe to
# remove after the node reboot restores GPU6/NVML.
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

DEVICE_CSV=${DEVICE// /,}
DEVICE_LOCAL=$(seq -s' ' 0 $(($(echo $DEVICE | wc -w) - 1)))
DEV1=${DEVICE%% *}

for f in $PKL_DIR/qtof_maxmin_train.pkl $PKL_DIR/qtof_maxmin_test.pkl \
         $PKL_DIR/qtof_maxmin_test.mgf; do
    [ -s "$f" ] || { echo "MISSING input: $f"; exit 1; }
done

# ============================================================================
# PART 1 — TCN training from scratch, mse loss (skipped if checkpoint exists)
# ============================================================================

if [ -s "$TCN_CKPT" ]; then
    echo "TCN checkpoint already present: $TCN_CKPT — skipping PART 1"
else
    if pgrep -f "train_tcn_gpus_cl.py" > /dev/null; then
        echo "A train_tcn_gpus_cl.py process is already running — refusing to launch."
        exit 1
    fi
    # One CL training holds its noise-augmented Q-TOF dataset in RAM (~77GB).
    avail_ram_gb=$(free -g | awk '/^Mem:/ {print $7}')
    if [ "$avail_ram_gb" -lt 100 ]; then
        echo "Only ${avail_ram_gb}GB RAM available — need ~100GB for Q-TOF training."
        exit 1
    fi

    CUDA_VISIBLE_DEVICES=$DEVICE_CSV python -u scripts/train_tcn_gpus_cl.py \
    --train_data $PKL_DIR/qtof_maxmin_train.pkl \
    --test_data $PKL_DIR/qtof_maxmin_test.pkl \
    --config_path $CONFIG \
    --formula_loss mse \
    --checkpoint_path $TCN_CKPT \
    --result_path ./result/fiddle_tcn_qtof_$TAG.csv \
    --device $DEVICE_LOCAL > tcn_qtof_$TAG.log 2>&1

    echo "=== tail of tcn_qtof_$TAG.log ==="; tail -5 tcn_qtof_$TAG.log
    [ -s "$TCN_CKPT" ] || { echo "MISSING: $TCN_CKPT — check the training log"; exit 1; }

    python scripts/analyze_count_ambiguity.py \
    --result ./result/fiddle_tcn_qtof_$TAG.csv \
    --test_data $PKL_DIR/qtof_maxmin_test.pkl
fi

# ============================================================================
# PART 2 — Rescore retrain on this model's enumeration pairs
# ============================================================================

# Own subdirectory: the pair filenames are fixed by the train stem, and the
# poisson run's pairs already live in $PKL_DIR.
RESCORE_DIR=$PKL_DIR/rescore_$TAG
mkdir -p $RESCORE_DIR

(
python scripts/prepare_augment_rescore.py \
--train_data $PKL_DIR/qtof_maxmin_train.pkl \
--test_data $PKL_DIR/qtof_maxmin_test.pkl \
--config_path $CONFIG \
--resume_path $TCN_CKPT \
--rescore_dir $RESCORE_DIR \
--formula_loss mse \
--pos_cap 10 --neg_per_pos 8 --tolerance 50 --num_workers 8 --device $DEV1

python scripts/train_rescore.py \
--train_data $RESCORE_DIR/qtof_maxmin_rescore_train.pkl \
--test_data $RESCORE_DIR/qtof_maxmin_rescore_test.pkl \
--config_path $CONFIG \
--resume_path $TCN_CKPT \
--checkpoint_path $RESCORE_CKPT --device $DEV1
) > rescore_qtof_$TAG.log 2>&1

echo "=== tail of rescore_qtof_$TAG.log ==="; tail -5 rescore_qtof_$TAG.log
[ -s "$RESCORE_CKPT" ] || { echo "MISSING: $RESCORE_CKPT — check the log"; exit 1; }

# ============================================================================
# PART 3 — Held-out evaluation
# ============================================================================

(
python scripts/run_fiddle.py --test_data $PKL_DIR/qtof_maxmin_test.mgf \
--config_path $CONFIG \
--resume_path $TCN_CKPT \
--formula_loss mse \
--rescore_resume_path $RESCORE_CKPT \
--result_path ./result/fiddle_qtof_$TAG.csv --device $DEV1
python scripts/eval_topk.py --result ./result/fiddle_qtof_$TAG.csv \
--test_data $PKL_DIR/qtof_maxmin_test.pkl --topk 20
) > eval_qtof_$TAG.log 2>&1

echo "mse arm complete; compare eval_qtof_$TAG.log vs eval_qtof_082426_v2.log"
tail -5 eval_qtof_$TAG.log

# ============================================================================
# OPTIONAL CALIBRATION — v2.1 checkpoint on the new test set (~10 min).
# Run it from a SEPARATE clone at the v2.1 tag: v2.1 encodes collision energy
# on the old fraction scale, so replaying it with current code would feed the
# CE embedding out-of-distribution inputs and unfairly tank the baseline.
#
#   git clone <repo> FIDDLE_v21 && cd FIDDLE_v21 && git checkout v2.1.0
#   python scripts/run_fiddle.py \
#     --test_data ../FIDDLE_TCN/data/cl_pkl_082426/qtof_maxmin_test.mgf \
#     --config_path ./msfiddle/config/fiddle_tcn_qtof.yml \
#     --resume_path <v2.1 qtof checkpoint> \
#     --rescore_resume_path <v2.1 qtof rescore checkpoint> \
#     --result_path ./result/fiddle_qtof_v21_on_082426.csv --device 0
#   python scripts/eval_topk.py --result ./result/fiddle_qtof_v21_on_082426.csv \
#     --test_data ../FIDDLE_TCN/data/cl_pkl_082426/qtof_maxmin_test.pkl --topk 10
# ============================================================================
