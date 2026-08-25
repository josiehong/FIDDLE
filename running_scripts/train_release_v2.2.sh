# ============================================================================
# TRACK 2 — v2.2 release cycle: rebuild the data, then train FROM SCRATCH.
#
# Evidence and design in logs/experiment_v2.2.md. Summary vs v2.1:
#   - formula_refinement: complete bounded enumeration (13x faster, provably
#     complete within the caps) ranked by (edit distance, |dmass|).
#   - Rebuilt pkls (PART 1) with the 08/2026 preprocessing fixes: contrastive
#     pairing, isotope-removal direction, NCE scale, -2H2O labels, GNPS
#     adducts, crash guards. Never mix generations: old checkpoints pair with
#     cl_pkl_060526, everything trained here pairs with cl_pkl_$DATA_TAG.
#   - Rescore models retrained on enumeration-generated pairs.
#   - Optionally the Poisson count-likelihood formula loss, if the Track 1
#     fine-tune experiments (experiments_finetune_loss.sh, old data) win.
#
# PREREQUISITES
#   (a) Config update in msfiddle/config/fiddle_tcn_{qtof,orbitrap}.yml:
#         refine_atom_type: ['C','O','N','H','Cl','Br','F','I','S','P']
#         refine_atom_num:  [5, 5, 5, -1, 2, 1, 2, 1, 2, 1]
#         top_k: 20
#       Orbitrap mass_tolerance: 5.0 only if the measured-mass check passes
#       (PPM-field distribution of the orbitrap test MGF; keep 10.0 if >~1%
#       of spectra exceed 5 ppm).
#   (b) FORMULA_LOSS set from the Track 1 verdict (default mse).
#
# Env overrides: DATA_TAG (pkl generation), TAG (model tag), FORMULA_LOSS,
#   DEVICE_QTOF / DEVICE_ORBITRAP (space-separated lists allowed, e.g. "6 7").
# ============================================================================

DATA_TAG=${DATA_TAG:-082426}
TAG=${TAG:-$DATA_TAG}
FORMULA_LOSS=${FORMULA_LOSS:-mse}
DEVICE_QTOF=${DEVICE_QTOF:-"0 1"}
DEVICE_ORBITRAP=${DEVICE_ORBITRAP:-"3 4"}
PKL_DIR=./data/cl_pkl_$DATA_TAG

# ============================================================================
# PART 1 — Rebuild the pkl data (CPU-only; skipped if $PKL_DIR is complete)
# ============================================================================

REBUILD_OUTPUTS="$PKL_DIR/qtof_maxmin_train.pkl $PKL_DIR/qtof_maxmin_train_pairs.pkl \
$PKL_DIR/qtof_maxmin_test.pkl $PKL_DIR/orbitrap_maxmin_train.pkl \
$PKL_DIR/orbitrap_maxmin_train_pairs.pkl $PKL_DIR/orbitrap_maxmin_test.pkl"

rebuild_done=true
for f in $REBUILD_OUTPUTS; do [ -s "$f" ] || rebuild_done=false; done

if [ "$rebuild_done" = true ]; then
    echo "Rebuilt data already present at $PKL_DIR — skipping PART 1"
else
    # Refuse to run on a stale checkout: the whole point is the fixed encoders.
    python - <<'PY' || exit 1
import sys
from msfiddle.utils.pkl_utils import parse_collision_energy
ce, nce = parse_collision_energy("35 nce", precursor_mz=400.0)
if abs(nce - 35.0) > 1e-6:
    sys.exit("STALE CODE: parse_collision_energy still fraction-scaled — "
             "upload the fixed msfiddle/ and scripts/ files first")
print("preprocessing fixes present — ok")
PY

    avail_gb=$(df -Pk ./data | awk 'NR==2 {print int($4/1048576)}')
    if [ "$avail_gb" -lt 80 ]; then
        echo "Only ${avail_gb}GB free under ./data — need ~80GB for a full rebuild."
        exit 1
    fi

    mkdir -p $PKL_DIR

    python -u scripts/prepare_msms.py \
    --dataset agilent nist20 nist23 mona waters gnps \
    --instrument_type qtof \
    --config_path ./msfiddle/config/fiddle_tcn_qtof.yml \
    --pkl_dir $PKL_DIR/ --maxmin_pick > prep_qtof_$DATA_TAG.log 2>&1 &

    python -u scripts/prepare_msms.py \
    --dataset nist20 nist23 mona gnps \
    --instrument_type orbitrap \
    --config_path ./msfiddle/config/fiddle_tcn_orbitrap.yml \
    --pkl_dir $PKL_DIR/ --maxmin_pick > prep_orbitrap_$DATA_TAG.log 2>&1 &

    wait  # both preprocessing runs done (hours; NIST23 dominates)

    for log in prep_qtof_$DATA_TAG.log prep_orbitrap_$DATA_TAG.log; do
        echo "=== tail of $log ==="; tail -8 "$log"
    done
    for f in $REBUILD_OUTPUTS; do
        [ -s "$f" ] || { echo "MISSING: $f — check the logs above"; exit 1; }
    done
    echo "rebuild complete: $PKL_DIR"
fi

# ============================================================================
# PART 2 — TCN training from scratch (both instruments in parallel)
# ============================================================================

python -u scripts/train_tcn_gpus_cl.py \
--train_data $PKL_DIR/qtof_maxmin_train.pkl \
--test_data $PKL_DIR/qtof_maxmin_test.pkl \
--config_path ./msfiddle/config/fiddle_tcn_qtof.yml \
--formula_loss $FORMULA_LOSS \
--checkpoint_path ./check_point/fiddle_tcn_qtof_$TAG.pt \
--result_path ./result/fiddle_tcn_qtof_$TAG.csv \
--device $DEVICE_QTOF > tcn_qtof_$TAG.log 2>&1 &

python -u scripts/train_tcn_gpus_cl.py \
--train_data $PKL_DIR/orbitrap_maxmin_train.pkl \
--test_data $PKL_DIR/orbitrap_maxmin_test.pkl \
--config_path ./msfiddle/config/fiddle_tcn_orbitrap.yml \
--formula_loss $FORMULA_LOSS \
--checkpoint_path ./check_point/fiddle_tcn_orbitrap_$TAG.pt \
--result_path ./result/fiddle_tcn_orbitrap_$TAG.csv \
--device $DEVICE_ORBITRAP > tcn_orbitrap_$TAG.log 2>&1 &

wait

for log in tcn_qtof_$TAG.log tcn_orbitrap_$TAG.log; do
    echo "=== tail of $log ==="; tail -5 "$log"
done
for f in ./check_point/fiddle_tcn_qtof_$TAG.pt ./check_point/fiddle_tcn_orbitrap_$TAG.pt; do
    [ -s "$f" ] || { echo "MISSING: $f — check the training log above"; exit 1; }
done

python scripts/analyze_count_ambiguity.py \
--result ./result/fiddle_tcn_qtof_$TAG.csv \
--test_data $PKL_DIR/qtof_maxmin_test.pkl
python scripts/analyze_count_ambiguity.py \
--result ./result/fiddle_tcn_orbitrap_$TAG.csv \
--test_data $PKL_DIR/orbitrap_maxmin_test.pkl

# ============================================================================
# PART 3 — Rescore retrain on enumeration-generated pairs, against the NEW
# TCNs and data (each instrument's prepare precedes its own training, so the
# pipeline is sequential inside each subshell and parallel across GPUs; the
# rescore/eval steps are single-GPU, so use the first index of each list)
# ============================================================================

TCN_QTOF=./check_point/fiddle_tcn_qtof_$TAG.pt
TCN_ORBITRAP=./check_point/fiddle_tcn_orbitrap_$TAG.pt
DEV1_QTOF=${DEVICE_QTOF%% *}
DEV1_ORBITRAP=${DEVICE_ORBITRAP%% *}

(
python scripts/prepare_augment_rescore.py \
--train_data $PKL_DIR/qtof_maxmin_train.pkl \
--test_data $PKL_DIR/qtof_maxmin_test.pkl \
--config_path ./msfiddle/config/fiddle_tcn_qtof.yml \
--resume_path $TCN_QTOF \
--rescore_dir $PKL_DIR \
--pos_cap 10 --neg_per_pos 8 --tolerance 50 --num_workers 8 --device $DEV1_QTOF

python scripts/train_rescore.py \
--train_data $PKL_DIR/qtof_maxmin_rescore_train.pkl \
--test_data $PKL_DIR/qtof_maxmin_rescore_test.pkl \
--config_path ./msfiddle/config/fiddle_tcn_qtof.yml \
--resume_path $TCN_QTOF \
--checkpoint_path ./check_point/fiddle_rescore_qtof_$TAG.pt --device $DEV1_QTOF
) > rescore_qtof_$TAG.log 2>&1 &

(
python scripts/prepare_augment_rescore.py \
--train_data $PKL_DIR/orbitrap_maxmin_train.pkl \
--test_data $PKL_DIR/orbitrap_maxmin_test.pkl \
--config_path ./msfiddle/config/fiddle_tcn_orbitrap.yml \
--resume_path $TCN_ORBITRAP \
--rescore_dir $PKL_DIR \
--pos_cap 10 --neg_per_pos 8 --tolerance 50 --num_workers 8 --device $DEV1_ORBITRAP

python scripts/train_rescore.py \
--train_data $PKL_DIR/orbitrap_maxmin_rescore_train.pkl \
--test_data $PKL_DIR/orbitrap_maxmin_rescore_test.pkl \
--config_path ./msfiddle/config/fiddle_tcn_orbitrap.yml \
--resume_path $TCN_ORBITRAP \
--checkpoint_path ./check_point/fiddle_rescore_orbitrap_$TAG.pt --device $DEV1_ORBITRAP
) > rescore_orbitrap_$TAG.log 2>&1 &

wait  # both rescore models trained

for log in rescore_qtof_$TAG.log rescore_orbitrap_$TAG.log; do
    echo "=== tail of $log ==="; tail -5 "$log"
done
for f in ./check_point/fiddle_rescore_qtof_$TAG.pt ./check_point/fiddle_rescore_orbitrap_$TAG.pt; do
    [ -s "$f" ] || { echo "MISSING: $f — check the log above"; exit 1; }
done

# ============================================================================
# PART 4 — Held-out evaluation: new TCN + new rescorer (parallel)
# ============================================================================

(
python scripts/run_fiddle.py --test_data $PKL_DIR/qtof_maxmin_test.mgf \
--config_path ./msfiddle/config/fiddle_tcn_qtof.yml \
--resume_path $TCN_QTOF \
--rescore_resume_path ./check_point/fiddle_rescore_qtof_$TAG.pt \
--result_path ./result/fiddle_qtof_$TAG.csv --device $DEV1_QTOF
python scripts/eval_topk.py --result ./result/fiddle_qtof_$TAG.csv \
--test_data $PKL_DIR/qtof_maxmin_test.pkl --topk 10
) > eval_qtof_$TAG.log 2>&1 &

(
python scripts/run_fiddle.py --test_data $PKL_DIR/orbitrap_maxmin_test.mgf \
--config_path ./msfiddle/config/fiddle_tcn_orbitrap.yml \
--resume_path $TCN_ORBITRAP \
--rescore_resume_path ./check_point/fiddle_rescore_orbitrap_$TAG.pt \
--result_path ./result/fiddle_orbitrap_$TAG.csv --device $DEV1_ORBITRAP
python scripts/eval_topk.py --result ./result/fiddle_orbitrap_$TAG.csv \
--test_data $PKL_DIR/orbitrap_maxmin_test.pkl --topk 10
) > eval_orbitrap_$TAG.log 2>&1 &

wait
echo "v2.2 release cycle complete; see eval_qtof_$TAG.log / eval_orbitrap_$TAG.log"

# Post-eval checks:
# - top_k sensitivity: rerun eval_topk with --topk 10 and --topk 20 on the
#   result CSVs; confirm accuracy peaks at (or before) K=20.
# - Compare eval_topk numbers against the v2.1 release (Q-TOF top-1 70.2%).
