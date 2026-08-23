# ============================================================================
# Released models — maxmin rebuild (QTOF + Orbitrap), date 060526
#
# Reference catalog (not run top-to-bottom). Two parts:
#   PART 1 — TRAINING:   clean rebuild of both release models.
#   PART 2 — EVALUATION: held-out top-k accuracy (per spectrum & per compound,
#                        via eval_topk.py) + external CASMI / EMBL tests.
#
# The diagnostics that motivated this build (training-set composition, precursor
# isotope feasibility, baseline per-element accuracy) are concluded and written up
# in logs/experiment_v3.md; their one-off scripts have been removed.
#
# What's baked in (see logs/experiment_v3.md):
#   - preprocessing fixes: reproducible split; [M-H2O-H]- adduct (these spectra
#     used to crash preprocessing -> re-preprocessing now includes them).
#   - halogen reachability: both configs (qtof + orbitrap) have expanded
#     refine_atom_type + top_k=10; prepare_augment_rescore trains the rescorer on
#     those candidates. (Orbitrap is halogen-rich, so expect a smaller effect there.)
#   - fast_mass refinement speedup (msfiddle/utils/refine_utils.py).
# Note: the mz_val (column 1) preprocessing change is NOT model-visible (the model
#   reads spec[:,0]); the rebuild is driven by the data-level fixes above.
# Adjust --device to your GPUs.
# ============================================================================


# ============================================================================
# PART 1 — RELEASED MODEL: TRAINING  (clean rebuild, date 060526)
# ============================================================================

# --- 1a. Preprocess (maxmin split, fixed prepare_msms.py) --------------------
python scripts/prepare_msms.py \
--dataset agilent nist20 nist23 mona waters gnps \
--instrument_type qtof \
--config_path ./msfiddle/config/fiddle_tcn_qtof.yml \
--pkl_dir ./data/cl_pkl_060526/ --maxmin_pick

python scripts/prepare_msms.py \
--dataset nist20 nist23 mona gnps \
--instrument_type orbitrap \
--config_path ./msfiddle/config/fiddle_tcn_orbitrap.yml \
--pkl_dir ./data/cl_pkl_060526/ --maxmin_pick

# --- 1b. Train TCN -----------------------------------------------------------
nohup python -u train_tcn_gpus_cl.py \
--train_data ./data/cl_pkl_060526/qtof_maxmin_train.pkl \
--test_data ./data/cl_pkl_060526/qtof_maxmin_test.pkl \
--config_path ./msfiddle/config/fiddle_tcn_qtof.yml \
--checkpoint_path ./check_point/fiddle_tcn_qtof_060526.pt \
--result_path ./result/fiddle_tcn_qtof_060526.csv --device 6 7 > fiddle_tcn_qtof_060526.log 2>&1 &

python -u train_tcn_gpus_cl.py \
--train_data ./data/cl_pkl_060526/orbitrap_maxmin_train.pkl \
--test_data ./data/cl_pkl_060526/orbitrap_maxmin_test.pkl \
--config_path ./msfiddle/config/fiddle_tcn_orbitrap.yml \
--checkpoint_path ./check_point/fiddle_tcn_orbitrap_060526.pt \
--result_path ./result/fiddle_tcn_orbitrap_060526.csv --device 6 7

# --- 1c. Rescorer (FIDDLES) — trains on the config's refine_atom_type + top_k --
python scripts/prepare_augment_rescore.py \
--train_data ./data/cl_pkl_060526/qtof_maxmin_train.pkl \
--test_data ./data/cl_pkl_060526/qtof_maxmin_test.pkl \
--config_path ./msfiddle/config/fiddle_tcn_qtof.yml \
--resume_path ./check_point/fiddle_tcn_qtof_060526.pt \
--rescore_dir ./data/cl_pkl_060526 \
--pos_cap 10 --neg_per_pos 8 --tolerance 50 --num_workers 8 --device 5
python scripts/train_rescore.py \
--train_data ./data/cl_pkl_060526/qtof_maxmin_rescore_train.pkl \
--test_data ./data/cl_pkl_060526/qtof_maxmin_rescore_test.pkl \
--config_path ./msfiddle/config/fiddle_tcn_qtof.yml \
--resume_path ./check_point/fiddle_tcn_qtof_060526.pt \
--checkpoint_path ./check_point/fiddle_rescore_qtof_060526.pt --device 5

python scripts/prepare_augment_rescore.py \
--train_data ./data/cl_pkl_060526/orbitrap_maxmin_train.pkl \
--test_data ./data/cl_pkl_060526/orbitrap_maxmin_test.pkl \
--config_path ./msfiddle/config/fiddle_tcn_orbitrap.yml \
--resume_path ./check_point/fiddle_tcn_orbitrap_060526.pt \
--rescore_dir ./data/cl_pkl_060526 \
--pos_cap 10 --neg_per_pos 8 --tolerance 50 --num_workers 8 --device 4
python scripts/train_rescore.py \
--train_data ./data/cl_pkl_060526/orbitrap_maxmin_rescore_train.pkl \
--test_data ./data/cl_pkl_060526/orbitrap_maxmin_rescore_test.pkl \
--config_path ./msfiddle/config/fiddle_tcn_orbitrap.yml \
--resume_path ./check_point/fiddle_tcn_orbitrap_060526.pt \
--checkpoint_path ./check_point/fiddle_rescore_orbitrap_060526.pt --device 4


# ============================================================================
# PART 2 — EVALUATION  (new release, 060526)
# ============================================================================

# --- 2a. Held-out test + top-k accuracy (per spectrum & per compound): QTOF --
python scripts/run_fiddle.py --test_data ./data/cl_pkl_060526/qtof_maxmin_test.mgf \
--config_path ./msfiddle/config/fiddle_tcn_qtof.yml \
--resume_path ./check_point/fiddle_tcn_qtof_060526.pt \
--rescore_resume_path ./check_point/fiddle_rescore_qtof_060526.pt \
--result_path ./result/fiddle_qtof_060526.csv --device 5
python scripts/eval_topk.py --result ./result/fiddle_qtof_060526.csv \
--test_data ./data/cl_pkl_060526/qtof_maxmin_test.pkl --topk 5
# Joined: 12627 spectra / 1540 compounds  (matched 12627/12627 result rows)
#             top-1   top-2   top-3   top-4   top-5 
# spectra    70.2%   74.9%   75.8%   76.0%   76.2%
# compound   72.8%   77.3%   78.2%   78.5%   78.7%

# --- 2b. Held-out test + top-k accuracy (per spectrum & per compound): Orbitrap
python scripts/run_fiddle.py --test_data ./data/cl_pkl_060526/orbitrap_maxmin_test.mgf \
--config_path ./msfiddle/config/fiddle_tcn_orbitrap.yml \
--resume_path ./check_point/fiddle_tcn_orbitrap_060526.pt \
--rescore_resume_path ./check_point/fiddle_rescore_orbitrap_060526.pt \
--result_path ./result/fiddle_orbitrap_060526.csv --device 4
python scripts/eval_topk.py --result ./result/fiddle_orbitrap_060526.csv \
--test_data ./data/cl_pkl_060526/orbitrap_maxmin_test.pkl --topk 5

# --- 2c. External test: CASMI and EMBL (QTOF model) --------------------------
python scripts/run_fiddle.py --test_data ./data/casmi2016.mgf \
--config_path ./msfiddle/config/fiddle_tcn_casmi.yml \
--resume_path ./check_point/fiddle_tcn_qtof_060526.pt \
--rescore_resume_path ./check_point/fiddle_rescore_qtof_060526.pt \
--result_path ./result/fiddle_casmi16_060526.csv
python scripts/run_fiddle.py --test_data ./data/casmi2017.mgf \
--config_path ./msfiddle/config/fiddle_tcn_casmi.yml \
--resume_path ./check_point/fiddle_tcn_qtof_060526.pt \
--rescore_resume_path ./check_point/fiddle_rescore_qtof_060526.pt \
--result_path ./result/fiddle_casmi17_060526.csv
python scripts/run_fiddle.py --test_data ./data/embl_mcf_2.0.mgf \
--config_path ./msfiddle/config/fiddle_tcn_embl.yml \
--resume_path ./check_point/fiddle_tcn_qtof_060526.pt \
--rescore_resume_path ./check_point/fiddle_rescore_qtof_060526.pt \
--result_path ./result/fiddle_embl_060526.csv


# ============================================================================
# PART 3 — EXPERIMENT: CE-stratified per-compound cap (vs PART 1/2 baseline)
#   Trims each compound to <=K spectra spanning its collision-energy range, so
#   heavily-measured compounds stop dominating the per-spectrum training signal.
#   Only the TRAIN split is capped; the test split is identical to PART 2, so the
#   capped model is directly comparable to the PART 1/2 baseline on the same test.
#   Pick K from the spectra/compound distribution that balance_ce_cap.py prints
#   (median is a sensible start: ~8 Q-TOF, higher for Orbitrap).
#   Note: the *_rescore_test.pkl name is derived from the (unchanged) test file,
#   so it overwrites PART 1c's copy. Harmless — it is rescore-training validation
#   only; final accuracy comes from run_fiddle + eval_topk below.
# ============================================================================

# --- 3a. Q-TOF: cap -> TCN -> rescorer -> eval (K=8) ------------------------
python scripts/balance_ce_cap.py \
--train_data ./data/cl_pkl_060526/qtof_maxmin_train.pkl \
--out ./data/cl_pkl_060526/qtof_maxmin_cek8_train.pkl \
--cap 8 --ce_bins 3 --seed 42

nohup python -u train_tcn_gpus_cl.py \
--train_data ./data/cl_pkl_060526/qtof_maxmin_cek8_train.pkl \
--test_data ./data/cl_pkl_060526/qtof_maxmin_test.pkl \
--config_path ./msfiddle/config/fiddle_tcn_qtof.yml \
--checkpoint_path ./check_point/fiddle_tcn_qtof_060526_cek8.pt \
--result_path ./result/fiddle_tcn_qtof_060526_cek8.csv --device 4 5 > fiddle_tcn_qtof_060526_cek8.log 2>&1 &

python scripts/prepare_augment_rescore.py \
--train_data ./data/cl_pkl_060526/qtof_maxmin_cek8_train.pkl \
--test_data ./data/cl_pkl_060526/qtof_maxmin_test.pkl \
--config_path ./msfiddle/config/fiddle_tcn_qtof.yml \
--resume_path ./check_point/fiddle_tcn_qtof_060526_cek8.pt \
--rescore_dir ./data/cl_pkl_060526 \
--pos_cap 10 --neg_per_pos 8 --tolerance 50 --num_workers 8 --device 5
python scripts/train_rescore.py \
--train_data ./data/cl_pkl_060526/qtof_maxmin_cek8_rescore_train.pkl \
--test_data ./data/cl_pkl_060526/qtof_maxmin_rescore_test.pkl \
--config_path ./msfiddle/config/fiddle_tcn_qtof.yml \
--resume_path ./check_point/fiddle_tcn_qtof_060526_cek8.pt \
--checkpoint_path ./check_point/fiddle_rescore_qtof_060526_cek8.pt --device 5

python scripts/run_fiddle.py --test_data ./data/cl_pkl_060526/qtof_maxmin_test.mgf \
--config_path ./msfiddle/config/fiddle_tcn_qtof.yml \
--resume_path ./check_point/fiddle_tcn_qtof_060526_cek8.pt \
--rescore_resume_path ./check_point/fiddle_rescore_qtof_060526_cek8.pt \
--result_path ./result/fiddle_qtof_060526_cek8.csv --device 5
python scripts/eval_topk.py --result ./result/fiddle_qtof_060526_cek8.csv \
--test_data ./data/cl_pkl_060526/qtof_maxmin_test.pkl --topk 5
# Compare against PART 2a (baseline). Per-compound row is the headline.

# --- 3b. Orbitrap: cap -> TCN -> rescorer -> eval (set K from the printout) --
python scripts/balance_ce_cap.py \
--train_data ./data/cl_pkl_060526/orbitrap_maxmin_train.pkl \
--out ./data/cl_pkl_060526/orbitrap_maxmin_cek16_train.pkl \
--cap 16 --ce_bins 3 --seed 42

python -u train_tcn_gpus_cl.py \
--train_data ./data/cl_pkl_060526/orbitrap_maxmin_cek16_train.pkl \
--test_data ./data/cl_pkl_060526/orbitrap_maxmin_test.pkl \
--config_path ./msfiddle/config/fiddle_tcn_orbitrap.yml \
--checkpoint_path ./check_point/fiddle_tcn_orbitrap_060526_cek16.pt \
--result_path ./result/fiddle_tcn_orbitrap_060526_cek16.csv --device 4 7

python scripts/prepare_augment_rescore.py \
--train_data ./data/cl_pkl_060526/orbitrap_maxmin_cek16_train.pkl \
--test_data ./data/cl_pkl_060526/orbitrap_maxmin_test.pkl \
--config_path ./msfiddle/config/fiddle_tcn_orbitrap.yml \
--resume_path ./check_point/fiddle_tcn_orbitrap_060526_cek16.pt \
--rescore_dir ./data/cl_pkl_060526 \
--pos_cap 10 --neg_per_pos 8 --tolerance 50 --num_workers 8 --device 4
python scripts/train_rescore.py \
--train_data ./data/cl_pkl_060526/orbitrap_maxmin_cek16_rescore_train.pkl \
--test_data ./data/cl_pkl_060526/orbitrap_maxmin_rescore_test.pkl \
--config_path ./msfiddle/config/fiddle_tcn_orbitrap.yml \
--resume_path ./check_point/fiddle_tcn_orbitrap_060526_cek16.pt \
--checkpoint_path ./check_point/fiddle_rescore_orbitrap_060526_cek16.pt --device 4

python scripts/run_fiddle.py --test_data ./data/cl_pkl_060526/orbitrap_maxmin_test.mgf \
--config_path ./msfiddle/config/fiddle_tcn_orbitrap.yml \
--resume_path ./check_point/fiddle_tcn_orbitrap_060526_cek16.pt \
--rescore_resume_path ./check_point/fiddle_rescore_orbitrap_060526_cek16.pt \
--result_path ./result/fiddle_orbitrap_060526_cek16.csv --device 4
python scripts/eval_topk.py --result ./result/fiddle_orbitrap_060526_cek16.csv \
--test_data ./data/cl_pkl_060526/orbitrap_maxmin_test.pkl --topk 5
# Compare against PART 2b (baseline).


