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
#   - fast_mass refinement speedup (utils/refine_utils.py).
# Note: the mz_val (column 1) preprocessing change is NOT model-visible (the model
#   reads spec[:,0]); the rebuild is driven by the data-level fixes above.
# Adjust --device to your GPUs.
# ============================================================================


# ============================================================================
# PART 1 — RELEASED MODEL: TRAINING  (clean rebuild, date 060526)
# ============================================================================

# --- 1a. Preprocess (maxmin split, fixed prepare_msms.py) --------------------
python prepare_msms.py \
--dataset agilent nist20 nist23 mona waters gnps \
--instrument_type qtof \
--config_path ./config/fiddle_tcn_qtof.yml \
--pkl_dir ./data/cl_pkl_060526/ --maxmin_pick

python prepare_msms.py \
--dataset nist20 nist23 mona gnps \
--instrument_type orbitrap \
--config_path ./config/fiddle_tcn_orbitrap.yml \
--pkl_dir ./data/cl_pkl_060526/ --maxmin_pick

# --- 1b. Train TCN -----------------------------------------------------------
python -u train_tcn_gpus_cl.py \
--train_data ./data/cl_pkl_060526/qtof_maxmin_train.pkl \
--test_data ./data/cl_pkl_060526/qtof_maxmin_test.pkl \
--config_path ./config/fiddle_tcn_qtof.yml \
--checkpoint_path ./check_point/fiddle_tcn_qtof_060526.pt \
--result_path ./result/fiddle_tcn_qtof_060526.csv --device 4 5

python -u train_tcn_gpus_cl.py \
--train_data ./data/cl_pkl_060526/orbitrap_maxmin_train.pkl \
--test_data ./data/cl_pkl_060526/orbitrap_maxmin_test.pkl \
--config_path ./config/fiddle_tcn_orbitrap.yml \
--checkpoint_path ./check_point/fiddle_tcn_orbitrap_060526.pt \
--result_path ./result/fiddle_tcn_orbitrap_060526.csv --device 4 7

# --- 1c. Rescorer (FIDDLES) — trains on the config's refine_atom_type + top_k --
python prepare_augment_rescore.py \
--train_data ./data/cl_pkl_060526/qtof_maxmin_train.pkl \
--test_data ./data/cl_pkl_060526/qtof_maxmin_test.pkl \
--config_path ./config/fiddle_tcn_qtof.yml \
--resume_path ./check_point/fiddle_tcn_qtof_060526.pt \
--rescore_dir ./data/cl_pkl_060526 \
--pos_cap 10 --neg_per_pos 8 --tolerance 50 --num_workers 8 --device 5
python train_rescore.py \
--train_data ./data/cl_pkl_060526/qtof_maxmin_rescore_train.pkl \
--test_data ./data/cl_pkl_060526/qtof_maxmin_rescore_test.pkl \
--config_path ./config/fiddle_tcn_qtof.yml \
--resume_path ./check_point/fiddle_tcn_qtof_060526.pt \
--checkpoint_path ./check_point/fiddle_rescore_qtof_060526.pt --device 5

python prepare_augment_rescore.py \
--train_data ./data/cl_pkl_060526/orbitrap_maxmin_train.pkl \
--test_data ./data/cl_pkl_060526/orbitrap_maxmin_test.pkl \
--config_path ./config/fiddle_tcn_orbitrap.yml \
--resume_path ./check_point/fiddle_tcn_orbitrap_060526.pt \
--rescore_dir ./data/cl_pkl_060526 \
--pos_cap 10 --neg_per_pos 8 --tolerance 50 --num_workers 8 --device 4
python train_rescore.py \
--train_data ./data/cl_pkl_060526/orbitrap_maxmin_rescore_train.pkl \
--test_data ./data/cl_pkl_060526/orbitrap_maxmin_rescore_test.pkl \
--config_path ./config/fiddle_tcn_orbitrap.yml \
--resume_path ./check_point/fiddle_tcn_orbitrap_060526.pt \
--checkpoint_path ./check_point/fiddle_rescore_orbitrap_060526.pt --device 4


# ============================================================================
# PART 2 — EVALUATION  (new release, 060526)
# ============================================================================

# --- 2a. Held-out test + top-k accuracy (per spectrum & per compound): QTOF --
python run_fiddle.py --test_data ./data/cl_pkl_060526/qtof_maxmin_test.mgf \
--config_path ./config/fiddle_tcn_qtof.yml \
--resume_path ./check_point/fiddle_tcn_qtof_060526.pt \
--rescore_resume_path ./check_point/fiddle_rescore_qtof_060526.pt \
--result_path ./result/fiddle_qtof_060526.csv --device 5
python eval_topk.py --result ./result/fiddle_qtof_060526.csv \
--test_data ./data/cl_pkl_060526/qtof_maxmin_test.pkl --topk 5
# Joined: 12627 spectra / 1540 compounds  (matched 12627/12627 result rows)
#             top-1   top-2   top-3   top-4   top-5 
# spectra    70.2%   74.9%   75.8%   76.0%   76.2%
# compound   72.8%   77.3%   78.2%   78.5%   78.7%

# --- 2b. Held-out test + top-k accuracy (per spectrum & per compound): Orbitrap
python run_fiddle.py --test_data ./data/cl_pkl_060526/orbitrap_maxmin_test.mgf \
--config_path ./config/fiddle_tcn_orbitrap.yml \
--resume_path ./check_point/fiddle_tcn_orbitrap_060526.pt \
--rescore_resume_path ./check_point/fiddle_rescore_orbitrap_060526.pt \
--result_path ./result/fiddle_orbitrap_060526.csv --device 4
python eval_topk.py --result ./result/fiddle_orbitrap_060526.csv \
--test_data ./data/cl_pkl_060526/orbitrap_maxmin_test.pkl --topk 5

# --- 2c. External test: CASMI and EMBL (QTOF model) --------------------------
python run_fiddle.py --test_data ./data/casmi2016.mgf \
--config_path ./config/fiddle_tcn_casmi.yml \
--resume_path ./check_point/fiddle_tcn_qtof_060526.pt \
--rescore_resume_path ./check_point/fiddle_rescore_qtof_060526.pt \
--result_path ./result/fiddle_casmi16_060526.csv
python run_fiddle.py --test_data ./data/casmi2017.mgf \
--config_path ./config/fiddle_tcn_casmi.yml \
--resume_path ./check_point/fiddle_tcn_qtof_060526.pt \
--rescore_resume_path ./check_point/fiddle_rescore_qtof_060526.pt \
--result_path ./result/fiddle_casmi17_060526.csv
python run_fiddle.py --test_data ./data/embl_mcf_2.0.mgf \
--config_path ./config/fiddle_tcn_embl.yml \
--resume_path ./check_point/fiddle_tcn_qtof_060526.pt \
--rescore_resume_path ./check_point/fiddle_rescore_qtof_060526.pt \
--result_path ./result/fiddle_embl_060526.csv


