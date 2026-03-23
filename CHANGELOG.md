# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-03-23

### Added

- `test_caffeine.py`: inference scripts for caffeine (C8H10N4O2) GNPS spectra.
- `running_scripts/retrain_031826.sh`: end-to-end retraining script for both Orbitrap and Q-TOF (031826 data).
- `train_rescore.py`: Siamese rescore trainer. Freezes the TCN spectrum encoder; trains `FormulaEncoder` + `RescoreHead` with BCE loss. Checkpoint stores `formula_encoder_state_dict` and `rescore_head_state_dict`.
- `prepare_augment_rescore.py`: unified rescore data preparation script. Takes the TCN train and test sets directly, runs inference on both, and augments the train split — capping positives per formula (`--pos_cap`), generating cross-spectrum negatives within a precursor m/z window (`--tolerance` ppm), and downsampling to 1:1 positive:negative ratio. Formula refinement is parallelised via `multiprocessing.Pool` (`--num_workers`). The test split is saved without augmentation.
- `model_tcn.py`: added `FormulaEncoder` (atom-count vector → 512-dim L2-normalised embedding) and `RescoreHead` (element-wise product `z_spec ⊙ z_form` → scalar logit).
- `environment_mac.yml` for macOS ARM64 (Apple Silicon) setup.
- Docstrings added across `model_tcn.py`, `dataset.py`, `utils/mol_utils.py`, `utils/msms_utils.py`, `utils/pkl_utils.py`, and `utils/refine_utils.py`.

### Changed

- `train_rescore.py`, `run_fiddle.py`: replaced `FDRNet`-based reranking with the Siamese rescore architecture. Output CSV columns renamed from `FDR (k)` to `Rescore (k)`.
- Rescore pipeline (`train_rescore.py`, `run_fiddle.py`, `test_caffeine.py`): `env[:, 0]` (precursor m/z) is zeroed before the spectrum encoder to prevent the model from learning a mass-based frequency prior.

### Removed

- `model_tcn.py`: removed `FDRNet` class.

### Fixed

- `config/fiddle_tcn_orbitrap.yml`: added `'ftms'` to the `gnps_orbitrap` instrument allowlist. Orbitrap dataset expanded to 28,751 training / 3,195 test compounds.
- `prepare_augment_rescore.py`, `run_fiddle.py`: `refine_atom_type` is now extended with atoms present in the predicted formula before calling `formula_refinement`, ensuring the refinement search space at training time matches inference.
- `train_tcn_gpus_cl.py`: fixed result export loading from `resume_path` instead of `checkpoint_path`.
- `train_tcn_gpus_cl.py`: embeddings are L2-normalized before contrastive loss; gradient clipping added (`max_norm=1.0`); division by zero in H/C ratio target fixed (`y[:, 0].clamp(min=1)`); checkpoint saved only when `formula_acc` (with H) improves.
- `model_tcn.py`: multi-scale feature collection now uses `isinstance(layer, TemporalBlock)` instead of `i % 2 == 0`.

## [1.1.0] - 2025-08-20

### Added

- Ablation study scripts (`prepare_msms_ablation.py`, `prepare_msms_ablation_ins.py`) and corresponding running scripts for systematic evaluation of model components.
- Chimeric spectra experiment (`prepare_msms_chimeric.py`, `running_scripts/experiments_test_chimeric.sh`) to evaluate robustness to co-eluting compounds.
- Noised spectra experiment (`prepare_msms_noised.py`, `running_scripts/experiments_test_noised.sh`) to evaluate robustness to spectral noise.
- Demo training and evaluation script (`running_scripts/experiments_demo.sh`).

## [1.0.0] - 2024-11-26

### Added

- Initial FIDDLE version.
