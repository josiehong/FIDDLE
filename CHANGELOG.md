# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Docstrings added to commonly used functions and classes across `model_tcn.py`, `dataset.py`, `utils/mol_utils.py`, `utils/msms_utils.py`, `utils/pkl_utils.py`, and `utils/refine_utils.py`.
- `environment_mac.yml` for macOS ARM64 (Apple Silicon) setup with CPU-only PyTorch and compatible package versions.
- `test_caffeine.py`: debugging script that runs the full FIDDLE inference pipeline on a caffeine (C8H10N4O2) spectrum fetched live from GNPS (CCMSLIB00016149314), reporting rescore-ranked results. Requires no local data files.
- `model_tcn.py`: added `FormulaEncoder` — a small MLP (`output_dim → 64 → 256 → embedding_dim`, LayerNorm + ReLU, L2-normalised output) that maps a formula atom-count vector into the same embedding space as the TCN spectrum encoder.
- `model_tcn.py`: added `RescoreHead` — a lightweight MLP (`embedding_dim → 256 → 64 → 1`, ReLU + Dropout(0.2)) that takes the element-wise product `z_spec ⊙ z_form` and produces a scalar logit.
- `train_rescore.py`: Siamese rescore trainer. Freezes the TCN spectrum encoder; trains `FormulaEncoder` + `RescoreHead` jointly with BCE loss on standard per-sample `RescoreDataset`. Score = `sigmoid(RescoreHead(z_spec ⊙ FormulaEncoder(f)))`. Checkpoint stores `formula_encoder_state_dict` and `rescore_head_state_dict`.

### Changed

- `train_rescore.py`: replaced the original `FDRNet`-based trainer (spectrum embedding concatenated with formula vector → MLP → BCE) with the Siamese interaction-head design.
- `run_fiddle.py`: replaced `FDRNet`-based reranking with Siamese rescoring (`rescore_candidates`). Candidates are ranked directly by the rescore score. Output CSV columns renamed from `FDR (k)` to `Rescore (k)`.
- `train_rescore.py`, `run_fiddle.py`, `test_caffeine_orbitrap.py`, `test_caffeine_qtof.py`: `env[:, 0]` (precursor m/z) is zeroed before passing to the spectrum encoder during rescoring. The original model was learning a global mass-based frequency prior via the precursor m/z shortcut instead of using spectral features.
- `prepare_rescore.py`: removed `--train_data` argument; rescore data is now prepared solely from the test set to avoid distribution shift caused by the formula model memorizing training spectra. The test set is split into rescore train/test splits at the compound (SMILES) level, ensuring no compound appears on both sides. Outputs `_rescore_train.pkl` and `_rescore_test.pkl`. A `--train_ratio` argument (default `0.8`) controls the compound split fraction.

### Removed

- `model_tcn.py`: removed `FDRNet` class. Reranking is now handled by the Siamese rescore architecture (`FormulaEncoder` + `RescoreHead`).

### Fixed

- `config/fiddle_tcn_orbitrap.yml`: added `'ftms'` to the `gnps_orbitrap` instrument allowlist. The Orbitrap dataset was expanded to 28,751 training and 3,195 test compounds, with 894,762 training and 121,991 test spectra.
- `prepare_rescore.py`: `refine_atom_type` is now extended with atoms present in the predicted formula before calling `formula_refinement`, matching the search space used at inference time. Previously, atoms predicted by the model but absent from the config (e.g. F, S) were frozen in the refinement search, making rescore training candidates inconsistent with inference.
- `run_fiddle.py`: same atom type extension fix as `prepare_rescore.py`. When the model incorrectly predicts a rare atom (e.g. F), the refinement search can now navigate away from it instead of being stuck in a formula subspace that excludes the true answer.
- `train_tcn_gpus_cl.py`: fixed `--result_path` export loading from `resume_path` instead of `checkpoint_path`, which caused the wrong (pre-training) model to be evaluated after training.
- `train_tcn_gpus_cl.py`: embeddings are now L2-normalized (`F.normalize(z, dim=1)`) before being passed to the contrastive loss, projecting representations onto the unit hypersphere for more stable similarity computation.
- `train_tcn_gpus_cl.py`: added gradient clipping (`clip_grad_norm_`, max norm 1.0) after the backward pass to prevent exploding gradients during contrastive training.
- `train_tcn_gpus_cl.py`: fixed division by zero in H/C ratio target (`y[:, 0].clamp(min=1)`) to prevent NaN loss for samples with no carbon.
- `train_tcn_gpus_cl.py`: checkpoint is now saved only when `formula_acc` (with H) improves, replacing the previous OR condition across three metrics that made early stopping unreliable.
- `model_tcn.py`: multi-scale feature collection now uses `isinstance(layer, TemporalBlock)` instead of `i % 2 == 0` to correctly identify TCN layers regardless of layer ordering.

## [1.1.0] - 2025-08-20

### Added

- Ablation study scripts (`prepare_msms_ablation.py`, `prepare_msms_ablation_ins.py`) and corresponding running scripts for systematic evaluation of model components.
- Chimeric spectra experiment (`prepare_msms_chimeric.py`, `running_scripts/experiments_test_chimeric.sh`) to evaluate robustness to co-eluting compounds.
- Noised spectra experiment (`prepare_msms_noised.py`, `running_scripts/experiments_test_noised.sh`) to evaluate robustness to spectral noise.
- Demo training and evaluation script (`running_scripts/experiments_demo.sh`).

## [1.0.0] - 2024-11-26

### Added

- Initial FIDDLE version.
