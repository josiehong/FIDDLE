# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Docstrings added to commonly used functions and classes across `model_tcn.py`, `dataset.py`, `utils/mol_utils.py`, `utils/msms_utils.py`, `utils/pkl_utils.py`, and `utils/refine_utils.py`.
- `environment_mac.yml` for macOS ARM64 (Apple Silicon) setup with CPU-only PyTorch and compatible package versions.
- `test_caffine.py` debugging script that fetches caffeine spectra from GNPS and local filtered MGF files, runs the full FIDDLE inference pipeline, and compares FDR-ranked vs. combined (FDR × exp(−ppm/λ)) scoring across multiple sources and collision energies.

### Fixed

- `config/fiddle_tcn_orbitrap.yml`: added `'ftms'` to the `gnps_orbitrap` instrument allowlist. The Orbitrap dataset was expanded to 28,751 training and 3,195 test compounds, with 894,762 training and 121,991 test spectra.
- `config/fiddle_tcn_orbitrap.yml`: added missing `'[M-H2O-H]-': '-1'` entry to `type2charge`, which caused a `KeyError` during data preparation when this adduct appeared in the GNPS Orbitrap dataset.
- `config/fiddle_tcn_orbitrap.yml`: reverted learning rate from `0.002` back to `0.001` to match the standard training setup.
- `config/fiddle_tcn_{casmi,demo,demo_wo_da,embl,qtof}.yml`: corrected `'[M-H2O-H]+'` → `'[M-H2O-H]-'` in `type2charge`; the adduct carries a negative charge and the `+` sign was a typo.
- `prepare_fdr.py`: `refine_atom_type` is now extended with atoms present in the predicted formula before calling `formula_refinement`, matching the search space used at inference time. Previously, atoms predicted by the model but absent from the config (e.g. F, S) were frozen in the refinement search, making FDR training candidates inconsistent with inference.
- `prepare_fdr.py`: corrected a misleading comment that said "experimental precursor m/z" when the code actually uses the true monoisotopic mass from the formula.
- `run_fiddle.py`: same atom type extension fix as `prepare_fdr.py`. When the model incorrectly predicts a rare atom (e.g. F), the refinement search can now navigate away from it instead of being stuck in a formula subspace that excludes the true answer.
- `train_tcn_gpus_cl.py`: fixed `--result_path` export loading from `resume_path` instead of `checkpoint_path`, which caused the wrong (pre-training) model to be evaluated after training.
- `train_tcn_gpus_cl.py`: embeddings are now L2-normalized (`F.normalize(z, dim=1)`) before being passed to the contrastive loss, projecting representations onto the unit hypersphere for more stable similarity computation.
- `train_tcn_gpus_cl.py`: added gradient clipping (`clip_grad_norm_`, max norm 1.0) after the backward pass to prevent exploding gradients during contrastive training.
- `train_tcn_gpus_cl.py`: fixed division by zero in H/C ratio target (`y[:, 0].clamp(min=1)`) to prevent NaN loss for samples with no carbon.
- `train_tcn_gpus_cl.py`: checkpoint is now saved only when `formula_acc` (with H) improves, replacing the previous OR condition across three metrics that made early stopping unreliable.
- `train_tcn_gpus_cl.py`: `WarmUpScheduler` now properly inherits from `_LRScheduler` (calls `super().__init__`) so its state is correctly saved and restored via `state_dict`.
- `train_tcn_gpus_cl.py`: resume now loads the checkpoint file once instead of five separate times.
- `model_tcn.py`: `FDRNet._build_decoder` renamed to `_build_fdr_decoder` to avoid overriding the parent's method during `__init__`, which was silently building `decoder_formula/mass/atomnum/hcnum` with the wrong input dimension.
- `model_tcn.py`: removed `LeakyReLU` from the final layer of `FDRNet`'s FDR decoder; sigmoid is applied externally in `rerank_by_fdr`, so the pre-sigmoid logits are now unconstrained.
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
