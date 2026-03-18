# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `environment_mac.yml` for macOS ARM64 (Apple Silicon) setup with CPU-only PyTorch and compatible package versions.
- `test_caffine.py` debugging script that fetches caffeine spectra from GNPS and local filtered MGF files, runs the full FIDDLE inference pipeline, and compares FDR-ranked vs. combined (FDR × exp(−ppm/λ)) scoring across multiple sources and collision energies.

### Fixed

- `config/fiddle_tcn_orbitrap.yml`: added `'ftms'` to the `gnps_orbitrap` instrument allowlist. The Orbitrap dataset was expanded to 28,751 training and 3,195 test compounds, with 894,762 training and 121,991 test spectra.
- `prepare_fdr.py`: `refine_atom_type` is now extended with atoms present in the predicted formula before calling `formula_refinement`, matching the search space used at inference time. Previously, atoms predicted by the model but absent from the config (e.g. F, S) were frozen in the refinement search, making FDR training candidates inconsistent with inference.
- `prepare_fdr.py`: corrected a misleading comment that said "experimental precursor m/z" when the code actually uses the true monoisotopic mass from the formula.
- `run_fiddle.py`: same atom type extension fix as `prepare_fdr.py`. When the model incorrectly predicts a rare atom (e.g. F), the refinement search can now navigate away from it instead of being stuck in a formula subspace that excludes the true answer.

## [1.1.0] - 2025-08-20

### Added

- Ablation study scripts (`prepare_msms_ablation.py`, `prepare_msms_ablation_ins.py`) and corresponding running scripts for systematic evaluation of model components.
- Chimeric spectra experiment (`prepare_msms_chimeric.py`, `running_scripts/experiments_test_chimeric.sh`) to evaluate robustness to co-eluting compounds.
- Noised spectra experiment (`prepare_msms_noised.py`, `running_scripts/experiments_test_noised.sh`) to evaluate robustness to spectral noise.
- Demo training and evaluation script (`running_scripts/experiments_demo.sh`).

## [1.0.0] - 2024-11-26

### Added

- Initial FIDDLE version. 
