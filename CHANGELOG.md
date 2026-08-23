# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Merged the [msfiddle](https://pypi.org/project/msfiddle/) PyPI package repository into this repo. The `msfiddle/` package directory is now the single source of truth for the shared core (`model_tcn.py`, `dataset.py`, `utils/`, `config/`, `demo/`); the research scripts import from it and the former top-level copies are removed. Packaging (`setup.py`), docs (readthedocs), package tests, and the PyPI publish workflow now live here. Pre-merge package release history is appended at the bottom of this file.
- The package picks up fixes that had only landed on the research side: `fast_mass` refinement speedup, the `generate_ms` per-bin m/z fix, the Python 3.11 `random.sample` fix, and the halogen-extended refinement configs.
- Research scripts moved from the repo root into `scripts/` (shell drivers stay in `running_scripts/`, with paths updated). Each script bootstraps the repo root onto `sys.path` and the conda environments now install the package in editable mode, so `msfiddle` imports resolve regardless of working directory; scripts should still be run from the repo root so their relative default paths resolve.

## [2.1.0] - 2026-06-13

### Added

- `eval_topk.py`: top-1..top-K formula accuracy from a `run_fiddle` result CSV and a test set, reported both per spectrum (micro) and per compound (macro, each compound weighted equally — robust to replicate-rich compounds).
- `tests/test_generate_ms.py`: first pytest in the repo; regression tests for the spectral-binning m/z channel.
- `running_scripts/experiments_release_v2.1.sh`: end-to-end maxmin rebuild + evaluation of both release models (Q-TOF + Orbitrap) on the corrected pipeline.
- `logs/experiment_v3.md`: experiment notes for this cycle (training-set composition, precursor-isotope feasibility, halogen reachability/rescoring, efficiency).

### Changed

- `config/fiddle_tcn_qtof.yml`, `config/fiddle_tcn_orbitrap.yml`: refinement search space extended with halogens (`refine_atom_type` now `C/O/N/H + Cl/Br/F/I`) and candidate budget widened (`top_k` 5 → 10), so halogenated formulas are reachable and not crowded out of the returned set. On the Q-TOF release split this raises chlorine top-1 from 3.4% to ~26% while preserving overall accuracy (~70%).
- `utils/refine_utils.py`: monoisotopic mass in the refinement search now uses a memoised `fast_mass` (~19× faster), numerically identical to the previous implementation up to floating-point summation order (far below the ppm acceptance window).
- `utils/pkl_utils.py` (`generate_ms`): the per-bin representative-m/z channel now records the m/z of the most intense peak in each bin (previously a dead branch left it identically zero). Not consumed by the current model.
- Released Q-TOF and Orbitrap models retrained on the corrected pipeline.

### Fixed

- `prepare_msms.py`: replaced the O(n²) train/test partition (linear membership tests) with set lookups; made the train/test split reproducible by sorting the deduplicated compound list (hash-randomised set iteration previously made splits differ run-to-run even with a fixed seed).
- `utils/msms_utils.py`: the `[M-H2O-H]-` adduct is now handled (treated as equivalent to `[M-H-H2O]-`); it was accepted by the data filter but unrecognised by the m/z calculator, which crashed preprocessing.
- `train_tcn_gpus.py`, `train_tcn_gpus_cl.py`: early-stop best-metric initialisation changed from `0` to `-1` (`best_formula_acc`, `best_formula_wo_acc`), so a first epoch with 0 accuracy counts as an improvement instead of spuriously incrementing the early-stop counter at epoch 1.

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

---

# msfiddle package (pre-merge history)

Releases of the `msfiddle` PyPI package while it lived in its own repository (`josiehong/msfiddle`, now merged here).

## [2.1.0] - 2026-06-05

### Added
- Accept native/original BUDDY/msbuddy output for `--buddy_path` (and the Python API): an `msbuddy_result_summary.tsv` file, or the full output directory. When the directory contains per-query `formula_results.tsv` files (msbuddy `-d`), their per-candidate FDR scores are used for ranks 2–5.
- Accept native/original SIRIUS formula summaries for `--sirius_path` (and the Python API): a `formula_identifications` file (TSV/CSV/XLSX) or a SIRIUS summary output directory.

### Deprecated
- The msfiddle-normalized BUDDY and SIRIUS CSV formats are deprecated and will be removed in 3.0.0. Pass native/original msbuddy or SIRIUS output instead. Loading a normalized CSV now emits a `DeprecationWarning`.

## [2.0.1] - 2026-05-02

### Added
- Added `MsFiddlePredictor` for reusable Python inference with single-spectrum, batch, and MGF prediction methods.
- Added `predict_from_spectrum`, `predict_batch_from_spectra`, and `predict_from_mgf` convenience APIs.
- Added the optional `inference` extra for installing PyTorch with `pip install "msfiddle[inference]"`.

### Changed
- Refactored the CLI to use the shared predictor internals while preserving the existing command-line interface and CSV output shape.
- Deferred checkpoint warnings/errors until prediction instead of warning during import.
- Set package metadata to require Python 3.8+, matching the existing pandas 2 dependency.
- Derived checkpoint downloads from the package major version, so all `2.*.*` releases use the FIDDLE `v2.0.0` checkpoint assets.

## [2.0.0] - 2026-03-23

### Changed
- Replaced `FDRNet` with a Siamese-style rescoring architecture: new `FormulaEncoder` (MLP → L2-normalised embedding) and `RescoreHead` (element-wise product → scalar logit) classes in `model_tcn.py`
- Renamed `FDRDataset` → `RescoreDataset` in `dataset.py` and updated references from `prepare_fdr.py` to `prepare_rescore.py`
- Renamed `train_fdr` config section to `train_rescore` across all four config YAMLs
- Reduced `early_stop_step` from 10 to 5 in Orbitrap and Q-TOF training configs

### Added
- `formula_dim: 64` parameter added to Orbitrap and Q-TOF model configs

## [0.1.0] - 2025-03-20

### Added
- Initial release
- Chemical formula prediction from tandem mass spectra (MS/MS) using pre-trained TCN models
- Support for Orbitrap and Q-TOF instrument types
- Formula refinement with confidence scoring (FDR)
- Integration with BUDDY and SIRIUS results
- `msfiddle` CLI for running predictions
- `msfiddle-download-models` CLI for downloading pre-trained model weights
- `msfiddle-checkpoint-paths` CLI for inspecting model locations
- Demo data for quick testing (`--demo` flag)
