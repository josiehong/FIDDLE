# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `running_scripts/train_release_v2.3.sh`: rebuilds the Q-TOF data, trains the Q-TOF TCN, retrains both rescorers over several seeds (keeping the best by validation top-1) and evaluates them.
- `prepare_msms.py --report_only` loads and filters the raw libraries and reports what each dataset keeps and drops, including isotope-peak precursors and stepped collision energies, without the split, encoding or output files.
- Isotope-error search ([#8](https://github.com/JosieHong/FIDDLE/issues/8)): `post_processing.isotope_errors` (default `[0]`) lists 13C isotope offsets k to refine at, for precursors read from the M+k isotope peak, and `isotope_penalty` (default 0.5) is subtracted from the rescore per unit of k. Results gain `Isotope Shift (k)` columns when enabled. `prepare_augment_rescore.py` adds the candidates found at those offsets as decoys, so the rescorer is trained on them.
- `msfiddle.settings.load_config`: every config is read through it. It fills defaults, rejects invalid post-processing settings, and warns when a dataset's `ppm_tolerance` is wider than `mass_tolerance`.
- `scripts/run_buddy_sirius.py` runs msbuddy and SIRIUS on an MGF file, setting `ION` from `PRECURSOR_TYPE` so both tools see the adduct, and `running_scripts/compare.sh` runs FIDDLE alone, with BUDDY, and with BUDDY and SIRIUS for each experiment. The experiment scripts run BUDDY and SIRIUS through it again.
- `scripts/check_precursor_error.py` classifies each spectrum's precursor mass error as within tolerance, a 13C isotope peak, or other, and flags spectra within tolerance on the m/z but not on the neutral mass. Preprocessing reports isotope-peak precursors as `isotope_error` instead of `ppm_tolerance`; they are still dropped.

### Changed

- Rescore checkpoints are chosen by top-1 on validation spectra instead of pair AUC on the test pairs. `prepare_augment_rescore.py` holds out a share of the training compounds (`--val_ratio`, default 0.1) as validation pairs and no longer builds test pairs; `train_rescore.py` takes `--val_data` instead of `--test_data`. The test set is then used only for the final evaluation. Pair-level AUC barely changes between retrains whose top-1 differs by several points.
- The refinement window `mass_tolerance` is in ppm of the precursor m/z, the unit of the data filters' `ppm_tolerance`, and is converted to each adduct's neutral mass ([#9](https://github.com/JosieHong/FIDDLE/issues/9)). It used to be ppm of the neutral mass, so for adducts lighter than their ion, such as small `[M+Na]+`, spectra the filter admitted could fall outside the window. The CASMI and EMBL configs' windows are raised to their filters' tolerances (10 and 50 ppm), which changes those benchmark results. Rescore training pairs use the same window.
- Spectra with a stepped or ranged collision energy (`HCD (NCE 20-30-40%)`, `30-50 eV`) are skipped with one warning that gives the count, where `predict` used to drop them without saying why; `predict_from_spectrum` raises a clear error. Preprocessing reports how many it drops. The models were trained on single-energy spectra, so these are not encoded as a mean energy. ([#7](https://github.com/JosieHong/FIDDLE/issues/7))
- BUDDY and SIRIUS results are looked up by spectrum ID through an index, not by scanning the whole table for every spectrum. Results are unchanged.

### Fixed

- Preprocessing dropped every spectrum of a library passed in as a one-pass MGF reader: `filter_spec` read its input twice, and the second pass was empty. One Q-TOF training library was affected; data built since the August 2026 preprocessing speed-up, including the 2.2 release data, lacks its spectra.

### Removed

- `running_scripts/train_release_v2.2.sh`, replaced by `train_release_v2.3.sh`.
- `scripts/train_tcn_gpus.py`; `scripts/train_tcn_gpus_cl.py` trains the TCN.
- `--test_title_list` in `prepare_msms.py` and `prepare_msms_nist23.py`, which nothing used.
- Unused imports and a duplicate collision-energy pattern in the package, and the manual test scripts at the end of the `msfiddle/utils` modules.

## [2.2.1] - 2026-10-19

### Added

- `msfiddle` subcommands: `predict`, `download` and `paths`. Arguments without a subcommand still go to `predict`, `python -m msfiddle` works without installing, and `msfiddle-download-models` / `msfiddle-checkpoint-paths` remain as aliases.
- `msfiddle predict --eval`: after predicting, report top-k accuracy per spectrum and per compound against the input MGF's SMILES or FORMULA fields, and save it as `<result>_eval.csv`.
- `--num_workers` for `msfiddle predict` and `num_workers` for `predict_mgf` / `predict_from_mgf`: formula refinement in parallel processes, with identical results.

### Changed

- Research evaluation uses the package CLI (`python -m msfiddle predict --eval`), so reported accuracy comes from the same code users run. `scripts/run_fiddle.py` and `scripts/eval_topk.py` are removed. The package computes the neutral mass from the full-precision precursor m/z, which can swap a decoy candidate at the edge of the mass window; top-k accuracy is unchanged on a 300-spectrum check.
- New FIDDLE logo, with a dark-mode variant, in the README and docs. The README opens with a quick start.
- Faster rescoring: the rescorer's spectrum embedding is computed in the batched model pass, not once more per spectrum. `predict` shows a "Rescore" progress bar. `scripts/train_rescore.py` also embeds each spectrum once, not once per candidate pair.
- The experiment scripts in `running_scripts/` run again: they called scripts and flags that no longer existed. The benchmark, chimeric and noise experiments use the released models, and the NIST23 experiment trains on its own split. The BUDDY and SIRIUS runner steps are removed; FIDDLE + BUDDY runs read existing BUDDY output.

### Fixed

- Evaluation scored spectra with neutral-loss adducts (`[M+H-H2O]+`, `[M-H-H2O]-`, `[M+NH4]+`, `[M+H-NH3]+`, `[M-H-CO2]-`, ...) against the test .pkl target, which is the formula without the neutral loss, while predictions are the intact molecule. Every such spectrum counted as wrong, which affects Orbitrap more than Q-TOF. Evaluation now uses the MGF's SMILES or FORMULA. Accuracies reported for 2.2.0 and earlier are therefore too low; comparisons between models evaluated the same way are unaffected.
- `prepare_augment_rescore.py` built rescore training pairs from the .pkl target, which folds neutral losses into the formula (candidates around M - H2O, positives M - H2O for `[M+H-H2O]+`), while inference rescores candidates around the intact molecule. Pairs are now built for the intact molecule, using each compound's SMILES. Spectra without neutral losses keep the same positives. Rescore models retrained on the corrected pairs performed on par with the 2.2.0 ones, so 2.2.1 keeps the 2.2.0 checkpoints.

### Removed

- `msfiddle.test_step` and `msfiddle.rescore_candidates`, unused leftovers of the pre-API CLI.
- `scripts/analyze_count_ambiguity.py`.
- `running_scripts/experiments_demo.sh` and `experiments_ablation_study.sh`, which depended on removed scripts and configs, with `scripts/prepare_msms_ablation.py` and `prepare_msms_ablation_ins.py`.
- `running_scripts/train_released_models.sh`, `train_release_v2.1.sh` and `scripts/balance_ce_cap.py`; `train_release_v2.2.sh` trains the current models.
- `scripts/plot_training_log.py` and `scripts/test_caffeine.py`.
- Unused `ce2nce` and `exceed_refine_atom_limit` utilities.

### Known issues

- The Orbitrap refinement window is 5 ppm. For small molecules (below about 200 Da), a precursor error under 1 mDa can exceed 5 ppm, and the true formula is then not generated. Widening the window to 20 ppm lowered held-out accuracy, so it stays at 5 ppm until a ppm-plus-absolute tolerance is added.

## [2.2.0] - 2026-10-18

### Added

- `msfiddle/optim.py`: Muon optimizer (momentum + Newton-Schulz orthogonalisation) for the TCN's matrix weights, with AdamW for biases, weight-norm gains and embeddings. CL training now uses it.
- Gradient accumulation for CL training (`grad_accum_steps` in the config), so an effective batch of 512 fits on 11 GB GPUs.
- `scripts/plot_training_log.py`: plots a training log and estimates the remaining time.
- `scripts/analyze_count_ambiguity.py`: error anatomy (exact / H-only / heavy-atom) of a result CSV.

### Changed

- Formula refinement: the best-first search is replaced by a complete bounded enumeration of every formula within the caps, ranked by heavy-atom edit distance from the prediction and then mass error. Faster, and the true formula reaches the candidate set more often.
- Post-processing configs: S/P refinement caps, `top_k` 10 → 20 for both instruments, Orbitrap `mass_tolerance` 20 → 5 ppm to match the data filters.
- Training: fused Siamese forward pass, optional mixed precision (`--amp`), cuDNN autotuning, ReduceLROnPlateau instead of StepLR, and early stopping on validation MAE with a minimum-epoch floor and a rolling last-epoch checkpoint.
- Preprocessing: faster binning, O(n) pair generation, and parallel filtering/encoding.
- Pre-trained checkpoints: from 2.2 on, each minor version downloads from its own release tag (`v2.2.0`), and the local cache is keyed by that tag (`~/.msfiddle/check_point/v2.2.0/`), so upgrading never silently reuses older weights. Run `msfiddle-download-models` after upgrading.
- Released Q-TOF and Orbitrap models retrained from scratch with Muon on rebuilt data, with rescore models retrained on enumeration-generated candidates. Held-out top-1 (per spectrum / per compound): Q-TOF 78.8% / 82.9%, Orbitrap 63.2% / 68.7%.
- Merged the [msfiddle](https://pypi.org/project/msfiddle/) PyPI package repository into this repo. The `msfiddle/` package directory is now the single source of truth for the shared core (`model_tcn.py`, `dataset.py`, `utils/`, `config/`, `demo/`); the research scripts import from it and the former top-level copies are removed. Packaging (`setup.py`), docs (readthedocs), package tests, and the PyPI publish workflow now live here. Pre-merge package release history is appended at the bottom of this file.
- The package picks up fixes that had only landed on the research side: `fast_mass` refinement speedup, the `generate_ms` per-bin m/z fix, the Python 3.11 `random.sample` fix, and the halogen-extended refinement configs.
- `msfiddle/utils/refine_utils.py`: refinement search internals optimised (deque frontier with O(1) membership, `f0_list` parsed once per call, hydrogen adjustment without string round-trips) — ~3.4× faster, output verified identical against the previous implementation on 400 randomised cases.
- Research scripts moved from the repo root into `scripts/` (shell drivers stay in `running_scripts/`, with paths updated). Each script bootstraps the repo root onto `sys.path` and the conda environments now install the package in editable mode, so `msfiddle` imports resolve regardless of working directory; scripts should still be run from the repo root so their relative default paths resolve.

### Fixed

- Preprocessing: contrastive augmentation paired the wrong spectra when `noised_times >= 2`; precursor-isotope removal stepped downward and deleted real fragments; the NCE feature mixed fraction and percent scales; `[M+H-2H2O]+` losses were unparsed (formula labels off by 2 H2O); GNPS adduct handling dropped or corrupted some spectra; plus crash guards for missing molecules, SDF properties, unsupported adducts and peaks at exactly `max_mz`.
- `dataset.py`: padding no longer upcasts spectra to float64, halving the training set's memory.
- `prepare_augment_rescore.py`: each split is loaded once instead of twice.

### Removed

- The Poisson count-likelihood formula loss and `--formula_loss`: an ablation on the rebuilt data showed MSE ahead at every k, so MSE is the only loss.

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
