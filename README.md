# FIDDLE

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19181279-blue)](https://doi.org/10.5281/zenodo.19181279)
[![Release](https://img.shields.io/github/v/release/JosieHong/FIDDLE?label=Release)](https://github.com/JosieHong/FIDDLE/releases)

[<img src="https://raw.githubusercontent.com/JosieHong/FIDDLE/main/img/fiddle_logo.png" align="right" width="220">](https://github.com/JosieHong/FIDDLE)

FIDDLE is a deep learning method for predicting molecular formulas from MS/MS spectra. This repository contains the full research codebase for model training, evaluation, and paper reproduction, plus the [`msfiddle`](https://pypi.org/project/msfiddle/) PyPI package (in [`./msfiddle/`](https://github.com/JosieHong/FIDDLE/tree/main/msfiddle)) for inference-only use.

- **Paper:** [Nature Communications (2025)](https://www.nature.com/articles/s41467-025-66060-9)
- **CLI and Python API:** `pip install "msfiddle[inference]"` ([package docs](https://msfiddle.readthedocs.io))
- **Try this demo!** [FIDDLE on Hugging Face](https://huggingface.co/spaces/J0siee/FIDDLE)

<br clear="right"/>

> **Breaking change (v2.0.0):** The rescore model has been redesigned (Siamese architecture), see details in [CHANGELOG.md](https://github.com/JosieHong/FIDDLE/blob/main/CHANGELOG.md).

## Set up

### Requirements

1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/), if not already installed.

2. Create the environment with the necessary packages:

```bash
conda env create -f environment.yml
```

On Apple Silicon (macOS ARM64), use `environment_mac.yml` instead.

3. (optional) Install [BUDDY](https://github.com/Philipbear/msbuddy) and [SIRIUS](https://v6.docs.sirius-ms.io/) following the respective installation instructions provided in each tool's documentation. 

## Usage

Download the pre-trained models once per release, then predict formulas for an MGF file:

```bash
msfiddle download
msfiddle predict --test_data spectra.mgf --instrument_type orbitrap --result_path results.csv
```

Use `--instrument_type qtof` for Q-TOF data. Each spectrum needs `TITLE`, `PRECURSOR_MZ`, `PRECURSOR_TYPE` and `COLLISION_ENERGY`. [`msfiddle/demo/input_msms.mgf`](https://github.com/JosieHong/FIDDLE/blob/main/msfiddle/demo/input_msms.mgf) has 21 example spectra from EMBL-MCF 2.0, and `msfiddle predict --demo --result_path demo.csv` runs on them.

When the true formulas are known (a `SMILES` or `FORMULA` field in each spectrum), add `--eval` to also report top-k accuracy, saved as `results_eval.csv`.

The [documentation](https://msfiddle.readthedocs.io) covers the Python API, BUDDY and SIRIUS integration, custom checkpoints, and the output columns. [`scripts/test_caffeine.py`](https://github.com/JosieHong/FIDDLE/blob/main/scripts/test_caffeine.py) runs FIDDLE on a caffeine Orbitrap spectrum fetched from GNPS.

Models download to `~/.msfiddle/check_point/<release>/`. Research scripts that expect `./check_point/` can use `msfiddle download -d ./check_point`.

## Reproduce paper results

All scripts should be run from the repository root (`FIDDLE/`).

| Script | Description |
|---|---|
| `running_scripts/experiments_test_benchmark.sh` | Evaluate on external benchmarks (CASMI 2016, CASMI 2017, EMBL-MCF 2.0) |
| `running_scripts/experiments_test_nist23.sh` | Evaluate on NIST23 |
| `running_scripts/experiments_test_chimeric.sh` | Evaluate on chimeric spectra |
| `running_scripts/experiments_test_noised.sh` | Evaluate under noise conditions |
| `running_scripts/experiments_ablation_study.sh` | Run ablation study |
| `running_scripts/experiments_demo.sh` | Run demo experiment |
| `running_scripts/train_released_models.sh` | Train TCN and rescore models for both Orbitrap and Q-TOF |

For training from scratch, see the train scripts (`scripts/train_tcn_gpus.py`, `scripts/train_tcn_gpus_cl.py`, `scripts/train_rescore.py`) and the corresponding config files in `./msfiddle/config/`.

## Citation

```
@article{hong2025fiddle,
  title={FIDDLE: a deep learning method for chemical formulas prediction from tandem mass spectra},
  author={Hong, Yuhui and Li, Sujun and Ye, Yuzhen and Tang, Haixu},
  journal={Nature Communications},
  volume={16},
  number={1},
  pages={11102},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```
