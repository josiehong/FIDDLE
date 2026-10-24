<p align="center">
  <a href="https://github.com/JosieHong/FIDDLE">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/JosieHong/FIDDLE/main/img/fiddle-logo-dark@2x.png">
      <img src="https://raw.githubusercontent.com/JosieHong/FIDDLE/main/img/fiddle-logo-light@2x.png" alt="FIDDLE" width="480">
    </picture>
  </a>
</p>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19181279-blue)](https://doi.org/10.5281/zenodo.19181279)
[![Release](https://img.shields.io/github/v/release/JosieHong/FIDDLE?label=Release)](https://github.com/JosieHong/FIDDLE/releases)

FIDDLE predicts molecular formulas from MS/MS spectra with deep learning.

**[Paper](https://www.nature.com/articles/s41467-025-66060-9)** · **[Documentation](https://msfiddle.readthedocs.io)** · **[Try the demo on Hugging Face](https://huggingface.co/spaces/J0siee/FIDDLE)**

## Quick start

Install the [`msfiddle`](https://pypi.org/project/msfiddle/) package, download the pre-trained models once, then predict formulas:

```bash
pip install "msfiddle[inference]"
msfiddle download
msfiddle predict --demo --result_path demo.csv        # 21 bundled example spectra

# your own spectra
msfiddle predict --test_data spectra.mgf --instrument_type orbitrap --result_path results.csv
```

Use `--instrument_type qtof` for Q-TOF data and `--instrument_type timstof` for timsTOF data. Each spectrum needs `TITLE`, `PRECURSOR_MZ`, `PRECURSOR_TYPE` and `COLLISION_ENERGY`. The results list ranked candidates per spectrum in `Refined Formula (0)`, `Refined Formula (1)`, … with their scores in `Rescore (k)`.

When the true formulas are known (a `SMILES` or `FORMULA` field in each spectrum), add `--eval` to also report top-k accuracy, saved as `results_eval.csv`.

The [documentation](https://msfiddle.readthedocs.io) covers the Python API, BUDDY and SIRIUS integration, custom checkpoints, all output columns, and [benchmark results](https://msfiddle.readthedocs.io/en/latest/benchmarks.html).

## Reproduce the paper

This repository also holds the research code for training and evaluation. Run everything from the repository root (`FIDDLE/`).

1. Create the environment with [Anaconda](https://docs.anaconda.com/anaconda/install/) (use `environment_mac.yml` on Apple Silicon):

```bash
conda env create -f environment.yml
```

2. (optional) Install [BUDDY](https://github.com/Philipbear/msbuddy) and [SIRIUS](https://v6.docs.sirius-ms.io/) for the comparison experiments. SIRIUS needs a free account; log in once as `install_sirius.sh` prints at the end.

```bash
bash running_scripts/install_buddy.sh
bash running_scripts/install_sirius.sh
```

3. Download the models into `./check_point/`, where the scripts expect them:

```bash
msfiddle download -d ./check_point
```

| Script | Description |
|---|---|
| `running_scripts/train_release_v2.3.sh` | Build the data, train the release models (TCN and rescore) for Q-TOF, Orbitrap and timsTOF in parallel and evaluate them; `INSTRUMENTS=timstof` runs one of them. timsTOF trains on the Enveda CASMI 2026 library at `data/origin/casmi2026/train.parquet` |
| `running_scripts/experiments_test_benchmark.sh` | Evaluate on external benchmarks (CASMI 2016, CASMI 2017, EMBL-MCF 2.0) |
| `running_scripts/experiments_test_nist23.sh` | Train and evaluate with compounds new in NIST23 held out |
| `running_scripts/experiments_test_chimeric.sh` | Evaluate on chimeric spectra |
| `running_scripts/experiments_test_noised.sh` | Evaluate on spectra with added noise peaks |

The benchmark, chimeric and noise experiments use the released models; the chimeric and noise ones also need the data those models were trained on.

To train from scratch, see `scripts/train_tcn_gpus_cl.py` and `scripts/train_rescore.py` with the configs in [`msfiddle/config/`](https://github.com/JosieHong/FIDDLE/tree/main/msfiddle/config).

## Citation

If you use FIDDLE, please cite:

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
