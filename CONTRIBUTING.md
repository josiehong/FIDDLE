# Contributing

Thank you for your interest in contributing! This document explains how to set up your development environment, run tests, and submit changes.

This repository contains both the research codebase (training/evaluation scripts in `scripts/`, experiment drivers in `running_scripts/`) and the installable `msfiddle` package (in `msfiddle/`). The package directory is the single source of truth for the shared core (`msfiddle/model_tcn.py`, `msfiddle/dataset.py`, `msfiddle/utils/`, `msfiddle/config/`) — research scripts import from it.

## Development Setup

**1. Clone the repository**

```bash
git clone https://github.com/JosieHong/FIDDLE.git
cd FIDDLE
```

**2. Create a conda environment**

```bash
conda create -n msfiddle-dev python=3.10
conda activate msfiddle-dev
```

(For the full research environment, use `conda env create -f environment.yml` instead.)

**3. Install the package in editable mode**

```bash
pip install -e .
pip install pytest black
```

**4. Install PyTorch** (required to run the full pipeline, not needed for utility tests)

Follow the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for your system, or use `pip install -e ".[inference]"`.

## Running Tests

```bash
pytest tests/ -v
```

The utility and format tests do **not** require PyTorch or downloaded model weights; torch-dependent tests are skipped when torch is unavailable.

## Code Style

The `msfiddle/` package and `tests/` use [Black](https://black.readthedocs.io/) for code formatting (the research scripts at the repo root are not Black-formatted). Before submitting a pull request that touches the package, format with:

```bash
black msfiddle tests
```

To check formatting without making changes:

```bash
black --check msfiddle tests
```

## What to Contribute

Good candidates for contributions:

- **Bug fixes** in utility functions (`msfiddle/utils/`)
- **New precursor type support** in `msfiddle/utils/msms_utils.py`
- **Test coverage** for currently untested edge cases
- **Documentation** improvements

Please open an issue before starting work on larger changes.

## Submitting a Pull Request

1. Fork the repository and create a branch from `main`
2. Make your changes and add tests if applicable
3. Format package code: `black msfiddle tests`
4. Ensure all tests pass: `pytest tests/ -v`
5. Open a pull request against `main` with a clear description of what was changed and why

## Reporting Issues

Please use [GitHub Issues](https://github.com/JosieHong/FIDDLE/issues) to report bugs. Include:

- Your OS and Python version
- The full error traceback
- A minimal reproducible example if possible
