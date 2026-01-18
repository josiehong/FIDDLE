# FIDDLE

[![DOI](https://zenodo.org/badge/720138825.svg)](https://doi.org/10.5281/zenodo.17172711)

**F**ormula **ID**entification from tandem mass spectra by **D**eep **LE**arning

The source code for the training and evaluation of FIDDLE, as well as for the inference of FIDDLE using results from SIRIUS and BUDDY, is provided (see detailed commands in `./running_scripts/`). A PyPI package and a website-based service for FIDDLE will be available soon. 

Paper: https://www.nature.com/articles/s41467-025-66060-9

Command-line tool: https://pypi.org/project/msfiddle/

## Set up

### Requirements

1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/), if not already installed.

2. Create the environment with the necessary packages:

```bash
conda env create -f environment.yml
```

3. (optional) Install [BUDDY](https://github.com/Philipbear/msbuddy) and [SIRIUS](https://v6.docs.sirius-ms.io/) following the respective installation instructions provided in each tool's documentation. 

### Pre-trained Model Weights

To use the pre-trained models, please use the following scripts to download the weights from the [release page](https://github.com/JosieHong/FIDDLE/releases/tag/v1.0.0) and place them in the `./check_point/` directory:

- **Orbitrap models**:
  - `fiddle_tcn_orbitrap.pt`: formula prediction model on Orbitrap spectra
  - `fiddle_fdr_orbitrap.pt`: confidence score prediction model on Orbitrap spectra
- **Q-TOF models**:
  - `fiddle_tcn_qtof.pt`: formula prediction model on Q-TOF spectra
  - `fiddle_fdr_qtof.pt`: confidence score prediction model on Q-TOF spectra

```bash
bash ./running_scripts/download_models.sh
```

## Usage

The input format is `mgf`, where `title`, `precursor_mz`, `precursor_type`, `collision_energy` fields are required. Here, we sampled 21 spectra from the EMBL-MCF 2.0 dataset as an example.

```mgf
BEGIN IONS
TITLE=EMBL_MCF_2_0_HRMS_Library000531
PEPMASS=129.01941
CHARGE=1-
PRECURSOR_TYPE=[M-H]-
PRECURSOR_MZ=129.01941
COLLISION_ENERGY=50.0
SMILES=[H]OC(=O)C([H])=C(C(=O)O[H])C([H])([H])[H]
FORMULA=C5H6O4
THEORETICAL_PRECURSOR_MZ=129.018785
PPM=4.844255818912111
SIMULATED_PRECURSOR_MZ=129.02032113281717
41.2041 0.410228
55.7698 0.503672
56.8647 0.461943
85.0296 100.0
129.0196 8.036902
END IONS
```

**Run FIDDLE!**

```bash
python run_fiddle.py --test_data ./demo/input_msms.mgf \
                    --config_path ./config/fiddle_tcn_orbitrap.yml \
                    --resume_path ./check_point/fiddle_tcn_orbitrap.pt \
                    --fdr_resume_path ./check_point/fiddle_fdr_orbitrap.pt \
                    --result_path ./demo/output_fiddle.csv --device 0
```

If you'd like to integrate the results from SIRIUS and BUDDY, please organize the results in the format shown in `./demo/buddy_output.csv` and `./demo/sirius_output.csv`, and provide them to run FIDDLE:

```bash
python run_fiddle.py --test_data ./demo/input_msms.mgf \
                    --config_path ./config/fiddle_tcn_orbitrap.yml \
                    --resume_path ./check_point/fiddle_tcn_orbitrap.pt \
                    --fdr_resume_path ./check_point/fiddle_fdr_orbitrap.pt \
                    --buddy_path ./demo/output_buddy.csv \
                    --sirius_path ./demo/output_sirius.csv \
                    --result_path ./demo/output_fiddle_all.csv --device 0
```

## Citation

```
@article{hong2024fiddle,
  title={FIDDLE: a deep learning method for chemical formulas prediction from tandem mass spectra},
  author={Hong, Yuhui and Li, Sujun and Ye, Yuzhen and Tang, Haixu},
  journal={bioRxiv},
  pages={2024--11},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```