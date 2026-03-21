"""
Dataset for formula prediction (remove some functions to ./utils/ later)
- Positive and negative pairs for training
- Spectra in pkl format for both training and test set
- Modified to extract MoNA-QTOF and sample NIST20-Orbitrap data
"""

import os
import re
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import pickle
from pyteomics import mgf
import pandas as pd
import random

from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator

# ignore the warning
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

from utils import sdf2mgf, filter_spec, mgf_key_order, spec2arr, spec2pair
from utils import mgf_key_order

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess the data for formula prediction"
    )
    parser.add_argument(
        "--raw_dir", type=str, default="./data/origin/", help="Path to raw data"
    )
    parser.add_argument(
        "--pkl_dir", type=str, default="./data/ablation/", help="Path to pkl data"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio for train set (default: 0.8)",
    )
    parser.add_argument(
        "--config_qtof_path",
        type=str,
        default="./config/fiddle_tcn_qtof.yml",
        help="Path to QTOF configuration",
    )
    parser.add_argument(
        "--config_orbitrap_path",
        type=str,
        default="./config/fiddle_tcn_orbitrap.yml",
        help="Path to Orbitrap configuration",
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    assert args.train_ratio < 1.0

    # Set random seeds for reproducibility
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    os.makedirs(args.pkl_dir, exist_ok=True)

    # Check required files exist
    assert os.path.exists(
        os.path.join(args.raw_dir, "MoNA-export-All_LC-MS-MS_QTOF.sdf")
    ), "MoNA QTOF file not found"
    assert os.path.exists(
        os.path.join(args.raw_dir, "hr_msms_nist.SDF")
    ), "NIST20 file not found"

    # load the configurations
    with open(args.config_qtof_path, "r") as f:
        config_qtof = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.config_orbitrap_path, "r") as f:
        config_orbitrap = yaml.load(f, Loader=yaml.FullLoader)

    # 1. Load and convert original format to mgf
    print("\n>>> Step 1: Load and convert original format to mgf")

    # Load MoNA-QTOF data
    print("Loading MoNA-QTOF data...")
    mona_qtof_spectra = sdf2mgf(
        path=os.path.join(args.raw_dir, "MoNA-export-All_LC-MS-MS_QTOF.sdf"),
        prefix="mona_qtof",
    )
    print(f"Loaded {len(mona_qtof_spectra)} MoNA-QTOF spectra")

    # Load NIST20 data
    print("Loading NIST20 data...")
    nist20_spectra = sdf2mgf(
        path=os.path.join(args.raw_dir, "hr_msms_nist.SDF"), prefix="nist20"
    )
    print(f"Loaded {len(nist20_spectra)} NIST20 spectra")

    # Randomly sample 20000 from NIST20-Orbitrap
    nist20_sample_size = 20000
    if len(nist20_spectra) > nist20_sample_size:
        print(f"Randomly sampling {nist20_sample_size} spectra from NIST20...")
        sampled_indices = random.sample(range(len(nist20_spectra)), nist20_sample_size)
        nist20_spectra = [nist20_spectra[i] for i in sampled_indices]
        print(f"Sampled {len(nist20_spectra)} NIST20 spectra")
    else:
        print(
            f"NIST20 has fewer than {nist20_sample_size} spectra, using all {len(nist20_spectra)} spectra"
        )

    # Process datasets
    datasets = {
        "mona_qtof": {
            "spectra": mona_qtof_spectra,
            "config_key": "mona_qtof",
            "instrument": "qtof",
            "config": config_qtof,
        },
        "nist20_orbitrap": {
            "spectra": nist20_spectra,
            "config_key": "nist20_orbitrap",
            "instrument": "orbitrap",
            "config": config_orbitrap,
        },
    }

    # 2. Filter spectra and split into train/test for each dataset
    print("\n>>> Step 2-4: Filter, split, and encode data")

    for dataset_name, dataset_info in datasets.items():
        print(f"\n--- Processing {dataset_name} ---")

        # Filter spectra
        config_key = dataset_info["config_key"]
        current_config = dataset_info["config"]
        if config_key not in current_config.keys():
            print(f"Configuration for {config_key} not found, skipping...")
            continue

        print(f"Filtering {dataset_name} spectra...")
        filtered_spectra, filtered_smiles_list = filter_spec(
            dataset_info["spectra"],
            current_config[config_key],
            type2charge=current_config["encoding"]["type2charge"],
        )

        filtered_smiles_list = list(set(filtered_smiles_list))
        print(
            f"After filtering: {len(filtered_spectra)} spectra, {len(filtered_smiles_list)} unique compounds"
        )

        if len(filtered_spectra) == 0:
            print(
                f"No spectra remaining after filtering for {dataset_name}, skipping..."
            )
            continue

        # Split by SMILES (molecules)
        print(f"Splitting {dataset_name} by molecules...")
        train_smiles_count = int(len(filtered_smiles_list) * args.train_ratio)
        train_smiles_indices = random.sample(
            range(len(filtered_smiles_list)), train_smiles_count
        )

        train_smiles_set = set([filtered_smiles_list[i] for i in train_smiles_indices])
        test_smiles_set = set(filtered_smiles_list) - train_smiles_set

        print(
            f"Train molecules: {len(train_smiles_set)}, Test molecules: {len(test_smiles_set)}"
        )

        # Split spectra based on SMILES
        train_spectra = []
        test_spectra = []

        for spectrum in tqdm(
            filtered_spectra, desc=f"Splitting {dataset_name} spectra"
        ):
            smiles = spectrum["params"]["smiles"]
            if smiles in train_smiles_set:
                train_spectra.append(spectrum)
            else:
                test_spectra.append(spectrum)

        print(f"Train spectra: {len(train_spectra)}, Test spectra: {len(test_spectra)}")

        # Save MGF files
        instrument = dataset_info["instrument"]
        file_prefix = f"ablation_{instrument}"

        # Save test MGF
        test_mgf_path = os.path.join(args.pkl_dir, f"{file_prefix}_test.mgf")
        mgf.write(test_spectra, test_mgf_path, key_order=mgf_key_order, file_mode="w")
        print(f"Saved test MGF: {test_mgf_path}")

        # Save train MGF
        train_mgf_path = os.path.join(args.pkl_dir, f"{file_prefix}_train.mgf")
        mgf.write(train_spectra, train_mgf_path, key_order=mgf_key_order, file_mode="w")
        print(f"Saved train MGF: {train_mgf_path}")

        # Convert spectra to arrays and save PKL files
        print(f"Converting {dataset_name} spectra to arrays...")

        # Process test data
        test_data, _ = spec2arr(test_spectra, current_config["encoding"])
        test_pkl_path = os.path.join(args.pkl_dir, f"{file_prefix}_test.pkl")
        with open(test_pkl_path, "wb") as f:
            pickle.dump(test_data, f)
        print(f"Saved test PKL: {test_pkl_path} ({len(test_data)} samples)")

        # Process train data
        train_data, bad_title = spec2arr(train_spectra, current_config["encoding"])
        train_pkl_path = os.path.join(args.pkl_dir, f"{file_prefix}_train.pkl")
        with open(train_pkl_path, "wb") as f:
            pickle.dump(train_data, f)
        print(f"Saved train PKL: {train_pkl_path} ({len(train_data)} samples)")

        # Generate training pairs for contrastive learning
        print(f"Generating training pairs for {dataset_name}...")
        train_pairs = spec2pair(train_data, bad_title, current_config["encoding"])
        pairs_pkl_path = os.path.join(args.pkl_dir, f"{file_prefix}_train_pairs.pkl")
        with open(pairs_pkl_path, "wb") as f:
            pickle.dump(train_pairs, f)
        print(
            f'Saved train pairs PKL: {pairs_pkl_path} ({len(train_pairs["idx1"])} pairs)'
        )

        # Clean up memory
        del (
            filtered_spectra,
            train_spectra,
            test_spectra,
            train_data,
            test_data,
            train_pairs,
        )

    print("\n>>> Processing completed!")
    print(f"Files saved to: {args.pkl_dir}")
