import re
import copy
import pickle

import numpy as np
import pandas as pd
import torch
from pyteomics import mgf
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import (
    ATOMS_INDEX,
    generate_ms,
    parse_collision_energy,
    unify_precursor_type,
    formula_to_dict,
    formula_to_vector,
)


# Used for training and evaluation
class MS2FDataset_CL(Dataset):
    """Dataset for contrastive learning, yielding matched spectrum pairs with labels.

    Loads a preprocessed pkl file of encoded spectra and a corresponding pairs pkl
    (same path with '_train.pkl' replaced by '_train_pairs.pkl') that defines
    positive (same SMILES) and negative (different SMILES) pairs.

    Args:
            path: Path to the training pkl file (e.g. 'orbitrap_maxmin_train.pkl').
            noised_times: Number of noise-augmented copies to append per spectrum.
                          Gaussian noise ~N(0, 0.1) is applied only to non-zero bins.
            padding_dim: Number of zero rows to append to each spectrum array so that
                         the length is divisible by pooling strides.
    """

    def __init__(self, path, noised_times=0, padding_dim=0):
        with open(path, "rb") as file:
            self.data = pickle.load(file)
            print(f"Loaded {len(self.data)} data items from {path}")

        with open(path.replace("_train.pkl", "_train_pairs.pkl"), "rb") as file:
            self.pairs = pickle.load(file)
            self.length = len(self.pairs["idx1"])
            print(
                f'Loaded {self.length} pairs from {path.replace("_train.pkl", "_train_pairs.pkl")}'
            )

        if padding_dim:
            self.padding_for_pooling(padding_dim)
            print(f"Padded ({padding_dim}) zeros for divisibility in pooling layers")

        if noised_times:
            self.data, self.pairs, self.length = self.augment_with_noise(
                self.data, self.pairs, noised_times
            )
            print(f"Data augmented with noise ~N(0, 0.1) {noised_times} times")

    def padding_for_pooling(self, padding_dim):
        for data_item in self.data:
            spec = data_item["spec"]
            data_item["spec"] = np.concatenate(
                (spec, np.zeros((padding_dim, spec.shape[1]))), axis=0
            )

    def augment_with_noise(self, data, pairs, noised_times):
        """Append noise-augmented copies of each spectrum and extend the pairs index.

        For each spectrum, creates noised_times copies by adding Gaussian noise
        ~N(0, 0.1) only to non-zero intensity bins (preserving the zero structure).
        Pair indices are shifted accordingly so augmented copies are paired with
        their corresponding augmented counterparts.

        Args:
                data: List of spectrum data dicts.
                pairs: Dict with keys 'idx1', 'idx2', 'label'.
                noised_times: Number of augmented copies per original spectrum.

        Returns:
                tuple: (augmented_data, augmented_pairs, new_length).
        """
        original_length = len(data)
        augmented_data = copy.deepcopy(data)
        for data_item in tqdm(data, desc="Data Augmentation"):
            spec = data_item["spec"][:, 0]
            spec_mask = np.where(spec > 0, 1.0, 0.0)

            for _ in range(noised_times):
                noise = np.random.normal(0, 0.1, len(spec))
                new_spec = spec + noise * spec_mask

                new_data_item = copy.deepcopy(data_item)
                new_data_item["spec"][:, 0] = new_spec
                augmented_data.append(new_data_item)

        augmented_pairs = copy.deepcopy(pairs)
        for idx1, idx2, label in zip(pairs["idx1"], pairs["idx2"], pairs["label"]):
            for i in range(1, noised_times + 1):
                augmented_pairs["idx1"].append(int(idx1 + original_length * i))
                augmented_pairs["idx2"].append(int(idx2 + original_length * i))
                augmented_pairs["label"].append(label)

        return augmented_data, augmented_pairs, len(augmented_pairs["idx1"])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # print('group_idx', idx, 'idx1', self.pairs['idx1'][idx], 'idx2', self.pairs['idx2'][idx])
        data_item1 = self.data[self.pairs["idx1"][idx]]
        data_item2 = self.data[self.pairs["idx2"][idx]]
        label = self.pairs["label"][idx]

        return (
            data_item1["title"],
            data_item1["formula"],
            data_item1["spec"][:, 0],
            data_item1["mass"],
            data_item1["env"],
            data_item2["title"],
            data_item2["formula"],
            data_item2["spec"][:, 0],
            data_item2["mass"],
            data_item2["env"],
            label,
        )


# Used for training and evaluation
class MS2FDataset(Dataset):
    """Dataset for single-spectrum training and evaluation (non-contrastive).

    Loads a preprocessed pkl file of encoded spectra. Supports optional noise
    augmentation and zero-padding for pooling layer divisibility.

    Args:
            path: Path to the pkl file of encoded spectra.
            noised_times: Number of noise-augmented copies to append per spectrum.
            padding_dim: Number of zero rows to append to each spectrum array.
    """

    def __init__(self, path, noised_times=0, padding_dim=0):
        with open(path, "rb") as file:
            self.data = pickle.load(file)
            print(f"Loaded {len(self.data)} data items from {path}")

        if padding_dim:
            self.padding_for_pooling(padding_dim)
            print(f"Padded ({padding_dim}) zeros for divisibility in pooling layers")

        if noised_times:
            self.data = self.augment_with_noise(self.data, noised_times)
            print(f"Data augmented with noise ~N(0, 0.1) {noised_times} times")

    def padding_for_pooling(self, padding_dim):
        for data_item in self.data:
            spec = data_item["spec"]
            data_item["spec"] = np.concatenate(
                (spec, np.zeros((padding_dim, spec.shape[1]))), axis=0
            )

    def augment_with_noise(self, data, noised_times):
        augmented_data = copy.deepcopy(data)
        for data_item in tqdm(data, desc="Data Augmentation"):
            spec = data_item["spec"][:, 0]
            spec_mask = np.where(spec > 0, 1.0, 0.0)

            for _ in range(noised_times):
                noise = np.random.normal(0, 0.1, len(spec))
                new_spec = spec + noise * spec_mask

                new_data_item = copy.deepcopy(data_item)
                new_data_item["spec"][:, 0] = new_spec
                augmented_data.append(new_data_item)

        return augmented_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        return (
            data_item["title"],
            data_item["formula"],
            data_item["spec"][:, 0],
            data_item["mass"],
            data_item["env"],
        )


# Used for testing
class MGFDataset(Dataset):
    """Dataset for inference from a raw MGF file.

    Reads spectra directly from an MGF file, applies basic validity filters
    (m/z range, minimum peak count, required metadata keys), encodes each
    spectrum into an array, and resolves neutral losses from the precursor type.

    Args:
            path: Path to the input MGF file.
            encoder: Dict with encoding parameters: 'resolution', 'max_mz',
                     'type2charge', 'precursor_type', 'use_simulated_precursor_mz'.
    """

    def __init__(self, path, encoder):
        self.data = []
        self.general_filter_config = {
            "min_mz": 50,
            "max_mz": 1500,
            "min_peak_num": 5,
        }
        self.use_simulated_precursor_mz = encoder["use_simulated_precursor_mz"]
        if self.use_simulated_precursor_mz:
            self.precursor_mz_key = "simulated_precursor_mz"
        else:
            self.precursor_mz_key = "precursor_mz"

        # read data from mgf file
        spectra = mgf.read(path)
        print(len(spectra), "spectra loaded from", path)

        # filter out invalid data (add the other rules if needed)
        spectra, _ = self.filter_spec(
            spectra, self.general_filter_config, type2charge=encoder["type2charge"]
        )
        print(len(spectra), "spectra left after filtering")

        # convert mgf to pkl
        self.load_mgf_spectra(spectra, encoder)
        print(len(self.data), "spectra loaded into the dataset")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.data[idx]["title"],
            self.data[idx]["precursor_type"],
            self.data[idx]["spec"][:, 0],
            self.data[idx]["env"],
            self.data[idx]["neutral_add"],
        )

    def filter_spec(self, spectra, general_filter_config, type2charge):
        clean_spectra = []
        invalid_spectra = []
        for spectrum in spectra:
            if not self.has_all_keys(spectrum):
                print(
                    "MGFError: lacking necessary keys in mgf file, skip this spectrum"
                )
                print(
                    "expected keys in mgf: ('title', '{}', 'precursor_type', 'collision_energy')".format(
                        self.precursor_mz_key
                    )
                )
                continue

            # filter out invalid data
            maxium_mz = np.max(spectrum["m/z array"])
            minium_mz = np.min(spectrum["m/z array"])
            if (
                maxium_mz < general_filter_config["min_mz"]
                or maxium_mz > general_filter_config["max_mz"]
                or len(spectrum["m/z array"]) < general_filter_config["min_peak_num"]
            ):
                invalid_spectra.append(spectrum)
            else:
                clean_spectra.append(spectrum)
        return clean_spectra, invalid_spectra

    def has_all_keys(self, spec):
        if self.use_simulated_precursor_mz:
            keys = [
                "title",
                "simulated_precursor_mz",
                "precursor_type",
                "collision_energy",
            ]
        else:
            keys = ["title", "precursor_mz", "precursor_type", "collision_energy"]
        for k in keys:
            if k not in spec["params"].keys():
                return False
        return True

    def load_mgf_spectra(self, spectra, encoder):
        """Encode a list of filtered spectra and append valid items to self.data.

        For each spectrum: bins the m/z array into a fixed-length array via
        generate_ms, resolves neutral losses from the precursor type, and parses
        collision energy. Spectra that fail any of these steps are silently skipped.

        Args:
                spectra: List of spectra dicts (pyteomics MGF format).
                encoder: Dict with 'resolution', 'max_mz', 'type2charge',
                         'precursor_type', and 'use_simulated_precursor_mz'.
        """
        for spectrum in spectra:
            good_spec, _, _, spec_arr = generate_ms(
                x=spectrum["m/z array"],
                y=spectrum["intensity array"],
                precursor_mz=float(spectrum["params"][self.precursor_mz_key]),
                resolution=encoder["resolution"],
                max_mz=encoder["max_mz"],
                charge=int(
                    encoder["type2charge"][spectrum["params"]["precursor_type"]]
                ),
            )
            if not good_spec:
                continue

            adjust_neutral_add_vec, adjust_precursor_type = self.melt_neutral_precursor(
                spectrum["params"]["precursor_type"]
            )

            ce, nce = parse_collision_energy(
                ce_str=spectrum["params"]["collision_energy"],
                precursor_mz=float(spectrum["params"][self.precursor_mz_key]),
                charge=abs(
                    int(encoder["type2charge"][spectrum["params"]["precursor_type"]])
                ),
            )
            if ce == None and nce == None:  # can not process '30-50 eV'
                continue

            env_arr = np.array(
                [
                    float(spectrum["params"][self.precursor_mz_key]),
                    nce,
                    encoder["precursor_type"][adjust_precursor_type],
                ]
            )

            na_arr = np.array(adjust_neutral_add_vec)

            self.data.append(
                {
                    "title": spectrum["params"]["title"],
                    "precursor_type": spectrum["params"]["precursor_type"],
                    "spec": spec_arr,
                    "env": env_arr,
                    "neutral_add": na_arr,
                }
            )

    def melt_neutral_precursor(self, precursor_type):  # Used for testing only
        precursor_type = unify_precursor_type(precursor_type)

        neutrue_list = ["CH4O2", "CH2O2", "H2O", "NH3", "CO2"]
        neutrue_counts = {}

        # Step 1: Get the count of neutral losses/adducts
        for neutrue in neutrue_list:
            if neutrue in precursor_type:
                pattern = r"([-+]?\d*)" + neutrue
                count = re.findall(pattern, precursor_type)
                if count:
                    count = count[0]
                    if count == "+":
                        count = 1
                    elif count == "-":
                        count = -1
                    else:
                        count = int(count)
                    neutrue_counts[neutrue] = count
                    break  # only one neutral adduct is allowed

        # Step 2: Remove neutral losses/adducts from the precursor type string
        pattern = r"([-+]?\d*)(?:" + "|".join(neutrue_list) + ")"
        adjust_precursor_type = re.sub(pattern, "", precursor_type)

        # Step 3: Convert the neutral losses/adducts into vector
        adjust_neutral_add = {}
        for neutrue, count in neutrue_counts.items():
            n_dict = formula_to_dict(neutrue)
            for k, v in n_dict.items():
                if k in adjust_neutral_add:
                    adjust_neutral_add[k] += v * count
                else:
                    adjust_neutral_add[k] = v * count
        adjust_neutral_add_vec = self.formula_dict_to_vector(adjust_neutral_add)

        return adjust_neutral_add_vec, adjust_precursor_type

    def formula_dict_to_vector(self, formula_dict):
        vector = [0] * len(ATOMS_INDEX)

        for atom, count in formula_dict.items():
            index = ATOMS_INDEX.get(atom, None)
            if index is not None:
                vector[index] = int(count) if count else 1

        return vector



class RescoreDataset(Dataset):
    """Dataset for FDR model training and evaluation.

    Loads a pkl file produced by prepare_rescore.py, where each item contains a
    spectrum array, environment vector, a predicted formula string, and a binary
    label (1 = correct formula, 0 = decoy). The formula string is converted to a
    fixed-length atom-count vector at load time.

    Args:
            path: Path to the FDR pkl file (e.g. 'orbitrap_maxmin_fdr_train.pkl').
    """

    def __init__(self, path):
        with open(path, "rb") as file:
            self.data = pickle.load(file)

        # convert formula string to formula vector
        for d in self.data:
            f_vec = np.array(formula_to_vector(d["pred_formula"]))
            # f_vec = (f_vec - f_vec.min()) / (f_vec.max() - f_vec.min()) # normalize formula vector
            d["f"] = f_vec

        print("Load {} data from {}".format(len(self.data), path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.data[idx]["title"],
            self.data[idx]["spec"][:, 0],
            self.data[idx]["env"],
            self.data[idx]["f"],
            self.data[idx]["label"],
        )
