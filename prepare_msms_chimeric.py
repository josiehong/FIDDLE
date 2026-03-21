import os
import argparse
import pickle
import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm
from pyteomics import mgf
import yaml
import copy

from utils import parse_collision_energy, mgf_key_order


def combine_spectra(spec1_mz, spec1_intensity, spec2_mz, spec2_intensity, alpha):
    spec1_intensity = spec1_intensity / np.sum(spec1_intensity)
    spec2_intensity = spec2_intensity / np.sum(spec2_intensity)

    # Concatenate the arrays
    chim_spec_mz = np.concatenate([spec1_mz, spec2_mz])
    chim_spec_intensity = np.concatenate(
        [alpha * spec1_intensity, (1 - alpha) * spec2_intensity]
    )

    # Get the sorted indices of `a`
    sorted_indices = np.argsort(chim_spec_mz)

    # Apply the sorted indices to both arrays
    chim_spec_mz_sorted = chim_spec_mz[sorted_indices]
    chim_spec_intensity_sorted = chim_spec_intensity[sorted_indices]
    return chim_spec_mz_sorted, chim_spec_intensity_sorted


def bin_mz(mz, bin_size=0.1):
    return round(mz / bin_size) * bin_size


def extract_ce(spectrum, encoder):
    charge = int(encoder["type2charge"][spectrum["params"]["precursor_type"]])
    ce, _ = parse_collision_energy(
        ce_str=spectrum["params"]["collision_energy"],
        precursor_mz=float(spectrum["params"]["theoretical_precursor_mz"]),
        charge=abs(charge),
    )
    if ce is None:
        return None
    else:
        return int(ce)


def group_entries(data, encoder, mz_tol=0.5):
    """Group entries by (NCE, precursor_type, binned precursor m/z) for fast lookup."""
    grouped = defaultdict(list)
    for d in data:
        ce = extract_ce(d, encoder)
        precursor_type = d["params"]["precursor_type"]
        mz_bin = bin_mz(float(d["params"]["precursor_mz"]), bin_size=mz_tol / 2)
        key = (ce, precursor_type, mz_bin)
        grouped[key].append(d)
    return grouped


def make_chimeric_dataset(data, alpha, encoder, mz_tol=0.5, num_pairs=1000, seed=42):
    """
    Generate chimeric spectra with matching precursor m/z (±mz_tol), same NCE and precursor type.
    Use formula and env from spectrum A.
    """
    chimeric_data = []
    random.seed(seed)
    np.random.seed(seed)

    grouped = group_entries(data, encoder, mz_tol=mz_tol)
    all_entries = [
        (
            d,
            bin_mz(float(d["params"]["precursor_mz"]), bin_size=mz_tol / 2),
            extract_ce(d, encoder),
            d["params"]["precursor_type"],
        )
        for d in data
    ]

    for _ in tqdm(
        range(num_pairs),
        desc=f"Generating chimeric {int(alpha * 100)}:{int((1-alpha)*100)}",
    ):
        entry_a, mz_bin_a, ce_a, type_a = random.choice(all_entries)

        candidates = []
        key = (ce_a, type_a, mz_bin_a)

        for entry_b in grouped.get(key, []):
            if entry_b["params"]["smiles"] != entry_a["params"]["smiles"]:
                candidates.append(entry_b)

        if not candidates:
            continue

        entry_b = random.choice(candidates)

        chim_spec_mz, chim_spec_intensity = combine_spectra(
            entry_a["m/z array"],
            entry_a["intensity array"],
            entry_b["m/z array"],
            entry_b["intensity array"],
            alpha,
        )

        chim_spec = copy.deepcopy(entry_a)
        chim_spec["m/z array"] = chim_spec_mz
        chim_spec["intensity array"] = (
            chim_spec_intensity * 1000
        )  # Scale intensity to match original
        chim_spec["params"][
            "title"
        ] = f"{entry_a['params']['title']}_{entry_b['params']['title']}_chimera_{int(alpha*100)}"
        chim_spec["params"]["smiles_a"] = entry_a["params"]["smiles"]
        chim_spec["params"]["smiles_b"] = entry_b["params"]["smiles"]
        chimeric_data.append(chim_spec)

    print(
        f"Generated {len(chimeric_data)} chimeric spectra with ratio {int(alpha * 100)}:{int((1 - alpha) * 100)}"
    )
    return chimeric_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate chimeric spectra")
    parser.add_argument(
        "--mgf_data",
        type=str,
        default="./data/cl_pkl_1007/qtof_test.mgf",
        help="Path to .mgf file",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./data/chimeric/",
        help="Directory to save individual chimeric spectra files",
    )
    parser.add_argument(
        "--mix_ratios",
        type=int,
        nargs="+",
        default=[90, 70, 50],
        help="List of mixing ratios (e.g., 90 70 50)",
    )
    parser.add_argument(
        "--num_pairs",
        type=int,
        default=1000,
        help="Number of unique pairs to generate chimeras from",
    )
    parser.add_argument(
        "--mz_tol",
        type=float,
        default=0.01,
        help="Tolerance for precursor m/z matching",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./config/fiddle.yml",
        help="Path to configuration",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        encoder = config["encoding"]

    # load mgf data
    data = mgf.read(args.mgf_data)
    data = [d for d in data]  # convert to list
    print(f"Loaded {len(data)} spectra from {args.mgf_data}")

    # create output directory if it doesn't exist
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # construct chimeric datasets for each mix ratio
    for ratio in args.mix_ratios:
        alpha = ratio / 100.0
        chimeric_data = make_chimeric_dataset(
            data, alpha, encoder, mz_tol=args.mz_tol, num_pairs=args.num_pairs
        )

        out_path = os.path.join(
            args.out_dir,
            "{}_chimeric{}.mgf".format(
                args.mgf_data.split("/")[-1].replace(".mgf", ""), ratio
            ),
        )

        mgf.write(chimeric_data, out_path, key_order=mgf_key_order, file_mode="w")

        print(
            f"Saved {len(chimeric_data)} spectra for ratio {ratio}:{100-ratio} to {out_path}"
        )
