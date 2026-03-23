#!/usr/bin/env python3

import os
import argparse
import numpy as np
import random
import copy
from tqdm import tqdm
from pyteomics import mgf
import yaml

from utils import mgf_key_order


def add_noise_to_spectrum(intensity_array, noise_std=0.1):
    """
    Add Gaussian noise to spectrum intensities.

    Args:
        intensity_array: intensity values
        noise_std: standard deviation of noise (default 0.1)

    Returns:
        noised_intensity: intensity values with added noise
    """
    # Create mask for non-zero intensities
    spec_mask = np.where(intensity_array > 0, 1.0, 0.0)

    # Generate Gaussian noise
    noise = np.random.normal(0, noise_std, len(intensity_array))

    # Add noise only to non-zero intensities
    noised_intensity = intensity_array + noise * spec_mask

    # Ensure no negative intensities
    noised_intensity = np.maximum(noised_intensity, 0)

    return noised_intensity


def make_noised_dataset(spectra, noise_times, num_spectra=1000, noise_std=0.1, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    noised_datasets = {}

    for noise_time in noise_times:
        print(f"Generating noised spectra by adding noise {noise_time} time(s)...")

        # Randomly select spectra for this noise time
        if len(spectra) <= num_spectra:
            selected_spectra = spectra.copy()
            print(
                f"  Using all {len(selected_spectra)} available spectra (requested {num_spectra})"
            )
        else:
            selected_spectra = random.sample(spectra, num_spectra)
            print(
                f"  Randomly selected {len(selected_spectra)} spectra from {len(spectra)} available"
            )

        noised_data = []

        for spectrum in tqdm(
            selected_spectra, desc=f"Adding noise {noise_time} time(s)"
        ):
            # Skip empty spectra
            if (
                "intensity array" not in spectrum
                or len(spectrum["intensity array"]) == 0
            ):
                continue

            # Create a deep copy of the spectrum
            noised_spectrum = copy.deepcopy(spectrum)

            # Start with original intensity array
            current_intensity = spectrum["intensity array"].copy()

            # Add noise multiple times (each time separately)
            for i in range(noise_time):
                current_intensity = add_noise_to_spectrum(current_intensity, noise_std)

            # Update the spectrum with noised intensities
            noised_spectrum["intensity array"] = current_intensity

            # Add noise level annotation to parameters
            noised_spectrum["params"]["noised_times"] = str(noise_time)

            # Update title if it exists
            if "title" in noised_spectrum["params"]:
                original_title = noised_spectrum["params"]["title"]
                noised_spectrum["params"][
                    "title"
                ] = f"{original_title}_noise{noise_time}times"

            noised_data.append(noised_spectrum)

        noised_datasets[noise_time] = noised_data
        print(
            f"Generated {len(noised_data)} noised spectra with noise added {noise_time} time(s)"
        )

    return noised_datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate noised spectra")
    parser.add_argument(
        "--mgf_data",
        type=str,
        default="./data/cl_pkl_1007/qtof_test.mgf",
        help="Path to .mgf file",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="./data/noisy/noised_qtof_test.mgf",
        help="Output path for the combined noised spectra MGF file",
    )
    parser.add_argument(
        "--noise_times",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="List of noise times (e.g., 1 2 3 4 5)",
    )
    parser.add_argument(
        "--num_spectra",
        type=int,
        default=1000,
        help="Number of spectra to process for each noise time",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./config/fiddle.yml",
        help="Path to configuration",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    # Load configuration
    config = {}
    if os.path.exists(args.config_path):
        with open(args.config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        print(
            f"Warning: Config file {args.config_path} not found. Using default settings."
        )

    # Get noise standard deviation from config or use default
    noise_std = config.get("noise_std", 0.1)

    # Load MGF data using pyteomics
    print(f"Loading spectra from {args.mgf_data}...")
    data = mgf.read(args.mgf_data)
    data = [d for d in data]  # Convert to list
    print(f"Loaded {len(data)} spectra from {args.mgf_data}")

    # Generate noised datasets for each noise level
    noised_datasets = make_noised_dataset(
        data,
        args.noise_times,
        num_spectra=args.num_spectra,
        noise_std=noise_std,
        seed=42,
    )

    # Combine all noised spectra into a single list
    all_noised_spectra = []
    for noise_time in args.noise_times:
        if noise_time in noised_datasets:
            all_noised_spectra.extend(noised_datasets[noise_time])

    # Save all noised spectra to a single MGF file
    mgf.write(all_noised_spectra, args.out_path, key_order=mgf_key_order, file_mode="w")

    print(f"Saved {len(all_noised_spectra)} total noised spectra to {args.out_path}")

    # Print summary of spectra by noise times
    for noise_time in args.noise_times:
        if noise_time in noised_datasets:
            count = len(noised_datasets[noise_time])
            print(f"  - {count} spectra with noise added {noise_time} time(s)")

    print("Noise generation completed!")
