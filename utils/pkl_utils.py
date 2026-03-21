import numpy as np
import re
from decimal import Decimal
from tqdm import tqdm
import pandas as pd
import random
import copy

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

RDLogger.DisableLog("rdApp.*")

from .mol_utils import (
    monoisotopic_mass_calculator,
    formula_to_dict,
    formula_to_vector,
    dict_to_formula,
    ATOMS_INDEX,
)

# ------------------------ Main Functions ------------------------


def spec2arr(spectra, encoder):
    """data format
    [
        {'title': <str>, 'formula': <numpy array>, 'env': <numpy array>, 'spec': <numpy array>},
        {'title': <str>, 'formula': <numpy array>, 'env': <numpy array>, 'spec': <numpy array>},
        ....
    ]
    """
    data = []
    bad_data = []
    for idx, spectrum in enumerate(tqdm(spectra)):
        data_item = spec2arr_single(spectrum, encoder)
        if data_item != None:
            data.append(data_item)
        else:
            bad_data.append(spectrum["params"]["title"])

    return data, bad_data


def spec2pair(data, bad_title, encoder):
    # Extract df of spectra index and smiles
    df = {"index": [], "smiles": []}

    for idx, data_item in enumerate(tqdm(data)):
        if data_item["title"] in bad_title:
            continue

        df["index"].append(idx)
        df["smiles"].append(data_item["smiles"])
    df = pd.DataFrame(df)

    # Extract positive pairs
    print("Extracting positive and negative pairs...")
    positive_pairs, negative_pairs = generate_pairs(df)

    # Ensure balanced samples
    num_samples = min(len(positive_pairs), len(negative_pairs))
    sampled_positive_pairs = random.sample(positive_pairs, num_samples)
    sampled_negative_pairs = random.sample(negative_pairs, num_samples)

    # Convert pairs to dataframe
    data_pairs = {"idx1": [], "idx2": [], "label": []}
    for idx1, idx2 in sampled_positive_pairs:
        data_pairs["idx1"].append(idx1)
        data_pairs["idx2"].append(idx2)
        data_pairs["label"].append(1)
    for idx1, idx2 in sampled_negative_pairs:
        data_pairs["idx1"].append(idx1)
        data_pairs["idx2"].append(idx2)
        data_pairs["label"].append(0)

    return data_pairs


# ------------------------ Helper Functions ------------------------


def generate_pairs(df):
    df_group = df.groupby("smiles")
    groups = list(df_group.groups.values())

    positive_pairs = set()
    negative_pairs = set()

    for _, row in tqdm(df.iterrows(), total=len(df)):
        group = df_group.get_group(row["smiles"])

        # Positive pair (from the same group)
        if len(group) > 1:
            positive_pair_candidates = list(set(group.index) - {row.name})
            positive_pair = random.sample(positive_pair_candidates, 1)[0]
            positive_pairs.add(tuple(sorted((row.name, positive_pair))))

        # Negative pair (from a different group)
        other_groups = [g for g in groups if row.name not in g]
        if other_groups:
            random_group = random.choice(other_groups)
            negative_pair = random.choice(random_group)
            negative_pairs.add(tuple(sorted((row.name, negative_pair))))

    return positive_pairs, negative_pairs


def spec2arr_single(spectrum, encoder):
    # convert molecular object to formula string
    mol = Chem.MolFromSmiles(spectrum["params"]["smiles"])
    mol = Chem.AddHs(mol)
    try:
        formula = CalcMolFormula(mol)  # formula string
    except:
        print(
            "Failed to convert smiles to formula: {}".format(
                spectrum["params"]["smiles"]
            )
        )
        return None

    charge = int(encoder["type2charge"][spectrum["params"]["precursor_type"]])

    # spec array
    good_spec, mz_array, int_array, spec_arr = generate_ms(
        x=spectrum["m/z array"],
        y=spectrum["intensity array"],
        precursor_mz=float(spectrum["params"]["theoretical_precursor_mz"]),
        resolution=encoder["resolution"],
        max_mz=encoder["max_mz"],
        charge=charge,
    )
    if not good_spec:
        print("Failed to generate mass spectra: {}".format(spectrum["params"]["title"]))
        return None  # invalid spectra

    # env array
    adjust_formula, adjust_precursor_type = melt_neutral_precursor(
        formula, spectrum["params"]["precursor_type"]
    )
    ce, nce = parse_collision_energy(
        ce_str=spectrum["params"]["collision_energy"],
        precursor_mz=float(spectrum["params"]["theoretical_precursor_mz"]),
        charge=abs(charge),
    )
    if ce == None and nce == None:
        print(
            'Failed to parse collision energy: "{}"'.format(
                spectrum["params"]["collision_energy"]
            )
        )
        return None  # can not process '30-50 eV'
    env_arr = np.array(
        [
            float(spectrum["params"]["theoretical_precursor_mz"]),
            nce,
            encoder["precursor_type"][adjust_precursor_type],
        ]
    )

    # mass
    # iso_mass = monoisotopic_mass_calculator(mol, mode='mol')
    iso_mass = monoisotopic_mass_calculator(adjust_formula, mode="f")

    # formula array
    formula_vec = formula_to_vector(adjust_formula)
    f_arr = np.array(formula_vec)

    return {
        "title": spectrum["params"]["title"],
        "smiles": spectrum["params"]["smiles"],
        "formula": f_arr.astype(np.float32),
        "mass": iso_mass,
        "charge": charge,
        "spec": spec_arr.astype(np.float32),
        "env": env_arr.astype(np.float32),
        "mz": mz_array,
        "intensity": int_array,
    }


def generate_ms(x, y, precursor_mz, resolution=1, max_mz=1500, charge=1):
    """
    Input:  x   [float list denotes the x-coordinate of peaks]
            y   [float list denotes the y-coordinate of peaks]
            precursor_mz    [float denotes the parent ion]
            resolution      [float denotes the resolution of spectra]
            max_mz          [integer denotes the maximum m/z value of spectra]
            charge          [float denotes the charge of spectra]
    Return: ms  [numpy array denotes the mass spectra]
    """
    assert len(x) == len(y), "The length of x and y should be the same."

    x, y = remove_precursor_isotopic_peaks(
        x, y, precursor_mz, charge, tolerance=resolution / 10
    )
    x = np.array(x)
    y = np.array(y)
    if len(x) == 0 or y.max() - y.min() == 0:
        return False, None, None, None

    # (1) Smooth the MS/MS (no benefit)
    # y = np.sqrt(y)
    # (2) Sharpen the MS/MS
    # y = y ** 2

    # Prepare parameters using Decimal for precision
    resolution = Decimal(str(resolution))
    max_mz = Decimal(str(max_mz))

    # Initialize mass spectra vector with zeros
    intensity_val = np.zeros(int(max_mz // resolution))
    mz_val = np.zeros(
        int(max_mz // resolution)
    )  # Save the max delta m/z falling this bin

    # Convert x, y to indices and accumulate in the mass spectra vector
    for mz, intensity in zip(x, y):
        idx = int(round(Decimal(str(mz)) // resolution))

        intensity_val[idx] += intensity  # Accumulate the intensity in the bin
        if intensity_val[idx] < intensity:  # Update the max delta m/z falling this bin
            mz_val[idx] = mz

    # Normalize to 0-1 range
    intensity_val = (intensity_val - intensity_val.min()) / (
        intensity_val.max() - intensity_val.min()
    )

    # Stack the arrays horizontally
    stacked_ms = np.column_stack((intensity_val, mz_val))

    return True, x, y, stacked_ms


def remove_precursor_isotopic_peaks(x, y, precursor_mz, charge, tolerance=0.1):
    """Remove precursor and its isotopic peaks from an MS/MS spectrum.

    Args:
        x: m/z array of fragment peaks.
        y: Intensity array of fragment peaks.
        precursor_mz: Precursor ion m/z.
        charge: Precursor charge state (determines isotope spacing = 1/|charge|).
        tolerance: m/z window (Da) to match a peak to a theoretical isotope.

    Returns:
        tuple[list, list]: Filtered (x, y) with isotopic peaks removed.
    """
    # Calculate the spacing between isotopic peaks
    spacing = 1 / abs(charge)

    # Generate theoretical isotopic peaks
    max_isotopes = (
        5  # Typically, we consider up to 5 isotopic peaks for large molecules
    )
    theoretical_isotopic_peaks = [
        precursor_mz - i * spacing for i in range(max_isotopes)
    ]

    # Identify isotopic peaks
    isotopic_peaks = []
    for value in x:
        for theoretical_peak in theoretical_isotopic_peaks:
            if abs(value - theoretical_peak) <= tolerance:
                isotopic_peaks.append(value)
                break

    # Remove isotopic peaks from x and y
    filtered_x = []
    filtered_y = []
    for xi, yi in zip(x, y):
        if xi not in isotopic_peaks:
            filtered_x.append(xi)
            filtered_y.append(yi)

    return filtered_x, filtered_y


def parse_collision_energy(ce_str, precursor_mz, charge=1):
    """Parse a free-text collision energy string into (ce, nce) values.

    Handles a wide range of formats from NIST, MassBank, and CASMI.
    If only one of ce/nce can be determined, the other is derived via the
    charge-factor formula. Returns (None, None) if the string is unparseable.

    Args:
        ce_str: Raw collision energy string from spectrum metadata.
        precursor_mz: Precursor m/z, used for ce↔nce conversion.
        charge: Precursor charge state (default 1).

    Returns:
        tuple[float | None, float | None]: (ce in eV, nce dimensionless).
    """
    # ratio constants for NCE
    # charge = int(charge.lstrip('+').lstrip('-'))
    charge_factor = {1: 1, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.75, 6: 0.75, 7: 0.75, 8: 0.75}

    ce = None
    nce = None

    # match collision energy (eV)
    matches_ev = {
        # NIST20
        r"^[\d]+[.]?[\d]*$": lambda x: float(x),
        r"^[\d]+[.]?[\d]*[ ]?eV$": lambda x: float(x.rstrip(" eV")),
        r"^[\d]+[.]?[\d]*[ ]?ev$": lambda x: float(x.rstrip(" ev")),
        r"^[\d]+[.]?[\d]*[ ]?v$": lambda x: float(x.rstrip(" v")),
        r"^[\d]+[.]?[\d]*[ ]?V$": lambda x: float(x.rstrip(" V")),
        r"^NCE=[\d]+[.]?[\d]*% [\d]+[.]?[\d]*eV$": lambda x: float(
            x.split()[1].rstrip("eV")
        ),
        r"^nce=[\d]+[.]?[\d]*% [\d]+[.]?[\d]*ev$": lambda x: float(
            x.split()[1].rstrip("ev")
        ),
        # MassBank
        r"^[\d]+[.]?[\d]*[ ]?V$": lambda x: float(x.rstrip(" V")),
        r"^hcd[\d]+[.]?[\d]*$": lambda x: float(x.lstrip("hcd")),
        r"^[\d]+HCD$": lambda x: float(x.rstrip("HCD")),  # 35HCD
    }
    for k, v in matches_ev.items():
        if re.match(k, ce_str):
            ce = v(ce_str)
            break
    # match collision energy (NCE)
    matches_nce = {
        # MassBank
        r"^[\d]+[.]?[\d]*[ ]?[%]? \(nominal\)$": lambda x: float(
            x.rstrip("% (nominal)")
        ),
        r"^[\d]+[.]?[\d]*[ ]?nce$": lambda x: float(x.rstrip(" nce")),
        r"^[\d]+[.]?[\d]*[ ]?\(nce\)$": lambda x: float(x.rstrip(" (nce)")),
        r"^NCE=[\d]+\%$": lambda x: float(x.lstrip("NCE=").rstrip("%")),
        r"^[\d]+[.]?[\d]*\([Nn][Cc][Ee]\)$": lambda x: float(
            x.split("(")[0]
        ),  # 90(NCE)
        r"^HCD \(NCE [\d]+[.]?[\d]*%\)$": lambda x: float(
            x.split(" ")[-1].rstrip("%)")
        ),  # HCD (NCE 40%)
        # CASMI
        r"^[\d]+[.]?[\d]*[ ]?\(nominal\)$": lambda x: float(
            x.rstrip("(nominal)").rstrip(" ")
        ),
    }
    for k, v in matches_nce.items():
        if re.match(k, ce_str):
            nce = v(ce_str) * 0.01
            break

    # unknown collision energy
    if ce_str == "Unknown":
        ce = 40

    if nce == None and ce != None:
        nce = ce * 500 * charge_factor[charge] / precursor_mz
    elif ce == None and nce != None:
        ce = nce * precursor_mz / (500 * charge_factor[charge])
    else:
        # raise Exception('Collision energy parse error: {}'.format(ce_str))
        return None, None
    return ce, nce


def melt_neutral_precursor(formula, precursor_type):
    """Fold neutral losses/adducts from the precursor type into the molecular formula.

    For example, '[M+H-H2O]+' removes H2O from the formula and returns '[M+H]+'.
    This normalizes the formula so it reflects the neutral molecule after accounting
    for common in-source fragmentations.

    Args:
        formula: Molecular formula string of the neutral compound.
        precursor_type: Adduct/loss string, e.g. '[M+H-H2O]+'.

    Returns:
        tuple[str, str]: (adjusted_formula, simplified_precursor_type).
    """
    precursor_type = unify_precursor_type(precursor_type)

    neutrue_list = ["H2O", "NH3", "CO2", "CH4O2", "CH2O2"]
    neutrue_counts = {}

    # Step 1: Get the count of neutral losses/adducts
    for neutrue in neutrue_list:
        if neutrue in precursor_type:
            pattern = r"([-+]?\d*)\b" + neutrue + r"\b"
            matches = re.findall(pattern, precursor_type)
            if matches:
                count = matches[0]
                if count == "+":
                    count = 1
                elif count == "-":
                    count = -1
                else:
                    try:
                        count = int(count)
                    except Exception as e:
                        print("Get the count of neutral losses/adducts", e)
                        print(precursor_type, neutrue, matches, count)
                neutrue_counts[neutrue] = count
                break  # only one neutral adduct is allowed

    # Step 2: Remove neutral losses/adducts from the precursor type string
    pattern = r"([-+]?\d*)(?:" + "|".join(neutrue_list) + ")" + r"\b"
    adjust_precursor_type = re.sub(pattern, "", precursor_type)

    # Step 3: Add neutral losses/adducts to formula
    f_dict = formula_to_dict(formula)
    for neutrue, count in neutrue_counts.items():
        n_dict = formula_to_dict(neutrue)
        for k, v in n_dict.items():
            if k in f_dict.keys():
                try:
                    f_dict[k] += int(v) * count
                except Exception as e:
                    print("Add neutral losses/adducts to formula", e)
                    print(f_dict, int(v) * count)
            else:
                f_dict[k] = int(v) * count
    adjust_formula = dict_to_formula(f_dict)

    return adjust_formula, adjust_precursor_type


def unify_precursor_type(precursor_type):
    # +1+-neutrue
    # '[M+H-H2O]+', '[M-H2O+H]+', '[M+H-2H2O]+', '[M+H-NH3]+', '[M+H+NH3]+', '[M+NH4]+', '[M+H-CH2O2]+', '[M+H-CH4O2]+'
    # -1+-neutrue
    # '[M-H-CO2]-', '[M-CHO2]-', '[M-H-H2O]-'
    if precursor_type == "[M+NH4]+":
        return "[M+H+NH3]+"
    elif precursor_type == "[M-CHO2]-":
        return "[M-H-CO2]-"
    elif precursor_type == "[M-H2O+H]+":
        return "[M+H-H2O]+"
    else:
        return precursor_type


if __name__ == "__main__":
    # melt_neutral_precursor
    formula = "C12H14O2"
    precursor_type = "[M+H-CH2O2]+"
    adjust_formula, adjust_precursor_type = melt_neutral_precursor(
        formula, precursor_type
    )
    print(adjust_formula, adjust_precursor_type)
    print(formula_to_vector(adjust_formula))

    # generate_ms
    x = [128.0511, 128.0553, 129.0544, 129.0585, 130.0665, 130.0725]
    y = [102.5, 2.3, 10.99, 1.0, 999.0, 22.68]
    precursor_mz = 130.06564948176495
    print("origin mz", x)
    print("origin intensity", y)
    resolution = 0.2
    max_mz = 1500
    charge = 1
    good_ms, mz, intensity, ms = generate_ms(
        x, y, precursor_mz, resolution, max_mz, charge
    )
    print(good_ms, ms)
    print("mz", mz)
    print("intensity", intensity)

    x = [1447.98, 279.24, 695.48, 1446.97, 415.23, 29.69, 1446.89]
    y = [1.0, 0.02231055, 0.02118375, 0.01257913, 0.01197476, 0.00163897, 0.0]
    precursor_mz = 1448.97222710619
    print("origin mz", x)
    print("origin intensity", y)
    resolution = 1.0
    max_mz = 1500
    charge = -1
    good_ms, mz, intensity, ms = generate_ms(
        x, y, precursor_mz, resolution, max_mz, charge
    )
    print(good_ms, ms)
    print("mz", mz)
    print("intensity", intensity)
