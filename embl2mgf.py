import re
import argparse
import yaml
import numpy as np
from pyteomics import mgf
from utils import filter_spec


# Function to parse the input text
def parse_input_text(text):
    data = {}

    # Extract ID (for TITLE)
    id_match = re.search(r">  <ID>\n(.+)", text)
    if id_match:
        data["id"] = id_match.group(1).strip()
    else:
        return None

    # Extract NAME (optional)
    name_match = re.search(r">  <NAME>\n(.+)\n", text)
    if name_match:
        data["name"] = name_match.group(1).strip()

    # Extract FORMULA
    formula_match = re.search(r"formula=([A-Za-z0-9]+)", text)
    if formula_match:
        data["formula"] = formula_match.group(1).strip()
    else:
        return None

    # Extract SMILES
    smiles_match = re.search(r"SMILES=([A-Za-z0-9@+\-\[\]\(\)=#%/\\]+)", text)
    if smiles_match:
        data["smiles"] = smiles_match.group(1).strip()

    # Extract PRECURSOR M/Z
    precursor_mz_match = re.search(r">  <PRECURSOR M/Z>\n([\d.]+)", text)
    if precursor_mz_match:
        data["precursor_mz"] = float(precursor_mz_match.group(1).strip())

    # Extract PRECURSOR TYPE
    precursor_type_match = re.search(r">  <PRECURSOR TYPE>\n(.+)", text)
    if precursor_type_match:
        data["precursor_type"] = precursor_type_match.group(1).strip()
    else:
        return None

    # Extract COLLISION ENERGY
    collision_energy_match = re.search(r">  <COLLISION ENERGY>\n(.+)", text)
    if collision_energy_match:
        collision_energy = collision_energy_match.group(1).strip()
        if "," in collision_energy:
            collision_energy = np.mean(
                [float(ce) for ce in collision_energy.split(",")]
            )  # average collision energy
        data["collision_energy"] = collision_energy

    # Extract M/Z array and intensity array from MASS SPECTRAL PEAKS
    peaks_match = re.search(r">  <MASS SPECTRAL PEAKS>\n([\d.\s]+)", text)
    if peaks_match:
        peaks = peaks_match.group(1).strip().splitlines()
        mz_array = []
        intensity_array = []
        for peak in peaks:
            mz, intensity = map(float, peak.split())
            mz_array.append(mz)
            intensity_array.append(intensity)
        data["mz_array"] = mz_array
        data["intensity_array"] = intensity_array
    else:
        return None

    return data


# Function to generate an MGF block
def generate_mgf_block(data, type2charge):
    # Extract the relevant fields from the input
    assert (
        "id" in data
        and "precursor_type" in data
        and "formula" in data
        and "mz_array" in data
        and "intensity_array" in data
    ), "Some required fields are missing in the input data"

    title = data["id"]
    precursor_mz = data.get("precursor_mz", 0)
    precursor_type = data.get("precursor_type", "")
    collision_energy = data.get("collision_energy", "")
    formula = data.get("formula", "")
    smiles = data.get("smiles", "")
    mz_array = data["mz_array"]
    intensity_array = data["intensity_array"]

    # Use the type2charge mapping to determine the charge
    assert precursor_type in type2charge, "Unknown precursor type: {}".format(
        precursor_type
    )
    charge = type2charge[precursor_type]

    # Build the MGF formatted string
    mgf_block = f"BEGIN IONS\n"
    mgf_block += f"TITLE={title}\n"
    mgf_block += f"PEPMASS={precursor_mz}\n"
    mgf_block += f"CHARGE={charge}\n"
    mgf_block += f"PRECURSOR_TYPE={precursor_type}\n"
    mgf_block += f"PRECURSOR_MZ={precursor_mz}\n"
    mgf_block += f"COLLISION_ENERGY={collision_energy}\n"
    mgf_block += f"SMILES={smiles}\n"
    mgf_block += f"FORMULA={formula}\n"

    # Add peaks with proper formatting (remove trailing zeros)
    for mz, intensity in zip(mz_array, intensity_array):
        mgf_block += f"{mz} {intensity}\n"

    mgf_block += "END IONS\n"
    return mgf_block


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the Data")
    parser.add_argument(
        "--raw_path",
        type=str,
        default="./data/benchmark/embl/MoNA-export-EMBL-MCF_2.0_HRMS_Library.sdf",
        help="path to raw data",
    )
    parser.add_argument(
        "--mgf_path",
        type=str,
        default="./data/embl_mcf_2.0.mgf",
        help="path to mgf data",
    )
    parser.add_argument(
        "--data_config_path",
        type=str,
        default="./config/fiddle_tcn_embl.yml",
        help="path to configuration",
    )
    args = parser.parse_args()

    # load the configurations
    with open(args.data_config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    type2charge = config["encoding"]["type2charge"]

    with open(args.raw_path, "r") as file:
        input_text = file.read()

    mgf_output = ""
    for input_item in input_text.split("$$$$"):
        # Parse the input text
        parsed_item = parse_input_text(input_item)
        if (
            parsed_item is None
            or parsed_item["precursor_type"] not in type2charge.keys()
        ):
            continue
        # Generate MGF block
        mgf_output += generate_mgf_block(parsed_item, type2charge)
        mgf_output += "\n"

    # Save to a .mgf file
    with open(args.mgf_path, "w") as mgf_file:
        mgf_file.write(mgf_output)
    print("MGF file generated successfully.")

    # Filter out invalid spectra
    spectra = mgf.read(args.mgf_path)
    print("Read {} spectra from {}".format(len(spectra), args.mgf_path))
    filtered_spectra, _ = filter_spec(spectra, config["embl"], type2charge)
    print(
        "Filtered out {} invalid spectra".format(len(spectra) - len(filtered_spectra))
    )
    print("Remaining {} spectra".format(len(filtered_spectra)))

    # Save to a new .mgf file
    mgf.write(filtered_spectra, args.mgf_path, file_mode="w")
    print("Filtered MGF file saved successfully.")
