import os
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import pandas as pd
from pyteomics import mgf

from rdkit import Chem

# ignore the warning
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from utils import (
    sdf2mgf,
    filter_spec,
    precursor_mz_calculator,
    mgf_key_order,
    monoisotopic_mass_calculator,
)

global CASMI_STAND_MODE, CASMI_MODE2TYPE

CASMI_STAND_MODE = {"Pos": "POSITIVE", "Neg": "NEGATIVE"}
CASMI_MODE2TYPE = {"POSITIVE": "[M+H]+", "NEGATIVE": "[M-H]-"}


def cal_ppm(theo_mass, precursor_type, real_mz):
    theo_mz = precursor_mz_calculator(precursor_type, float(theo_mass))
    real_mz = float(real_mz)
    return abs(theo_mz - real_mz) / theo_mz * 10**6


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the Data")
    parser.add_argument(
        "--raw_dir", type=str, default="./data/benchmark/", help="path to raw data"
    )
    parser.add_argument(
        "--mgf_dir", type=str, default="./data/", help="path to mgf data"
    )
    parser.add_argument(
        "--data_config_path",
        type=str,
        default="./config/fiddle_tcn_casmi.yml",
        help="path to configuration",
    )
    args = parser.parse_args()

    assert os.path.exists(
        os.path.join(args.raw_dir, "casmi2016", "MoNA-export-CASMI_2016.sdf")
    )

    assert os.path.exists(
        os.path.join(args.raw_dir, "casmi2017", "Chal1to45Summary.csv")
    )
    assert os.path.exists(
        os.path.join(args.raw_dir, "casmi2017", "CASMI-solutions.csv")
    )
    assert os.path.exists(
        os.path.join(args.raw_dir, "casmi2017", "challenges-001-045-msms-mgf-20170908/")
    )
    assert os.path.exists(
        os.path.join(args.raw_dir, "casmi2017", "challenges-001-045-ms-mgf-20170908/")
    )

    # load the configurations
    with open(args.data_config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # 1. convert original format to mgf
    """mgf format
	[{
		'params': {
			'title': prefix_<index>, 
			'precursor_type': <precursor_type (e.g. [M+NH4]+ and [M+H]+)>, 
			'precursor_mz': <precursor m/z>,
			'molmass': <isotopic mass>, 
			'ms_level': <ms_level>, 
			'ionmode': <POSITIVE|NEGATIVE>, 
			'source_instrument': <source_instrument>,
			'instrument_type': <instrument_type>, 
			'collision_energy': <collision energe>, 
			'smiles': <smiles>, 
		},
		'm/z array': mz_array,
		'intensity array': intensity_array
	}, ...]
	"""
    print("\n>>> Step 1: convert original format to mgf;")
    print("Convert CASMI2016...")
    casmi16_spectra = sdf2mgf(
        path=os.path.join(args.raw_dir, "casmi2016", "MoNA-export-CASMI_2016.sdf"),
        prefix="casmi16",
    )

    print("Convert CASMI2017...")
    df_sum1 = pd.read_csv(
        os.path.join(args.raw_dir, "casmi2017", "Chal1to45Summary.csv")
    )
    df_sum1["challengename"] = df_sum1["MS_files"].apply(
        lambda x: x.rstrip("-msms.txt")
    )
    df_sum2 = pd.read_csv(
        os.path.join(
            args.raw_dir,
            "casmi2017",
            "challenges-001-045-msms-mgf-20170908/",
            "summary-001-045.csv",
        ),
        sep="\t",
    )
    df_sol = pd.read_csv(
        os.path.join(args.raw_dir, "casmi2017", "CASMI-solutions.csv"), index_col=0
    )
    df_sol["SMILES"] = df_sol["SMILES"].apply(lambda x: x.rstrip("\t"))
    df_sum = pd.merge(
        df_sum1, df_sum2, left_on="challengename", right_on="challengename"
    )
    df_sum = pd.merge(df_sum, df_sol, left_on="challengename", right_on="Challenge")
    casmi17_spectra = []
    # casmi17_ms_spectra = []
    for idx, row in tqdm(df_sum.iterrows(), total=df_sum.shape[0]):
        # msms data
        msms_file_name = row["MSMS_files"].replace(".txt", ".mgf")
        msms_mgf_file = os.path.join(
            args.raw_dir,
            "casmi2017",
            "challenges-001-045-msms-mgf-20170908",
            msms_file_name,
        )

        ion_mode = CASMI_STAND_MODE[row["ESI Mode"]]
        precursor_type = CASMI_MODE2TYPE[ion_mode]
        smiles = row["SMILES"]
        ce = row["CollisionEnergy(eV)"]
        rt = row["RT_sec"]

        # ms data
        ms_file_name = row["MS_files"].replace(".txt", ".mgf")
        ms_mgf_file = os.path.join(
            args.raw_dir,
            "casmi2017",
            "challenges-001-045-ms-mgf-20170908",
            ms_file_name,
        )

        with mgf.read(ms_mgf_file) as reader:
            spectrum = reader[0]

            # calculate precursor m/z
            max_peak_idx = np.argmax(spectrum["intensity array"])
            precursor_mz = spectrum["m/z array"][max_peak_idx]

        # msms data
        with mgf.read(msms_mgf_file) as reader:
            org_spectrum = reader[0]

            spectrum = {"params": {}}
            spectrum["params"]["title"] = msms_file_name
            spectrum["params"]["rtinseconds"] = rt
            spectrum["params"]["precursor_type"] = precursor_type
            spectrum["params"]["precursor_mz"] = precursor_mz
            spectrum["params"]["molmass"] = monoisotopic_mass_calculator(
                Chem.MolFromSmiles(smiles), mode="mol"
            )
            spectrum["params"]["ms_level"] = "MS2"
            spectrum["params"]["ionmode"] = ion_mode
            spectrum["params"]["source_instrument"] = org_spectrum["params"][
                "source_instrument"
            ]
            spectrum["params"]["instrument_type"] = org_spectrum["params"][
                "source_instrument"
            ]
            spectrum["params"]["collision_energy"] = ce
            spectrum["params"]["smiles"] = smiles
            spectrum["m/z array"] = org_spectrum["m/z array"]
            spectrum["intensity array"] = org_spectrum["intensity array"]
        casmi17_spectra.append(spectrum)

    # write ms data
    # mgf.write(casmi17_ms_spectra, os.path.join(args.mgf_dir, 'casmi2017_ms.mgf'), file_mode="w")

    # 2. filter the spectra
    # 3. randomly split spectra into training and test set according to [smiles]
    # Note that there is not overlapped molecules between training set and tes set.
    print(
        "\n>>> Step 2 & 3: filter out spectra by certain rules; randomly split SMILES into training set and test set;"
    )
    print("Filter CASMI2016 spectra...")
    casmi16_spectra, casmi_smiles_list = filter_spec(
        casmi16_spectra,
        config["casmi16"],
        type2charge=config["encoding"]["type2charge"],
    )
    # calculate ppm
    for idx, spec in enumerate(casmi16_spectra):
        ppm = cal_ppm(
            spec["params"]["molmass"],
            spec["params"]["precursor_type"],
            spec["params"]["precursor_mz"],
        )  # charge 1
        casmi16_spectra[idx]["params"]["ppm"] = ppm
    print("Get {} spectra from CASMI2016".format(len(casmi_smiles_list)))
    mgf.write(
        casmi16_spectra,
        os.path.join(args.mgf_dir, "casmi2016.mgf"),
        key_order=mgf_key_order + ["ppm"],
        file_mode="w",
    )

    print("Filter CASMI2017 spectra...")
    casmi17_spectra, casmi_smiles_list = filter_spec(
        casmi17_spectra,
        config["casmi17"],
        type2charge=config["encoding"]["type2charge"],
    )
    # calculate ppm
    for idx, spec in enumerate(casmi17_spectra):
        ppm = cal_ppm(
            spec["params"]["molmass"],
            spec["params"]["precursor_type"],
            spec["params"]["precursor_mz"],
        )  # charge 1
        casmi17_spectra[idx]["params"]["ppm"] = ppm
    print("Get {} spectra from CASMI2017".format(len(casmi_smiles_list)))
    mgf.write(
        casmi17_spectra,
        os.path.join(args.mgf_dir, "casmi2017.mgf"),
        key_order=mgf_key_order + ["ppm"],
        file_mode="w",
    )

    print("Done!")
