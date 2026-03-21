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
        "--pkl_dir", type=str, default="./data/", help="Path to pkl data"
    )
    parser.add_argument(
        "--test_title_list", type=str, default="", help="Path to test title list"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        required=True,
        choices=["agilent", "nist20", "nist23", "mona", "waters", "gnps"],
        help="Dataset name",
    )
    parser.add_argument(
        "--instrument_type",
        type=str,
        nargs="+",
        required=True,
        choices=["qtof", "orbitrap"],
        help="Dataset name",
    )  # J0siee: remove later
    parser.add_argument(
        "--train_ratio", type=float, default=0.9, help="Ratio for train set"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./config/fiddle.yml",
        help="Path to configuration",
    )
    args = parser.parse_args()

    assert args.train_ratio < 1.0

    os.makedirs(args.pkl_dir, exist_ok=True)

    if "agilent" in args.dataset:
        assert os.path.exists(os.path.join(args.raw_dir, "Agilent_Combined.sdf"))
        assert os.path.exists(os.path.join(args.raw_dir, "Agilent_Metlin.sdf"))
    if "nist20" in args.dataset:
        assert os.path.exists(os.path.join(args.raw_dir, "hr_msms_nist.SDF"))
    if "nist23" in args.dataset:
        assert os.path.exists(os.path.join(args.raw_dir, "exported_hr_msms.mgf"))
        assert os.path.exists(os.path.join(args.raw_dir, "exported_hr_msms2.mgf"))
    if "mona" in args.dataset:
        assert os.path.exists(
            os.path.join(args.raw_dir, "MoNA-export-All_LC-MS-MS_QTOF.sdf")
        )
        assert os.path.exists(
            os.path.join(args.raw_dir, "MoNA-export-All_LC-MS-MS_Orbitrap.sdf")
        )
    if "waters" in args.dataset:
        assert os.path.exists(os.path.join(args.raw_dir, "waters_qtof.mgf"))
    if "gnps" in args.dataset:
        assert os.path.exists(os.path.join(args.raw_dir, "ALL_GNPS_cleaned.mgf"))
        assert os.path.exists(os.path.join(args.raw_dir, "ALL_GNPS_cleaned.csv"))

    # load the configurations
    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # configuration check (later)

    # 1. convert original format to mgf
    print("\n>>> Step 1: convert original format to mgf;")
    origin_spectra = {}
    if "agilent" in args.dataset:
        spectra1 = sdf2mgf(
            path=os.path.join(args.raw_dir, "Agilent_Combined.sdf"),
            prefix="agilent_combine",
        )
        spectra2 = sdf2mgf(
            path=os.path.join(args.raw_dir, "Agilent_Metlin.sdf"),
            prefix="agilent_metlin",
        )
        origin_spectra["agilent"] = spectra1 + spectra2
    if "nist20" in args.dataset:
        origin_spectra["nist20"] = sdf2mgf(
            path=os.path.join(args.raw_dir, "hr_msms_nist.SDF"), prefix="nist20"
        )
    if "nist23" in args.dataset:
        spectra1 = mgf.read(os.path.join(args.raw_dir, "exported_hr_msms.mgf"))
        spectra2 = mgf.read(os.path.join(args.raw_dir, "exported_hr_msms2.mgf"))
        origin_spectra["nist23"] = [spec for spec in spectra1] + [
            spec for spec in spectra2
        ]
        print(
            "Read {} data from {} and {}".format(
                len(origin_spectra["nist23"]),
                os.path.join(args.raw_dir, "exported_hr_msms.mgf"),
                os.path.join(args.raw_dir, "exported_hr_msms2.mgf"),
            )
        )
    if "mona" in args.dataset:
        spectra1 = sdf2mgf(
            path=os.path.join(args.raw_dir, "MoNA-export-All_LC-MS-MS_QTOF.sdf"),
            prefix="mona_qtof",
        )
        spectra2 = sdf2mgf(
            path=os.path.join(args.raw_dir, "MoNA-export-All_LC-MS-MS_Orbitrap.sdf"),
            prefix="mona_orbitrap",
        )
        origin_spectra["mona"] = spectra1 + spectra2
        # origin_spectra['mona'] = sdf2mgf(path=os.path.join(args.raw_dir, 'MoNA-export-All_LC-MS-MS_QTOF.sdf'), prefix='mona_qtof')
    if "waters" in args.dataset:
        origin_spectra["waters"] = mgf.read(
            os.path.join(args.raw_dir, "waters_qtof.mgf")
        )
        print(
            "Read {} data from {}".format(
                len(origin_spectra["waters"]),
                os.path.join(args.raw_dir, "waters_qtof.mgf"),
            )
        )
    if "gnps" in args.dataset:
        raw_spectra = mgf.read(os.path.join(args.raw_dir, "ALL_GNPS_cleaned.mgf"))
        all_metadata = pd.read_csv(os.path.join(args.raw_dir, "ALL_GNPS_cleaned.csv"))
        all_metadata = all_metadata[
            [
                "spectrum_id",
                "Adduct",
                "Precursor_MZ",
                "ExactMass",
                "Ion_Mode",
                "msMassAnalyzer",
                "msDissociationMethod",
                "Smiles",
                "InChIKey_smiles",
                "collision_energy",
            ]
        ]
        all_metadata["collision_energy"] = all_metadata["collision_energy"].fillna(
            "Unknown"
        )
        all_metadata = all_metadata.dropna()
        all_metadata["Adduct"] = all_metadata["Adduct"].apply(lambda x: x[:-2] + x[-1:])
        all_metadata["collision_energy"] = all_metadata["collision_energy"].astype(str)
        all_metadata["Ion_Mode"] = all_metadata["Ion_Mode"].apply(lambda x: x.upper())
        all_metadata = (
            all_metadata.rename(
                columns={
                    "spectrum_id": "title",
                    "Adduct": "precursor_type",
                    "Precursor_MZ": "precursor_mz",
                    "ExactMass": "molmass",
                    "Ion_Mode": "ionmode",
                    "msMassAnalyzer": "source_instrument",
                    "msDissociationMethod": "instrument_type",
                    "Smiles": "smiles",
                    "InChIKey_smiles": "inchi_key",
                }
            )
            .set_index("title")
            .to_dict("index")
        )
        # add meta-data into the spectra
        tmp_spectra = []
        for idx, spec in enumerate(tqdm(raw_spectra)):
            title = spec["params"]["title"]
            if title in all_metadata.keys():
                metadata = all_metadata[title]
                spec["params"] = metadata
                spec["params"]["title"] = "gnps_" + str(idx)
                tmp_spectra.append(spec)
        origin_spectra["gnps"] = tmp_spectra
        print(
            "Read {} data from {}".format(
                len(origin_spectra["gnps"]),
                os.path.join(args.raw_dir, "ALL_GNPS_cleaned.mgf"),
            )
        )

    # 2. filter the spectra
    # 3. split spectra into training and test set according to smiles
    # Note that there is not overlapped molecules between training set and tes set.
    # 4. generate 3d conformattions & encoding data into arrays
    print("\n>>> Step 2: filter out spectra by conditions and unify the SMILES; \n\
\tStep 3: split SMILES into training set and test set; \n\
\t(Molecules that are in nist23 but not in nist20 are put in test set.) \n\
\tStep 4: encode all the data into pkl format;")
    for ins in args.instrument_type:
        spectra = []
        smiles_list = []
        nist23_smiles = []
        for ds in args.dataset:
            config_name = ds + "_" + ins
            if config_name not in config.keys():
                print("Skip {}...".format(config_name))
                continue
            print(
                "({}) Filter {} spectra and Unify the SMILES...".format(
                    ins, config_name
                )
            )
            filter_spectra, filter_smiles_list = filter_spec(
                origin_spectra[ds],
                config[config_name],
                type2charge=config["encoding"]["type2charge"],
            )

            # mgf.write(origin_spectra[ds], './data/mgf_debug/original_{}.mgf'.format(config_name), file_mode="w") # save mgf for debug
            # mgf.write(filter_spectra, './data/mgf_debug/filtered_{}.mgf'.format(config_name), file_mode="w") # save mgf for debug
            filter_smiles_list = list(set(filter_smiles_list))
            spectra += filter_spectra
            smiles_list += filter_smiles_list
            # --------------------------------------------------
            # record the nist23 and nist20 smiles
            # (used for splitting the training and test set)
            # --------------------------------------------------
            if ds == "nist23":
                nist23_smiles = set(filter_smiles_list)
            elif ds == "nist20":
                nist20_smiles = set(filter_smiles_list)
            # --------------------------------------------------
            print(
                "# spectra: {} # compounds: {}".format(
                    len(filter_spectra), len(filter_smiles_list)
                )
            )
            del filter_spectra, filter_smiles_list
        smiles_list = list(set(smiles_list))
        print(
            "Total # spectra: {} # compounds: {}".format(len(spectra), len(smiles_list))
        )

        # split the training and test set
        train_spectra = []
        test_spectra = []
        if args.test_title_list:
            with open(args.test_title_list, "r") as f:
                test_title_list = f.readlines()
                test_title_list = [t.strip() for t in test_title_list]

            for _, spectrum in enumerate(tqdm(spectra)):
                if spectrum["params"]["title"] in test_title_list:
                    test_spectra.append(spectrum)
                else:
                    train_spectra.append(spectrum)
            del spectra, smiles_list, test_title_list
        else:
            unique_nist23_smiles = list(nist23_smiles - nist20_smiles)
            print(
                "({}) Unique nist23 smiles: {}".format(ins, len(unique_nist23_smiles))
            )
            non_nist23_smiles_list = list(set(smiles_list) - set(unique_nist23_smiles))
            print(
                "({}) Total smiles excluding nist23: {}".format(
                    ins, len(non_nist23_smiles_list)
                )
            )

            num_training_smiles = int(len(smiles_list) * args.train_ratio)
            num_test_smiles = len(smiles_list) - num_training_smiles
            # If unique_nist23_smiles can fully cover the test set, take all test compounds from it
            if len(unique_nist23_smiles) >= num_test_smiles:
                test_smiles_list = np.random.choice(
                    unique_nist23_smiles, num_test_smiles, replace=False
                ).tolist()
            # Otherwise, take all of unique_nist23_smiles and fill the gap from non_nist23_smiles_list
            else:
                test_smiles_list = unique_nist23_smiles.copy()
                remaining_test_smiles = num_test_smiles - len(test_smiles_list)
                test_smiles_list += np.random.choice(
                    non_nist23_smiles_list, remaining_test_smiles, replace=False
                ).tolist()
            # The training set is composed of all remaining compounds not in the test set
            train_smiles_list = [
                smiles for smiles in smiles_list if smiles not in test_smiles_list
            ]
            print(
                "({}) Get {} training compounds and {} test compounds".format(
                    ins, len(train_smiles_list), len(test_smiles_list)
                )
            )

            for _, spectrum in enumerate(tqdm(spectra)):
                smiles = spectrum["params"]["smiles"]
                if smiles in train_smiles_list and not spectrum["params"][
                    "title"
                ].startswith("nist23_"):
                    train_spectra.append(spectrum)
                else:
                    test_spectra.append(spectrum)
            del spectra, smiles_list, test_smiles_list
        print(
            "({}) Get {} training spectra and {} test spectra".format(
                ins, len(train_spectra), len(test_spectra)
            )
        )

        # save mgf for debug, SIRIUS, and BUDDY
        out_path = os.path.join(args.pkl_dir, "{}_test.mgf".format(ins))
        mgf.write(test_spectra, out_path, key_order=mgf_key_order, file_mode="w")
        print("Save {} for comparing with SIRIUS and BUDDY".format(out_path))
        out_path = os.path.join(args.pkl_dir, "{}_train.mgf".format(ins))
        mgf.write(train_spectra, out_path, key_order=mgf_key_order, file_mode="w")
        print("Save {} for comparing with SIRIUS and BUDDY".format(out_path))

        # convert the spectra into arrays
        print("({}) Convert spectra and molecules data into arrays...".format(ins))
        # test (do not apply contrastive learning on test set)
        test_data, _ = spec2arr(test_spectra, config["encoding"])
        out_path = os.path.join(args.pkl_dir, "{}_test.pkl".format(ins))
        with open(out_path, "wb") as f:
            pickle.dump(test_data, f)
            print("Save {} data to {}".format(len(test_data), out_path))
        # train
        train_data, bad_title = spec2arr(train_spectra, config["encoding"])
        out_path = os.path.join(args.pkl_dir, "{}_train.pkl".format(ins))
        with open(out_path, "wb") as f:
            pickle.dump(train_data, f)
            print("Save {} data to {}".format(len(train_data), out_path))
        # save the training pairs for contrastive learning
        train_pairs = spec2pair(train_data, bad_title, config["encoding"])
        out_path = os.path.join(args.pkl_dir, "{}_train_pairs.pkl".format(ins))
        with open(out_path, "wb") as f:
            pickle.dump(train_pairs, f)
            print("Save {} data to {}".format(len(train_pairs["idx1"]), out_path))

    print("Done!")
