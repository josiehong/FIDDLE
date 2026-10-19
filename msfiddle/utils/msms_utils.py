import numpy as np
from tqdm import tqdm

from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

RDLogger.DisableLog("rdApp.*")

from .mol_utils import ATOMS_WEIGHT, monoisotopic_mass_calculator

MSLEVEL_MAP = {
    "MS2": 2,
    "MS1": 1,
    "2": 2,
    "1": 1,
}
# this is the superset of spectra keys in our data
mgf_key_order = [
    "title",
    "precursor_type",
    "precursor_mz",
    "molmass",
    "ms_level",
    "ionmode",
    "source_instrument",
    "instrument_type",
    "collision_energy",
    "smiles",
    "inchi_key",
]


def sdf2mgf(path, prefix):
    """Read an SDF file and convert each record to an MGF-style spectrum dict.

    Args:
        path: Path to the SDF file.
        prefix: String prefix for spectrum titles. Each spectrum is titled
            ``<prefix>_<index>``.

    Returns:
        list[dict]: Each dict has keys ``params`` (metadata), ``m/z array``,
            and ``intensity array``. Records missing required SDF properties
            (MASS SPECTRAL PEAKS, PRECURSOR TYPE, PRECURSOR M/Z, SPECTRUM TYPE,
            COLLISION ENERGY, ION MODE) are skipped.
    """
    supp = Chem.SDMolSupplier(path)
    print("Read {} data from {}".format(len(supp), path))

    spectra = []
    for idx, mol in enumerate(tqdm(supp)):
        if (
            mol == None
            or not mol.HasProp("MASS SPECTRAL PEAKS")
            or not mol.HasProp("PRECURSOR TYPE")
            or not mol.HasProp("PRECURSOR M/Z")
            or not mol.HasProp("SPECTRUM TYPE")
            or not mol.HasProp("COLLISION ENERGY")
            or not mol.HasProp("ION MODE")
        ):
            continue

        mz_array = []
        intensity_array = []
        raw_ms = mol.GetProp("MASS SPECTRAL PEAKS").split("\n")
        for line in raw_ms:
            mz_array.append(float(line.split()[0]))
            intensity_array.append(float(line.split()[1]))
        mz_array = np.array(mz_array)
        intensity_array = np.array(intensity_array)

        inchi_key = (
            "Unknown" if not mol.HasProp("INCHIKEY") else mol.GetProp("INCHIKEY")
        )
        instrument = (
            "Unknown" if not mol.HasProp("INSTRUMENT") else mol.GetProp("INSTRUMENT")
        )
        # These two are also optional in real MoNA exports — default like the
        # fields above instead of letting GetProp raise KeyError mid-run.
        molmass = (
            "Unknown" if not mol.HasProp("EXACT MASS") else mol.GetProp("EXACT MASS")
        )
        instrument_type = (
            "Unknown"
            if not mol.HasProp("INSTRUMENT TYPE")
            else mol.GetProp("INSTRUMENT TYPE")
        )
        spectrum = {
            "params": {
                "title": prefix + "_" + str(idx),
                "precursor_type": mol.GetProp("PRECURSOR TYPE"),
                "precursor_mz": mol.GetProp("PRECURSOR M/Z"),
                "molmass": molmass,
                "ms_level": mol.GetProp("SPECTRUM TYPE"),
                "ionmode": mol.GetProp("ION MODE"),
                "source_instrument": instrument,
                "instrument_type": instrument_type,
                "collision_energy": mol.GetProp("COLLISION ENERGY"),
                "smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
                "inchi_key": inchi_key,
            },
            "m/z array": mz_array,
            "intensity array": intensity_array,
        }
        spectra.append(spectrum)
    return spectra


def _filter_spec_single(spectrum, config, type2charge):
    """Filter/clean one spectrum. Returns (kept_spectrum_or_None, reject_key).

    reject_key is the records category the rejection counts under, or None for
    rejections the original code did not record (bad SMILES, atom count).
    """
    smiles = spectrum["params"]["smiles"]
    # Guard before AddHs: AddHs(None) raises instead of returning None,
    # so one malformed SMILES would abort the whole run.
    mol = Chem.MolFromSmiles(smiles)
    if mol == None:
        return None, None
    mol = Chem.AddHs(mol)

    # Unify the smiles format
    smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    spectrum["params"]["smiles"] = smiles

    # Filter by instrument type
    if "instrument_type" in config.keys():
        instrument_type = spectrum["params"]["instrument_type"]
        if instrument_type not in config["instrument_type"]:
            return None, "instrument_type"

    # Filter by instrument (MoNA contains too many intrument names to filter out)
    if "instrument" in config.keys():
        instrument = spectrum["params"]["source_instrument"]
        if instrument not in config["instrument"]:
            return None, "instrument"

    # Filter by mslevel
    if "ms_level" in config.keys():
        mslevel = spectrum["params"]["ms_level"]
        if mslevel != config["ms_level"]:
            return None, "ms_level"

    # Filter by atom number and atom type
    if (
        len(mol.GetAtoms()) > config["max_atom_num"]
        or len(mol.GetAtoms()) < config["min_atom_num"]
    ):
        return None, None
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in config["atom_type"]:
            return None, "atom_type"

    # Filter by precursor type
    precursor_type = spectrum["params"]["precursor_type"]
    if precursor_type not in config["precursor_type"]:
        return None, "precursor_type"

    # Filt by peak number
    if len(spectrum["m/z array"]) < config["min_peak_num"]:
        return None, "peak_num"

    # Filter by max m/z
    if (
        np.max(spectrum["m/z array"]) < config["min_mz"]
        or np.max(spectrum["m/z array"]) > config["max_mz"]
    ):
        return None, "max_mz"

    # Theoretical precursor m/z and mass error are always computed (they
    # are stamped on every surviving spectrum for downstream encoders);
    # the ppm cut itself only applies when the config sets a tolerance.
    f = CalcMolFormula(mol)
    molmass = monoisotopic_mass_calculator(f, "f")
    theo_mz = precursor_mz_calculator(precursor_type, molmass)
    ppm = abs(theo_mz - float(spectrum["params"]["precursor_mz"])) / theo_mz * 10**6

    # Filter by ppm (mass error)
    if "ppm_tolerance" in config.keys() and ppm > config["ppm_tolerance"]:
        return None, "ppm_tolerance"

    # Remove the invalid spectra
    if spectrum["intensity array"].max() == spectrum["intensity array"].min():
        return None, "invalid_intensity_array"

    spectrum["params"]["theoretical_precursor_mz"] = theo_mz
    spectrum["params"]["ppm"] = ppm
    spectrum["params"]["simulated_precursor_mz"] = (
        simulate_experimental_mz(theo_mz, config["ppm_tolerance"])
        if "ppm_tolerance" in config.keys()
        else theo_mz
    )
    spectrum["params"]["charge"] = type2charge[precursor_type]  # add charge
    return spectrum, None


def filter_spec(spectra, config, type2charge, num_workers=0):
    """Filter and clean a list of spectra according to a configuration dict.

    Applies sequential filters: instrument type/name, MS level, atom count/type,
    precursor type, peak count, m/z range, and ppm mass error. Each spectrum is
    independent, so with num_workers > 1 the work runs in a process pool
    (order-preserving, output identical to serial).

    Args:
            spectra: List of spectra dicts in MGF format.
            config: Dict of filter thresholds (keys: 'instrument_type', 'ms_level',
                    'atom_type', 'precursor_type', 'min_peak_num', 'min_mz', 'max_mz',
                    'ppm_tolerance', etc.).
            type2charge: Dict mapping precursor type string to charge int.
            num_workers: Process count for parallel filtering (0/1 = serial).

    Returns:
            tuple: (clean_spectra, smiles_list) — filtered spectra and their SMILES.
    """
    if num_workers and num_workers > 1:
        from functools import partial
        from multiprocessing import Pool

        with Pool(num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(
                        partial(
                            _filter_spec_single,
                            config=config,
                            type2charge=type2charge,
                        ),
                        spectra,
                        chunksize=256,
                    ),
                    total=len(spectra),
                )
            )
    else:
        results = [_filter_spec_single(s, config, type2charge) for s in tqdm(spectra)]

    clean_spectra = []
    smiles_list = []
    records = {
        "instrument_type": [],
        "instrument": [],
        "ms_level": [],
        "atom_type": [],
        "precursor_type": [],
        "peak_num": [],
        "max_mz": [],
        "invalud_formula": [],
        "ppm_tolerance": [],
        "invalid_intensity_array": [],
    }
    for spectrum, (kept, reject_key) in zip(spectra, results):
        if kept is not None:
            clean_spectra.append(kept)
            smiles_list.append(kept["params"]["smiles"])
        elif reject_key is not None:
            records[reject_key].append(spectrum["params"]["title"])

    print("Filtering records: ")
    for k, v in records.items():
        print("\t{}: {}".format(k, len(v)))

    return clean_spectra, smiles_list


def simulate_experimental_mz(theoretical_mz, relative_mass_tolerance_ppm):
    """Simulate experimental precursor m/z by shifting the theoretical value
    within a Gaussian distribution of mass deviations.

    Args:
        theoretical_mz: Theoretical precursor m/z value.
        relative_mass_tolerance_ppm: Relative mass tolerance in ppm.

    Returns:
        float: Simulated experimental precursor m/z value.
    """

    # Calculate the standard deviation of the Gaussian distribution as 1/3 of the relative mass tolerance
    std_dev = (relative_mass_tolerance_ppm / 1e6 * theoretical_mz) / 3

    # Randomly sample a mass deviation from the Gaussian distribution
    mass_deviation = np.random.normal(0, std_dev)

    # Shift the theoretical m/z value by the sampled mass deviation to simulate the experimental value
    experimental_mz = theoretical_mz + mass_deviation

    return experimental_mz


def precursor_mz_calculator(precursor_type, mass):
    """Compute the expected precursor m/z from neutral monoisotopic mass.

    Args:
            precursor_type: Adduct string, e.g. '[M+H]+'.
            mass: Neutral monoisotopic mass in Da.

    Returns:
            float: Theoretical precursor m/z.

    Raises:
            ValueError: If precursor_type is not supported.
    """
    if precursor_type == "[M+H]+":
        return mass + ATOMS_WEIGHT["H"]
    elif precursor_type == "[M+2H]2+":
        return (mass + 2 * ATOMS_WEIGHT["H"]) / 2
    elif precursor_type == "[M+Na]+":
        return mass + ATOMS_WEIGHT["Na"]
    elif precursor_type == "[M-H]-":
        return mass - ATOMS_WEIGHT["H"]
    elif precursor_type == "[M+H-H2O]+" or precursor_type == "[M-H2O+H]+":
        return mass + ATOMS_WEIGHT["H"] - (ATOMS_WEIGHT["H"] * 2 + ATOMS_WEIGHT["O"])
    elif precursor_type == "[2M+H]+":
        return 2 * mass + ATOMS_WEIGHT["H"]
    elif precursor_type == "[2M-H]-":
        return 2 * mass - ATOMS_WEIGHT["H"]
    elif precursor_type == "[M+H-2H2O]+":
        return (
            mass + ATOMS_WEIGHT["H"] - 2 * (ATOMS_WEIGHT["H"] * 2 + ATOMS_WEIGHT["O"])
        )
    elif precursor_type == "[M+H-NH3]+":
        return mass + ATOMS_WEIGHT["H"] - (ATOMS_WEIGHT["N"] + ATOMS_WEIGHT["H"] * 3)
    elif precursor_type == "[M+H+NH3]+" or precursor_type == "[M+NH4]+":
        return mass + ATOMS_WEIGHT["H"] + (ATOMS_WEIGHT["N"] + ATOMS_WEIGHT["H"] * 3)
    elif precursor_type == "[M+H-CH2O2]+":
        return (
            mass
            + ATOMS_WEIGHT["H"]
            - (ATOMS_WEIGHT["C"] + ATOMS_WEIGHT["H"] * 2 + ATOMS_WEIGHT["O"] * 2)
        )
    elif precursor_type == "[M+H-CH4O2]+":
        return (
            mass
            + ATOMS_WEIGHT["H"]
            - (ATOMS_WEIGHT["C"] + ATOMS_WEIGHT["H"] * 4 + ATOMS_WEIGHT["O"] * 2)
        )
    elif precursor_type == "[M-H-CO2]-" or precursor_type == "[M-CHO2]-":
        return mass - ATOMS_WEIGHT["H"] - (ATOMS_WEIGHT["C"] + ATOMS_WEIGHT["O"] * 2)
    elif precursor_type == "[M-H-H2O]-" or precursor_type == "[M-H2O-H]-":
        return mass - ATOMS_WEIGHT["H"] - (ATOMS_WEIGHT["H"] * 2 + ATOMS_WEIGHT["O"])
    else:
        raise ValueError("Unsupported precursor type: {}".format(precursor_type))


def mass_calculator(precursor_type, precursor_mz):
    """Back-calculate neutral monoisotopic mass from observed precursor m/z.

    The inverse of precursor_mz_calculator.

    Args:
            precursor_type: Adduct string, e.g. '[M+H]+'.
            precursor_mz: Observed precursor m/z.

    Returns:
            float: Neutral monoisotopic mass in Da.

    Raises:
            ValueError: If precursor_type is not supported.
    """
    if precursor_type == "[M+H]+":
        return precursor_mz - ATOMS_WEIGHT["H"]
    elif precursor_type == "[M+2H]2+":
        return precursor_mz * 2 - 2 * ATOMS_WEIGHT["H"]
    elif precursor_type == "[M+Na]+":
        return precursor_mz - ATOMS_WEIGHT["Na"]
    elif precursor_type == "[M-H]-":
        return precursor_mz + ATOMS_WEIGHT["H"]
    elif precursor_type == "[M+H-H2O]+" or precursor_type == "[M-H2O+H]+":
        return (
            precursor_mz
            - ATOMS_WEIGHT["H"]
            + (ATOMS_WEIGHT["H"] * 2 + ATOMS_WEIGHT["O"])
        )
    elif precursor_type == "[2M+H]+":
        return (precursor_mz - ATOMS_WEIGHT["H"]) / 2
    elif precursor_type == "[2M-H]-":
        return (precursor_mz + ATOMS_WEIGHT["H"]) / 2
    elif precursor_type == "[M+H-2H2O]+":
        return (
            precursor_mz
            - ATOMS_WEIGHT["H"]
            + 2 * (ATOMS_WEIGHT["H"] * 2 + ATOMS_WEIGHT["O"])
        )
    elif precursor_type == "[M+H-NH3]+":
        return (
            precursor_mz
            - ATOMS_WEIGHT["H"]
            + (ATOMS_WEIGHT["N"] + ATOMS_WEIGHT["H"] * 3)
        )
    elif precursor_type == "[M+H+NH3]+" or precursor_type == "[M+NH4]+":
        return (
            precursor_mz
            - ATOMS_WEIGHT["H"]
            - (ATOMS_WEIGHT["N"] + ATOMS_WEIGHT["H"] * 3)
        )
    elif precursor_type == "[M+H-CH2O2]+":
        return (
            precursor_mz
            - ATOMS_WEIGHT["H"]
            + (ATOMS_WEIGHT["C"] + ATOMS_WEIGHT["H"] * 2 + ATOMS_WEIGHT["O"] * 2)
        )
    elif precursor_type == "[M+H-CH4O2]+":
        return (
            precursor_mz
            - ATOMS_WEIGHT["H"]
            + (ATOMS_WEIGHT["C"] + ATOMS_WEIGHT["H"] * 4 + ATOMS_WEIGHT["O"] * 2)
        )
    elif precursor_type == "[M-H-CO2]-" or precursor_type == "[M-CHO2]-":
        return (
            precursor_mz
            + ATOMS_WEIGHT["H"]
            + (ATOMS_WEIGHT["C"] + ATOMS_WEIGHT["O"] * 2)
        )
    elif precursor_type == "[M-H-H2O]-" or precursor_type == "[M-H2O-H]-":
        return (
            precursor_mz
            + ATOMS_WEIGHT["H"]
            + (ATOMS_WEIGHT["H"] * 2 + ATOMS_WEIGHT["O"])
        )
    else:
        raise ValueError("Unsupported precursor type: {}".format(precursor_type))


if __name__ == "__main__":
    # test sdf2mgf
    spectra = sdf2mgf("../data/origin/Agilent_Combined.sdf", "agilent_combine")
    print("Number of spectra: {}".format(len(spectra)))
    for spec in spectra:
        if spec["params"]["precursor_type"] == "[M-H]-":
            print(spec)
            break

    # test filter_spec
    config = {
        "instrument_type": "ESI-QTOF",
        "instrument": ["Unknown"],
        "ms_level": "2",
        "atom_type": [
            "C",
            "O",
            "N",
            "H",
            "P",
            "S",
            "F",
            "Cl",
            "B",
            "Br",
            "I",
            "Na",
            "K",
        ],
        "precursor_type": ["[M+H]+", "[M-H]-"],
        "min_mz": 50,
        "max_mz": 1500,
        "min_peak_num": 5,
        "max_atom_num": 300,
        "min_atom_num": 10,
        "ppm_tolerance": 10,
    }
    type2charge = {"[M+H]+": 1, "[M-H]-": -1}
    clean_spectra, smiles_list = filter_spec(spectra, config, type2charge)
    print("Number of clean spectra: {}".format(len(clean_spectra)))
    print("Number of smiles: {}".format(len(smiles_list)))
    for spec in clean_spectra:
        if spec["params"]["precursor_type"] == "[M-H]-":
            print(spec)
            break
