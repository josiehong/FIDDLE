"""Remove the compounds of a model's training data from benchmark MGF files.

Writes <name>_unseen.mgf next to each input MGF, holding the spectra whose
compound is not in the training data, so benchmarks can be reported on all
compounds and on unseen ones. Compounds match on the first block of the
InChIKey (the structure without stereochemistry), so a stereoisomer of a
training compound counts as seen.

Usage:
    python scripts/remove_training_compounds.py \\
        --train_data ./data/cl_pkl_102026/qtof_maxmin_train.pkl \\
                     ./data/cl_pkl_102026/qtof_maxmin_val.pkl \\
        --mgf ./data/casmi2016.mgf ./data/casmi2017.mgf ./data/embl.mgf
"""

import argparse
import os
import pickle
import sys

from pyteomics import mgf
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# Make the repo root importable so `msfiddle` resolves without an editable install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from msfiddle.utils import mgf_key_order


def compound_key(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(mol)[:14] if mol is not None else None


def training_smiles(path):
    """SMILES of a training data file: a pkl from prepare_msms.py or an MGF."""
    if path.endswith(".mgf"):
        return {s["params"]["smiles"] for s in mgf.read(path)}
    with open(path, "rb") as f:
        return {d["smiles"] for d in pickle.load(f)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--train_data",
        nargs="+",
        required=True,
        help="Training (and validation) data of the model, .pkl or .mgf",
    )
    parser.add_argument("--mgf", nargs="+", required=True, help="Benchmark MGF files")
    args = parser.parse_args()

    seen = set()
    for path in args.train_data:
        seen |= {compound_key(s) for s in training_smiles(path)}
        print(f"Read the training compounds of {path}")
    seen.discard(None)

    for path in args.mgf:
        spectra = list(mgf.read(path))
        unseen = [s for s in spectra if compound_key(s["params"]["smiles"]) not in seen]
        out_path = path[: -len(".mgf")] + "_unseen.mgf"
        mgf.write(unseen, out_path, key_order=mgf_key_order, file_mode="w")
        n_compounds = len({s["params"]["smiles"] for s in spectra})
        n_unseen = len({s["params"]["smiles"] for s in unseen})
        print(
            f"{path}: {n_unseen} of {n_compounds} compounds unseen "
            f"({len(unseen)} of {len(spectra)} spectra) -> {out_path}"
        )
