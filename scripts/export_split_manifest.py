"""Write the timsTOF compound split of a data build: one row per compound
and split, with its InChIKey and source libraries, but no spectra.

Only the timsTOF data (the public CASMI 2026 libraries) is exported; the
Q-TOF and Orbitrap splits include licensed NIST data and stay private.

The TCN and the rescorer train on the same training compounds and select
their checkpoints and seeds on the same validation compounds, so one list
covers both stages.

Usage:
    python scripts/export_split_manifest.py --pkl_dir ./data/cl_pkl_v2.5 \
        --out ./timstof_split_v2.5.0.csv.gz
"""

import argparse
import csv
import gzip
import os
from collections import defaultdict

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def read_compounds(path):
    """SMILES -> [spectrum count, source libraries] of an MGF; only the
    SMILES and SOURCE_INSTRUMENT lines are read, not the peak lists."""
    compounds = defaultdict(lambda: [0, set()])
    smiles = source = None
    with open(path) as f:
        for line in f:
            if line.startswith("BEGIN IONS"):
                smiles = source = None
            elif line.startswith("SMILES="):
                smiles = line[7:].strip()
            elif line.startswith("SOURCE_INSTRUMENT="):
                source = line[18:].strip()
            elif line.startswith("END IONS") and smiles:
                compounds[smiles][0] += 1
                if source:
                    compounds[smiles][1].add(source)
    return compounds


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pkl_dir", required=True)
    parser.add_argument("--split", default="maxmin")
    parser.add_argument("--out", required=True, help=".csv, or .csv.gz to compress")
    args = parser.parse_args()

    opener = gzip.open if args.out.endswith(".gz") else open
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with opener(args.out, "wt", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "inchikey", "n_spectra", "sources"])
        for inst in ["timstof"]:
            for part in ("train", "val", "test"):
                path = os.path.join(args.pkl_dir, f"{inst}_{args.split}_{part}.mgf")
                if not os.path.exists(path):
                    print(f"Skip missing {path}")
                    continue
                # Different SMILES of one compound (e.g. other hydrogen
                # notation) merge under its InChIKey
                rows = defaultdict(lambda: [0, set()])
                for smiles, (n, sources) in read_compounds(path).items():
                    mol = Chem.MolFromSmiles(smiles)
                    key = Chem.MolToInchiKey(mol) if mol is not None else ""
                    rows[key][0] += n
                    rows[key][1] |= sources
                for key in sorted(rows):
                    n, sources = rows[key]
                    writer.writerow([part, key, n, ";".join(sorted(sources))])
                print(f"{inst} {part}: {len(rows)} compounds")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
