"""Remove the compounds of a model's training data from benchmark MGF files.

Writes <name>_unseen.mgf next to each input MGF, holding the spectra whose
compound is not in the training data, so benchmarks can be reported on all
compounds and on unseen ones. Compounds match on the first block of the
InChIKey (the structure without stereochemistry), so a stereoisomer of a
training compound counts as seen.

The split can be saved to a manifest (--manifest_out), a CSV with a row per
benchmark spectrum: benchmark, title, InChIKey block and whether it is unseen.
With the manifest (--manifest) instead of the training data, the same split
is made anywhere; spectra that differ from the manifest's (a re-exported
benchmark) are reported, and spectra of compounds it does not list are left
out of the unseen set.

Usage:
    # where the models were trained: split and save the manifest
    python scripts/remove_training_compounds.py \\
        --train_data ./data/cl_pkl_112026/qtof_maxmin_train.mgf \\
                     ./data/cl_pkl_112026/qtof_maxmin_val.mgf \\
        --mgf ./data/casmi2017.mgf --manifest_out ./benchmarks/unseen_v2.4.0.csv
    # anywhere: split by the manifest
    python scripts/remove_training_compounds.py \\
        --manifest ./benchmarks/unseen_v2.4.0.csv --mgf ./data/casmi2017.mgf
"""

import argparse
import csv
import os
import sys

from pyteomics import mgf
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# Make the repo root importable so `msfiddle` resolves without an editable install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from msfiddle.utils import mgf_key_order

MANIFEST_FIELDS = ["benchmark", "title", "inchikey_block", "unseen"]


def compound_key(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(mol)[:14] if mol is not None else None


def training_smiles(path):
    """SMILES of a training MGF; only the SMILES lines are read, since parsing
    every peak list would be far slower."""
    with open(path) as f:
        return {line[7:].strip() for line in f if line.startswith("SMILES=")}


def benchmark_name(path):
    return os.path.basename(path)[: -len(".mgf")]


def read_manifest(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(path, rows, benchmarks):
    """Replace the rows of these benchmarks in the manifest, keeping the others."""
    kept = [r for r in read_manifest(path) if r["benchmark"] not in benchmarks] if os.path.exists(path) else []
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(sorted(kept + rows, key=lambda r: (r["benchmark"], r["title"])))
    os.replace(tmp, path)


def manifest_key(spectrum):
    return compound_key(spectrum["params"]["smiles"]) or ""


def split_by_manifest(name, spectra, manifest):
    """Unseen spectra of a benchmark by its manifest rows, reporting differences."""
    rows = [r for r in manifest if r["benchmark"] == name]
    if not rows:
        sys.exit(f"The manifest has no rows for {name}")
    unseen_keys = {r["inchikey_block"] for r in rows if r["unseen"] == "1"}
    seen_keys = {r["inchikey_block"] for r in rows if r["unseen"] == "0"}
    titles = {r["title"] for r in rows}
    found = {s["params"]["title"] for s in spectra}
    if found != titles:
        print(
            f"Warning: {name} differs from the manifest: {len(titles - found)} of its "
            f"{len(titles)} spectra are missing and {len(found - titles)} are new"
        )
    unknown = {k for s in spectra if (k := manifest_key(s)) not in unseen_keys | seen_keys}
    if unknown:
        print(f"Warning: {len(unknown)} compounds of {name} are not in the manifest; left out of the unseen set")
    return [s for s in spectra if manifest_key(s) in unseen_keys]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--train_data",
        nargs="+",
        help="Training (and validation) MGF files of the model, as prepare_msms.py "
        "writes them",
    )
    source.add_argument("--manifest", help="A manifest written with --manifest_out")
    parser.add_argument("--mgf", nargs="+", required=True, help="Benchmark MGF files")
    parser.add_argument(
        "--manifest_out", help="Save the split to this manifest (with --train_data)"
    )
    args = parser.parse_args()
    if args.manifest_out and not args.train_data:
        parser.error("--manifest_out needs --train_data")

    if args.train_data:
        seen = set()
        for path in args.train_data:
            seen |= {compound_key(s) for s in training_smiles(path)}
            print(f"Read the training compounds of {path}")
        seen.discard(None)
    else:
        manifest = read_manifest(args.manifest)

    rows = []
    for path in args.mgf:
        name = benchmark_name(path)
        spectra = list(mgf.read(path, use_index=False))
        if args.train_data:
            unseen = [s for s in spectra if compound_key(s["params"]["smiles"]) not in seen]
            unseen_titles = {s["params"]["title"] for s in unseen}
            rows += [
                {
                    "benchmark": name,
                    "title": s["params"]["title"],
                    "inchikey_block": manifest_key(s),
                    "unseen": "1" if s["params"]["title"] in unseen_titles else "0",
                }
                for s in spectra
            ]
        else:
            unseen = split_by_manifest(name, spectra, manifest)
        out_path = path[: -len(".mgf")] + "_unseen.mgf"
        mgf.write(unseen, out_path, key_order=mgf_key_order, file_mode="w")
        n_compounds = len({s["params"]["smiles"] for s in spectra})
        n_unseen = len({s["params"]["smiles"] for s in unseen})
        print(
            f"{path}: {n_unseen} of {n_compounds} compounds unseen "
            f"({len(unseen)} of {len(spectra)} spectra) -> {out_path}"
        )

    if args.manifest_out:
        write_manifest(args.manifest_out, rows, {benchmark_name(p) for p in args.mgf})
        print(f"Saved the split to {args.manifest_out}")
