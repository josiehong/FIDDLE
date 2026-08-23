#!/usr/bin/env python
"""Top-k formula accuracy, averaged per spectrum and per compound.

Joins a run_fiddle result CSV (one row per spectrum, key column "ID") to a test
set (.pkl preferred, .mgf supported) by ID == title, then reports cumulative
top-1..top-K accuracy two ways:

  - per spectrum (micro): fraction of spectra whose true formula is in the top k.
  - per compound (macro): mean over compounds of the per-compound spectrum
    accuracy, each compound weighted equally. Robust to replicate-rich compounds
    (the per-spectrum number is dominated by heavily-measured molecules).

A formula is "correct at top k" if the true formula's atom-count vector matches
one of the first k refined predictions ("Refined Formula (0..k-1)").

Usage:
  python scripts/eval_topk.py --result ./result/fiddle_qtof_060526.csv \
      --test_data ./data/cl_pkl_060526/qtof_maxmin_test.pkl --topk 5

The test .pkl is preferred: it stores the exact training target (`formula`
vector, already neutral-loss-adjusted), whereas a .mgf derives the formula from
SMILES, which is subtly wrong for neutral-loss adducts (e.g. [M+H-H2O]+).
"""
import argparse
import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from msfiddle.utils.mol_utils import formula_to_vector  # noqa: E402

try:
    from pyteomics import mgf as pyteomics_mgf
except Exception:
    pyteomics_mgf = None


_VEC_CACHE = {}


def formula_vec(formula):
    """Formula string -> tuple atom-count vector (cached); None on empty/failure."""
    if not isinstance(formula, str):
        return None
    f = formula.strip()
    if f == "" or f.lower() == "nan":
        return None
    if f not in _VEC_CACHE:
        try:
            _VEC_CACHE[f] = tuple(int(round(x)) for x in formula_to_vector(f))
        except Exception:
            _VEC_CACHE[f] = None
    return _VEC_CACHE[f]


def load_truth(path):
    """title -> (formula_vec, smiles). Prefers the test .pkl's stored target."""
    truth = {}
    if path.endswith(".pkl"):
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        for d in data:
            title, fvec = d.get("title"), d.get("formula")
            if title is None or fvec is None:
                continue
            truth[str(title)] = (
                tuple(int(round(float(x))) for x in fvec),
                d.get("smiles"),
            )
    else:  # .mgf
        if pyteomics_mgf is None:
            raise SystemExit("pyteomics is required to read .mgf truth")
        from rdkit import Chem
        from rdkit.Chem.rdMolDescriptors import CalcMolFormula

        for spec in pyteomics_mgf.read(path):
            p = spec.get("params", {})
            title, smiles = p.get("title"), p.get("smiles")
            if not title or not smiles:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            truth[str(title)] = (formula_vec(CalcMolFormula(mol)), smiles)
    return truth


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result", required=True, help="run_fiddle result CSV")
    ap.add_argument("--test_data", required=True,
                    help="test set .pkl (preferred) or .mgf")
    ap.add_argument("--topk", type=int, default=5, help="max k to report (default 5)")
    args = ap.parse_args()

    truth = load_truth(args.test_data)
    df = pd.read_csv(args.result)
    if "ID" not in df.columns:
        raise SystemExit("result CSV has no 'ID' column")

    refined_cols = [c for c in df.columns if c.startswith("Refined Formula (")]
    refined_cols.sort(key=lambda c: int(c.split("(")[1].rstrip(")")))
    if not refined_cols:
        raise SystemExit("result CSV has no 'Refined Formula (k)' columns")
    topk = min(args.topk, len(refined_cols))
    used_cols = refined_cols[:topk]

    n_rows = len(df)
    spectrum_hits = []          # per spectrum: list[topk] of 0/1 (cumulative top-k)
    compound_hits = {}          # smiles -> list of those hit vectors

    for _, row in df.iterrows():
        key = str(row["ID"])
        if key not in truth:
            continue
        true_vec, smiles = truth[key]
        rank = None
        if true_vec is not None:
            for i, c in enumerate(used_cols):
                if formula_vec(row[c]) == true_vec:
                    rank = i
                    break
        hits = [1 if (rank is not None and rank <= k) else 0 for k in range(topk)]
        spectrum_hits.append(hits)
        compound_hits.setdefault(smiles, []).append(hits)

    n_joined = len(spectrum_hits)
    if n_joined == 0:
        raise SystemExit("No rows joined — check that result 'ID' matches test 'title'.")

    # per-spectrum (micro): every spectrum weighted equally
    spec_acc = [sum(h[k] for h in spectrum_hits) / n_joined for k in range(topk)]
    # per-compound (macro): each compound's mean accuracy, then averaged
    comp_means = [
        [sum(h[k] for h in hlist) / len(hlist) for k in range(topk)]
        for hlist in compound_hits.values()
    ]
    n_comp = len(comp_means)
    comp_acc = [sum(m[k] for m in comp_means) / n_comp for k in range(topk)]

    print("Eval: {}  vs  {}".format(args.result, args.test_data))
    print("Joined: {} spectra / {} compounds  (matched {}/{} result rows)".format(
        n_joined, n_comp, n_joined, n_rows))
    print("          " + "".join("  top-{:<2d}".format(k + 1) for k in range(topk)))
    print("spectra " + "".join("  {:5.1f}%".format(100 * a) for a in spec_acc))
    print("compound" + "".join("  {:5.1f}%".format(100 * a) for a in comp_acc))


if __name__ == "__main__":
    main()
