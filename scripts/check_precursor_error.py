"""Classify the precursor mass error of MGF spectra with known formulas.

For each spectrum with a SMILES or FORMULA, the measured precursor m/z is
compared with the theoretical one and put in one class:

  within      |error| <= --ppm on the precursor m/z
  isotope_k   the measured m/z is the k-th 13C isotope peak (k = 1..--max_isotope)
  other       anything else (wrong adduct, wrong annotation, ...)

`neutral_miss` counts spectra that are `within` on m/z but exceed --ppm on the
neutral mass, which refinement's ppm window is applied to.

Usage:
    python scripts/check_precursor_error.py ./data/casmi2016.mgf ./data/embl.mgf \\
        --ppm 10 --out ./result/precursor_error.csv
"""

import argparse
import os
import re
import sys
from collections import Counter

import pandas as pd
from pyteomics import mgf
from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from msfiddle.utils.mol_utils import monoisotopic_mass_calculator  # noqa: E402
from msfiddle.utils.msms_utils import (  # noqa: E402
    C13_SHIFT,
    mass_calculator,
    precursor_mz_calculator,
)

RDLogger.DisableLog("rdApp.*")


def truth_formula(params):
    mol = Chem.MolFromSmiles(params["smiles"]) if params.get("smiles") else None
    return CalcMolFormula(mol) if mol is not None else params.get("formula")


def charge_of(precursor_type):
    match = re.search(r"\](\d*)[+-]$", precursor_type)
    return int(match.group(1) or 1) if match else 1


def classify(params, ppm, max_isotope):
    formula, adduct = truth_formula(params), params.get("precursor_type")
    mz = params.get("precursor_mz", (params.get("pepmass") or [None])[0])
    if not formula or not adduct or mz is None:
        return None
    mz = float(mz)
    mass = monoisotopic_mass_calculator(formula, "f")
    theo_mz = precursor_mz_calculator(adduct, mass)
    error_mz = mz - theo_mz
    row = {
        "title": params.get("title"),
        "precursor_type": adduct,
        "theoretical_mz": theo_mz,
        "error_ppm": error_mz / theo_mz * 1e6,
        "neutral_error_ppm": (mass_calculator(adduct, mz) - mass) / mass * 1e6,
    }
    for k in range(max_isotope + 1):
        shifted = error_mz - k * C13_SHIFT / charge_of(adduct)
        if abs(shifted) / theo_mz * 1e6 <= ppm:
            row["class"] = "within" if k == 0 else f"isotope_{k}"
            break
    else:
        row["class"] = "other"
    row["neutral_miss"] = (
        row["class"] == "within" and abs(row["neutral_error_ppm"]) > ppm
    )
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify precursor mass errors")
    parser.add_argument("mgf", nargs="+")
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--max_isotope", type=int, default=2)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    rows = []
    for path in args.mgf:
        file_rows = []
        for spec in mgf.read(path):
            row = classify(spec["params"], args.ppm, args.max_isotope)
            if row is not None:
                file_rows.append({"file": os.path.basename(path), **row})
        counts = Counter(r["class"] for r in file_rows)
        misses = sum(r["neutral_miss"] for r in file_rows)
        n = len(file_rows)
        summary = ", ".join(
            f"{c} {counts[c]} ({counts[c] / n:.1%})" for c in sorted(counts)
        )
        print(f"{path}: {n} spectra with truth: {summary}; neutral_miss {misses}")
        rows.extend(file_rows)

    if args.out:
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"Saved per-spectrum errors to {args.out}")
