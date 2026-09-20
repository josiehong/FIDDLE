"""Analyze count-prediction ambiguity and error anatomy from a result CSV.

Reads the `Y Pred` float vectors and `Pred Formula` from a train-export or
run_fiddle result CSV, ground truth from the test pkl, and reports:
  - distribution of distance-to-nearest-integer (the ~0.5 ambiguity)
  - ambiguity split by correct vs wrong predictions
  - exact / H-only / heavy-atom error anatomy

Example:
  python scripts/analyze_count_ambiguity.py \
      --result ./result/fiddle_tcn_qtof_082426.csv \
      --test_data ./data/cl_pkl_060526/qtof_maxmin_test.pkl
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from msfiddle.utils.mol_utils import ATOMS_INDEX_re  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result", required=True, help="result CSV with Y Pred column")
    ap.add_argument("--test_data", required=True, help="test set .pkl (ground truth)")
    args = ap.parse_args()

    with open(args.test_data, "rb") as fh:
        data = pickle.load(fh)
    truth = {
        str(d["title"]): np.round(np.asarray(d["formula"], dtype=float)).astype(int)
        for d in data
        if d.get("title") is not None and d.get("formula") is not None
    }
    del data

    sep = "\t" if args.result.endswith(".tsv") else None
    df = pd.read_csv(args.result, sep=sep, engine="python")
    ypred_sep = ";" if df["Y Pred"].iloc[0].count(";") else ","

    amb_correct, amb_wrong = [], []
    n = exact = h_only = 0
    n_dim = len(ATOMS_INDEX_re)
    per_elem = np.zeros(n_dim)
    for _, row in df.iterrows():
        tv = truth.get(str(row["ID"]))
        yp = row["Y Pred"]
        if tv is None or not isinstance(yp, str):
            continue
        c = np.array([float(x) for x in yp.split(ypred_sep)])
        if len(c) != n_dim:
            continue
        n += 1
        r = np.round(c).astype(int)
        r[r < 0] = 0
        d = np.abs(c - np.round(c))
        per_elem += d >= 0.35
        heavy_ok = all(r[j] == tv[j] for j in range(n_dim) if ATOMS_INDEX_re[j] != "H")
        correct = bool((r == tv).all())
        (amb_correct if correct else amb_wrong).append(d.max())
        if correct:
            exact += 1
        elif heavy_ok:
            h_only += 1

    n_wrong = n - exact - h_only
    print(f"{n} spectra   result: {args.result}")
    print(
        f"\nexact: {100*exact/n:.1f}%   H-only error: {100*h_only/n:.1f}%"
        f"   heavy-atom error: {100*n_wrong/n:.1f}%"
    )
    amb_all = np.array(amb_correct + amb_wrong)
    print("\nmax per-spectrum distance to nearest integer:")
    for thr in (0.25, 0.35, 0.45):
        print(f"  >=1 element d >= {thr}: {100*(amb_all>=thr).mean():.1f}%")
    if amb_correct and amb_wrong:
        print(
            f"mean max-ambiguity  correct: {np.mean(amb_correct):.3f}"
            f"   wrong: {np.mean(amb_wrong):.3f}"
        )
    top = np.argsort(-per_elem)[:5]
    print(
        "elements most often ambiguous (d>=0.35): "
        + "  ".join(f"{ATOMS_INDEX_re[i]}: {100*per_elem[i]/n:.1f}%" for i in top)
    )


if __name__ == "__main__":
    main()
