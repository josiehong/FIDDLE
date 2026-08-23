#!/usr/bin/env python
"""Cap the number of spectra per compound by collision-energy-stratified sampling.

Rebalancing experiment: heavily-measured compounds dominate the per-spectrum
training signal. This trims each compound to at most `--cap` spectra, chosen to
span the compound's collision-energy range (so we keep CE coverage rather than a
random clump at one energy). Compounds with <= cap spectra are kept whole;
nothing is up-sampled.

Operates on an existing train .pkl (the output of prepare_msms.py) and writes a
capped train .pkl plus its companion contrastive-pairs file (the CL trainer reads
`<stem>_train_pairs.pkl`); the pairs index into the spectra list, so they must be
rebuilt for the capped subset. The TEST split is never touched, so a model trained
on the output is directly comparable to the full-train model on the same test set.

Each spectrum's normalized collision energy is read from env[1] (the encoded
[precursor_mz, nce, precursor_type] vector).

The --out name must end with `_train.pkl` (e.g. qtof_maxmin_cek8_train.pkl) so the
pairs file and the downstream rescore filenames derive cleanly.

Usage:
  python scripts/balance_ce_cap.py \
      --train_data ./data/cl_pkl_060526/qtof_maxmin_train.pkl \
      --out        ./data/cl_pkl_060526/qtof_maxmin_cek8_train.pkl \
      --cap 8 --ce_bins 3 --seed 42
"""
import argparse
import os
import pickle
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from msfiddle.utils.pkl_utils import spec2pair  # noqa: E402


def ce_stratified_pick(nce, cap, n_bins, rng):
    """Indices (into the compound's spectra) to keep: <=cap, spread over CE.

    Buckets spectra into CE quantile bins, then draws round-robin from low to
    high CE so the kept set covers the energy range. Random within a bin.
    """
    n = len(nce)
    if n <= cap:
        return list(range(n))
    nce = np.asarray(nce, dtype=float)
    finite = nce[np.isfinite(nce)]
    if finite.size == 0:  # no usable CE -> plain random cap
        return sorted(rng.choice(n, cap, replace=False).tolist())

    nb = int(min(n_bins, max(1, np.unique(finite).size)))
    edges = np.quantile(finite, np.linspace(0.0, 1.0, nb + 1))
    bins = np.digitize(nce, edges[1:-1])  # 0..nb-1
    bins[~np.isfinite(nce)] = 0           # missing CE -> lowest bin

    buckets = defaultdict(list)
    for i in range(n):
        buckets[int(bins[i])].append(i)
    for b in buckets:
        buckets[b] = list(rng.permutation(buckets[b]))

    picked, order = [], sorted(buckets)
    while len(picked) < cap:
        progressed = False
        for b in order:
            if buckets[b]:
                picked.append(buckets[b].pop())
                progressed = True
                if len(picked) == cap:
                    break
        if not progressed:
            break
    return sorted(picked)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_data", required=True, help="input train .pkl")
    ap.add_argument("--out", required=True, help="output capped train .pkl")
    ap.add_argument("--cap", type=int, required=True, help="max spectra per compound")
    ap.add_argument("--ce_bins", type=int, default=3, help="CE strata (default 3)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.train_data, "rb") as f:
        data = pickle.load(f)

    groups = defaultdict(list)
    for i, d in enumerate(data):
        groups[d.get("smiles")].append(i)

    # report the pre-cap distribution so the cap can be chosen sensibly
    sizes = np.array([len(v) for v in groups.values()])
    pct = np.percentile(sizes, [50, 90, 99])
    print("Compounds: {}  | spectra: {}".format(len(groups), len(data)))
    print("Spectra/compound  median={:.0f}  p90={:.0f}  p99={:.0f}  max={}".format(
        pct[0], pct[1], pct[2], sizes.max()))

    rng = np.random.default_rng(args.seed)
    keep = []
    n_capped = 0
    for smiles in sorted(groups, key=lambda s: (s is None, s)):  # deterministic
        idxs = groups[smiles]
        nce = [float(data[i]["env"][1]) for i in idxs]
        sel = ce_stratified_pick(nce, args.cap, args.ce_bins, rng)
        if len(sel) < len(idxs):
            n_capped += 1
        keep.extend(idxs[s] for s in sel)

    keep.sort()
    capped = [data[i] for i in keep]
    with open(args.out, "wb") as f:
        pickle.dump(capped, f)

    print("Capped {} / {} compounds (>{} spectra).".format(
        n_capped, len(groups), args.cap))
    print("Spectra: {} -> {}  ({:.1%} kept).  Wrote {}".format(
        len(data), len(capped), len(capped) / len(data), args.out))

    # Rebuild contrastive pairs for the capped subset (indices into `capped`).
    if not args.out.endswith("_train.pkl"):
        raise SystemExit("--out must end with '_train.pkl' so the pairs file derives cleanly")
    random.seed(args.seed)
    pairs = spec2pair(capped, set(), None)  # spec2pair ignores the encoder arg
    pairs_path = args.out.replace("_train.pkl", "_train_pairs.pkl")
    with open(pairs_path, "wb") as f:
        pickle.dump(pairs, f)
    print("Wrote {} pairs to {}".format(len(pairs["idx1"]), pairs_path))


if __name__ == "__main__":
    main()
