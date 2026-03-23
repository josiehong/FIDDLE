"""
Prepare and augment rescore train/test data directly from TCN train/test sets.

Train pipeline (--train_data):
  1. Run TCN inference → top-K candidates labelled correct/incorrect
  2. Cap positives per formula to --pos_cap (default 10)
  3. Generate cross-spectrum negatives (--neg_per_pos per positive, --tolerance ppm)
  4. Pool original + cross-spectrum negatives, downsample to 1:1
  → <rescore_dir>/qtof_maxmin_rescore_train.pkl

Test pipeline (--test_data):
  1. Run TCN inference → top-K candidates labelled correct/incorrect
  2. No augmentation
  → <rescore_dir>/qtof_maxmin_rescore_test.pkl

Usage:
    python prepare_augment_rescore.py \\
        --train_data ./data/cl_pkl_031826/qtof_maxmin_train.pkl \\
        --test_data  ./data/cl_pkl_031826/qtof_maxmin_test.pkl \\
        --config_path ./config/fiddle_tcn_qtof.yml \\
        --resume_path ./check_point/fiddle_tcn_qtof_031826.pt \\
        --rescore_dir ./data/cl_pkl_031826 \\
        --pos_cap 10 \\
        --neg_per_pos 8 \\
        --tolerance 50 \\
        --device 2
"""

import argparse
import os
import pickle
import random
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MS2FDataset
from model_tcn import MS2FNet_tcn
from utils import ATOMS_INDEX_re, formula_to_dict, formula_refinement


def _refine_worker(args):
    """Module-level worker for multiprocessing: runs formula_refinement for one spectrum."""
    (
        pred_f,
        m,
        mass_tolerance,
        ppm_mode,
        top_k,
        max_miss,
        time_out,
        atom_types_base,
        atom_nums_base,
    ) = args
    refine_atom_type = list(atom_types_base)
    refine_atom_num = list(atom_nums_base)
    for atom, cnt in formula_to_dict(pred_f).items():
        if atom == "H" or atom in refine_atom_type:
            continue
        refine_atom_type.append(atom)
        refine_atom_num.append(max(1, int(cnt)))
    refined = formula_refinement(
        [pred_f],
        m,
        mass_tolerance,
        ppm_mode,
        top_k,
        max_miss,
        time_out,
        refine_atom_type,
        refine_atom_num,
    )
    return refined["formula"]  # list of top_k formula strings (or None)


# ---------------------------------------------------------------------------
# TCN inference helpers (from prepare_rescore.py)
# ---------------------------------------------------------------------------


def vec2formula(vec, withH=True):
    formula = ""
    for idx, v in enumerate(vec):
        v = round(float(v))
        if v <= 0:
            continue
        elif not withH and ATOMS_INDEX_re[idx] == "H":
            continue
        elif v == 1:
            formula += ATOMS_INDEX_re[idx]
        else:
            formula += ATOMS_INDEX_re[idx] + str(v)
    return formula


def run_inference(model, loader, device):
    model.eval()
    spec_ids, y_true_all, y_pred_all, mass_true_all = [], [], [], []
    with tqdm(total=len(loader), desc="Inference") as bar:
        for _, batch in enumerate(loader):
            spec_id, y, x, mass, env = batch
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            env = env.to(device, dtype=torch.float32)
            mass = mass.to(device, dtype=torch.float32)
            with torch.no_grad():
                _, pred_f, _, _, _ = model(x, env)
            spec_ids.extend(list(spec_id))
            y_true_all.append(y.cpu())
            y_pred_all.append(pred_f.cpu())
            mass_true_all.append(mass.cpu())
            bar.update(1)
    return (
        spec_ids,
        torch.cat(y_true_all),
        torch.cat(y_pred_all),
        torch.cat(mass_true_all),
    )


def prepare_rescore_records(data_path, model, device, config, pkl_data, num_workers=8):
    """Run TCN inference on a dataset and return flat list of rescore dicts."""
    dataset = MS2FDataset(data_path)
    loader = DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["train"]["num_workers"],
        drop_last=False,
    )

    spec_ids, y_true, y_pred, mass_true = run_inference(model, loader, device)

    formula_pred = [vec2formula(v) for v in y_pred]
    formula_true = [vec2formula(v) for v in y_true]

    # Formula refinement (post-processing) — parallelised across CPU cores
    pp = config["post_processing"]
    top_k = pp["top_k"]

    worker_args = [
        (
            pred_f,
            m.item(),
            pp["mass_tolerance"],
            pp["ppm_mode"],
            top_k,
            pp["maxium_miss_atom_num"],
            pp["time_out"],
            pp["refine_atom_type"],
            pp["refine_atom_num"],
        )
        for pred_f, m in zip(formula_pred, mass_true)
    ]

    print(f"Post-processing {len(worker_args)} spectra with {num_workers} workers...")
    with Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(_refine_worker, worker_args, chunksize=32),
                total=len(worker_args),
                desc="Post-processing",
            )
        )

    # results[i] is a list of top_k formula strings
    formula_refined = {
        f"Refined Formula ({k})": [r[k] for r in results] for k in range(top_k)
    }

    # Label and collect records
    res_df = pd.DataFrame({"ID": spec_ids, "Formula": formula_true, **formula_refined})

    records = []
    for k in range(top_k):
        col = f"Refined Formula ({k})"
        res_df[f"Label ({k})"] = res_df.apply(
            lambda row: formula_to_dict(row["Formula"]) == formula_to_dict(row[col]),
            axis=1,
        )
        for label_val, flag in [(1.0, True), (0.0, False)]:
            sub = res_df[res_df[f"Label ({k})"] == flag].dropna(subset=[col])
            for _, row in sub.iterrows():
                title = row["ID"]
                spec, env = pkl_data[title]
                records.append(
                    {
                        "title": title,
                        "pred_formula": row[col],
                        "spec": spec,
                        "env": env,
                        "label": label_val,
                    }
                )
            tag = "correct" if label_val == 1.0 else "incorrect"
            print(f"  k={k} {tag}: {len(sub)}")

    return records


# ---------------------------------------------------------------------------
# Augmentation helpers (from augment_rescore.py)
# ---------------------------------------------------------------------------


def within_tolerance(mz_a, mz_b, tolerance, ppm_mode):
    if ppm_mode:
        return abs(mz_a - mz_b) / mz_a * 1e6 < tolerance
    else:
        return abs(mz_a - mz_b) < tolerance


def augment(data, pos_cap, neg_per_pos, tolerance, ppm_mode, seed):
    random.seed(seed)
    np.random.seed(seed)

    all_positives = [d for d in data if d["label"] == 1.0]
    all_negatives = [d for d in data if d["label"] == 0.0]
    print(
        f"Before capping — Positives: {len(all_positives)}  Negatives: {len(all_negatives)}"
    )

    # Step 1: cap positives per formula
    formula_groups = defaultdict(list)
    for d in all_positives:
        formula_groups[d["pred_formula"]].append(d)

    capped_positives = []
    for formula, items in formula_groups.items():
        capped_positives.extend(random.sample(items, min(pos_cap, len(items))))
    random.shuffle(capped_positives)

    print(
        f"After capping (max {pos_cap}/formula) — Positives: {len(capped_positives)} "
        f"from {len(formula_groups)} unique formulas"
    )

    # Step 2: cross-spectrum negatives using mz binning
    bin_size = 1.0
    mz_bins = defaultdict(list)
    for pos in capped_positives:
        mz = float(pos["env"][0])
        mz_bins[int(mz / bin_size)].append(pos)

    cross_negatives = []
    skipped = 0
    for pos_a in tqdm(capped_positives, desc="Building cross-spectrum negatives"):
        f_a = pos_a["pred_formula"]
        mz_a = float(pos_a["env"][0])
        bin_key = int(mz_a / bin_size)

        candidates = [
            pos_b
            for bk in (bin_key - 1, bin_key, bin_key + 1)
            for pos_b in mz_bins.get(bk, [])
            if pos_b["pred_formula"] != f_a
            and within_tolerance(mz_a, float(pos_b["env"][0]), tolerance, ppm_mode)
        ]

        if not candidates:
            skipped += 1
            continue

        for pos_b in random.sample(candidates, min(neg_per_pos, len(candidates))):
            cross_negatives.append(
                {
                    "title": pos_b["title"],
                    "pred_formula": f_a,
                    "spec": pos_b["spec"],
                    "env": pos_b["env"],
                    "label": 0.0,
                }
            )

    print(f"Cross-spectrum negatives generated: {len(cross_negatives)}")
    print(f"Positives with no mass-matched partner: {skipped}")

    # Step 3: pool all negatives, downsample to 1:1
    neg_pool = all_negatives + cross_negatives
    random.shuffle(neg_pool)
    sampled_negatives = neg_pool[: len(capped_positives)]

    augmented = capped_positives + sampled_negatives
    random.shuffle(augmented)

    print(f"\nFinal dataset: {len(augmented)} items")
    print(f"  Positives            : {len(capped_positives)}")
    print(f"  Original negatives   : {len(all_negatives)}")
    print(f"  Cross-spec negatives : {len(cross_negatives)}")
    print(
        f"  Sampled negatives    : {len(sampled_negatives)} (from pool of {len(neg_pool)})"
    )
    print(
        f"  Ratio                : 1:{len(sampled_negatives) / len(capped_positives):.1f}"
    )
    return augmented


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Prepare and augment rescore data from TCN train/test sets"
    )
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="TCN training pkl (e.g. qtof_maxmin_train.pkl)",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="TCN test pkl (e.g. qtof_maxmin_test.pkl)",
    )
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument(
        "--resume_path", type=str, required=True, help="Pretrained TCN checkpoint"
    )
    parser.add_argument(
        "--rescore_dir", type=str, required=True, help="Directory to save rescore pkls"
    )
    parser.add_argument(
        "--pos_cap", type=int, default=10, help="Max positives per formula (default 10)"
    )
    parser.add_argument(
        "--neg_per_pos",
        type=int,
        default=8,
        help="Cross-spectrum negatives to generate per positive (default 8)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Override mass tolerance from config (ppm or Da)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="CPU workers for parallel post-processing (default 8)",
    )
    parser.add_argument("--device", type=int, nargs="+", default=[0])
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    tolerance = (
        args.tolerance
        if args.tolerance is not None
        else config["post_processing"]["mass_tolerance"]
    )
    ppm_mode = config["post_processing"]["ppm_mode"]
    print(f"Mass tolerance: {tolerance} {'ppm' if ppm_mode else 'Da'}")

    device = (
        torch.device(f"cuda:{args.device[0]}")
        if torch.cuda.is_available() and not args.no_cuda
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    # Load TCN model
    model = MS2FNet_tcn(config["model"]).to(device)
    if len(args.device) > 1:
        model = nn.DataParallel(model, device_ids=args.device)
    state = torch.load(args.resume_path, map_location=device)["model_state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded TCN model from {args.resume_path}")

    # Derive output paths from input filenames
    train_stem = os.path.basename(args.train_data).replace("_train.pkl", "")
    test_stem = os.path.basename(args.test_data).replace("_test.pkl", "")
    rescore_train_path = os.path.join(
        args.rescore_dir, f"{train_stem}_rescore_train.pkl"
    )
    rescore_test_path = os.path.join(args.rescore_dir, f"{test_stem}_rescore_test.pkl")

    # ── Train split: inference + augmentation ────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Train split: {args.train_data}")
    with open(args.train_data, "rb") as f:
        train_raw = pickle.load(f)
    print(f"Loaded {len(train_raw)} train items")
    train_pkl_data = {d["title"]: [d["spec"], d["env"]] for d in train_raw}

    train_records = prepare_rescore_records(
        args.train_data, model, device, config, train_pkl_data, args.num_workers
    )
    print(f"\nRaw rescore train records: {len(train_records)}")

    print("\nAugmenting train split...")
    train_augmented = augment(
        train_records, args.pos_cap, args.neg_per_pos, tolerance, ppm_mode, args.seed
    )
    with open(rescore_train_path, "wb") as f:
        pickle.dump(train_augmented, f)
    print(f"Saved {len(train_augmented)} items to {rescore_train_path}")

    # ── Test split: inference only, no augmentation ──────────────────────────
    print(f"\n{'='*60}")
    print(f"Test split: {args.test_data}")
    with open(args.test_data, "rb") as f:
        test_raw = pickle.load(f)
    print(f"Loaded {len(test_raw)} test items")
    test_pkl_data = {d["title"]: [d["spec"], d["env"]] for d in test_raw}

    test_records = prepare_rescore_records(
        args.test_data, model, device, config, test_pkl_data, args.num_workers
    )
    print(f"\nRaw rescore test records: {len(test_records)}")
    with open(rescore_test_path, "wb") as f:
        pickle.dump(test_records, f)
    print(f"Saved {len(test_records)} items to {rescore_test_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
