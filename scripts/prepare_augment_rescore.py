"""
Prepare rescore training and validation pairs.

  1. Run TCN inference and refinement on the TCN training and validation sets:
     top-K candidates labelled correct or not
  2. Validation pairs: the validation set's raw candidates (no augmentation),
     used to pick rescore checkpoints by top-1. The TCN never trained on them,
     and the test set is not used.
  3. Training pairs: from the training set, cap positives per formula to
     --pos_cap, add cross-spectrum negatives (--neg_per_pos per positive,
     within --tolerance), and downsample to 1:1
  → <rescore_dir>/<stem>_rescore_train.pkl and <stem>_rescore_val.pkl

Usage:
    python scripts/prepare_augment_rescore.py \\
        --train_data ./data/cl_pkl_102026/qtof_maxmin_train.pkl \\
        --val_data ./data/cl_pkl_102026/qtof_maxmin_val.pkl \\
        --config_path ./msfiddle/config/fiddle_tcn_qtof.yml \\
        --resume_path ./check_point/fiddle_tcn_qtof_102026_muon.pt \\
        --rescore_dir ./data/rescore_102026_muon \\
        --pos_cap 10 --neg_per_pos 8 --tolerance 50 --device 0
"""

import argparse
import os
import pickle
import random
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys

# Make the repo root importable so `msfiddle` resolves without an editable install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from msfiddle.settings import load_config
from msfiddle.dataset import MS2FDataset
from msfiddle.model_tcn import MS2FNet_tcn
from msfiddle.utils import (
    ATOMS_INDEX_re,
    formula_refinement,
    formula_to_dict,
    formula_to_vector,
    monoisotopic_mass_calculator,
)
from msfiddle.utils.msms_utils import isotope_mass_shift, precursor_window_da


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


def intact_offset(smiles, stored_vec):
    """Atom-count offset from the stored target to the intact molecule.

    The pkl target folds neutral losses into the formula ([M+H-H2O]+ stores
    M - H2O); inference adds them back, so candidates and labels must be built
    for the intact molecule. Zero when there is no neutral loss or no SMILES.
    """
    zero = np.zeros(len(stored_vec), dtype=np.float32)
    if not smiles:
        return zero
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return zero
    return np.asarray(formula_to_vector(CalcMolFormula(mol)), dtype=np.float32) - (
        np.round(np.asarray(stored_vec, dtype=np.float32))
    )


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


def prepare_rescore_records(data_path, model, device, config, num_workers=8):
    """Run TCN inference on a dataset and return flat list of rescore dicts."""
    dataset = MS2FDataset(data_path)
    print(f"Loaded {len(dataset)} items")
    # Lookup built from the dataset's own records (no second copy of the pkl:
    # the Orbitrap train pkl alone is ~50GB in RAM).
    pkl_data = {d["title"]: [d["spec"], d["env"]] for d in dataset.data}
    loader = DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["train"]["num_workers"],
        drop_last=False,
    )

    spec_ids, y_true, y_pred, _ = run_inference(model, loader, device)

    # Work in intact-molecule space, as inference does: shift the prediction
    # and the target by each spectrum's neutral-loss offset and refine around
    # the intact mass.
    smiles = {d["title"]: d.get("smiles") for d in dataset.data}
    offsets = np.stack(
        [intact_offset(smiles[t], y) for t, y in zip(spec_ids, y_true.numpy())]
    )
    print(f"Spectra with a neutral-loss offset: {int((offsets != 0).any(1).sum())}")
    formula_pred = [vec2formula(v) for v in y_pred.numpy() + offsets]
    formula_true = [vec2formula(v) for v in np.round(y_true.numpy()) + offsets]
    mass_true = [monoisotopic_mass_calculator(f, mode="f") for f in formula_true]

    # Formula refinement as at inference: the ppm tolerance applies to the
    # precursor m/z, and each 13C isotope offset k is refined too (its
    # candidates are decoys here, since the training masses are exact).
    # Parallelised across CPU cores.
    pp = config["post_processing"]
    index_to_adduct = {v: t for t, v in config["encoding"]["precursor_type"].items()}
    worker_args, owners = [], []
    for i, (pred_f, m, title) in enumerate(zip(formula_pred, mass_true, spec_ids)):
        env = pkl_data[title][1]
        adduct = index_to_adduct[int(env[2])]
        window = (
            precursor_window_da(adduct, float(env[0]), pp["mass_tolerance"])
            if pp["ppm_mode"]
            else pp["mass_tolerance"]
        )
        for k in pp["isotope_errors"]:
            worker_args.append(
                (
                    pred_f,
                    m - isotope_mass_shift(adduct, k),
                    window,
                    False,  # window is in Da
                    pp["top_k"],
                    pp["maxium_miss_atom_num"],
                    pp["time_out"],
                    pp["refine_atom_type"],
                    pp["refine_atom_num"],
                )
            )
            owners.append((i, k))

    print(
        f"Post-processing {len(worker_args)} refinements with {num_workers} workers..."
    )
    with Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(_refine_worker, worker_args, chunksize=32),
                total=len(worker_args),
                desc="Post-processing",
            )
        )

    # Pool each spectrum's candidates across offsets, keeping the first
    candidates = [{} for _ in spec_ids]  # formula -> isotope offset
    for (i, k), formulas in zip(owners, results):
        for formula in formulas:
            if formula is not None:
                candidates[i].setdefault(formula, k)

    # Label and collect records
    records = []
    counts = {"correct": 0, "incorrect": 0, "from isotope offsets": 0}
    for title, truth, found in zip(spec_ids, formula_true, candidates):
        spec, env = pkl_data[title]
        truth = formula_to_dict(truth)
        for formula, k in found.items():
            label = float(formula_to_dict(formula) == truth)
            counts["correct" if label else "incorrect"] += 1
            counts["from isotope offsets"] += k > 0
            records.append(
                {
                    "title": title,
                    "pred_formula": formula,
                    "spec": spec,
                    "env": env,
                    "label": label,
                }
            )
    print("  " + ", ".join(f"{name}: {n}" for name, n in counts.items()))

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
    for items in formula_groups.values():
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
        description="Prepare rescore training and validation pairs"
    )
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="TCN training pkl (e.g. qtof_maxmin_train.pkl)",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        required=True,
        help="TCN validation pkl (e.g. qtof_maxmin_val.pkl)",
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

    config = load_config(args.config_path)

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

    os.makedirs(args.rescore_dir, exist_ok=True)
    stem = os.path.basename(args.train_data).replace("_train.pkl", "")
    rescore_train_path = os.path.join(args.rescore_dir, f"{stem}_rescore_train.pkl")
    rescore_val_path = os.path.join(args.rescore_dir, f"{stem}_rescore_val.pkl")

    print(f"Validation set: {args.val_data}")
    val_records = prepare_rescore_records(
        args.val_data, model, device, config, args.num_workers
    )
    with open(rescore_val_path, "wb") as f:
        pickle.dump(val_records, f)
    print(f"Saved {len(val_records)} items to {rescore_val_path}")

    print(f"\nTraining set: {args.train_data}")
    train_records = prepare_rescore_records(
        args.train_data, model, device, config, args.num_workers
    )
    print(f"Raw rescore training records: {len(train_records)}")

    print("\nAugmenting the training pairs...")
    train_augmented = augment(
        train_records, args.pos_cap, args.neg_per_pos, tolerance, ppm_mode, args.seed
    )
    with open(rescore_train_path, "wb") as f:
        pickle.dump(train_augmented, f)
    print(f"Saved {len(train_augmented)} items to {rescore_train_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
