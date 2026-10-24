"""Train a Siamese rescore model (element-wise interaction + BCE).

Architecture
------------
1. Frozen spectrum encoder (MS2FNet_tcn) → z_spec  (L2-normalised, 512-dim)
2. Trainable FormulaEncoder              → z_form  (L2-normalised, 512-dim)
3. score = sigmoid(RescoreHead(z_spec ⊙ z_form))

Only FormulaEncoder and RescoreHead are trained.  The spectrum encoder
weights are frozen at the pre-trained TCN checkpoint.

The checkpoint is chosen by top-1 on the validation spectra (pairs from
held-out training compounds, see prepare_augment_rescore.py): the fraction of
spectra whose highest-scored candidate is correct, which is what prediction
reports. Pair-level AUC barely moves between runs whose top-1 differs by
several points.

Usage:
    python scripts/train_rescore.py \\
        --train_data ./data/rescore_102026_muon/qtof_maxmin_rescore_train.pkl \\
        --val_data ./data/rescore_102026_muon/qtof_maxmin_rescore_val.pkl \\
        --config_path ./msfiddle/config/fiddle_tcn_qtof.yml \\
        --resume_path ./check_point/fiddle_tcn_qtof_102026_muon.pt \\
        --checkpoint_path ./check_point/fiddle_rescore_qtof_102026_muon.pt \\
        --seed 1 --device 0
"""

import argparse
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import os
import sys

# Make the repo root importable so `msfiddle` resolves without an editable install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from msfiddle.settings import load_config
from msfiddle.model_tcn import MS2FNet_tcn, FormulaEncoder, RescoreHead, tcn_hash
from msfiddle.utils.mol_utils import formula_to_vector

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_pairs(path, spec_encoder, batch_size, device):
    """Load a rescore pkl and embed each distinct spectrum once.

    The spectrum encoder is frozen and deterministic, so every pair of a spectrum
    shares one embedding; the raw spectra are dropped once embedded.

    Returns:
        z:   (S, 512) L2-normalised spectrum embeddings (CPU)
        idx: (N,) row of z for each pair
        f:   (N, 13) formula vectors
        y:   (N,) labels
    """
    with open(path, "rb") as file:
        data = pickle.load(file)
    print("Load {} pairs from {}".format(len(data), path))

    rows, specs = {}, []
    for d in data:
        if d["title"] not in rows:
            rows[d["title"]] = len(specs)
            specs.append((d["spec"][:, 0], d["env"]))
    idx = torch.tensor([rows[d["title"]] for d in data])
    f = torch.tensor(
        np.array([formula_to_vector(d["pred_formula"]) for d in data]),
        dtype=torch.float32,
    )
    y = torch.tensor([d["label"] for d in data], dtype=torch.float32)
    del data, rows

    z = []
    with torch.no_grad():
        for i in tqdm(range(0, len(specs), batch_size), desc="Embed spectra"):
            chunk = specs[i : i + batch_size]
            x = torch.tensor(np.stack([s for s, _ in chunk]), dtype=torch.float32)
            env = torch.tensor(np.stack([e for _, e in chunk]), dtype=torch.float32)
            env[:, 0] = 0.0  # zero out precursor_mz to prevent mass-based shortcut
            z_specs, _, _, _, _ = spec_encoder(x.to(device), env.to(device))
            z.append(F.normalize(z_specs, dim=1).cpu())
    print(f"Embedded {len(specs)} spectra")
    return torch.cat(z), idx, f, y


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------


def run_epoch(pairs, formula_encoder, rescore_head, batch_size, device, optimizer=None):
    """One pass over the pairs; trains when an optimizer is given."""
    z, idx, f, y = pairs
    train = optimizer is not None
    formula_encoder.train(train)
    rescore_head.train(train)

    criterion = nn.BCEWithLogitsLoss()
    order = torch.randperm(len(y)) if train else torch.arange(len(y))
    total_loss = 0.0
    seen, scores = [], []

    with torch.set_grad_enabled(train), tqdm(
        order.split(batch_size), desc="Train" if train else "Valid"
    ) as bar:
        for b in bar:
            z_specs = z[idx[b]].to(device)
            z_forms = formula_encoder(f[b].to(device))  # (B, 512), L2-normalised
            logits = rescore_head(z_specs * z_forms)  # (B,)
            loss = criterion(logits, y[b].to(device))

            if train:
                if torch.isnan(loss):
                    continue
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                bar.set_postfix(loss=loss.item())

            total_loss += loss.item() * len(b)
            seen.append(b)
            scores.append(torch.sigmoid(logits).detach().cpu())

    y_true = y[torch.cat(seen)]
    y_score = torch.cat(scores)
    acc = ((y_score > 0.5).float() == y_true).float().mean().item()
    auc = roc_auc_score(y_true.numpy(), y_score.numpy())
    top1 = top1_accuracy(idx[torch.cat(seen)].numpy(), y_true.numpy(), y_score.numpy())
    return total_loss / len(y_true), acc, auc, top1


def top1_accuracy(spectrum, y_true, y_score):
    """Fraction of spectra whose highest-scored candidate is correct."""
    order = np.lexsort((-y_score, spectrum))  # by spectrum, best score first
    grouped = spectrum[order]
    first = np.r_[True, grouped[1:] != grouped[:-1]]
    return float(y_true[order][first].mean())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Siamese rescore model (interaction head + BCE)"
    )
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument(
        "--val_data",
        type=str,
        required=True,
        help="Validation pairs from prepare_augment_rescore.py (*_rescore_val.pkl)",
    )
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument(
        "--resume_path", type=str, required=True, help="Pretrained TCN checkpoint"
    )
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, nargs="+", default=[0])
    parser.add_argument("--no_cuda", action="store_true", default=False)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    config = load_config(args.config_path)
    print(f"Config: {args.config_path}")

    device = (
        torch.device(f"cuda:{args.device[0]}")
        if torch.cuda.is_available() and not args.no_cuda
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    # Frozen spectrum encoder: embeds each distinct spectrum once
    spec_encoder = MS2FNet_tcn(config["model"]).to(device)
    state_dict = torch.load(args.resume_path, map_location=device)["model_state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    spec_encoder.load_state_dict(state_dict, strict=False)
    weights_hash = tcn_hash(state_dict)
    spec_encoder.eval()
    print(f"Loaded frozen spectrum encoder from {args.resume_path}")

    batch_size = config["train_rescore"]["batch_size"]
    train_pairs = load_pairs(args.train_data, spec_encoder, batch_size, device)
    valid_pairs = load_pairs(args.val_data, spec_encoder, batch_size, device)
    del spec_encoder

    # Trainable FormulaEncoder + RescoreHead
    formula_encoder = FormulaEncoder(config["model"]).to(device)
    rescore_head = RescoreHead(config["model"]).to(device)

    n_params = sum(p.numel() for p in formula_encoder.parameters()) + sum(
        p.numel() for p in rescore_head.parameters()
    )
    print(f"# Trainable params (FormulaEncoder + RescoreHead): {n_params}")

    trainable = list(formula_encoder.parameters()) + list(rescore_head.parameters())
    optimizer = optim.AdamW(trainable, lr=config["train_rescore"]["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=config["train_rescore"]["patience"],
    )

    best_top1 = -1.0
    early_stop_patience = 0
    early_stop_step = config["train_rescore"]["early_stop_step"]

    for epoch in range(1, config["train_rescore"]["epochs"] + 1):
        print(f"\n=====Epoch {epoch}")
        train_loss, train_acc, train_auc, _ = run_epoch(
            train_pairs, formula_encoder, rescore_head, batch_size, device, optimizer
        )
        print(
            f"Train loss: {train_loss:.4f}  acc: {train_acc:.4f}  auc: {train_auc:.4f}"
        )

        valid_loss, valid_acc, valid_auc, valid_top1 = run_epoch(
            valid_pairs, formula_encoder, rescore_head, batch_size, device
        )
        print(
            f"Validation loss: {valid_loss:.4f}  acc: {valid_acc:.4f}  "
            f"auc: {valid_auc:.4f}  top-1: {valid_top1:.4f}"
        )

        if valid_top1 > best_top1:
            best_top1 = valid_top1
            torch.save(
                {
                    "epoch": epoch,
                    "formula_encoder_state_dict": formula_encoder.state_dict(),
                    "rescore_head_state_dict": rescore_head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_top1": best_top1,
                    "seed": args.seed,
                    "tcn_hash": weights_hash,
                },
                args.checkpoint_path,
            )
            early_stop_patience = 0
            print("Saving checkpoint... Early stop patience reset")
        else:
            early_stop_patience += 1
            print(f"Early stop count: {early_stop_patience}/{early_stop_step}")

        scheduler.step(valid_top1)
        print(f"Best top-1 so far: {best_top1:.4f}")

        if early_stop_patience == early_stop_step:
            print("Early stop!")
            break

    print("Done!")
