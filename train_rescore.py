"""Train a Siamese rescore model (interaction head + BCE).

Architecture
------------
1. Frozen spectrum encoder (MS2FNet_tcn) → z_spec  (L2-normalised, D-dim)
2. Trainable FormulaEncoder              → z_form  (L2-normalised, D-dim)
3. interaction = z_spec ⊙ z_form         (element-wise product, D-dim)
4. RescoreHead(interaction)              → scalar logit → sigmoid → BCE

Only FormulaEncoder and RescoreHead are trained.  The spectrum encoder
weights are frozen at the pre-trained TCN checkpoint.

Usage:
    python train_rescore.py \\
        --train_data ./data/cl_pkl_031826/qtof_maxmin_rescore_train.pkl \\
        --test_data  ./data/cl_pkl_031826/qtof_maxmin_rescore_test.pkl \\
        --config_path ./config/fiddle_tcn_qtof.yml \\
        --resume_path ./check_point/fiddle_tcn_qtof_031826.pt \\
        --checkpoint_path ./check_point/fiddle_rescore_qtof_031826.pt \\
        --device 2
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RescoreDataset
from model_tcn import MS2FNet_tcn, FormulaEncoder, RescoreHead

# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------


def train_step(spec_encoder, formula_encoder, rescore_head, loader, optimizer, device):
    formula_encoder.train()
    rescore_head.train()
    spec_encoder.eval()

    criterion = nn.BCELoss()
    total_loss = 0.0
    total_acc = 0
    y_true = []
    y_scores = []

    with tqdm(total=len(loader)) as bar:
        for batch in loader:
            _, x, env, f, y = batch
            x = x.to(device, dtype=torch.float32)
            env = env.to(device, dtype=torch.float32)
            f = f.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            env = env.clone()
            env[:, 0] = 0.0  # zero out precursor_mz to prevent mass-based shortcut

            optimizer.zero_grad()

            with torch.no_grad():
                z_specs, _, _, _, _ = spec_encoder(x, env)
                z_specs = F.normalize(z_specs, dim=1)

            z_forms = formula_encoder(f)  # (B, D), L2-normed
            interaction = z_specs * z_forms  # element-wise product (B, D)
            logits = rescore_head(interaction)  # (B,)
            y_hat = torch.sigmoid(logits)

            loss = criterion(y_hat, y)
            if torch.isnan(loss):
                bar.update(1)
                continue

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_acc += torch.eq(y.round(), y_hat.round()).float().sum().item()
            y_true.extend(y.cpu().numpy())
            y_scores.extend(y_hat.detach().cpu().numpy())

            bar.set_description("Train")
            bar.set_postfix(loss=loss.item())
            bar.update(1)

    auc = roc_auc_score(y_true, y_scores)
    return total_loss / len(loader.dataset), total_acc / len(loader.dataset), auc


def eval_step(spec_encoder, formula_encoder, rescore_head, loader, device):
    formula_encoder.eval()
    rescore_head.eval()
    spec_encoder.eval()

    criterion = nn.BCELoss()
    total_loss = 0.0
    total_acc = 0
    y_true = []
    y_scores = []

    with torch.no_grad():
        for batch in loader:
            _, x, env, f, y = batch
            x = x.to(device, dtype=torch.float32)
            env = env.to(device, dtype=torch.float32)
            f = f.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            env = env.clone()
            env[:, 0] = 0.0

            z_specs, _, _, _, _ = spec_encoder(x, env)
            z_specs = F.normalize(z_specs, dim=1)

            z_forms = formula_encoder(f)
            interaction = z_specs * z_forms
            logits = rescore_head(interaction)
            y_hat = torch.sigmoid(logits)

            loss = criterion(y_hat, y)
            total_loss += loss.item() * x.size(0)
            total_acc += torch.eq(y.round(), y_hat.round()).float().sum().item()
            y_true.extend(y.cpu().numpy())
            y_scores.extend(y_hat.detach().cpu().numpy())

    auc = roc_auc_score(y_true, y_scores)
    return total_loss / len(loader.dataset), total_acc / len(loader.dataset), auc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Siamese rescore model (interaction head + BCE)"
    )
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
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

    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(f"Config: {args.config_path}")

    device = (
        torch.device(f"cuda:{args.device[0]}")
        if torch.cuda.is_available() and not args.no_cuda
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    # Data
    train_set = RescoreDataset(args.train_data)
    train_loader = DataLoader(
        train_set,
        batch_size=config["train_rescore"]["batch_size"],
        shuffle=True,
        num_workers=config["train_rescore"]["num_workers"],
        drop_last=True,
    )
    valid_set = RescoreDataset(args.test_data)
    valid_loader = DataLoader(
        valid_set,
        batch_size=config["train_rescore"]["batch_size"],
        shuffle=False,
        num_workers=config["train_rescore"]["num_workers"],
        drop_last=True,
    )

    # Frozen spectrum encoder
    spec_encoder = MS2FNet_tcn(config["model"]).to(device)
    state_dict = torch.load(args.resume_path, map_location=device)["model_state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    spec_encoder.load_state_dict(state_dict, strict=False)
    for p in spec_encoder.parameters():
        p.requires_grad = False
    spec_encoder.eval()
    print(f"Loaded frozen spectrum encoder from {args.resume_path}")

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
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_auc = 0.0
    early_stop_patience = 0
    early_stop_step = config["train_rescore"]["early_stop_step"]

    for epoch in range(1, config["train_rescore"]["epochs"] + 1):
        print(f"\n=====Epoch {epoch}")
        train_loss, train_acc, train_auc = train_step(
            spec_encoder, formula_encoder, rescore_head, train_loader, optimizer, device
        )
        print(
            f"Train loss: {train_loss:.4f}  acc: {train_acc:.4f}  auc: {train_auc:.4f}"
        )

        valid_loss, valid_acc, valid_auc = eval_step(
            spec_encoder, formula_encoder, rescore_head, valid_loader, device
        )
        print(
            f"Validation loss: {valid_loss:.4f}  acc: {valid_acc:.4f}  auc: {valid_auc:.4f}"
        )

        if valid_auc > best_auc:
            best_auc = valid_auc
            torch.save(
                {
                    "epoch": epoch,
                    "formula_encoder_state_dict": formula_encoder.state_dict(),
                    "rescore_head_state_dict": rescore_head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_auc": best_auc,
                },
                args.checkpoint_path,
            )
            early_stop_patience = 0
            print("Saving checkpoint... Early stop patience reset")
        else:
            early_stop_patience += 1
            print(f"Early stop count: {early_stop_patience}/{early_stop_step}")

        scheduler.step(valid_auc)
        print(f"Best auc so far: {best_auc:.4f}")

        if early_stop_patience == early_stop_step:
            print("Early stop!")
            break

    print("Done!")
