import argparse
from tqdm import tqdm
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from dataset import FDRDataset
from model_tcn import FDRNet


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train_step(model, loader, optimizer, device):
    model.train()
    criterion = nn.BCELoss()

    total_loss = 0
    total_acc = 0
    y_true = []
    y_scores = []
    with tqdm(total=len(loader)) as bar:
        for step, batch in enumerate(loader):
            _, x, env, f, y = batch
            x = x.to(device, dtype=torch.float32)
            env = env.to(device, dtype=torch.float32)
            f = f.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            y_hat = model(x, env, f)
            y_hat = torch.sigmoid(y_hat)

            # loss = criterion(y, y_hat) # wrong
            loss = criterion(y_hat, y)  # correct
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected during training.")
                continue

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_acc += torch.eq(y.round(), y_hat.round()).float().sum().item()

            y_true.extend(y.cpu().numpy())
            y_scores.extend(y_hat.detach().cpu().numpy())

            bar.set_description("Train")
            bar.set_postfix(lr=get_lr(optimizer), loss=loss.item())
            bar.update(1)

    auc_score = roc_auc_score(y_true, y_scores)
    return total_loss / len(loader.dataset), total_acc / len(loader.dataset), auc_score


def eval_step(model, loader, device):
    model.eval()
    criterion = nn.BCELoss()

    total_loss = 0
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

            y_hat = model(x, env, f)
            y_hat = torch.sigmoid(y_hat)

            loss = criterion(y_hat, y)
            total_loss += loss.item() * x.size(0)
            total_acc += torch.eq(y.round(), y_hat.round()).float().sum().item()

            y_true.extend(y.cpu().numpy())
            y_scores.extend(y_hat.detach().cpu().numpy())

    auc_score = roc_auc_score(y_true, y_scores)
    return total_loss / len(loader.dataset), total_acc / len(loader.dataset), auc_score


def init_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(
        description="Mass Spectra and formula to FDR (train)"
    )
    parser.add_argument(
        "--train_data", type=str, required=True, help="Path to training data (.pkl)"
    )
    parser.add_argument(
        "--test_data", type=str, required=True, help="Path to test data (.pkl)"
    )
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to configuration (.yaml)"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to save checkpoint"
    )
    parser.add_argument(
        "--transfer", action="store_true", help="Whether to load the pretrained encoder"
    )
    parser.add_argument(
        "--resume_path", type=str, default="", help="Path to pretrained model"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for random functions"
    )
    parser.add_argument(
        "--device",
        type=int,
        nargs="+",
        default=[0],
        help="Which GPUs to use if any (default: [0]). Accepts multiple values separated by space.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", default=False, help="Disables CUDA training"
    )
    args = parser.parse_args()

    init_random_seed(args.seed)
    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print("Load the model & training configuration from {}".format(args.config_path))

    device_1st = (
        torch.device("cuda:" + str(args.device[0]))
        if torch.cuda.is_available() and not args.no_cuda
        else torch.device("cpu")
    )
    print(f"Device(s): {args.device}")

    # 1. Data
    train_set = FDRDataset(args.train_data)
    train_loader = DataLoader(
        train_set,
        batch_size=config["train_fdr"]["batch_size"],
        shuffle=True,
        num_workers=config["train_fdr"]["num_workers"],
        drop_last=True,
    )
    valid_set = FDRDataset(args.test_data)
    valid_loader = DataLoader(
        valid_set,
        batch_size=config["train_fdr"]["batch_size"],
        shuffle=False,
        num_workers=config["train_fdr"]["num_workers"],
        drop_last=True,
    )

    # 2. Model
    model = FDRNet(config["model"]).to(device_1st)
    num_params = sum(p.numel() for p in model.parameters())
    # print(f'{str(model)} #Params: {num_params}')
    print(f"# FDRNet Params: {num_params}")

    if len(args.device) > 1:  # Wrap the model with nn.DataParallel
        model = nn.DataParallel(model, device_ids=args.device)
    # need to do something when using one GPU

    # 3. Train FDRNet
    # Define the hyperparameters
    optimizer = optim.AdamW(model.parameters(), lr=config["train_fdr"]["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # Load the checkpoints
    if args.transfer and args.resume_path != "":
        print("Load the pretrained encoder (freeze the encoder)")
        state_dict = torch.load(args.resume_path, map_location=device_1st)[
            "model_state_dict"
        ]
        encoder_dict = {}
        for name, param in state_dict.items():
            if name.startswith("encoder"):
                param.requires_grad = False  # freeze the encoder
                encoder_dict[name] = param
        model.load_state_dict(encoder_dict, strict=False)

        epoch_start = 0
        best_valid_auc = 0
    elif args.resume_path:
        print("Load the checkpoints of the whole model")
        checkpoint = torch.load(args.resume_path, map_location=device_1st)
        epoch_start = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_valid_auc = checkpoint["best_val_auc"]
    else:
        epoch_start = 0
        best_valid_auc = 0

    # Train
    early_stop_patience = 0
    for epoch in range(epoch_start + 1, config["train_fdr"]["epochs"] + 1):
        print(f"\n=====Epoch {epoch}")
        train_loss, train_acc, train_auc = train_step(
            model, train_loader, optimizer, device_1st
        )
        print(f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} auc: {train_auc:.4f}")

        valid_loss, valid_acc, valid_auc = eval_step(model, valid_loader, device_1st)
        print(
            f"Validation loss: {valid_loss:.4f} acc: {valid_acc:.4f} auc: {valid_auc:.4f}"
        )

        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc

            if args.checkpoint_path:
                print("Saving checkpoint...")
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "num_params": num_params,
                    "best_val_auc": best_valid_auc,
                }
                torch.save(checkpoint, args.checkpoint_path)

            early_stop_patience = 0
            print("Early stop patience reset")
        else:
            early_stop_patience += 1
            print(
                f'Early stop count: {early_stop_patience}/{config["train_fdr"]["early_stop_step"]}'
            )

        scheduler.step(valid_auc)  # ReduceLROnPlateau
        print(f"Best auc so far: {best_valid_auc}")

        if early_stop_patience == config["train_fdr"]["early_stop_step"]:
            print("Early stop!")
            break

    print("Done!")
