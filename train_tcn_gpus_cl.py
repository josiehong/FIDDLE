import argparse
from tqdm import tqdm
import yaml
import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, RandomSampler
from torch.optim.lr_scheduler import _LRScheduler

from dataset import MS2FDataset_CL, MS2FDataset
from model_tcn import MS2FNet_tcn
from utils import ATOMS_WEIGHT, ATOMS_INDEX, ATOMS_INDEX_re, vector_to_formula


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, z1, z2, label):
        euclidean_distance = F.pairwise_distance(z1, z2)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2)
            + label
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive


class WarmUpScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, warmup_start_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            progress = self.last_epoch / max(self.warmup_steps, 1)
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * progress
                for base_lr in self.base_lrs
            ]
        return self.base_lrs


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def formula_criterion(outputs, targets, weights):
    # weighted
    outputs = torch.mul(outputs, weights)
    targets = torch.mul(targets, weights)
    # mse loss
    t = nn.MSELoss()
    loss = t(outputs.float(), targets.float())
    return loss.to(torch.float32)


def mse_criterion(outputs, targets):
    t = nn.MSELoss()
    loss = t(outputs.float(), targets.float())
    return loss.to(torch.float32)


def train_step(
    model,
    loader,
    optimizer,
    warmup_scheduler,
    warmup_steps,  # warmup_scheduler is updated each step
    device,
    batch_size,
    weight,
):

    mae = []
    mass_mae = []
    atomnum_mae = []
    hcnum_mae = []
    record_loss = []
    cl_criterion = ContrastiveLoss(margin=1.0)

    with tqdm(total=len(loader)) as bar:
        for step, batch in enumerate(loader):
            _, y, x, mass, env, _, y2, x2, mass2, env2, label = batch
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            env = env.to(device, dtype=torch.float32)
            mass = mass.to(device, dtype=torch.float32)
            atomnum = torch.sum(y, dim=1)
            hcnum = y[:, 1] / y[:, 0].clamp(min=1)

            x2 = x2.to(device, dtype=torch.float32)
            y2 = y2.to(device, dtype=torch.float32)
            env2 = env2.to(device, dtype=torch.float32)
            mass2 = mass2.to(device, dtype=torch.float32)
            atomnum2 = torch.sum(y2, dim=1)
            hcnum2 = y2[:, 1] / y2[:, 0].clamp(min=1)

            label = label.to(device, dtype=torch.float32)

            # -------------------------
            # train MS2FNet_tcn (generator)
            # -------------------------
            model.zero_grad()
            z1, pred_f, pred_mass, pred_atomnum, pred_hcnum = model(x, env)
            z2, pred_f2, pred_mass2, pred_atomnum2, pred_hcnum2 = model(x2, env2)

            loss = (
                mse_criterion(mass, pred_mass) * 0.01
                + mse_criterion(atomnum, pred_atomnum)
                + mse_criterion(hcnum, pred_hcnum) * 10
                + formula_criterion(pred_f, y, weight)
                + mse_criterion(mass2, pred_mass2) * 0.01
                + mse_criterion(atomnum2, pred_atomnum2)
                + mse_criterion(hcnum2, pred_hcnum2) * 10
                + formula_criterion(pred_f2, y2, weight)
                + cl_criterion(F.normalize(z1, dim=1), F.normalize(z2, dim=1), label)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update the learning rate during warm-up phase
            # After warm-up phase, the learning rate ctrled by warmup_scheduler is fixed and we won't use it
            if warmup_scheduler.last_epoch < warmup_steps:
                warmup_scheduler.step()
                # print('Warmup lr: {:.6f}'.format(get_lr(optimizer))) # Josie: for debug

            # (tmp) for debug
            mae = (
                mae
                + torch.mean(torch.abs(y - pred_f), dim=1).tolist()
                + torch.mean(torch.abs(y2 - pred_f2), dim=1).tolist()
            )
            mass_mae = (
                mass_mae
                + torch.abs(mass - pred_mass).tolist()
                + torch.abs(mass2 - pred_mass2).tolist()
            )
            atomnum_mae = (
                atomnum_mae
                + torch.abs(atomnum - pred_atomnum).tolist()
                + torch.abs(atomnum2 - pred_atomnum2).tolist()
            )
            hcnum_mae = (
                hcnum_mae
                + torch.abs(hcnum - pred_hcnum).tolist()
                + torch.abs(hcnum2 - pred_hcnum2).tolist()
            )
            record_loss.append(loss.item())

            bar.set_description("Train")
            bar.set_postfix(lr=get_lr(optimizer), loss=loss.item())
            bar.update(1)

    return (mae, mass_mae, atomnum_mae, hcnum_mae, record_loss)


def eval_step(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    spec_ids = []
    mae = []
    mass_true = []
    mass_pred = []
    mass_mae = []
    with tqdm(total=len(loader)) as bar:
        for _, batch in enumerate(loader):
            spec_id, y, x, mass, env = batch
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            env = env.to(device, dtype=torch.float32)
            mass = mass.to(device, dtype=torch.float32)

            with torch.no_grad():
                _, pred_f, pred_mass, _, _ = model(x, env)

            bar.set_description("Eval")
            bar.update(1)

            y_true.append(y.detach().cpu())
            y_pred.append(pred_f.detach().cpu())

            mae = mae + torch.mean(torch.abs(y - pred_f), dim=1).tolist()
            spec_ids = spec_ids + list(spec_id)

            mass_true.append(mass.detach().cpu())
            mass_pred.append(pred_mass.detach().cpu())
            mass_mae = mass_mae + torch.abs(mass - pred_mass).tolist()

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    mass_true = torch.cat(mass_true, dim=0)
    mass_pred = torch.cat(mass_pred, dim=0)
    return spec_ids, y_true, y_pred, mae, mass_true, mass_pred, mass_mae


def init_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="Mass Spectra to formula (train)")
    parser.add_argument(
        "--train_data", type=str, required=True, help="Path to data (.pkl)"
    )
    parser.add_argument(
        "--test_data", type=str, required=True, help="Path to data (.pkl)"
    )
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to configuration (.yaml)"
    )

    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to save checkpoint"
    )
    parser.add_argument(
        "--resume_path", type=str, default="", help="Path to pretrained model"
    )
    parser.add_argument("--transfer", action="store_true", default=False)

    parser.add_argument(
        "--ex_model_path",
        type=str,
        default="",
        help="Path to export the whole model (structure & weights)",
    )
    parser.add_argument(
        "--result_path", type=str, default="", help="Path to save predicted results"
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

    # predifined tensors:
    weight = torch.tensor(
        [ATOMS_WEIGHT[ATOMS_INDEX_re[i]] for i in range(len(ATOMS_INDEX_re))]
    ).to(device_1st)

    # 1. Data
    train_set = MS2FDataset_CL(
        args.train_data,
        noised_times=config["train"]["noised_times"],
        padding_dim=config["model"]["padding_dim"],
    )
    train_loader = DataLoader(
        train_set,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["train"]["num_workers"],
        drop_last=True,
    )
    valid_set = MS2FDataset(args.test_data, padding_dim=config["model"]["padding_dim"])
    valid_loader = DataLoader(
        valid_set,
        batch_size=int(config["train"]["batch_size"] / 4),
        shuffle=False,
        num_workers=config["train"]["num_workers"],
        drop_last=True,
    )

    # 2. Models
    model = MS2FNet_tcn(config["model"]).to(device_1st)
    num_params = sum(p.numel() for p in model.parameters())
    # print(f'{str(model)} #Params: {num_params}')
    print(f"# MS2FNet_tcn Params: {num_params}")

    if len(args.device) > 1:  # Wrap the model with nn.DataParallel
        model = nn.DataParallel(model, device_ids=args.device)

    # 4. Train MS2FNet_tcn & FNet
    # define the hyperparameters
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=config['train']['patience'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.5)

    # load the checkpoints
    if args.resume_path != "" and not args.transfer:  # start from a checkpoint
        print("Load the checkpoints of the whole model")
        ckpt = torch.load(args.resume_path, map_location=device_1st)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        epoch_start = ckpt["epoch"]
        best_valid_mae = ckpt["best_val_mae"]
        best_formula_acc = ckpt["best_val_acc"]
        best_formula_wo_acc = ckpt["best_val_wo_acc"]
        warmup_steps = 0  # do not use the warm-up scheduler when resuming

    elif args.resume_path != "" and args.transfer:  # transfer learning
        print("Transfer learning from the pretrained model")
        model.load_state_dict(
            torch.load(args.resume_path, map_location=device_1st)["model_state_dict"]
        )

        epoch_start = 0
        best_valid_mae = 9999
        best_formula_acc = 0
        best_formula_wo_acc = 0
        warmup_steps = int(
            len(train_set)
            / config["train"]["batch_size"]
            * config["train"]["warmup_ratio"]
        )
    else:
        epoch_start = 0
        best_valid_mae = 9999
        best_formula_acc = 0
        best_formula_wo_acc = 0
        warmup_steps = int(
            len(train_set)
            / config["train"]["batch_size"]
            * config["train"]["warmup_ratio"]
        )  # 10% of the total steps in the first epoch

    # Create the warm-up scheduler
    warmup_start_lr = 1e-6
    warmup_scheduler = WarmUpScheduler(optimizer, warmup_steps, warmup_start_lr)
    print(
        "Warm-up steps: {}, Warm-up start lr: {}".format(warmup_steps, warmup_start_lr)
    )

    # train!
    early_stop_patience = 0
    Tensor = (
        torch.cuda.FloatTensor
        if torch.cuda.is_available() and not args.no_cuda
        else torch.FloatTensor
    )
    for epoch in range(epoch_start + 1, config["train"]["epochs"] + 1):
        print("\n=====Epoch {}".format(epoch))

        train_mae, train_mass_mae, train_atomnum_mae, train_hcnum_mae, train_loss = (
            train_step(
                model,
                train_loader,
                optimizer,
                warmup_scheduler,
                warmup_steps,
                device_1st,
                batch_size=config["train"]["batch_size"],
                weight=weight,
            )
        )
        print(
            "Train error: {:.4f} mass error: {:.4f} atom number error: {:.4f} H/C number error: {:.4f} loss: {:.4f}".format(
                np.mean(train_mae),
                np.mean(train_mass_mae),
                np.mean(train_atomnum_mae),
                np.mean(train_hcnum_mae),
                np.mean(train_loss),
            )
        )

        spec_ids, y_true, y_pred, mae, mass_true, mass_pred, mass_mae = eval_step(
            model, valid_loader, device_1st
        )
        valid_mae = np.mean(mae)
        valid_mass_mae = np.mean(mass_mae)
        print(
            "Validation error: {:.4f} mass error: {:.4f}".format(
                valid_mae, valid_mass_mae
            )
        )

        # calculate the formula (tmp)
        formula_true = [vector_to_formula(y) for y in y_true]
        formula_pred = [vector_to_formula(y) for y in y_pred]
        print("Exp samples: {}, \nmass: {}".format(formula_true[:5], mass_true[:5]))
        print("Pred samples: {}, \nmass: {}".format(formula_pred[:5], mass_pred[:5]))
        formula_acc = sum(
            [1 for f1, f2 in zip(formula_true, formula_pred) if f1 == f2]
        ) / len(formula_pred)
        print("formula w acc: {:.4f}".format(formula_acc))
        # calculate the formula without H (tmp)
        formula_wo_true = [vector_to_formula(y, withH=False) for y in y_true]
        formula_wo_pred = [vector_to_formula(y, withH=False) for y in y_pred]
        formula_wo_acc = sum(
            [1 for f1, f2 in zip(formula_wo_true, formula_wo_pred) if f1 == f2]
        ) / len(formula_wo_pred)
        print("formula w/o acc: {:.4f}".format(formula_wo_acc))

        # Save checkpoint only when formula accuracy (with H) improves — single primary metric
        if best_formula_acc < formula_acc:
            best_formula_acc = formula_acc
            best_valid_mae = valid_mae
            best_formula_wo_acc = formula_wo_acc

            if args.checkpoint_path != "":
                print("Saving checkpoint...")
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "num_params": num_params,
                    "best_val_mae": best_valid_mae,
                    "best_val_acc": best_formula_acc,
                    "best_val_wo_acc": best_formula_wo_acc,
                }
                torch.save(checkpoint, args.checkpoint_path)

            early_stop_patience = 0
            print("Early stop patience reset")
        else:
            early_stop_patience += 1
            print(
                "Early stop count: {}/{}".format(
                    early_stop_patience, config["train"]["early_stop_step"]
                )
            )

        # Update the cosine annealing scheduler after the warmup phase
        if epoch > warmup_steps // len(train_loader):
            # scheduler.step(valid_mae) # ReduceLROnPlateau
            scheduler.step()  # StepLR
        print(f"Best absolute error so far: {best_valid_mae}")
        print(f"Best formula accuracy with H so far: {best_formula_acc}")
        print(f"Best formula accuracy without H so far: {best_formula_wo_acc}")

        if early_stop_patience == config["train"]["early_stop_step"]:
            print("Early stop!")
            break

    # 4. Export the model
    if args.ex_model_path != "":
        print("Export the model...")
        model_scripted = torch.jit.script(model)  # Export to TorchScript
        model_scripted.save(args.ex_model_path)  # Save

    # 5. Output the prediction results
    if args.result_path != "":
        print("Loading the best model...")
        model.load_state_dict(
            torch.load(args.checkpoint_path, map_location=device_1st)[
                "model_state_dict"
            ]
        )
        spec_ids, y_true, y_pred, mae, mass_true, mass_pred, mass_mae = eval_step(
            model, valid_loader, device_1st
        )
        valid_mae = np.mean(mae)
        valid_mass_mae = np.mean(mass_mae)
        print(
            "Validation error: {:.4f} mass error: {:.4f}".format(
                valid_mae, valid_mass_mae
            )
        )
        # calculate the formula
        formula_true = [vector_to_formula(y) for y in y_true]
        formula_pred = [vector_to_formula(y) for y in y_pred]

        # y_true = [','.join(y) for y in y_true.numpy().astype('str')]
        y_pred = [",".join(y) for y in y_pred.numpy().astype("str")]

        print("Save the predicted results...")
        out_dict = {
            "ID": spec_ids,
            "Y Pred": y_pred,
            "Formula": formula_true,
            "Pred Formula": formula_pred,
            "Mass": mass_true.tolist(),
            "Pred Mass": mass_pred.tolist(),
        }
        res_df = pd.DataFrame(out_dict)
        res_df.to_csv(args.result_path, sep="\t")

    print("Done!")
