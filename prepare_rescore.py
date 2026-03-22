"""
Prepare rescore training data.
- Correct predicted formula (label=1)
- Incorrect predicted formula (label=0)
"""

import os
import pickle
import argparse
from tqdm import tqdm
import yaml
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler

from dataset import MS2FDataset
from model_tcn import MS2FNet_tcn
from utils import ATOMS_INDEX_re, formula_to_dict, formula_refinement


def convert_to_list_of_dicts(data_dict):
    keys = list(data_dict.keys())
    num_items = len(data_dict[keys[0]])
    result = []

    for i in range(num_items):
        item_dict = {}
        for key in keys:
            item_dict[key] = data_dict[key][i]
        result.append(item_dict)

    return result


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
            x = x.to(device).to(torch.float32)
            y = y.to(device).to(torch.float32)
            env = env.to(device).to(torch.float32)
            mass = mass.to(device).to(torch.float32)

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


def split_indices_by_compound(data, train_ratio, seed):
    """Split dataset indices by compound (SMILES) so no compound appears in both splits.

    Args:
        data: list of data dicts, each with a 'smiles' key.
        train_ratio: fraction of compounds to assign to the training split.
        seed: random seed for reproducibility.

    Returns:
        train_indices, test_indices: lists of integer indices into data.
    """
    rng = np.random.default_rng(seed)

    # Map each unique SMILES to its spectrum indices
    smiles_to_indices = {}
    for i, item in enumerate(data):
        smiles = item["smiles"]
        smiles_to_indices.setdefault(smiles, []).append(i)

    unique_smiles = np.array(list(smiles_to_indices.keys()))
    rng.shuffle(unique_smiles)

    n_train = int(len(unique_smiles) * train_ratio)
    train_smiles = set(unique_smiles[:n_train])

    train_indices, test_indices = [], []
    for smiles, indices in smiles_to_indices.items():
        if smiles in train_smiles:
            train_indices.extend(indices)
        else:
            test_indices.extend(indices)

    print(
        f"Compound split: {len(train_smiles)} train compounds ({len(train_indices)} spectra) / "
        f"{len(unique_smiles) - len(train_smiles)} test compounds ({len(test_indices)} spectra)"
    )
    return train_indices, test_indices


def prepare_rescore_split(
    data_path, indices, model, device, config, out_path, pkl_data
):
    """Run model inference and post-processing on a subset of data, then save rescore pkl.

    Args:
        data_path: path to the source pkl (used only to build the full dataset).
        indices: integer indices of spectra to include in this split.
        model: loaded MS2FNet_tcn model.
        device: torch device.
        config: parsed YAML config dict.
        out_path: output path for the rescore pkl.
        pkl_data: dict mapping title -> [spec, env] for fast lookup.
    """
    dataset = MS2FDataset(data_path)

    sampler = SubsetRandomSampler(indices)
    loader = DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        sampler=sampler,
        num_workers=config["train"]["num_workers"],
        drop_last=False,
    )

    # Prediction
    spec_ids, y_true, y_pred, mae, mass_true, mass_pred, mass_mae = eval_step(
        model, loader, device
    )

    formula_pred = [vec2formula(y) for y in y_pred]
    formula_true = [vec2formula(y) for y in y_true]

    # Post-processing
    formula_redined = {
        "Refined Formula ({})".format(str(k)): []
        for k in range(config["post_processing"]["top_k"])
    }
    for pred_f, m in tqdm(
        zip(formula_pred, mass_true), total=len(mass_true), desc="Post"
    ):
        refine_atom_type = list(config["post_processing"]["refine_atom_type"])
        refine_atom_num = list(config["post_processing"]["refine_atom_num"])
        for atom, cnt in formula_to_dict(pred_f).items():
            if atom == "H" or atom in refine_atom_type:
                continue
            refine_atom_type.append(atom)
            refine_atom_num.append(max(1, int(cnt)))

        refined_results = formula_refinement(
            [pred_f],
            m.item(),
            config["post_processing"]["mass_tolerance"],
            config["post_processing"]["ppm_mode"],
            config["post_processing"]["top_k"],
            config["post_processing"]["maxium_miss_atom_num"],
            config["post_processing"]["time_out"],
            refine_atom_type,
            refine_atom_num,
        )

        for i, (refined_f, refined_m) in enumerate(
            zip(refined_results["formula"], refined_results["mass"])
        ):
            formula_redined["Refined Formula ({})".format(str(i))].append(refined_f)

    # Label rescore data
    info_dict = {"ID": spec_ids, "Formula": formula_true}
    res_df = pd.DataFrame({**info_dict, **formula_redined})

    data = {"title": [], "pred_formula": [], "spec": [], "env": [], "label": []}
    for k in range(config["post_processing"]["top_k"]):
        res_df["Label ({})".format(str(k))] = res_df.apply(
            lambda x: formula_to_dict(x["Formula"])
            == formula_to_dict(x["Refined Formula ({})".format(str(k))]),
            axis=1,
        )

        correct_df = res_df[res_df["Label ({})".format(str(k))] == True]
        correct_df = correct_df.dropna(subset=["Refined Formula ({})".format(str(k))])
        titles = correct_df["ID"].tolist()
        data["title"].extend(titles)
        data["pred_formula"].extend(
            correct_df["Refined Formula ({})".format(str(k))].tolist()
        )
        data["label"].extend([1.0] * len(titles))
        for title in titles:
            spec, env = pkl_data[title]
            data["spec"].append(spec)
            data["env"].append(env)
        print(k, "correct", len(titles))

        incorrect_df = res_df[res_df["Label ({})".format(str(k))] == False]
        incorrect_df = incorrect_df.dropna(
            subset=["Refined Formula ({})".format(str(k))]
        )
        titles = incorrect_df["ID"].tolist()
        data["title"].extend(titles)
        data["pred_formula"].extend(
            incorrect_df["Refined Formula ({})".format(str(k))].tolist()
        )
        data["label"].extend([0.0] * len(titles))
        for title in titles:
            spec, env = pkl_data[title]
            data["spec"].append(spec)
            data["env"].append(env)
        print(k, "incorrect", len(titles))

    print("\nSave the rescore dataset...")
    with open(out_path, "wb") as f:
        data = convert_to_list_of_dicts(data)
        pickle.dump(data, f)
        print("Save {} rescore data to {}".format(len(data), out_path))
    print("Done!")


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(
        description="Preprocess the data for rescore model training"
    )
    parser.add_argument(
        "--test_data", type=str, required=True, help="Path to test data (.pkl)"
    )
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to configuration (.yaml)"
    )
    parser.add_argument(
        "--resume_path", type=str, required=True, help="Path to pretrained model"
    )
    parser.add_argument(
        "--rescore_dir", type=str, required=True, help="Path to save rescore dataset"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of compounds used for rescore training split (default: 0.8)",
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
    parser.add_argument("--no_cuda", action="store_true", help="Disables CUDA")
    args = parser.parse_args()

    init_random_seed(args.seed)
    start_time = time.time()

    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print("Load the model & training configuration from {}".format(args.config_path))

    device_1st = (
        torch.device("cuda:" + str(args.device[0]))
        if torch.cuda.is_available() and not args.no_cuda
        else torch.device("cpu")
    )
    print(f"Device(s): {args.device}")

    # 1. Model
    model = MS2FNet_tcn(config["model"]).to(device_1st)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"# MS2FNet_tcn Params: {num_params}")

    if len(args.device) > 1:
        model = nn.DataParallel(model, device_ids=args.device)

    print("Loading the best model...")
    model.load_state_dict(
        torch.load(args.resume_path, map_location=device_1st)["model_state_dict"]
    )

    # 2. Load test data and split by compound (SMILES)
    print(f"Loading test data from {args.test_data}")
    with open(args.test_data, "rb") as f:
        raw_data = pickle.load(f)
    print(f"Loaded {len(raw_data)} data items")

    train_indices, test_indices = split_indices_by_compound(
        raw_data, args.train_ratio, args.seed
    )

    # Build title -> [spec, env] lookup for rescore labelling
    pkl_data = {d["title"]: [d["spec"], d["env"]] for d in raw_data}

    # 3. Output paths
    rescore_train_path = args.test_data.replace("_test.pkl", "_rescore_train.pkl")
    rescore_test_path = args.test_data.replace("_test.pkl", "_rescore_test.pkl")

    # 4. Prepare rescore splits
    for split_name, indices, out_path in [
        ("train", train_indices, rescore_train_path),
        ("test", test_indices, rescore_test_path),
    ]:
        print(f"\n===== Rescore {split_name} split ({len(indices)} spectra) =====")
        prepare_rescore_split(
            args.test_data, indices, model, device_1st, config, out_path, pkl_data
        )
