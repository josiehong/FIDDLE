import os
import argparse
from tqdm import tqdm
import yaml
import time

import numpy as np
import pandas as pd
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import MGFDataset
from model_tcn import MS2FNet_tcn, FormulaEncoder, SiameseFDRHead
from utils import (
    formula_refinement,
    mass_calculator,
    vector_to_formula,
    formula_to_vector,
    formula_to_dict,
)


def test_step(model, loader, device):
    model.eval()
    spec_ids = []
    y_pred = []
    exp_precursor_mz = []
    exp_precursor_type = []
    mass_pred = []
    atomnum_pred = []
    hcnum_pred = []
    with tqdm(total=len(loader)) as bar:
        for _, batch in enumerate(loader):
            spec_id, exp_pre_type, x, env, neutral_add = batch
            x = x.to(device, dtype=torch.float32)
            env = env.to(device, dtype=torch.float32)
            neutral_add = neutral_add.to(device, dtype=torch.float32)
            exp_pre_mz = env[:, 0]

            with torch.no_grad():
                _, pred_f, pred_mass, pred_atomnum, pred_hcnum = model(x, env)
            pred_f = pred_f - neutral_add  # add the neutral adduct

            bar.set_description("Eval")
            bar.update(1)

            spec_ids = spec_ids + list(spec_id)
            y_pred.append(pred_f.detach().cpu())
            exp_precursor_mz.append(exp_pre_mz.detach().cpu())
            exp_precursor_type = exp_precursor_type + list(exp_pre_type)
            mass_pred.append(pred_mass.detach().cpu())
            atomnum_pred.append(pred_atomnum.detach().cpu())
            hcnum_pred.append(pred_hcnum.detach().cpu())

    y_pred = torch.cat(y_pred, dim=0)
    exp_precursor_mz = torch.cat(exp_precursor_mz, dim=0)
    mass_pred = torch.cat(mass_pred, dim=0)
    atomnum_pred = torch.cat(atomnum_pred, dim=0)
    hcnum_pred = torch.cat(hcnum_pred, dim=0)
    return (
        spec_ids,
        y_pred,
        exp_precursor_mz,
        exp_precursor_type,
        mass_pred,
        atomnum_pred,
        hcnum_pred,
    )


def rerank_by_siamese(
    spec_encoder, formula_encoder, fdr_head, spec, env, refined_results, device, K
):
    """Rerank candidates using the Siamese interaction head.

    Score = sigmoid(SiameseFDRHead(z_spec ⊙ FormulaEncoder(formula_vec))).
    Candidates are ranked by siamese score directly.
    """
    formula_encoder.eval()
    fdr_head.eval()
    spec_encoder.eval()

    refine_f = [f for f in refined_results["formula"] if f is not None]
    refine_m = [m for m in refined_results["mass"] if m is not None]
    if not refine_f:
        refined_results["siamese"] = [0.0] * K
        return refined_results

    f_vecs = torch.from_numpy(np.array([formula_to_vector(s) for s in refine_f]))
    spec_t = spec.to(device, dtype=torch.float32)
    env_t = env.to(device, dtype=torch.float32).clone()
    env_t[:, 0] = 0.0  # zero out precursor_mz to match training

    with torch.no_grad():
        z_spec, _, _, _, _ = spec_encoder(spec_t, env_t)
        z_spec = F.normalize(z_spec, dim=1)  # (1, D)
        z_spec_rep = z_spec.expand(len(refine_f), -1)  # (K, D)

        f_t = f_vecs.to(device, dtype=torch.float32)
        z_form = formula_encoder(f_t)  # (K, D)

        interaction = z_spec_rep * z_form  # (K, D)
        logits = fdr_head(interaction)  # (K,)
        siamese_scores = torch.sigmoid(logits).cpu().numpy()

    ranked = sorted(
        zip(siamese_scores, refine_f, refine_m),
        key=lambda x: x[0],
        reverse=True,
    )
    sorted_siamese, sorted_f, sorted_m = map(list, zip(*ranked))

    while len(sorted_f) < K:
        sorted_f.append(None)
        sorted_siamese.append(0.0)
        sorted_m.append(None)

    return {"formula": sorted_f, "mass": sorted_m, "siamese": sorted_siamese}


def init_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mass Spectra to formula (prediction)")
    parser.add_argument(
        "--test_data", type=str, required=True, help="Path to data (.mgf)"
    )
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to configuration (.yaml)"
    )
    parser.add_argument(
        "--resume_path", type=str, required=True, help="Path to pretrained TCN model"
    )
    parser.add_argument(
        "--fdr_resume_path",
        type=str,
        required=True,
        help="Path to pretrained Siamese FDR model",
    )
    parser.add_argument(
        "--buddy_path", type=str, default="", help="Path to saved BUDDY's results"
    )
    parser.add_argument(
        "--sirius_path", type=str, default="", help="Path to saved SIRIUS's results"
    )
    parser.add_argument(
        "--result_path", type=str, required=True, help="Path to save predicted results"
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

    # 1. Data
    valid_set = MGFDataset(args.test_data, config["encoding"])
    valid_loader = DataLoader(
        valid_set, batch_size=1, shuffle=False, num_workers=0, drop_last=True
    )

    # 2. Spectrum encoder (MS2FNet_tcn)
    model = MS2FNet_tcn(config["model"]).to(device_1st)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"# MS2FNet_tcn Params: {num_params}")
    if len(args.device) > 1:
        model = nn.DataParallel(model, device_ids=args.device)

    print("Loading the best formula prediction model...")
    state_dict = torch.load(args.resume_path, map_location=device_1st)[
        "model_state_dict"
    ]
    is_multi_gpu = any(key.startswith("module.") for key in state_dict.keys())
    if is_multi_gpu and len(args.device) == 1:
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_state_dict[key[7:] if key.startswith("module.") else key] = value
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    # 3. Siamese FDR model (FormulaEncoder + SiameseFDRHead)
    formula_encoder = FormulaEncoder(config["model"]).to(device_1st)
    fdr_head = SiameseFDRHead(config["model"]).to(device_1st)
    n_params = sum(p.numel() for p in formula_encoder.parameters()) + sum(
        p.numel() for p in fdr_head.parameters()
    )
    print(f"# Siamese FDR Params: {n_params}")

    print("Loading the best Siamese FDR model...")
    ckpt = torch.load(args.fdr_resume_path, map_location=device_1st)
    formula_encoder.load_state_dict(ckpt["formula_encoder_state_dict"])
    fdr_head.load_state_dict(ckpt["fdr_head_state_dict"])
    formula_encoder.eval()
    fdr_head.eval()

    # 4. Formula Prediction
    (
        spec_ids,
        y_pred,
        exp_precursor_mz,
        exp_precursor_type,
        mass_pred,
        atomnum_pred,
        hcnum_pred,
    ) = test_step(model, valid_loader, device_1st)

    prediction_time = time.time() - start_time
    prediction_time /= len(valid_set)

    formula_pred = [vector_to_formula(y) for y in y_pred]
    y_pred = [";".join(y) for y in y_pred.numpy().astype("str")]

    spectra = []
    environments = []
    for batch in valid_loader:
        _, _, spec, env, _ = batch
        spectra.append(spec)
        environments.append(env)

    # 5. Post-processing
    if args.buddy_path != "":
        buddy_df = pd.read_csv(args.buddy_path)
    if args.sirius_path != "":
        sirius_df = pd.read_csv(args.sirius_path)

    formula_redined = {
        "Refined Formula ({})".format(str(k)): []
        for k in range(config["post_processing"]["top_k"])
    }
    mass_redined = {
        "Refined Mass ({})".format(str(k)): []
        for k in range(config["post_processing"]["top_k"])
    }
    siamese_refined = {
        "Siamese ({})".format(str(k)): []
        for k in range(config["post_processing"]["top_k"])
    }
    running_time = []
    exp_mass = []

    for idx, pred_f, exp_pre_mz, exp_pre_type, spec, env in tqdm(
        zip(
            spec_ids,
            formula_pred,
            exp_precursor_mz,
            exp_precursor_type,
            spectra,
            environments,
        ),
        total=len(exp_precursor_mz),
        desc="Post",
    ):
        m = mass_calculator(exp_pre_type, exp_pre_mz)
        exp_mass.append(m.item())

        f0_list = [pred_f]
        if args.buddy_path != "" and len(buddy_df.loc[buddy_df["ID"] == idx]) > 0:
            buddy_f = (
                buddy_df.loc[buddy_df["ID"] == idx]
                .iloc[0][
                    [
                        "Pred Formula (1)",
                        "Pred Formula (2)",
                        "Pred Formula (3)",
                        "Pred Formula (4)",
                        "Pred Formula (5)",
                    ]
                ]
                .tolist()
            )
            buddy_fdr = (
                buddy_df.loc[buddy_df["ID"] == idx]
                .iloc[0][
                    [
                        "BUDDY Score (1)",
                        "BUDDY Score (2)",
                        "BUDDY Score (3)",
                        "BUDDY Score (4)",
                        "BUDDY Score (5)",
                    ]
                ]
                .tolist()
            )
            buddy_f = [
                x
                for x, fdr in zip(buddy_f, buddy_fdr)
                if str(x) != "nan" and fdr < config["post_processing"]["buddy_fdr_thr"]
            ]
            f0_list.extend(buddy_f)
        if args.sirius_path != "":
            sirius_f = (
                sirius_df.loc[sirius_df["ID"] == idx]
                .iloc[0][
                    [
                        "Pred Formula (1)",
                        "Pred Formula (2)",
                        "Pred Formula (3)",
                        "Pred Formula (4)",
                        "Pred Formula (5)",
                    ]
                ]
                .tolist()
            )
            sirius_score = (
                sirius_df.loc[sirius_df["ID"] == idx]
                .iloc[0][
                    [
                        "SIRIUS Score (1)",
                        "SIRIUS Score (2)",
                        "SIRIUS Score (3)",
                        "SIRIUS Score (4)",
                        "SIRIUS Score (5)",
                    ]
                ]
                .tolist()
            )
            sirius_f = [
                x
                for x, score in zip(sirius_f, sirius_score)
                if str(x) != "nan"
                and score > config["post_processing"]["sirius_score_thr"]
            ]
            f0_list.extend(sirius_f)

        f0_list = list(set(f0_list))
        refine_atom_type = list(config["post_processing"]["refine_atom_type"])
        refine_atom_num = list(config["post_processing"]["refine_atom_num"])
        for f0 in f0_list:
            for atom, cnt in formula_to_dict(f0).items():
                if atom == "H" or atom in refine_atom_type:
                    continue
                refine_atom_type.append(atom)
                refine_atom_num.append(max(1, int(cnt)))

        start_time = time.time()
        refined_results = formula_refinement(
            f0_list,
            m.item(),
            config["post_processing"]["mass_tolerance"],
            config["post_processing"]["ppm_mode"],
            config["post_processing"]["top_k"],
            config["post_processing"]["maxium_miss_atom_num"],
            config["post_processing"]["time_out"],
            refine_atom_type,
            refine_atom_num,
        )

        refined_results = rerank_by_siamese(
            model,
            formula_encoder,
            fdr_head,
            spec,
            env,
            refined_results,
            device_1st,
            config["post_processing"]["top_k"],
        )

        for i, (refined_f, refined_m, refined_s) in enumerate(
            zip(
                refined_results["formula"],
                refined_results["mass"],
                refined_results["siamese"],
            )
        ):
            formula_redined[f"Refined Formula ({i})"].append(refined_f)
            mass_redined[f"Refined Mass ({i})"].append(refined_m)
            siamese_refined[f"Siamese ({i})"].append(refined_s)
        refinement_time = time.time() - start_time
        running_time.append(prediction_time + refinement_time)

    # 6. Save the final results
    print("\nSave the predicted results...")
    out_dict = {
        "ID": spec_ids,
        "Y Pred": y_pred,
        "Mass": exp_mass,
        "Pred Formula": formula_pred,
        "Pred Mass": mass_pred.tolist(),
        "Pred Atom Num": atomnum_pred.tolist(),
        "Pred H/C Num": hcnum_pred.tolist(),
        "Running Time": running_time,
    }
    res_df = pd.DataFrame(
        {**out_dict, **formula_redined, **mass_redined, **siamese_refined}
    )
    res_df.to_csv(args.result_path, index=False)
    print("Done!")
