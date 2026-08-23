import os
import argparse
import sys

from .download import get_checkpoint_dir
from .api import MsFiddlePredictor


def _checkpoint_error_message(missing_paths):
    missing = "\n".join(f"  - {path}" for path in missing_paths)
    return (
        "Required msfiddle checkpoint file(s) were not found:\n"
        f"{missing}\n\n"
        "Download the pre-trained checkpoints before running predictions:\n"
        "  msfiddle-download-models\n\n"
        "To inspect checkpoint locations, run:\n"
        "  msfiddle-checkpoint-paths"
    )


def validate_checkpoint_paths(resume_path, rescore_resume_path):
    missing_paths = [
        path for path in (resume_path, rescore_resume_path) if not os.path.exists(path)
    ]
    if missing_paths:
        raise FileNotFoundError(_checkpoint_error_message(missing_paths))


def test_step(model, loader, device):
    import torch
    from tqdm import tqdm

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


def rescore_candidates(
    spec_encoder, formula_encoder, rescore_head, spec, env, refined_results, device, K
):
    """Rescore candidates using the Siamese interaction head.

    Score = sigmoid(RescoreHead(z_spec ⊙ FormulaEncoder(formula_vec))).
    Candidates are ranked by rescore score directly.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    from .utils.mol_utils import formula_to_vector

    formula_encoder.eval()
    rescore_head.eval()
    spec_encoder.eval()

    refine_f = [f for f in refined_results["formula"] if f is not None]
    refine_m = [m for m in refined_results["mass"] if m is not None]
    if not refine_f:
        refined_results["rescore"] = [0.0] * K
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
        logits = rescore_head(interaction)  # (K,)
        rescore_scores = torch.sigmoid(logits).cpu().numpy()

    ranked = sorted(
        zip(rescore_scores, refine_f, refine_m),
        key=lambda x: x[0],
        reverse=True,
    )
    sorted_rescore, sorted_f, sorted_m = map(list, zip(*ranked))

    while len(sorted_f) < K:
        sorted_f.append(None)
        sorted_rescore.append(0.0)
        sorted_m.append(None)

    return {"formula": sorted_f, "mass": sorted_m, "rescore": sorted_rescore}


def init_random_seed(seed):
    import numpy as np

    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return


def main():
    parser = argparse.ArgumentParser(description="msfiddle")

    # Define two exclusive argument groups
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--demo", action="store_true", help="Run prediction on demo data"
    )
    mode_group.add_argument("--test_data", type=str, help="Path to data (.mgf)")

    # Add instrument type option
    parser.add_argument(
        "--instrument_type",
        type=str,
        choices=["qtof", "orbitrap"],
        default="orbitrap",
        help="Instrument type: qtof or orbitrap (default: orbitrap)",
    )

    # Add other required arguments
    parser.add_argument(
        "--result_path", type=str, required=True, help="Path to save predicted results"
    )

    # Add optional arguments
    parser.add_argument(
        "--buddy_path",
        type=str,
        default="",
        help=(
            "Path to BUDDY results: native/original msbuddy output directory "
            "or summary TSV. The msfiddle-normalized CSV is also accepted but "
            "deprecated and will be removed in 3.0.0."
        ),
    )
    parser.add_argument(
        "--sirius_path",
        type=str,
        default="",
        help=(
            "Path to SIRIUS results: native/original formula_identifications "
            "summary or SIRIUS summary output directory. The "
            "msfiddle-normalized CSV is also accepted but deprecated and will "
            "be removed in 3.0.0."
        ),
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

    # Add advanced arguments for custom configurations
    advanced_group = parser.add_argument_group("Advanced options")
    advanced_group.add_argument(
        "--config_path", type=str, help="Custom path to configuration (.yaml)"
    )
    advanced_group.add_argument(
        "--resume_path", type=str, help="Custom path to pretrained TCN model"
    )
    advanced_group.add_argument(
        "--rescore_resume_path",
        type=str,
        help="Custom path to pretrained rescore model",
    )

    args = parser.parse_args()

    # Initialize random seed
    init_random_seed(args.seed)

    # Get package directory
    package_dir = os.path.dirname(os.path.abspath(__file__))

    # Set default paths based on mode and instrument type
    if args.demo:
        test_data_path = os.path.join(package_dir, "demo", "input_msms.mgf")
        print(f"Using demo data: {test_data_path}")
    else:
        test_data_path = args.test_data
        print(f"Using custom data: {test_data_path}")

    # Set paths based on instrument type
    instrument_suffix = args.instrument_type

    # Set config path
    if args.config_path:
        config_path = args.config_path
    else:
        config_path = os.path.join(
            package_dir, "config", f"fiddle_tcn_{instrument_suffix}.yml"
        )
    print(f"Using config: {config_path}")

    # Set model paths
    checkpoint_dir = get_checkpoint_dir()

    if args.resume_path:
        resume_path = args.resume_path
    else:
        resume_path = os.path.join(checkpoint_dir, f"fiddle_tcn_{instrument_suffix}.pt")
    print(f"Using TCN model: {resume_path}")

    if args.rescore_resume_path:
        rescore_resume_path = args.rescore_resume_path
    else:
        rescore_resume_path = os.path.join(
            checkpoint_dir, f"fiddle_rescore_{instrument_suffix}.pt"
        )
    print(f"Using rescore model: {rescore_resume_path}")

    try:
        validate_checkpoint_paths(resume_path, rescore_resume_path)
    except FileNotFoundError as exc:
        sys.stdout.flush()
        parser.exit(status=1, message=f"{exc}\n")

    predictor = MsFiddlePredictor(
        instrument_type=args.instrument_type,
        device=args.device,
        no_cuda=args.no_cuda,
        config_path=config_path,
        resume_path=resume_path,
        rescore_resume_path=rescore_resume_path,
        download_models=False,
        verbose=True,
    )
    res_df = predictor.predict_mgf(
        test_data_path,
        buddy_path=args.buddy_path,
        sirius_path=args.sirius_path,
    )

    print("\nSaving predicted results...")
    res_df.to_csv(args.result_path, index=False)
    print(f"Done! Results saved to {args.result_path}")


if __name__ == "__main__":
    main()
