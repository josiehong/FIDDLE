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


SUBCOMMANDS = ("predict", "validate", "download", "paths")


def _add_predict_args(parser):
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
        choices=["qtof", "orbitrap", "timstof"],
        default="orbitrap",
        help="Instrument type: qtof, orbitrap or timstof (default: orbitrap)",
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
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Spectra per inference batch"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Processes for formula refinement (default: 1)",
    )

    parser.add_argument(
        "--eval",
        action="store_true",
        help=(
            "After predicting, report top-k accuracy against each spectrum's "
            "SMILES or FORMULA field and save it as <result>_eval.csv"
        ),
    )
    parser.add_argument(
        "--eval_details",
        "--eval-details",
        action="store_true",
        help=(
            "With --eval (implied), also save one row per spectrum to "
            "<result>_eval_details.csv and accuracy by precursor m/z, adduct, "
            "peak count and NCE to <result>_eval_groups.csv"
        ),
    )

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


def _predict(args, parser):
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
        batch_size=args.batch_size,
        verbose=True,
    )
    res_df = predictor.predict_mgf(
        test_data_path,
        buddy_path=args.buddy_path,
        sirius_path=args.sirius_path,
        num_workers=args.num_workers,
    )

    print("\nSaving predicted results...")
    res_df.to_csv(args.result_path, index=False)
    print(f"Done! Results saved to {args.result_path}")

    if args.eval or args.eval_details:
        kept_index = getattr(predictor, "_kept_index", None)
        _evaluate(res_df, test_data_path, args, parser, kept_index)


def _evaluate(res_df, truth_path, args, parser, kept_index=None):
    from .evaluate import load_eval_inputs, topk_accuracy

    stem = os.path.splitext(args.result_path)[0]
    try:
        inputs = load_eval_inputs(truth_path, kept_index)
    except ValueError as exc:  # e.g. an unparseable truth value in the MGF
        parser.exit(
            status=1, message=f"Could not read the truth in {truth_path}: {exc}\n"
        )
    try:
        result = topk_accuracy(res_df, inputs.truth)
    except ValueError as exc:
        # With no matched rows the details still show why (no_truth and
        # not_predicted rows), so save them before failing.
        if getattr(args, "eval_details", False):
            try:
                _save_eval_details(res_df, inputs, args, stem)
            except ValueError:
                pass
        parser.exit(status=1, message=f"{exc}\n")
    eval_path = stem + "_eval.csv"
    result.to_frame().to_csv(eval_path, index=False)
    print(f"\nEval against {truth_path}")
    print(result.format())
    print(f"Saved to {eval_path}")
    if getattr(args, "eval_details", False):
        _save_eval_details(res_df, inputs, args, stem)


def _save_eval_details(res_df, inputs, args, stem):
    from .evaluate import coverage, grouped_accuracy, topk_details

    details = topk_details(res_df, inputs.truth, info=inputs.info)
    details.insert(2, "instrument_type", getattr(args, "instrument_type", None))
    # A compound without SMILES is keyed by its formula vector; write the formula
    details["compound"] = details["compound"].where(
        details["compound"].map(lambda c: isinstance(c, str) or c is None),
        details["true_formula"],
    )
    details.to_csv(stem + "_eval_details.csv", index=False)
    grouped_accuracy(details).to_csv(stem + "_eval_groups.csv", index=False)
    print("\nCoverage")
    for key, value in coverage(details, len(res_df), inputs).items():
        print(f"  {key:<30s}{value}")
    print(f"Saved to {stem}_eval_details.csv and {stem}_eval_groups.csv")


def _validate(args, parser):
    from .api import MsFiddlePredictor, summarize_validation

    # Load the config and read the MGF in separate steps, so an error names
    # the file it came from
    try:
        predictor = MsFiddlePredictor(
            instrument_type=args.instrument_type, config_path=args.config_path
        )
    except Exception as exc:  # a missing or invalid config
        config = args.config_path or f"of {args.instrument_type}"
        parser.exit(status=1, message=f"Could not load the config {config}: {exc}\n")
    try:
        report = predictor.validate_mgf(args.test_data)
    except Exception as exc:  # a missing or unparseable MGF file
        parser.exit(status=1, message=f"Could not read {args.test_data}: {exc}\n")
    print(summarize_validation(report))
    if args.report_path:
        report.to_csv(args.report_path, index=False)
        print(f"Per-spectrum report saved to {args.report_path}")
    if (report["status"] == "rejected").all():
        parser.exit(status=1, message="No spectrum can be predicted.\n")


def _download(args, parser):
    from .download import download_models

    download_models(destination=args.destination, instruments=args.instruments)


def _paths(args, parser):
    from .download import print_checkpoint_paths

    print_checkpoint_paths()


def build_parser():
    parser = argparse.ArgumentParser(
        prog="msfiddle",
        description=(
            "msfiddle: chemical formula prediction from MS/MS spectra. "
            "Without a subcommand, the arguments are passed to 'predict'."
        ),
    )
    sub = parser.add_subparsers(
        dest="command", metavar="{predict,validate,download,paths}"
    )

    p = sub.add_parser("predict", help="Predict formulas for an MGF file")
    _add_predict_args(p)
    p.set_defaults(func=_predict)

    p = sub.add_parser(
        "validate",
        help="Check an MGF file's spectra without loading the models",
    )
    p.add_argument("--test_data", type=str, required=True, help="Path to data (.mgf)")
    p.add_argument(
        "--instrument_type",
        type=str,
        choices=["qtof", "orbitrap", "timstof"],
        default="orbitrap",
        help="Instrument type whose encoding to check against (default: orbitrap)",
    )
    p.add_argument(
        "--report_path",
        type=str,
        help="Save the status and reason for each spectrum to this CSV",
    )
    p.add_argument(
        "--config_path", type=str, help="Custom path to configuration (.yaml)"
    )
    p.set_defaults(func=_validate)

    p = sub.add_parser("download", help="Download the pre-trained models")
    p.add_argument("-d", "--destination", help="Directory to save the models")
    p.add_argument(
        "-i",
        "--instruments",
        nargs="+",
        choices=["qtof", "orbitrap", "timstof"],
        help="Instruments whose models to download (default: all)",
    )
    p.set_defaults(func=_download)

    p = sub.add_parser("paths", help="Show the checkpoint directory and models")
    p.set_defaults(func=_paths)
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Keep the pre-subcommand form working: `msfiddle --test_data ...`
    if argv and argv[0] not in SUBCOMMANDS and argv[0] not in ("-h", "--help"):
        argv = ["predict"] + argv
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        parser.exit(status=1)
    args.func(args, parser)


if __name__ == "__main__":
    main()
