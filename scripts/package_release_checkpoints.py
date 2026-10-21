r"""Package trained checkpoints as release zips for msfiddle-download-models.

Each training checkpoint check_point/fiddle_<model>_<tag>.pt is reduced to the
weights inference needs (TCN: model_state_dict without the DataParallel
"module." prefix; rescore: formula encoder + rescore head), plus the epoch and
best validation metrics for provenance. Optimizer and scheduler state are
dropped. The result is written as <out_dir>/fiddle_<model>.zip containing a
single fiddle_<model>.pt, the layout download.py expects, and each zip is
reloaded and compared against its source before the script reports success.

Usage:
  python scripts/package_release_checkpoints.py --tag 102026_muon \
      --out_dir ./release_v2.3.0
  # one instrument at a time, e.g. on the server that trained it
  python scripts/package_release_checkpoints.py --tag 102026_muon \
      --out_dir ./release_v2.3.0 --models tcn_timstof rescore_timstof
  # checkpoints already named fiddle_<model>.pt: omit --tag
  python scripts/package_release_checkpoints.py \
      --checkpoint_dir ./release_v2.0.0 --out_dir ./release_v2.0.0
"""

import argparse
import os
import sys
import tempfile
import zipfile

import torch

MODELS = [
    f"{kind}_{inst}"
    for inst in ("qtof", "orbitrap", "timstof")
    for kind in ("tcn", "rescore")
]
WEIGHT_KEYS = {
    "tcn": ["model_state_dict"],
    "rescore": ["formula_encoder_state_dict", "rescore_head_state_dict"],
}
META_KEYS = [
    "epoch",
    "num_params",
    "best_val_mae",
    "best_val_acc",
    "best_val_wo_acc",
    "best_val_auc",
]


def strip_module_prefix(state_dict):
    return {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }


def release_checkpoint(ckpt, kind):
    """Weights-only copy of a training checkpoint for model kind 'tcn'/'rescore'."""
    missing = [k for k in WEIGHT_KEYS[kind] if k not in ckpt]
    if missing:
        raise KeyError(f"{kind} checkpoint lacks {missing}")
    out = {k: strip_module_prefix(ckpt[k]) for k in WEIGHT_KEYS[kind]}
    out.update({k: ckpt[k] for k in META_KEYS if k in ckpt})
    return out


def same_weights(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def package(src_path, model, out_dir):
    """Write <out_dir>/fiddle_<model>.zip from src_path; return the zip path."""
    kind = model.split("_")[0]
    ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
    rel = release_checkpoint(ckpt, kind)

    name = f"fiddle_{model}.pt"
    zip_path = os.path.join(out_dir, f"fiddle_{model}.zip")
    with tempfile.TemporaryDirectory() as tmp:
        pt_path = os.path.join(tmp, name)
        torch.save(rel, pt_path)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(pt_path, arcname=name)

        # Verify what a user would get: extract the zip and compare weights.
        check_dir = os.path.join(tmp, "check")
        with zipfile.ZipFile(zip_path) as zf:
            if zf.namelist() != [name]:
                raise RuntimeError(f"{zip_path}: unexpected entries {zf.namelist()}")
            zf.extractall(check_dir)
        loaded = torch.load(
            os.path.join(check_dir, name), map_location="cpu", weights_only=False
        )
    for k in WEIGHT_KEYS[kind]:
        if not same_weights(strip_module_prefix(ckpt[k]), loaded[k]):
            raise RuntimeError(f"{zip_path}: {k} differs from {src_path}")
    return zip_path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--tag",
        default="",
        help="checkpoint suffix, e.g. 102026_muon (omit for fiddle_<model>.pt)",
    )
    ap.add_argument("--checkpoint_dir", default="./check_point")
    ap.add_argument("--out_dir", required=True, help="where the zips are written")
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    args = ap.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    srcs = {
        m: os.path.join(args.checkpoint_dir, f"fiddle_{m}{suffix}.pt")
        for m in args.models
    }
    missing = [p for p in srcs.values() if not os.path.isfile(p)]
    if missing:
        sys.exit("Missing checkpoints:\n  " + "\n  ".join(missing))

    os.makedirs(args.out_dir, exist_ok=True)
    for model, src in srcs.items():
        zip_path = package(src, model, args.out_dir)
        mb = os.path.getsize(zip_path) / 2**20
        print(f"{src} -> {zip_path} ({mb:.1f} MB, verified)")


if __name__ == "__main__":
    main()
