r"""Package trained checkpoints as release zips for msfiddle-download-models.

Each training checkpoint check_point/fiddle_<model>_<tag>.pt is reduced to the
weights inference needs (TCN: model_state_dict without the DataParallel
"module." prefix; rescore: formula encoder + rescore head), plus the epoch and
best validation metrics for provenance. Optimizer and scheduler state are
dropped. An instrument's TCN and rescore model go into one zip,
<out_dir>/fiddle_<instrument>.zip holding fiddle_tcn_<instrument>.pt and
fiddle_rescore_<instrument>.pt, the layout download.py expects from v2.3.0 on.

The rescore model only works with the TCN it was trained on, so its
checkpoint carries a hash of the TCN's weights: rescore checkpoints from
train_rescore.py already record it, and packaging stops if it does not match
the TCN; older ones get it here. Each zip is reloaded and compared against
its sources before the script reports success.

Usage:
  python scripts/package_release_checkpoints.py --tag 102026_muon \
      --out_dir ./release_v2.3.0
  # one instrument at a time, e.g. on the server that trained it
  python scripts/package_release_checkpoints.py --tag 102026_muon \
      --out_dir ./release_v2.3.0 --instruments timstof
"""

import argparse
import os
import sys
import tempfile
import zipfile

import torch

# Make the repo root importable so `msfiddle` resolves without an editable install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from msfiddle.download import INSTRUMENTS, model_files
from msfiddle.model_tcn import tcn_hash

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
    "best_val_top1",
    "seed",
    "tcn_hash",
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


def package(srcs, instrument, out_dir):
    """Write <out_dir>/fiddle_<instrument>.zip from the TCN and rescore
    checkpoints in srcs ({'tcn': path, 'rescore': path}); return the zip path."""
    ckpts = {
        kind: torch.load(path, map_location="cpu", weights_only=False)
        for kind, path in srcs.items()
    }
    rels = {kind: release_checkpoint(ckpt, kind) for kind, ckpt in ckpts.items()}
    digest = tcn_hash(rels["tcn"]["model_state_dict"])
    recorded = rels["rescore"].setdefault("tcn_hash", digest)
    if recorded != digest:
        raise RuntimeError(
            f"{srcs['rescore']} was trained with a different TCN than {srcs['tcn']}"
        )

    names = dict(zip(("tcn", "rescore"), model_files(instrument)))
    zip_path = os.path.join(out_dir, f"fiddle_{instrument}.zip")
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for kind, name in names.items():
                torch.save(rels[kind], os.path.join(tmp, name))
                zf.write(os.path.join(tmp, name), arcname=name)

        # Verify what a user would get: extract the zip and compare weights.
        check_dir = os.path.join(tmp, "check")
        with zipfile.ZipFile(zip_path) as zf:
            if sorted(zf.namelist()) != sorted(names.values()):
                raise RuntimeError(f"{zip_path}: unexpected entries {zf.namelist()}")
            zf.extractall(check_dir)
        loaded = {
            kind: torch.load(
                os.path.join(check_dir, name), map_location="cpu", weights_only=False
            )
            for kind, name in names.items()
        }
    for kind in names:
        for k in WEIGHT_KEYS[kind]:
            if not same_weights(strip_module_prefix(ckpts[kind][k]), loaded[kind][k]):
                raise RuntimeError(f"{zip_path}: {k} differs from {srcs[kind]}")
    if loaded["rescore"]["tcn_hash"] != tcn_hash(loaded["tcn"]["model_state_dict"]):
        raise RuntimeError(f"{zip_path}: the TCN hash does not match")
    return zip_path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--tag",
        default="",
        help="suffix of the training checkpoints, e.g. 102026_muon to read "
        "fiddle_tcn_qtof_102026_muon.pt; the released files drop it",
    )
    ap.add_argument("--checkpoint_dir", default="./check_point")
    ap.add_argument("--out_dir", required=True, help="where the zips are written")
    ap.add_argument(
        "--instruments", nargs="+", default=list(INSTRUMENTS), choices=INSTRUMENTS
    )
    args = ap.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    srcs = {
        inst: {
            kind: os.path.join(args.checkpoint_dir, f"fiddle_{kind}_{inst}{suffix}.pt")
            for kind in ("tcn", "rescore")
        }
        for inst in args.instruments
    }
    missing = [p for s in srcs.values() for p in s.values() if not os.path.isfile(p)]
    if missing:
        sys.exit("Missing checkpoints:\n  " + "\n  ".join(missing))

    os.makedirs(args.out_dir, exist_ok=True)
    for inst, src in srcs.items():
        zip_path = package(src, inst, args.out_dir)
        mb = os.path.getsize(zip_path) / 2**20
        print(f"{src['tcn']} + {src['rescore']} -> {zip_path} ({mb:.1f} MB, verified)")


if __name__ == "__main__":
    main()
