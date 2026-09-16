"""Plot training curves and estimate completion from a train_tcn_gpus_cl log.

Parses the per-epoch blocks (train/validation errors, accuracies, LR, early
stop counter, epoch wall time) and writes a multi-panel PNG next to the log.
Also prints an ETA summary: average epoch time, patience state, and the time
to the epoch cap if early stopping never fires.

Usage:
  python scripts/plot_training_log.py tcn_qtof_082426.log [--epochs-cap 100]
  python scripts/plot_training_log.py tcn_orbitrap_082426.log [--epochs-cap 100]
"""

import argparse
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPOCH_RE = re.compile(r"^=====Epoch (\d+)")
TRAIN_RE = re.compile(
    r"^Train error: ([\d.]+) mass error: ([\d.]+) atom number error: ([\d.]+) "
    r"H/C number error: ([\d.]+) loss: (-?[\d.]+)"
)
VALID_RE = re.compile(r"^Validation error: ([\d.]+) mass error: ([\d.]+)")
ACC_RE = re.compile(r"^formula w acc: ([\d.]+)")
ACC_WO_RE = re.compile(r"^formula w/o acc: ([\d.]+)")
STOP_RE = re.compile(r"^Early stop count: (\d+)/(\d+)")
RESET_RE = re.compile(r"^Early stop patience reset")
# epoch duration and final lr from the completed tqdm bar; elapsed time is
# MM:SS for epochs under an hour and H:MM:SS above (Orbitrap epochs are ~1h)
BAR_RE = re.compile(r"100%\|.*\| \d+/\d+ \[(?:(\d+):)?(\d+):(\d+)<.*lr=([\d.e+-]+)")


def parse_log(path):
    rows = []
    cur = None
    for line in open(path, errors="replace"):
        line = line.rstrip("\n")
        m = EPOCH_RE.match(line)
        if m:
            if cur:
                rows.append(cur)
            cur = {"epoch": int(m.group(1)), "patience": 0}
            continue
        if cur is None:
            continue
        m = BAR_RE.search(line)
        if m and "minutes" not in cur:
            hours = int(m.group(1)) if m.group(1) else 0
            cur["minutes"] = hours * 60 + int(m.group(2)) + int(m.group(3)) / 60
            cur["lr"] = float(m.group(4))
        m = TRAIN_RE.match(line)
        if m:
            cur["train_mae"], cur["train_mass"] = float(m.group(1)), float(m.group(2))
            cur["train_loss"] = float(m.group(5))
        m = VALID_RE.match(line)
        if m:
            cur["val_mae"], cur["val_mass"] = float(m.group(1)), float(m.group(2))
        m = ACC_RE.match(line)
        if m:
            cur["acc"] = float(m.group(1))
        m = ACC_WO_RE.match(line)
        if m:
            cur["acc_wo"] = float(m.group(1))
        m = STOP_RE.match(line)
        if m:
            cur["patience"], cur["patience_max"] = int(m.group(1)), int(m.group(2))
    if cur and "val_mae" in cur:
        rows.append(cur)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", help="training log file")
    ap.add_argument("--out", default="", help="output PNG (default: <log>.png)")
    ap.add_argument(
        "--epochs-cap",
        type=int,
        default=0,
        help="config train.epochs, for the no-early-stop ETA bound",
    )
    args = ap.parse_args()

    rows = parse_log(args.log)
    if not rows:
        sys.exit(f"no epoch blocks parsed from {args.log}")
    ep = [r["epoch"] for r in rows]

    def series(key):
        return (
            [r["epoch"] for r in rows if key in r],
            [r[key] for r in rows if key in r],
        )

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    ax = axes[0][0]
    ax.plot(*series("train_loss"), marker=".")
    ax.set_title("train loss (epoch avg)")
    ax = axes[0][1]
    ax.plot(*series("train_mass"), marker=".", label="train")
    ax.plot(*series("val_mass"), marker=".", label="validation")
    ax.set_title("mass error (Da)")
    ax.legend()
    ax = axes[0][2]
    ax.plot(*series("train_mae"), marker=".", label="train (raw space pre-fix runs)")
    ax.plot(*series("val_mae"), marker=".", label="validation (count space)")
    ax.set_title("count MAE")
    ax.legend()
    ax = axes[1][0]
    ax.plot(*series("acc"), marker=".", label="with H")
    ax.plot(*series("acc_wo"), marker=".", label="without H")
    ax.set_title("formula accuracy (log-decoded)")
    ax.legend()
    ax = axes[1][1]
    ax.plot(*series("lr"), marker=".")
    ax.set_yscale("log")
    ax.set_title("learning rate")
    ax = axes[1][2]
    ax.plot(*series("patience"), marker=".")
    if rows and "patience_max" in rows[-1]:
        ax.axhline(rows[-1]["patience_max"], color="r", ls="--", label="stop")
        ax.legend()
    ax.set_title("early-stop counter")
    for row_axes in axes:
        for a in row_axes:
            a.set_xlabel("epoch")
            a.grid(alpha=0.3)
    fig.tight_layout()
    out = args.out or args.log + ".png"
    fig.savefig(out, dpi=120)
    print(f"plot: {out}")

    # ETA summary
    minutes = [r["minutes"] for r in rows if "minutes" in r]
    avg_min = sum(minutes[-10:]) / len(minutes[-10:]) if minutes else 0
    last = rows[-1]
    pat, pat_max = last.get("patience", 0), last.get("patience_max", 15)
    print(
        f"epochs logged: {ep[-1]}   avg epoch: {avg_min:.1f} min"
        f"   patience: {pat}/{pat_max}"
    )
    resets = [r["epoch"] for r in rows if r.get("patience", 1) == 0]
    if resets:
        gaps = [b - a for a, b in zip(resets, resets[1:])] or [1]
        print(
            f"last improvement: epoch {resets[-1]}"
            f"   recent improvement gap: ~{max(gaps[-3:]):d} epochs"
        )
    print(
        f"earliest early stop: {pat_max - pat} more non-improving epochs"
        f" (~{(pat_max - pat) * avg_min / 60:.1f} h)"
    )
    if args.epochs_cap:
        remaining = max(0, args.epochs_cap - ep[-1])
        print(
            f"no-early-stop bound: {remaining} epochs to cap"
            f" (~{remaining * avg_min / 60:.1f} h)"
        )


if __name__ == "__main__":
    main()
