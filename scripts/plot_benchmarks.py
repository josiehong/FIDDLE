"""Summarize and plot the external benchmarks of FIDDLE with BUDDY and SIRIUS.

Reads the results that experiments_test_benchmark.sh writes
(result/<method>_<benchmark>_eval.csv) and, with --subsets all unseen, those of
experiments_test_benchmark_unseen.sh (result/<method>_<benchmark>_unseen_eval.csv),
writes the top-1/3/5 accuracy of every method to a CSV and plots it: one column
per benchmark and one row per subset, with top-1/3/5 accuracy bars per method,
each labelled with its value.

Usage:
    python scripts/plot_benchmarks.py --subsets all unseen \\
        --csv ./result/benchmarks.csv --figure ./result/benchmarks.png
    # for a release, copy result/benchmarks.png to docs/_static/ by hand
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Benchmark -> title, naming the models it is run with
BENCHMARKS = {
    "casmi2016": "CASMI 2016\n(Orbitrap models)",
    "casmi2017": "CASMI 2017\n(Q-TOF models)",
    "embl": "EMBL-MCF 2.0\n(Orbitrap models)",
}
# Method name -> axis label, in plotting order (the paper's names)
METHODS = {
    "fiddle": "FIDDLE",
    "fiddle_buddy": "FIDDLE w.\nBUDDY",
    "fiddle_buddy_sirius": "FIDDLE w.\nBUDDY &\nSIRIUS",
}
TOPK = (1, 3, 5)
# One ordinal blue ramp, darkest for top-1; the lightest step still clears 2:1
COLORS = {1: "#184f95", 3: "#3987e5", 5: "#86b6ef"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


SUBSETS = {"all": ("", "All compounds"), "unseen": ("_unseen", "Unseen compounds")}


def summarize(result_dir, subsets=("all",), methods=tuple(METHODS)):
    rows = []
    for bench in BENCHMARKS:
        for subset in subsets:
            suffix, label = SUBSETS[subset]
            for method in methods:
                path = os.path.join(result_dir, f"{method}_{bench}{suffix}_eval.csv")
                d = pd.read_csv(path).set_index("k")
                rows.append(
                    {
                        "benchmark": bench,
                        "compounds": label,
                        "method": method,
                        "n_spectra": int(d["n_spectra"].iloc[0]),
                        "n_compounds": int(d["n_compounds"].iloc[0]),
                        **{f"top{k}": 100 * d.loc[k, "spectrum_acc"] for k in TOPK},
                    }
                )
    return pd.DataFrame(rows)


def plot(summary, path):
    methods = [m for m in METHODS if m in set(summary["method"])]
    labels = list(dict.fromkeys(summary["compounds"]))
    n = len(labels)
    width = 0.27
    fig, axes = plt.subplots(
        n,
        len(BENCHMARKS),
        figsize=(7.2, 0.7 + 2.4 * n),
        gridspec_kw={"hspace": 0.2, "wspace": 0.06},
        sharex="col",
        sharey="row",
        squeeze=False,
    )
    for col, (bench, title) in enumerate(BENCHMARKS.items()):
        for row, label in enumerate(labels):
            ax = axes[row, col]
            d = summary[(summary.benchmark == bench) & (summary.compounds == label)]
            d = d.set_index("method").loc[methods]
            xs = range(len(methods))
            for i, k in enumerate(TOPK):
                bx = [x + (i - 1) * width for x in xs]
                ax.bar(
                    bx,
                    d[f"top{k}"],
                    width=width - 0.03,  # surface gap between adjacent bars
                    color=COLORS[k],
                    label=f"Top-{k}",
                    zorder=2,
                )
                for x, value in zip(bx, d[f"top{k}"]):
                    ax.text(
                        x,
                        value + 1.2,
                        f"{value:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=6.5,
                        color=INK,
                        rotation=90,
                    )
            r0 = d.iloc[0]
            ax.text(
                0.5,
                1.02,
                f"n = {r0.n_spectra} spectra, {r0.n_compounds} compounds",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=7,
                color=MUTED,
            )
            if row == 0:
                ax.set_title(title, fontsize=9, color=INK, pad=14)
            ax.set_ylim(0, 124)
            ax.set_yticks(range(0, 101, 20))
            ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(MUTED)
            ax.tick_params(axis="y", length=0, labelsize=7, colors=MUTED)
            ax.tick_params(axis="x", length=0, labelsize=6.5, colors=INK)
            ax.set_xticks(list(xs))
            ax.set_xticklabels(
                [METHODS[m] for m in methods], ha="center", multialignment="center"
            )
            if col == 0:
                ax.set_ylabel(f"{label}\naccuracy (%)", fontsize=8, color=INK)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        fontsize=8,
        ncol=len(legend_labels),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.955 if n > 1 else 1.0),
    )
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.05, facecolor="white")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--result_dir", default="./result")
    parser.add_argument(
        "--subsets",
        nargs="+",
        choices=list(SUBSETS),
        default=["all"],
        help="One row each, in this order (default: all)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHODS),
        default=list(METHODS),
        help="Methods to include (default: all)",
    )
    parser.add_argument("--csv", default="./result/benchmarks.csv")
    parser.add_argument("--figure", default="./result/benchmarks.png")
    args = parser.parse_args()

    summary = summarize(args.result_dir, args.subsets, args.methods)
    for path in (args.csv, args.figure):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    summary.to_csv(args.csv, index=False, float_format="%.1f")
    print(summary.to_string(index=False, float_format="%.1f"))
    plot(summary, args.figure)
    print(f"Saved {args.csv} and {args.figure}")
