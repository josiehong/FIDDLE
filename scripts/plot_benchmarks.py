"""Summarize and plot the external benchmarks of FIDDLE with BUDDY and SIRIUS.

Reads the results that experiments_test_benchmark.sh writes
(result/<method>_<benchmark>[_unseen]_eval.csv), writes the top-1/3/5 accuracy
of every method to a CSV and plots it: one column per benchmark and one row
each for all and unseen compounds, with top-1/3/5 accuracy bars per method
above a matrix marking the tools each method combines.

Usage:
    python scripts/plot_benchmarks.py --csv ./result/benchmarks.csv \\
        --figure ./docs/_static/benchmarks.png
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
TOOLS = ["FIDDLE", "BUDDY", "SIRIUS"]
# Method name -> tools it combines, in plotting order
METHODS = {
    "fiddle": ["FIDDLE"],
    "fiddle_buddy": ["FIDDLE", "BUDDY"],
    "fiddle_buddy_sirius": ["FIDDLE", "BUDDY", "SIRIUS"],
}
SUBSETS = {"": "All", "_unseen": "Unseen"}
TOPK = (1, 3, 5)
# One ordinal blue ramp, darkest for top-1; the lightest step still clears 2:1
COLORS = {1: "#184f95", 3: "#3987e5", 5: "#86b6ef"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def summarize(result_dir):
    rows = []
    for bench in BENCHMARKS:
        for suffix, subset in SUBSETS.items():
            for method in METHODS:
                path = os.path.join(result_dir, f"{method}_{bench}{suffix}_eval.csv")
                d = pd.read_csv(path).set_index("k")
                rows.append(
                    {
                        "benchmark": bench,
                        "compounds": subset,
                        "method": method,
                        "n_spectra": int(d["n_spectra"].iloc[0]),
                        **{f"top{k}": 100 * d.loc[k, "spectrum_acc"] for k in (1, 3, 5)},
                    }
                )
    return pd.DataFrame(rows)


def plot(summary, path):
    methods = list(METHODS)
    width = 0.27
    fig, axes = plt.subplots(
        3,
        len(BENCHMARKS),
        figsize=(7.2, 5.4),
        gridspec_kw={"height_ratios": [3, 3, 1.1], "hspace": 0.25, "wspace": 0.08},
        sharex="col",
        sharey="row",
    )
    for col, (bench, title) in enumerate(BENCHMARKS.items()):
        for row, subset in enumerate(SUBSETS.values()):
            ax = axes[row, col]
            d = summary[(summary.benchmark == bench) & (summary.compounds == subset)]
            d = d.set_index("method").loc[methods]
            for i, k in enumerate(TOPK):
                ax.bar(
                    [x + (i - 1) * width for x in range(len(methods))],
                    d[f"top{k}"],
                    width=width - 0.03,  # surface gap between adjacent bars
                    color=COLORS[k],
                    label=f"Top-{k}",
                    zorder=2,
                )
            ax.text(
                1.0,
                -0.03,
                f"n = {d.n_spectra.iloc[0]} spectra",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                color=MUTED,
            )
            ax.set_ylim(0, 100)
            ax.set_yticks(range(0, 101, 20))
            ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(MUTED)
            ax.tick_params(axis="y", length=0, labelsize=7, colors=MUTED)
            ax.tick_params(axis="x", length=0)
            if row == 0:
                ax.set_title(title, fontsize=9, color=INK)
            if col == 0:
                ax.set_ylabel(f"{subset} compounds\naccuracy (%)", fontsize=8, color=INK)

        # Combination matrix: a filled dot for each tool the method uses
        mat = axes[-1, col]
        for x, method in enumerate(methods):
            used = [TOOLS.index(t) for t in METHODS[method]]
            for r in range(len(TOOLS)):
                mat.scatter(x, r, s=22, color=INK if r in used else GRID, zorder=3)
            if len(used) > 1:
                mat.plot([x, x], [min(used), max(used)], color=INK, lw=1.5, zorder=2)
        mat.set_ylim(len(TOOLS) - 0.5, -0.5)
        mat.set_yticks(range(len(TOOLS)))
        mat.set_yticklabels(TOOLS, fontsize=7, color=INK)
        mat.set_xticks([])
        mat.tick_params(axis="y", length=0)
        for spine in mat.spines.values():
            spine.set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=8,
        ncol=len(labels),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.95),
    )
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--result_dir", default="./result")
    parser.add_argument("--csv", default="./result/benchmarks.csv")
    parser.add_argument("--figure", default="./result/benchmarks.png")
    args = parser.parse_args()

    summary = summarize(args.result_dir)
    for path in (args.csv, args.figure):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    summary.to_csv(args.csv, index=False, float_format="%.1f")
    print(summary.to_string(index=False, float_format="%.1f"))
    plot(summary, args.figure)
    print(f"Saved {args.csv} and {args.figure}")
