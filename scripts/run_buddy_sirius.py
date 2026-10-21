"""Run BUDDY (msbuddy) and SIRIUS on an MGF file for comparison with FIDDLE.

Writes <out_dir>/input.mgf (the spectra with ION set from PRECURSOR_TYPE, the
field both tools read the adduct from), then <out_dir>/buddy/ and
<out_dir>/sirius/, which `msfiddle predict` takes as --buddy_path and
--sirius_path. A tool that is not on PATH is skipped.

Usage:
    python scripts/run_buddy_sirius.py --mgf ./data/casmi2016.mgf \\
        --instrument_type qtof --out_dir ./external/casmi2016
"""

import argparse
import os
import shutil
import subprocess

from pyteomics import mgf


def with_ion(path):
    for spec in mgf.read(path):
        spec["params"]["ion"] = spec["params"]["precursor_type"]
        yield spec


def run(cmd):
    if shutil.which(cmd[0]) is None:
        print(f"{cmd[0]} not found on PATH; skipping")
        return False
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BUDDY and SIRIUS on an MGF")
    parser.add_argument("--mgf", type=str, required=True)
    parser.add_argument(
        "--instrument_type",
        type=str,
        required=True,
        choices=["qtof", "orbitrap", "timstof"],
        help="timstof uses the qtof profiles of BUDDY and SIRIUS",
    )
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    profile = "qtof" if args.instrument_type == "timstof" else args.instrument_type
    # Start clean: SIRIUS adds to an existing project, so results for spectra
    # of an earlier run (e.g. an older test set) would be pooled in
    for old in ("buddy", "sirius", "sirius_project"):
        shutil.rmtree(os.path.join(args.out_dir, old), ignore_errors=True)
    os.makedirs(args.out_dir, exist_ok=True)
    input_mgf = os.path.join(args.out_dir, "input.mgf")
    mgf.write(with_ion(args.mgf), input_mgf, file_mode="w")

    run(
        ["msbuddy", "-mgf", input_mgf, "-output", os.path.join(args.out_dir, "buddy")]
        + ["-ms", profile, "-d", "-p"]
    )

    project = os.path.join(args.out_dir, "sirius_project")
    if run(
        ["sirius", "--input", input_mgf, "--project", project]
        + ["formulas", "--profile", profile]
    ):
        run(
            ["sirius", "--project", project, "summaries", "--top-k-summary=5"]
            + ["--output", os.path.join(args.out_dir, "sirius")]
        )
