"""Run BUDDY (msbuddy) and SIRIUS on an MGF file for comparison with FIDDLE.

Writes <out_dir>/input.mgf for the tools: the spectra with the fields both
read the precursor from (PEPMASS from PRECURSOR_MZ, ION and CHARGE from
PRECURSOR_TYPE), MS_LEVEL as a number, which SIRIUS requires, FEATURE_ID set
to the title, which SIRIUS reports its results under, and without the true
structure and formula, which SIRIUS would otherwise take as the answer. ION is
the only field starting with "ION": SIRIUS takes any such field as the adduct,
so IONMODE=P would become [M+P]+.
Then it writes <out_dir>/buddy/ and
<out_dir>/sirius/, which `msfiddle predict` takes as --buddy_path and
--sirius_path. A tool that is not on PATH is skipped.

Usage:
    python scripts/run_buddy_sirius.py --mgf ./data/casmi2016.mgf \\
        --instrument_type qtof --out_dir ./external/casmi2016
"""

import argparse
import os
import re
import shutil
import subprocess
import sys

from pyteomics import mgf

# Fields that give away the answer; SIRIUS fixes the formula to them when present.
TRUTH_KEYS = (
    "smiles",
    "inchi",
    "inchikey",
    "inchi_key",
    "formula",
    "molmass",
    "theoretical_precursor_mz",
    "simulated_precursor_mz",
    "ppm",
)


def with_ion(path):
    for spec in mgf.read(path):
        params = spec["params"]
        for key in list(params):
            if key in TRUTH_KEYS or key.startswith("ion") or "adduct" in key:
                del params[key]
        params["feature_id"] = params["title"]
        params["pepmass"] = float(params["precursor_mz"])
        params["ion"] = params["precursor_type"]
        n, sign = re.search(r"\](\d*)([+-])$", params["ion"]).groups()
        params["charge"] = (n or "1") + sign
        if "ms_level" in params:
            params["ms_level"] = str(params["ms_level"]).lstrip("MS")
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
    # of an earlier run (e.g. an older test set) would be pooled in. SIRIUS 6
    # keeps a project in one file.
    for old in ("buddy", "sirius", "sirius_project", "sirius_project.sirius"):
        old = os.path.join(args.out_dir, old)
        if os.path.isdir(old):
            shutil.rmtree(old)
        elif os.path.exists(old):
            os.remove(old)
    os.makedirs(args.out_dir, exist_ok=True)
    input_mgf = os.path.join(args.out_dir, "input.mgf")
    mgf.write(with_ion(args.mgf), input_mgf, file_mode="w")

    run(
        ["msbuddy", "-mgf", input_mgf, "-output", os.path.join(args.out_dir, "buddy")]
        + ["-ms", profile, "-d", "-p"]
    )

    project = os.path.join(args.out_dir, "sirius_project.sirius")
    if run(
        ["sirius", "--input", input_mgf, "--project", project]
        + ["formulas", "--profile", profile]
    ):
        sirius_out = os.path.join(args.out_dir, "sirius")
        run(
            ["sirius", "--project", project, "summaries", "--top-k-summary=5"]
            + ["--output", sirius_out]
        )
        # SIRIUS skips spectra it cannot parse and still exits with 0
        if not os.path.isdir(sirius_out):
            sys.exit(f"SIRIUS wrote no results to {sirius_out}; see its log above")
