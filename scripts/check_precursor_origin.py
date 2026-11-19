"""Check whether each training library's precursor m/z is measured or computed.

For every spectrum, the precursor m/z is compared with the theoretical m/z of
its structure and adduct, as preprocessing computes it. A library that stores
computed values puts most spectra within the rounding error (< 0.2 ppm at four
decimals); measured values scatter by a few ppm. Run it on a new library
before training on it: precursor m/z that is computed from the answer teaches
the model nothing about real mass error, and evaluating on it overstates
accuracy.

The libraries are read from --raw_dir with the names prepare_msms.py uses, or
from the zip archives in --raw_dir when a file is not unpacked (nist20 from
nist.zip, for example). Libraries whose files are missing are skipped.

Usage:
    python scripts/check_precursor_origin.py --raw_dir ./data/origin/ \\
        [--dataset nist20 nist23 ...] [--limit N] [--out precursor_origin.csv]
"""

import argparse
import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd

# Make the repo root importable so `msfiddle` resolves without an editable install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from msfiddle.utils.msms_utils import precursor_mz_calculator  # noqa: E402

DATASETS = ["agilent", "nist20", "nist23", "mona", "waters", "gnps", "casmi2026"]
# |ppm| bands; computed values fall in the first
BANDS = [0, 0.2, 1, 5, 20, np.inf]
BAND_NAMES = ["<0.2", "0.2-1", "1-5", "5-20", ">20"]
# A library with more than this share below 0.2 ppm looks computed
COMPUTED_SHARE = 0.5


class Source:
    """A raw file in raw_dir, or the same file inside one of its zips."""

    def __init__(self, raw_dir, name, zip_name):
        self.path = os.path.join(raw_dir, name)
        self.zip = os.path.join(raw_dir, zip_name) if zip_name else None
        self.name = name

    def exists(self):
        return os.path.exists(self.path) or bool(self.zip and os.path.exists(self.zip))

    def binary(self):
        if os.path.exists(self.path):
            return open(self.path, "rb")
        # Read from the archive in pieces: the members are too large to load,
        # and pyteomics needs a seekable stream, which a pipe is not
        return zipfile.ZipFile(self.zip).open(self.name)

    def text(self):
        return io.TextIOWrapper(self.binary(), encoding="utf-8", errors="replace")


def ppm_of(mz, adduct, mass):
    try:
        theo = float(precursor_mz_calculator(adduct, float(mass)))
        return (float(mz) - theo) / theo * 1e6
    except Exception:  # unsupported adduct or unparseable value
        return np.nan


_MASS = {}


def mass_of_smiles(smiles):
    if smiles not in _MASS:
        from rdkit import Chem
        from rdkit.Chem.rdMolDescriptors import CalcExactMolWt

        mol = Chem.MolFromSmiles(smiles)
        _MASS[smiles] = CalcExactMolWt(mol) if mol is not None else np.nan
    return _MASS[smiles]


def sdf_errors(source, limit):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.rdMolDescriptors import CalcExactMolWt

    RDLogger.DisableLog("rdApp.*")
    out = []
    for mol in Chem.ForwardSDMolSupplier(source.binary(), sanitize=False):
        if limit and len(out) >= limit:
            break
        if mol is None or not (
            mol.HasProp("PRECURSOR M/Z") and mol.HasProp("PRECURSOR TYPE")
        ):
            continue
        try:
            mol.UpdatePropertyCache(strict=False)
            mass = CalcExactMolWt(mol)
        except Exception:
            continue
        out.append(
            ppm_of(mol.GetProp("PRECURSOR M/Z"), mol.GetProp("PRECURSOR TYPE"), mass)
        )
    return out


def mgf_errors(sources, limit):
    """MGF spectra with PRECURSOR_TYPE, PRECURSOR_MZ (or PEPMASS) and SMILES
    (or MOLMASS)."""
    from pyteomics import mgf

    out = []
    for source in sources:
        for spec in mgf.read(source.text(), use_index=False):
            if limit and len(out) >= limit:
                return out
            p = spec["params"]
            mz = p.get("precursor_mz", p.get("pepmass"))
            if isinstance(mz, (tuple, list)):
                mz = mz[0]
            if mz is None or "precursor_type" not in p:
                continue
            if "smiles" in p:
                mass = mass_of_smiles(p["smiles"])
            elif "molmass" in p:
                mass = p["molmass"]
            else:
                continue
            out.append(ppm_of(mz, p["precursor_type"], mass))
    return out


def gnps_errors(source, limit):
    """GNPS keeps its metadata in a CSV, split by sub-library."""
    df = pd.read_csv(
        source.binary(),
        usecols=["Adduct", "Precursor_MZ", "ExactMass", "GNPS_library_membership"],
        nrows=limit or None,
    ).dropna()
    df["Adduct"] = df["Adduct"].str.replace(r"\]1([+-])$", r"]\1", regex=True)
    df["ppm"] = [
        ppm_of(m, a, e) for m, a, e in zip(df.Precursor_MZ, df.Adduct, df.ExactMass)
    ]
    return df.groupby("GNPS_library_membership")["ppm"]


def summarize(library, errors):
    ppm = np.abs(np.asarray(errors, dtype=float))
    ppm = ppm[np.isfinite(ppm)]
    row = {"library": library, "n": len(ppm)}
    if len(ppm):
        shares = np.histogram(ppm, BANDS)[0] / len(ppm)
        row["median_ppm"] = float(np.median(ppm))
        row.update({f"share_{b}": float(s) for b, s in zip(BAND_NAMES, shares)})
        row["looks_computed"] = bool(shares[0] > COMPUTED_SHARE)
    return row


def check(raw_dir, datasets, limit):
    def src(name, zip_name=None):
        return Source(raw_dir, name, zip_name)

    rows = []
    if "agilent" in datasets:
        for name in ("Agilent_Combined.sdf", "Agilent_Metlin.sdf"):
            s = src(name, "agilent.zip")
            if s.exists():
                rows.append(summarize(f"agilent {name}", sdf_errors(s, limit)))
    if "nist20" in datasets:
        s = src("hr_msms_nist.SDF", "nist.zip")
        if s.exists():
            rows.append(summarize("nist20", sdf_errors(s, limit)))
    if "nist23" in datasets:
        sources = [src(n) for n in ("exported_hr_msms.mgf", "exported_hr_msms2.mgf")]
        if all(s.exists() for s in sources):
            rows.append(summarize("nist23", mgf_errors(sources, limit)))
    if "mona" in datasets:
        for name in (
            "MoNA-export-All_LC-MS-MS_QTOF.sdf",
            "MoNA-export-All_LC-MS-MS_Orbitrap.sdf",
        ):
            s = src(name, "mona.zip")
            if s.exists():
                rows.append(summarize(f"mona {name}", sdf_errors(s, limit)))
    if "waters" in datasets:
        s = src("waters_qtof.mgf", "waters_qtof.zip")
        if s.exists():
            rows.append(summarize("waters", mgf_errors([s], limit)))
    if "gnps" in datasets:
        s = src("ALL_GNPS_cleaned.csv", "gnps.zip")
        if s.exists():
            groups = gnps_errors(s, limit)
            rows.append(summarize("gnps", groups.obj))
            for lib, ppm in groups:
                rows.append(summarize(f"gnps {lib}", ppm))
    if "casmi2026" in datasets:
        path = os.path.join(raw_dir, "casmi2026", "train.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path, columns=["ingest_lib", "precursor_error_ppm"])
            if limit:
                df = df.head(limit)
            rows.append(summarize("casmi2026", df.precursor_error_ppm))
            for lib, d in df.groupby("ingest_lib"):
                rows.append(summarize(f"casmi2026 {lib}", d.precursor_error_ppm))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--raw_dir", default="./data/origin/", help="Raw libraries")
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=DATASETS,
        default=DATASETS,
        help="Libraries to check (default: all whose files exist)",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Spectra per library (default: all)"
    )
    parser.add_argument("--out", default="", help="Save the table as CSV")
    args = parser.parse_args()

    table = check(args.raw_dir, args.dataset, args.limit)
    found = {lib.split()[0] for lib in table.get("library", [])}
    skipped = [d for d in args.dataset if d not in found]
    if skipped:
        print(f"Skipped, files not found in {args.raw_dir}: {', '.join(skipped)}")
    if table.empty:
        sys.exit(f"No library found in {args.raw_dir}")
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(
            table.to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}",
                formatters={f"share_{b}": "{:.1%}".format for b in BAND_NAMES},
            )
        )
    computed = table.loc[
        table.get("looks_computed", False) == True, "library"
    ]  # noqa: E712
    if len(computed):
        print(
            "\nPrecursor m/z looks computed (most spectra within 0.2 ppm of the "
            "theoretical value): " + ", ".join(computed)
        )
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"Saved {args.out}")
