"""Read timsTOF libraries in the Enveda CASMI 2026 layout into FIDDLE spectra.

The Enveda CASMI 2026 Kaggle training set (``train.parquet``) and the
Enveda-180 deposit (Zenodo record 20436851) hold one MS/MS spectrum per row
with the peak lists as list columns. Column names differ slightly between the
two, so fields are looked up through ``COLUMN_ALIASES``. Rows become the
pyteomics-style spectrum dicts that ``filter_spec`` and ``spec2arr`` read, so
scripts/prepare_msms.py treats them like any other library.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

INSTRUMENT_TYPE = "timsTOF"  # as spelled in train.parquet's instrument_type column
DEFAULT_LIBRARY_PREFIX = "enveda"  # the timsTOF libraries, when there is no such column

# FIDDLE field -> candidate column names; the first present column wins,
# except for the collision energy, where the first non-null value per row wins
COLUMN_ALIASES = {
    "mz": ("ms2_mzs", "mzs", "mz_array"),
    "intensity": (
        "ms2_normalized_intensities",
        "ms2_intensities",
        "intensities",
        "intensity_array",
    ),
    "smiles": ("normalized_smiles", "smiles"),
    "precursor_mz": ("precursor_mz",),
    "adduct": ("adduct", "precursor_type"),
    "collision_energy": (
        "collision_energy_ev",
        "collision_energy",
        "collision_energy_orig",
    ),
    "library": ("ingest_lib", "library", "source"),
    "instrument": ("instrument_type", "instrument"),
    "id": ("spectrum_id", "molecule_id"),
    "inchikey": ("inchikey", "inchi_key", "inchikey14"),
}
REQUIRED = ("mz", "intensity", "smiles", "precursor_mz", "adduct")

# Adduct spellings in these libraries -> the spelling FIDDLE encodes
ADDUCT_ALIASES = {
    "[M+FA-H]-": "[M+CH2O2-H]-",
    "[M+HCOO]-": "[M+CH2O2-H]-",
    "[M+HCO2]-": "[M+CH2O2-H]-",
    "[M+HCOOH-H]-": "[M+CH2O2-H]-",
}


def normalize_adduct(adduct: Any) -> str:
    """'[M+H]1+' -> '[M+H]+', '[M+FA-H]-' -> '[M+CH2O2-H]-'. Unknown adducts
    are returned stripped of whitespace, so the precursor-type filter drops them."""
    text = re.sub(r"\s+", "", str(adduct))
    text = re.sub(r"\]1([+-])$", r"]\1", text)  # explicit charge 1
    return ADDUCT_ALIASES.get(text, text)


def collision_energy_string(value: Any) -> str:
    """The string parse_collision_energy reads: numbers become '40.0', lists
    '20,40,60' (a stepped energy), negative energies (negative mode on Bruker
    instruments) lose their sign and missing values become 'Unknown'."""
    if isinstance(value, (list, tuple, np.ndarray)):
        parts = [p for p in map(collision_energy_string, value) if p != "Unknown"]
        return ",".join(parts) if parts else "Unknown"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return "Unknown" if math.isnan(value) else str(abs(float(value)))
    text = "" if value is None else str(value).strip()
    if text == "" or text.lower() in ("nan", "none", "null"):
        return "Unknown"
    return str(abs(float(text))) if re.fullmatch(r"-?\d+(\.\d+)?", text) else text


def resolve_columns(names: Iterable[str]) -> dict[str, list[str]]:
    """Map each FIDDLE field to the parquet columns that provide it."""
    names = set(names)
    resolved = {
        f: [c for c in cols if c in names] for f, cols in COLUMN_ALIASES.items()
    }
    missing = [f for f in REQUIRED if not resolved[f]]
    if missing:
        raise ValueError(
            "Parquet file lacks columns for {}; known names: {}".format(
                ", ".join(missing), {f: COLUMN_ALIASES[f] for f in missing}
            )
        )
    return {f: cols for f, cols in resolved.items() if cols}


def _first_non_null(row: dict[str, Any], columns: list[str]) -> Any:
    for column in columns:
        value = row.get(column)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return None


def row_to_spectrum(row: dict[str, Any], columns: dict[str, list[str]], index: int):
    """One parquet row (a dict) as a FIDDLE spectrum dict, or None when the
    peak list or a required field is missing."""
    mz, intensity = row.get(columns["mz"][0]), row.get(columns["intensity"][0])
    smiles = _first_non_null(row, columns["smiles"])
    precursor_mz = _first_non_null(row, columns["precursor_mz"])
    adduct = _first_non_null(row, columns["adduct"])
    if (
        not mz
        or not intensity
        or len(mz) != len(intensity)
        or None in (smiles, precursor_mz, adduct)
    ):
        return None
    adduct = normalize_adduct(adduct)
    instrument = _first_non_null(row, columns.get("instrument", [])) or INSTRUMENT_TYPE
    library = str(_first_non_null(row, columns.get("library", [])) or "unknown")
    spectrum_id = _first_non_null(row, columns.get("id", []))
    return {
        "params": {
            "title": f"{library if spectrum_id is None else spectrum_id}_{index}",
            "precursor_type": adduct,
            "precursor_mz": float(precursor_mz),
            "molmass": "Unknown",
            "ms_level": "MS2",
            "ionmode": "N" if adduct.endswith("-") else "P",
            "source_instrument": library,
            "instrument_type": str(instrument),
            "collision_energy": collision_energy_string(
                _first_non_null(row, columns.get("collision_energy", []))
            ),
            "smiles": str(smiles),
            "inchi_key": str(
                _first_non_null(row, columns.get("inchikey", [])) or "Unknown"
            ),
        },
        "m/z array": np.asarray(mz, dtype=float),
        "intensity array": np.asarray(intensity, dtype=float),
    }


def select_libraries(
    available: Iterable[str], requested: Iterable[str] | None
) -> list[str] | None:
    """The requested libraries (checked against the available ones), else None
    when the file has an instrument column to select on, else every library
    whose name starts with DEFAULT_LIBRARY_PREFIX, else all of them."""
    available = sorted({str(a) for a in available if a is not None})
    if requested:
        unknown = [r for r in requested if r not in available]
        if unknown:
            raise ValueError(f"Unknown libraries {unknown}; available: {available}")
        return list(requested)
    chosen = [a for a in available if a.lower().startswith(DEFAULT_LIBRARY_PREFIX)]
    return chosen or available


def iter_spectra(
    path: str,
    libraries: Iterable[str] | None = None,
    max_spectra_per_compound: int = 0,
    batch_size: int = 65536,
    verbose: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yield FIDDLE spectrum dicts from a CASMI 2026 / Enveda-180 parquet file.

    ``libraries`` restricts the rows to those ``ingest_lib`` values. Without
    it, the rows whose instrument column says timsTOF are read (train.parquet
    marks the two Enveda libraries so), or the Enveda libraries when there is
    no instrument column. ``max_spectra_per_compound`` keeps the first N
    spectra of each SMILES in file order (0 = all).
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    reader = pq.ParquetFile(path)
    columns = resolve_columns(reader.schema_arrow.names)
    needed = sorted({c for cols in columns.values() for c in cols})
    # row selection: by library when requested, else by the instrument column
    select_column = chosen = None
    if libraries or ("library" in columns and "instrument" not in columns):
        if "library" not in columns:
            raise ValueError("The parquet file has no library column to select on")
        select_column = columns["library"][0]
        available = pq.read_table(path, columns=[select_column]).column(0).unique()
        chosen = select_libraries(available.to_pylist(), libraries)
    elif "instrument" in columns:
        select_column, chosen = columns["instrument"][0], [INSTRUMENT_TYPE]
    if verbose and chosen:
        print(f"Read rows with {select_column} in {chosen} from {path}")

    per_compound: Counter = Counter()
    summary: dict[str, Counter] = {
        k: Counter() for k in ("library", "adduct", "energy")
    }
    index = kept = 0
    for batch in reader.iter_batches(batch_size=batch_size, columns=needed):
        if chosen is not None:
            batch = batch.filter(
                pc.is_in(batch.column(select_column), value_set=pa.array(chosen))
            )
        for row in batch.to_pylist():
            index += 1
            spectrum = row_to_spectrum(row, columns, index)
            if spectrum is None:
                continue
            params = spectrum["params"]
            if max_spectra_per_compound:
                if per_compound[params["smiles"]] >= max_spectra_per_compound:
                    continue
                per_compound[params["smiles"]] += 1
            kept += 1
            for key, field in (
                ("library", "source_instrument"),
                ("adduct", "precursor_type"),
                ("energy", "collision_energy"),
            ):
                summary[key][params[field]] += 1
            yield spectrum
    if verbose:
        print(f"Read {kept} of {index} rows from {path}")
        for key, counts in summary.items():
            print(f"  {key}: {dict(counts.most_common(12))}")
