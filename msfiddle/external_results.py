"""Load external formula candidates from BUDDY/msbuddy and SIRIUS outputs."""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Iterable

import pandas as pd

_NORMALIZED_CSV_DEPRECATION = (
    "The msfiddle-normalized {tool} CSV format is deprecated and will be "
    "removed in msfiddle 3.0.0. Pass the native/original {tool} output "
    "(directory or summary file) instead."
)

BUDDY_FORMULA_COLUMNS = tuple(f"Pred Formula ({idx})" for idx in range(1, 6))
BUDDY_SCORE_COLUMNS = tuple(f"BUDDY Score ({idx})" for idx in range(1, 6))
BUDDY_COLUMNS = ("ID", "Adduct", *BUDDY_FORMULA_COLUMNS, *BUDDY_SCORE_COLUMNS)

SIRIUS_FORMULA_COLUMNS = tuple(f"Pred Formula ({idx})" for idx in range(1, 6))
SIRIUS_ADDUCT_COLUMNS = tuple(f"Pred Adduct ({idx})" for idx in range(1, 6))
SIRIUS_SCORE_COLUMNS = tuple(f"SIRIUS Score ({idx})" for idx in range(1, 6))
SIRIUS_COLUMNS = (
    "ID",
    *(
        value
        for idx in range(1, 6)
        for value in (
            f"Pred Formula ({idx})",
            f"Pred Adduct ({idx})",
            f"SIRIUS Score ({idx})",
        )
    ),
)

_BUDDY_NATIVE_COLUMNS = (
    "identifier",
    "adduct",
    "formula_rank_1",
    "estimated_fdr",
)
_SIRIUS_FORMULA_FILES = (
    "formula_identifications_top-",
    "formula_identifications_all",
    "formula_identifications_adducts_top-",
    "formula_identifications_adducts_all",
    "formula_identifications",
)
_SIRIUS_ID_COLUMNS = (
    "name",
    "mappingFeatureId",
    "alignedFeatureId",
    "featureId",
    "compoundId",
    "id",
)
_SIRIUS_FORMULA_COLUMNS = ("molecularFormula", "formula", "Formula")
_SIRIUS_ADDUCT_COLUMNS = ("adduct", "ionType", "ion", "precursorIonType")
_SIRIUS_SCORE_COLUMNS = (
    "SiriusScore",
    "SIRIUS Score",
    "rankingScore",
    "ZodiacScore",
    "ConfidenceScore",
    "score",
)
_SIRIUS_RANK_COLUMNS = ("rank", "formulaRank")


def load_buddy_results(path: str | Path) -> pd.DataFrame:
    """Load native/original or msfiddle-normalized BUDDY/msbuddy results.

    Native/original msbuddy inputs may be either the
    ``msbuddy_result_summary.tsv`` file or the output directory that contains
    it. If detailed ``formula_results.tsv`` files are present, their
    per-candidate FDR values are used.
    """

    resolved = _resolve_path(path, "BUDDY")
    if resolved.is_dir():
        return _load_msbuddy_directory(resolved)

    df = _read_table(resolved)
    if _has_columns(df, BUDDY_COLUMNS):
        warnings.warn(
            _NORMALIZED_CSV_DEPRECATION.format(tool="BUDDY"),
            DeprecationWarning,
            stacklevel=2,
        )
        return _normalize_buddy(df)
    if _has_columns(df, _BUDDY_NATIVE_COLUMNS):
        return _normalize_msbuddy_summary(df)
    raise ValueError(
        "Unsupported BUDDY result format. Expected a native/original msbuddy "
        "summary or output directory, or an msfiddle-normalized CSV. Native "
        "summary columns: "
        f"{', '.join(_BUDDY_NATIVE_COLUMNS)}"
    )


def load_sirius_results(path: str | Path) -> pd.DataFrame:
    """Load native/original or msfiddle-normalized SIRIUS results."""

    resolved = _resolve_path(path, "SIRIUS")
    if resolved.is_dir():
        resolved = _find_sirius_formula_file(resolved)

    df = _read_table(resolved)
    if _has_columns(df, SIRIUS_COLUMNS):
        warnings.warn(
            _NORMALIZED_CSV_DEPRECATION.format(tool="SIRIUS"),
            DeprecationWarning,
            stacklevel=2,
        )
        return _normalize_sirius(df)
    return _normalize_sirius_native(df)


def _resolve_path(path: str | Path, label: str) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} result path not found: {resolved}")
    return resolved


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return _clean_columns(pd.read_excel(path))
    if suffix == ".tsv":
        return _clean_columns(pd.read_csv(path, sep="\t"))
    if suffix == ".csv":
        return _clean_columns(pd.read_csv(path))
    return _clean_columns(pd.read_csv(path, sep=None, engine="python"))


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(column).strip().lstrip("\ufeff") for column in df.columns]
    return df


def _has_columns(df: pd.DataFrame, columns: Iterable[str]) -> bool:
    df_columns = set(df.columns)
    return all(column in df_columns for column in columns)


def _normalize_buddy(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.loc[:, BUDDY_COLUMNS].copy()
    normalized["ID"] = normalized["ID"].map(_clean_identifier)
    normalized["Adduct"] = normalized["Adduct"].map(_clean_text)
    for column in BUDDY_FORMULA_COLUMNS:
        normalized[column] = normalized[column].map(_clean_formula)
    for column in BUDDY_SCORE_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized.dropna(subset=["ID"]).reset_index(drop=True)


def _load_msbuddy_directory(path: Path) -> pd.DataFrame:
    summary_path = path / "msbuddy_result_summary.tsv"
    if not summary_path.exists():
        raise FileNotFoundError(
            "Native msbuddy output directory must contain "
            f"msbuddy_result_summary.tsv: {path}"
        )
    summary = _read_table(summary_path)
    details = _load_msbuddy_details(path, summary)
    return _normalize_msbuddy_summary(summary, details=details)


def _normalize_msbuddy_summary(
    df: pd.DataFrame,
    *,
    details: dict[str, list[tuple[str, float]]] | None = None,
) -> pd.DataFrame:
    if not _has_columns(df, _BUDDY_NATIVE_COLUMNS):
        raise ValueError(
            "Native msbuddy summary is missing required columns: "
            f"{', '.join(_BUDDY_NATIVE_COLUMNS)}"
        )

    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        spec_id = _clean_identifier(row["identifier"])
        out: dict[str, object] = {
            "ID": spec_id,
            "Adduct": _clean_text(row.get("adduct", pd.NA)),
        }

        detail_candidates = details.get(spec_id, []) if details else []
        if detail_candidates:
            for idx, (formula, score) in enumerate(detail_candidates[:5], start=1):
                out[f"Pred Formula ({idx})"] = _clean_formula(formula)
                out[f"BUDDY Score ({idx})"] = score
        else:
            for idx in range(1, 6):
                out[f"Pred Formula ({idx})"] = _clean_formula(
                    row.get(f"formula_rank_{idx}", pd.NA)
                )
                # msbuddy summary exposes estimated FDR for rank 1 only.
                out[f"BUDDY Score ({idx})"] = (
                    _to_number(row.get("estimated_fdr", pd.NA)) if idx == 1 else pd.NA
                )
        rows.append(out)
    return _normalize_buddy(pd.DataFrame(rows, columns=BUDDY_COLUMNS))


def _load_msbuddy_details(
    path: Path, summary: pd.DataFrame
) -> dict[str, list[tuple[str, float]]]:
    # Detail directories follow `{sanitized_id}_mz_{mz}_rt_{rt}`. Index by the
    # sanitized id so we don't depend on how msbuddy formats mz.
    by_sanitized_id: dict[str, str] = {}
    for _, row in summary.iterrows():
        spec_id = _clean_identifier(row["identifier"])
        if pd.isna(spec_id):
            continue
        by_sanitized_id[_sanitize_msbuddy_identifier(spec_id)] = spec_id

    details: dict[str, list[tuple[str, float]]] = {}
    for result_path in path.glob("**/formula_results.tsv"):
        folder = result_path.parent.name
        sanitized_id, _, _ = folder.partition("_mz_")
        if not sanitized_id:
            continue
        spec_id = by_sanitized_id.get(sanitized_id)
        if spec_id is None:
            continue
        df = _read_table(result_path)
        if not _has_columns(df, ("rank", "formula", "estimated_fdr")):
            continue
        df = df.copy()
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
        df["estimated_fdr"] = pd.to_numeric(df["estimated_fdr"], errors="coerce")
        df = df.sort_values("rank", na_position="last")
        candidates: list[tuple[str, float]] = []
        for _, row in df.iterrows():
            formula = _clean_formula(row.get("formula", pd.NA))
            if pd.isna(formula):
                continue
            candidates.append((str(formula), row.get("estimated_fdr", pd.NA)))
            if len(candidates) == 5:
                break
        if candidates:
            details[spec_id] = candidates
    return details


def _sanitize_msbuddy_identifier(spec_id: str) -> str:
    return spec_id.replace("/", "").replace(":", "").replace(" ", "").strip()


def _find_sirius_formula_file(path: Path) -> Path:
    candidates = [
        file_path
        for file_path in path.glob("**/*")
        if file_path.is_file()
        and file_path.suffix.lower() in {".tsv", ".csv", ".xlsx", ".xls"}
        and file_path.stem.startswith("formula_identifications")
    ]
    if not candidates:
        raise FileNotFoundError(
            "SIRIUS output directory must contain a formula_identifications "
            f"summary file: {path}"
        )

    def priority(file_path: Path) -> tuple[int, str]:
        stem = file_path.stem
        for idx, prefix in enumerate(_SIRIUS_FORMULA_FILES):
            if stem.startswith(prefix):
                return idx, file_path.name
        return len(_SIRIUS_FORMULA_FILES), file_path.name

    return sorted(candidates, key=priority)[0]


def _normalize_sirius(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.loc[:, SIRIUS_COLUMNS].copy()
    normalized["ID"] = normalized["ID"].map(_clean_identifier)
    for column in SIRIUS_FORMULA_COLUMNS:
        normalized[column] = normalized[column].map(_clean_formula)
    for column in SIRIUS_ADDUCT_COLUMNS:
        normalized[column] = normalized[column].map(_clean_text)
    for column in SIRIUS_SCORE_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized.dropna(subset=["ID"]).reset_index(drop=True)


def _normalize_sirius_native(df: pd.DataFrame) -> pd.DataFrame:
    id_column = _first_existing(df, _SIRIUS_ID_COLUMNS)
    formula_column = _first_existing(df, _SIRIUS_FORMULA_COLUMNS)
    adduct_column = _first_existing(df, _SIRIUS_ADDUCT_COLUMNS)
    score_column = _first_existing(df, _SIRIUS_SCORE_COLUMNS)
    rank_column = _first_existing(df, _SIRIUS_RANK_COLUMNS)
    if not all((id_column, formula_column, adduct_column, score_column)):
        raise ValueError(
            "Unsupported SIRIUS result format. Expected a native/original "
            "formula_identifications summary or output directory, or an "
            "msfiddle-normalized CSV. Native summaries must include "
            "identifier, formula, adduct, and score columns."
        )

    native = df.copy()
    native["_msfiddle_id"] = native[id_column].map(_clean_identifier)
    native["_msfiddle_formula"] = native[formula_column].map(_clean_formula)
    native["_msfiddle_adduct"] = native[adduct_column].map(_clean_text)
    native["_msfiddle_score"] = pd.to_numeric(native[score_column], errors="coerce")
    if rank_column is not None:
        native["_msfiddle_rank"] = pd.to_numeric(native[rank_column], errors="coerce")
    else:
        native["_msfiddle_rank"] = pd.NA
    native = native.dropna(subset=["_msfiddle_id", "_msfiddle_formula"])

    rows: list[dict[str, object]] = []
    for spec_id, group in native.groupby("_msfiddle_id", sort=False):
        group = group.sort_values(
            by=["_msfiddle_rank", "_msfiddle_score"],
            ascending=[True, False],
            na_position="last",
        )
        out: dict[str, object] = {"ID": spec_id}
        for idx, (_, row) in enumerate(group.head(5).iterrows(), start=1):
            out[f"Pred Formula ({idx})"] = row["_msfiddle_formula"]
            out[f"Pred Adduct ({idx})"] = row["_msfiddle_adduct"]
            out[f"SIRIUS Score ({idx})"] = row["_msfiddle_score"]
        rows.append(out)
    return _normalize_sirius(pd.DataFrame(rows, columns=SIRIUS_COLUMNS))


def _first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    df_columns = list(df.columns)
    for candidate in candidates:
        if candidate in df_columns:
            return candidate
    canonical_columns = {_canonical_column(column): column for column in df_columns}
    for candidate in candidates:
        match = canonical_columns.get(_canonical_column(candidate))
        if match is not None:
            return match
    return None


def _canonical_column(column: object) -> str:
    text = str(column).strip().lstrip("\ufeff")
    if text.lower().startswith("sirius_"):
        text = text[7:]
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _clean_identifier(value: object) -> object:
    text = _clean_text(value)
    if pd.isna(text):
        return pd.NA
    return str(text)


def _clean_formula(value: object) -> object:
    text = _clean_text(value)
    if pd.isna(text):
        return pd.NA
    # SIRIUS summaries may represent adduct-resolved formulas as
    # "[C20H17NO6 + H]+". Keep only the neutral molecular formula.
    match = re.match(r"^\[?([A-Z][A-Za-z0-9]*)", str(text))
    return match.group(1) if match else text


def _clean_text(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if text.lower() in {"", "na", "nan", "none", "null"}:
        return pd.NA
    return text


def _to_number(value: object) -> object:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return number if pd.notna(number) else pd.NA
