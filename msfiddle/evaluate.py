"""Top-k formula accuracy of a prediction CSV against known formulas.

A result row is correct at top k if the true formula's atom-count vector
matches one of its first k ``Refined Formula (i)`` columns. Accuracy is
reported two ways:

- per spectrum (micro): the fraction of spectra whose true formula is in the
  top k;
- per compound (macro): the mean over compounds of each compound's
  per-spectrum accuracy, so heavily replicated compounds do not dominate.

The truth is the intact molecule from each MGF spectrum's ``SMILES`` field, or
its ``FORMULA`` field when there is no SMILES. Predictions are intact-molecule
formulas too: neutral losses in the adduct (e.g. ``[M+H-H2O]+``) are added back
after the model predicts the formula without them.

``topk_details`` gives the per-spectrum rows behind these numbers, with each
spectrum's metadata, and ``grouped_accuracy`` and ``coverage`` summarize them.
A missing ``true_rank`` only means the true formula is not among the returned
candidates: the result CSV does not hold the full refinement candidate pool, so
it cannot tell a ranking miss from a candidate-generation miss.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .utils.mol_utils import formula_to_vector, vector_to_formula

_VEC_CACHE: dict[str, Optional[tuple[int, ...]]] = {}


def formula_vec(formula) -> Optional[tuple[int, ...]]:
    """Formula string to an atom-count tuple (cached); None if empty or invalid."""
    if not isinstance(formula, str):
        return None
    f = formula.strip()
    if f == "" or f.lower() == "nan":
        return None
    if f not in _VEC_CACHE:
        try:
            _VEC_CACHE[f] = tuple(int(round(x)) for x in formula_to_vector(f))
        except Exception:
            _VEC_CACHE[f] = None
    return _VEC_CACHE[f]


def _read_mgf(path: str | os.PathLike[str]) -> list[dict]:
    """Each spectrum's params plus its peak count, in file order."""
    from pyteomics import mgf

    # Sequential read: the indexed reader drops untitled and repeated titles.
    return [
        {**spec.get("params", {}), "_peak_count": len(spec.get("m/z array", []))}
        for spec in mgf.read(str(path), use_index=False)
    ]


def _truth_from_params(params: dict) -> Optional[tuple]:
    smiles, formula = params.get("smiles"), params.get("formula")
    vec = None
    if smiles:
        from rdkit import Chem
        from rdkit.Chem.rdMolDescriptors import CalcMolFormula

        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            vec = formula_vec(CalcMolFormula(mol))
    if vec is None and formula:
        vec = formula_vec(formula)
    return None if vec is None else (vec, smiles or vec)


def load_truth(path: str | os.PathLike[str]) -> dict[str, tuple]:
    """Map each MGF spectrum title to (formula vector, compound key).

    A repeated title takes its last spectrum's truth (prediction keeps the
    last usable spectrum of a title, normally of the same compound).
    """
    return _truth_from_records(_read_mgf(path))


def _truth_from_records(records: list[dict]) -> dict[str, tuple]:
    truth = {}
    for params in records:
        title = params.get("title")
        if not title:
            continue
        found = _truth_from_params(params)
        if found is not None:
            truth[str(title)] = found
        else:
            truth.pop(str(title), None)
    return truth


@dataclass
class EvalInputs:
    """What evaluation needs from the test MGF, read in one pass."""

    truth: dict[str, tuple]
    info: pd.DataFrame  # one row per title: precursor_mz, adduct, CE, peaks
    n_input: int  # spectra in the file
    n_duplicate_titles: int  # spectra whose title an earlier one already used


def load_eval_inputs(path: str | os.PathLike[str]) -> EvalInputs:
    records = _read_mgf(path)
    info = {}
    titles = []
    for params in records:
        title = params.get("title")
        if not title:
            continue
        titles.append(str(title))
        mz = params.get("precursor_mz", params.get("pepmass"))
        if isinstance(mz, (tuple, list)):
            mz = mz[0]
        info[str(title)] = {
            "precursor_mz": pd.to_numeric(mz, errors="coerce"),
            "adduct": params.get("precursor_type"),
            "collision_energy": params.get("collision_energy"),
            "peak_count": params["_peak_count"],
        }
    info_df = pd.DataFrame.from_dict(
        info,
        orient="index",
        columns=["precursor_mz", "adduct", "collision_energy", "peak_count"],
    )
    return EvalInputs(
        truth=_truth_from_records(records),
        info=info_df,
        n_input=len(records),
        n_duplicate_titles=len(titles) - len(set(titles)),
    )


@dataclass
class TopKResult:
    spectrum_acc: list[float]  # cumulative top-1..top-k, per spectrum
    compound_acc: list[float]  # cumulative top-1..top-k, per compound
    n_spectra: int  # result rows joined to the truth
    n_compounds: int
    n_rows: int  # result rows in total

    def to_frame(self) -> pd.DataFrame:
        """One row per k: cumulative top-k accuracy per spectrum and compound."""
        return pd.DataFrame(
            {
                "k": range(1, len(self.spectrum_acc) + 1),
                "spectrum_acc": self.spectrum_acc,
                "compound_acc": self.compound_acc,
                "n_spectra": self.n_spectra,
                "n_compounds": self.n_compounds,
            }
        )

    def format(self) -> str:
        k = len(self.spectrum_acc)
        return "\n".join(
            [
                "Joined: {} spectra / {} compounds  (matched {}/{} result rows)".format(
                    self.n_spectra, self.n_compounds, self.n_spectra, self.n_rows
                ),
                "          " + "".join("  top-{:<2d}".format(i + 1) for i in range(k)),
                "spectra "
                + "".join("  {:5.1f}%".format(100 * a) for a in self.spectrum_acc),
                "compound"
                + "".join("  {:5.1f}%".format(100 * a) for a in self.compound_acc),
            ]
        )


def _refined_columns(result: pd.DataFrame, topk: Optional[int]) -> list[str]:
    if "ID" not in result.columns:
        raise ValueError("result has no 'ID' column")
    cols = sorted(
        (c for c in result.columns if c.startswith("Refined Formula (")),
        key=lambda c: int(c.split("(")[1].rstrip(")")),
    )
    if not cols:
        raise ValueError("result has no 'Refined Formula (k)' columns")
    topk = len(cols) if topk is None else min(topk, len(cols))
    return cols[:topk]


def topk_details(
    result: pd.DataFrame,
    truth: dict[str, tuple],
    topk: Optional[int] = None,
    info: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """One row per result row, then one per truth spectrum with no result row.

    ``eval_status`` is ``evaluated`` (has a truth; counted in the accuracy),
    ``no_truth`` (no SMILES or FORMULA for its ID; excluded) or
    ``not_predicted`` (in the truth but not in the result, e.g. rejected by
    validation; excluded). ``true_rank`` is 1-based and empty when the true
    formula is not among the first ``topk`` candidates; ``top{k}_correct`` is
    cumulative. ``info`` (from ``load_eval_inputs``) adds spectrum metadata.
    """
    cols = _refined_columns(result, topk)
    k = len(cols)
    rows = []
    for _, row in result.iterrows():
        key = str(row["ID"])
        rec = {"ID": key, "top1_formula": row[cols[0]]}
        if key not in truth:
            rec["eval_status"] = "no_truth"
            rows.append(rec)
            continue
        true_vec, compound = truth[key]
        rank = next(
            (i for i, c in enumerate(cols) if formula_vec(row[c]) == true_vec), None
        )
        rec.update(
            eval_status="evaluated",
            compound=compound,
            true_formula=vector_to_formula(true_vec),
            true_rank=None if rank is None else rank + 1,
        )
        rows.append(rec)
    predicted = {r["ID"] for r in rows}
    for key, (true_vec, compound) in truth.items():
        if key not in predicted:
            rows.append(
                {
                    "ID": key,
                    "eval_status": "not_predicted",
                    "compound": compound,
                    "true_formula": vector_to_formula(true_vec),
                }
            )

    columns = ["ID", "eval_status", "compound", "true_formula", "top1_formula"]
    details = pd.DataFrame(rows).reindex(columns=columns + ["true_rank"])
    details["true_rank"] = details["true_rank"].astype("Int64")
    evaluated = details["eval_status"] == "evaluated"
    for i in range(1, k + 1):
        hit = evaluated & details["true_rank"].le(i).fillna(False)
        details[f"top{i}_correct"] = hit.where(evaluated).astype("boolean")
    if info is not None:
        details = details.join(info, on="ID")
    return details


def _accuracy_from_details(details: pd.DataFrame) -> tuple[list, list, int, int]:
    ev = details[details["eval_status"] == "evaluated"]
    hit_cols = [c for c in details.columns if re.fullmatch(r"top\d+_correct", c)]
    hits = ev[hit_cols].astype(float)
    spectrum_acc = hits.mean().tolist()
    groups = hits.groupby(ev["compound"].map(_compound_key), sort=False).mean()
    compound_acc = groups.mean().tolist()
    return spectrum_acc, compound_acc, len(ev), len(groups)


def _compound_key(compound):
    return compound if isinstance(compound, str) else ("vec", tuple(compound))


def topk_accuracy(
    result: pd.DataFrame, truth: dict[str, tuple], topk: Optional[int] = None
) -> TopKResult:
    """Cumulative top-1..top-k accuracy of ``result`` rows that have a truth.

    ``topk`` defaults to every ``Refined Formula`` column.
    """
    details = topk_details(result, truth, topk)
    if not (details["eval_status"] == "evaluated").any():
        raise ValueError(
            "no result rows joined; check that 'ID' matches the test titles"
        )
    spectrum_acc, compound_acc, n, n_compounds = _accuracy_from_details(details)
    return TopKResult(spectrum_acc, compound_acc, n, n_compounds, len(result))


_MZ_BINS = [0, 200, 400, 600, 800, np.inf]
_PEAK_BINS = [0, 10, 20, 50, np.inf]
_NCE_BINS = [0, 20, 40, 60, np.inf]


def _bin_labels(edges, unit=""):
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        labels.append(f">={lo:g}{unit}" if np.isinf(hi) else f"{lo:g}-{hi:g}{unit}")
    return labels


def _nce(row) -> float:
    """The NCE the model is given for a spectrum, NaN if not parseable."""
    from .utils.pkl_utils import is_stepped_collision_energy, parse_collision_energy

    ce, mz = row.get("collision_energy"), row.get("precursor_mz")
    if ce is None or pd.isna(ce) or pd.isna(mz) or is_stepped_collision_energy(ce):
        return np.nan
    match = re.search(r"\](\d*)[+-]$", str(row.get("adduct") or ""))
    charge = int(match.group(1) or 1) if match else 1
    try:
        _, nce = parse_collision_energy(str(ce), float(mz), min(charge, 8))
    except (KeyError, ValueError, ZeroDivisionError):
        return np.nan
    return np.nan if nce is None else nce


def grouped_accuracy(details: pd.DataFrame) -> pd.DataFrame:
    """Per-spectrum cumulative top-k accuracy of evaluated rows, by precursor
    m/z, adduct, peak count and NCE bin. NCE is the model's input (eV converted
    with the precursor m/z); stepped or unparseable energies are 'unknown'."""
    ev = details[details["eval_status"] == "evaluated"].copy()
    hit_cols = [c for c in details.columns if re.fullmatch(r"top\d+_correct", c)]
    groups = {}
    if "precursor_mz" in ev:
        groups["precursor_mz"] = pd.cut(
            ev["precursor_mz"], _MZ_BINS, right=False, labels=_bin_labels(_MZ_BINS)
        )
    if "adduct" in ev:
        groups["adduct"] = ev["adduct"]
    if "peak_count" in ev:
        groups["peak_count"] = pd.cut(
            ev["peak_count"], _PEAK_BINS, right=False, labels=_bin_labels(_PEAK_BINS)
        )
    if "collision_energy" in ev:
        nce = ev.apply(_nce, axis=1) if len(ev) else pd.Series(dtype=float)
        groups["nce"] = pd.cut(
            nce, _NCE_BINS, right=False, labels=_bin_labels(_NCE_BINS)
        )

    frames = []
    for name, key in groups.items():
        # Bins in their natural order, other values sorted, 'unknown' last
        if isinstance(key.dtype, pd.CategoricalDtype):
            order = list(key.cat.categories)
        else:
            order = sorted(key.dropna().astype(str).unique())
        key = key.astype(object).where(key.notna(), "unknown").astype(str)
        g = ev.assign(_key=key.values).groupby("_key")
        frame = g[hit_cols].mean()
        frame.columns = [c.replace("_correct", "_acc") for c in hit_cols]
        frame.insert(
            0,
            "n_compounds",
            g["compound"].agg(lambda c: c.map(_compound_key).nunique()),
        )
        frame.insert(0, "n_spectra", g.size())
        frame = frame.reindex([v for v in order + ["unknown"] if v in frame.index])
        frame = frame.rename_axis("value").reset_index()
        frame.insert(0, "group", name)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["group", "value", "n_spectra", "n_compounds"])
    return pd.concat(frames, ignore_index=True)


def coverage(
    details: pd.DataFrame, n_result_rows: int, inputs: Optional[EvalInputs] = None
) -> dict[str, int]:
    """Counts behind the accuracy denominators."""
    status = details["eval_status"].value_counts()
    result_ids = details.loc[details["eval_status"] != "not_predicted", "ID"]
    out = {}
    if inputs is not None:
        out["input_spectra"] = inputs.n_input
        out["duplicate_input_titles"] = inputs.n_duplicate_titles
        out["spectra_with_truth"] = len(inputs.truth)
    out.update(
        result_rows=n_result_rows,
        duplicate_result_ids=int(result_ids.duplicated().sum()),
        evaluated_rows=int(status.get("evaluated", 0)),
        result_rows_without_truth=int(status.get("no_truth", 0)),
        truth_spectra_not_predicted=int(status.get("not_predicted", 0)),
    )
    return out
