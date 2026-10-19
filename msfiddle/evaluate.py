"""Top-k formula accuracy of a prediction CSV against known formulas.

A result row is correct at top k if the true formula's atom-count vector
matches one of its first k ``Refined Formula (i)`` columns. Accuracy is
reported two ways:

- per spectrum (micro): the fraction of spectra whose true formula is in the
  top k;
- per compound (macro): the mean over compounds of each compound's
  per-spectrum accuracy, so heavily replicated compounds do not dominate.

The truth comes from a test ``.pkl`` (the stored training target, already
adjusted for neutral-loss adducts) or an ``.mgf`` (``SMILES`` if present,
otherwise ``FORMULA``).
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .utils.mol_utils import formula_to_vector

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


def load_truth(path: str | os.PathLike[str]) -> dict[str, tuple]:
    """Map spectrum title to (formula vector, compound key)."""
    path = str(path)
    truth = {}
    if path.endswith(".pkl"):
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        for d in data:
            title, fvec = d.get("title"), d.get("formula")
            if title is None or fvec is None:
                continue
            vec = tuple(int(round(float(x))) for x in fvec)
            truth[str(title)] = (vec, d.get("smiles") or vec)
        return truth

    from pyteomics import mgf

    for spec in mgf.read(path):
        p = spec.get("params", {})
        title, smiles, formula = p.get("title"), p.get("smiles"), p.get("formula")
        if not title:
            continue
        vec = None
        if smiles:
            from rdkit import Chem
            from rdkit.Chem.rdMolDescriptors import CalcMolFormula

            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                vec = formula_vec(CalcMolFormula(mol))
        if vec is None and formula:
            vec = formula_vec(formula)
        if vec is not None:
            truth[str(title)] = (vec, smiles or vec)
    return truth


@dataclass
class TopKResult:
    spectrum_acc: list[float]  # cumulative top-1..top-k, per spectrum
    compound_acc: list[float]  # cumulative top-1..top-k, per compound
    n_spectra: int  # result rows joined to the truth
    n_compounds: int
    n_rows: int  # result rows in total

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


def topk_accuracy(
    result: pd.DataFrame, truth: dict[str, tuple], topk: int = 5
) -> TopKResult:
    """Cumulative top-1..top-k accuracy of ``result`` rows that have a truth."""
    if "ID" not in result.columns:
        raise ValueError("result has no 'ID' column")
    cols = sorted(
        (c for c in result.columns if c.startswith("Refined Formula (")),
        key=lambda c: int(c.split("(")[1].rstrip(")")),
    )
    if not cols:
        raise ValueError("result has no 'Refined Formula (k)' columns")
    topk = min(topk, len(cols))
    cols = cols[:topk]

    spectrum_hits = []
    compound_hits: dict = {}
    for _, row in result.iterrows():
        key = str(row["ID"])
        if key not in truth:
            continue
        true_vec, compound = truth[key]
        rank = next(
            (i for i, c in enumerate(cols) if formula_vec(row[c]) == true_vec), None
        )
        hits = [int(rank is not None and rank <= k) for k in range(topk)]
        spectrum_hits.append(hits)
        compound_hits.setdefault(compound, []).append(hits)

    n = len(spectrum_hits)
    if n == 0:
        raise ValueError(
            "no result rows joined; check that 'ID' matches the test titles"
        )
    spectrum_acc = [sum(h[k] for h in spectrum_hits) / n for k in range(topk)]
    per_compound = [
        [sum(h[k] for h in hs) / len(hs) for k in range(topk)]
        for hs in compound_hits.values()
    ]
    compound_acc = [
        sum(m[k] for m in per_compound) / len(per_compound) for k in range(topk)
    ]
    return TopKResult(spectrum_acc, compound_acc, n, len(per_compound), len(result))
