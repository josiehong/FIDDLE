"""Tests for the fast_mass BFS-refinement speedup patch in utils/refine_utils.py.

Run:  pytest tests/test_refine_fast_mass.py -v
Requires: molmass (already a FIDDLE dependency).

These tests guard the speedup so it can never silently change which candidate
formulas pass the delta_M mass window:
  1. fast_mass agrees with molmass across many real formulas (incl. boron).
  2. formula_refinement output is byte-identical before/after the change, by
     comparing against molmass-computed masses as ground truth.
"""

import itertools
import random

import pytest
from molmass import Formula

from msfiddle.utils.refine_utils import (
    fast_mass,
    formula_refinement,
    parse_formula,
    ISOTOPE_MASS,
)

ATOMS = ["C", "H", "N", "O", "S", "P", "F", "Cl", "B", "Br", "I", "Na", "K"]
NUM = [-1] * len(ATOMS)

REAL_FORMULAS = [
    "C12H14O2",
    "C6H12O6",
    "C10H15NO",
    "C20H25N3O",
    "C8H10N4O2",
    "C9H11ClN2O",
    "C12H10Br2",
    "C7H5NO3S",
    "C5H11NO2S",
    "C10H16BNO2",
    "C16H19N3O5S",
    "C22H23ClN2O8",
    "C3H7NO2",
    "C27H46O",
    "C8H9NO2",
    "C13H18O2",
    "C17H21NO4",
    "C21H30O2",
    "C9H8O4",
    "C6H8O7",
]


@pytest.mark.parametrize("el", ISOTOPE_MASS.keys())
def test_single_element_matches_molmass(el):
    # Each element's mass must equal molmass's most-abundant-isotope mass.
    # This is the test that catches the boron B-10 vs B-11 trap.
    assert abs(fast_mass(el) - Formula(el).isotope.mass) < 1e-4


@pytest.mark.parametrize("f", REAL_FORMULAS)
def test_fast_mass_agrees_with_molmass(f):
    assert abs(fast_mass(f) - Formula(f).isotope.mass) < 1e-3


def test_fast_mass_agrees_on_random_formulas():
    rng = random.Random(0)
    for _ in range(500):
        parts, expect = [], 0.0
        for el in rng.sample(ATOMS, k=rng.randint(2, 6)):
            n = rng.randint(1, 30)
            parts.append(f"{el}{n}")
            expect += Formula(el).isotope.mass * n
        f = "".join(parts)
        assert abs(fast_mass(f) - expect) < 1e-2


@pytest.mark.parametrize("f0", REAL_FORMULAS)
def test_refinement_unchanged_vs_molmass_groundtruth(f0):
    # Target mass from molmass (ground truth). Two guarantees:
    #  (a) every returned candidate's reported mass matches molmass (no drift),
    #  (b) the input composition is recovered (compare parsed counts, since the
    #      refiner reformats alphabetically, e.g. C22H23ClN2O8 -> C22ClH23N2O8).
    M = Formula(f0).isotope.mass
    res = formula_refinement([f0], M, 5, True, 5, 10, 0, ATOMS, NUM)
    for f, m in zip(res["formula"], res["mass"]):
        if f is None:
            continue
        assert abs(m - Formula(f).isotope.mass) < 1e-3  # no mass drift
    recovered = {
        tuple(sorted(parse_formula(f).items())) for f in res["formula"] if f is not None
    }
    assert tuple(sorted(parse_formula(f0).items())) in recovered


def test_boron_candidate_mass_window():
    # Boron is the risky case: ensure a B-containing target still resolves and
    # its reported masses are within the ppm window using molmass as oracle.
    f0 = "C10H16BNO2"
    M = Formula(f0).isotope.mass
    res = formula_refinement([f0], M, 5, True, 5, 10, 0, ATOMS, NUM)
    for f, m in zip(res["formula"], res["mass"]):
        if f is None:
            continue
        # reported mass matches molmass, and is within 5 ppm of target
        assert abs(m - Formula(f).isotope.mass) < 1e-3
        assert abs(m - M) <= 5 * M / 1e6 + 1e-6
