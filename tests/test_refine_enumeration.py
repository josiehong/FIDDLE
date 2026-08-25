"""Tests for the Tier 2 enumeration refinement (complete bounded search).

formula_refinement dispatches to a vectorized enumeration of the capped search
box when it is tractable (all production configs) and falls back to the BFS
otherwise. These tests pin the dispatch behaviour and the enumeration's
guarantees: every returned formula is valid, the list is proximity-ranked, and
the enumeration never misses a formula the BFS can find.
"""

import pytest
from molmass import Formula

from msfiddle.utils.refine_utils import (
    _enumerate_valid_formulas,
    exceed_refine_atom_limit,
    fast_mass,
    formula_refinement,
    formula_refinement_bfs,
    parse_formula,
    passes_senior_rule,
)

# Production Q-TOF post_processing caps (halogen-extended, v2.1)
TYPES = ["C", "O", "N", "H", "Cl", "Br", "F", "I"]
NUMS = [5, 5, 5, -1, 2, 1, 2, 1]
PPM, K, D, T = 10.0, 10, 50, 0

CASES = [
    # (predicted formula, true formula) — spanning exact hits, C/N/O swaps,
    # and the halogen-introduction cases the v2.1 configs were tuned for.
    ("C8H10N4O2", "C8H10N4O2"),
    ("C9H12N2O5", "C9H11ClN2O"),
    ("C17H21NO4", "C17H21NO4"),
    ("C12H10O2", "C12H10Br2"),
    ("C6H14N2O2", "C6H12O6"),
    ("C10H14ClN", "C10H15NO"),
]


def _delta_da(M):
    return PPM * M / 1e6


@pytest.mark.parametrize("f0,target", CASES)
def test_enumeration_valid_and_sorted(f0, target):
    M = Formula(target).isotope.mass
    full = _enumerate_valid_formulas([f0], M, _delta_da(M), D, TYPES, NUMS)
    assert full is not None, "production caps must take the enumeration path"
    initial = {f0}
    dists = [abs(fast_mass(f) - M) for f in full]
    assert dists == sorted(dists)
    for f in full:
        m = fast_mass(f)
        assert M - _delta_da(M) <= m <= M + _delta_da(M)
        assert passes_senior_rule(f)
        if f not in initial:
            assert not exceed_refine_atom_limit(parse_formula(f), [f0], TYPES, NUMS)


@pytest.mark.parametrize("f0,target", CASES)
def test_enumeration_superset_of_bfs(f0, target):
    M = Formula(target).isotope.mass
    full = _enumerate_valid_formulas([f0], M, _delta_da(M), D, TYPES, NUMS)
    bfs = formula_refinement_bfs([f0], M, PPM, True, K, D, T, TYPES, NUMS)
    bfs_found = {f for f in bfs["formula"] if f is not None}
    assert bfs_found <= set(full)


def test_halogen_introduction_reaches_true_formula():
    # f0 has no Cl; the true formula does. The whole point of the halogen-
    # extended caps: the enumeration must surface the chlorinated candidate.
    f0, target = "C9H12N2O5", "C9H11ClN2O"
    M = Formula(target).isotope.mass
    res = formula_refinement([f0], M, PPM, True, K, D, T, TYPES, NUMS)
    target_counts = parse_formula(target)
    assert any(
        f is not None and parse_formula(f) == target_counts for f in res["formula"]
    )


def test_output_shape_and_padding():
    f0 = "C3H7NO2"
    M = Formula(f0).isotope.mass
    res = formula_refinement([f0], M, PPM, True, K, D, T, TYPES, NUMS)
    assert len(res["formula"]) == K and len(res["mass"]) == K
    n = sum(1 for f in res["formula"] if f is not None)
    assert res["formula"][:n] == [f for f in res["formula"] if f is not None]
    for f, m in zip(res["formula"], res["mass"]):
        assert (f is None) == (m is None)
        if f is not None:
            assert m == fast_mass(f)


def test_multi_f0_with_payload_atom():
    # One prediction carries S (not in the base cap list at call time here);
    # candidates derived from it must keep S, others must not gain it.
    f0s = ["C16H19N3O5S", "C16H21N3O5"]
    M = Formula("C16H19N3O5S").isotope.mass
    full = _enumerate_valid_formulas(f0s, M, _delta_da(M), D, TYPES, NUMS)
    assert full is not None
    assert any(parse_formula(f).get("S", 0) == 1 for f in full)
    for f in full:
        assert parse_formula(f).get("S", 0) in (0, 1)


def test_ranking_prefers_fewest_atom_changes():
    # When the predicted composition itself is valid, it must rank first even
    # if a near-isobar sits closer in mass; ranks must be non-decreasing in
    # heavy-atom edit distance from the prediction.
    f0 = "C8H10N4O2"
    M = Formula(f0).isotope.mass
    res = formula_refinement([f0], M, PPM, True, K, D, T, TYPES, NUMS)
    assert res["formula"][0] is not None
    assert parse_formula(res["formula"][0]) == parse_formula(f0)
    f0c = parse_formula(f0)

    def hdist(f):
        c = parse_formula(f)
        atoms = (set(c) | set(f0c)) - {"H"}
        return sum(abs(c.get(a, 0) - f0c.get(a, 0)) for a in atoms)

    dists = [hdist(f) for f in res["formula"] if f is not None]
    assert dists == sorted(dists)


def test_uncapped_search_falls_back_to_bfs():
    # All-unlimited caps over many atoms: the box is intractable, the
    # enumeration must decline, and formula_refinement must still answer.
    atoms = ["C", "H", "N", "O", "S", "P", "F", "Cl", "B", "Br", "I", "Na", "K"]
    nums = [-1] * len(atoms)
    f0 = "C8H10N4O2"
    M = Formula(f0).isotope.mass
    assert _enumerate_valid_formulas([f0], M, _delta_da(M), 10, atoms, nums) is None
    res = formula_refinement([f0], M, 5, True, 5, 10, 0, atoms, nums)
    bfs = formula_refinement_bfs([f0], M, 5, True, 5, 10, 0, atoms, nums)
    assert res == bfs
