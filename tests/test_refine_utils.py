import pytest
from molmass import Formula

from msfiddle.utils.refine_utils import (
    ISOTOPE_MASS,
    _enumerate_valid_formulas,
    fast_mass,
    format_formula,
    formula_refinement,
    formula_refinement_bfs,
    parse_formula,
    passes_senior_rule,
)

# Production Q-TOF post_processing caps (halogen-extended)
TYPES = ["C", "O", "N", "H", "Cl", "Br", "F", "I"]
NUMS = [5, 5, 5, -1, 2, 1, 2, 1]
PPM, K, D, T = 10.0, 10, 50, 0


def _window(M):
    return PPM * M / 1e6


def test_parse_and_format_formula():
    assert parse_formula("C10H12N2O3") == {"C": 10, "H": 12, "N": 2, "O": 3}
    assert parse_formula("C2H5Cl") == {"C": 2, "H": 5, "Cl": 1}
    assert format_formula({"H": 4, "C": 1}) == "CH4"
    assert parse_formula(format_formula({"C": 6, "H": 6})) == {"C": 6, "H": 6}


@pytest.mark.parametrize(
    "formula, valid",
    [
        ("CH4", True),
        ("C6H6", True),
        ("C4H4N2O2", True),
        ("N2", True),
        ("CH3", False),
        ("HN", False),
        ("N", False),
    ],
)
def test_senior_rule(formula, valid):
    assert passes_senior_rule(formula) is valid


@pytest.mark.parametrize("el", ISOTOPE_MASS.keys())
def test_fast_mass_uses_most_abundant_isotope(el):
    # Catches the boron B-10 vs B-11 trap.
    assert fast_mass(el) == pytest.approx(Formula(el).isotope.mass, abs=1e-4)


@pytest.mark.parametrize(
    "f0, target",
    [
        ("C8H10N4O2", "C8H10N4O2"),
        ("C9H12N2O5", "C9H11ClN2O"),
        ("C12H10O2", "C12H10Br2"),
        ("C6H14N2O2", "C6H12O6"),
        ("C10H14ClN", "C10H15NO"),
    ],
)
def test_enumeration_is_valid_sorted_and_capped(f0, target):
    M = Formula(target).isotope.mass
    full = _enumerate_valid_formulas([f0], M, _window(M), D, TYPES, NUMS)
    assert full is not None, "production caps must take the enumeration path"
    dists = [abs(fast_mass(f) - M) for f in full]
    assert dists == sorted(dists)
    f0c = parse_formula(f0)
    for f in full:
        assert abs(fast_mass(f) - M) <= _window(M)
        assert passes_senior_rule(f)
        c = parse_formula(f)
        for t, n in zip(TYPES, NUMS):
            assert n == -1 or abs(c.get(t, 0) - f0c.get(t, 0)) <= n


def test_refinement_introduces_missing_halogen():
    f0, target = "C9H12N2O5", "C9H11ClN2O"
    res = formula_refinement(
        [f0], Formula(target).isotope.mass, PPM, True, K, D, T, TYPES, NUMS
    )
    assert parse_formula(target) in [parse_formula(f) for f in res["formula"] if f]


def test_refinement_ranks_prediction_first_and_pads():
    f0 = "C8H10N4O2"
    res = formula_refinement(
        [f0], Formula(f0).isotope.mass, PPM, True, K, D, T, TYPES, NUMS
    )
    assert len(res["formula"]) == len(res["mass"]) == K
    assert parse_formula(res["formula"][0]) == parse_formula(f0)
    found = [f for f in res["formula"] if f is not None]
    assert res["formula"][: len(found)] == found  # None only as trailing padding
    for f, m in zip(res["formula"], res["mass"]):
        assert (f is None) == (m is None)

    # Non-decreasing heavy-atom edit distance from the prediction
    f0c = parse_formula(f0)

    def hdist(f):
        c = parse_formula(f)
        return sum(
            abs(c.get(a, 0) - f0c.get(a, 0)) for a in (set(c) | set(f0c)) - {"H"}
        )

    dists = [hdist(f) for f in found]
    assert dists == sorted(dists)


def test_enumeration_keeps_payload_atom():
    # S is outside the cap list: candidates may carry it from a prediction
    # that has it, but never gain more.
    f0s = ["C16H19N3O5S", "C16H21N3O5"]
    M = Formula("C16H19N3O5S").isotope.mass
    full = _enumerate_valid_formulas(f0s, M, _window(M), D, TYPES, NUMS)
    assert any(parse_formula(f).get("S", 0) == 1 for f in full)
    assert all(parse_formula(f).get("S", 0) in (0, 1) for f in full)


def test_uncapped_search_falls_back_to_bfs():
    atoms = ["C", "H", "N", "O", "S", "P", "F", "Cl", "B", "Br", "I", "Na", "K"]
    nums = [-1] * len(atoms)
    f0 = "C8H10N4O2"
    M = Formula(f0).isotope.mass
    assert _enumerate_valid_formulas([f0], M, _window(M), 10, atoms, nums) is None
    assert formula_refinement(
        [f0], M, 5, True, 5, 10, 0, atoms, nums
    ) == formula_refinement_bfs([f0], M, 5, True, 5, 10, 0, atoms, nums)
