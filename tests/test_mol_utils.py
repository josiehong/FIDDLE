import pytest
from rdkit import Chem

from msfiddle.utils.mol_utils import (
    ATOMS_INDEX,
    dict_to_formula,
    formula_to_dict,
    formula_to_vector,
    monoisotopic_mass_calculator,
    vector_to_formula,
)


@pytest.mark.parametrize(
    "formula, expected",
    [
        ("C", {"C": 1}),
        ("C10H12N2O3", {"C": 10, "H": 12, "N": 2, "O": 3}),
        ("C2H5Cl", {"C": 2, "H": 5, "Cl": 1}),
        (None, {}),
    ],
)
def test_formula_to_dict(formula, expected):
    assert formula_to_dict(formula) == expected


def test_dict_to_formula_roundtrip():
    d = {"C": 10, "H": 12, "N": 2, "O": 3}
    assert formula_to_dict(dict_to_formula(d)) == d


def test_formula_vector_roundtrip():
    vec = formula_to_vector("C6H6O")
    assert len(vec) == len(ATOMS_INDEX)
    assert vec[ATOMS_INDEX["C"]] == 6 and vec[ATOMS_INDEX["N"]] == 0
    assert formula_to_dict(vector_to_formula(vec)) == {"C": 6, "H": 6, "O": 1}
    assert "H" not in vector_to_formula(vec, withH=False)


def test_monoisotopic_mass():
    assert monoisotopic_mass_calculator("CH4", "f") == pytest.approx(16.0313, abs=1e-3)
    methane = Chem.AddHs(Chem.MolFromSmiles("C"))
    assert monoisotopic_mass_calculator(methane, "mol") == pytest.approx(
        16.0313, abs=1e-3
    )
    with pytest.raises(AssertionError):
        monoisotopic_mass_calculator("CH4", "invalid")
