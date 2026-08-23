import pytest
from msfiddle.utils.mol_utils import (
    ATOMS_INDEX,
    formula_to_dict,
    dict_to_formula,
    formula_to_vector,
    vector_to_formula,
    monoisotopic_mass_calculator,
)


class TestFormulaToDict:
    def test_simple_formula(self):
        assert formula_to_dict("CH4") == {"C": 1, "H": 4}

    def test_multi_element(self):
        assert formula_to_dict("C10H12N2O3") == {"C": 10, "H": 12, "N": 2, "O": 3}

    def test_single_atom(self):
        assert formula_to_dict("C") == {"C": 1}

    def test_two_char_element(self):
        result = formula_to_dict("C2H5Cl")
        assert result == {"C": 2, "H": 5, "Cl": 1}

    def test_non_string_returns_empty(self):
        assert formula_to_dict(None) == {}


class TestDictToFormula:
    def test_roundtrip(self):
        d = {"C": 10, "H": 12, "N": 2, "O": 3}
        assert formula_to_dict(dict_to_formula(d)) == d

    def test_single_count_no_number(self):
        result = dict_to_formula({"C": 1, "H": 4})
        assert "C" in result
        assert "H4" in result
        assert "C1" not in result

    def test_zero_count_excluded(self):
        # v <= 0 is skipped by vector_to_formula; dict_to_formula only includes v==1 or v>1
        result = dict_to_formula({"C": 2, "H": 0})
        assert "H" not in result


class TestFormulaVector:
    def test_roundtrip(self):
        formula = "C10H12N2O3"
        vec = formula_to_vector(formula)
        result = vector_to_formula(vec)
        # Roundtrip should preserve atom counts
        assert formula_to_dict(result) == formula_to_dict(formula)

    def test_vector_length(self):
        vec = formula_to_vector("C6H6")
        assert len(vec) == len(ATOMS_INDEX)

    def test_correct_positions(self):
        vec = formula_to_vector("C6H6")
        assert vec[ATOMS_INDEX["C"]] == 6
        assert vec[ATOMS_INDEX["H"]] == 6
        assert vec[ATOMS_INDEX["O"]] == 0

    def test_withH_false(self):
        vec = formula_to_vector("C6H6")
        result = vector_to_formula(vec, withH=False)
        assert "H" not in result
        assert "C6" in result


class TestMonoisotopicMass:
    def test_formula_mode(self):
        # C: 12.0, H4: 4*1.007825 = 4.0313 → ~16.031
        mass = monoisotopic_mass_calculator("CH4", "f")
        assert abs(mass - 16.0313) < 1e-3

    def test_known_formula(self):
        # C4H4N2O2 (uracil neutral): ~112.027
        mass = monoisotopic_mass_calculator("C4H4N2O2", "f")
        assert abs(mass - 112.027) < 0.01

    def test_mol_mode(self):
        from rdkit import Chem
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.AddHs(Chem.MolFromSmiles("C"))  # methane
        mass = monoisotopic_mass_calculator(mol, "mol")
        assert abs(mass - 16.0313) < 1e-3

    def test_invalid_mode(self):
        with pytest.raises(AssertionError):
            monoisotopic_mass_calculator("CH4", "invalid")
