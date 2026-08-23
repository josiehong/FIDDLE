import pytest
from msfiddle.utils.refine_utils import (
    parse_formula,
    format_formula,
    passes_senior_rule,
)


class TestParseFormula:
    def test_simple(self):
        assert parse_formula("CH4") == {"C": 1, "H": 4}

    def test_multi_digit(self):
        assert parse_formula("C10H12N2O3") == {"C": 10, "H": 12, "N": 2, "O": 3}

    def test_two_char_element(self):
        result = parse_formula("C2H5Cl")
        assert result == {"C": 2, "H": 5, "Cl": 1}

    def test_single_atom(self):
        assert parse_formula("C") == {"C": 1}


class TestFormatFormula:
    def test_sorts_alphabetically(self):
        result = format_formula({"H": 4, "C": 1})
        assert result == "CH4"  # C before H alphabetically

    def test_single_count_no_digit(self):
        result = format_formula({"C": 1})
        assert result == "C"
        assert "1" not in result

    def test_roundtrip(self):
        original = {"C": 6, "H": 6}
        assert parse_formula(format_formula(original)) == original


class TestPassesSeniorRule:
    # Valid molecules
    def test_methane_valid(self):
        assert passes_senior_rule("CH4") is True

    def test_benzene_valid(self):
        assert passes_senior_rule("C6H6") is True

    def test_uracil_valid(self):
        assert passes_senior_rule("C4H4N2O2") is True

    def test_dinitrogen_valid(self):
        assert passes_senior_rule("N2") is True

    # Invalid molecules
    def test_methyl_radical_invalid(self):
        # CH3 has odd total odd-valence sum (3 H, valence 1 each → sum=3, odd)
        assert passes_senior_rule("CH3") is False

    def test_hn_invalid(self):
        # HN: total_valence=4, max_valence=3 → 4 < 2*3=6 → False
        assert passes_senior_rule("HN") is False

    def test_single_nitrogen_invalid(self):
        # N alone: odd-valence sum = 3 (odd) → False
        assert passes_senior_rule("N") is False
