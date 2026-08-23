import pytest
from msfiddle.utils.mol_utils import ATOMS_WEIGHT
from msfiddle.utils.msms_utils import (
    precursor_mz_calculator,
    mass_calculator,
    ce2nce,
)

# A known neutral mass to use across roundtrip tests
MASS = 200.0


PRECURSOR_TYPES = [
    "[M+H]+",
    "[M+2H]2+",
    "[M+Na]+",
    "[M-H]-",
    "[M+H-H2O]+",
    "[M-H2O+H]+",
    "[2M+H]+",
    "[2M-H]-",
    "[M+H-2H2O]+",
    "[M+H-NH3]+",
    "[M+H+NH3]+",
    "[M+NH4]+",
    "[M+H-CH2O2]+",
    "[M+H-CH4O2]+",
    "[M-H-CO2]-",
    "[M-CHO2]-",
    "[M-H-H2O]-",
    "[M-H2O-H]-",
]


class TestPrecursorRoundtrip:
    @pytest.mark.parametrize("precursor_type", PRECURSOR_TYPES)
    def test_mz_mass_roundtrip(self, precursor_type):
        """mass_calculator should be the inverse of precursor_mz_calculator."""
        mz = precursor_mz_calculator(precursor_type, MASS)
        recovered_mass = mass_calculator(precursor_type, mz)
        assert (
            abs(recovered_mass - MASS) < 1e-6
        ), f"{precursor_type}: expected {MASS}, got {recovered_mass}"

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError):
            precursor_mz_calculator("[M+K]+", MASS)

    def test_unsupported_type_mass_raises(self):
        with pytest.raises(ValueError):
            mass_calculator("[M+K]+", 201.0)


class TestPrecursorMzValues:
    def test_m_plus_h(self):
        mz = precursor_mz_calculator("[M+H]+", MASS)
        assert abs(mz - (MASS + ATOMS_WEIGHT["H"])) < 1e-9

    def test_m_minus_h(self):
        mz = precursor_mz_calculator("[M-H]-", MASS)
        assert abs(mz - (MASS - ATOMS_WEIGHT["H"])) < 1e-9

    def test_m_plus_na(self):
        mz = precursor_mz_calculator("[M+Na]+", MASS)
        assert abs(mz - (MASS + ATOMS_WEIGHT["Na"])) < 1e-9

    def test_2m_plus_h(self):
        mz = precursor_mz_calculator("[2M+H]+", MASS)
        assert abs(mz - (2 * MASS + ATOMS_WEIGHT["H"])) < 1e-9


class TestCe2nce:
    def test_charge_1(self):
        # charge_factor[1] = 1.0
        result = ce2nce(ce=50, precursor_mz=500, charge=1)
        assert abs(result - 50.0) < 1e-9  # 50 * 500 * 1 / 500 = 50

    def test_charge_2(self):
        # charge_factor[2] = 0.9
        result = ce2nce(ce=50, precursor_mz=500, charge=2)
        assert abs(result - 45.0) < 1e-9  # 50 * 500 * 0.9 / 500 = 45

    def test_scales_with_precursor_mz(self):
        r1 = ce2nce(ce=30, precursor_mz=300, charge=1)
        r2 = ce2nce(ce=30, precursor_mz=600, charge=1)
        assert r1 == pytest.approx(r2 * 2)
