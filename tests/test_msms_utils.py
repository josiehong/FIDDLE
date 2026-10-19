import pytest

from msfiddle.utils.mol_utils import ATOMS_WEIGHT
from msfiddle.utils.msms_utils import mass_calculator, precursor_mz_calculator

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


@pytest.mark.parametrize("precursor_type", PRECURSOR_TYPES)
def test_mass_calculator_inverts_precursor_mz(precursor_type):
    mz = precursor_mz_calculator(precursor_type, MASS)
    assert mass_calculator(precursor_type, mz) == pytest.approx(MASS, abs=1e-6)


@pytest.mark.parametrize(
    "precursor_type, expected",
    [
        ("[M+H]+", MASS + ATOMS_WEIGHT["H"]),
        ("[M-H]-", MASS - ATOMS_WEIGHT["H"]),
        ("[M+Na]+", MASS + ATOMS_WEIGHT["Na"]),
        ("[2M+H]+", 2 * MASS + ATOMS_WEIGHT["H"]),
    ],
)
def test_precursor_mz(precursor_type, expected):
    assert precursor_mz_calculator(precursor_type, MASS) == pytest.approx(expected)


def test_unsupported_precursor_type_raises():
    with pytest.raises(ValueError):
        precursor_mz_calculator("[M+K]+", MASS)
    with pytest.raises(ValueError):
        mass_calculator("[M+K]+", 201.0)
