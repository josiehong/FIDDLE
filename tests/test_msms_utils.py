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


def test_filter_counts_isotope_errors_separately(capsys):
    import numpy as np

    from msfiddle.utils.mol_utils import monoisotopic_mass_calculator
    from msfiddle.utils.msms_utils import C13_SHIFT, filter_spec

    theo = precursor_mz_calculator("[M+H]+", monoisotopic_mass_calculator("C2H6O", "f"))

    def spectrum(title, mz):
        params = {"title": title, "smiles": "CCO", "precursor_type": "[M+H]+"}
        return {
            "params": {**params, "precursor_mz": str(mz)},
            "m/z array": np.array([31.0, 45.0, 46.0]),
            "intensity array": np.array([10.0, 100.0, 30.0]),
        }

    config = {
        "min_peak_num": 2,
        "min_mz": 20,
        "max_mz": 1500,
        "max_atom_num": 100,
        "min_atom_num": 2,
        "atom_type": ["C", "H", "O"],
        "precursor_type": ["[M+H]+"],
        "ppm_tolerance": 10,
    }
    spectra = [
        spectrum("ok", theo),
        spectrum("isotope", theo + C13_SHIFT),
        spectrum("off", theo + 0.5),
    ]
    # A one-pass reader (like pyteomics' IndexedMGF) must work too
    kept, _ = filter_spec(iter(spectra), config, {"[M+H]+": "+1"})
    assert [s["params"]["title"] for s in kept] == ["ok"]
    out = capsys.readouterr().out
    assert "ppm_tolerance: 1" in out and "isotope_error: 1" in out


@pytest.mark.parametrize(
    "precursor_type, per_mz, charge",
    [
        ("[M+H]+", 1.0, 1),
        ("[M+Na]+", 1.0, 1),
        ("[M+2H]2+", 2.0, 2),
        ("[2M+H]+", 0.5, 1),
    ],
)
def test_precursor_window_and_isotope_shift(precursor_type, per_mz, charge):
    from msfiddle.utils.msms_utils import (
        C13_SHIFT,
        isotope_mass_shift,
        precursor_window_da,
    )

    # A ppm tolerance applies to the m/z, then scales to the neutral mass
    assert precursor_window_da(precursor_type, 400.0, 5) == pytest.approx(
        5 * 400.0 / 1e6 * per_mz
    )
    # The k-th isotope peak sits k * 1.00336 / z higher in m/z
    assert isotope_mass_shift(precursor_type, 2) == pytest.approx(
        2 * C13_SHIFT / charge * per_mz
    )
