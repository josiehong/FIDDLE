import pytest

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
    "[M+CH2O2-H]-",
]


@pytest.mark.parametrize("precursor_type", PRECURSOR_TYPES)
def test_mass_calculator_inverts_precursor_mz(precursor_type):
    mz = precursor_mz_calculator(precursor_type, MASS)
    assert mass_calculator(precursor_type, mz) == pytest.approx(MASS, abs=1e-6)


CAFFEINE = 194.080376  # C8H10N4O2, monoisotopic
PROTON = 1.007276467


@pytest.mark.parametrize(
    "precursor_type, expected",
    [
        ("[M+H]+", 195.087652),  # M plus a proton, not a hydrogen atom
        ("[M-H]-", 193.073100),
        ("[M+Na]+", 217.069597),
        ("[M+2H]2+", (CAFFEINE + 2 * PROTON) / 2),
        ("[2M+H]+", 2 * CAFFEINE + PROTON),
        ("[M+NH4]+", 212.114202),
    ],
)
def test_precursor_mz_accounts_for_the_electron(precursor_type, expected):
    mz = precursor_mz_calculator(precursor_type, CAFFEINE)
    assert mz == pytest.approx(expected, abs=2e-6)
    assert mass_calculator(precursor_type, expected) == pytest.approx(
        CAFFEINE, abs=2e-6
    )


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


def test_simulated_precursor_mz_is_seeded_and_realistic():
    import numpy as np

    from msfiddle.utils.msms_utils import simulate_precursor_mz

    def spectra():
        return [
            {
                "params": {
                    "precursor_mz": "195.0877",
                    "precursor_type": "[M+H]+",
                    "smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
                }
            }
            for _ in range(2000)
        ]

    a = simulate_precursor_mz(spectra(), ppm=5, seed=0)
    b = simulate_precursor_mz(spectra(), ppm=5, seed=0)
    assert [s["params"]["precursor_mz"] for s in a] == [
        s["params"]["precursor_mz"] for s in b
    ]
    assert a[0]["params"]["library_precursor_mz"] == "195.0877"
    theo = precursor_mz_calculator("[M+H]+", CAFFEINE)
    ppm = np.array([(s["params"]["precursor_mz"] - theo) / theo * 1e6 for s in a])
    assert abs(ppm.mean()) < 0.15  # centred on the theoretical m/z
    assert ppm.std() == pytest.approx(5 / 3, rel=0.1)


def test_filter_simulates_the_precursor_mz_when_asked():
    import numpy as np

    from msfiddle.utils.mol_utils import monoisotopic_mass_calculator
    from msfiddle.utils.msms_utils import filter_spec

    theo = precursor_mz_calculator("[M+H]+", monoisotopic_mass_calculator("C2H6O", "f"))

    def spectra():
        return [
            {
                "params": {
                    "title": f"s{i}",
                    "smiles": "CCO",
                    "precursor_type": "[M+H]+",
                    "precursor_mz": f"{theo:.4f}",
                },
                "m/z array": np.array([31.0, 45.0, 46.0]),
                "intensity array": np.array([10.0, 100.0, 30.0]),
            }
            for i in range(5)
        ]

    config = {
        "min_peak_num": 2,
        "min_mz": 20,
        "max_mz": 1500,
        "max_atom_num": 100,
        "min_atom_num": 2,
        "atom_type": ["C", "H", "O"],
        "precursor_type": ["[M+H]+"],
        "ppm_tolerance": 10,
        "simulate_precursor_ppm": 10,
    }
    kept, _ = filter_spec(spectra(), config, {"[M+H]+": "+1"})
    simulated = [s["params"]["precursor_mz"] for s in kept]
    assert all(s["params"]["library_precursor_mz"] == f"{theo:.4f}" for s in kept)
    assert len(set(simulated)) == 5  # each spectrum gets its own error
    assert all(abs(mz - theo) / theo * 1e6 < 10 for mz in simulated)

    # The same values on every run, and in a process pool
    again, _ = filter_spec(spectra(), config, {"[M+H]+": "+1"}, num_workers=2)
    assert [s["params"]["precursor_mz"] for s in again] == simulated
    # Spectra sharing a title still get their own errors
    shared = spectra()
    for spectrum in shared:
        spectrum["params"]["title"] = "same"
    kept_shared, _ = filter_spec(shared, config, {"[M+H]+": "+1"})
    assert len({s["params"]["precursor_mz"] for s in kept_shared}) == 5
    # Off by default: the library value is kept
    plain, _ = filter_spec(
        spectra(),
        {k: v for k, v in config.items() if k != "simulate_precursor_ppm"},
        {"[M+H]+": "+1"},
    )
    assert all("library_precursor_mz" not in s["params"] for s in plain)
