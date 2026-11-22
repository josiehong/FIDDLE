import random

import numpy as np
import pandas as pd
import pytest

from msfiddle.settings import load_config
from msfiddle.utils.pkl_utils import (
    generate_ms,
    generate_pairs,
    is_stepped_collision_energy,
    precursor_mz_key,
    spec2arr_single,
)

# Precursor far from the test peaks so isotopic-peak removal leaves them intact.
PRECURSOR_MZ = 1000.0


@pytest.mark.parametrize("mz, idx", [(20.2, 101), (22.2, 111)])
def test_peak_on_bin_boundary(mz, idx):
    # Exact multiples of the resolution: naive float floor gives 20.2 / 0.2 = 100.99...
    ok, _, _, ms = generate_ms([mz, 300.0], [1.0, 2.0], PRECURSOR_MZ, resolution=0.2)
    assert ok
    assert ms[idx, 1] == pytest.approx(mz)


def test_mz_channel_records_strongest_peak_per_bin():
    ok, _, _, ms = generate_ms(
        [50.01, 50.05, 200.0], [10.0, 3.0, 5.0], PRECURSOR_MZ, resolution=0.2
    )
    assert ok
    assert ms[250, 1] == pytest.approx(50.01)  # 50.01 and 50.05 share a bin
    assert ms[1000, 1] == pytest.approx(200.0)
    assert ms[617, 1] == 0.0  # empty bin


def test_generate_pairs():
    rng = random.Random(1)
    rows = [f"compound{c}" for c in range(40) for _ in range(rng.randint(1, 6))]
    df = pd.DataFrame({"index": range(len(rows)), "smiles": rows})
    random.seed(7)
    positive, negative = generate_pairs(df)
    smiles = df["smiles"].to_dict()
    assert positive and negative
    assert all(a != b and smiles[a] == smiles[b] for a, b in positive)
    assert all(smiles[a] != smiles[b] for a, b in negative)
    # Every compound with replicates appears in a positive pair
    counts = df["smiles"].value_counts()
    replicated = {i for i, s in smiles.items() if counts[s] > 1}
    assert replicated <= {i for pair in positive for i in pair}


@pytest.mark.parametrize(
    "ce, stepped",
    [
        ("HCD (NCE 20-30-40%)", True),
        ("30-50 eV", True),
        ("stepped 20,40,60", True),
        ("35", False),
        ("NCE=35%", False),
        ("HCD (NCE 40%)", False),
        ("35.5 eV", False),
        ("Unknown", False),
    ],
)
def test_stepped_collision_energy(ce, stepped):
    assert is_stepped_collision_energy(ce) is stepped


def _caffeine_spectrum():
    # [M+H]+ of caffeine (theoretical m/z 195.087652), measured 2 ppm high
    return {
        "m/z array": np.array([110.0713, 138.0662, 195.0880]),
        "intensity array": np.array([20.0, 100.0, 50.0]),
        "params": {
            "title": "caffeine",
            "smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
            "precursor_type": "[M+H]+",
            "collision_energy": "30",
            "precursor_mz": 195.0880,
            "simulated_precursor_mz": 195.0870,
            "theoretical_precursor_mz": 195.087652,
        },
    }


@pytest.mark.parametrize(
    "use_simulated, expected",
    [(False, "precursor_mz"), (True, "simulated_precursor_mz")],
)
def test_training_encodes_the_configured_precursor_mz(use_simulated, expected):
    # Training must see the precursor m/z prediction sees, not the theoretical one
    encoder = dict(
        load_config("./msfiddle/config/fiddle_tcn_qtof.yml")["encoding"],
        use_simulated_precursor_mz=use_simulated,
    )
    assert precursor_mz_key(encoder) == expected
    spectrum = _caffeine_spectrum()
    record = spec2arr_single(spectrum, encoder)
    assert record["env"][0] == pytest.approx(spectrum["params"][expected], abs=1e-4)
