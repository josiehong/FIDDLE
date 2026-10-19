import random

import pandas as pd
import pytest

from msfiddle.utils.pkl_utils import (
    generate_ms,
    generate_pairs,
    is_stepped_collision_energy,
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
