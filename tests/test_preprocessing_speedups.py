"""Regression tests for the preprocessing speedups (08/2026).

Pins: float binning parity with the old per-peak Decimal formula (including
exact bin-boundary m/z values), the O(n) pair sampler's structural
properties, and parallel == serial output for spec2arr and filter_spec.
"""

import random
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from msfiddle.utils.msms_utils import filter_spec
from msfiddle.utils.pkl_utils import generate_pairs, generate_ms, spec2arr


class TestBinningParity:
    @pytest.mark.parametrize("resolution", [0.2, 1])
    def test_matches_decimal_formula(self, resolution):
        # Boundary-adversarial values (exact multiples of the resolution,
        # where naive float floor under-bins: 100.2 * 5 = 500.99999...) plus
        # random peaks with realistic 4-5 decimal precision.
        rng = random.Random(0)
        mzs = [round(k * resolution, 4) for k in range(1, 400)]
        mzs += [
            round(rng.uniform(20, 1499), rng.choice([2, 3, 4, 5])) for _ in range(2000)
        ]
        mzs = sorted(set(m for m in mzs if 20 <= m < 1500))

        ys = [1.0 + (i % 7) for i in range(len(mzs))]
        ok, _, _, ms = generate_ms(
            x=mzs, y=ys, precursor_mz=5000.0, resolution=resolution, max_mz=1500
        )
        assert ok

        n_bins = ms.shape[0]
        expected_intensity = np.zeros(n_bins)
        res_d = Decimal(str(resolution))
        for mz, y in zip(mzs, ys):
            idx = int(round(Decimal(str(mz)) // res_d))  # the old formula
            if idx < n_bins:
                expected_intensity[idx] += y
        rng_norm = expected_intensity.max() - expected_intensity.min()
        expected_intensity = (expected_intensity - expected_intensity.min()) / rng_norm

        assert np.allclose(ms[:, 0], expected_intensity)


class TestGeneratePairs:
    def _df(self):
        # 40 compounds, replicate counts 1-6 (some singletons: no positives)
        rows = []
        rng = random.Random(1)
        for c in range(40):
            for _ in range(rng.randint(1, 6)):
                rows.append({"index": len(rows), "smiles": f"compound{c}"})
        return pd.DataFrame(rows)

    def test_pair_structure(self):
        df = self._df()
        random.seed(7)
        positive, negative = generate_pairs(df)
        smiles = df["smiles"].to_dict()
        assert positive and negative
        for a, b in positive:
            assert a != b and smiles[a] == smiles[b]
        for a, b in negative:
            assert smiles[a] != smiles[b]

    def test_every_replicated_row_gets_a_positive(self):
        df = self._df()
        counts = df["smiles"].value_counts()
        replicated = {i for i, s in df["smiles"].items() if counts[s] > 1}
        random.seed(7)
        positive, _ = generate_pairs(df)
        covered = {i for pair in positive for i in pair}
        assert replicated <= covered


def _toy_spectra(n=6):
    spectra = []
    for i in range(n):
        spectra.append(
            {
                "params": {
                    "title": f"toy_{i}",
                    "smiles": "CCO" if i % 2 else "C1=CC=CC=C1",
                    "precursor_type": "[M+H]+",
                    "precursor_mz": "47.0491" if i % 2 else "79.0542",
                    "theoretical_precursor_mz": "47.0491" if i % 2 else "79.0542",
                    "collision_energy": "40",
                    "instrument_type": "qtof",
                    "source_instrument": "toy",
                    "ms_level": "MS2",
                },
                "m/z array": np.array([31.0184, 41.0386, 45.0335 + i * 0.5]),
                "intensity array": np.array([10.0, 100.0, 30.0 + i]),
            }
        )
    return spectra


_ENCODER = {
    "resolution": 0.2,
    "max_mz": 1500,
    "type2charge": {"[M+H]+": "1"},
    "precursor_type": {"[M+H]+": 1},
}

_FILTER_CONFIG = {
    "min_peak_num": 2,
    "min_mz": 20,
    "max_mz": 1500,
    "max_atom_num": 100,
    "min_atom_num": 2,
    "atom_type": ["C", "H", "O", "N"],
    "precursor_type": ["[M+H]+"],
}


class TestParallelEqualsSerial:
    def test_spec2arr(self):
        serial, bad_s = spec2arr(_toy_spectra(), _ENCODER, num_workers=0)
        parallel, bad_p = spec2arr(_toy_spectra(), _ENCODER, num_workers=2)
        assert bad_s == bad_p
        assert len(serial) == len(parallel) > 0
        for a, b in zip(serial, parallel):
            assert a["title"] == b["title"]
            assert np.allclose(a["spec"], b["spec"])
            assert np.allclose(a["env"], b["env"])
            assert np.allclose(a["formula"], b["formula"])

    def test_filter_spec(self):
        serial, smiles_s = filter_spec(
            _toy_spectra(), _FILTER_CONFIG, _ENCODER["type2charge"], num_workers=0
        )
        parallel, smiles_p = filter_spec(
            _toy_spectra(), _FILTER_CONFIG, _ENCODER["type2charge"], num_workers=2
        )
        assert smiles_s == smiles_p
        assert len(serial) == len(parallel) > 0
        for a, b in zip(serial, parallel):
            assert a["params"]["title"] == b["params"]["title"]
            assert a["params"]["theoretical_precursor_mz"] == pytest.approx(
                b["params"]["theoretical_precursor_mz"]
            )
