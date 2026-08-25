"""Regression tests for the preprocessing bugs found in the 08/2026 review.

Covers: the contrastive-pair augmentation index corruption, the -2H2O
neutral-loss parsing, the precursor-isotope ladder direction, the NCE scale
unification, the max_mz bin overflow, and the GNPS adduct normalization.
"""

import re

import numpy as np
import pytest

from msfiddle.dataset import MS2FDataset_CL
from msfiddle.utils.pkl_utils import (
    generate_ms,
    melt_neutral_precursor,
    parse_collision_energy,
    remove_precursor_isotopic_peaks,
)


class TestAugmentPairIndices:
    def _make_data(self, n):
        return [
            {"title": f"spec{j}", "spec": np.full((4, 2), float(j))} for j in range(n)
        ]

    @pytest.mark.parametrize("noised_times", [1, 2, 3])
    def test_augmented_pairs_reference_same_spectra(self, noised_times):
        # Every shifted pair must point at (noised copies of) the same two
        # spectra as the original pair — the interleaved layout broke this
        # for noised_times >= 2.
        data = self._make_data(5)
        pairs = {"idx1": [0, 2], "idx2": [1, 4], "label": [1, 0]}
        ds = MS2FDataset_CL.__new__(MS2FDataset_CL)
        aug_data, aug_pairs, length = ds.augment_with_noise(data, pairs, noised_times)
        assert length == len(pairs["idx1"]) * (1 + noised_times)
        assert len(aug_data) == len(data) * (1 + noised_times)
        n_orig = len(pairs["idx1"])
        for p in range(len(aug_pairs["idx1"])):
            # originals first, then noised_times appended copies per pair
            orig = p if p < n_orig else (p - n_orig) // noised_times
            assert aug_pairs["label"][p] == pairs["label"][orig]
            assert (
                aug_data[aug_pairs["idx1"][p]]["title"]
                == data[pairs["idx1"][orig]]["title"]
            )
            assert (
                aug_data[aug_pairs["idx2"][p]]["title"]
                == data[pairs["idx2"][orig]]["title"]
            )

    def test_noise_only_on_nonzero_bins(self):
        data = [{"title": "s", "spec": np.array([[1.0, 0.0], [0.0, 0.0]])}]
        ds = MS2FDataset_CL.__new__(MS2FDataset_CL)
        aug_data, _, _ = ds.augment_with_noise(
            data, {"idx1": [], "idx2": [], "label": []}, 2
        )
        for item in aug_data[1:]:
            assert item["spec"][1, 0] == 0.0  # zero bins stay zero


class TestMeltNeutralPrecursor:
    def test_double_water_loss(self):
        f, t = melt_neutral_precursor("C6H12O6", "[M+H-2H2O]+")
        assert t == "[M+H]+"
        assert f == "C6H8O4"

    def test_single_water_loss(self):
        f, t = melt_neutral_precursor("C6H12O6", "[M+H-H2O]+")
        assert t == "[M+H]+"
        assert f == "C6H10O5"

    def test_ch2o2_not_confused_with_h2o(self):
        f, t = melt_neutral_precursor("C6H12O6", "[M+H-CH2O2]+")
        assert t == "[M+H]+"
        assert f == "C5H10O4"

    def test_no_neutral_loss_untouched(self):
        f, t = melt_neutral_precursor("C6H12O6", "[M+H]+")
        assert t == "[M+H]+"
        assert f == "C6H12O6"


class TestIsotopeRemovalDirection:
    def test_removes_precursor_and_upward_isotopes_only(self):
        x = [298.10, 299.10, 300.10, 301.10, 302.10]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        fx, fy = remove_precursor_isotopic_peaks(x, y, 300.10, charge=1)
        # fragments below the precursor survive; precursor + isotopes removed
        assert fx == [298.10, 299.10]
        assert fy == [1.0, 2.0]


class TestCollisionEnergyScale:
    def test_direct_nce_is_percent_scale(self):
        ce, nce = parse_collision_energy("35 nce", precursor_mz=400.0)
        assert nce == pytest.approx(35.0)
        assert ce == pytest.approx(35.0 * 400.0 / 500.0)  # 28 eV, not 0.28

    def test_ev_derived_nce_matches_scale(self):
        ce, nce = parse_collision_energy("40", precursor_mz=400.0)
        assert ce == pytest.approx(40.0)
        assert nce == pytest.approx(40.0 * 500.0 / 400.0)  # 50.0, percent scale

    def test_roundtrip_consistency(self):
        # eV -> NCE -> eV should be the identity at charge 1
        ce1, nce1 = parse_collision_energy("40", precursor_mz=400.0)
        ce2, nce2 = parse_collision_energy(f"{nce1} nce", precursor_mz=400.0)
        assert ce2 == pytest.approx(ce1)


class TestGenerateMsBounds:
    def test_peak_at_exactly_max_mz_does_not_crash(self):
        ok, _, _, ms = generate_ms(
            x=[100.0, 1500.0],
            y=[10.0, 5.0],
            precursor_mz=800.0,
            resolution=1,
            max_mz=1500,
        )
        assert ok
        assert ms.shape[0] == 1500


class TestGnpsAdductNormalization:
    NORM = staticmethod(lambda x: re.sub(r"\]1([+-])$", r"]\1", x))

    def test_explicit_charge_one_stripped(self):
        assert self.NORM("[M+H]1+") == "[M+H]+"
        assert self.NORM("[M-H]1-") == "[M-H]-"

    def test_multiply_charged_untouched(self):
        assert self.NORM("[M+2H]2+") == "[M+2H]2+"

    def test_already_normalized_untouched(self):
        assert self.NORM("[M+H]+") == "[M+H]+"
        assert self.NORM("[M-H]-") == "[M-H]-"
