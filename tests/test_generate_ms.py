"""Regression tests for the mz_val channel in utils.pkl_utils.generate_ms (TODO A3).

Before the A3 fix, intensity was accumulated into the bin *before* comparing it
to the incoming peak, so `if intensity_val[idx] < intensity` was always False and
the `mz_val` channel stayed 0 for every bin. These tests lock in the fixed
behaviour: the m/z of the strongest peak in each bin is recorded.
"""

import os
import sys
from decimal import Decimal

# Make the repo root importable when running pytest from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from msfiddle.utils.pkl_utils import generate_ms  # noqa: E402

RES = 0.2
# Precursor placed far from the test peaks so isotopic-peak removal leaves them intact.
PRECURSOR_MZ = 1000.0


def _bin_idx(mz, resolution=RES):
    return int(round(Decimal(str(mz)) // Decimal(str(resolution))))


def test_single_peak_per_bin_records_mz():
    # Two peaks, each alone in its bin. Previously both mz_val entries were 0.
    x = [100.0, 200.0]
    y = [5.0, 3.0]
    ok, _, _, ms = generate_ms(
        x, y, precursor_mz=PRECURSOR_MZ, resolution=RES, charge=1
    )
    assert ok
    mz_val = ms[:, 1]
    assert abs(mz_val[_bin_idx(100.0)] - 100.0) < 1e-9
    assert abs(mz_val[_bin_idx(200.0)] - 200.0) < 1e-9
    # An empty bin must remain at its initialised 0.
    assert mz_val[_bin_idx(123.4)] == 0.0


def test_strongest_peak_in_shared_bin_wins():
    # Two peaks in the same 0.2 Da bin; the stronger one's m/z should be recorded.
    x = [50.01, 50.05]
    y = [10.0, 3.0]
    ok, _, _, ms = generate_ms(
        x, y, precursor_mz=PRECURSOR_MZ, resolution=RES, charge=1
    )
    assert ok
    idx = _bin_idx(50.01)
    assert _bin_idx(50.05) == idx  # sanity: both peaks share the bin
    assert abs(ms[idx, 1] - 50.01) < 1e-9
