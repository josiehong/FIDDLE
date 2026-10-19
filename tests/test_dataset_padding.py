import pickle

import numpy as np
import pytest

from msfiddle.dataset import MS2FDataset, MS2FDataset_CL


def _write_pkls(tmp_path, n=3, length=7500):
    rng = np.random.default_rng(0)
    data = [
        {
            "title": f"s{i}",
            "formula": np.zeros(13, dtype=np.float32),
            "spec": rng.random((length, 2), dtype=np.float32),
            "mass": 100.0 + i,
            "env": np.array([100.0 + i, 20.0, 1.0], dtype=np.float32),
        }
        for i in range(n)
    ]
    train = tmp_path / "x_train.pkl"
    with open(train, "wb") as f:
        pickle.dump(data, f)
    with open(tmp_path / "x_train_pairs.pkl", "wb") as f:
        pickle.dump({"idx1": [0, 1], "idx2": [1, 2], "label": [0, 1]}, f)
    return str(train), data


@pytest.mark.parametrize(
    "make",
    [
        lambda p: MS2FDataset(p, padding_dim=20),
        lambda p: MS2FDataset_CL(p, padding_dim=20),
    ],
    ids=["MS2FDataset", "MS2FDataset_CL"],
)
def test_padding_keeps_float32(tmp_path, make):
    # np.zeros defaults to float64; concatenating it would double the RAM of
    # every spectrum (about 54 -> 107 GB for the Orbitrap training set).
    path, original = _write_pkls(tmp_path)
    ds = make(path)
    for item, orig in zip(ds.data, original):
        spec = item["spec"]
        assert spec.dtype == np.float32
        assert spec.shape == (7520, 2)
        np.testing.assert_array_equal(spec[:7500], orig["spec"])
        assert not spec[7500:].any()
