import importlib.util
import os

import pytest

from msfiddle import download
from msfiddle.download import checkpoint_release_for_package_version


@pytest.mark.parametrize(
    "version, release",
    [
        ("2.0.0", "v2.0.0"),
        ("2.0.1.dev0", "v2.0.0"),
        ("2.1.0", "v2.0.0"),
        ("2.2.0", "v2.2.0"),
        ("2.2.3", "v2.2.0"),
        ("2.2.0.dev0", "v2.2.0"),
        ("2.3.0", "v2.3.0"),
        ("3.4.5", "v3.4.0"),
    ],
)
def test_checkpoint_release_for_package_version(version, release):
    assert checkpoint_release_for_package_version(version) == release


def test_checkpoint_dir_is_keyed_by_release():
    assert os.path.basename(download.get_checkpoint_dir()) == download.FIDDLE_RELEASE


def test_release_asset_urls_bundle_each_instrument_from_v2_3():
    assert [
        u.rsplit("/", 1)[-1] for u in download.release_asset_urls("qtof", "v2.3.0")
    ] == ["fiddle_qtof.zip"]
    assert [
        u.rsplit("/", 1)[-1] for u in download.release_asset_urls("qtof", "v2.2.0")
    ] == [
        "fiddle_tcn_qtof.zip",
        "fiddle_rescore_qtof.zip",
    ]


def _fake_checkpoints(tmp_path, recorded_hash=None):
    torch = pytest.importorskip("torch")
    tcn = {"model_state_dict": {"module.w": torch.randn(3, 4)}, "epoch": 7}
    rescore = {
        "formula_encoder_state_dict": {"f": torch.randn(2)},
        "rescore_head_state_dict": {"h": torch.randn(2)},
        "optimizer_state_dict": {},
    }
    if recorded_hash:
        rescore["tcn_hash"] = recorded_hash
    paths = {"tcn": tmp_path / "tcn.pt", "rescore": tmp_path / "rescore.pt"}
    torch.save(tcn, paths["tcn"])
    torch.save(rescore, paths["rescore"])
    return paths


def _packaging():
    spec = importlib.util.spec_from_file_location(
        "package_release_checkpoints",
        os.path.join(
            os.path.dirname(__file__), "..", "scripts", "package_release_checkpoints.py"
        ),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_packaged_pair_downloads_together(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    from msfiddle.model_tcn import tcn_hash

    zip_path = _packaging().package(_fake_checkpoints(tmp_path), "qtof", str(tmp_path))
    monkeypatch.setattr(
        download, "release_asset_urls", lambda inst: [f"file://{zip_path}"]
    )
    download.download_models(destination=str(tmp_path / "out"), instruments=["qtof"])

    tcn = torch.load(tmp_path / "out" / "fiddle_tcn_qtof.pt", weights_only=False)
    rescore = torch.load(
        tmp_path / "out" / "fiddle_rescore_qtof.pt", weights_only=False
    )
    assert list(tcn["model_state_dict"]) == ["w"]  # DataParallel prefix stripped
    assert "optimizer_state_dict" not in rescore
    assert rescore["tcn_hash"] == tcn_hash(tcn["model_state_dict"])


def test_packaging_refuses_a_rescore_model_of_another_tcn(tmp_path):
    srcs = _fake_checkpoints(tmp_path, recorded_hash="0123456789abcdef")
    with pytest.raises(RuntimeError, match="different TCN"):
        _packaging().package(srcs, "qtof", str(tmp_path))
