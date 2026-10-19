import importlib.util
import os
import zipfile

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_script():
    path = os.path.join(REPO, "scripts", "package_release_checkpoints.py")
    spec = importlib.util.spec_from_file_location("package_release_checkpoints", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pkg = _load_script()


def _extract(zip_path, tmp_path):
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        zf.extractall(tmp_path / "out")
    return names, torch.load(tmp_path / "out" / names[0], weights_only=False)


def test_tcn_zip_keeps_weights_drops_optimizer(tmp_path):
    w = {"module.fc.weight": torch.randn(3, 2), "module.fc.bias": torch.randn(3)}
    src = tmp_path / "fiddle_tcn_qtof_x.pt"
    torch.save(
        {
            "epoch": 7,
            "model_state_dict": w,
            "optimizer_state_dict": {"state": {0: torch.randn(3, 2)}},
            "scheduler_state_dict": {},
            "best_val_mae": 0.3,
        },
        src,
    )
    names, rel = _extract(pkg.package(str(src), "tcn_qtof", str(tmp_path)), tmp_path)
    assert names == ["fiddle_tcn_qtof.pt"]
    assert set(rel) == {"model_state_dict", "epoch", "best_val_mae"}
    assert set(rel["model_state_dict"]) == {"fc.weight", "fc.bias"}
    assert torch.equal(rel["model_state_dict"]["fc.weight"], w["module.fc.weight"])


def test_rescore_zip_keeps_both_heads(tmp_path):
    src = tmp_path / "fiddle_rescore_orbitrap_x.pt"
    torch.save(
        {
            "epoch": 3,
            "formula_encoder_state_dict": {"a": torch.ones(2)},
            "rescore_head_state_dict": {"b": torch.zeros(2)},
            "optimizer_state_dict": {},
            "best_val_auc": 0.9,
        },
        src,
    )
    names, rel = _extract(
        pkg.package(str(src), "rescore_orbitrap", str(tmp_path)), tmp_path
    )
    assert names == ["fiddle_rescore_orbitrap.pt"]
    assert set(rel) == {
        "formula_encoder_state_dict",
        "rescore_head_state_dict",
        "epoch",
        "best_val_auc",
    }


def test_wrong_checkpoint_kind_is_rejected(tmp_path):
    src = tmp_path / "fiddle_tcn_qtof_x.pt"
    torch.save({"formula_encoder_state_dict": {}, "rescore_head_state_dict": {}}, src)
    with pytest.raises(KeyError):
        pkg.package(str(src), "tcn_qtof", str(tmp_path))
