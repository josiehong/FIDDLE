import pytest
import yaml

from msfiddle.config import load_config

POST_PROCESSING = {
    "refine_atom_type": ["C", "H"],
    "refine_atom_num": [5, -1],
    "mass_tolerance": 5.0,
    "ppm_mode": True,
    "top_k": 20,
}


def _write(tmp_path, post_processing, **sections):
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump({"post_processing": post_processing, **sections}))
    return path


def test_defaults_are_filled(tmp_path):
    pp = load_config(_write(tmp_path, dict(POST_PROCESSING)))["post_processing"]
    assert pp["isotope_errors"] == [0] and pp["isotope_penalty"] == 0.5


@pytest.mark.parametrize("offsets", [[1], [0, 0], [0, -1], [0, 1.5]])
def test_invalid_isotope_errors_raise(tmp_path, offsets):
    with pytest.raises(ValueError, match="isotope_errors"):
        load_config(_write(tmp_path, {**POST_PROCESSING, "isotope_errors": offsets}))


def test_filter_wider_than_refinement_window_warns(tmp_path):
    path = _write(tmp_path, dict(POST_PROCESSING), casmi={"ppm_tolerance": 10})
    with pytest.warns(UserWarning, match="casmi.ppm_tolerance"):
        load_config(path)
