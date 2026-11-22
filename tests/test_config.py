import pytest
import yaml

from msfiddle.settings import load_config

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


@pytest.mark.parametrize("instrument, ppm", [("qtof", 10), ("orbitrap", 5)])
def test_only_nist_precursor_mz_is_simulated(instrument, ppm):
    # NIST20/23 list a precursor m/z computed from the structure; the paper
    # simulates it, and every other library keeps its measured value
    config = load_config(f"./msfiddle/config/fiddle_tcn_{instrument}.yml")
    simulated = {
        name: section["simulate_precursor_ppm"]
        for name, section in config.items()
        if isinstance(section, dict) and "simulate_precursor_ppm" in section
    }
    assert simulated == {f"nist20_{instrument}": ppm, f"nist23_{instrument}": ppm}


def test_timstof_precursor_mz_is_measured():
    config = load_config("./msfiddle/config/fiddle_tcn_timstof.yml")
    assert not any(
        isinstance(section, dict) and section.get("simulate_precursor_ppm")
        for section in config.values()
    )
