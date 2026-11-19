import warnings

import pytest

from msfiddle import main as cli
from msfiddle.api import (
    REJECTION_REASONS,
    MsFiddlePredictor,
    summarize_validation,
    validate_mgf,
)

PEAKS = [(60.0, 10.0), (85.0, 50.0), (100.0, 20.0), (125.0, 35.0), (150.0, 15.0)]


def _ion(title=None, mz="180.063", adduct="[M+H]+", ce="35", peaks=PEAKS):
    lines = ["BEGIN IONS"]
    for key, value in (
        ("TITLE", title),
        ("PRECURSOR_MZ", mz),
        ("PRECURSOR_TYPE", adduct),
        ("COLLISION_ENERGY", ce),
    ):
        if value is not None:
            lines.append(f"{key}={value}")
    lines += [f"{m} {i}" for m, i in peaks]
    return "\n".join(lines + ["END IONS", ""])


# (title, ion kwargs, expected status, expected reason) for the default encoder
CASES = [
    ("ok", {}, "rejected", "duplicate_id"),  # repeated below; the last is kept
    ("no_ce", {"ce": None}, "rejected", "missing_metadata"),
    (None, {}, "rejected", "missing_metadata"),
    ("bad_mz", {"mz": "abc"}, "rejected", "invalid_precursor_mz"),
    ("adduct", {"adduct": "[M+K]+"}, "rejected", "unsupported_adduct"),
    ("few", {"peaks": PEAKS[:4]}, "rejected", "insufficient_peaks"),
    ("high", {"peaks": PEAKS + [(1600.0, 1.0)]}, "rejected", "peak_mz_out_of_range"),
    ("stepped", {"ce": "HCD (NCE 20-30-40%)"}, "rejected", "stepped_collision_energy"),
    ("bad_ce", {"ce": "very high"}, "rejected", "invalid_collision_energy"),
    (
        "flat",
        {"peaks": [(m, 5.0) for m, _ in PEAKS]},
        "rejected",
        "encoding_failure",
    ),
    ("ok", {"ce": "40"}, "warning", "duplicate_id"),
]


@pytest.fixture
def mgf_path(tmp_path):
    path = tmp_path / "in.mgf"
    path.write_text("".join(_ion(title, **kw) for title, kw, _, _ in CASES))
    return path


def test_every_spectrum_gets_a_status_and_reason(mgf_path):
    report = validate_mgf(mgf_path)
    assert len(report) == len(CASES)
    assert list(report["status"]) == [status for _, _, status, _ in CASES]
    assert list(report["reason"]) == [reason for _, _, _, reason in CASES]
    rejected = report[report["status"] == "rejected"]
    assert set(rejected["reason"]) <= set(REJECTION_REASONS)
    assert rejected["message"].str.len().gt(0).all()


def test_predict_keeps_exactly_the_spectra_validation_accepts(mgf_path):
    # and the same ones the indexed reader used before: the last of a TITLE
    predictor = MsFiddlePredictor()
    report = predictor.validate_mgf(mgf_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        encoded = predictor._load_mgf_spectra(mgf_path)
    kept = report[report["status"] != "rejected"]
    assert [e.id for e in encoded] == list(kept["id"])
    assert [e.collision_energy for e in encoded] == ["40"]


def test_timstof_averages_stepped_energies_with_a_warning(tmp_path):
    path = tmp_path / "in.mgf"
    path.write_text(_ion("s", ce="20,40,60"))
    report = validate_mgf(path, instrument_type="timstof")
    assert report.loc[0, "status"] == "warning"
    assert report.loc[0, "reason"] == "stepped_collision_energy_averaged"


def test_validation_needs_no_checkpoints(mgf_path, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("models must not be loaded")

    monkeypatch.setattr(MsFiddlePredictor, "_ensure_models_loaded", fail)
    monkeypatch.setattr(MsFiddlePredictor, "_ensure_model_files", fail)
    validate_mgf(mgf_path)


def test_summary_counts_reconcile(mgf_path):
    text = summarize_validation(validate_mgf(mgf_path))
    assert text.splitlines()[0] == "11 spectra: 1 usable (1 with warnings), 10 rejected"
    assert "  missing_metadata            2" in text


def test_cli_writes_report(mgf_path, tmp_path, capsys):
    out = tmp_path / "report.csv"
    cli.main(["validate", "--test_data", str(mgf_path), "--report_path", str(out)])
    assert "10 rejected" in capsys.readouterr().out
    assert out.read_text().startswith("index,id,status,reason,message")


def test_cli_fails_on_malformed_mgf(tmp_path, capsys):
    path = tmp_path / "bad.mgf"
    path.write_text("BEGIN IONS\nTITLE=a\n100 abc\nEND IONS\n")
    with pytest.raises(SystemExit) as exc:
        cli.main(["validate", "--test_data", str(path)])
    assert exc.value.code == 1
    assert "Could not read" in capsys.readouterr().err


def test_cli_fails_when_nothing_is_usable(tmp_path):
    path = tmp_path / "in.mgf"
    path.write_text(_ion("a", adduct="[M+K]+"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["validate", "--test_data", str(path)])
    assert exc.value.code == 1


def test_loading_matches_the_indexed_reader(tmp_path):
    from pyteomics import mgf

    path = tmp_path / "in.mgf"
    path.write_text(
        _ion("a", ce="20") + _ion("b") + _ion(None) + _ion("a", ce="40") + _ion("c")
    )
    indexed = [
        (s["params"]["title"], s["params"]["collision_energy"])
        for s in mgf.read(str(path))
    ]
    encoded = MsFiddlePredictor()._load_mgf_spectra(path)
    assert [(e.id, e.collision_energy) for e in encoded] == indexed


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"mz": "0"}, "invalid_precursor_mz"),
        ({"mz": "-180.063"}, "invalid_precursor_mz"),
        ({"peaks": [(-40.0, 5.0)] + PEAKS[1:]}, "invalid_peaks"),
    ],
)
def test_non_positive_masses_are_rejected(tmp_path, kwargs, reason):
    path = tmp_path / "in.mgf"
    path.write_text(_ion("s", **kwargs))
    report = validate_mgf(path)
    assert (report.loc[0, "status"], report.loc[0, "reason"]) == ("rejected", reason)
