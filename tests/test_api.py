import pytest
import numpy as np
import pandas as pd

import msfiddle.api as api
from msfiddle.api import MsFiddlePredictor, _EncodedSpectrum, _ModelPrediction

VALID_MZ = [60.0, 85.0, 100.0, 125.0, 150.0]
VALID_INTENSITY = [10.0, 50.0, 20.0, 35.0, 15.0]
VALID_PRECURSOR_MZ = 180.063
VALID_ADDUCT = "[M+H]+"


def test_predict_from_spectrum_rejects_unequal_array_lengths():
    with pytest.raises(ValueError, match="same length"):
        api.predict_from_spectrum(
            mz_array=VALID_MZ,
            intensity_array=VALID_INTENSITY[:-1],
            precursor_mz=VALID_PRECURSOR_MZ,
            adduct=VALID_ADDUCT,
        )


def test_predict_from_spectrum_rejects_too_few_peaks():
    with pytest.raises(ValueError, match="at least 5 peaks"):
        api.predict_from_spectrum(
            mz_array=VALID_MZ[:4],
            intensity_array=VALID_INTENSITY[:4],
            precursor_mz=VALID_PRECURSOR_MZ,
            adduct=VALID_ADDUCT,
        )


def test_predict_from_spectrum_rejects_unsupported_adduct():
    with pytest.raises(ValueError, match="Unsupported adduct"):
        api.predict_from_spectrum(
            mz_array=VALID_MZ,
            intensity_array=VALID_INTENSITY,
            precursor_mz=VALID_PRECURSOR_MZ,
            adduct="[M+K]+",
        )


def test_predict_batch_rejects_malformed_item():
    with pytest.raises(ValueError, match="missing 'intensity_array'"):
        api.predict_batch_from_spectra(
            [
                {
                    "mz_array": VALID_MZ,
                    "precursor_mz": VALID_PRECURSOR_MZ,
                    "adduct": VALID_ADDUCT,
                }
            ]
        )


def test_predict_from_spectrum_reports_missing_checkpoints(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing msfiddle checkpoint"):
        api.predict_from_spectrum(
            mz_array=VALID_MZ,
            intensity_array=VALID_INTENSITY,
            precursor_mz=VALID_PRECURSOR_MZ,
            adduct=VALID_ADDUCT,
            resume_path=tmp_path / "missing_tcn.pt",
            rescore_resume_path=tmp_path / "missing_rescore.pt",
        )


def test_predict_from_spectrum_delegates_to_predictor(monkeypatch):
    calls = {}

    class DummyPredictor:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def predict_spectrum(self, *args, **kwargs):
            calls["predict"] = {"args": args, "kwargs": kwargs}
            return [
                {
                    "formula": "C8H9N3S",
                    "score": 0.9,
                    "mass": 179.05,
                    "metadata": {"raw_rescore": 0.9},
                }
            ]

    monkeypatch.setattr(api, "MsFiddlePredictor", DummyPredictor)

    result = api.predict_from_spectrum(
        mz_array=VALID_MZ,
        intensity_array=VALID_INTENSITY,
        precursor_mz=VALID_PRECURSOR_MZ,
        adduct=VALID_ADDUCT,
        device="cpu",
        download_models=False,
    )

    assert calls["init"]["device"] == "cpu"
    assert calls["init"]["download_models"] is False
    assert calls["predict"]["kwargs"]["top_k"] == 5
    assert result[0]["formula"] == "C8H9N3S"


def test_predict_batch_preserves_order_ids_and_candidate_shape(monkeypatch, tmp_path):
    predictor = MsFiddlePredictor(
        resume_path=tmp_path / "missing_tcn.pt",
        rescore_resume_path=tmp_path / "missing_rescore.pt",
    )

    def fake_predict_encoded(self, encoded, *, batch_size):
        return [
            _ModelPrediction(
                spectrum=item,
                vector_string="1;2;3",
                pred_formula="C8H10O",
                pred_mass=122.1,
                pred_atom_count=19.0,
                pred_hc_count=1.25,
                prediction_time=0.01,
            )
            for item in encoded
        ]

    def fake_refine_and_rescore(self, prediction, top_k, *, extra_formulas=None):
        return (
            {
                "formula": ["C8H10O", None],
                "mass": [122.073, None],
                "rescore": [1.2, 0.0],
            },
            122.073,
        )

    monkeypatch.setattr(MsFiddlePredictor, "_predict_encoded", fake_predict_encoded)
    monkeypatch.setattr(
        MsFiddlePredictor, "_refine_and_rescore", fake_refine_and_rescore
    )

    result = predictor.predict_batch(
        [
            {
                "id": "a",
                "mz_array": VALID_MZ,
                "intensity_array": VALID_INTENSITY,
                "precursor_mz": VALID_PRECURSOR_MZ,
                "adduct": VALID_ADDUCT,
            },
            {
                "id": "b",
                "mz_array": [70.0, 90.0, 110.0, 130.0, 160.0],
                "intensity_array": VALID_INTENSITY,
                "precursor_mz": 190.063,
                "adduct": VALID_ADDUCT,
            },
        ],
        top_k=2,
    )

    assert [record["id"] for record in result] == ["a", "b"]
    assert result[0]["candidates"][0]["formula"] == "C8H10O"
    assert result[0]["candidates"][0]["score"] == 1.0
    assert result[0]["candidates"][0]["mass"] == pytest.approx(122.073)
    assert result[0]["candidates"][0]["metadata"]["raw_rescore"] == 1.2
    assert result[0]["metadata"]["adduct"] == VALID_ADDUCT


def test_predict_mgf_loads_native_external_outputs(monkeypatch, tmp_path):
    predictor = MsFiddlePredictor(
        resume_path=tmp_path / "missing_tcn.pt",
        rescore_resume_path=tmp_path / "missing_rescore.pt",
    )
    spectrum = _EncodedSpectrum(
        id="spec-1",
        precursor_type="[M-H]-",
        precursor_mz=100.0,
        collision_energy="50",
        spec=np.zeros(10, dtype=np.float32),
        env=np.zeros(3, dtype=np.float32),
        neutral_add=np.zeros(13, dtype=np.float32),
    )

    buddy_dir = tmp_path / "buddy_output"
    buddy_dir.mkdir()
    pd.DataFrame(
        [
            {
                "identifier": "spec-1",
                "mz": 100.0,
                "rt": "NA",
                "adduct": "[M-H]-",
                "formula_rank_1": "C4H6N4O3",
                "estimated_fdr": 0.2,
                "formula_rank_2": "C5H7NO2",
                "formula_rank_3": "NA",
                "formula_rank_4": "NA",
                "formula_rank_5": "NA",
            }
        ]
    ).to_csv(buddy_dir / "msbuddy_result_summary.tsv", sep="\t", index=False)
    detail_dir = buddy_dir / "spec-1_mz_100.0_rt_NA"
    detail_dir.mkdir()
    pd.DataFrame(
        [
            {"rank": 1, "formula": "C4H6N4O3", "estimated_fdr": 0.2},
            {"rank": 2, "formula": "C5H7NO2", "estimated_fdr": 0.4},
        ]
    ).to_csv(detail_dir / "formula_results.tsv", sep="\t", index=False)

    sirius_path = tmp_path / "formula_identifications_top-5.tsv"
    pd.DataFrame(
        [
            {
                "name": "spec-1",
                "rank": 1,
                "molecularFormula": "C6H12O6",
                "adduct": "[M - H]-",
                "SiriusScore": 1.0,
            }
        ]
    ).to_csv(sirius_path, sep="\t", index=False)

    seen_extra_formulas = []

    def fake_load_mgf_spectra(self, test_data_path):
        return [spectrum]

    def fake_predict_encoded(self, encoded, *, batch_size):
        return [
            _ModelPrediction(
                spectrum=encoded[0],
                vector_string="1;2;3",
                pred_formula="C3H7NO2",
                pred_mass=89.0,
                pred_atom_count=12.0,
                pred_hc_count=1.5,
                prediction_time=0.01,
            )
        ]

    def fake_refine_and_rescore(self, prediction, top_k, *, extra_formulas=None):
        seen_extra_formulas.extend(extra_formulas or [])
        return (
            {"formula": ["C4H6N4O3"], "mass": [146.0], "rescore": [0.9]},
            146.0,
        )

    monkeypatch.setattr(MsFiddlePredictor, "_load_mgf_spectra", fake_load_mgf_spectra)
    monkeypatch.setattr(MsFiddlePredictor, "_predict_encoded", fake_predict_encoded)
    monkeypatch.setattr(
        MsFiddlePredictor, "_refine_and_rescore", fake_refine_and_rescore
    )

    result = predictor.predict_mgf(
        tmp_path / "unused.mgf",
        buddy_path=buddy_dir,
        sirius_path=sirius_path,
        top_k=1,
    )

    assert seen_extra_formulas == ["C4H6N4O3", "C5H7NO2", "C6H12O6"]
    assert result.loc[0, "ID"] == "spec-1"
    assert result.loc[0, "Refined Formula (0)"] == "C4H6N4O3"
