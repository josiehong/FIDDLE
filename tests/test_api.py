import warnings

import pytest
import numpy as np
import pandas as pd

import msfiddle.api as api
from msfiddle.api import MsFiddlePredictor, _EncodedSpectrum, _ModelPrediction

VALID_MZ = [60.0, 85.0, 100.0, 125.0, 150.0]
VALID_INTENSITY = [10.0, 50.0, 20.0, 35.0, 15.0]
VALID_PRECURSOR_MZ = 180.063
VALID_ADDUCT = "[M+H]+"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"intensity_array": VALID_INTENSITY[:-1]}, "same length"),
        (
            {"mz_array": VALID_MZ[:4], "intensity_array": VALID_INTENSITY[:4]},
            "at least 5 peaks",
        ),
        ({"adduct": "[M+K]+"}, "Unsupported adduct"),
    ],
)
def test_predict_from_spectrum_rejects_invalid_input(overrides, message):
    spectrum = {
        "mz_array": VALID_MZ,
        "intensity_array": VALID_INTENSITY,
        "precursor_mz": VALID_PRECURSOR_MZ,
        "adduct": VALID_ADDUCT,
        **overrides,
    }
    with pytest.raises(ValueError, match=message):
        api.predict_from_spectrum(**spectrum)


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

    def fake_refinement_args(
        self, prediction, top_k, *, extra_formulas=None, isotope_shift=0
    ):
        seen_extra_formulas.extend(extra_formulas or [])
        return (), 146.0

    def fake_run_refinement(jobs, num_workers, verbose):
        return [({"formula": ["C4H6N4O3"], "mass": [146.0]}, 0.0) for _ in jobs]

    def fake_rescore(self, spectrum, refined_results, top_k):
        return {**refined_results, "rescore": [0.9]}

    monkeypatch.setattr(MsFiddlePredictor, "_load_mgf_spectra", fake_load_mgf_spectra)
    monkeypatch.setattr(MsFiddlePredictor, "_predict_encoded", fake_predict_encoded)
    monkeypatch.setattr(MsFiddlePredictor, "_refinement_args", fake_refinement_args)
    monkeypatch.setattr(api, "_run_refinement", fake_run_refinement)
    monkeypatch.setattr(MsFiddlePredictor, "_rescore_candidates", fake_rescore)

    result = predictor.predict_mgf(
        tmp_path / "unused.mgf",
        buddy_path=buddy_dir,
        sirius_path=sirius_path,
        top_k=1,
    )

    assert seen_extra_formulas == ["C4H6N4O3", "C5H7NO2", "C6H12O6"]
    assert result.loc[0, "ID"] == "spec-1"
    assert result.loc[0, "Refined Formula (0)"] == "C4H6N4O3"


def test_predict_mgf_skips_stepped_collision_energy_with_warning(tmp_path):
    peaks = "".join(f"{mz} {i}\n" for mz, i in zip(VALID_MZ, VALID_INTENSITY))
    path = tmp_path / "in.mgf"
    path.write_text(
        "".join(
            f"BEGIN IONS\nTITLE={t}\nPRECURSOR_MZ={VALID_PRECURSOR_MZ}\n"
            f"PRECURSOR_TYPE={VALID_ADDUCT}\nCOLLISION_ENERGY={ce}\n{peaks}END IONS\n"
            for t, ce in [("single", "35"), ("stepped", "HCD (NCE 20-30-40%)")]
        )
    )
    predictor = MsFiddlePredictor(
        resume_path=tmp_path / "missing_tcn.pt",
        rescore_resume_path=tmp_path / "missing_rescore.pt",
    )
    with pytest.warns(UserWarning, match="Skipped 1 spectra with a stepped"):
        encoded = predictor._load_mgf_spectra(path)
    assert [e.id for e in encoded] == ["single"]


def test_isotope_offsets_pool_candidates_and_penalize_shifted_ones():
    parts = [
        (1, {"formula": ["C7H14O6", "C8H10N4O2"], "mass": [194.08, 194.08]}),
        (0, {"formula": ["C8H10N4O2", None], "mass": [194.08, None]}),
    ]
    merged, shift_of = api._merge_isotope_refinements(parts)
    assert merged["formula"] == ["C8H10N4O2", "C7H14O6"]  # smallest k kept
    assert shift_of == {"C8H10N4O2": 0, "C7H14O6": 1}

    rescored = {**merged, "rescore": [0.5, 0.9]}
    out = api._apply_isotope_penalty(rescored, shift_of, 0.5, top_k=3)
    assert out["formula"] == ["C8H10N4O2", "C7H14O6", None]  # 0.9 - 0.5 < 0.5
    assert out["isotope_shift"] == [0, 1, None]
    assert out["rescore"] == [0.5, 0.9, 0.0]


@pytest.mark.parametrize("matching", [True, False])
def test_warns_when_the_rescore_model_belongs_to_another_tcn(tmp_path, matching):
    torch = pytest.importorskip("torch")
    from msfiddle.model_tcn import (
        FormulaEncoder,
        MS2FNet_tcn,
        RescoreHead,
        tcn_hash,
    )

    predictor = MsFiddlePredictor(
        instrument_type="qtof",
        resume_path=tmp_path / "tcn.pt",
        rescore_resume_path=tmp_path / "rescore.pt",
        device="cpu",
    )
    model_config = predictor.config["model"]
    tcn = MS2FNet_tcn(model_config).state_dict()
    torch.save({"model_state_dict": tcn}, predictor.resume_path)
    torch.save(
        {
            "formula_encoder_state_dict": FormulaEncoder(model_config).state_dict(),
            "rescore_head_state_dict": RescoreHead(model_config).state_dict(),
            "tcn_hash": tcn_hash(tcn) if matching else "0" * 16,
        },
        predictor.rescore_resume_path,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        predictor._ensure_models_loaded()
    mismatch = [w for w in caught if "different TCN" in str(w.message)]
    assert len(mismatch) == (0 if matching else 1)
