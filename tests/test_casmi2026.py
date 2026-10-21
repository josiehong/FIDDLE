import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from msfiddle import api, main as cli
from msfiddle.api import MsFiddlePredictor, _normalize_instrument_type
from msfiddle.download import INSTRUMENTS
from msfiddle.settings import load_config
from msfiddle.utils import filter_spec, spec2arr, split_by_scaffold
from msfiddle.utils.casmi2026 import (
    collision_energy_string,
    iter_spectra,
    normalize_adduct,
    resolve_columns,
)
from msfiddle.utils.pkl_utils import mean_collision_energy, resolve_collision_energy

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "msfiddle", "config")
TIMSTOF_CONFIG = os.path.join(CONFIG_DIR, "fiddle_tcn_timstof.yml")
QTOF_CONFIG = os.path.join(CONFIG_DIR, "fiddle_tcn_qtof.yml")

CAFFEINE = "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"  # C8H10N4O2, 194.0804
PEAKS = [56.05, 69.045, 83.06, 110.071, 138.066, 195.088]
INTENSITIES = [0.1, 0.4, 0.2, 0.6, 1.0, 0.3]


@pytest.fixture
def parquet(tmp_path):
    """Six rows in the Kaggle train.parquet layout."""
    base = {
        "ms2_mzs": PEAKS,
        "ms2_normalized_intensities": INTENSITIES,
        "normalized_smiles": CAFFEINE,
        "inchikey": "RYYVLZVUVIJVGH-UHFFFAOYSA-N",
        "instrument_type": "timsTOF",
        "collision_energy_ev": [40.0],
        "collision_energy_orig": "40",
    }
    rows = [
        {
            **base,
            "ingest_lib": "enveda-180",
            "adduct": "[M+H]+",
            "precursor_mz": 195.0877,
        },
        {
            **base,
            "ingest_lib": "enveda-180",
            "adduct": "[M+FA-H]-",
            "precursor_mz": 239.0786,
            "collision_energy_orig": "-40",
        },
        {
            **base,
            "ingest_lib": "enveda-180",
            "adduct": "[M+K]+",
            "precursor_mz": 233.0435,
        },
        {
            **base,
            "ingest_lib": "gnps",
            "instrument_type": "Orbitrap",
            "adduct": "[M+H]+",
            "precursor_mz": 195.0877,
        },
        {
            **base,
            "ingest_lib": "enveda-np-examples",
            "adduct": "[M+H]1+",
            "precursor_mz": 195.0877,
            "collision_energy_ev": [20.0, 40.0, 60.0],
            "collision_energy_orig": "20,40,60",
        },
        {
            **base,
            "ingest_lib": "enveda-180",
            "adduct": "[M+H]+",
            "precursor_mz": 195.0877,
            "ms2_mzs": [],
            "ms2_normalized_intensities": [],
        },
        {
            **base,
            "ingest_lib": "enveda-180",
            "adduct": "[2M+Na]+",
            "precursor_mz": 411.1500,
        },
    ]
    path = tmp_path / "train.parquet"
    pq.write_table(pa.table({k: [r[k] for r in rows] for k in rows[0]}), path)
    return str(path)


@pytest.mark.parametrize(
    "adduct, expected",
    [
        ("[M+H]+", "[M+H]+"),
        ("[M+H]1+", "[M+H]+"),
        ("[M-H]1-", "[M-H]-"),
        ("[M+FA-H]-", "[M+CH2O2-H]-"),
        ("[M+HCOO]-", "[M+CH2O2-H]-"),
        (" [M+Na]+ ", "[M+Na]+"),
        ("[M+K]+", "[M+K]+"),
    ],
)
def test_normalize_adduct(adduct, expected):
    assert normalize_adduct(adduct) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (40, "40.0"),
        (-40, "40.0"),
        ("40", "40.0"),
        ("-60", "60.0"),
        ("20,40,60", "20,40,60"),
        ([20, 40, 60], "20.0,40.0,60.0"),
        ("HCD (NCE 35%)", "HCD (NCE 35%)"),
        (None, "Unknown"),
        (float("nan"), "Unknown"),
        ("", "Unknown"),
    ],
)
def test_collision_energy_string(value, expected):
    assert collision_energy_string(value) == expected


def test_resolve_columns_reports_missing():
    with pytest.raises(ValueError, match="lacks columns for mz, intensity"):
        resolve_columns(["normalized_smiles", "precursor_mz", "adduct"])


def test_iter_spectra_reads_timstof_rows_by_default(parquet):
    spectra = list(iter_spectra(parquet, verbose=False))
    assert len(spectra) == 5  # the Orbitrap row and the empty peak list are skipped
    assert {s["params"]["source_instrument"] for s in spectra} == {
        "enveda-180",
        "enveda-np-examples",
    }
    assert {s["params"]["instrument_type"] for s in spectra} == {"timsTOF"}
    assert {s["params"]["inchi_key"] for s in spectra} == {
        "RYYVLZVUVIJVGH-UHFFFAOYSA-N"
    }
    assert len({s["params"]["title"] for s in spectra}) == 5

    formate = next(
        s for s in spectra if s["params"]["precursor_type"] == "[M+CH2O2-H]-"
    )
    assert formate["params"]["ionmode"] == "N"
    assert formate["params"]["collision_energy"] == "40.0"
    assert isinstance(formate["m/z array"], np.ndarray)

    merged = next(
        s for s in spectra if s["params"]["source_instrument"] == "enveda-np-examples"
    )
    assert merged["params"]["precursor_type"] == "[M+H]+"
    assert merged["params"]["collision_energy"] == "20.0,40.0,60.0"


def test_iter_spectra_library_selection_and_cap(parquet):
    gnps = list(iter_spectra(parquet, libraries=["gnps"], verbose=False))
    assert [s["params"]["source_instrument"] for s in gnps] == ["gnps"]
    assert gnps[0]["params"]["instrument_type"] == "Orbitrap"
    with pytest.raises(ValueError, match="Unknown libraries"):
        list(iter_spectra(parquet, libraries=["nist"], verbose=False))
    assert (
        len(list(iter_spectra(parquet, max_spectra_per_compound=2, verbose=False))) == 2
    )


def _filter_and_encode(parquet, config):
    spectra = list(iter_spectra(parquet, verbose=False))
    kept, smiles = filter_spec(
        spectra,
        config["casmi2026_timstof"],
        type2charge=config["encoding"]["type2charge"],
    )
    data, bad = spec2arr(kept, config["encoding"])
    return kept, smiles, data, bad


def test_timstof_config_filters_and_encodes_merged_spectra(parquet):
    config = load_config(TIMSTOF_CONFIG)
    kept, smiles, data, bad = _filter_and_encode(parquet, config)
    assert sorted(s["params"]["precursor_type"] for s in kept) == [
        "[M+CH2O2-H]-",
        "[M+H]+",
        "[M+H]+",
    ]
    assert len(set(smiles)) == 1  # [M+K]+ and [2M+Na]+ dropped, one compound
    assert len(data) == 3 and not bad

    single = next(d for d in data if d["title"] == "enveda-180_1")
    formate = next(d for d in data if d["title"] == "enveda-180_2")
    merged = next(d for d in data if d["title"].startswith("enveda-np-examples"))
    # merged 20,40,60 eV is encoded at 40 eV, like the single 40 eV spectrum
    assert merged["env"][1] == pytest.approx(single["env"][1])
    # the formate adduct is folded into [M-H]- and the formula gains CH2O2
    assert single["env"][2] == config["encoding"]["precursor_type"]["[M+H]+"]
    assert formate["env"][2] == config["encoding"]["precursor_type"]["[M-H]-"]
    assert formate["formula"][0] == single["formula"][0] + 1
    assert formate["mass"] == pytest.approx(single["mass"] + 46.005479, abs=1e-4)


def test_qtof_config_still_drops_merged_spectra(parquet):
    config = load_config(QTOF_CONFIG)
    config["casmi2026_timstof"] = load_config(TIMSTOF_CONFIG)["casmi2026_timstof"]
    assert len(_filter_and_encode(parquet, config)[2]) == 2


@pytest.mark.parametrize(
    "ce, expected",
    [
        ("20,40,60", "40.0"),
        ("30-50 eV", "40.0"),
        ("HCD (NCE 20-30-40%)", "30.0 nce"),
        ("stepped", None),
    ],
)
def test_mean_collision_energy(ce, expected):
    assert mean_collision_energy(ce) == expected


def test_resolve_collision_energy_modes():
    assert resolve_collision_energy("40", "drop") == "40"
    assert resolve_collision_energy("20,40,60", "drop") is None
    assert resolve_collision_energy("20,40,60", "mean") == "40.0"
    with pytest.raises(ValueError, match="stepped_collision_energy"):
        resolve_collision_energy("20,40,60", "average")


def test_split_by_scaffold_keeps_scaffolds_together():
    benzenes = ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C"]
    cyclohexanes = ["C1CCCCC1O", "C1CCCCC1N"]
    acyclic = ["CCO", "CCCO", "CCCCO", "CCCCCO", "CCCCCCO"]
    smiles = benzenes + cyclohexanes + acyclic
    train, val, test = split_by_scaffold(smiles, train_ratio=0.8, val_ratio=0.1, seed=1)
    assert train | val | test == set(smiles)
    assert not (train & val) and not (train & test) and not (val & test)
    assert len(test) >= 2 and len(val) >= 1
    for group in (benzenes, cyclohexanes):
        assert (
            len({s in train for s in group}) == 1
            and len({s in test for s in group}) == 1
        )


def test_timstof_model_matches_qtof_for_transfer():
    timstof, qtof = load_config(TIMSTOF_CONFIG), load_config(QTOF_CONFIG)
    assert timstof["model"] == qtof["model"]
    assert timstof["encoding"]["precursor_type"] == qtof["encoding"]["precursor_type"]
    assert timstof["encoding"]["stepped_collision_energy"] == "mean"
    assert timstof["encoding"]["type2charge"]["[M+CH2O2-H]-"] == "-1"


def test_api_accepts_timstof_and_downloads_its_checkpoints(monkeypatch, tmp_path):
    assert _normalize_instrument_type("TimsTOF") == "timstof"
    assert "timstof" in INSTRUMENTS
    predictor = MsFiddlePredictor(
        instrument_type="timstof",
        resume_path=tmp_path / "tcn.pt",
        rescore_resume_path=tmp_path / "rescore.pt",
        download_models=True,
    )
    assert predictor.config_path.name == "fiddle_tcn_timstof.yml"
    requested = []
    monkeypatch.setattr(
        api,
        "download_pretrained_models",
        lambda instruments: requested.extend(instruments),
    )
    with pytest.raises(FileNotFoundError):
        predictor._ensure_model_files()
    assert requested == ["timstof"]


def test_api_encodes_merged_energy_and_formate_with_timstof_config(tmp_path):
    predictor = MsFiddlePredictor(
        instrument_type="timstof",
        resume_path=tmp_path / "tcn.pt",
        rescore_resume_path=tmp_path / "rescore.pt",
    )

    def encode(**kw):
        return predictor._encode_spectrum(
            mz_array=PEAKS, intensity_array=INTENSITIES, spec_id="s", **kw
        )

    merged = encode(precursor_mz=195.0877, adduct="[M+H]+", collision_energy="20,40,60")
    single = encode(precursor_mz=195.0877, adduct="[M+H]+", collision_energy=40)
    assert merged.env[1] == pytest.approx(single.env[1])
    formate = encode(precursor_mz=239.0786, adduct="[M+FA-H]-", collision_energy=40)
    assert formate.env[2] == predictor.config["encoding"]["precursor_type"]["[M-H]-"]
    assert list(formate.neutral_add[:3]) == [1, 2, 2]  # C, H, O of CH2O2


def test_cli_accepts_timstof(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "_predict", lambda args, parser: seen.update(vars(args)))
    cli.main(["--demo", "--instrument_type", "timstof", "--result_path", "o.csv"])
    assert seen["instrument_type"] == "timstof"
