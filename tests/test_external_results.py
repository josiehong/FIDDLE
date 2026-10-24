import pandas as pd
import pytest

from msfiddle.external_results import load_buddy_results, load_sirius_results


def _write_msbuddy_summary(path, rows):
    """rows: (identifier, mz, [formulas], top-1 fdr); ranks beyond the list are NA."""
    records = []
    for identifier, mz, formulas, fdr in rows:
        record = {"identifier": identifier, "mz": mz, "rt": "NA", "adduct": "[M-H]-"}
        for k in range(1, 6):
            record[f"formula_rank_{k}"] = (
                formulas[k - 1] if k <= len(formulas) else "NA"
            )
        record["estimated_fdr"] = fdr
        records.append(record)
    pd.DataFrame(records).to_csv(path, sep="\t", index=False)


def test_load_buddy_results_converts_msbuddy_summary(tmp_path):
    path = tmp_path / "msbuddy_result_summary.tsv"
    _write_msbuddy_summary(
        path,
        [
            ("NA", 100.0, ["C4H6N4O3"], 0.2),  # missing identifier: dropped
            ("spec-1", 100.0, ["C4H6N4O3", "C5H7NO2"], 0.2),
        ],
    )

    result = load_buddy_results(path)

    assert list(result["ID"]) == ["spec-1"]
    assert result.loc[0, "Adduct"] == "[M-H]-"
    assert result.loc[0, "Pred Formula (1)"] == "C4H6N4O3"
    assert result.loc[0, "BUDDY Score (1)"] == pytest.approx(0.2)
    assert result.loc[0, "Pred Formula (2)"] == "C5H7NO2"
    assert pd.isna(result.loc[0, "BUDDY Score (2)"])


def test_load_buddy_results_uses_detail_scores_from_directory(tmp_path):
    # The summary pads mz with zeros while the directory name rounds it; the
    # identifier prefix alone must match them.
    _write_msbuddy_summary(
        tmp_path / "msbuddy_result_summary.tsv",
        [("spec-1", 111.020160000, ["C4H4N2O2"], 0.0)],
    )
    detail_dir = tmp_path / "spec-1_mz_111.0202_rt_NA"
    detail_dir.mkdir()
    pd.DataFrame(
        [
            {"rank": 2, "formula": "C5H8O2", "estimated_fdr": 0.5},
            {"rank": 1, "formula": "C4H4N2O2", "estimated_fdr": 0.0},
        ]
    ).to_csv(detail_dir / "formula_results.tsv", sep="\t", index=False)

    result = load_buddy_results(tmp_path)

    assert result.loc[0, "Pred Formula (1)"] == "C4H4N2O2"
    assert result.loc[0, "Pred Formula (2)"] == "C5H8O2"
    assert result.loc[0, "BUDDY Score (2)"] == pytest.approx(0.5)


def test_load_sirius_results_converts_native_top_k_summary(tmp_path):
    path = tmp_path / "formula_identifications_top-5.tsv"
    pd.DataFrame(
        [
            {
                "name": "spec-1",
                "rank": 2,
                "molecularFormula": "C5H7NO2",
                "adduct": "[M - H]-",
                "SiriusScore": 5.0,
            },
            {
                "name": "spec-1",
                "rank": 1,
                "molecularFormula": "[C4H6N4O3 - H]-",
                "adduct": "[M - H]-",
                "SiriusScore": 7.0,
            },
        ]
    ).to_csv(path, sep="\t", index=False)

    result = load_sirius_results(path)

    assert result.loc[0, "ID"] == "spec-1"
    assert result.loc[0, "Pred Formula (1)"] == "C4H6N4O3"
    assert result.loc[0, "Pred Adduct (1)"] == "[M - H]-"
    assert result.loc[0, "SIRIUS Score (1)"] == pytest.approx(7.0)
    assert result.loc[0, "Pred Formula (2)"] == "C5H7NO2"


def test_load_sirius_results_finds_formula_summary_in_directory(tmp_path):
    pd.DataFrame(
        [
            {
                "mappingFeatureId": "spec-1",
                "molecularFormula": "C4H6N4O3",
                "adduct": "[M - H]-",
                "rankingScore": 0.8,
            }
        ]
    ).to_csv(tmp_path / "formula_identifications.tsv", sep="\t", index=False)

    result = load_sirius_results(tmp_path)

    assert result.loc[0, "ID"] == "spec-1"
    assert result.loc[0, "Pred Formula (1)"] == "C4H6N4O3"
    assert result.loc[0, "SIRIUS Score (1)"] == pytest.approx(0.8)


def test_load_sirius_results_recovers_titles_from_sirius6_mapping_ids(tmp_path):
    path = tmp_path / "formula_identifications_top-5.tsv"
    pd.DataFrame(
        [
            {
                "formulaRank": 1,
                "molecularFormula": "C10H9N",
                "adduct": "[M + H]+",
                "SiriusScore": 12.9,
                "alignedFeatureId": 901721886549270209,
                "mappingFeatureId": mapping_id,
            }
            for mapping_id in ["901721886549270209_UNKNOWN_casmi16_620", "spec-2"]
        ]
    ).to_csv(path, sep="\t", index=False)

    result = load_sirius_results(path)

    assert result["ID"].tolist() == ["casmi16_620", "spec-2"]


def test_load_sirius_results_accepts_prefixed_column_variants(tmp_path):
    path = tmp_path / "formula_identifications_top-5.tsv"
    pd.DataFrame(
        [
            {
                "SIRIUS_mappingFeatureId": "spec-1",
                "SIRIUS_formula": formula,
                "SIRIUS_ionType": "[M - H]-",
                "SIRIUS_rankingScore": score,
            }
            for formula, score in [("C5H7NO2", 0.4), ("C4H6N4O3", 0.9)]
        ]
    ).to_csv(path, sep="\t", index=False)

    result = load_sirius_results(path)

    assert result.loc[0, "Pred Formula (1)"] == "C4H6N4O3"
    assert result.loc[0, "SIRIUS Score (1)"] == pytest.approx(0.9)
    assert result.loc[0, "Pred Formula (2)"] == "C5H7NO2"


@pytest.mark.parametrize(
    "loader, tool, columns",
    [
        (load_buddy_results, "BUDDY", ["Pred Formula", "BUDDY Score"]),
        (
            load_sirius_results,
            "SIRIUS",
            ["Pred Formula", "Pred Adduct", "SIRIUS Score"],
        ),
    ],
)
def test_legacy_normalized_csv_is_deprecated_but_loads(tmp_path, loader, tool, columns):
    row = {"ID": "spec-1", "Adduct": "[M-H]-"}
    for k in range(1, 6):
        for col in columns:
            row[f"{col} ({k})"] = "NA"
    row["Pred Formula (1)"] = "C4H6N4O3"
    row[f"{tool} Score (1)"] = "0.1"
    if "Pred Adduct" in columns:
        row["Pred Adduct (1)"] = "[M - H]-"
    path = tmp_path / f"{tool.lower()}.csv"
    pd.DataFrame([row]).to_csv(path, index=False)

    with pytest.warns(DeprecationWarning, match=f"msfiddle-normalized {tool}"):
        result = loader(path)

    assert result.loc[0, "ID"] == "spec-1"
    assert result.loc[0, "Pred Formula (1)"] == "C4H6N4O3"
    assert result.loc[0, f"{tool} Score (1)"] == pytest.approx(0.1)
    assert pd.isna(result.loc[0, "Pred Formula (2)"])
