import pandas as pd
import pytest

from msfiddle.external_results import load_buddy_results, load_sirius_results


def test_load_buddy_results_preserves_normalized_csv(tmp_path):
    path = tmp_path / "buddy.csv"
    pd.DataFrame(
        [
            {
                "ID": "spec-1",
                "Adduct": "[M-H]-",
                "Pred Formula (1)": "C4H6N4O3",
                "Pred Formula (2)": "NA",
                "Pred Formula (3)": "NA",
                "Pred Formula (4)": "NA",
                "Pred Formula (5)": "NA",
                "BUDDY Score (1)": "0.1",
                "BUDDY Score (2)": "NA",
                "BUDDY Score (3)": "NA",
                "BUDDY Score (4)": "NA",
                "BUDDY Score (5)": "NA",
            }
        ]
    ).to_csv(path, index=False)

    with pytest.warns(DeprecationWarning, match="msfiddle-normalized BUDDY"):
        result = load_buddy_results(path)

    assert result.loc[0, "ID"] == "spec-1"
    assert result.loc[0, "Pred Formula (1)"] == "C4H6N4O3"
    assert result.loc[0, "BUDDY Score (1)"] == pytest.approx(0.1)
    assert pd.isna(result.loc[0, "Pred Formula (2)"])


def test_load_buddy_results_converts_msbuddy_summary(tmp_path):
    path = tmp_path / "msbuddy_result_summary.tsv"
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
    ).to_csv(path, sep="\t", index=False)

    result = load_buddy_results(path)

    assert result.loc[0, "ID"] == "spec-1"
    assert result.loc[0, "Adduct"] == "[M-H]-"
    assert result.loc[0, "Pred Formula (1)"] == "C4H6N4O3"
    assert result.loc[0, "BUDDY Score (1)"] == pytest.approx(0.2)
    assert result.loc[0, "Pred Formula (2)"] == "C5H7NO2"
    assert pd.isna(result.loc[0, "BUDDY Score (2)"])


def test_load_buddy_results_uses_msbuddy_detail_scores_from_directory(tmp_path):
    summary = tmp_path / "msbuddy_result_summary.tsv"
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
    ).to_csv(summary, sep="\t", index=False)
    detail_dir = tmp_path / "spec-1_mz_100.0_rt_NA"
    detail_dir.mkdir()
    pd.DataFrame(
        [
            {"rank": 2, "formula": "C5H7NO2", "estimated_fdr": 0.4},
            {"rank": 1, "formula": "C4H6N4O3", "estimated_fdr": 0.2},
        ]
    ).to_csv(detail_dir / "formula_results.tsv", sep="\t", index=False)

    result = load_buddy_results(tmp_path)

    assert result.loc[0, "Pred Formula (1)"] == "C4H6N4O3"
    assert result.loc[0, "BUDDY Score (1)"] == pytest.approx(0.2)
    assert result.loc[0, "Pred Formula (2)"] == "C5H7NO2"
    assert result.loc[0, "BUDDY Score (2)"] == pytest.approx(0.4)


def test_load_buddy_results_matches_detail_dir_when_summary_mz_format_differs(tmp_path):
    # Real msbuddy rounds mz in the summary, but a future version (or a
    # hand-edited summary) could disagree with the directory name. The
    # directory's identifier prefix alone should be enough to match.
    summary = tmp_path / "msbuddy_result_summary.tsv"
    pd.DataFrame(
        [
            {
                "identifier": "spec-1",
                "mz": 111.020160000,  # padded zeros vs. dir name's "111.0202"
                "rt": "NA",
                "adduct": "[M-H]-",
                "formula_rank_1": "C4H4N2O2",
                "estimated_fdr": 0.0,
                "formula_rank_2": "NA",
                "formula_rank_3": "NA",
                "formula_rank_4": "NA",
                "formula_rank_5": "NA",
            }
        ]
    ).to_csv(summary, sep="\t", index=False)
    detail_dir = tmp_path / "spec-1_mz_111.0202_rt_NA"
    detail_dir.mkdir()
    pd.DataFrame(
        [
            {"rank": 1, "formula": "C4H4N2O2", "estimated_fdr": 0.0},
            {"rank": 2, "formula": "C5H8O2", "estimated_fdr": 0.5},
        ]
    ).to_csv(detail_dir / "formula_results.tsv", sep="\t", index=False)

    result = load_buddy_results(tmp_path)

    assert result.loc[0, "Pred Formula (1)"] == "C4H4N2O2"
    assert result.loc[0, "Pred Formula (2)"] == "C5H8O2"
    assert result.loc[0, "BUDDY Score (2)"] == pytest.approx(0.5)


def test_load_buddy_results_drops_rows_with_missing_identifier(tmp_path):
    path = tmp_path / "msbuddy_result_summary.tsv"
    pd.DataFrame(
        [
            {
                "identifier": "NA",
                "mz": 100.0,
                "rt": "NA",
                "adduct": "[M-H]-",
                "formula_rank_1": "C4H6N4O3",
                "estimated_fdr": 0.2,
                "formula_rank_2": "NA",
                "formula_rank_3": "NA",
                "formula_rank_4": "NA",
                "formula_rank_5": "NA",
            },
            {
                "identifier": "spec-1",
                "mz": 100.0,
                "rt": "NA",
                "adduct": "[M-H]-",
                "formula_rank_1": "C5H7NO2",
                "estimated_fdr": 0.3,
                "formula_rank_2": "NA",
                "formula_rank_3": "NA",
                "formula_rank_4": "NA",
                "formula_rank_5": "NA",
            },
        ]
    ).to_csv(path, sep="\t", index=False)

    result = load_buddy_results(path)

    assert len(result) == 1
    assert result.loc[0, "ID"] == "spec-1"


def test_load_sirius_results_preserves_normalized_csv(tmp_path):
    path = tmp_path / "sirius.csv"
    pd.DataFrame(
        [
            {
                "ID": "spec-1",
                "Pred Formula (1)": "C3H7NO2",
                "Pred Adduct (1)": "[M - H]-",
                "SIRIUS Score (1)": "12.5",
                "Pred Formula (2)": "NA",
                "Pred Adduct (2)": "NA",
                "SIRIUS Score (2)": "NA",
                "Pred Formula (3)": "NA",
                "Pred Adduct (3)": "NA",
                "SIRIUS Score (3)": "NA",
                "Pred Formula (4)": "NA",
                "Pred Adduct (4)": "NA",
                "SIRIUS Score (4)": "NA",
                "Pred Formula (5)": "NA",
                "Pred Adduct (5)": "NA",
                "SIRIUS Score (5)": "NA",
            }
        ]
    ).to_csv(path, index=False)

    with pytest.warns(DeprecationWarning, match="msfiddle-normalized SIRIUS"):
        result = load_sirius_results(path)

    assert result.loc[0, "ID"] == "spec-1"
    assert result.loc[0, "Pred Formula (1)"] == "C3H7NO2"
    assert result.loc[0, "SIRIUS Score (1)"] == pytest.approx(12.5)
    assert pd.isna(result.loc[0, "Pred Formula (2)"])


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


def test_load_sirius_results_accepts_prefixed_column_variants(tmp_path):
    path = tmp_path / "formula_identifications_top-5.tsv"
    pd.DataFrame(
        [
            {
                "SIRIUS_mappingFeatureId": "spec-1",
                "SIRIUS_formula": "C5H7NO2",
                "SIRIUS_ionType": "[M - H]-",
                "SIRIUS_rankingScore": 0.4,
            },
            {
                "SIRIUS_mappingFeatureId": "spec-1",
                "SIRIUS_formula": "C4H6N4O3",
                "SIRIUS_ionType": "[M - H]-",
                "SIRIUS_rankingScore": 0.9,
            },
        ]
    ).to_csv(path, sep="\t", index=False)

    result = load_sirius_results(path)

    assert result.loc[0, "ID"] == "spec-1"
    assert result.loc[0, "Pred Formula (1)"] == "C4H6N4O3"
    assert result.loc[0, "SIRIUS Score (1)"] == pytest.approx(0.9)
    assert result.loc[0, "Pred Formula (2)"] == "C5H7NO2"
