import pandas as pd
import pytest

from msfiddle.evaluate import formula_vec, load_truth, topk_accuracy


def _result(rows):
    """rows: list of (ID, [refined formulas])."""
    k = max(len(f) for _, f in rows)
    data = {"ID": [r[0] for r in rows]}
    for i in range(k):
        data[f"Refined Formula ({i})"] = [f[i] if i < len(f) else None for _, f in rows]
    return pd.DataFrame(data)


def test_topk_counts_cumulative_hits_per_spectrum_and_compound():
    truth = {
        "a1": (formula_vec("C8H10N4O2"), "caffeine"),
        "a2": (formula_vec("C8H10N4O2"), "caffeine"),
        "b1": (formula_vec("C6H12O6"), "glucose"),
    }
    result = _result(
        [
            ("a1", ["C8H10N4O2", "C7H8N4O2"]),  # hit at rank 1
            ("a2", ["C7H8N4O2", "C8H10N4O2"]),  # hit at rank 2
            ("b1", ["C6H10O5", None]),  # miss
            ("x9", ["C6H12O6", None]),  # no truth: not joined
        ]
    )
    r = topk_accuracy(result, truth, topk=2)
    assert (r.n_spectra, r.n_compounds, r.n_rows) == (3, 2, 4)
    assert r.spectrum_acc == pytest.approx([1 / 3, 2 / 3])
    # caffeine 0.5 then 1.0, glucose 0 then 0, averaged per compound
    assert r.compound_acc == pytest.approx([0.25, 0.5])


def test_formula_matching_ignores_element_order():
    truth = {"s": (formula_vec("C8H10N4O2"), "c")}
    r = topk_accuracy(_result([("s", ["C8H10O2N4"])]), truth, topk=1)
    assert r.spectrum_acc == [1.0]


def test_topk_is_capped_at_available_columns():
    truth = {"s": (formula_vec("CH4"), "c")}
    r = topk_accuracy(_result([("s", ["CH4", "C2H6"])]), truth, topk=10)
    assert len(r.spectrum_acc) == 2


def test_no_joined_rows_is_an_error():
    with pytest.raises(ValueError):
        topk_accuracy(_result([("s", ["CH4"])]), {}, topk=1)


def test_load_truth_from_mgf_formula_field(tmp_path):
    path = tmp_path / "t.mgf"
    path.write_text(
        "BEGIN IONS\nTITLE=s1\nFORMULA=C5H6O4\n85.0 100.0\nEND IONS\n"
        "BEGIN IONS\nTITLE=s2\n85.0 100.0\nEND IONS\n"
    )
    truth = load_truth(path)
    assert set(truth) == {"s1"}
    assert truth["s1"][0] == formula_vec("C5H6O4")


def _mgf(tmp_path, ions):
    """ions: list of (title, formula or None, precursor m/z, adduct, CE, peaks)."""
    text = ""
    for title, formula, mz, adduct, ce, n_peaks in ions:
        text += "BEGIN IONS\n"
        text += f"TITLE={title}\n" if title else ""
        text += f"FORMULA={formula}\n" if formula else ""
        text += f"PRECURSOR_MZ={mz}\nPRECURSOR_TYPE={adduct}\nCOLLISION_ENERGY={ce}\n"
        text += "".join(f"{60 + i}.0 {10 + i}.0\n" for i in range(n_peaks))
        text += "END IONS\n"
    path = tmp_path / "t.mgf"
    path.write_text(text)
    return path


def _example(tmp_path):
    path = _mgf(
        tmp_path,
        [
            ("a1", "C8H10N4O2", 195.09, "[M+H]+", "30", 12),
            ("a2", "C8H10N4O2", 195.09, "[M+H]+", "HCD (NCE 35%)", 6),
            ("b1", "C6H12O6", 179.06, "[M-H]-", "20", 25),
            ("c1", "CH4", 450.0, "[M+Na]+", "40", 60),  # not predicted
            ("d1", None, 300.0, "[M+H]+", "30", 8),  # no truth
            (None, "CH4", 300.0, "[M+H]+", "30", 8),  # untitled
        ],
    )
    result = _result(
        [
            ("a1", ["C8H10N4O2", "C7H8N4O2"]),
            ("a2", ["C7H8N4O2", "C8H10N4O2"]),
            ("b1", ["C6H10O5", None]),
            ("d1", ["C6H12O6", None]),
        ]
    )
    return path, result


def test_details_reproduce_the_aggregate_metrics(tmp_path):
    from msfiddle.evaluate import _accuracy_from_details, load_eval_inputs, topk_details

    path, result = _example(tmp_path)
    inputs = load_eval_inputs(path)
    details = topk_details(result, inputs.truth, info=inputs.info)
    agg = topk_accuracy(result, inputs.truth)
    spectrum_acc, compound_acc, n, n_compounds = _accuracy_from_details(details)
    assert spectrum_acc == pytest.approx(agg.spectrum_acc)
    assert compound_acc == pytest.approx(agg.compound_acc)
    assert (n, n_compounds) == (agg.n_spectra, agg.n_compounds) == (3, 2)

    by_id = details.set_index("ID")
    assert list(by_id["eval_status"]) == [
        "evaluated",
        "evaluated",
        "evaluated",
        "no_truth",
        "not_predicted",
    ]
    assert by_id.loc["a2", "true_rank"] == 2
    assert pd.isna(by_id.loc["b1", "true_rank"])
    assert by_id.loc["a2", "top1_formula"] == "C7H8N4O2"
    assert (by_id.loc["a2", "top1_correct"], by_id.loc["a2", "top2_correct"]) == (
        False,
        True,
    )
    assert pd.isna(by_id.loc["d1", "top1_correct"])  # excluded, not a miss
    assert by_id.loc["b1", "peak_count"] == 25
    assert by_id.loc["b1", "adduct"] == "[M-H]-"


def test_grouped_accuracy_and_coverage(tmp_path):
    from msfiddle.evaluate import (
        coverage,
        grouped_accuracy,
        load_eval_inputs,
        topk_details,
    )

    path, result = _example(tmp_path)
    inputs = load_eval_inputs(path)
    details = topk_details(result, inputs.truth, info=inputs.info)
    groups = grouped_accuracy(details).set_index(["group", "value"])
    assert groups.loc[("adduct", "[M+H]+"), "n_spectra"] == 2
    assert groups.loc[("adduct", "[M+H]+"), "top1_acc"] == 0.5
    assert groups.loc[("precursor_mz", "0-200"), "n_spectra"] == 3
    assert groups.loc[("peak_count", "20-50"), "top2_acc"] == 0.0
    assert groups.loc[("nce", "20-40"), "n_spectra"] == 1  # a2: NCE 35
    for name in ("precursor_mz", "adduct", "peak_count", "nce"):
        assert groups.loc[name, "n_spectra"].sum() == 3

    assert coverage(details, len(result), inputs) == {
        "input_spectra": 6,
        "duplicate_input_titles": 0,
        "spectra_with_truth": 4,
        "result_rows": 4,
        "duplicate_result_ids": 0,
        "evaluated_rows": 3,
        "result_rows_without_truth": 1,
        "truth_spectra_not_predicted": 1,
    }


def test_repeated_titles_take_the_last_truth_and_are_counted(tmp_path):
    from msfiddle.evaluate import coverage, load_eval_inputs, topk_details

    path = _mgf(
        tmp_path,
        [
            ("s", "CH4", 100.0, "[M+H]+", "30", 5),
            ("s", "C2H6", 100.0, "[M+H]+", "30", 5),
        ],
    )
    inputs = load_eval_inputs(path)
    assert inputs.truth["s"][0] == formula_vec("C2H6")
    assert load_truth(path)["s"][0] == formula_vec("C2H6")
    result = _result([("s", ["C2H6"]), ("s", ["CH4"])])
    details = topk_details(result, inputs.truth)
    cov = coverage(details, len(result), inputs)
    assert cov["duplicate_input_titles"] == 1
    assert cov["duplicate_result_ids"] == 1
    assert cov["evaluated_rows"] == 2  # each result row counts, as before


def test_truth_follows_the_copy_prediction_kept(tmp_path):
    from msfiddle.api import MsFiddlePredictor
    from msfiddle.evaluate import load_eval_inputs

    peaks = "60.0 10.0\n85.0 50.0\n100.0 20.0\n125.0 35.0\n150.0 15.0\n"

    def ion(smiles, ce):
        return (
            f"BEGIN IONS\nTITLE=s\nPRECURSOR_MZ=180.063\nPRECURSOR_TYPE=[M+H]+\n"
            f"COLLISION_ENERGY={ce}\nSMILES={smiles}\n{peaks}END IONS\n"
        )

    # The later copy has a stepped energy, which the Orbitrap model skips
    path = tmp_path / "dup.mgf"
    path.write_text(ion("OCC(O)CO", "35") + ion("CCCCCC", "HCD (NCE 20-30-40%)"))
    predictor = MsFiddlePredictor(instrument_type="orbitrap")
    with pytest.warns(UserWarning, match="stepped"):
        predictor._load_mgf_spectra(path)
    assert predictor._kept_index == {"s": 0}
    inputs = load_eval_inputs(path, predictor._kept_index)
    assert inputs.truth["s"][0] == formula_vec("C3H8O3")
    assert load_eval_inputs(path).truth["s"][0] == formula_vec("C6H14")
