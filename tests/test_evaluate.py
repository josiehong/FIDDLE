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
