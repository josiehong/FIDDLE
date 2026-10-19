import argparse

import pandas as pd
import pytest

from msfiddle import main as cli
from msfiddle.main import validate_checkpoint_paths


def test_cli_checkpoint_validation_reports_download_command(tmp_path):
    tcn_path = tmp_path / "fiddle_tcn_orbitrap.pt"
    rescore_path = tmp_path / "fiddle_rescore_orbitrap.pt"

    with pytest.raises(FileNotFoundError) as exc_info:
        validate_checkpoint_paths(str(tcn_path), str(rescore_path))

    message = str(exc_info.value)
    assert str(tcn_path) in message
    assert str(rescore_path) in message
    assert "msfiddle-download-models" in message
    assert "msfiddle-checkpoint-paths" in message


@pytest.mark.parametrize(
    "argv",
    [
        ["--demo", "--result_path", "out.csv"],  # pre-subcommand form
        ["predict", "--demo", "--result_path", "out.csv"],
    ],
)
def test_predict_is_the_default_subcommand(monkeypatch, argv):
    seen = {}
    monkeypatch.setattr(cli, "_predict", lambda args, parser: seen.update(vars(args)))
    cli.main(argv)
    assert seen["demo"] and seen["result_path"] == "out.csv"


@pytest.mark.parametrize(
    "extra, expected",
    [([], None), (["--eval"], True), (["--eval", "truth.pkl"], "truth.pkl")],
)
def test_predict_eval_flag(monkeypatch, extra, expected):
    seen = {}
    monkeypatch.setattr(cli, "_predict", lambda args, parser: seen.update(vars(args)))
    cli.main(["--demo", "--result_path", "out.csv"] + extra)
    assert seen["eval"] == expected


def test_missing_truth_file_fails_before_predicting(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(
            ["--demo", "--result_path", str(tmp_path / "out.csv")]
            + ["--eval", str(tmp_path / "missing.pkl")]
        )
    assert exc.value.code == 1
    assert "Truth file not found" in capsys.readouterr().err


def test_evaluate_prints_and_saves_table(tmp_path, capsys):
    truth = tmp_path / "t.mgf"
    truth.write_text("BEGIN IONS\nTITLE=s1\nFORMULA=CH4\n85.0 100.0\nEND IONS\n")
    res = pd.DataFrame(
        {"ID": ["s1"], "Refined Formula (0)": ["C2H6"], "Refined Formula (1)": ["CH4"]}
    )
    args = argparse.Namespace(result_path=str(tmp_path / "out.csv"))
    cli._evaluate(res, str(truth), args, cli.build_parser())
    out = capsys.readouterr().out
    assert out.split("\nspectra")[1].split()[:2] == ["0.0%", "100.0%"]
    saved = pd.read_csv(tmp_path / "out_eval.csv")
    assert list(saved["k"]) == [1, 2]
    assert list(saved["spectrum_acc"]) == [0.0, 1.0]


def test_no_arguments_prints_help_and_fails(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 1
    assert "predict" in capsys.readouterr().out
