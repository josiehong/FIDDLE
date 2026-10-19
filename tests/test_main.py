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


def test_eval_subcommand_prints_topk_table(tmp_path, capsys):
    truth = tmp_path / "t.mgf"
    truth.write_text("BEGIN IONS\nTITLE=s1\nFORMULA=CH4\n85.0 100.0\nEND IONS\n")
    result = tmp_path / "r.csv"
    result.write_text("ID,Refined Formula (0),Refined Formula (1)\ns1,C2H6,CH4\n")
    cli.main(["eval", "--result", str(result), "--test_data", str(truth)])
    out = capsys.readouterr().out
    assert "top-2" in out
    assert out.split("\nspectra")[1].split()[:2] == ["0.0%", "100.0%"]


def test_no_arguments_prints_help_and_fails(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 1
    assert "predict" in capsys.readouterr().out
