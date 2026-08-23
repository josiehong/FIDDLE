import pytest

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
