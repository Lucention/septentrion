from pathlib import Path

from typer.testing import CliRunner

from septentrion.cli import app

runner = CliRunner()


def test_run_skeleton_command(tmp_path: Path):
    result = runner.invoke(app, ["run-skeleton", "--out-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "skeleton" / "metrics.json").exists()
    assert "separation_auc" in result.output
