from __future__ import annotations

from pathlib import Path

from tests.path_helpers import REPO_ROOT
from typer.testing import CliRunner

from tests.gabion.cli.cli_commands_cases import _invoke


# gabion:evidence E:call_footprint::tests/test_cli_live_repo.py::test_cli_docflow::cli.py::gabion.cli.app
# gabion:behavior primary=desired
def test_cli_docflow() -> None:
    runner = CliRunner()
    result = _invoke(
        runner,
        [
            "docflow",
            "--root",
            str(REPO_ROOT),
            "--no-fail-on-violations",
        ],
    )
    assert result.exit_code == 0


# gabion:evidence E:call_footprint::tests/test_cli_live_repo.py::test_cli_sppf_graph_and_status_consistency::cli.py::gabion.cli.app
# gabion:behavior primary=desired
def test_cli_sppf_graph_and_status_consistency(tmp_path: Path) -> None:
    graph_json = tmp_path / "graph.json"
    status_json = tmp_path / "status.json"

    runner = CliRunner()
    graph_result = _invoke(
        runner,
        [
            "sppf-graph",
            "--root",
            str(REPO_ROOT),
            "--json-output",
            str(graph_json),
        ],
    )
    assert graph_result.exit_code == 0
    assert graph_json.exists()

    status_result = _invoke(
        runner,
        [
            "status-consistency",
            "--root",
            str(REPO_ROOT),
            "--json-output",
            str(status_json),
            "--no-fail-on-violations",
        ],
    )
    assert status_result.exit_code == 0
    assert status_json.exists()


# gabion:behavior primary=desired
def test_cli_commands_cases_no_longer_import_repo_root() -> None:
    module_text = (REPO_ROOT / "tests/gabion/cli/cli_commands_cases.py").read_text(
        encoding="utf-8"
    )
    assert "REPO_ROOT" not in module_text
