from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


def _has_pygls() -> bool:
    return importlib.util.find_spec("pygls") is not None


@pytest.mark.skipif(not _has_pygls(), reason="pygls not installed")
def test_lsp_execute_command(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    sample = tmp_path / "sample.py"
    sample.write_text("def alpha(x):\n    return x\n")
    from gabion.lsp_client import CommandRequest, run_command

    result = run_command(
        CommandRequest(
            "gabion.dataflowAudit",
            [{"paths": [str(tmp_path)], "fail_on_violations": False}],
        ),
        root=tmp_path,
    )
    assert "exit_code" in result
    snapshot_result = run_command(
        CommandRequest(
            "gabion.dataflowAudit",
            [
                {
                    "paths": [str(tmp_path)],
                    "fail_on_violations": False,
                    "structure_tree": "-",
                    "structure_metrics": "-",
                }
            ],
        ),
        root=tmp_path,
    )
    assert "structure_tree" in snapshot_result
    assert "structure_metrics" in snapshot_result
    synth_result = run_command(
        CommandRequest(
            "gabion.synthesisPlan",
            [
                {
                    "bundles": [{"bundle": ["ctx"], "tier": 2}],
                    "min_bundle_size": 1,
                    "allow_singletons": True,
                    "existing_names": ["CtxBundle"],
                }
            ],
        ),
        root=repo_root,
    )
    assert "protocols" in synth_result


@pytest.mark.skipif(not _has_pygls(), reason="pygls not installed")
def test_lsp_execute_command_writes_structure_snapshot(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.lsp_client import CommandRequest, run_command

    sample = tmp_path / "sample.py"
    sample.write_text("def alpha(a, b):\n    return a + b\n")
    snapshot = tmp_path / "structure.json"
    result = run_command(
        CommandRequest(
            "gabion.dataflowAudit",
            [
                {
                    "paths": [str(sample)],
                    "fail_on_violations": False,
                    "structure_tree": str(snapshot),
                }
            ],
        ),
        root=tmp_path,
    )
    assert "exit_code" in result
    assert snapshot.exists()
    assert "\"format_version\"" in snapshot.read_text()
