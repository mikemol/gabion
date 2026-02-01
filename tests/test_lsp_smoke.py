from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


def _has_pygls() -> bool:
    return importlib.util.find_spec("pygls") is not None


@pytest.mark.skipif(not _has_pygls(), reason="pygls not installed")
def test_lsp_execute_command() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.lsp_client import run_command

    result = run_command(
        "gabion.dataflowAudit",
        [{"paths": [str(repo_root)], "fail_on_violations": False}],
        root=repo_root,
    )
    assert "exit_code" in result
    synth_result = run_command(
        "gabion.synthesisPlan",
        [
            {
                "bundles": [{"bundle": ["ctx"], "tier": 2}],
                "min_bundle_size": 1,
                "allow_singletons": True,
                "existing_names": ["CtxBundle"],
            }
        ],
        root=repo_root,
    )
    assert "protocols" in synth_result
