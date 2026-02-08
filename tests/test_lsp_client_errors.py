from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


def _has_pygls() -> bool:
    return importlib.util.find_spec("pygls") is not None


# gabion:evidence E:decision_surface/direct::lsp_client.py::gabion.lsp_client._read_response::request_id
@pytest.mark.skipif(not _has_pygls(), reason="pygls not installed")
def test_run_command_unknown_command_raises() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.lsp_client import CommandRequest, LspClientError, run_command

    with pytest.raises(LspClientError):
        run_command(CommandRequest("gabion.unknown", []), root=repo_root)
