from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import sys

import pytest


def _has_pygls() -> bool:
    return importlib.util.find_spec("pygls") is not None


class _FakeProcess:
    def __init__(self, stdout_bytes: bytes) -> None:
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(stdout_bytes)
        self.stderr = io.BytesIO()
        self.returncode = 0

    def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
        return (b"", b"")


def _rpc_response(msg_id: int, payload: dict) -> bytes:
    message = {"jsonrpc": "2.0", "id": msg_id}
    message.update(payload)
    body = json.dumps(message).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
    return header + body


def _fake_process_factory(stdout_bytes: bytes):
    def _factory(*_args, **_kwargs):
        return _FakeProcess(stdout_bytes)

    return _factory


# gabion:evidence E:decision_surface/direct::lsp_client.py::gabion.lsp_client._read_response::request_id
@pytest.mark.skipif(not _has_pygls(), reason="pygls not installed")
def test_run_command_unknown_command_raises() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.lsp_client import CommandRequest, LspClientError, run_command

    stdout_bytes = b"".join(
        [
            _rpc_response(1, {"result": {}}),
            _rpc_response(
                2,
                {"error": {"code": -32601, "message": "Command not found"}},
            ),
            _rpc_response(3, {"result": {}}),
        ]
    )
    with pytest.raises(LspClientError):
        run_command(
            CommandRequest("gabion.unknown", []),
            root=repo_root,
            timeout_ticks=10_000,
            timeout_tick_ns=1_000_000,
            process_factory=_fake_process_factory(stdout_bytes),
        )
