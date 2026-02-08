from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest

from gabion.lsp_client import CommandRequest, LspClientError, run_command


def _rpc_message(payload: dict) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
    return header + body


class _FakeProc:
    # dataflow-bundle: stderr_bytes, stdout_bytes
    def __init__(self, stdout_bytes: bytes, stderr_bytes: bytes, returncode: int | None) -> None:
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(stdout_bytes)
        self.stderr = io.BytesIO(stderr_bytes)
        self.returncode = returncode
        self.last_timeout: float | None = None

    def communicate(self, timeout: float | None = None):
        self.last_timeout = timeout
        return (b"", self.stderr.read())


def _make_proc(returncode: int | None, stderr_bytes: bytes) -> _FakeProc:
    init = _rpc_message({"jsonrpc": "2.0", "id": 1, "result": {}})
    cmd = _rpc_message({"jsonrpc": "2.0", "id": 2, "result": {}})
    shutdown = _rpc_message({"jsonrpc": "2.0", "id": 3, "result": {}})
    return _FakeProc(init + cmd + shutdown, stderr_bytes, returncode)


def _make_proc_with_cmd_result(returncode: int | None, stderr_bytes: bytes, cmd_result: object) -> _FakeProc:
    init = _rpc_message({"jsonrpc": "2.0", "id": 1, "result": {}})
    cmd = _rpc_message({"jsonrpc": "2.0", "id": 2, "result": cmd_result})
    shutdown = _rpc_message({"jsonrpc": "2.0", "id": 3, "result": {}})
    return _FakeProc(init + cmd + shutdown, stderr_bytes, returncode)


# gabion:evidence E:decision_surface/direct::lsp_client.py::gabion.lsp_client._read_response::request_id
def test_run_command_raises_on_nonzero_returncode() -> None:
    def factory(*_args, **_kwargs):
        return _make_proc(1, b"boom")

    with pytest.raises(LspClientError) as exc:
        run_command(CommandRequest("gabion.dataflowAudit", []), root=Path("."), process_factory=factory)
    assert "server failed" in str(exc.value).lower()


# gabion:evidence E:decision_surface/direct::lsp_client.py::gabion.lsp_client._read_response::request_id
def test_run_command_raises_on_stderr_output() -> None:
    def factory(*_args, **_kwargs):
        return _make_proc(0, b"warning")

    with pytest.raises(LspClientError) as exc:
        run_command(CommandRequest("gabion.dataflowAudit", []), root=Path("."), process_factory=factory)
    assert "error output" in str(exc.value).lower()


# gabion:evidence E:decision_surface/direct::lsp_client.py::gabion.lsp_client._read_response::request_id
def test_run_command_allows_blank_stderr() -> None:
    def factory(*_args, **_kwargs):
        return _make_proc(0, b"\n")

    result = run_command(CommandRequest("gabion.dataflowAudit", []), root=Path("."), process_factory=factory)
    assert result == {}


# gabion:evidence E:decision_surface/direct::lsp_client.py::gabion.lsp_client._read_response::request_id
def test_run_command_rejects_non_object_result() -> None:
    def factory(*_args, **_kwargs):
        return _make_proc_with_cmd_result(0, b"", [])

    with pytest.raises(LspClientError) as exc:
        run_command(CommandRequest("gabion.dataflowAudit", []), root=Path("."), process_factory=factory)
    assert "unexpected lsp result" in str(exc.value).lower()


# gabion:evidence E:function_site::lsp_client.py::gabion.lsp_client.run_command
def test_run_command_uses_env_timeout() -> None:
    proc = _make_proc(0, b"")

    def factory(*_args, **_kwargs):
        return proc

    previous = os.environ.get("GABION_LSP_TIMEOUT_SECONDS")
    os.environ["GABION_LSP_TIMEOUT_SECONDS"] = "1.0"
    try:
        result = run_command(CommandRequest("gabion.dataflowAudit", []), root=Path("."), process_factory=factory)
    finally:
        if previous is None:
            os.environ.pop("GABION_LSP_TIMEOUT_SECONDS", None)
        else:
            os.environ["GABION_LSP_TIMEOUT_SECONDS"] = previous
    assert result == {}
    assert proc.last_timeout is not None
    assert 0 < proc.last_timeout <= 1.0


# gabion:evidence E:function_site::lsp_client.py::gabion.lsp_client.run_command
def test_run_command_ignores_invalid_env_timeout() -> None:
    proc = _make_proc(0, b"")

    def factory(*_args, **_kwargs):
        return proc

    previous = os.environ.get("GABION_LSP_TIMEOUT_SECONDS")
    os.environ["GABION_LSP_TIMEOUT_SECONDS"] = "nope"
    try:
        result = run_command(CommandRequest("gabion.dataflowAudit", []), root=Path("."), process_factory=factory)
    finally:
        if previous is None:
            os.environ.pop("GABION_LSP_TIMEOUT_SECONDS", None)
        else:
            os.environ["GABION_LSP_TIMEOUT_SECONDS"] = previous
    assert result == {}
    assert proc.last_timeout is not None


# gabion:evidence E:function_site::lsp_client.py::gabion.lsp_client.run_command
def test_run_command_env_timeout_zero_ignored() -> None:
    proc = _make_proc(0, b"")

    def factory(*_args, **_kwargs):
        return proc

    previous = os.environ.get("GABION_LSP_TIMEOUT_SECONDS")
    os.environ["GABION_LSP_TIMEOUT_SECONDS"] = "0"
    try:
        result = run_command(CommandRequest("gabion.dataflowAudit", []), root=Path("."), process_factory=factory)
    finally:
        if previous is None:
            os.environ.pop("GABION_LSP_TIMEOUT_SECONDS", None)
        else:
            os.environ["GABION_LSP_TIMEOUT_SECONDS"] = previous
    assert result == {}
    assert proc.last_timeout is not None
