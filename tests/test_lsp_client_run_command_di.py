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


def _extract_rpc_messages(buffer: bytes) -> list[dict]:
    messages: list[dict] = []
    offset = 0
    while True:
        header_end = buffer.find(b"\r\n\r\n", offset)
        if header_end < 0:
            break
        header = buffer[offset:header_end].decode("utf-8")
        length = None
        for line in header.split("\r\n"):
            if line.lower().startswith("content-length:"):
                length = int(line.split(":", 1)[1].strip())
                break
        if length is None:
            break
        body_start = header_end + 4
        body_end = body_start + length
        if body_end > len(buffer):
            break
        payload = json.loads(buffer[body_start:body_end].decode("utf-8"))
        messages.append(payload)
        offset = body_end
    return messages


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


# gabion:evidence E:function_site::lsp_client.py::gabion.lsp_client.run_command
def test_run_command_injects_analysis_timeout_seconds() -> None:
    proc = _make_proc(0, b"")

    def factory(*_args, **_kwargs):
        return proc

    previous = os.environ.get("GABION_LSP_TIMEOUT_SECONDS")
    os.environ["GABION_LSP_TIMEOUT_SECONDS"] = "1.0"
    try:
        run_command(
            CommandRequest("gabion.dataflowAudit", [{"paths": ["."]}]),
            root=Path("."),
            process_factory=factory,
        )
    finally:
        if previous is None:
            os.environ.pop("GABION_LSP_TIMEOUT_SECONDS", None)
        else:
            os.environ["GABION_LSP_TIMEOUT_SECONDS"] = previous
    messages = _extract_rpc_messages(proc.stdin.getvalue())
    execute = next(msg for msg in messages if msg.get("method") == "workspace/executeCommand")
    payload = execute["params"]["arguments"][0]
    assert 0 < payload.get("analysis_timeout_seconds", 0) <= 1.0


# gabion:evidence E:function_site::lsp_client.py::gabion.lsp_client.run_command
def test_run_command_preserves_lower_analysis_timeout_seconds() -> None:
    proc = _make_proc(0, b"")

    def factory(*_args, **_kwargs):
        return proc

    previous = os.environ.get("GABION_LSP_TIMEOUT_SECONDS")
    os.environ["GABION_LSP_TIMEOUT_SECONDS"] = "5.0"
    try:
        run_command(
            CommandRequest(
                "gabion.dataflowAudit",
                [{"analysis_timeout_seconds": 0.01, "paths": ["."]}],
            ),
            root=Path("."),
            process_factory=factory,
        )
    finally:
        if previous is None:
            os.environ.pop("GABION_LSP_TIMEOUT_SECONDS", None)
        else:
            os.environ["GABION_LSP_TIMEOUT_SECONDS"] = previous
    messages = _extract_rpc_messages(proc.stdin.getvalue())
    execute = next(msg for msg in messages if msg.get("method") == "workspace/executeCommand")
    payload = execute["params"]["arguments"][0]
    assert payload.get("analysis_timeout_seconds") == 0.01


# gabion:evidence E:function_site::lsp_client.py::gabion.lsp_client.run_command
def test_run_command_overrides_invalid_analysis_timeout_seconds() -> None:
    proc = _make_proc(0, b"")

    def factory(*_args, **_kwargs):
        return proc

    previous = os.environ.get("GABION_LSP_TIMEOUT_SECONDS")
    os.environ["GABION_LSP_TIMEOUT_SECONDS"] = "1.0"
    try:
        run_command(
            CommandRequest(
                "gabion.dataflowAudit",
                [{"analysis_timeout_seconds": "nope", "paths": ["."]}],
            ),
            root=Path("."),
            process_factory=factory,
        )
    finally:
        if previous is None:
            os.environ.pop("GABION_LSP_TIMEOUT_SECONDS", None)
        else:
            os.environ["GABION_LSP_TIMEOUT_SECONDS"] = previous
    messages = _extract_rpc_messages(proc.stdin.getvalue())
    execute = next(msg for msg in messages if msg.get("method") == "workspace/executeCommand")
    payload = execute["params"]["arguments"][0]
    assert 0 < payload.get("analysis_timeout_seconds", 0) <= 1.0
