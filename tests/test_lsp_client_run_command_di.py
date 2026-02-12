from __future__ import annotations

import io
import json
import os
import subprocess
from pathlib import Path

import pytest

from gabion.lsp_client import (
    CommandRequest,
    LspClientError,
    run_command,
    _env_timeout_ticks,
    _analysis_timeout_slack_ns,
    _remaining_deadline_ns,
)
from gabion.exceptions import NeverThrown


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


class _TimeoutOnCommunicateProc(_FakeProc):
    def __init__(self, stdout_bytes: bytes, stderr_bytes: bytes, returncode: int | None) -> None:
        super().__init__(stdout_bytes, stderr_bytes, returncode)
        self._communicate_calls = 0
        self.killed = False

    def communicate(self, timeout: float | None = None):
        self._communicate_calls += 1
        if self._communicate_calls == 1:
            raise subprocess.TimeoutExpired(cmd=["python", "-m", "gabion.server"], timeout=timeout or 0.0)
        return super().communicate(timeout=timeout)

    def kill(self) -> None:
        self.killed = True


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


def _set_env(values: dict[str, str | None]) -> dict[str, str | None]:
    previous = {key: os.environ.get(key) for key in values}
    for key, value in values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    return previous


def _restore_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


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

    previous_ticks = os.environ.get("GABION_LSP_TIMEOUT_TICKS")
    previous_tick_ns = os.environ.get("GABION_LSP_TIMEOUT_TICK_NS")
    os.environ["GABION_LSP_TIMEOUT_TICKS"] = "1000"
    os.environ["GABION_LSP_TIMEOUT_TICK_NS"] = "1000000"
    try:
        result = run_command(CommandRequest("gabion.dataflowAudit", []), root=Path("."), process_factory=factory)
    finally:
        if previous_ticks is None:
            os.environ.pop("GABION_LSP_TIMEOUT_TICKS", None)
        else:
            os.environ["GABION_LSP_TIMEOUT_TICKS"] = previous_ticks
        if previous_tick_ns is None:
            os.environ.pop("GABION_LSP_TIMEOUT_TICK_NS", None)
        else:
            os.environ["GABION_LSP_TIMEOUT_TICK_NS"] = previous_tick_ns
    assert result == {}
    assert proc.last_timeout is not None
    assert 2.0 < proc.last_timeout <= 3.0


# gabion:evidence E:function_site::lsp_client.py::gabion.lsp_client.run_command
def test_run_command_rejects_invalid_env_timeout() -> None:
    proc = _make_proc(0, b"")

    def factory(*_args, **_kwargs):
        return proc

    previous = os.environ.get("GABION_LSP_TIMEOUT_TICKS")
    os.environ["GABION_LSP_TIMEOUT_TICKS"] = "nope"
    try:
        with pytest.raises(NeverThrown):
            run_command(CommandRequest("gabion.dataflowAudit", []), root=Path("."), process_factory=factory)
    finally:
        if previous is None:
            os.environ.pop("GABION_LSP_TIMEOUT_TICKS", None)
        else:
            os.environ["GABION_LSP_TIMEOUT_TICKS"] = previous


def test_analysis_timeout_slack_floor() -> None:
    assert _analysis_timeout_slack_ns(10_000_000) == 2_000_000_000


def test_analysis_timeout_slack_cap() -> None:
    assert _analysis_timeout_slack_ns(1_000_000_000_000) == 120_000_000_000


# gabion:evidence E:function_site::lsp_client.py::gabion.lsp_client.run_command
def test_run_command_env_timeout_zero_rejected() -> None:
    proc = _make_proc(0, b"")

    def factory(*_args, **_kwargs):
        return proc

    previous = os.environ.get("GABION_LSP_TIMEOUT_TICKS")
    os.environ["GABION_LSP_TIMEOUT_TICKS"] = "0"
    try:
        with pytest.raises(NeverThrown):
            run_command(CommandRequest("gabion.dataflowAudit", []), root=Path("."), process_factory=factory)
    finally:
        if previous is None:
            os.environ.pop("GABION_LSP_TIMEOUT_TICKS", None)
        else:
            os.environ["GABION_LSP_TIMEOUT_TICKS"] = previous


# gabion:evidence E:function_site::lsp_client.py::gabion.lsp_client.run_command
def test_run_command_injects_analysis_timeout_ticks() -> None:
    proc = _make_proc(0, b"")

    def factory(*_args, **_kwargs):
        return proc

    previous_ticks = os.environ.get("GABION_LSP_TIMEOUT_TICKS")
    previous_tick_ns = os.environ.get("GABION_LSP_TIMEOUT_TICK_NS")
    os.environ["GABION_LSP_TIMEOUT_TICKS"] = "1000"
    os.environ["GABION_LSP_TIMEOUT_TICK_NS"] = "1000000"
    try:
        run_command(
            CommandRequest("gabion.dataflowAudit", [{"paths": ["."]}]),
            root=Path("."),
            process_factory=factory,
        )
    finally:
        if previous_ticks is None:
            os.environ.pop("GABION_LSP_TIMEOUT_TICKS", None)
        else:
            os.environ["GABION_LSP_TIMEOUT_TICKS"] = previous_ticks
        if previous_tick_ns is None:
            os.environ.pop("GABION_LSP_TIMEOUT_TICK_NS", None)
        else:
            os.environ["GABION_LSP_TIMEOUT_TICK_NS"] = previous_tick_ns
    messages = _extract_rpc_messages(proc.stdin.getvalue())
    execute = next(msg for msg in messages if msg.get("method") == "workspace/executeCommand")
    payload = execute["params"]["arguments"][0]
    assert payload.get("analysis_timeout_tick_ns") == 1000000
    assert 0 < payload.get("analysis_timeout_ticks", 0) <= 1000


# gabion:evidence E:function_site::lsp_client.py::gabion.lsp_client.run_command
def test_run_command_preserves_lower_analysis_timeout_ticks() -> None:
    proc = _make_proc(0, b"")

    def factory(*_args, **_kwargs):
        return proc

    previous_ticks = os.environ.get("GABION_LSP_TIMEOUT_TICKS")
    previous_tick_ns = os.environ.get("GABION_LSP_TIMEOUT_TICK_NS")
    os.environ["GABION_LSP_TIMEOUT_TICKS"] = "5000"
    os.environ["GABION_LSP_TIMEOUT_TICK_NS"] = "1000000"
    try:
        run_command(
            CommandRequest(
                "gabion.dataflowAudit",
                [{"analysis_timeout_ticks": 1, "analysis_timeout_tick_ns": 1000000, "paths": ["."]}],
            ),
            root=Path("."),
            process_factory=factory,
        )
    finally:
        if previous_ticks is None:
            os.environ.pop("GABION_LSP_TIMEOUT_TICKS", None)
        else:
            os.environ["GABION_LSP_TIMEOUT_TICKS"] = previous_ticks
        if previous_tick_ns is None:
            os.environ.pop("GABION_LSP_TIMEOUT_TICK_NS", None)
        else:
            os.environ["GABION_LSP_TIMEOUT_TICK_NS"] = previous_tick_ns
    messages = _extract_rpc_messages(proc.stdin.getvalue())
    execute = next(msg for msg in messages if msg.get("method") == "workspace/executeCommand")
    payload = execute["params"]["arguments"][0]
    assert payload.get("analysis_timeout_ticks") == 1
    assert payload.get("analysis_timeout_tick_ns") == 1000000


# gabion:evidence E:function_site::lsp_client.py::gabion.lsp_client.run_command
def test_run_command_overrides_invalid_analysis_timeout_ticks() -> None:
    proc = _make_proc(0, b"")

    def factory(*_args, **_kwargs):
        return proc

    previous_ticks = os.environ.get("GABION_LSP_TIMEOUT_TICKS")
    previous_tick_ns = os.environ.get("GABION_LSP_TIMEOUT_TICK_NS")
    os.environ["GABION_LSP_TIMEOUT_TICKS"] = "1000"
    os.environ["GABION_LSP_TIMEOUT_TICK_NS"] = "1000000"
    try:
        with pytest.raises(NeverThrown):
            run_command(
                CommandRequest(
                    "gabion.dataflowAudit",
                    [{"analysis_timeout_ticks": "nope", "analysis_timeout_tick_ns": "bad", "paths": ["."]}],
                ),
                root=Path("."),
                process_factory=factory,
            )
    finally:
        if previous_ticks is None:
            os.environ.pop("GABION_LSP_TIMEOUT_TICKS", None)
        else:
            os.environ["GABION_LSP_TIMEOUT_TICKS"] = previous_ticks
        if previous_tick_ns is None:
            os.environ.pop("GABION_LSP_TIMEOUT_TICK_NS", None)
        else:
            os.environ["GABION_LSP_TIMEOUT_TICK_NS"] = previous_tick_ns


def test_run_command_rejects_missing_analysis_timeout_tick_ns() -> None:
    proc = _make_proc(0, b"")

    def factory(*_args, **_kwargs):
        return proc

    previous = _set_env(
        {
            "GABION_LSP_TIMEOUT_TICKS": None,
            "GABION_LSP_TIMEOUT_TICK_NS": None,
            "GABION_LSP_TIMEOUT_MS": None,
            "GABION_LSP_TIMEOUT_SECONDS": None,
        }
    )
    try:
        with pytest.raises(NeverThrown):
            run_command(
                CommandRequest(
                    "gabion.dataflowAudit",
                    [{"analysis_timeout_ticks": 1, "paths": ["."]}],
                ),
                root=Path("."),
                process_factory=factory,
            )
    finally:
        _restore_env(previous)


def test_run_command_rejects_zero_analysis_timeout_tick_ns() -> None:
    proc = _make_proc(0, b"")

    def factory(*_args, **_kwargs):
        return proc

    previous = _set_env(
        {
            "GABION_LSP_TIMEOUT_TICKS": None,
            "GABION_LSP_TIMEOUT_TICK_NS": None,
            "GABION_LSP_TIMEOUT_MS": None,
            "GABION_LSP_TIMEOUT_SECONDS": None,
        }
    )
    try:
        with pytest.raises(NeverThrown):
            run_command(
                CommandRequest(
                    "gabion.dataflowAudit",
                    [{"analysis_timeout_ticks": 1, "analysis_timeout_tick_ns": 0, "paths": ["."]}],
                ),
                root=Path("."),
                process_factory=factory,
            )
    finally:
        _restore_env(previous)


def test_run_command_preserves_lower_analysis_timeout_ms() -> None:
    proc = _make_proc(0, b"")

    def factory(*_args, **_kwargs):
        return proc

    previous = _set_env(
        {
            "GABION_LSP_TIMEOUT_TICKS": "5000",
            "GABION_LSP_TIMEOUT_TICK_NS": "1000000",
        }
    )
    try:
        run_command(
            CommandRequest(
                "gabion.dataflowAudit",
                [{"analysis_timeout_ms": 1, "paths": ["."]}],
            ),
            root=Path("."),
            process_factory=factory,
        )
    finally:
        _restore_env(previous)
    messages = _extract_rpc_messages(proc.stdin.getvalue())
    execute = next(msg for msg in messages if msg.get("method") == "workspace/executeCommand")
    payload = execute["params"]["arguments"][0]
    assert payload.get("analysis_timeout_ms") == 1
    assert "analysis_timeout_ticks" not in payload


def test_run_command_preserves_lower_analysis_timeout_seconds() -> None:
    proc = _make_proc(0, b"")

    def factory(*_args, **_kwargs):
        return proc

    previous = _set_env(
        {
            "GABION_LSP_TIMEOUT_TICKS": "5000",
            "GABION_LSP_TIMEOUT_TICK_NS": "1000000",
        }
    )
    try:
        run_command(
            CommandRequest(
                "gabion.dataflowAudit",
                [{"analysis_timeout_seconds": "0.001", "paths": ["."]}],
            ),
            root=Path("."),
            process_factory=factory,
        )
    finally:
        _restore_env(previous)
    messages = _extract_rpc_messages(proc.stdin.getvalue())
    execute = next(msg for msg in messages if msg.get("method") == "workspace/executeCommand")
    payload = execute["params"]["arguments"][0]
    assert payload.get("analysis_timeout_seconds") == "0.001"
    assert "analysis_timeout_ticks" not in payload


def test_run_command_overrides_invalid_analysis_timeout_seconds() -> None:
    proc = _make_proc(0, b"")

    def factory(*_args, **_kwargs):
        return proc

    previous = _set_env(
        {
            "GABION_LSP_TIMEOUT_TICKS": "1000",
            "GABION_LSP_TIMEOUT_TICK_NS": "1000000",
        }
    )
    try:
        with pytest.raises(NeverThrown):
            run_command(
                CommandRequest(
                    "gabion.dataflowAudit",
                    [{"analysis_timeout_seconds": "nope", "paths": ["."]}],
                ),
                root=Path("."),
                process_factory=factory,
            )
    finally:
        _restore_env(previous)


def test_run_command_rejects_zero_analysis_timeout_seconds() -> None:
    proc = _make_proc(0, b"")

    def factory(*_args, **_kwargs):
        return proc

    previous = _set_env(
        {
            "GABION_LSP_TIMEOUT_TICKS": "1000",
            "GABION_LSP_TIMEOUT_TICK_NS": "1000000",
        }
    )
    try:
        with pytest.raises(NeverThrown):
            run_command(
                CommandRequest(
                    "gabion.dataflowAudit",
                    [{"analysis_timeout_seconds": "0", "paths": ["."]}],
                ),
                root=Path("."),
                process_factory=factory,
            )
    finally:
        _restore_env(previous)


def test_run_command_overrides_invalid_analysis_timeout_ms() -> None:
    proc = _make_proc(0, b"")

    def factory(*_args, **_kwargs):
        return proc

    previous = _set_env(
        {
            "GABION_LSP_TIMEOUT_TICKS": "1000",
            "GABION_LSP_TIMEOUT_TICK_NS": "1000000",
        }
    )
    try:
        with pytest.raises(NeverThrown):
            run_command(
                CommandRequest(
                    "gabion.dataflowAudit",
                    [{"analysis_timeout_ms": "nope", "paths": ["."]}],
                ),
                root=Path("."),
                process_factory=factory,
            )
    finally:
        _restore_env(previous)


def test_run_command_rejects_zero_analysis_timeout_ms() -> None:
    proc = _make_proc(0, b"")

    def factory(*_args, **_kwargs):
        return proc

    previous = _set_env(
        {
            "GABION_LSP_TIMEOUT_TICKS": "1000",
            "GABION_LSP_TIMEOUT_TICK_NS": "1000000",
        }
    )
    try:
        with pytest.raises(NeverThrown):
            run_command(
                CommandRequest(
                    "gabion.dataflowAudit",
                    [{"analysis_timeout_ms": 0, "paths": ["."]}],
                ),
                root=Path("."),
                process_factory=factory,
            )
    finally:
        _restore_env(previous)


def test_run_command_tick_ns_clamps_to_remaining() -> None:
    proc = _make_proc(0, b"")

    def factory(*_args, **_kwargs):
        return proc

    run_command(
        CommandRequest("gabion.dataflowAudit", [{"paths": ["."]}]),
        root=Path("."),
        timeout_ticks=1,
        timeout_tick_ns=1_000_000_000,
        process_factory=factory,
    )
    messages = _extract_rpc_messages(proc.stdin.getvalue())
    execute = next(msg for msg in messages if msg.get("method") == "workspace/executeCommand")
    payload = execute["params"]["arguments"][0]
    assert payload.get("analysis_timeout_ticks") == 1
    assert 0 < payload.get("analysis_timeout_tick_ns", 0) <= 1_000_000_000


def test_env_timeout_ticks_with_tick_ns() -> None:
    previous = _set_env(
        {
            "GABION_LSP_TIMEOUT_TICKS": "5",
            "GABION_LSP_TIMEOUT_TICK_NS": "2000000",
            "GABION_LSP_TIMEOUT_MS": None,
            "GABION_LSP_TIMEOUT_SECONDS": None,
        }
    )
    try:
        assert _env_timeout_ticks() == (5, 2000000)
    finally:
        _restore_env(previous)


def test_env_timeout_ms_parsing() -> None:
    previous = _set_env(
        {
            "GABION_LSP_TIMEOUT_TICKS": None,
            "GABION_LSP_TIMEOUT_TICK_NS": None,
            "GABION_LSP_TIMEOUT_MS": "15",
            "GABION_LSP_TIMEOUT_SECONDS": None,
        }
    )
    try:
        assert _env_timeout_ticks() == (15, 1_000_000)
    finally:
        _restore_env(previous)


def test_env_timeout_tick_ns_invalid_falls_back_to_ms() -> None:
    previous = _set_env(
        {
            "GABION_LSP_TIMEOUT_TICKS": "5",
            "GABION_LSP_TIMEOUT_TICK_NS": "nope",
            "GABION_LSP_TIMEOUT_MS": "10",
            "GABION_LSP_TIMEOUT_SECONDS": None,
        }
    )
    try:
        with pytest.raises(NeverThrown):
            _env_timeout_ticks()
    finally:
        _restore_env(previous)


def test_env_timeout_tick_ns_missing_raises() -> None:
    previous = _set_env(
        {
            "GABION_LSP_TIMEOUT_TICKS": "5",
            "GABION_LSP_TIMEOUT_TICK_NS": None,
            "GABION_LSP_TIMEOUT_MS": None,
            "GABION_LSP_TIMEOUT_SECONDS": None,
        }
    )
    try:
        with pytest.raises(NeverThrown):
            _env_timeout_ticks()
    finally:
        _restore_env(previous)


def test_env_timeout_ms_invalid_returns_none() -> None:
    previous = _set_env(
        {
            "GABION_LSP_TIMEOUT_TICKS": None,
            "GABION_LSP_TIMEOUT_TICK_NS": None,
            "GABION_LSP_TIMEOUT_MS": "nope",
            "GABION_LSP_TIMEOUT_SECONDS": None,
        }
    )
    try:
        with pytest.raises(NeverThrown):
            _env_timeout_ticks()
    finally:
        _restore_env(previous)


def test_env_timeout_seconds_parsing() -> None:
    previous = _set_env(
        {
            "GABION_LSP_TIMEOUT_TICKS": None,
            "GABION_LSP_TIMEOUT_TICK_NS": None,
            "GABION_LSP_TIMEOUT_MS": None,
            "GABION_LSP_TIMEOUT_SECONDS": "0.25",
        }
    )
    try:
        assert _env_timeout_ticks() == (250, 1000000)
    finally:
        _restore_env(previous)


def test_env_timeout_seconds_invalid_returns_none() -> None:
    previous = _set_env(
        {
            "GABION_LSP_TIMEOUT_TICKS": None,
            "GABION_LSP_TIMEOUT_TICK_NS": None,
            "GABION_LSP_TIMEOUT_MS": None,
            "GABION_LSP_TIMEOUT_SECONDS": "nope",
        }
    )
    try:
        with pytest.raises(NeverThrown):
            _env_timeout_ticks()
    finally:
        _restore_env(previous)


def test_env_timeout_missing_raises() -> None:
    previous = _set_env(
        {
            "GABION_LSP_TIMEOUT_TICKS": None,
            "GABION_LSP_TIMEOUT_TICK_NS": None,
            "GABION_LSP_TIMEOUT_MS": None,
            "GABION_LSP_TIMEOUT_SECONDS": None,
        }
    )
    try:
        with pytest.raises(NeverThrown):
            _env_timeout_ticks()
    finally:
        _restore_env(previous)


def test_run_command_clamps_zero_timeout_ticks() -> None:
    proc = _make_proc(0, b"")

    def factory(*_args, **_kwargs):
        return proc

    run_command(
        CommandRequest("gabion.dataflowAudit", [{"paths": ["."]}]),
        root=Path("."),
        timeout_ticks=0,
        timeout_tick_ns=1_000_000_000,
        process_factory=factory,
    )
    messages = _extract_rpc_messages(proc.stdin.getvalue())
    execute = next(msg for msg in messages if msg.get("method") == "workspace/executeCommand")
    payload = execute["params"]["arguments"][0]
    assert payload.get("analysis_timeout_ticks", 0) > 0


def test_run_command_clamps_zero_tick_ns() -> None:
    proc = _make_proc(0, b"")

    def factory(*_args, **_kwargs):
        return proc

    run_command(
        CommandRequest("gabion.dataflowAudit", [{"paths": ["."]}]),
        root=Path("."),
        timeout_ticks=1_000_000_000,
        timeout_tick_ns=0,
        process_factory=factory,
    )
    messages = _extract_rpc_messages(proc.stdin.getvalue())
    execute = next(msg for msg in messages if msg.get("method") == "workspace/executeCommand")
    payload = execute["params"]["arguments"][0]
    assert payload.get("analysis_timeout_ticks", 0) > 0


def test_remaining_deadline_ns_raises_when_expired() -> None:
    with pytest.raises(NeverThrown):
        _remaining_deadline_ns(0)


def test_run_command_uses_unbuffered_stdio() -> None:
    proc = _make_proc(0, b"")
    captured: dict[str, object] = {}

    def factory(*_args, **kwargs):
        captured.update(kwargs)
        return proc

    result = run_command(
        CommandRequest("gabion.dataflowAudit", [{"paths": ["."]}]),
        root=Path("."),
        process_factory=factory,
    )
    assert result == {}
    assert captured.get("bufsize") == 0


def test_run_command_handles_shutdown_timeout() -> None:
    init = _rpc_message({"jsonrpc": "2.0", "id": 1, "result": {}})
    cmd = _rpc_message({"jsonrpc": "2.0", "id": 2, "result": {}})
    shutdown = _rpc_message({"jsonrpc": "2.0", "id": 3, "result": {}})
    proc = _TimeoutOnCommunicateProc(init + cmd + shutdown, b"", 0)

    def factory(*_args, **_kwargs):
        return proc

    result = run_command(
        CommandRequest("gabion.dataflowAudit", [{"paths": ["."]}]),
        root=Path("."),
        process_factory=factory,
    )
    assert result == {}
    assert proc.killed is True
