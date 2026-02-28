from __future__ import annotations

import io
import json
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.env_helpers import restore_env as _restore_env
from tests.env_helpers import set_env as _set_env
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
    def __init__(self, streams: "_FakeProcStreams", returncode: int | None) -> None:
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(streams.stdout_bytes)
        self._stderr_bytes = streams.stderr_bytes
        self.returncode = returncode
        self.last_timeout: float | None = None

    def communicate(self, timeout: float | None = None):
        self.last_timeout = timeout
        return (b"", self._stderr_bytes)


@dataclass(frozen=True)
class _FakeProcStreams:
    stdout_bytes: bytes
    stderr_bytes: bytes


class _TimeoutOnCommunicateProc(_FakeProc):
    def __init__(self, streams: _FakeProcStreams, returncode: int | None) -> None:
        super().__init__(streams, returncode)
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
    return _FakeProc(_FakeProcStreams(stdout_bytes=init + cmd + shutdown, stderr_bytes=stderr_bytes), returncode)


def _make_proc_with_cmd_result(returncode: int | None, stderr_bytes: bytes, cmd_result: object) -> _FakeProc:
    init = _rpc_message({"jsonrpc": "2.0", "id": 1, "result": {}})
    cmd = _rpc_message({"jsonrpc": "2.0", "id": 2, "result": cmd_result})
    shutdown = _rpc_message({"jsonrpc": "2.0", "id": 3, "result": {}})
    return _FakeProc(_FakeProcStreams(stdout_bytes=init + cmd + shutdown, stderr_bytes=stderr_bytes), returncode)


# gabion:evidence E:decision_surface/direct::lsp_client.py::gabion.lsp_client._read_response::request_id E:decision_surface/direct::lsp_client.py::gabion.lsp_client._read_response::stale_3eb94cc82057
def test_run_command_raises_on_nonzero_returncode() -> None:
    def factory(*_args, **_kwargs):
        return _make_proc(1, b"boom")

    with pytest.raises(LspClientError) as exc:
        run_command(CommandRequest("gabion.dataflowAudit", [{}]), root=Path("."), process_factory=factory)
    assert "server failed" in str(exc.value).lower()


# gabion:evidence E:call_footprint::tests/test_lsp_client_run_command_di.py::test_run_command_rejects_missing_payload_arguments::lsp_client.py::gabion.lsp_client.run_command::test_lsp_client_run_command_di.py::tests.test_lsp_client_run_command_di._make_proc
def test_run_command_rejects_missing_payload_arguments() -> None:
    created = False

    def factory(*_args, **_kwargs):
        nonlocal created
        created = True
        return _make_proc(0, b"")

    with pytest.raises(NeverThrown):
        run_command(
            CommandRequest("gabion.dataflowAudit"),
            root=Path("."),
            process_factory=factory,
        )
    assert created is False


# gabion:evidence E:decision_surface/direct::lsp_client.py::gabion.lsp_client._read_response::request_id E:decision_surface/direct::lsp_client.py::gabion.lsp_client._read_response::stale_6763a4f41f9c
def test_run_command_raises_on_stderr_output() -> None:
    def factory(*_args, **_kwargs):
        return _make_proc(0, b"warning")

    with pytest.raises(LspClientError) as exc:
        run_command(CommandRequest("gabion.dataflowAudit", [{}]), root=Path("."), process_factory=factory)
    assert "error output" in str(exc.value).lower()


# gabion:evidence E:decision_surface/direct::lsp_client.py::gabion.lsp_client._read_response::request_id E:decision_surface/direct::lsp_client.py::gabion.lsp_client._read_response::stale_299037338a45
def test_run_command_allows_blank_stderr() -> None:
    def factory(*_args, **_kwargs):
        return _make_proc(0, b"\n")

    result = run_command(CommandRequest("gabion.dataflowAudit", [{}]), root=Path("."), process_factory=factory)
    assert result == {}


# gabion:evidence E:decision_surface/direct::lsp_client.py::gabion.lsp_client._read_response::request_id E:decision_surface/direct::lsp_client.py::gabion.lsp_client._read_response::stale_8a892f76e117
def test_run_command_rejects_non_object_result() -> None:
    def factory(*_args, **_kwargs):
        return _make_proc_with_cmd_result(0, b"", [])

    with pytest.raises(LspClientError) as exc:
        run_command(CommandRequest("gabion.dataflowAudit", [{}]), root=Path("."), process_factory=factory)
    assert "unexpected lsp result" in str(exc.value).lower()


# gabion:evidence E:function_site::lsp_client.py::gabion.lsp_client.run_command E:decision_surface/direct::lsp_client.py::gabion.lsp_client.run_command::stale_9c0e201410b1
def test_run_command_uses_env_timeout() -> None:
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = run_command(
                CommandRequest("gabion.dataflowAudit", [{}]),
                root=Path("."),
                process_factory=factory,
                remaining_deadline_ns_fn=lambda _deadline_ns: 3_000_000_000,
            )
    finally:
        _restore_env(previous)
    assert result == {}
    assert proc.last_timeout is not None
    assert proc.last_timeout == 3.0


# gabion:evidence E:function_site::lsp_client.py::gabion.lsp_client.run_command E:decision_surface/direct::lsp_client.py::gabion.lsp_client.run_command::stale_b8ecbad5ac3e
def test_run_command_rejects_invalid_env_timeout() -> None:
    proc = _make_proc(0, b"")

    def factory(*_args, **_kwargs):
        return proc

    previous = _set_env({"GABION_LSP_TIMEOUT_TICKS": "nope"})
    try:
        with pytest.raises(NeverThrown):
            run_command(CommandRequest("gabion.dataflowAudit", [{}]), root=Path("."), process_factory=factory)
    finally:
        _restore_env(previous)


# gabion:evidence E:call_footprint::tests/test_lsp_client_run_command_di.py::test_analysis_timeout_slack_floor::lsp_client.py::gabion.lsp_client._analysis_timeout_slack_ns
def test_analysis_timeout_slack_floor() -> None:
    assert _analysis_timeout_slack_ns(10_000_000) == 2_000_000_000


# gabion:evidence E:call_footprint::tests/test_lsp_client_run_command_di.py::test_analysis_timeout_slack_cap::lsp_client.py::gabion.lsp_client._analysis_timeout_slack_ns
def test_analysis_timeout_slack_cap() -> None:
    assert _analysis_timeout_slack_ns(1_000_000_000_000) == 120_000_000_000


# gabion:evidence E:function_site::lsp_client.py::gabion.lsp_client.run_command E:decision_surface/direct::lsp_client.py::gabion.lsp_client.run_command::stale_e3e791608ab3_79cfac5a
def test_run_command_env_timeout_zero_rejected() -> None:
    proc = _make_proc(0, b"")

    def factory(*_args, **_kwargs):
        return proc

    previous = _set_env({"GABION_LSP_TIMEOUT_TICKS": "0"})
    try:
        with pytest.raises(NeverThrown):
            run_command(CommandRequest("gabion.dataflowAudit", [{}]), root=Path("."), process_factory=factory)
    finally:
        _restore_env(previous)


# gabion:evidence E:function_site::lsp_client.py::gabion.lsp_client.run_command E:decision_surface/direct::lsp_client.py::gabion.lsp_client.run_command::stale_ac0db42135fa
def test_run_command_injects_analysis_timeout_ticks() -> None:
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
        run_command(
            CommandRequest("gabion.dataflowAudit", [{"paths": ["."]}]),
            root=Path("."),
            process_factory=factory,
            remaining_deadline_ns_fn=lambda _deadline_ns: 1_000_000_000,
        )
    finally:
        _restore_env(previous)
    messages = _extract_rpc_messages(proc.stdin.getvalue())
    execute = next(msg for msg in messages if msg.get("method") == "workspace/executeCommand")
    payload = execute["params"]["arguments"][0]
    assert payload.get("analysis_timeout_tick_ns") == 1000000
    assert payload.get("analysis_timeout_ticks") == 1000


# gabion:evidence E:function_site::lsp_client.py::gabion.lsp_client.run_command E:decision_surface/direct::lsp_client.py::gabion.lsp_client.run_command::stale_2fae7f37df36
def test_run_command_preserves_lower_analysis_timeout_ticks() -> None:
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
                [{"analysis_timeout_ticks": 1, "analysis_timeout_tick_ns": 1000000, "paths": ["."]}],
            ),
            root=Path("."),
            process_factory=factory,
        )
    finally:
        _restore_env(previous)
    messages = _extract_rpc_messages(proc.stdin.getvalue())
    execute = next(msg for msg in messages if msg.get("method") == "workspace/executeCommand")
    payload = execute["params"]["arguments"][0]
    assert payload.get("analysis_timeout_ticks") == 1
    assert payload.get("analysis_timeout_tick_ns") == 1000000


# gabion:evidence E:function_site::lsp_client.py::gabion.lsp_client.run_command E:decision_surface/direct::lsp_client.py::gabion.lsp_client.run_command::stale_9225503d98a8
def test_run_command_overrides_invalid_analysis_timeout_ticks() -> None:
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
                    [{"analysis_timeout_ticks": "nope", "analysis_timeout_tick_ns": "bad", "paths": ["."]}],
                ),
                root=Path("."),
                process_factory=factory,
            )
    finally:
        _restore_env(previous)


# gabion:evidence E:call_footprint::tests/test_lsp_client_run_command_di.py::test_run_command_rejects_missing_analysis_timeout_tick_ns::env_helpers.py::tests.env_helpers.restore_env::env_helpers.py::tests.env_helpers.set_env::lsp_client.py::gabion.lsp_client.run_command::test_lsp_client_run_command_di.py::tests.test_lsp_client_run_command_di._make_proc
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


# gabion:evidence E:call_footprint::tests/test_lsp_client_run_command_di.py::test_run_command_rejects_zero_analysis_timeout_tick_ns::env_helpers.py::tests.env_helpers.restore_env::env_helpers.py::tests.env_helpers.set_env::lsp_client.py::gabion.lsp_client.run_command::test_lsp_client_run_command_di.py::tests.test_lsp_client_run_command_di._make_proc
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


# gabion:evidence E:call_footprint::tests/test_lsp_client_run_command_di.py::test_run_command_preserves_lower_analysis_timeout_ms::env_helpers.py::tests.env_helpers.restore_env::env_helpers.py::tests.env_helpers.set_env::lsp_client.py::gabion.lsp_client.run_command::test_lsp_client_run_command_di.py::tests.test_lsp_client_run_command_di._extract_rpc_messages::test_lsp_client_run_command_di.py::tests.test_lsp_client_run_command_di._make_proc
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


# gabion:evidence E:call_footprint::tests/test_lsp_client_run_command_di.py::test_run_command_preserves_lower_analysis_timeout_seconds::env_helpers.py::tests.env_helpers.restore_env::env_helpers.py::tests.env_helpers.set_env::lsp_client.py::gabion.lsp_client.run_command::test_lsp_client_run_command_di.py::tests.test_lsp_client_run_command_di._extract_rpc_messages::test_lsp_client_run_command_di.py::tests.test_lsp_client_run_command_di._make_proc
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


# gabion:evidence E:call_footprint::tests/test_lsp_client_run_command_di.py::test_run_command_overrides_invalid_analysis_timeout_seconds::env_helpers.py::tests.env_helpers.restore_env::env_helpers.py::tests.env_helpers.set_env::lsp_client.py::gabion.lsp_client.run_command::test_lsp_client_run_command_di.py::tests.test_lsp_client_run_command_di._make_proc
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


# gabion:evidence E:call_footprint::tests/test_lsp_client_run_command_di.py::test_run_command_rejects_zero_analysis_timeout_seconds::env_helpers.py::tests.env_helpers.restore_env::env_helpers.py::tests.env_helpers.set_env::lsp_client.py::gabion.lsp_client.run_command::test_lsp_client_run_command_di.py::tests.test_lsp_client_run_command_di._make_proc
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


# gabion:evidence E:call_footprint::tests/test_lsp_client_run_command_di.py::test_run_command_overrides_invalid_analysis_timeout_ms::env_helpers.py::tests.env_helpers.restore_env::env_helpers.py::tests.env_helpers.set_env::lsp_client.py::gabion.lsp_client.run_command::test_lsp_client_run_command_di.py::tests.test_lsp_client_run_command_di._make_proc
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


# gabion:evidence E:call_footprint::tests/test_lsp_client_run_command_di.py::test_run_command_rejects_zero_analysis_timeout_ms::env_helpers.py::tests.env_helpers.restore_env::env_helpers.py::tests.env_helpers.set_env::lsp_client.py::gabion.lsp_client.run_command::test_lsp_client_run_command_di.py::tests.test_lsp_client_run_command_di._make_proc
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


# gabion:evidence E:call_footprint::tests/test_lsp_client_run_command_di.py::test_run_command_tick_ns_clamps_to_remaining::lsp_client.py::gabion.lsp_client.run_command::test_lsp_client_run_command_di.py::tests.test_lsp_client_run_command_di._extract_rpc_messages::test_lsp_client_run_command_di.py::tests.test_lsp_client_run_command_di._make_proc
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
        remaining_deadline_ns_fn=lambda _deadline_ns: 123_456_789,
    )
    messages = _extract_rpc_messages(proc.stdin.getvalue())
    execute = next(msg for msg in messages if msg.get("method") == "workspace/executeCommand")
    payload = execute["params"]["arguments"][0]
    assert payload.get("analysis_timeout_ticks") == 1
    assert payload.get("analysis_timeout_tick_ns") == 123_456_789


# gabion:evidence E:call_footprint::tests/test_lsp_client_run_command_di.py::test_env_timeout_ticks_with_tick_ns::env_helpers.py::tests.env_helpers.restore_env::env_helpers.py::tests.env_helpers.set_env::lsp_client.py::gabion.lsp_client._env_timeout_ticks
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


# gabion:evidence E:call_footprint::tests/test_lsp_client_run_command_di.py::test_env_timeout_ms_parsing::env_helpers.py::tests.env_helpers.restore_env::env_helpers.py::tests.env_helpers.set_env::lsp_client.py::gabion.lsp_client._env_timeout_ticks
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


# gabion:evidence E:call_footprint::tests/test_lsp_client_run_command_di.py::test_env_timeout_tick_ns_invalid_falls_back_to_ms::env_helpers.py::tests.env_helpers.restore_env::env_helpers.py::tests.env_helpers.set_env::lsp_client.py::gabion.lsp_client._env_timeout_ticks
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


# gabion:evidence E:call_footprint::tests/test_lsp_client_run_command_di.py::test_env_timeout_tick_ns_missing_raises::env_helpers.py::tests.env_helpers.restore_env::env_helpers.py::tests.env_helpers.set_env::lsp_client.py::gabion.lsp_client._env_timeout_ticks
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


# gabion:evidence E:call_footprint::tests/test_lsp_client_run_command_di.py::test_env_timeout_ms_invalid_returns_none::env_helpers.py::tests.env_helpers.restore_env::env_helpers.py::tests.env_helpers.set_env::lsp_client.py::gabion.lsp_client._env_timeout_ticks
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


# gabion:evidence E:call_footprint::tests/test_lsp_client_run_command_di.py::test_env_timeout_seconds_parsing::env_helpers.py::tests.env_helpers.restore_env::env_helpers.py::tests.env_helpers.set_env::lsp_client.py::gabion.lsp_client._env_timeout_ticks
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


# gabion:evidence E:call_footprint::tests/test_lsp_client_run_command_di.py::test_env_timeout_seconds_invalid_returns_none::env_helpers.py::tests.env_helpers.restore_env::env_helpers.py::tests.env_helpers.set_env::lsp_client.py::gabion.lsp_client._env_timeout_ticks
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


# gabion:evidence E:call_footprint::tests/test_lsp_client_run_command_di.py::test_env_timeout_seconds_sub_millisecond_rejected::env_helpers.py::tests.env_helpers.restore_env::env_helpers.py::tests.env_helpers.set_env::lsp_client.py::gabion.lsp_client._env_timeout_ticks
def test_env_timeout_seconds_sub_millisecond_rejected() -> None:
    previous = _set_env(
        {
            "GABION_LSP_TIMEOUT_TICKS": None,
            "GABION_LSP_TIMEOUT_TICK_NS": None,
            "GABION_LSP_TIMEOUT_MS": None,
            "GABION_LSP_TIMEOUT_SECONDS": "0.0001",
        }
    )
    try:
        with pytest.raises(NeverThrown):
            _env_timeout_ticks()
    finally:
        _restore_env(previous)


# gabion:evidence E:call_footprint::tests/test_lsp_client_run_command_di.py::test_env_timeout_missing_raises::env_helpers.py::tests.env_helpers.restore_env::env_helpers.py::tests.env_helpers.set_env::lsp_client.py::gabion.lsp_client._env_timeout_ticks
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


# gabion:evidence E:call_footprint::tests/test_lsp_client_run_command_di.py::test_run_command_rejects_zero_timeout_ticks::lsp_client.py::gabion.lsp_client.run_command::test_lsp_client_run_command_di.py::tests.test_lsp_client_run_command_di._make_proc
def test_run_command_rejects_zero_timeout_ticks() -> None:
    proc = _make_proc(0, b"")

    def factory(*_args, **_kwargs):
        return proc

    with pytest.raises(NeverThrown):
        run_command(
            CommandRequest("gabion.dataflowAudit", [{"paths": ["."]}]),
            root=Path("."),
            timeout_ticks=0,
            timeout_tick_ns=1_000_000_000,
            process_factory=factory,
        )


# gabion:evidence E:call_footprint::tests/test_lsp_client_run_command_di.py::test_run_command_rejects_zero_tick_ns::lsp_client.py::gabion.lsp_client.run_command::test_lsp_client_run_command_di.py::tests.test_lsp_client_run_command_di._make_proc
def test_run_command_rejects_zero_tick_ns() -> None:
    proc = _make_proc(0, b"")

    def factory(*_args, **_kwargs):
        return proc

    with pytest.raises(NeverThrown):
        run_command(
            CommandRequest("gabion.dataflowAudit", [{"paths": ["."]}]),
            root=Path("."),
            timeout_ticks=1_000_000_000,
            timeout_tick_ns=0,
            process_factory=factory,
        )


# gabion:evidence E:call_footprint::tests/test_lsp_client_run_command_di.py::test_remaining_deadline_ns_raises_when_expired::lsp_client.py::gabion.lsp_client._remaining_deadline_ns
def test_remaining_deadline_ns_raises_when_expired() -> None:
    with pytest.raises(NeverThrown):
        _remaining_deadline_ns(0)


# gabion:evidence E:call_footprint::tests/test_lsp_client_run_command_di.py::test_run_command_uses_unbuffered_stdio::lsp_client.py::gabion.lsp_client.run_command::test_lsp_client_run_command_di.py::tests.test_lsp_client_run_command_di._make_proc
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


# gabion:evidence E:call_footprint::tests/test_lsp_client_run_command_di.py::test_run_command_handles_shutdown_timeout::lsp_client.py::gabion.lsp_client.run_command::test_lsp_client_run_command_di.py::tests.test_lsp_client_run_command_di._rpc_message
def test_run_command_handles_shutdown_timeout() -> None:
    init = _rpc_message({"jsonrpc": "2.0", "id": 1, "result": {}})
    cmd = _rpc_message({"jsonrpc": "2.0", "id": 2, "result": {}})
    shutdown = _rpc_message({"jsonrpc": "2.0", "id": 3, "result": {}})
    proc = _TimeoutOnCommunicateProc(_FakeProcStreams(stdout_bytes=init + cmd + shutdown, stderr_bytes=b""), 0)

    def factory(*_args, **_kwargs):
        return proc

    result = run_command(
        CommandRequest("gabion.dataflowAudit", [{"paths": ["."]}]),
        root=Path("."),
        process_factory=factory,
    )
    assert result == {}
    assert proc.killed is True
