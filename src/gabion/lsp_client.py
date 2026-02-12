from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import time
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

from gabion.json_types import JSONObject
from gabion import server
from gabion.analysis.timeout_context import Deadline, check_deadline, deadline_scope
from gabion.invariants import never


class LspClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class CommandRequest:
    command: str
    arguments: list[JSONObject] | None = None


def _has_env_timeout() -> bool:
    return any(
        os.getenv(key, "").strip()
        for key in (
            "GABION_LSP_TIMEOUT_TICKS",
            "GABION_LSP_TIMEOUT_TICK_NS",
            "GABION_LSP_TIMEOUT_MS",
            "GABION_LSP_TIMEOUT_SECONDS",
        )
    )


def _env_timeout_ticks() -> tuple[int, int]:
    raw_ticks = os.getenv("GABION_LSP_TIMEOUT_TICKS", "").strip()
    raw_tick_ns = os.getenv("GABION_LSP_TIMEOUT_TICK_NS", "").strip()
    if raw_ticks:
        try:
            ticks = int(raw_ticks)
        except ValueError:
            ticks = -1
        if ticks > 0:
            if not raw_tick_ns:
                never("missing env timeout tick_ns", ticks=raw_ticks)
            try:
                tick_ns = int(raw_tick_ns)
            except ValueError:
                tick_ns = -1
            if tick_ns > 0:
                return ticks, tick_ns
            never("invalid env timeout tick_ns", tick_ns=raw_tick_ns)
        never("invalid env timeout ticks", ticks=raw_ticks)
    raw_ms = os.getenv("GABION_LSP_TIMEOUT_MS", "").strip()
    if raw_ms:
        try:
            ticks = int(raw_ms)
        except ValueError:
            ticks = -1
        if ticks > 0:
            return ticks, 1_000_000
        never("invalid env timeout ms", ms=raw_ms)
    raw_seconds = os.getenv("GABION_LSP_TIMEOUT_SECONDS", "").strip()
    if raw_seconds:
        try:
            seconds = Decimal(raw_seconds)
        except (InvalidOperation, ValueError):
            seconds = Decimal(-1)
        if seconds > 0:
            millis = int(seconds * Decimal(1000))
            if millis > 0:
                return millis, 1_000_000
        never("invalid env timeout seconds", seconds=raw_seconds)
    never("missing env timeout configuration")


def _analysis_timeout_total_ns(payload: dict) -> int | None:
    existing_ticks = payload.get("analysis_timeout_ticks")
    existing_tick_ns = payload.get("analysis_timeout_tick_ns")
    if existing_ticks not in (None, "") or existing_tick_ns not in (None, ""):
        if existing_ticks in (None, "") or existing_tick_ns in (None, ""):
            never(
                "missing analysis timeout tick_ns",
                ticks=existing_ticks,
                tick_ns=existing_tick_ns,
            )
        try:
            ticks_value = int(existing_ticks)
            tick_ns_value = int(existing_tick_ns)
        except (TypeError, ValueError):
            never(
                "invalid analysis timeout ticks",
                ticks=existing_ticks,
                tick_ns=existing_tick_ns,
            )
        if ticks_value <= 0 or tick_ns_value <= 0:
            never(
                "invalid analysis timeout ticks",
                ticks=existing_ticks,
                tick_ns=existing_tick_ns,
            )
        return ticks_value * tick_ns_value
    existing_ms = payload.get("analysis_timeout_ms")
    if existing_ms not in (None, ""):
        try:
            ms_value = int(existing_ms)
        except (TypeError, ValueError):
            never("invalid analysis timeout ms", ms=existing_ms)
        if ms_value <= 0:
            never("invalid analysis timeout ms", ms=existing_ms)
        return ms_value * 1_000_000
    existing_seconds = payload.get("analysis_timeout_seconds")
    if existing_seconds not in (None, ""):
        try:
            seconds_value = Decimal(str(existing_seconds))
        except (InvalidOperation, ValueError):
            never("invalid analysis timeout seconds", seconds=existing_seconds)
        if seconds_value <= 0:
            never("invalid analysis timeout seconds", seconds=existing_seconds)
        return int(seconds_value * Decimal(1_000_000_000))
    return None


def _analysis_timeout_slack_ns(total_ns: int) -> int:
    slack_ns = total_ns // 5
    if slack_ns < 1_000_000_000:
        slack_ns = 1_000_000_000
    if slack_ns > 60_000_000_000:
        slack_ns = 60_000_000_000
    return slack_ns


def _wait_readable(stream, deadline_ns: int | None) -> None:
    if deadline_ns is None:
        return
    fileno = getattr(stream, "fileno", None)
    if fileno is None:
        return
    try:
        fd = fileno()
    except Exception:
        return
    remaining_ns = deadline_ns - time.monotonic_ns()
    timeout = max(0.0, remaining_ns / 1_000_000_000)
    ready, _, _ = select.select([fd], [], [], timeout)
    if not ready:
        raise LspClientError("LSP response timed out")


def _remaining_deadline_ns(deadline_ns: int) -> int:
    remaining_ns = deadline_ns - time.monotonic_ns()
    if remaining_ns <= 0:
        never("lsp deadline already expired")
    return remaining_ns


def _read_rpc(stream, deadline_ns: int | None = None) -> JSONObject:
    check_deadline()
    header = b""
    while b"\r\n\r\n" not in header:
        check_deadline()
        _wait_readable(stream, deadline_ns)
        chunk = stream.read(1)
        if not chunk:
            raise LspClientError("LSP stream closed")
        header += chunk
    head, _, rest = header.partition(b"\r\n\r\n")
    length = 0
    for line in head.split(b"\r\n"):
        check_deadline()
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1].strip())
            break
    if length <= 0:
        raise LspClientError("Invalid LSP Content-Length")
    body = rest
    if len(body) < length:
        _wait_readable(stream, deadline_ns)
        body += stream.read(length - len(body))
    message = json.loads(body.decode("utf-8"))
    if not isinstance(message, dict):
        raise LspClientError("Invalid LSP message payload")
    return message


def _write_rpc(stream, message: JSONObject) -> None:
    payload = json.dumps(message).encode("utf-8")
    header = f"Content-Length: {len(payload)}\r\n\r\n".encode("utf-8")
    stream.write(header + payload)
    stream.flush()


def _read_response(
    stream, request_id: int, deadline_ns: int | None = None
) -> JSONObject:
    check_deadline()
    while True:
        check_deadline()
        message = _read_rpc(stream, deadline_ns)
        if message.get("id") == request_id:
            return message


def run_command(
    request: CommandRequest,
    *,
    root: Path | None = None,
    timeout_ticks: int | None = None,
    timeout_tick_ns: int = 1_000_000,
    process_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
) -> JSONObject:
    if _has_env_timeout():
        timeout_ticks, timeout_tick_ns = _env_timeout_ticks()
    if timeout_ticks is None:
        timeout_ticks = 100
    ticks_value = int(timeout_ticks)
    tick_ns_value = int(timeout_tick_ns)
    if ticks_value <= 0:
        ticks_value = 1
    if tick_ns_value <= 0:
        tick_ns_value = 1

    command_args = list(request.arguments or [])
    if command_args and isinstance(command_args[0], dict):
        payload = dict(command_args[0])
        command_args[0] = payload
    else:
        payload = {}
        command_args = [payload]

    existing_total_ns = _analysis_timeout_total_ns(payload)
    if existing_total_ns is None:
        analysis_target_ns = ticks_value * tick_ns_value
    else:
        analysis_target_ns = existing_total_ns
    slack_ns = _analysis_timeout_slack_ns(analysis_target_ns)
    base_total_ns = ticks_value * tick_ns_value
    lsp_total_ns = max(base_total_ns, analysis_target_ns + slack_ns)
    lsp_ticks = max(1, (lsp_total_ns + tick_ns_value - 1) // tick_ns_value)

    deadline = Deadline.from_timeout_ticks(lsp_ticks, tick_ns_value)
    deadline_ns = deadline.deadline_ns
    proc = process_factory(
        [sys.executable, "-m", "gabion.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None

    root_uri = (root or Path.cwd()).resolve().as_uri()
    with deadline_scope(deadline):
        initialize_id = 1
        _write_rpc(
            proc.stdin,
            {
                "jsonrpc": "2.0",
                "id": initialize_id,
                "method": "initialize",
                "params": {"rootUri": root_uri, "capabilities": {}},
            },
        )
        _read_response(proc.stdout, initialize_id, deadline_ns)
        _write_rpc(proc.stdin, {"jsonrpc": "2.0", "method": "initialized", "params": {}})

        cmd_id = 2
        remaining_ns = _remaining_deadline_ns(deadline_ns)
        if existing_total_ns is None or existing_total_ns > remaining_ns:
            target_ns = min(analysis_target_ns, remaining_ns)
            tick_ns_value = min(tick_ns_value, target_ns)
            ticks_value = max(1, target_ns // tick_ns_value)
            payload["analysis_timeout_ticks"] = int(ticks_value)
            payload["analysis_timeout_tick_ns"] = int(tick_ns_value)
        _write_rpc(
            proc.stdin,
            {
                "jsonrpc": "2.0",
                "id": cmd_id,
                "method": "workspace/executeCommand",
                "params": {"command": request.command, "arguments": command_args},
            },
        )
        response = _read_response(proc.stdout, cmd_id, deadline_ns)

        shutdown_id = 3
        _write_rpc(proc.stdin, {"jsonrpc": "2.0", "id": shutdown_id, "method": "shutdown"})
        _read_response(proc.stdout, shutdown_id, deadline_ns)
        _write_rpc(proc.stdin, {"jsonrpc": "2.0", "method": "exit"})
        remaining_ns = deadline_ns - time.monotonic_ns()
        remaining = max(0.0, remaining_ns / 1_000_000_000)
        out, err = proc.communicate(timeout=remaining)
        if response.get("error"):
            raise LspClientError(f"LSP error: {response['error']}")
        if proc.returncode not in (0, None):
            detail = err.decode("utf-8", errors="replace").strip()
            raise LspClientError(f"LSP server failed (exit {proc.returncode}): {detail}")
        if err:
            detail = err.decode("utf-8", errors="replace").strip()
            if detail:
                raise LspClientError(f"LSP server error output: {detail}")
        result = response.get("result", {})
        if not isinstance(result, dict):
            raise LspClientError(f"Unexpected LSP result payload: {type(result).__name__}")
        return result


def run_command_direct(
    request: CommandRequest,
    *,
    root: Path | None = None,
) -> JSONObject:
    payload: dict = {}
    if request.arguments:
        if isinstance(request.arguments[0], dict):
            payload = dict(request.arguments[0])
        else:
            never("direct command payload must be a dict", payload=request.arguments[0])
    if "analysis_timeout_ticks" not in payload and "analysis_timeout_ms" not in payload and "analysis_timeout_seconds" not in payload:
        ticks_value, tick_ns_value = _env_timeout_ticks() if _has_env_timeout() else (100, 1_000_000)
        payload["analysis_timeout_ticks"] = int(ticks_value)
        payload["analysis_timeout_tick_ns"] = int(tick_ns_value)
    workspace = SimpleNamespace(root_path=str((root or Path.cwd()).resolve()))
    ls = SimpleNamespace(workspace=workspace)
    if request.command == server.DATAFLOW_COMMAND:
        return server.execute_command(ls, payload)
    if request.command == server.STRUCTURE_DIFF_COMMAND:
        return server.execute_structure_diff(ls, payload)
    if request.command == server.STRUCTURE_REUSE_COMMAND:
        return server.execute_structure_reuse(ls, payload)
    if request.command == server.DECISION_DIFF_COMMAND:
        return server.execute_decision_diff(ls, payload)
    if request.command == server.SYNTHESIS_COMMAND:
        return server.execute_synthesis(ls, payload)
    if request.command == server.REFACTOR_COMMAND:
        return server.execute_refactor(ls, payload)
    raise LspClientError(f"Unsupported direct command: {request.command}")
