from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import time
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Mapping

from gabion.json_types import JSONObject
from gabion import server
from gabion.analysis.timeout_context import (
    Deadline,
    check_deadline,
    deadline_clock_scope,
    deadline_scope,
)
from gabion.deadline_clock import GasMeter
from gabion.invariants import never
from gabion.order_contract import ordered_or_sorted


class LspClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class CommandRequest:
    command: str
    arguments: list[JSONObject] = field(default_factory=list)


def _normalized_command_payload(
    request: CommandRequest,
) -> tuple[list[JSONObject], dict[str, object]]:
    command_args = list(request.arguments)
    if not command_args:
        never("missing command payload arguments", command=request.command)
    payload_arg = command_args[0]
    if not isinstance(payload_arg, dict):
        never(
            "command payload must be a dict",
            command=request.command,
            payload_type=type(payload_arg).__name__,
        )
    payload = dict(payload_arg)
    command_args[0] = payload
    return command_args, payload


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


def _has_analysis_timeout(payload: Mapping[str, object]) -> bool:
    return any(
        payload.get(key) not in (None, "")
        for key in (
            "analysis_timeout_ticks",
            "analysis_timeout_tick_ns",
            "analysis_timeout_ms",
            "analysis_timeout_seconds",
        )
    )


def _analysis_timeout_total_ns(payload: Mapping[str, object]) -> int:
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
    never(
        "missing analysis timeout",
        payload_keys=ordered_or_sorted(
            (str(key) for key in payload.keys()),
            source="_analysis_timeout_ns.payload_keys",
        ),
    )


def _analysis_timeout_slack_ns(total_ns: int) -> int:
    slack_ns = total_ns // 3
    if slack_ns < 2_000_000_000:
        slack_ns = 2_000_000_000
    if slack_ns > 120_000_000_000:
        slack_ns = 120_000_000_000
    return slack_ns


def _wait_readable(stream, deadline_ns: int) -> None:
    read = getattr(stream, "read", None)
    fileno = getattr(stream, "fileno", None)
    if fileno is None:
        if read is None:
            raise LspClientError("LSP stream does not expose fileno")
        if time.monotonic_ns() >= deadline_ns:
            raise LspClientError("LSP response timed out")
        return
    try:
        fd = fileno()
    except (OSError, ValueError) as exc:
        if read is None:
            raise LspClientError("LSP stream fileno failed") from exc
        if time.monotonic_ns() >= deadline_ns:
            raise LspClientError("LSP response timed out")
        return
    remaining_ns = deadline_ns - time.monotonic_ns()
    timeout = max(0.0, remaining_ns / 1_000_000_000)
    ready, _, _ = select.select([fd], [], [], timeout)
    if not ready:
        raise LspClientError("LSP response timed out")


def _read_exact(stream, length: int, deadline_ns: int) -> bytes:
    body = bytearray()
    while len(body) < length:
        check_deadline()
        _wait_readable(stream, deadline_ns)
        chunk = stream.read(length - len(body))
        if not chunk:
            raise LspClientError("LSP stream closed")
        body.extend(chunk)
    return bytes(body)


def _remaining_deadline_ns(deadline_ns: int) -> int:
    remaining_ns = deadline_ns - time.monotonic_ns()
    if remaining_ns <= 0:
        never("lsp deadline already expired")
    return remaining_ns


def _read_rpc(stream, deadline_ns: int) -> JSONObject:
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
        body += _read_exact(stream, length - len(body), deadline_ns)
    elif len(body) > length:
        body = body[:length]
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
    stream,
    request_id: int,
    deadline_ns: int,
    *,
    notification_callback: Callable[[JSONObject], None] | None = None,
) -> JSONObject:
    check_deadline()
    while True:
        check_deadline()
        message = _read_rpc(stream, deadline_ns)
        if "id" not in message:
            if notification_callback is not None:
                notification_callback(message)
            continue
        if message.get("id") == request_id:
            return message


def run_command(
    request: CommandRequest,
    *,
    root: Path | None = None,
    timeout_ticks: int | None = None,
    timeout_tick_ns: int = 1_000_000,
    process_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
    remaining_deadline_ns_fn: Callable[[int], int] | None = None,
    notification_callback: Callable[[JSONObject], None] | None = None,
) -> JSONObject:
    if _has_env_timeout():
        timeout_ticks, timeout_tick_ns = _env_timeout_ticks()
    if timeout_ticks is None:
        timeout_ticks = 100
    ticks_value = int(timeout_ticks)
    tick_ns_value = int(timeout_tick_ns)
    if ticks_value <= 0:
        never("invalid lsp timeout ticks", ticks=timeout_ticks)
    if tick_ns_value <= 0:
        never("invalid lsp timeout tick_ns", tick_ns=timeout_tick_ns)

    command_args, payload = _normalized_command_payload(request)

    base_total_ns = ticks_value * tick_ns_value
    has_existing_analysis_timeout = _has_analysis_timeout(payload)
    analysis_target_ns = (
        _analysis_timeout_total_ns(payload)
        if has_existing_analysis_timeout
        else base_total_ns
    )
    slack_ns = _analysis_timeout_slack_ns(analysis_target_ns)
    lsp_total_ns = max(base_total_ns, analysis_target_ns + slack_ns)
    lsp_ticks = max(1, (lsp_total_ns + tick_ns_value - 1) // tick_ns_value)

    deadline = Deadline.from_timeout_ticks(lsp_ticks, tick_ns_value)
    deadline_ns = deadline.deadline_ns
    proc = process_factory(
        [sys.executable, "-m", "gabion.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None

    root_uri = (root or Path.cwd()).resolve().as_uri()
    logical_limit = max(10_000, int(lsp_ticks) * 1_000)
    remaining_deadline_ns = remaining_deadline_ns_fn or _remaining_deadline_ns
    with deadline_scope(deadline):
        with deadline_clock_scope(GasMeter(limit=logical_limit)):
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
            _read_response(
                proc.stdout,
                initialize_id,
                deadline_ns,
                notification_callback=notification_callback,
            )
            _write_rpc(proc.stdin, {"jsonrpc": "2.0", "method": "initialized", "params": {}})

            cmd_id = 2
            remaining_ns = remaining_deadline_ns(deadline_ns)
            if not has_existing_analysis_timeout or analysis_target_ns > remaining_ns:
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
            response = _read_response(
                proc.stdout,
                cmd_id,
                deadline_ns,
                notification_callback=notification_callback,
            )

            shutdown_id = 3
            _write_rpc(
                proc.stdin,
                {"jsonrpc": "2.0", "id": shutdown_id, "method": "shutdown"},
            )
            _read_response(
                proc.stdout,
                shutdown_id,
                deadline_ns,
                notification_callback=notification_callback,
            )
            _write_rpc(proc.stdin, {"jsonrpc": "2.0", "method": "exit"})
            remaining_ns = remaining_deadline_ns(deadline_ns)
            remaining = max(1.0, remaining_ns / 1_000_000_000)
            try:
                out, err = proc.communicate(timeout=remaining)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, err = proc.communicate(timeout=1.0)
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
    notification_callback: Callable[[JSONObject], None] | None = None,
    execute_dataflow_fn: Callable[[object, JSONObject], JSONObject] | None = None,
) -> JSONObject:
    _, payload = _normalized_command_payload(request)
    if (
        "analysis_timeout_ticks" not in payload
        and "analysis_timeout_ms" not in payload
        and "analysis_timeout_seconds" not in payload
    ):
        never("missing analysis timeout in direct command payload")
    workspace = SimpleNamespace(root_path=str((root or Path.cwd()).resolve()))

    def _send_notification(method: str, params: object) -> None:
        if notification_callback is None:
            return
        if not isinstance(params, Mapping):
            return
        notification_callback(
            {
                "jsonrpc": "2.0",
                "method": str(method),
                "params": {str(key): params[key] for key in params},
            }
        )

    ls = SimpleNamespace(workspace=workspace, send_notification=_send_notification)
    if request.command == server.DATAFLOW_COMMAND:
        execute_fn = execute_dataflow_fn or server.execute_command
        return execute_fn(ls, payload)
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
    if request.command == server.IMPACT_COMMAND:
        return server.execute_impact(ls, payload)
    raise LspClientError(f"Unsupported direct command: {request.command}")
