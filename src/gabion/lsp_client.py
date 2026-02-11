from __future__ import annotations

import json
import math
import os
import select
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

from gabion.json_types import JSONObject
from gabion import server
from gabion.analysis.timeout_context import check_deadline


class LspClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class CommandRequest:
    command: str
    arguments: list[JSONObject] | None = None


def _env_timeout_seconds() -> float | None:
    raw = os.getenv("GABION_LSP_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return value


def _wait_readable(stream, deadline: float | None) -> None:
    if deadline is None:
        return
    fileno = getattr(stream, "fileno", None)
    if fileno is None:
        return
    try:
        fd = fileno()
    except Exception:
        return
    timeout = max(0.0, deadline - time.monotonic())
    ready, _, _ = select.select([fd], [], [], timeout)
    if not ready:
        raise LspClientError("LSP response timed out")


def _read_rpc(stream, deadline: float | None = None) -> JSONObject:
    check_deadline()
    header = b""
    while b"\r\n\r\n" not in header:
        check_deadline()
        _wait_readable(stream, deadline)
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
        _wait_readable(stream, deadline)
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
    stream, request_id: int, deadline: float | None = None
) -> JSONObject:
    check_deadline()
    while True:
        check_deadline()
        message = _read_rpc(stream, deadline)
        if message.get("id") == request_id:
            return message


def run_command(
    request: CommandRequest,
    *,
    root: Path | None = None,
    timeout: float = 5.0,
    process_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
) -> JSONObject:
    env_timeout = _env_timeout_seconds()
    if env_timeout is not None:
        timeout = env_timeout
    proc = process_factory(
        [sys.executable, "-m", "gabion.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None

    root_uri = (root or Path.cwd()).resolve().as_uri()
    deadline = None
    if timeout is not None:
        deadline = time.monotonic() + max(timeout, 0.0)
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
    _read_response(proc.stdout, initialize_id, deadline)
    _write_rpc(proc.stdin, {"jsonrpc": "2.0", "method": "initialized", "params": {}})

    cmd_id = 2
    command_args = list(request.arguments or [])
    if deadline is not None and command_args and isinstance(command_args[0], dict):
        remaining = max(0.0, deadline - time.monotonic())
        if remaining > 0:
            payload = dict(command_args[0])
            existing = payload.get("analysis_timeout_seconds")
            try:
                existing_value = float(existing) if existing is not None else None
            except (TypeError, ValueError):
                existing_value = None
            if existing_value is None or existing_value > remaining:
                payload["analysis_timeout_seconds"] = remaining
            command_args[0] = payload
    _write_rpc(
        proc.stdin,
        {
            "jsonrpc": "2.0",
            "id": cmd_id,
            "method": "workspace/executeCommand",
            "params": {"command": request.command, "arguments": command_args},
        },
    )
    response = _read_response(proc.stdout, cmd_id, deadline)

    shutdown_id = 3
    _write_rpc(proc.stdin, {"jsonrpc": "2.0", "id": shutdown_id, "method": "shutdown"})
    _read_response(proc.stdout, shutdown_id, deadline)
    _write_rpc(proc.stdin, {"jsonrpc": "2.0", "method": "exit"})
    remaining = None
    if deadline is not None:
        remaining = max(0.0, deadline - time.monotonic())
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
    payload = {}
    if request.arguments:
        payload = request.arguments[0]
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
