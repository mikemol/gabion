from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any


class LspClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class CommandRequest:
    command: str
    arguments: list[dict] | None = None


def _read_rpc(stream) -> dict:
    header = b""
    while b"\r\n\r\n" not in header:
        chunk = stream.read(1)
        if not chunk:
            raise LspClientError("LSP stream closed")
        header += chunk
    head, _, rest = header.partition(b"\r\n\r\n")
    length = 0
    for line in head.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1].strip())
            break
    if length <= 0:
        raise LspClientError("Invalid LSP Content-Length")
    body = rest
    if len(body) < length:
        body += stream.read(length - len(body))
    return json.loads(body.decode("utf-8"))


def _write_rpc(stream, message: dict) -> None:
    payload = json.dumps(message).encode("utf-8")
    header = f"Content-Length: {len(payload)}\r\n\r\n".encode("utf-8")
    stream.write(header + payload)
    stream.flush()


def _read_response(stream, request_id: int) -> dict:
    while True:
        message = _read_rpc(stream)
        if message.get("id") == request_id:
            return message


def run_command(
    request: CommandRequest,
    *,
    root: Path | None = None,
    timeout: float = 5.0,
) -> dict[str, Any]:
    proc = subprocess.Popen(
        [sys.executable, "-m", "gabion.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None

    root_uri = (root or Path.cwd()).resolve().as_uri()
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
    _read_response(proc.stdout, initialize_id)
    _write_rpc(proc.stdin, {"jsonrpc": "2.0", "method": "initialized", "params": {}})

    cmd_id = 2
    _write_rpc(
        proc.stdin,
        {
            "jsonrpc": "2.0",
            "id": cmd_id,
            "method": "workspace/executeCommand",
            "params": {"command": request.command, "arguments": request.arguments or []},
        },
    )
    response = _read_response(proc.stdout, cmd_id)

    shutdown_id = 3
    _write_rpc(proc.stdin, {"jsonrpc": "2.0", "id": shutdown_id, "method": "shutdown"})
    _read_response(proc.stdout, shutdown_id)
    _write_rpc(proc.stdin, {"jsonrpc": "2.0", "method": "exit"})
    out, err = proc.communicate(timeout=timeout)
    if response.get("error"):
        raise LspClientError(f"LSP error: {response['error']}")
    if proc.returncode not in (0, None):
        detail = err.decode("utf-8", errors="replace").strip()
        raise LspClientError(f"LSP server failed (exit {proc.returncode}): {detail}")
    if err:
        detail = err.decode("utf-8", errors="replace").strip()
        if detail:
            raise LspClientError(f"LSP server error output: {detail}")
    return response.get("result", {})
