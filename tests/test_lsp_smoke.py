from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import pytest

_TIMEOUT_TICKS = 5_000
_TIMEOUT_TICK_NS = 1_000_000

def _has_pygls() -> bool:
    return importlib.util.find_spec("pygls") is not None

class _FakeProcess:
    def __init__(self, stdout_bytes: bytes, on_start=None) -> None:
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(stdout_bytes)
        self.stderr = io.BytesIO()
        self.returncode = 0
        if on_start:
            on_start()

    def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
        return (b"", b"")

def _rpc_response(msg_id: int, result: dict) -> bytes:
    # dataflow-bundle: msg_id, result
    payload = json.dumps({"jsonrpc": "2.0", "id": msg_id, "result": result}).encode(
        "utf-8"
    )
    header = f"Content-Length: {len(payload)}\r\n\r\n".encode("utf-8")
    return header + payload

def _fake_process_factory(stdout_bytes: bytes, on_start=None):
    def _factory(*_args, **_kwargs):
        return _FakeProcess(stdout_bytes, on_start=on_start)

    return _factory

# gabion:evidence E:decision_surface/direct::lsp_client.py::gabion.lsp_client._read_response::request_id E:decision_surface/direct::lsp_client.py::gabion.lsp_client._read_response::stale_da244cdbb037
@pytest.mark.skipif(not _has_pygls(), reason="pygls not installed")
def test_lsp_execute_command(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sample = tmp_path / "sample.py"
    sample.write_text("def alpha(x):\n    return x\n")
    from gabion.lsp_client import CommandRequest, run_command

    stdout_bytes = b"".join(
        [
            _rpc_response(1, {}),
            _rpc_response(2, {"exit_code": 0}),
            _rpc_response(3, {}),
        ]
    )
    result = run_command(
        CommandRequest(
            "gabion.dataflowAudit",
            [{"paths": [str(tmp_path)], "fail_on_violations": False}],
        ),
        root=tmp_path,
        timeout_ticks=_TIMEOUT_TICKS,
        timeout_tick_ns=_TIMEOUT_TICK_NS,
        process_factory=_fake_process_factory(stdout_bytes),
    )
    assert "exit_code" in result
    snapshot_bytes = b"".join(
        [
            _rpc_response(1, {}),
            _rpc_response(2, {"structure_tree": "{}", "structure_metrics": "{}"}),
            _rpc_response(3, {}),
        ]
    )
    snapshot_result = run_command(
        CommandRequest(
            "gabion.dataflowAudit",
            [
                {
                    "paths": [str(tmp_path)],
                    "fail_on_violations": False,
                    "structure_tree": "-",
                    "structure_metrics": "-",
                }
            ],
        ),
        root=tmp_path,
        timeout_ticks=_TIMEOUT_TICKS,
        timeout_tick_ns=_TIMEOUT_TICK_NS,
        process_factory=_fake_process_factory(snapshot_bytes),
    )
    assert "structure_tree" in snapshot_result
    assert "structure_metrics" in snapshot_result
    synth_bytes = b"".join(
        [
            _rpc_response(1, {}),
            _rpc_response(2, {"protocols": []}),
            _rpc_response(3, {}),
        ]
    )
    synth_result = run_command(
        CommandRequest(
            "gabion.synthesisPlan",
            [
                {
                    "bundles": [{"bundle": ["ctx"], "tier": 2}],
                    "min_bundle_size": 1,
                    "allow_singletons": True,
                    "existing_names": ["CtxBundle"],
                }
            ],
        ),
        root=repo_root,
        timeout_ticks=_TIMEOUT_TICKS,
        timeout_tick_ns=_TIMEOUT_TICK_NS,
        process_factory=_fake_process_factory(synth_bytes),
    )
    assert "protocols" in synth_result

# gabion:evidence E:decision_surface/direct::lsp_client.py::gabion.lsp_client._read_response::request_id E:decision_surface/direct::lsp_client.py::gabion.lsp_client._read_response::stale_161b964c3edd
@pytest.mark.skipif(not _has_pygls(), reason="pygls not installed")
def test_lsp_execute_command_writes_structure_snapshot(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    from gabion.lsp_client import CommandRequest, run_command

    sample = tmp_path / "sample.py"
    sample.write_text("def alpha(a, b):\n    return a + b\n")
    snapshot = tmp_path / "structure.json"
    stdout_bytes = b"".join(
        [
            _rpc_response(1, {}),
            _rpc_response(2, {"exit_code": 0}),
            _rpc_response(3, {}),
        ]
    )
    def _write_snapshot() -> None:
        snapshot.write_text("{\"format_version\": 1}")

    result = run_command(
        CommandRequest(
            "gabion.dataflowAudit",
            [
                {
                    "paths": [str(sample)],
                    "fail_on_violations": False,
                    "structure_tree": str(snapshot),
                }
            ],
        ),
        root=tmp_path,
        timeout_ticks=_TIMEOUT_TICKS,
        timeout_tick_ns=_TIMEOUT_TICK_NS,
        process_factory=_fake_process_factory(stdout_bytes, on_start=_write_snapshot),
    )
    assert "exit_code" in result
    assert snapshot.exists()
    assert "\"format_version\"" in snapshot.read_text()
