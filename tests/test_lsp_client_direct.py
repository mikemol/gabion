from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from gabion import server
from gabion.commands import command_ids
from gabion.lsp_client import (
    CommandRequest,
    LspClientError,
    _analysis_timeout_total_ns,
    _wait_readable,
    run_command_direct,
)
from gabion.exceptions import NeverThrown


class _NoFileno:
    pass


class _ReadOnlyTimedOut:
    def read(self, _length: int) -> bytes:
        return b""


class _FilenoRaisesWithRead:
    def fileno(self) -> int:
        raise OSError("boom")

    def read(self, _length: int) -> bytes:
        return b""


class _FilenoRaisesNoRead:
    def fileno(self) -> int:
        raise OSError("boom")


# gabion:evidence E:function_site::lsp_client.py::gabion.lsp_client._wait_readable
def test_wait_readable_rejects_missing_fileno() -> None:
    with pytest.raises(LspClientError):
        _wait_readable(_NoFileno(), time.monotonic_ns() + 100_000_000)


# gabion:evidence E:function_site::lsp_client.py::gabion.lsp_client._wait_readable
def test_wait_readable_times_out() -> None:
    read_fd, write_fd = os.pipe()
    try:
        with os.fdopen(read_fd, "rb", closefd=True) as reader, os.fdopen(
            write_fd, "wb", closefd=True
        ):
            with pytest.raises(LspClientError):
                _wait_readable(reader, time.monotonic_ns())
    finally:
        try:
            os.close(write_fd)
        except OSError:
            pass


# gabion:evidence E:call_footprint::tests/test_lsp_client_direct.py::test_wait_readable_times_out_when_only_read_method_exists::lsp_client.py::gabion.lsp_client._wait_readable
def test_wait_readable_times_out_when_only_read_method_exists() -> None:
    with pytest.raises(LspClientError):
        _wait_readable(_ReadOnlyTimedOut(), time.monotonic_ns())


# gabion:evidence E:call_footprint::tests/test_lsp_client_direct.py::test_wait_readable_handles_fileno_failure_with_and_without_read::lsp_client.py::gabion.lsp_client._wait_readable
def test_wait_readable_handles_fileno_failure_with_and_without_read() -> None:
    with pytest.raises(LspClientError):
        _wait_readable(_FilenoRaisesNoRead(), time.monotonic_ns() + 100_000_000)
    with pytest.raises(LspClientError):
        _wait_readable(_FilenoRaisesWithRead(), time.monotonic_ns())


# gabion:evidence E:call_footprint::tests/test_lsp_client_direct.py::test_wait_readable_returns_when_stream_is_ready::lsp_client.py::gabion.lsp_client._wait_readable
def test_wait_readable_returns_when_stream_is_ready() -> None:
    read_fd, write_fd = os.pipe()
    try:
        with os.fdopen(read_fd, "rb", closefd=True) as reader, os.fdopen(
            write_fd, "wb", closefd=True
        ) as writer:
            writer.write(b"x")
            writer.flush()
            _wait_readable(reader, time.monotonic_ns() + 1_000_000_000)
            assert reader.read(1) == b"x"
    finally:
        try:
            os.close(write_fd)
        except OSError:
            pass


# gabion:evidence E:function_site::lsp_client.py::gabion.lsp_client.run_command_direct
def test_run_command_direct_structure_reuse_and_decision_diff(tmp_path: Path) -> None:
    reuse_request = CommandRequest(
        server.STRUCTURE_REUSE_COMMAND,
        [{"analysis_timeout_ticks": 100, "analysis_timeout_tick_ns": 1_000_000}],
    )
    reuse_result = run_command_direct(reuse_request, root=tmp_path)
    assert reuse_result["exit_code"] == 2

    diff_request = CommandRequest(
        server.DECISION_DIFF_COMMAND,
        [{"analysis_timeout_ticks": 100, "analysis_timeout_tick_ns": 1_000_000}],
    )
    diff_result = run_command_direct(diff_request, root=tmp_path)
    assert diff_result["exit_code"] == 2


# gabion:evidence E:call_footprint::tests/test_lsp_client_direct.py::test_run_command_direct_forwards_notifications::lsp_client.py::gabion.lsp_client.run_command_direct
def test_run_command_direct_forwards_notifications(tmp_path: Path) -> None:
    seen: list[dict[str, object]] = []

    def _fake_execute_command(ls, _payload=None):
        ls.send_notification(
            "$/progress",
            {
                "token": "gabion.dataflowAudit/progress-v1",
                "value": {"resume_checkpoint": {"status": "checkpoint_loaded"}},
            },
        )
        return {"exit_code": 0}

    request = CommandRequest(
        server.DATAFLOW_COMMAND,
        [{"analysis_timeout_ticks": 100, "analysis_timeout_tick_ns": 1_000_000}],
    )
    result = run_command_direct(
        request,
        root=tmp_path,
        notification_callback=seen.append,
        execute_dataflow_fn=_fake_execute_command,
    )
    assert result["exit_code"] == 0
    assert seen == [
        {
            "jsonrpc": "2.0",
            "method": "$/progress",
            "params": {
                "token": "gabion.dataflowAudit/progress-v1",
                "value": {"resume_checkpoint": {"status": "checkpoint_loaded"}},
            },
        }
    ]


# gabion:evidence E:call_footprint::tests/test_lsp_client_direct.py::test_run_command_direct_routes_check_through_dataflow_di_executor::lsp_client.py::gabion.lsp_client.run_command_direct
def test_run_command_direct_routes_check_through_dataflow_di_executor(
    tmp_path: Path,
) -> None:
    seen_commands: list[str] = []

    def _fake_execute_command(_ls, _payload=None):
        seen_commands.append("called")
        return {"exit_code": 0, "analysis_state": "succeeded"}

    request = CommandRequest(
        command_ids.CHECK_COMMAND,
        [{"analysis_timeout_ticks": 100, "analysis_timeout_tick_ns": 1_000_000}],
    )
    result = run_command_direct(
        request,
        root=tmp_path,
        execute_dataflow_fn=_fake_execute_command,
    )
    assert result["exit_code"] == 0
    assert seen_commands == ["called"]


# gabion:evidence E:call_footprint::tests/test_lsp_client_direct.py::test_run_command_direct_ignores_non_mapping_notifications::lsp_client.py::gabion.lsp_client.run_command_direct
def test_run_command_direct_ignores_non_mapping_notifications(
    tmp_path: Path,
) -> None:
    seen: list[dict[str, object]] = []

    def _fake_execute_command(ls, _payload=None):
        ls.send_notification("$/progress", "not-a-mapping")
        return {"exit_code": 0}

    request = CommandRequest(
        server.DATAFLOW_COMMAND,
        [{"analysis_timeout_ticks": 100, "analysis_timeout_tick_ns": 1_000_000}],
    )
    result = run_command_direct(
        request,
        root=tmp_path,
        notification_callback=seen.append,
        execute_dataflow_fn=_fake_execute_command,
    )
    assert result["exit_code"] == 0
    assert seen == []


# gabion:evidence E:function_site::lsp_client.py::gabion.lsp_client.run_command_direct
def test_run_command_direct_rejects_unknown_command(tmp_path: Path) -> None:
    with pytest.raises(LspClientError):
        run_command_direct(
            CommandRequest(
                "gabion.unknown",
                [{"analysis_timeout_ticks": 100, "analysis_timeout_tick_ns": 1_000_000}],
            ),
            root=tmp_path,
        )


# gabion:evidence E:call_footprint::tests/test_lsp_client_direct.py::test_run_command_direct_rejects_non_mapping_execute_result::lsp_client.py::gabion.lsp_client.run_command_direct
def test_run_command_direct_rejects_non_mapping_execute_result(tmp_path: Path) -> None:
    request = CommandRequest(
        server.DATAFLOW_COMMAND,
        [{"analysis_timeout_ticks": 100, "analysis_timeout_tick_ns": 1_000_000}],
    )
    with pytest.raises(LspClientError):
        run_command_direct(
            request,
            root=tmp_path,
            execute_dataflow_fn=lambda _ls, _payload=None: [],
        )


# gabion:evidence E:call_footprint::tests/test_lsp_client_direct.py::test_run_command_direct_rejects_non_dict_payload::lsp_client.py::gabion.lsp_client.run_command_direct
def test_run_command_direct_rejects_non_dict_payload(tmp_path: Path) -> None:
    with pytest.raises(NeverThrown):
        run_command_direct(CommandRequest(server.DATAFLOW_COMMAND, [123]), root=tmp_path)


# gabion:evidence E:call_footprint::tests/test_lsp_client_direct.py::test_run_command_direct_requires_analysis_timeout::lsp_client.py::gabion.lsp_client.run_command_direct
def test_run_command_direct_requires_analysis_timeout(tmp_path: Path) -> None:
    with pytest.raises(NeverThrown):
        run_command_direct(CommandRequest(server.DATAFLOW_COMMAND, [{"paths": ["."]}]), root=tmp_path)


# gabion:evidence E:call_footprint::tests/test_lsp_client_direct.py::test_analysis_timeout_total_ns_requires_timeout_fields::lsp_client.py::gabion.lsp_client._analysis_timeout_total_ns
def test_analysis_timeout_total_ns_requires_timeout_fields() -> None:
    with pytest.raises(NeverThrown):
        _analysis_timeout_total_ns({})
