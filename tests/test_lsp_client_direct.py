from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from gabion import server
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


def test_wait_readable_times_out_when_only_read_method_exists() -> None:
    with pytest.raises(LspClientError):
        _wait_readable(_ReadOnlyTimedOut(), time.monotonic_ns())


def test_wait_readable_handles_fileno_failure_with_and_without_read() -> None:
    with pytest.raises(LspClientError):
        _wait_readable(_FilenoRaisesNoRead(), time.monotonic_ns() + 100_000_000)
    with pytest.raises(LspClientError):
        _wait_readable(_FilenoRaisesWithRead(), time.monotonic_ns())


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


def test_run_command_direct_rejects_non_dict_payload(tmp_path: Path) -> None:
    with pytest.raises(NeverThrown):
        run_command_direct(CommandRequest(server.DATAFLOW_COMMAND, [123]), root=tmp_path)


def test_run_command_direct_requires_analysis_timeout(tmp_path: Path) -> None:
    with pytest.raises(NeverThrown):
        run_command_direct(CommandRequest(server.DATAFLOW_COMMAND, [{"paths": ["."]}]), root=tmp_path)


def test_analysis_timeout_total_ns_requires_timeout_fields() -> None:
    with pytest.raises(NeverThrown):
        _analysis_timeout_total_ns({})
