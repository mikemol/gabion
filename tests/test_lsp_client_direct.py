from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from gabion import server
from gabion.lsp_client import CommandRequest, LspClientError, _wait_readable, run_command_direct
from gabion.exceptions import NeverThrown


class _NoFileno:
    pass


# gabion:evidence E:function_site::lsp_client.py::gabion.lsp_client._wait_readable
def test_wait_readable_skips_missing_fileno() -> None:
    _wait_readable(_NoFileno(), time.monotonic() + 0.1)


# gabion:evidence E:function_site::lsp_client.py::gabion.lsp_client._wait_readable
def test_wait_readable_times_out() -> None:
    read_fd, write_fd = os.pipe()
    try:
        with os.fdopen(read_fd, "rb", closefd=True) as reader, os.fdopen(
            write_fd, "wb", closefd=True
        ):
            with pytest.raises(LspClientError):
                _wait_readable(reader, time.monotonic())
    finally:
        try:
            os.close(write_fd)
        except OSError:
            pass


# gabion:evidence E:function_site::lsp_client.py::gabion.lsp_client.run_command_direct
def test_run_command_direct_structure_reuse_and_decision_diff(tmp_path: Path) -> None:
    reuse_request = CommandRequest(server.STRUCTURE_REUSE_COMMAND, [{}])
    reuse_result = run_command_direct(reuse_request, root=tmp_path)
    assert reuse_result["exit_code"] == 2

    diff_request = CommandRequest(server.DECISION_DIFF_COMMAND, [{}])
    diff_result = run_command_direct(diff_request, root=tmp_path)
    assert diff_result["exit_code"] == 2


# gabion:evidence E:function_site::lsp_client.py::gabion.lsp_client.run_command_direct
def test_run_command_direct_rejects_unknown_command(tmp_path: Path) -> None:
    with pytest.raises(LspClientError):
        run_command_direct(CommandRequest("gabion.unknown", []), root=tmp_path)


def test_run_command_direct_rejects_non_dict_payload(tmp_path: Path) -> None:
    with pytest.raises(NeverThrown):
        run_command_direct(CommandRequest(server.DATAFLOW_COMMAND, [123]), root=tmp_path)
