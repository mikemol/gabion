from __future__ import annotations

from gabion.server_core import dataflow_runtime_contract as runtime_contract


def test_collection_checkpoint_flush_due_semantic_interval() -> None:
    assert runtime_contract.collection_checkpoint_flush_due(
        intro_changed=False,
        remaining_files=3,
        semantic_substantive_progress=True,
        now_ns=2_000_000_000,
        last_flush_ns=1_000_000_000,
    ) is True


def test_collection_report_flush_due_terminal_and_stride() -> None:
    assert runtime_contract.collection_report_flush_due(
        completed_files=8,
        remaining_files=2,
        now_ns=0,
        last_flush_ns=0,
        last_flush_completed=0,
    ) is True
    assert runtime_contract.collection_report_flush_due(
        completed_files=1,
        remaining_files=0,
        now_ns=1,
        last_flush_ns=1,
        last_flush_completed=1,
    ) is True


def test_progress_heartbeat_seconds_bounds() -> None:
    assert runtime_contract.progress_heartbeat_seconds({"progress_heartbeat_seconds": " "}) == (
        runtime_contract.DEFAULT_PROGRESS_HEARTBEAT_SECONDS
    )
    assert runtime_contract.progress_heartbeat_seconds({"progress_heartbeat_seconds": "1"}) == (
        runtime_contract.MIN_PROGRESS_HEARTBEAT_SECONDS
    )
