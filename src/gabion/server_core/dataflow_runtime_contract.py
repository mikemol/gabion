from __future__ import annotations

"""Server-core ownership module and single source of truth for progress/deadline/report runtime contract values."""

from pathlib import Path
from typing import Literal, Mapping

from gabion.json_types import JSONValue
from gabion.runtime import path_policy

DEFAULT_PHASE_TIMELINE_MD = Path("artifacts/audit_reports/dataflow_phase_timeline.md")
DEFAULT_PHASE_TIMELINE_JSONL = Path("artifacts/audit_reports/dataflow_phase_timeline.jsonl")
DEFAULT_REPORT_SECTION_JOURNAL = Path("artifacts/audit_reports/dataflow_report_sections.json")

COLLECTION_CHECKPOINT_FLUSH_INTERVAL_NS = 2_000_000_000
COLLECTION_CHECKPOINT_MEANINGFUL_MIN_INTERVAL_NS = 1_000_000_000
COLLECTION_REPORT_FLUSH_INTERVAL_NS = 10_000_000_000
COLLECTION_REPORT_FLUSH_COMPLETED_STRIDE = 8

DEFAULT_PROGRESS_HEARTBEAT_SECONDS = 55.0
MIN_PROGRESS_HEARTBEAT_SECONDS = 5.0
PROGRESS_DEADLINE_FLUSH_SECONDS = 5.0
PROGRESS_DEADLINE_WATCHDOG_SECONDS = 10.0
PROGRESS_HEARTBEAT_POLL_SECONDS = 0.05
PROGRESS_DEADLINE_FLUSH_MARGIN_SECONDS = 0.5

LSP_PROGRESS_NOTIFICATION_METHOD = "$/progress"
LSP_PROGRESS_TOKEN_V2 = "gabion.dataflowAudit/progress-v2"
LSP_PROGRESS_TOKEN = LSP_PROGRESS_TOKEN_V2

STDOUT_ALIAS = "-"
STDOUT_PATH = "/dev/stdout"

PHASE_PRIMARY_UNITS: Mapping[str, str] = {
    "collection": "collection_files",
    "forest": "forest_mutable_steps",
    "edge": "edge_tasks",
    "post": "post_tasks",
}


def is_stdout_target(target: object) -> bool:
    return path_policy.is_stdout_target(
        target,
        stdout_alias=STDOUT_ALIAS,
        stdout_path=STDOUT_PATH,
    )


def deadline_tick_budget_allows_check(clock: object) -> bool:
    limit = getattr(clock, "limit", None)
    current = getattr(clock, "current", None)
    if isinstance(limit, int) and isinstance(current, int):
        return (limit - current) > 1
    return True


def collection_checkpoint_flush_due(
    *,
    intro_changed: bool,
    remaining_files: int,
    semantic_substantive_progress: bool = False,
    now_ns: int,
    last_flush_ns: int,
) -> bool:
    if intro_changed or remaining_files == 0:
        return True
    elapsed_ns = max(0, now_ns - last_flush_ns)
    if semantic_substantive_progress:
        return (
            elapsed_ns >= COLLECTION_CHECKPOINT_MEANINGFUL_MIN_INTERVAL_NS
            or elapsed_ns >= COLLECTION_CHECKPOINT_FLUSH_INTERVAL_NS
        )
    return elapsed_ns >= COLLECTION_CHECKPOINT_FLUSH_INTERVAL_NS


def collection_report_flush_due(
    *,
    completed_files: int,
    remaining_files: int,
    now_ns: int,
    last_flush_ns: int,
    last_flush_completed: int,
) -> bool:
    if last_flush_completed < 0:
        return True
    if completed_files - last_flush_completed >= COLLECTION_REPORT_FLUSH_COMPLETED_STRIDE:
        return True
    if now_ns - last_flush_ns >= COLLECTION_REPORT_FLUSH_INTERVAL_NS:
        return True
    return remaining_files == 0


def projection_phase_flush_due(
    *,
    phase: Literal["collection", "forest", "edge", "post"],
    now_ns: int,
    last_flush_ns: int,
) -> bool:
    if phase == "post":
        return True
    return now_ns - last_flush_ns >= COLLECTION_REPORT_FLUSH_INTERVAL_NS


def progress_heartbeat_seconds(payload: Mapping[str, JSONValue]) -> float:
    raw = payload.get("progress_heartbeat_seconds")
    if isinstance(raw, bool):
        return DEFAULT_PROGRESS_HEARTBEAT_SECONDS
    if isinstance(raw, (int, float)):
        parsed = float(raw)
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            return DEFAULT_PROGRESS_HEARTBEAT_SECONDS
        try:
            parsed = float(text)
        except ValueError:
            return DEFAULT_PROGRESS_HEARTBEAT_SECONDS
    elif raw is None:
        return DEFAULT_PROGRESS_HEARTBEAT_SECONDS
    else:
        return DEFAULT_PROGRESS_HEARTBEAT_SECONDS
    if parsed <= 0:
        return 0.0
    if parsed < MIN_PROGRESS_HEARTBEAT_SECONDS:
        return MIN_PROGRESS_HEARTBEAT_SECONDS
    return parsed
