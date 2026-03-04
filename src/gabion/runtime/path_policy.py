# gabion:decision_protocol_module
from __future__ import annotations

from pathlib import Path

DEFAULT_CHECK_REPORT_REL_PATH = Path("artifacts/audit_reports/dataflow_report.md")
DEFAULT_PHASE_TIMELINE_MD_REL_PATH = Path("artifacts/audit_reports/dataflow_phase_timeline.md")
DEFAULT_PHASE_TIMELINE_JSONL_REL_PATH = Path("artifacts/audit_reports/dataflow_phase_timeline.jsonl")
STDOUT_ALIAS = "-"
STDOUT_PATH = "/dev/stdout"


def resolve_report_path(path: Path | None, *, root: Path) -> Path:
    if path is not None:
        return path
    return root / DEFAULT_CHECK_REPORT_REL_PATH


def normalize_output_target(
    target: str | Path,
    *,
    stdout_alias: str = STDOUT_ALIAS,
    stdout_path: str = STDOUT_PATH,
) -> str:
    text = str(target).strip()
    if text == stdout_alias:
        return stdout_path
    return text


def is_stdout_target(
    target: object,
    *,
    stdout_alias: str = STDOUT_ALIAS,
    stdout_path: str = STDOUT_PATH,
) -> bool:
    if not isinstance(target, (str, Path)):
        return False
    return normalize_output_target(
        target,
        stdout_alias=stdout_alias,
        stdout_path=stdout_path,
    ) == stdout_path
