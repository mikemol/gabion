# gabion:decision_protocol_module
from __future__ import annotations

from pathlib import Path

DEFAULT_CHECK_REPORT_REL_PATH = Path("artifacts/audit_reports/dataflow_report.md")
DEFAULT_TIMEOUT_PROGRESS_REPORT_REL_PATH = Path("artifacts/audit_reports/timeout_progress.md")
DEFAULT_CI_RESUME_CHECKPOINT_REL_PATH = Path(
    "artifacts/audit_reports/dataflow_resume_checkpoint_ci.json"
)
DEFAULT_LOCAL_RESUME_CHECKPOINT_REL_PATH = Path(
    "artifacts/audit_reports/dataflow_resume_checkpoint_local.json"
)
DEFAULT_PHASE_TIMELINE_MD_REL_PATH = Path("artifacts/audit_reports/dataflow_phase_timeline.md")
DEFAULT_PHASE_TIMELINE_JSONL_REL_PATH = Path("artifacts/audit_reports/dataflow_phase_timeline.jsonl")


def resolve_report_path(path: Path | None, *, root: Path) -> Path:
    if path is not None:
        return path
    return root / DEFAULT_CHECK_REPORT_REL_PATH
