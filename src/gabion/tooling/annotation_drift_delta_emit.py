from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Mapping

from gabion import server
from gabion.lsp_client import CommandRequest, run_command_direct

_DEFAULT_TIMEOUT_TICKS = "65000000"
_DEFAULT_TIMEOUT_TICK_NS = "1000000"
_DEFAULT_RESUME_CHECKPOINT_PATH = Path(
    "artifacts/audit_reports/dataflow_resume_checkpoint_ci.json"
)
_STATE_PATH = Path("artifacts/out/test_annotation_drift.json")
_DELTA_PATH = Path("artifacts/out/test_annotation_drift_delta.json")


def _timeout_ticks() -> int:
    raw = os.getenv("GABION_LSP_TIMEOUT_TICKS", _DEFAULT_TIMEOUT_TICKS)
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return int(_DEFAULT_TIMEOUT_TICKS)
    return parsed if parsed > 0 else int(_DEFAULT_TIMEOUT_TICKS)


def _timeout_tick_ns() -> int:
    raw = os.getenv("GABION_LSP_TIMEOUT_TICK_NS", _DEFAULT_TIMEOUT_TICK_NS)
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return int(_DEFAULT_TIMEOUT_TICK_NS)
    return parsed if parsed > 0 else int(_DEFAULT_TIMEOUT_TICK_NS)


def _build_payload(
    *,
    resume_checkpoint: Path | bool | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "analysis_timeout_ticks": _timeout_ticks(),
        "analysis_timeout_tick_ns": _timeout_tick_ns(),
        "fail_on_violations": False,
        "fail_on_type_ambiguities": False,
        "resume_on_timeout": 1,
        "emit_timeout_progress_report": True,
        "emit_test_annotation_drift_delta": True,
    }
    if _STATE_PATH.exists():
        payload["test_annotation_drift_state"] = str(_STATE_PATH)
    if resume_checkpoint is False:
        payload["resume_checkpoint"] = False
    elif isinstance(resume_checkpoint, Path):
        payload["resume_checkpoint"] = str(resume_checkpoint)
    elif _DEFAULT_RESUME_CHECKPOINT_PATH.exists():
        payload["resume_checkpoint"] = str(_DEFAULT_RESUME_CHECKPOINT_PATH)
    else:
        payload["resume_checkpoint"] = False
    return payload


def main(
    *,
    run_command_direct_fn: Callable[..., Mapping[str, object]] = run_command_direct,
    root_path: Path = Path("."),
    delta_path: Path = _DELTA_PATH,
    resume_checkpoint: Path | bool | None = None,
) -> int:
    result = run_command_direct_fn(
        CommandRequest(
            server.DATAFLOW_COMMAND,
            [_build_payload(resume_checkpoint=resume_checkpoint)],
        ),
        root=root_path,
    )
    exit_code = int(result.get("exit_code", 0))
    if exit_code != 0:
        print(f"Annotation drift delta emit failed (exit {exit_code}).")
        return exit_code
    if not delta_path.exists():
        print(f"Annotation drift delta emit failed: missing output {delta_path}.")
        return 1
    return 0

