# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal, Mapping

from gabion.analysis.timeout_context import check_deadline, deadline_loop_iter
from gabion.invariants import never
from gabion.json_types import JSONObject

TerminalStatus = Literal["success", "timeout_resume", "hard_failure"]


@dataclass(frozen=True)
class TerminalOutcomeInput:
    terminal_exit: int
    terminal_state: str
    terminal_stage: str
    terminal_status: str = "unknown"
    attempts_run: int = 0


@dataclass(frozen=True)
class TerminalOutcome:
    terminal_exit: int
    terminal_state: str
    terminal_stage: str
    terminal_status: TerminalStatus
    attempts_run: int

    def to_payload(self) -> JSONObject:
        return {
            "terminal_exit": self.terminal_exit,
            "terminal_state": self.terminal_state,
            "terminal_stage": self.terminal_stage,
            "terminal_status": self.terminal_status,
            "attempts_run": self.attempts_run,
        }

    def to_output_lines(self, *, stage_metrics: str | None = None) -> list[str]:
        lines = [
            f"attempts_run={self.attempts_run}",
            f"terminal_stage={self.terminal_stage}",
            f"terminal_status={self.terminal_status}",
            f"exit_code={self.terminal_exit}",
            f"analysis_state={self.terminal_state}",
        ]
        if isinstance(stage_metrics, str) and stage_metrics:
            lines.append(f"stage_metrics={stage_metrics}")
        return lines


# gabion:decision_protocol
def project_terminal_outcome(
    outcome_in: TerminalOutcomeInput,
) -> TerminalOutcome:
    status = _normalize_terminal_status(
        terminal_status=outcome_in.terminal_status,
        terminal_exit=int(outcome_in.terminal_exit),
        terminal_state=str(outcome_in.terminal_state or "none"),
    )
    return TerminalOutcome(
        terminal_exit=int(outcome_in.terminal_exit),
        terminal_state=str(outcome_in.terminal_state or "none"),
        terminal_stage=str(outcome_in.terminal_stage or "none").upper(),
        terminal_status=status,
        attempts_run=max(0, int(outcome_in.attempts_run)),
    )


# gabion:decision_protocol
def terminal_outcome_from_stage_results(
    results: list[Mapping[str, object]] | tuple[Mapping[str, object], ...],
) -> TerminalOutcome | None:
    if not results:
        return None
    terminal = results[-1]
    stage_id = str(terminal.get("stage_id", "") or "none")
    exit_code = int(terminal.get("exit_code", 0) or 0)
    analysis_state = str(terminal.get("analysis_state", "") or "none")
    return project_terminal_outcome(
        TerminalOutcomeInput(
            terminal_exit=exit_code,
            terminal_state=analysis_state,
            terminal_stage=stage_id,
            terminal_status="unknown",
            attempts_run=len(results),
        )
    )


# gabion:boundary_normalization
def read_terminal_outcome_artifact(path: Path) -> TerminalOutcome | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return None
    return project_terminal_outcome(
        TerminalOutcomeInput(
            terminal_exit=int(payload.get("terminal_exit", 0) or 0),
            terminal_state=str(payload.get("terminal_state", "") or "none"),
            terminal_stage=str(payload.get("terminal_stage", "") or "none"),
            terminal_status=str(payload.get("terminal_status", "") or "unknown"),
            attempts_run=int(payload.get("attempts_run", 0) or 0),
        )
    )


# gabion:decision_protocol
def write_terminal_outcome_artifact(path: Path, outcome: TerminalOutcome) -> None:
    check_deadline()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = outcome.to_payload()
    payload["format_version"] = 1
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


# gabion:decision_protocol
def render_terminal_outcome_line(outcome: TerminalOutcome) -> str:
    return (
        "terminal_stage="
        f"{outcome.terminal_stage} attempts={outcome.attempts_run} "
        f"exit_code={outcome.terminal_exit} "
        f"analysis_state={outcome.terminal_state} "
        f"status={outcome.terminal_status}"
    )


def _normalize_terminal_status(
    *,
    terminal_status: str,
    terminal_exit: int,
    terminal_state: str,
) -> TerminalStatus:
    normalized = terminal_status.strip().lower()
    explicit: dict[str, TerminalStatus] = {
        "success": "success",
        "timeout_resume": "timeout_resume",
        "hard_failure": "hard_failure",
    }
    if normalized in explicit:
        return explicit[normalized]
    if normalized not in {"", "unknown", "none"}:
        never("invalid terminal status token", terminal_status=terminal_status)
    if terminal_exit == 0:
        return "success"
    if terminal_state == "timed_out_progress_resume":
        return "timeout_resume"
    return "hard_failure"


__all__ = [
    "TerminalOutcome",
    "TerminalOutcomeInput",
    "TerminalStatus",
    "project_terminal_outcome",
    "terminal_outcome_from_stage_results",
    "render_terminal_outcome_line",
    "read_terminal_outcome_artifact",
    "write_terminal_outcome_artifact",
]
