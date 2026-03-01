#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from gabion.tooling import terminal_outcome_projector

def _emit(path: Path, label: str) -> None:
    if not path.exists():
        return
    print(f"===== {label} =====")
    print(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal-exit", required=True)
    parser.add_argument("--terminal-state", default="none")
    parser.add_argument("--terminal-stage", default="none")
    parser.add_argument("--terminal-status", default="unknown")
    parser.add_argument("--attempts-run", default="0")
    parser.add_argument(
        "--terminal-outcome-json",
        default="artifacts/audit_reports/dataflow_terminal_outcome.json",
    )
    args = parser.parse_args()

    outcome_path = Path(args.terminal_outcome_json)
    outcome = terminal_outcome_projector.read_terminal_outcome_artifact(outcome_path)
    if outcome is None:
        outcome = terminal_outcome_projector.project_terminal_outcome(
            terminal_outcome_projector.TerminalOutcomeInput(
                terminal_exit=int(args.terminal_exit),
                terminal_state=str(args.terminal_state or "none"),
                terminal_stage=str(args.terminal_stage or "none"),
                terminal_status=str(args.terminal_status or "unknown"),
                attempts_run=int(args.attempts_run),
            )
        )
    print(
        terminal_outcome_projector.render_terminal_outcome_line(outcome)
    )
    if outcome.terminal_status == "success":
        return 0
    _emit(Path("artifacts/audit_reports/dataflow_report.md"), "dataflow report")
    if outcome.terminal_status == "timeout_resume":
        print("Dataflow audit invocation timed out with resumable progress.")
        return 1
    print("Dataflow audit failed for a non-timeout reason.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
