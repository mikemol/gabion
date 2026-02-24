#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


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
    args = parser.parse_args()

    terminal_status = args.terminal_status
    if terminal_status == "unknown":
        if args.terminal_exit == "0":
            terminal_status = "success"
        elif args.terminal_state == "timed_out_progress_resume":
            terminal_status = "timeout_resume"
        else:
            terminal_status = "hard_failure"
    print(
        "terminal_stage="
        f"{args.terminal_stage} attempts={args.attempts_run} "
        f"exit_code={args.terminal_exit} analysis_state={args.terminal_state} status={terminal_status}"
    )
    if terminal_status == "success":
        return 0
    _emit(Path("artifacts/audit_reports/dataflow_report.md"), "dataflow report")
    _emit(Path("artifacts/audit_reports/timeout_progress.md"), "timeout progress")
    if terminal_status == "timeout_resume":
        print("Dataflow audit invocation timed out with resumable progress.")
        return 1
    print("Dataflow audit failed for a non-timeout reason.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
