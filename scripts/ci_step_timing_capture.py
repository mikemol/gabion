#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_TIMING_PATH = Path("artifacts/audit_reports/ci_step_timings.json")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _default_run_id() -> str:
    env_value = os.getenv("GABION_CI_STEP_TIMING_RUN_ID")
    if env_value:
        return env_value
    return f"local-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure a CI step wall clock duration and append it to a deterministic JSON timing ledger."
        ),
    )
    parser.add_argument("--label", required=True, help="Step label (for example: checks_docflow).")
    parser.add_argument(
        "--mode",
        default="local",
        help="Execution mode bucket (for example: checks, dataflow, local).",
    )
    parser.add_argument(
        "--run-id",
        default=_default_run_id(),
        help="Timing run identifier used to group entries.",
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=DEFAULT_TIMING_PATH,
        help=f"Timing artifact JSON path (default: {DEFAULT_TIMING_PATH}).",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to execute (use -- before command).",
    )
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing command (use: ... -- <command> ...)")
    return args


def _load_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "runs": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"failed to read timing artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"timing artifact must be a JSON object: {path}")
    runs = payload.get("runs")
    if not isinstance(runs, list):
        payload["runs"] = []
    if "schema_version" not in payload:
        payload["schema_version"] = 1
    return payload


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _run_command(command: list[str]) -> tuple[int, float, str, str]:
    started_at = _now_utc()
    start = time.monotonic()
    proc = subprocess.run(command, check=False)
    elapsed = time.monotonic() - start
    ended_at = _now_utc()
    return proc.returncode, elapsed, started_at, ended_at


def _append_entry(
    *,
    payload: dict[str, Any],
    run_id: str,
    label: str,
    mode: str,
    command: list[str],
    exit_code: int,
    elapsed_seconds: float,
    started_at_utc: str,
    ended_at_utc: str,
) -> None:
    runs = payload.get("runs")
    if not isinstance(runs, list):
        runs = []
        payload["runs"] = runs

    selected_run: dict[str, Any] | None = None
    for run in runs:
        if isinstance(run, dict) and run.get("run_id") == run_id:
            selected_run = run
            break

    if selected_run is None:
        selected_run = {
            "run_id": run_id,
            "created_at_utc": _now_utc(),
            "entries": [],
        }
        runs.append(selected_run)

    entries = selected_run.get("entries")
    if not isinstance(entries, list):
        entries = []
        selected_run["entries"] = entries

    sequence = len(entries) + 1
    entries.append(
        {
            "sequence": sequence,
            "label": label,
            "mode": mode,
            "command": command,
            "command_text": shlex.join(command),
            "exit_code": exit_code,
            "elapsed_seconds": round(elapsed_seconds, 6),
            "started_at_utc": started_at_utc,
            "ended_at_utc": ended_at_utc,
        },
    )
    payload["updated_at_utc"] = _now_utc()


def main() -> int:
    args = _parse_args()
    exit_code, elapsed_seconds, started_at_utc, ended_at_utc = _run_command(args.command)

    payload = _load_payload(args.artifact_path)
    _append_entry(
        payload=payload,
        run_id=args.run_id,
        label=args.label,
        mode=args.mode,
        command=args.command,
        exit_code=exit_code,
        elapsed_seconds=elapsed_seconds,
        started_at_utc=started_at_utc,
        ended_at_utc=ended_at_utc,
    )
    _write_payload(args.artifact_path, payload)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
