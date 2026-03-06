#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pty
import select
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

DEFAULT_MAX_GAP_SECONDS = float(os.getenv("GABION_OBSERVABILITY_MAX_GAP_SECONDS", "5"))
DEFAULT_MAX_WALL_SECONDS = float(os.getenv("GABION_OBSERVABILITY_MAX_WALL_SECONDS", "1200"))
DEFAULT_GAP_TOLERANCE_SECONDS = float(
    os.getenv("GABION_OBSERVABILITY_GAP_TOLERANCE_SECONDS", "0.1")
)
DEFAULT_ARTIFACT_PATH = Path("artifacts/audit_reports/observability_violations.json")


class Clock(Protocol):
    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class ChildProcess(Protocol):
    def poll(self) -> int | None: ...

    def read_chunk_if_ready(self, timeout_seconds: float) -> bytes: ...

    def terminate_group(self, clock: Clock) -> None: ...

    @property
    def returncode(self) -> int | None: ...


class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class PtyChildProcess:
    def __init__(self, *, command: list[str], cwd: Path) -> None:
        master, slave = pty.openpty()
        self._master = master
        self._proc = subprocess.Popen(
            command,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            cwd=str(cwd),
            close_fds=True,
            start_new_session=True,
        )
        os.close(slave)

    def poll(self) -> int | None:
        return self._proc.poll()

    def read_chunk_if_ready(self, timeout_seconds: float) -> bytes:
        rlist, _, _ = select.select([self._master], [], [], timeout_seconds)
        if not rlist:
            return b""
        try:
            return os.read(self._master, 8192)
        except OSError:
            return b""

    def terminate_group(self, clock: Clock) -> None:
        if self._proc.poll() is not None:
            return
        try:
            os.killpg(self._proc.pid, signal.SIGTERM)
        except OSError:
            return
        deadline = clock.monotonic() + 2.0
        while self._proc.poll() is None and clock.monotonic() < deadline:
            clock.sleep(0.05)
        if self._proc.poll() is None:
            try:
                os.killpg(self._proc.pid, signal.SIGKILL)
            except OSError:
                pass

    @property
    def returncode(self) -> int | None:
        return self._proc.returncode


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a command in a PTY with live output and enforce observability gaps "
            "for meaningful non-heartbeat progress lines."
        ),
    )
    parser.add_argument("--label", required=True, help="Step label for violation artifacts.")
    parser.add_argument(
        "--max-gap-seconds",
        type=float,
        default=DEFAULT_MAX_GAP_SECONDS,
        help="Maximum allowed silence between meaningful non-heartbeat lines.",
    )
    parser.add_argument(
        "--gap-tolerance-seconds",
        type=float,
        default=DEFAULT_GAP_TOLERANCE_SECONDS,
        help="Clock/scheduling tolerance applied to gap enforcement.",
    )
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=DEFAULT_MAX_WALL_SECONDS,
        help="Maximum allowed wall time before timeout failure.",
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=DEFAULT_ARTIFACT_PATH,
        help="JSON artifact path used to append observability violations.",
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        default=None,
        help="Optional working directory for the wrapped command.",
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


def _is_meaningful_text(fragment: str) -> bool:
    text = fragment.strip()
    if not text:
        return False
    lowered = text.lower()
    if "heartbeat" in lowered:
        return False
    return True


def _first_meaningful_fragment(text: str) -> str | None:
    for fragment in _normalize_chunk_for_lines(text).split("\n"):
        if _is_meaningful_text(fragment):
            return fragment.strip()
    return None


def _is_terminal_progress_line(text: str) -> bool:
    lowered = text.lower()
    if "| post |" in lowered and "| complete |" in lowered:
        return True
    if "classification=succeeded" in lowered and "| post |" in lowered:
        return True
    return False


def _normalize_chunk_for_lines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _parse_deadline_profile(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return {
        "checks_total": payload.get("checks_total"),
        "ticks_consumed": payload.get("ticks_consumed"),
        "wall_total_elapsed_ns": payload.get("wall_total_elapsed_ns"),
    }


def _last_timeline_row(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return None
    for line in reversed(lines):
        if line.startswith("| ") and not line.startswith("| ---") and not line.startswith("| ts_utc |"):
            return line
    return None


def _signal_run_dataflow_stage() -> list[int]:
    sigusr1 = getattr(signal, "SIGUSR1", None)
    if sigusr1 is None:
        return []
    try:
        proc = subprocess.run(
            ["ps", "-eo", "pid=,command="],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return []
    signaled: list[int] = []
    for raw_line in proc.stdout.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2:
            continue
        raw_pid, command = parts
        if "run-dataflow-stage" not in command:
            continue
        try:
            pid = int(raw_pid)
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, sigusr1)
        except OSError:
            continue
        signaled.append(pid)
    return signaled


def _collect_violation_sample(*, cwd: Path) -> dict[str, Any]:
    audit_root = cwd / "artifacts" / "audit_reports"
    return {
        "deadline_profile": _parse_deadline_profile(audit_root / "deadline_profile.json"),
        "phase_timeline_last_row": _last_timeline_row(audit_root / "dataflow_phase_timeline.md"),
        "signaled_run_dataflow_stage_pids": _signal_run_dataflow_stage(),
    }


def _append_violation_artifact(path: Path, violation: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"violations": []}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict):
            raw_violations = existing.get("violations")
            if isinstance(raw_violations, list):
                payload["violations"] = [entry for entry in raw_violations if isinstance(entry, dict)]
    payload["violations"].append(violation)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _violation_payload(
    *,
    label: str,
    command: list[str],
    reason: str,
    wall_seconds: float,
    max_gap_seconds: float,
    measured_gap_seconds: float,
    previous_line: str | None,
    next_line: str | None,
    cwd: Path,
) -> dict[str, Any]:
    # dataflow-bundle: max_gap_seconds, measured_gap_seconds, wall_seconds
    return {
        "ts_utc": _now_utc(),
        "label": label,
        "reason": reason,
        "command": command,
        "command_text": shlex.join(command),
        "wall_seconds": round(wall_seconds, 3),
        "max_gap_seconds": round(max_gap_seconds, 3),
        "measured_gap_seconds": round(measured_gap_seconds, 3),
        "previous_line": previous_line or "",
        "next_line": next_line or "",
        "sample": _collect_violation_sample(cwd=cwd),
    }


@dataclass
class _ObservabilityState:
    started: float
    buffer: str = ""
    last_any_ts: float | None = None
    last_any_line: str | None = None
    last_meaningful_ts: float | None = None
    last_meaningful_line: str | None = None
    max_gap_meaningful: float = 0.0
    violation: dict[str, Any] | None = None
    terminal_progress_seen: bool = False


def _record_meaningful_event(
    *,
    state: _ObservabilityState,
    event_ts: float,
    event_text: str,
    label: str,
    command: list[str],
    max_gap_seconds: float,
    gap_tolerance_seconds: float,
    artifact_path: Path,
    cwd: Path,
    child: ChildProcess,
    clock: Clock,
) -> None:
    if state.last_meaningful_ts is not None:
        state.max_gap_meaningful = max(state.max_gap_meaningful, event_ts - state.last_meaningful_ts)
    state.last_meaningful_ts = event_ts
    state.last_meaningful_line = event_text
    if _is_terminal_progress_line(event_text):
        state.terminal_progress_seen = True
    if (
        not state.terminal_progress_seen
        and max_gap_seconds > 0
        and state.max_gap_meaningful > (max_gap_seconds + gap_tolerance_seconds)
        and state.violation is None
    ):
        state.violation = _violation_payload(
            label=label,
            command=command,
            reason="max_gap_meaningful_line_exceeded",
            wall_seconds=event_ts - state.started,
            max_gap_seconds=max_gap_seconds,
            measured_gap_seconds=state.max_gap_meaningful,
            previous_line=state.last_any_line,
            next_line=event_text,
            cwd=cwd,
        )
        _append_violation_artifact(artifact_path, state.violation)
        child.terminate_group(clock)


def _record_wall_timeout(
    *,
    state: _ObservabilityState,
    now_ts: float,
    label: str,
    command: list[str],
    max_gap_seconds: float,
    max_wall_seconds: float,
    artifact_path: Path,
    cwd: Path,
    child: ChildProcess,
    clock: Clock,
) -> None:
    if max_wall_seconds <= 0 or state.violation is not None:
        return
    wall_seconds = now_ts - state.started
    if wall_seconds <= max_wall_seconds:
        return
    state.violation = _violation_payload(
        label=label,
        command=command,
        reason="max_wall_timeout",
        wall_seconds=wall_seconds,
        max_gap_seconds=max_gap_seconds,
        measured_gap_seconds=wall_seconds,
        previous_line=state.last_meaningful_line,
        next_line=None,
        cwd=cwd,
    )
    _append_violation_artifact(artifact_path, state.violation)
    child.terminate_group(clock)


def enforce_observability(
    *,
    label: str,
    command: list[str],
    max_gap_seconds: float,
    gap_tolerance_seconds: float,
    max_wall_seconds: float,
    artifact_path: Path,
    cwd: Path,
    clock: Clock,
    child: ChildProcess,
) -> int:
    state = _ObservabilityState(started=clock.monotonic())

    while True:
        now = clock.monotonic()
        _record_wall_timeout(
            state=state,
            now_ts=now,
            label=label,
            command=command,
            max_gap_seconds=max_gap_seconds,
            max_wall_seconds=max_wall_seconds,
            artifact_path=artifact_path,
            cwd=cwd,
            child=child,
            clock=clock,
        )
        select_timeout = 0.5
        if max_gap_seconds > 0 and state.violation is None and not state.terminal_progress_seen:
            reference_ts = state.last_meaningful_ts
            if reference_ts is None:
                reference_ts = state.last_any_ts
            if reference_ts is not None:
                remaining = (max_gap_seconds + gap_tolerance_seconds) - (now - reference_ts)
                select_timeout = max(0.0, min(select_timeout, remaining))

        chunk = child.read_chunk_if_ready(select_timeout)
        if chunk:
            text = chunk.decode("utf-8", errors="replace")
            sys.stdout.write(text)
            sys.stdout.flush()
            chunk_ts = clock.monotonic()
            state.last_any_ts = chunk_ts
            meaningful_fragment = _first_meaningful_fragment(text)
            if meaningful_fragment is not None:
                _record_meaningful_event(
                    state=state,
                    event_ts=chunk_ts,
                    event_text=meaningful_fragment,
                    label=label,
                    command=command,
                    max_gap_seconds=max_gap_seconds,
                    gap_tolerance_seconds=gap_tolerance_seconds,
                    artifact_path=artifact_path,
                    cwd=cwd,
                    child=child,
                    clock=clock,
                )
            state.buffer += _normalize_chunk_for_lines(text)
            while "\n" in state.buffer:
                line, state.buffer = state.buffer.split("\n", 1)
                line_ts = clock.monotonic()
                state.last_any_ts = line_ts
                state.last_any_line = line
                if _is_meaningful_text(line):
                    _record_meaningful_event(
                        state=state,
                        event_ts=line_ts,
                        event_text=line,
                        label=label,
                        command=command,
                        max_gap_seconds=max_gap_seconds,
                        gap_tolerance_seconds=gap_tolerance_seconds,
                        artifact_path=artifact_path,
                        cwd=cwd,
                        child=child,
                        clock=clock,
                    )

        if max_gap_seconds > 0 and state.violation is None and not state.terminal_progress_seen:
            now_after = clock.monotonic()
            if state.last_meaningful_ts is None:
                if (
                    state.last_any_ts is not None
                    and (now_after - state.last_any_ts) > (max_gap_seconds + gap_tolerance_seconds)
                ):
                    state.violation = _violation_payload(
                        label=label,
                        command=command,
                        reason="max_gap_before_first_meaningful_line",
                        wall_seconds=now_after - state.started,
                        max_gap_seconds=max_gap_seconds,
                        measured_gap_seconds=(now_after - state.last_any_ts),
                        previous_line=None,
                        next_line=None,
                        cwd=cwd,
                    )
                    _append_violation_artifact(artifact_path, state.violation)
                    child.terminate_group(clock)
            else:
                observed_gap = now_after - state.last_meaningful_ts
                if observed_gap > (max_gap_seconds + gap_tolerance_seconds):
                    state.violation = _violation_payload(
                        label=label,
                        command=command,
                        reason="max_gap_meaningful_line_exceeded",
                        wall_seconds=now_after - state.started,
                        max_gap_seconds=max_gap_seconds,
                        measured_gap_seconds=observed_gap,
                        previous_line=state.last_meaningful_line,
                        next_line=None,
                        cwd=cwd,
                    )
                    _append_violation_artifact(artifact_path, state.violation)
                    child.terminate_group(clock)

        _record_wall_timeout(
            state=state,
            now_ts=clock.monotonic(),
            label=label,
            command=command,
            max_gap_seconds=max_gap_seconds,
            max_wall_seconds=max_wall_seconds,
            artifact_path=artifact_path,
            cwd=cwd,
            child=child,
            clock=clock,
        )
        if child.poll() is not None:
            break

    if state.buffer:
        sys.stdout.write(state.buffer)
        sys.stdout.flush()

    exit_code = int(child.returncode or 0)
    if state.violation is not None:
        return 2 if exit_code == 0 else exit_code
    return exit_code


def main() -> int:
    args = _parse_args()
    max_gap_seconds = max(0.0, float(args.max_gap_seconds))
    gap_tolerance_seconds = max(0.0, float(args.gap_tolerance_seconds))
    max_wall_seconds = max(0.0, float(args.max_wall_seconds))
    cwd = args.cwd.resolve() if isinstance(args.cwd, Path) else Path.cwd()
    command = [str(token) for token in args.command]

    clock = SystemClock()
    child = PtyChildProcess(command=command, cwd=cwd)
    return enforce_observability(
        label=str(args.label),
        command=command,
        max_gap_seconds=max_gap_seconds,
        gap_tolerance_seconds=gap_tolerance_seconds,
        max_wall_seconds=max_wall_seconds,
        artifact_path=args.artifact_path,
        cwd=cwd,
        clock=clock,
        child=child,
    )


if __name__ == "__main__":
    raise SystemExit(main())
