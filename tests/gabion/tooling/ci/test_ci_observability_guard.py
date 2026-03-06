from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Protocol

from tests.path_helpers import SCRIPTS_ROOT


_GUARD_SCRIPT = SCRIPTS_ROOT / "ci" / "ci_observability_guard.py"


def _load_guard_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ci_observability_guard", _GUARD_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_guard(
    *,
    tmp_path: Path,
    label: str,
    max_gap_seconds: float,
    max_wall_seconds: float,
    command: list[str],
) -> subprocess.CompletedProcess[str]:
    artifact_path = tmp_path / "artifacts" / "audit_reports" / "observability_violations.json"
    args = [
        sys.executable,
        str(_GUARD_SCRIPT),
        "--label",
        label,
        "--max-gap-seconds",
        str(max_gap_seconds),
        "--max-wall-seconds",
        str(max_wall_seconds),
        "--artifact-path",
        str(artifact_path),
        "--",
        *command,
    ]
    return subprocess.run(
        args,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )


def _violations(tmp_path: Path) -> list[dict[str, object]]:
    artifact_path = tmp_path / "artifacts" / "audit_reports" / "observability_violations.json"
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    violations = payload.get("violations")
    assert isinstance(violations, list)
    return [entry for entry in violations if isinstance(entry, dict)]


class _Clock(Protocol):
    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


@dataclass
class _SyntheticEvent:
    delay_before_chunk: float
    data: bytes


class _SyntheticClock:
    def __init__(self) -> None:
        self._now = 0.0

    def monotonic(self) -> float:
        return self._now

    def sleep(self, seconds: float) -> None:
        self._now += seconds

    def advance(self, seconds: float) -> None:
        self._now += seconds


class _SyntheticChildProcess:
    def __init__(self, *, clock: _SyntheticClock, events: list[_SyntheticEvent], exit_code: int = 0) -> None:
        self._clock = clock
        self._events = events
        self._exit_code = exit_code
        self._terminated = False

    def poll(self) -> int | None:
        if self._terminated:
            return 1
        if self._events:
            return None
        return self._exit_code

    def read_chunk_if_ready(self, timeout_seconds: float) -> bytes:
        effective_timeout = timeout_seconds if timeout_seconds > 0 else 1e-6
        if self._terminated:
            self._clock.advance(effective_timeout)
            return b""
        if not self._events:
            self._clock.advance(effective_timeout)
            return b""

        next_event = self._events[0]
        if next_event.delay_before_chunk <= effective_timeout:
            self._clock.advance(next_event.delay_before_chunk)
            self._events.pop(0)
            return next_event.data

        self._clock.advance(effective_timeout)
        next_event.delay_before_chunk -= effective_timeout
        return b""

    def terminate_group(self, clock: _Clock) -> None:
        self._terminated = True

    @property
    def returncode(self) -> int:
        return 1 if self._terminated else self._exit_code


# This file intentionally keeps a narrow integration slice at the real subprocess/PTTY boundary.
# Remaining timing-heavy policy tests use synthetic clock/process injection through enforce_observability.


def test_guard_integration_fails_on_wall_timeout(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-c",
        "import time; print('start', flush=True); time.sleep(0.5)",
    ]
    result = _run_guard(
        tmp_path=tmp_path,
        label="integration_wall_case",
        max_gap_seconds=10.0,
        max_wall_seconds=0.15,
        command=command,
    )

    assert result.returncode != 0
    latest = _violations(tmp_path)[-1]
    assert latest.get("label") == "integration_wall_case"
    assert latest.get("reason") == "max_wall_timeout"


def test_guard_integration_passes_with_real_output(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import time; "
            "print('progress 1', flush=True); "
            "time.sleep(0.02); "
            "print('progress 2', flush=True); "
            "time.sleep(0.02); "
            "print('progress 3', flush=True)"
        ),
    ]
    result = _run_guard(
        tmp_path=tmp_path,
        label="integration_pass_case",
        max_gap_seconds=0.20,
        max_wall_seconds=5.0,
        command=command,
    )

    assert result.returncode == 0
    artifact_path = tmp_path / "artifacts" / "audit_reports" / "observability_violations.json"
    assert not artifact_path.exists()


def test_guard_synthetic_fails_before_first_meaningful_line(tmp_path: Path) -> None:
    guard = _load_guard_module()
    clock = _SyntheticClock()
    child = _SyntheticChildProcess(
        clock=clock,
        events=[
            _SyntheticEvent(delay_before_chunk=0.0, data=b"heartbeat tick\n"),
            _SyntheticEvent(delay_before_chunk=0.4, data=b"heartbeat tick 2\n"),
        ],
    )

    exit_code = guard.enforce_observability(
        label="first_meaningful_gap_case",
        command=["synthetic", "command"],
        max_gap_seconds=0.1,
        gap_tolerance_seconds=0.0,
        max_wall_seconds=5.0,
        artifact_path=tmp_path / "artifacts" / "audit_reports" / "observability_violations.json",
        cwd=tmp_path,
        clock=clock,
        child=child,
    )

    assert exit_code != 0
    latest = _violations(tmp_path)[-1]
    assert latest.get("reason") == "max_gap_before_first_meaningful_line"


def test_guard_synthetic_heartbeat_excluded_from_meaningful_gap(tmp_path: Path) -> None:
    guard = _load_guard_module()
    clock = _SyntheticClock()
    child = _SyntheticChildProcess(
        clock=clock,
        events=[
            _SyntheticEvent(delay_before_chunk=0.0, data=b"progress start\n"),
            _SyntheticEvent(delay_before_chunk=0.08, data=b"heartbeat tick 1\n"),
            _SyntheticEvent(delay_before_chunk=0.08, data=b"heartbeat tick 2\n"),
            _SyntheticEvent(delay_before_chunk=0.08, data=b"heartbeat tick 3\n"),
        ],
    )

    exit_code = guard.enforce_observability(
        label="heartbeat_case",
        command=["synthetic", "command"],
        max_gap_seconds=0.1,
        gap_tolerance_seconds=0.0,
        max_wall_seconds=5.0,
        artifact_path=tmp_path / "artifacts" / "audit_reports" / "observability_violations.json",
        cwd=tmp_path,
        clock=clock,
        child=child,
    )

    assert exit_code != 0
    latest = _violations(tmp_path)[-1]
    assert latest.get("reason") == "max_gap_meaningful_line_exceeded"


def test_guard_synthetic_terminal_progress_bypasses_gap_enforcement(tmp_path: Path) -> None:
    guard = _load_guard_module()
    clock = _SyntheticClock()
    child = _SyntheticChildProcess(
        clock=clock,
        events=[
            _SyntheticEvent(delay_before_chunk=0.0, data=b"| ts | progress | post | complete |\n"),
            _SyntheticEvent(delay_before_chunk=0.8, data=b"done\n"),
        ],
    )

    exit_code = guard.enforce_observability(
        label="terminal_cleanup_case",
        command=["synthetic", "command"],
        max_gap_seconds=0.1,
        gap_tolerance_seconds=0.0,
        max_wall_seconds=5.0,
        artifact_path=tmp_path / "artifacts" / "audit_reports" / "observability_violations.json",
        cwd=tmp_path,
        clock=clock,
        child=child,
    )

    assert exit_code == 0
    artifact_path = tmp_path / "artifacts" / "audit_reports" / "observability_violations.json"
    assert not artifact_path.exists()


def test_guard_synthetic_chunk_timing_for_partial_and_multiline_chunks(tmp_path: Path) -> None:
    guard = _load_guard_module()
    clock = _SyntheticClock()
    child = _SyntheticChildProcess(
        clock=clock,
        events=[
            _SyntheticEvent(delay_before_chunk=0.0, data=b"progress "),
            _SyntheticEvent(delay_before_chunk=0.03, data=b"part"),
            _SyntheticEvent(delay_before_chunk=0.03, data=b"ial\nprogress 2\nprogress 3\n"),
        ],
    )

    exit_code = guard.enforce_observability(
        label="chunk_and_multiline_case",
        command=["synthetic", "command"],
        max_gap_seconds=0.2,
        gap_tolerance_seconds=0.0,
        max_wall_seconds=5.0,
        artifact_path=tmp_path / "artifacts" / "audit_reports" / "observability_violations.json",
        cwd=tmp_path,
        clock=clock,
        child=child,
    )

    assert exit_code == 0
    artifact_path = tmp_path / "artifacts" / "audit_reports" / "observability_violations.json"
    assert not artifact_path.exists()
