#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Mapping
from xml.etree import ElementTree


_SCHEMA_VERSION = 1
_ARTIFACT_KIND = "delivery_flow_summary"
_GENERATED_BY = "gabion governance delivery-flow-emit"
_SEVERE_RUNTIME_RATIO = 1.2
_SEVERE_RUNTIME_DELTA_SECONDS = 30.0
_RED_STATE_DWELL_RUNS = 2
_CLOSURE_LAG_RUNS = 2
_UNSTABLE_LOOKBACK_RUNS = 4


@dataclass(frozen=True)
class CurrentSummary:
    suite_red_state: bool
    failing_test_case_count: int
    test_failure_count: int
    local_ci_failed_surface_ids: tuple[str, ...]
    local_ci_failed_relation_ids: tuple[str, ...]
    observability_violation_ids: tuple[str, ...]
    severe_runtime_regression_current_band: bool
    repeat_blocker_ids: tuple[str, ...]
    stalled_blocker_runs_by_id: Mapping[str, int]
    unstable_blocker_ids: tuple[str, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "suite_red_state": self.suite_red_state,
            "failing_test_case_count": self.failing_test_case_count,
            "test_failure_count": self.test_failure_count,
            "local_ci_failed_surface_ids": list(self.local_ci_failed_surface_ids),
            "local_ci_failed_relation_ids": list(self.local_ci_failed_relation_ids),
            "observability_violation_ids": list(self.observability_violation_ids),
            "severe_runtime_regression_current_band": self.severe_runtime_regression_current_band,
            "repeat_blocker_ids": list(self.repeat_blocker_ids),
            "stalled_blocker_runs_by_id": dict(self.stalled_blocker_runs_by_id),
            "unstable_blocker_ids": list(self.unstable_blocker_ids),
        }


@dataclass(frozen=True)
class TrendSummary:
    latest_total_runtime_seconds: float
    baseline_total_runtime_seconds: float | None
    runtime_regression_ratio: float | None
    runtime_delta_seconds: float | None
    red_state_dwell_runs: int
    recurring_loop_ids: tuple[str, ...]
    closure_lag_loop_ids: tuple[str, ...]
    max_time_to_correction_runs: int | None

    def as_payload(self) -> dict[str, object]:
        return {
            "latest_total_runtime_seconds": self.latest_total_runtime_seconds,
            "baseline_total_runtime_seconds": self.baseline_total_runtime_seconds,
            "runtime_regression_ratio": self.runtime_regression_ratio,
            "runtime_delta_seconds": self.runtime_delta_seconds,
            "red_state_dwell_runs": self.red_state_dwell_runs,
            "recurring_loop_ids": list(self.recurring_loop_ids),
            "closure_lag_loop_ids": list(self.closure_lag_loop_ids),
            "max_time_to_correction_runs": self.max_time_to_correction_runs,
        }


@dataclass(frozen=True)
class HistoryRow:
    run_id: str
    suite_red_state: bool
    open_blocker_ids: tuple[str, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "suite_red_state": self.suite_red_state,
            "open_blocker_ids": list(self.open_blocker_ids),
        }


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Emit a canonical delivery-flow summary artifact from current CI signals and recent governance telemetry."
        )
    )
    parser.add_argument("--run-id", default="", help="Stable run identifier.")
    parser.add_argument(
        "--history-window-runs",
        type=int,
        default=10,
        help="Maximum number of summary history rows to retain.",
    )
    parser.add_argument(
        "--junit",
        type=Path,
        default=Path("artifacts/test_runs/junit.xml"),
    )
    parser.add_argument(
        "--local-ci-contract",
        type=Path,
        default=Path("artifacts/out/local_ci_repro_contract.json"),
    )
    parser.add_argument(
        "--observability",
        type=Path,
        default=Path("artifacts/audit_reports/observability_violations.json"),
    )
    parser.add_argument(
        "--telemetry-history",
        type=Path,
        default=Path("artifacts/out/governance_telemetry_history.json"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("artifacts/out/delivery_flow_summary.json"),
    )
    return parser.parse_args(argv)


def _load_json_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, Mapping):
        return {}
    return {str(key): payload[key] for key in payload}


def _nested_int(payload: Mapping[str, Any], *keys: str) -> int:
    node: Any = payload
    for key in keys:
        if not isinstance(node, Mapping):
            return 0
        node = node.get(key)
    try:
        return int(node if node is not None else 0)
    except (TypeError, ValueError):
        return 0


def _nested_float(payload: Mapping[str, Any], *keys: str) -> float:
    node: Any = payload
    for key in keys:
        if not isinstance(node, Mapping):
            return 0.0
        node = node.get(key)
    try:
        return float(node if node is not None else 0.0)
    except (TypeError, ValueError):
        return 0.0


def _normalize_rel_path(raw_path: object) -> str:
    if raw_path is None:
        return ""
    text = str(raw_path).strip()
    if not text:
        return ""
    return text.replace("\\", "/")


def _rel_path_from_pytest_classname(classname: str) -> str:
    parts = [part.strip() for part in classname.split(".") if part.strip()]
    if not parts:
        return ""
    last_test_index = max(
        (index for index, part in enumerate(parts) if part.startswith("test_")),
        default=-1,
    )
    if last_test_index < 0:
        return ""
    return "/".join(parts[: last_test_index + 1]) + ".py"


def _load_failing_test_ids(path: Path) -> tuple[str, ...]:
    if not path.exists():
        return ()
    try:
        tree = ElementTree.parse(path)
    except ElementTree.ParseError:
        return ()
    failures: list[str] = []
    for index, testcase in enumerate(tree.iterfind(".//testcase"), start=1):
        failure_like = next(
            (child for child in testcase if child.tag in {"failure", "error"}),
            None,
        )
        if failure_like is None:
            continue
        raw_name = str(testcase.attrib.get("name", "")).strip()
        if not raw_name:
            continue
        classname = str(testcase.attrib.get("classname", "")).strip()
        rel_test_path = _normalize_rel_path(testcase.attrib.get("file"))
        if not rel_test_path and classname:
            rel_test_path = _rel_path_from_pytest_classname(classname)
        test_id = raw_name
        if rel_test_path:
            class_suffix = classname.split(".")[-1].strip() if classname else ""
            if class_suffix and class_suffix != raw_name:
                test_id = f"{rel_test_path}::{class_suffix}::{raw_name}"
            else:
                test_id = f"{rel_test_path}::{raw_name}"
        failures.append(test_id or f"testcase-{index}")
    return tuple(sorted(dict.fromkeys(failures)))


def _load_local_ci_failures(path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    payload = _load_json_mapping(path)
    surface_ids: list[str] = []
    raw_surfaces = payload.get("surfaces")
    if isinstance(raw_surfaces, list):
        for raw_surface in raw_surfaces:
            if not isinstance(raw_surface, Mapping):
                continue
            if str(raw_surface.get("status", "")).strip() == "pass":
                continue
            surface_id = str(raw_surface.get("surface_id", "")).strip()
            if surface_id:
                surface_ids.append(surface_id)
    relation_ids: list[str] = []
    raw_relations = payload.get("relations")
    if isinstance(raw_relations, list):
        for raw_relation in raw_relations:
            if not isinstance(raw_relation, Mapping):
                continue
            if str(raw_relation.get("status", "")).strip() == "pass":
                continue
            relation_id = str(raw_relation.get("relation_id", "")).strip()
            if relation_id:
                relation_ids.append(relation_id)
    return (
        tuple(sorted(dict.fromkeys(surface_ids))),
        tuple(sorted(dict.fromkeys(relation_ids))),
    )


def _load_observability_violation_ids(path: Path) -> tuple[str, ...]:
    payload = _load_json_mapping(path)
    raw_violations = payload.get("violations")
    if not isinstance(raw_violations, list):
        return ()
    ids: list[str] = []
    for index, raw_violation in enumerate(raw_violations, start=1):
        if not isinstance(raw_violation, Mapping):
            continue
        label = str(raw_violation.get("label", "")).strip()
        reason = str(raw_violation.get("reason", "")).strip()
        ids.append(label or reason or f"violation-{index}")
    return tuple(sorted(dict.fromkeys(ids)))


def _load_telemetry_runs(path: Path) -> list[dict[str, Any]]:
    payload = _load_json_mapping(path)
    raw_runs = payload.get("runs")
    if not isinstance(raw_runs, list):
        return []
    return [
        {str(key): run[key] for key in run}
        for run in raw_runs
        if isinstance(run, Mapping)
    ]


def _runtime_total_seconds(raw_run: Mapping[str, Any]) -> float:
    timings = raw_run.get("timings_seconds_by_step")
    if not isinstance(timings, Mapping):
        return 0.0
    total = 0.0
    for value in timings.values():
        try:
            total += float(value or 0.0)
        except (TypeError, ValueError):
            continue
    return total


def _loop_ids(raw_run: Mapping[str, Any]) -> tuple[str, ...]:
    raw_loops = raw_run.get("loops")
    if not isinstance(raw_loops, list):
        return ()
    loop_ids: list[str] = []
    for raw_loop in raw_loops:
        if not isinstance(raw_loop, Mapping):
            continue
        loop_id = str(raw_loop.get("loop_id", "")).strip()
        if loop_id and _nested_int(raw_loop, "violation_count") > 0:
            loop_ids.append(loop_id)
    return tuple(sorted(dict.fromkeys(loop_ids)))


def _recurring_loop_ids(raw_run: Mapping[str, Any]) -> tuple[str, ...]:
    raw_loops = raw_run.get("loops")
    if not isinstance(raw_loops, list):
        return ()
    loop_ids: list[str] = []
    for raw_loop in raw_loops:
        if not isinstance(raw_loop, Mapping):
            continue
        loop_id = str(raw_loop.get("loop_id", "")).strip()
        if not loop_id:
            continue
        violation_count = _nested_int(raw_loop, "violation_count")
        recurrence_rate = _nested_float(raw_loop, "recurrence_rate")
        trend_delta = _nested_int(raw_loop, "trend_delta")
        if violation_count > 0 and (recurrence_rate >= 0.5 or trend_delta > 0):
            loop_ids.append(loop_id)
    return tuple(sorted(dict.fromkeys(loop_ids)))


def _closure_lag_loop_ids(raw_run: Mapping[str, Any]) -> tuple[str, ...]:
    raw_loops = raw_run.get("loops")
    if not isinstance(raw_loops, list):
        return ()
    loop_ids: list[str] = []
    for raw_loop in raw_loops:
        if not isinstance(raw_loop, Mapping):
            continue
        loop_id = str(raw_loop.get("loop_id", "")).strip()
        time_to_correction = raw_loop.get("time_to_correction_runs")
        if not loop_id or time_to_correction is None:
            continue
        try:
            correction_runs = int(time_to_correction or 0)
        except (TypeError, ValueError):
            correction_runs = 0
        if correction_runs >= _CLOSURE_LAG_RUNS:
            loop_ids.append(loop_id)
    return tuple(sorted(dict.fromkeys(loop_ids)))


def _max_time_to_correction_runs(raw_run: Mapping[str, Any]) -> int | None:
    raw_loops = raw_run.get("loops")
    if not isinstance(raw_loops, list):
        return None
    values: list[int] = []
    for raw_loop in raw_loops:
        if not isinstance(raw_loop, Mapping):
            continue
        raw_value = raw_loop.get("time_to_correction_runs")
        if raw_value is None:
            continue
        try:
            values.append(int(raw_value or 0))
        except (TypeError, ValueError):
            continue
    return max(values) if values else None


def _history_rows_from_telemetry_runs(
    *,
    telemetry_runs: list[dict[str, Any]],
    history_window_runs: int,
) -> tuple[HistoryRow, ...]:
    rows: list[HistoryRow] = []
    for raw_run in telemetry_runs:
        run_id = str(raw_run.get("run_id", "")).strip()
        raw_blockers = raw_run.get("open_blocker_ids")
        blocker_ids = (
            tuple(
                sorted(str(item).strip() for item in raw_blockers if str(item).strip())
            )
            if isinstance(raw_blockers, list)
            else ()
        )
        rows.append(
            HistoryRow(
                run_id=run_id,
                suite_red_state=bool(raw_run.get("suite_red_state", False)),
                open_blocker_ids=blocker_ids,
            )
        )
    if history_window_runs > 0:
        rows = rows[-history_window_runs:]
    return tuple(rows)


def _consecutive_red_state_runs(rows: tuple[HistoryRow, ...]) -> int:
    streak = 0
    for row in reversed(rows):
        if not row.suite_red_state:
            break
        streak += 1
    return streak


def _repeat_blocker_ids(rows: tuple[HistoryRow, ...]) -> tuple[str, ...]:
    if len(rows) < 2:
        return ()
    current_ids = set(rows[-1].open_blocker_ids)
    previous_ids = set(rows[-2].open_blocker_ids)
    return tuple(sorted(current_ids & previous_ids))


def _stalled_blocker_runs_by_id(rows: tuple[HistoryRow, ...]) -> dict[str, int]:
    if not rows:
        return {}
    current_ids = set(rows[-1].open_blocker_ids)
    stalled: dict[str, int] = {}
    for blocker_id in sorted(current_ids):
        streak = 0
        for row in reversed(rows):
            if blocker_id not in row.open_blocker_ids:
                break
            streak += 1
        stalled[blocker_id] = streak
    return stalled


def _unstable_blocker_ids(rows: tuple[HistoryRow, ...]) -> tuple[str, ...]:
    if not rows:
        return ()
    lookback = rows[-_UNSTABLE_LOOKBACK_RUNS:]
    current_ids = set(rows[-1].open_blocker_ids)
    unstable: list[str] = []
    for blocker_id in sorted(current_ids):
        present_indexes = [
            index for index, row in enumerate(lookback) if blocker_id in row.open_blocker_ids
        ]
        if len(present_indexes) < 2:
            continue
        if any(
            right - left > 1
            for left, right in zip(present_indexes, present_indexes[1:], strict=False)
        ):
            unstable.append(blocker_id)
    return tuple(unstable)


def _build_current_and_trend_summary(
    *,
    failing_test_ids: tuple[str, ...],
    failed_surface_ids: tuple[str, ...],
    failed_relation_ids: tuple[str, ...],
    observability_violation_ids: tuple[str, ...],
    telemetry_runs: list[dict[str, Any]],
    history_rows: tuple[HistoryRow, ...],
) -> tuple[CurrentSummary, TrendSummary]:
    latest_run = telemetry_runs[-1] if telemetry_runs else {}
    latest_total_runtime_seconds = _runtime_total_seconds(latest_run)
    previous_totals = tuple(
        _runtime_total_seconds(raw_run) for raw_run in telemetry_runs[:-1]
    )
    baseline_total_runtime_seconds = (
        float(median(previous_totals)) if previous_totals else None
    )
    runtime_regression_ratio = None
    runtime_delta_seconds = None
    severe_runtime_regression_current_band = False
    if baseline_total_runtime_seconds is not None and baseline_total_runtime_seconds > 0:
        runtime_regression_ratio = (
            latest_total_runtime_seconds / baseline_total_runtime_seconds
        )
        runtime_delta_seconds = (
            latest_total_runtime_seconds - baseline_total_runtime_seconds
        )
        severe_runtime_regression_current_band = (
            latest_total_runtime_seconds
            >= baseline_total_runtime_seconds * _SEVERE_RUNTIME_RATIO
            and runtime_delta_seconds >= _SEVERE_RUNTIME_DELTA_SECONDS
        )
    current_blocker_ids = {
        *(f"test:{test_id}" for test_id in failing_test_ids),
        *(f"surface:{surface_id}" for surface_id in failed_surface_ids),
        *(f"relation:{relation_id}" for relation_id in failed_relation_ids),
        *(f"observability:{violation_id}" for violation_id in observability_violation_ids),
        *(f"loop:{loop_id}" for loop_id in _loop_ids(latest_run)),
    }
    if severe_runtime_regression_current_band:
        current_blocker_ids.add("runtime:severe_current_band")
    current = CurrentSummary(
        suite_red_state=bool(failing_test_ids),
        failing_test_case_count=len(failing_test_ids),
        test_failure_count=len(failing_test_ids),
        local_ci_failed_surface_ids=failed_surface_ids,
        local_ci_failed_relation_ids=failed_relation_ids,
        observability_violation_ids=observability_violation_ids,
        severe_runtime_regression_current_band=severe_runtime_regression_current_band,
        repeat_blocker_ids=_repeat_blocker_ids(history_rows),
        stalled_blocker_runs_by_id=_stalled_blocker_runs_by_id(history_rows),
        unstable_blocker_ids=_unstable_blocker_ids(history_rows),
    )
    trend = TrendSummary(
        latest_total_runtime_seconds=latest_total_runtime_seconds,
        baseline_total_runtime_seconds=baseline_total_runtime_seconds,
        runtime_regression_ratio=runtime_regression_ratio,
        runtime_delta_seconds=runtime_delta_seconds,
        red_state_dwell_runs=_consecutive_red_state_runs(history_rows),
        recurring_loop_ids=_recurring_loop_ids(latest_run),
        closure_lag_loop_ids=_closure_lag_loop_ids(latest_run),
        max_time_to_correction_runs=_max_time_to_correction_runs(latest_run),
    )
    return (
        current,
        trend,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    failing_test_ids = _load_failing_test_ids(args.junit)
    failed_surface_ids, failed_relation_ids = _load_local_ci_failures(
        args.local_ci_contract
    )
    observability_violation_ids = _load_observability_violation_ids(args.observability)
    telemetry_runs = _load_telemetry_runs(args.telemetry_history)
    history_rows = _history_rows_from_telemetry_runs(
        telemetry_runs=telemetry_runs,
        history_window_runs=args.history_window_runs,
    )
    current, trend = _build_current_and_trend_summary(
        failing_test_ids=failing_test_ids,
        failed_surface_ids=failed_surface_ids,
        failed_relation_ids=failed_relation_ids,
        observability_violation_ids=observability_violation_ids,
        telemetry_runs=telemetry_runs,
        history_rows=history_rows,
    )
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "artifact_kind": _ARTIFACT_KIND,
        "generated_at_utc": _now_utc(),
        "generated_by": _GENERATED_BY,
        "history_window_runs": args.history_window_runs,
        "current": current.as_payload(),
        "trend": trend.as_payload(),
        "history": [row.as_payload() for row in history_rows],
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
