#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class LoopMetric:
    loop_id: str
    domain: str
    violation_count: int
    recurrence_rate: float
    false_positive_overrides: int
    time_to_correction_runs: int | None
    trend_delta: int | None


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Emit governance telemetry JSON/markdown from policy/docflow/delta/baseline artifacts."
        )
    )
    parser.add_argument("--run-id", default="", help="Stable run identifier (defaults to UTC timestamp).")
    parser.add_argument("--window-runs", type=int, default=10, help="Recent run window for recurrence/trends.")
    parser.add_argument("--docflow-delta", type=Path, default=Path("artifacts/out/docflow_compliance_delta.json"))
    parser.add_argument("--obsolescence-delta", type=Path, default=Path("artifacts/out/test_obsolescence_delta.json"))
    parser.add_argument("--annotation-delta", type=Path, default=Path("artifacts/out/test_annotation_drift_delta.json"))
    parser.add_argument("--ambiguity-delta", type=Path, default=Path("artifacts/out/ambiguity_delta.json"))
    parser.add_argument("--branchless-baseline", type=Path, default=Path("baselines/branchless_policy_baseline.json"))
    parser.add_argument(
        "--defensive-baseline", type=Path, default=Path("baselines/defensive_fallback_policy_baseline.json")
    )
    parser.add_argument("--timings", type=Path, default=Path("artifacts/audit_reports/ci_step_timings.json"))
    parser.add_argument(
        "--overrides",
        type=Path,
        default=Path("artifacts/out/governance_false_positive_overrides.json"),
    )
    parser.add_argument("--history", type=Path, default=Path("artifacts/out/governance_telemetry_history.json"))
    parser.add_argument("--json-out", type=Path, default=Path("artifacts/out/governance_telemetry.json"))
    parser.add_argument(
        "--md-out",
        type=Path,
        default=Path("artifacts/audit_reports/governance_telemetry.md"),
    )
    return parser.parse_args(argv)


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, Mapping):
        return {}
    return {str(key): payload[key] for key in payload}


def _nested_int(payload: Mapping[str, Any], keys: tuple[str, ...]) -> int:
    node: Any = payload
    for key in keys:
        if not isinstance(node, Mapping):
            return 0
        node = node.get(key)
    try:
        return int(node if node is not None else 0)
    except (TypeError, ValueError):
        return 0


def _extract_loop_counts(args: argparse.Namespace) -> dict[str, int]:
    docflow = _load_json_object(args.docflow_delta)
    obsolescence = _load_json_object(args.obsolescence_delta)
    annotation = _load_json_object(args.annotation_delta)
    ambiguity = _load_json_object(args.ambiguity_delta)
    branchless = _load_json_object(args.branchless_baseline)
    defensive = _load_json_object(args.defensive_baseline)

    loop_counts = {
        "policy.branchless": len(branchless.get("violations", [])) if isinstance(branchless.get("violations"), list) else 0,
        "policy.defensive_fallback": len(defensive.get("violations", [])) if isinstance(defensive.get("violations"), list) else 0,
        "docflow.contradictions": _nested_int(docflow, ("summary", "current", "contradicts")),
        "delta.obsolescence_unmapped": _nested_int(obsolescence, ("summary", "counts", "current", "unmapped")),
        "delta.obsolescence_opaque": _nested_int(obsolescence, ("summary", "opaque_evidence", "current")),
        "delta.annotation_orphaned": _nested_int(annotation, ("summary", "current", "orphaned")),
        "delta.ambiguity_total": _nested_int(ambiguity, ("summary", "total", "current")),
    }
    if loop_counts["delta.annotation_orphaned"] == 0:
        loop_counts["delta.annotation_orphaned"] = _nested_int(annotation, ("summary", "current", "annotations_without_tests"))
    return loop_counts


def _loop_domains() -> dict[str, str]:
    return {
        "policy.branchless": "security",
        "policy.defensive_fallback": "security",
        "docflow.contradictions": "governance",
        "delta.obsolescence_unmapped": "ratchet",
        "delta.obsolescence_opaque": "ratchet",
        "delta.annotation_orphaned": "ratchet",
        "delta.ambiguity_total": "ratchet",
    }


def _load_overrides(path: Path) -> dict[str, int]:
    payload = _load_json_object(path)
    loops = payload.get("loops")
    if not isinstance(loops, Mapping):
        return {}
    counts: dict[str, int] = {}
    for key, value in loops.items():
        if isinstance(value, Mapping):
            counts[str(key)] = _nested_int(value, ("false_positive_overrides",))
        else:
            try:
                counts[str(key)] = int(value)
            except (TypeError, ValueError):
                counts[str(key)] = 0
    return counts


def _load_history(path: Path) -> list[dict[str, Any]]:
    payload = _load_json_object(path)
    runs = payload.get("runs")
    if not isinstance(runs, list):
        return []
    history: list[dict[str, Any]] = []
    for run in runs:
        if isinstance(run, Mapping):
            history.append({str(key): run[key] for key in run})
    return history


def _recent_violation_series(history: list[dict[str, Any]], loop_id: str, window: int) -> list[int]:
    series: list[int] = []
    for run in history[-window:]:
        loops = run.get("loops")
        if not isinstance(loops, list):
            continue
        value = 0
        for entry in loops:
            if isinstance(entry, Mapping) and str(entry.get("loop_id", "")) == loop_id:
                value = _nested_int(entry, ("violation_count",))
                break
        series.append(value)
    return series


def _time_to_correction_runs(history: list[dict[str, Any]], loop_id: str, current_count: int) -> int | None:
    if current_count != 0:
        return None
    streak = 0
    for run in reversed(history):
        loops = run.get("loops")
        if not isinstance(loops, list):
            continue
        prior = 0
        for entry in loops:
            if isinstance(entry, Mapping) and str(entry.get("loop_id", "")) == loop_id:
                prior = _nested_int(entry, ("violation_count",))
                break
        streak += 1
        if prior > 0:
            return streak
    return None


def _collect_loop_metrics(
    *,
    loop_counts: Mapping[str, int],
    history: list[dict[str, Any]],
    window_runs: int,
    overrides: Mapping[str, int],
) -> list[LoopMetric]:
    metrics: list[LoopMetric] = []
    domains = _loop_domains()
    for loop_id in sorted(loop_counts):
        current_count = int(loop_counts[loop_id])
        history_series = _recent_violation_series(history, loop_id, window_runs)
        recurrence_source = history_series + [current_count]
        recurring = [value for value in recurrence_source if value > 0]
        recurrence_rate = (len(recurring) / len(recurrence_source)) if recurrence_source else 0.0
        previous_count = history_series[-1] if history_series else None
        trend_delta = None if previous_count is None else current_count - previous_count
        metrics.append(
            LoopMetric(
                loop_id=loop_id,
                domain=domains.get(loop_id, "governance"),
                violation_count=current_count,
                recurrence_rate=round(recurrence_rate, 4),
                false_positive_overrides=int(overrides.get(loop_id, 0)),
                time_to_correction_runs=_time_to_correction_runs(history, loop_id, current_count),
                trend_delta=trend_delta,
            )
        )
    return metrics


def _load_timing_for_run(path: Path, run_id: str) -> dict[str, float]:
    payload = _load_json_object(path)
    runs = payload.get("runs")
    if not isinstance(runs, list):
        return {}
    for run in runs:
        if not isinstance(run, Mapping) or str(run.get("run_id", "")) != run_id:
            continue
        entries = run.get("entries")
        if not isinstance(entries, list):
            return {}
        durations: dict[str, float] = {}
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            label = str(entry.get("label", "")).strip()
            value = entry.get("elapsed_seconds")
            if not label:
                continue
            try:
                durations[label] = float(value)
            except (TypeError, ValueError):
                durations[label] = 0.0
        return durations
    return {}


def _build_slos(metrics: list[LoopMetric], window_runs: int) -> list[dict[str, Any]]:
    by_domain: dict[str, list[LoopMetric]] = {}
    for metric in metrics:
        by_domain.setdefault(metric.domain, []).append(metric)
    slo_spec = {
        "security": (f"No repeated SEC-* violation for {window_runs} runs", 0.0),
        "governance": (f"No repeated GOV-* contradiction for {window_runs} runs", 0.0),
        "ratchet": ("Ratchet domains trend non-increasing over recent runs", 1.0),
    }
    slos: list[dict[str, Any]] = []
    for domain in ("security", "governance", "ratchet"):
        domain_metrics = by_domain.get(domain, [])
        max_recurrence = max((m.recurrence_rate for m in domain_metrics), default=0.0)
        trend_ok = all((m.trend_delta or 0) <= 0 for m in domain_metrics)
        target = slo_spec[domain][1]
        if domain == "ratchet":
            status = "pass" if trend_ok else "fail"
        else:
            status = "pass" if max_recurrence <= target else "fail"
        slos.append(
            {
                "domain": domain,
                "objective": slo_spec[domain][0],
                "window_runs": window_runs,
                "status": status,
                "max_recurrence_rate": round(max_recurrence, 4),
                "requires_non_increasing_trend": domain == "ratchet",
            }
        )
    return slos


def _render_markdown(payload: Mapping[str, Any]) -> str:
    loops = payload.get("loops", [])
    slos = payload.get("convergence_slos", [])
    lines = [
        "# Governance Telemetry",
        "",
        f"- run_id: `{payload.get('run_id', 'unknown')}`",
        f"- generated_at_utc: `{payload.get('generated_at_utc', 'unknown')}`",
        f"- trend_window_runs: `{payload.get('trend_window_runs', 0)}`",
        "",
        "## Per-loop metrics",
        "",
        "| loop_id | domain | violations | trend_delta | recurrence_rate | false_positive_overrides | time_to_correction_runs |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    if isinstance(loops, list):
        for loop in loops:
            if not isinstance(loop, Mapping):
                continue
            trend = loop.get("trend_delta")
            ttc = loop.get("time_to_correction_runs")
            trend_text = "n/a" if trend is None else str(trend)
            ttc_text = "n/a" if ttc is None else str(ttc)
            lines.append(
                "| "
                f"`{loop.get('loop_id', '')}` | `{loop.get('domain', '')}` | "
                f"{_nested_int(loop, ('violation_count',))} | {trend_text} | "
                f"{float(loop.get('recurrence_rate', 0.0)):.2f} | "
                f"{_nested_int(loop, ('false_positive_overrides',))} | {ttc_text} |"
            )
    lines.extend(["", "## Convergence SLOs", ""])
    lines.extend([
        "| domain | objective | status | max_recurrence_rate |",
        "| --- | --- | --- | ---: |",
    ])
    if isinstance(slos, list):
        for slo in slos:
            if not isinstance(slo, Mapping):
                continue
            lines.append(
                f"| `{slo.get('domain', '')}` | {slo.get('objective', '')} | "
                f"`{slo.get('status', '')}` | {float(slo.get('max_recurrence_rate', 0.0)):.2f} |"
            )
    lines.append("")
    return "\n".join(lines)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    generated_at_utc = _now_utc()
    run_id = args.run_id or generated_at_utc.replace(":", "").replace("-", "")
    history = _load_history(args.history)
    overrides = _load_overrides(args.overrides)
    loop_counts = _extract_loop_counts(args)
    metrics = _collect_loop_metrics(
        loop_counts=loop_counts,
        history=history,
        window_runs=max(1, int(args.window_runs)),
        overrides=overrides,
    )
    durations = _load_timing_for_run(args.timings, run_id)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at_utc": generated_at_utc,
        "trend_window_runs": max(1, int(args.window_runs)),
        "timings_seconds_by_step": durations,
        "loops": [
            {
                "loop_id": metric.loop_id,
                "domain": metric.domain,
                "violation_count": metric.violation_count,
                "trend_delta": metric.trend_delta,
                "recurrence_rate": metric.recurrence_rate,
                "false_positive_overrides": metric.false_positive_overrides,
                "time_to_correction_runs": metric.time_to_correction_runs,
            }
            for metric in metrics
        ],
    }
    payload["convergence_slos"] = _build_slos(metrics, payload["trend_window_runs"])
    _write_json(args.json_out, payload)

    next_history = history + [payload]
    _write_json(args.history, {"schema_version": 1, "runs": next_history[-50:]})

    md = _render_markdown(payload)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(md + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
