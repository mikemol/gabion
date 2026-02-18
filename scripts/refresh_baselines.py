from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:  # pragma: no cover - import form depends on invocation mode
    from scripts.deadline_runtime import DeadlineBudget, deadline_scope_from_lsp_env
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from deadline_runtime import DeadlineBudget, deadline_scope_from_lsp_env
from gabion.analysis.timeout_context import check_deadline
from gabion.execution_plan import BaselineFacet, DeadlineFacet, ExecutionPlan


OBSOLESCENCE_DELTA_PATH = Path("artifacts/out/test_obsolescence_delta.json")
ANNOTATION_DRIFT_DELTA_PATH = Path("artifacts/out/test_annotation_drift_delta.json")
AMBIGUITY_DELTA_PATH = Path("artifacts/out/ambiguity_delta.json")
DOCFLOW_DELTA_PATH = Path("artifacts/out/docflow_compliance_delta.json")
OBSOLESCENCE_STATE_PATH = Path("artifacts/out/test_obsolescence_state.json")
ANNOTATION_DRIFT_STATE_PATH = Path("artifacts/out/test_annotation_drift.json")
AMBIGUITY_STATE_PATH = Path("artifacts/out/ambiguity_state.json")
DOCFLOW_BASELINE_PATH = Path("baselines/docflow_compliance_baseline.json")
DOCFLOW_CURRENT_PATH = Path("artifacts/out/docflow_compliance.json")

ENV_GATE_UNMAPPED = "GABION_GATE_UNMAPPED_DELTA"
ENV_GATE_ORPHANED = "GABION_GATE_ORPHANED_DELTA"
ENV_GATE_AMBIGUITY = "GABION_GATE_AMBIGUITY_DELTA"
_DEFAULT_TIMEOUT_TICKS = 120_000
_DEFAULT_TIMEOUT_TICK_NS = 1_000_000
_DEFAULT_TIMEOUT_BUDGET = DeadlineBudget(
    ticks=_DEFAULT_TIMEOUT_TICKS,
    tick_ns=_DEFAULT_TIMEOUT_TICK_NS,
)


def _deadline_scope():
    return deadline_scope_from_lsp_env(
        default_budget=_DEFAULT_TIMEOUT_BUDGET,
    )


def _run_check(flag: str, timeout: int | None, extra: list[str] | None = None) -> None:
    cmd = [
        sys.executable,
        "-m",
        "gabion",
        "check",
        "--no-fail-on-violations",
        "--no-fail-on-type-ambiguities",
        flag,
    ]
    if extra:
        cmd.extend(extra)
    env = dict(os.environ)
    env["GABION_DIRECT_RUN"] = "1"
    subprocess.run(cmd, check=True, timeout=timeout, env=env)


def _run_docflow_delta_emit(timeout: int | None) -> None:
    env = dict(os.environ)
    env.setdefault("GABION_DIRECT_RUN", "1")
    subprocess.run(
        [sys.executable, "scripts/docflow_delta_emit.py"],
        check=True,
        timeout=timeout,
        env=env,
    )


def _state_args(path: Path, flag: str) -> list[str]:
    if path.exists():
        return [flag, str(path)]
    return []


def _gate_enabled(env_flag: str) -> bool:
    value = os.getenv(env_flag, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_nested(payload: object, keys: list[str], default: int = 0) -> int:
    current = payload
    for key in keys:
        check_deadline()
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    try:
        return int(current) if current is not None else default
    except (TypeError, ValueError):
        return default


def _build_execution_plan(timeout: int | None) -> ExecutionPlan:
    plan = ExecutionPlan()
    plan.with_deadline(DeadlineFacet(timeout_seconds=timeout))
    return plan


def _risk_entries(
    *,
    obsolescence_payload: dict[str, object] | None = None,
    annotation_payload: dict[str, object] | None = None,
    ambiguity_payload: dict[str, object] | None = None,
    docflow_payload: dict[str, object] | None = None,
) -> tuple[tuple[str, int], ...]:
    values: list[tuple[str, int]] = []
    if obsolescence_payload is not None:
        values.append(
            (
                "obsolescence.opaque",
                _get_nested(obsolescence_payload, ["summary", "opaque_evidence", "delta"]),
            )
        )
        values.append(
            (
                "obsolescence.unmapped",
                _get_nested(obsolescence_payload, ["summary", "counts", "delta", "unmapped"]),
            )
        )
    if annotation_payload is not None:
        values.append(
            ("annotation.orphaned", _get_nested(annotation_payload, ["summary", "delta", "orphaned"]))
        )
    if ambiguity_payload is not None:
        values.append(("ambiguity.total", _get_nested(ambiguity_payload, ["summary", "total", "delta"])))
    if docflow_payload is not None:
        values.append(
            ("docflow.contradicts", _get_nested(docflow_payload, ["summary", "delta", "contradicts"]))
        )
    return tuple(values)


def _ensure_delta(
    flag: str,
    path: Path,
    timeout: int | None,
    *,
    extra: list[str] | None = None,
) -> dict[str, object]:
    _run_check(flag, timeout, extra)
    if not path.exists():
        raise FileNotFoundError(f"Missing delta output at {path}")
    return _load_json(path)


def _guard_obsolescence_delta(plan: ExecutionPlan) -> None:
    payload = _ensure_delta(
        "--emit-test-obsolescence-delta",
        OBSOLESCENCE_DELTA_PATH,
        plan.deadline.timeout_seconds,
        extra=_state_args(OBSOLESCENCE_STATE_PATH, "--test-obsolescence-state"),
    )
    plan.with_baseline(BaselineFacet(risks=_risk_entries(obsolescence_payload=payload)))
    if plan.baseline.risk("obsolescence.opaque") > 0:
        raise SystemExit(
            "Refusing to refresh obsolescence baseline: opaque evidence delta > 0."
        )
    if _gate_enabled(ENV_GATE_UNMAPPED) and plan.baseline.risk("obsolescence.unmapped") > 0:
        raise SystemExit(
            "Refusing to refresh obsolescence baseline: unmapped delta > 0."
        )


def _guard_annotation_drift_delta(plan: ExecutionPlan) -> None:
    if not _gate_enabled(ENV_GATE_ORPHANED):
        return
    payload = _ensure_delta(
        "--emit-test-annotation-drift-delta",
        ANNOTATION_DRIFT_DELTA_PATH,
        plan.deadline.timeout_seconds,
        extra=_state_args(ANNOTATION_DRIFT_STATE_PATH, "--test-annotation-drift-state"),
    )
    plan.with_baseline(BaselineFacet(risks=_risk_entries(annotation_payload=payload)))
    if plan.baseline.risk("annotation.orphaned") > 0:
        raise SystemExit(
            "Refusing to refresh annotation drift baseline: orphaned delta > 0."
        )


def _guard_ambiguity_delta(plan: ExecutionPlan) -> None:
    if not _gate_enabled(ENV_GATE_AMBIGUITY):
        return
    payload = _ensure_delta(
        "--emit-ambiguity-delta",
        AMBIGUITY_DELTA_PATH,
        plan.deadline.timeout_seconds,
        extra=_state_args(AMBIGUITY_STATE_PATH, "--ambiguity-state"),
    )
    plan.with_baseline(BaselineFacet(risks=_risk_entries(ambiguity_payload=payload)))
    if plan.baseline.risk("ambiguity.total") > 0:
        raise SystemExit(
            "Refusing to refresh ambiguity baseline: ambiguity delta > 0."
        )


def _guard_docflow_delta(plan: ExecutionPlan) -> None:
    _run_docflow_delta_emit(plan.deadline.timeout_seconds)
    if not DOCFLOW_DELTA_PATH.exists():
        raise FileNotFoundError(
            f"Missing docflow delta output at {DOCFLOW_DELTA_PATH}"
        )
    payload = _load_json(DOCFLOW_DELTA_PATH)
    if payload.get("baseline_missing"):
        return
    plan.with_baseline(BaselineFacet(risks=_risk_entries(docflow_payload=payload)))
    if plan.baseline.risk("docflow.contradicts") > 0:
        raise SystemExit(
            "Refusing to refresh docflow baseline: contradictions delta > 0."
        )


def main() -> int:
    with _deadline_scope():
        parser = argparse.ArgumentParser(
            description="Refresh baseline carriers via gabion check.",
        )
        parser.add_argument(
            "--obsolescence",
            action="store_true",
            help="Refresh baselines/test_obsolescence_baseline.json",
        )
        parser.add_argument(
            "--annotation-drift",
            action="store_true",
            help="Refresh baselines/test_annotation_drift_baseline.json",
        )
        parser.add_argument(
            "--ambiguity",
            action="store_true",
            help="Refresh baselines/ambiguity_baseline.json",
        )
        parser.add_argument(
            "--all",
            action="store_true",
            help="Refresh all baselines (default when no flags provided).",
        )
        parser.add_argument(
            "--docflow",
            action="store_true",
            help="Refresh baselines/docflow_compliance_baseline.json",
        )
        parser.add_argument(
            "--timeout",
            type=int,
            default=None,
            help="Seconds to wait for each gabion check (default: no timeout).",
        )
        args = parser.parse_args()

        if not (
            args.obsolescence
            or args.annotation_drift
            or args.ambiguity
            or args.docflow
            or args.all
        ):
            args.all = True

        plan = _build_execution_plan(args.timeout)

        if args.all or args.obsolescence:
            _guard_obsolescence_delta(plan)
            _run_check(
                "--write-test-obsolescence-baseline",
                plan.deadline.timeout_seconds,
                _state_args(OBSOLESCENCE_STATE_PATH, "--test-obsolescence-state"),
            )
        if args.all or args.annotation_drift:
            _guard_annotation_drift_delta(plan)
            _run_check(
                "--write-test-annotation-drift-baseline",
                plan.deadline.timeout_seconds,
                _state_args(
                    ANNOTATION_DRIFT_STATE_PATH, "--test-annotation-drift-state"
                ),
            )
        if args.all or args.ambiguity:
            _guard_ambiguity_delta(plan)
            _run_check(
                "--write-ambiguity-baseline",
                plan.deadline.timeout_seconds,
                _state_args(AMBIGUITY_STATE_PATH, "--ambiguity-state"),
            )
        if args.all or args.docflow:
            _guard_docflow_delta(plan)
            if not DOCFLOW_CURRENT_PATH.exists():
                raise FileNotFoundError(
                    f"Missing docflow compliance output at {DOCFLOW_CURRENT_PATH}"
                )
            DOCFLOW_BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(DOCFLOW_CURRENT_PATH, DOCFLOW_BASELINE_PATH)

        return 0


if __name__ == "__main__":
    raise SystemExit(main())
