# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

from gabion.commands import transport_policy
from gabion.runtime import json_io
from gabion.runtime import env_policy
from gabion.tooling.deadline_runtime import DeadlineBudget, deadline_scope_from_lsp_env
from gabion.analysis.timeout_context import check_deadline
from gabion.execution_plan import DocflowFacet, ExecutionPlan
from gabion.order_contract import sort_once

BASELINE_PATH = Path("baselines/docflow_compliance_baseline.json")
CURRENT_PATH = Path("artifacts/out/docflow_compliance.json")
DELTA_PATH = Path("artifacts/out/docflow_compliance_delta.json")

_DEFAULT_DELTA_TIMEOUT_TICKS = 120_000
_DEFAULT_DELTA_TIMEOUT_TICK_NS = 1_000_000
_DEFAULT_DELTA_TIMEOUT_BUDGET = DeadlineBudget(
    ticks=_DEFAULT_DELTA_TIMEOUT_TICKS,
    tick_ns=_DEFAULT_DELTA_TIMEOUT_TICK_NS,
)


def _delta_deadline_scope():
    return deadline_scope_from_lsp_env(
        default_budget=_DEFAULT_DELTA_TIMEOUT_BUDGET,
    )


def _run_docflow_audit(
    *,
    run_fn: Callable[..., subprocess.CompletedProcess[str] | None] = subprocess.run,
) -> None:
    command: list[str] = [sys.executable, "-m", "gabion"]
    override = transport_policy.transport_override()
    carrier = "direct"
    if override is not None and override.direct_requested is not None:
        carrier = "direct" if override.direct_requested else "lsp"
    command.extend(["--carrier", carrier])
    timeout_override = env_policy.lsp_timeout_override()
    if timeout_override is not None:
        command.extend(
            [
                "--timeout",
                env_policy.duration_text_from_ticks(
                    ticks=timeout_override.ticks,
                    tick_ns=timeout_override.tick_ns,
                ),
            ]
        )
    command.append("docflow")
    run_fn(
        command,
        check=True,
        env=dict(os.environ),
    )


def _changed_paths_from_git() -> tuple[str, ...]:
    try:
        out = subprocess.check_output(
            ["git", "diff", "--name-only", "HEAD~1..HEAD"],
            text=True,
        )
    except Exception:
        return ()
    paths = [line.strip() for line in out.splitlines() if line.strip()]
    return tuple(
        sort_once(
            set(paths),
            source="_changed_paths_from_git.paths",
        )
    )


def _build_execution_plan(
    *,
    changed_paths_fn: Callable[[], tuple[str, ...]] = _changed_paths_from_git,
) -> ExecutionPlan:
    plan = ExecutionPlan()
    plan.with_docflow(DocflowFacet(changed_paths=changed_paths_fn()))
    return plan


def _load_summary(path: Path) -> tuple[dict[str, int], bool]:
    if not path.exists():
        return {"compliant": 0, "contradicts": 0, "excess": 0, "proposed": 0}, True
    payload = json_io.load_json_object_path(path)
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    counts: dict[str, int] = {}
    for key in ("compliant", "contradicts", "excess", "proposed"):
        check_deadline()
        value = 0
        if isinstance(summary, dict):
            try:
                value = int(summary.get(key, 0))
            except (TypeError, ValueError):
                value = 0
        counts[key] = value
    return counts, False


def _delta_counts(
    baseline: dict[str, int],
    current: dict[str, int],
) -> dict[str, int]:
    delta: dict[str, int] = {}
    for key in ("compliant", "contradicts", "excess", "proposed"):
        check_deadline()
        delta[key] = int(current.get(key, 0)) - int(baseline.get(key, 0))
    return delta


def main(
    *,
    build_execution_plan_fn: Callable[[], ExecutionPlan] = _build_execution_plan,
    run_docflow_audit_fn: Callable[[], None] = _run_docflow_audit,
    load_summary_fn: Callable[[Path], tuple[dict[str, int], bool]] = _load_summary,
    delta_counts_fn: Callable[[dict[str, int], dict[str, int]], dict[str, int]] = _delta_counts,
    baseline_path: Path = BASELINE_PATH,
    current_path: Path = CURRENT_PATH,
    delta_path: Path = DELTA_PATH,
    write_text_fn: Callable[[Path, str], None] | None = None,
) -> int:
    with _delta_deadline_scope():
        plan = build_execution_plan_fn()
        try:
            run_docflow_audit_fn()
        except subprocess.CalledProcessError:
            print("Docflow delta emit failed: docflow audit did not succeed.")
            return 0
        if not current_path.exists():
            print("Docflow compliance output missing; delta emit skipped.")
            return 0
        baseline_counts, baseline_missing = load_summary_fn(baseline_path)
        current_counts, _ = load_summary_fn(current_path)
        payload = {
            "baseline": {"path": str(baseline_path)},
            "current": {"path": str(current_path)},
            "baseline_missing": baseline_missing,
            "summary": {
                "baseline": baseline_counts,
                "current": current_counts,
                "delta": delta_counts_fn(baseline_counts, current_counts),
            },
            "facets": {
                "docflow": {
                    "changed_paths": list(plan.docflow.changed_paths),
                }
            },
            "version": 1,
        }
        canonical_payload = json_io.canonicalize_json(payload)
        serialized = json_io.dump_json_pretty(canonical_payload)
        if write_text_fn is None:
            delta_path.write_text(serialized, encoding="utf-8")
        else:
            write_text_fn(delta_path, serialized)
        return 0
