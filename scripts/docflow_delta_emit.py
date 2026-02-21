from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

try:  # pragma: no cover - import form depends on invocation mode
    from scripts.deadline_runtime import DeadlineBudget, deadline_scope_from_lsp_env
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from deadline_runtime import DeadlineBudget, deadline_scope_from_lsp_env
from gabion.analysis.timeout_context import check_deadline
from gabion.execution_plan import DocflowFacet, ExecutionPlan
from gabion.order_contract import ordered_or_sorted

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


def _run_docflow_audit() -> None:
    env = dict(os.environ)
    env.setdefault("GABION_DIRECT_RUN", "1")
    subprocess.run(
        [sys.executable, "-m", "gabion", "docflow"],
        check=True,
        env=env,
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
        ordered_or_sorted(
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
    payload = json.loads(path.read_text(encoding="utf-8"))
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


def main() -> int:
    with _delta_deadline_scope():
        plan = _build_execution_plan()
        try:
            _run_docflow_audit()
        except subprocess.CalledProcessError:
            print("Docflow delta emit failed: docflow audit did not succeed.")
            return 0
        if not CURRENT_PATH.exists():
            print("Docflow compliance output missing; delta emit skipped.")
            return 0
        baseline_counts, baseline_missing = _load_summary(BASELINE_PATH)
        current_counts, _ = _load_summary(CURRENT_PATH)
        payload = {
            "baseline": {"path": str(BASELINE_PATH)},
            "current": {"path": str(CURRENT_PATH)},
            "baseline_missing": baseline_missing,
            "summary": {
                "baseline": baseline_counts,
                "current": current_counts,
                "delta": _delta_counts(baseline_counts, current_counts),
            },
            "facets": {
                "docflow": {
                    "changed_paths": list(plan.docflow.changed_paths),
                }
            },
            "version": 1,
        }
        DELTA_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
