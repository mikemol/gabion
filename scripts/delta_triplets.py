from __future__ import annotations

import concurrent.futures
import os
import subprocess
import sys

try:  # pragma: no cover - import form depends on invocation mode
    from scripts.deadline_runtime import DeadlineBudget, deadline_scope_from_lsp_env
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from deadline_runtime import DeadlineBudget, deadline_scope_from_lsp_env
from gabion.analysis.timeout_context import check_deadline


TRIPLETS: dict[str, list[str]] = {
    "obsolescence": [
        "scripts/obsolescence_delta_emit.py",
        "scripts/obsolescence_delta_advisory.py",
        "scripts/obsolescence_delta_gate.py",
        "scripts/obsolescence_delta_unmapped_gate.py",
    ],
    "annotation_drift": [
        "scripts/annotation_drift_delta_emit.py",
        "scripts/annotation_drift_delta_advisory.py",
        "scripts/annotation_drift_orphaned_gate.py",
    ],
    "ambiguity": [
        "scripts/ambiguity_delta_emit.py",
        "scripts/ambiguity_delta_advisory.py",
        "scripts/ambiguity_delta_gate.py",
    ],
    "docflow": [
        "scripts/docflow_delta_emit.py",
        "scripts/docflow_delta_advisory.py",
        "scripts/docflow_delta_gate.py",
    ],
}

_DEFAULT_TRIPLET_TIMEOUT_TICKS = 120_000
_DEFAULT_TRIPLET_TIMEOUT_TICK_NS = 1_000_000
_DEFAULT_TRIPLET_TIMEOUT_BUDGET = DeadlineBudget(
    ticks=_DEFAULT_TRIPLET_TIMEOUT_TICKS,
    tick_ns=_DEFAULT_TRIPLET_TIMEOUT_TICK_NS,
)


def _deadline_scope():
    return deadline_scope_from_lsp_env(
        default_budget=_DEFAULT_TRIPLET_TIMEOUT_BUDGET,
    )


def _run_triplet(name: str, steps: list[str]) -> int:
    exit_code = 0
    env = dict(os.environ)
    env.setdefault("GABION_DIRECT_RUN", "1")
    env.setdefault("GABION_LSP_TIMEOUT_TICKS", "65000000")
    env.setdefault("GABION_LSP_TIMEOUT_TICK_NS", "1000000")
    with _deadline_scope():
        for step in steps:
            check_deadline()
            result = subprocess.run([sys.executable, step], check=False, env=env)
            if result.returncode != 0:
                print(f"{name} step failed: {step} (exit {result.returncode})")
                exit_code = exit_code or result.returncode
                if step.endswith("_emit.py"):
                    return exit_code
    return exit_code


def main() -> int:
    failures = 0
    with _deadline_scope():
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(TRIPLETS)
        ) as executor:
            futures = {
                executor.submit(_run_triplet, name, steps): name
                for name, steps in TRIPLETS.items()
            }
            for future in concurrent.futures.as_completed(futures):
                check_deadline()
                name = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"{name} triplet crashed: {exc}")
                    failures += 1
                    continue
                if result != 0:
                    failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
