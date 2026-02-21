from __future__ import annotations

import concurrent.futures
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable, Mapping, Sequence

try:  # pragma: no cover - import form depends on invocation mode
    from scripts.deadline_runtime import DeadlineBudget, deadline_scope_from_lsp_env
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from deadline_runtime import DeadlineBudget, deadline_scope_from_lsp_env
from gabion.analysis.timeout_context import check_deadline
from gabion.order_contract import ordered_or_sorted


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
_DEFAULT_STEP_HEARTBEAT_SECONDS = 30.0
_DEFAULT_PENDING_HEARTBEAT_SECONDS = 30.0


def _deadline_scope():
    return deadline_scope_from_lsp_env(
        default_budget=_DEFAULT_TRIPLET_TIMEOUT_BUDGET,
    )


def _heartbeat_seconds(env_name: str, default_value: float) -> float:
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return default_value
    try:
        parsed = float(raw)
    except ValueError:
        return default_value
    return parsed if parsed > 0 else 0.0


def _run_step_process(
    *,
    # dataflow-bundle: name, step_index, step_total
    name: str,
    step: str,
    env: Mapping[str, str],
    step_index: int,
    step_total: int,
    step_heartbeat_seconds: float,
    print_fn: Callable[[str], None] = print,
    popen_fn: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    try:
        process = popen_fn([sys.executable, step], env=dict(env))
    except OSError as exc:
        print_fn(f"{name} step launch failed: {step} ({exc})")
        return 127
    started = monotonic_fn()
    last_heartbeat = started
    while True:
        check_deadline()
        return_code = process.poll()
        if return_code is not None:
            return int(return_code)
        if step_heartbeat_seconds > 0:
            now = monotonic_fn()
            if (now - last_heartbeat) >= step_heartbeat_seconds:
                elapsed = max(0.0, now - started)
                print_fn(
                    f"{name} step heartbeat {step_index}/{step_total}: "
                    f"{Path(step).name} elapsed_s={elapsed:.1f}"
                )
                last_heartbeat = now
        sleep_fn(0.5)


def _run_triplet(
    name: str,
    steps: list[str],
    *,
    run_step_fn: Callable[..., int] = _run_step_process,
    print_fn: Callable[[str], None] = print,
    monotonic_fn: Callable[[], float] = time.monotonic,
    step_heartbeat_seconds: float | None = None,
) -> int:
    heartbeat_seconds = (
        _heartbeat_seconds(
            "GABION_DELTA_TRIPLETS_STEP_HEARTBEAT_SECONDS",
            _DEFAULT_STEP_HEARTBEAT_SECONDS,
        )
        if step_heartbeat_seconds is None
        else max(0.0, float(step_heartbeat_seconds))
    )
    env = dict(os.environ)
    env.setdefault("GABION_DIRECT_RUN", "1")
    env.setdefault("GABION_LSP_TIMEOUT_TICKS", "65000000")
    env.setdefault("GABION_LSP_TIMEOUT_TICK_NS", "1000000")
    exit_code = 0
    print_fn(f"{name} triplet start: steps={len(steps)}")
    with _deadline_scope():
        for index, step in enumerate(steps, start=1):
            check_deadline()
            started = monotonic_fn()
            print_fn(f"{name} step start {index}/{len(steps)}: {step}")
            step_exit = int(
                run_step_fn(
                    name=name,
                    step=step,
                    env=env,
                    step_index=index,
                    step_total=len(steps),
                    step_heartbeat_seconds=heartbeat_seconds,
                    print_fn=print_fn,
                    monotonic_fn=monotonic_fn,
                )
            )
            elapsed = max(0.0, monotonic_fn() - started)
            print_fn(
                f"{name} step complete {index}/{len(steps)}: {step} "
                f"exit={step_exit} elapsed_s={elapsed:.1f}"
            )
            if step_exit != 0:
                print_fn(f"{name} step failed: {step} (exit {step_exit})")
                exit_code = exit_code or step_exit
                if step.endswith("_emit.py"):
                    print_fn(
                        f"{name} triplet aborting remaining steps because emit failed."
                    )
                    return exit_code
    print_fn(f"{name} triplet complete: exit={exit_code}")
    return exit_code


def main(
    *,
    triplets: Mapping[str, Sequence[str]] = TRIPLETS,
    run_triplet_fn: Callable[..., int] = _run_triplet,
    print_fn: Callable[[str], None] = print,
    pending_heartbeat_seconds: float | None = None,
) -> int:
    triplet_map = {str(name): [str(step) for step in steps] for name, steps in triplets.items()}
    if not triplet_map:
        print_fn("delta_triplets: no triplets configured.")
        return 0
    heartbeat_seconds = (
        _heartbeat_seconds(
            "GABION_DELTA_TRIPLETS_PENDING_HEARTBEAT_SECONDS",
            _DEFAULT_PENDING_HEARTBEAT_SECONDS,
        )
        if pending_heartbeat_seconds is None
        else max(0.0, float(pending_heartbeat_seconds))
    )
    failures = 0
    print_fn(f"delta_triplets: start triplets={len(triplet_map)}")
    with _deadline_scope():
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(triplet_map)
        ) as executor:
            futures = {
                executor.submit(run_triplet_fn, name, steps): name
                for name, steps in triplet_map.items()
            }
            pending = set(futures)
            while pending:
                check_deadline()
                done, pending = concurrent.futures.wait(
                    pending,
                    timeout=heartbeat_seconds if heartbeat_seconds > 0 else None,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                if not done:
                    pending_names = ", ".join(
                        ordered_or_sorted(
                            (futures[future] for future in pending),
                            source="main.pending_triplet_names",
                        )
                    )
                    print_fn(f"delta_triplets heartbeat: pending={pending_names}")
                    continue
                for future in done:
                    check_deadline()
                    name = futures[future]
                    try:
                        result = int(future.result())
                    except Exception as exc:
                        print_fn(f"{name} triplet crashed: {exc}")
                        failures += 1
                        continue
                    if result != 0:
                        failures += 1
    print_fn(f"delta_triplets: complete failures={failures} total={len(triplet_map)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
