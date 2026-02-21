from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Callable, Literal, Mapping, Sequence

from gabion.tooling import (
    ambiguity_delta_advisory,
    ambiguity_delta_emit,
    ambiguity_delta_gate,
    annotation_drift_delta_advisory,
    annotation_drift_delta_emit,
    annotation_drift_orphaned_gate,
    docflow_delta_advisory,
    docflow_delta_emit,
    docflow_delta_gate,
    obsolescence_delta_advisory,
    obsolescence_delta_emit,
    obsolescence_delta_gate,
    obsolescence_delta_unmapped_gate,
)
from gabion.tooling.deadline_runtime import DeadlineBudget, deadline_scope_from_lsp_env
from gabion.analysis.timeout_context import check_deadline
from gabion.order_contract import ordered_or_sorted


StepKind = Literal["emit", "advisory", "gate"]


@dataclass(frozen=True)
class StepSpec:
    id: str
    label: str
    kind: StepKind
    run: Callable[[], int]


def _triplet_resume_checkpoint_path(triplet_name: str) -> Path:
    normalized = triplet_name.strip().lower().replace("-", "_")
    return Path("artifacts/audit_reports") / f"dataflow_resume_checkpoint_ci_{normalized}.json"


def _run_obsolescence_emit(
    *,
    run_emit: Callable[..., int] = obsolescence_delta_emit.main,
) -> int:
    return run_emit(resume_checkpoint=_triplet_resume_checkpoint_path("obsolescence"))


def _run_annotation_drift_emit(
    *,
    run_emit: Callable[..., int] = annotation_drift_delta_emit.main,
) -> int:
    return run_emit(resume_checkpoint=_triplet_resume_checkpoint_path("annotation_drift"))


def _run_ambiguity_emit(
    *,
    run_emit: Callable[..., int] = ambiguity_delta_emit.main,
) -> int:
    return run_emit(resume_checkpoint=_triplet_resume_checkpoint_path("ambiguity"))


TRIPLETS: dict[str, tuple[StepSpec, ...]] = {
    "obsolescence": (
        StepSpec(
            id="obsolescence_delta_emit",
            label="obsolescence_delta_emit",
            kind="emit",
            run=_run_obsolescence_emit,
        ),
        StepSpec(
            id="obsolescence_delta_advisory",
            label="obsolescence_delta_advisory",
            kind="advisory",
            run=obsolescence_delta_advisory.main,
        ),
        StepSpec(
            id="obsolescence_delta_gate",
            label="obsolescence_delta_gate",
            kind="gate",
            run=obsolescence_delta_gate.main,
        ),
        StepSpec(
            id="obsolescence_delta_unmapped_gate",
            label="obsolescence_delta_unmapped_gate",
            kind="gate",
            run=obsolescence_delta_unmapped_gate.main,
        ),
    ),
    "annotation_drift": (
        StepSpec(
            id="annotation_drift_delta_emit",
            label="annotation_drift_delta_emit",
            kind="emit",
            run=_run_annotation_drift_emit,
        ),
        StepSpec(
            id="annotation_drift_delta_advisory",
            label="annotation_drift_delta_advisory",
            kind="advisory",
            run=annotation_drift_delta_advisory.main,
        ),
        StepSpec(
            id="annotation_drift_orphaned_gate",
            label="annotation_drift_orphaned_gate",
            kind="gate",
            run=annotation_drift_orphaned_gate.main,
        ),
    ),
    "ambiguity": (
        StepSpec(
            id="ambiguity_delta_emit",
            label="ambiguity_delta_emit",
            kind="emit",
            run=_run_ambiguity_emit,
        ),
        StepSpec(
            id="ambiguity_delta_advisory",
            label="ambiguity_delta_advisory",
            kind="advisory",
            run=ambiguity_delta_advisory.main,
        ),
        StepSpec(
            id="ambiguity_delta_gate",
            label="ambiguity_delta_gate",
            kind="gate",
            run=ambiguity_delta_gate.main,
        ),
    ),
    "docflow": (
        StepSpec(
            id="docflow_delta_emit",
            label="docflow_delta_emit",
            kind="emit",
            run=docflow_delta_emit.main,
        ),
        StepSpec(
            id="docflow_delta_advisory",
            label="docflow_delta_advisory",
            kind="advisory",
            run=docflow_delta_advisory.main,
        ),
        StepSpec(
            id="docflow_delta_gate",
            label="docflow_delta_gate",
            kind="gate",
            run=docflow_delta_gate.main,
        ),
    ),
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


def _set_default_env() -> None:
    os.environ.setdefault("GABION_DIRECT_RUN", "1")
    os.environ.setdefault("GABION_LSP_TIMEOUT_TICKS", "65000000")
    os.environ.setdefault("GABION_LSP_TIMEOUT_TICK_NS", "1000000")


def _run_step_callable(
    *,
    # dataflow-bundle: name, step_index, step_total
    name: str,
    step: StepSpec,
    step_index: int,
    step_total: int,
    step_heartbeat_seconds: float,
    print_fn: Callable[[str], None] = print,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> int:
    started = monotonic_fn()
    last_heartbeat = started
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(step.run)
        while True:
            check_deadline()
            try:
                return int(future.result(timeout=0.5))
            except concurrent.futures.TimeoutError:
                pass
            except Exception as exc:
                print_fn(f"{name} step failed: {step.label} ({exc})")
                return 1
            if step_heartbeat_seconds > 0:
                now = monotonic_fn()
                if (now - last_heartbeat) >= step_heartbeat_seconds:
                    elapsed = max(0.0, now - started)
                    print_fn(
                        f"{name} step heartbeat {step_index}/{step_total}: "
                        f"{step.label} elapsed_s={elapsed:.1f}"
                    )
                    last_heartbeat = now


def _run_triplet(
    name: str,
    steps: Sequence[StepSpec],
    *,
    run_step_fn: Callable[..., int] = _run_step_callable,
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
    normalized_steps = list(steps)
    exit_code = 0
    print_fn(f"{name} triplet start: steps={len(normalized_steps)}")
    with _deadline_scope():
        for index, step in enumerate(normalized_steps, start=1):
            check_deadline()
            started = monotonic_fn()
            print_fn(f"{name} step start {index}/{len(normalized_steps)}: {step.label}")
            step_exit = int(
                run_step_fn(
                    name=name,
                    step=step,
                    step_index=index,
                    step_total=len(normalized_steps),
                    step_heartbeat_seconds=heartbeat_seconds,
                    print_fn=print_fn,
                    monotonic_fn=monotonic_fn,
                )
            )
            elapsed = max(0.0, monotonic_fn() - started)
            print_fn(
                f"{name} step complete {index}/{len(normalized_steps)}: {step.label} "
                f"exit={step_exit} elapsed_s={elapsed:.1f}"
            )
            if step_exit != 0:
                print_fn(f"{name} step failed: {step.label} (exit {step_exit})")
                exit_code = exit_code or step_exit
                if step.kind == "emit":
                    print_fn(
                        f"{name} triplet aborting remaining steps because emit failed."
                    )
                    return exit_code
    print_fn(f"{name} triplet complete: exit={exit_code}")
    return exit_code


def main(
    *,
    triplets: Mapping[str, Sequence[StepSpec]] = TRIPLETS,
    run_triplet_fn: Callable[..., int] = _run_triplet,
    print_fn: Callable[[str], None] = print,
    pending_heartbeat_seconds: float | None = None,
) -> int:
    _set_default_env()
    triplet_map = {
        str(name): [step for step in steps]
        for name, steps in triplets.items()
    }
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
