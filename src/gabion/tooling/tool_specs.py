# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from gabion.order_contract import sort_once
from gabion.tooling import (
    ambiguity_delta_gate,
    annotation_drift_orphaned_gate,
    delta_advisory,
    delta_state_emit,
    docflow_delta_emit,
    docflow_delta_gate,
    obsolescence_delta_gate,
    obsolescence_delta_unmapped_gate,
)

StepKind = Literal["emit", "advisory", "gate"]


@dataclass(frozen=True)
class ToolSpec:
    id: str
    label: str
    kind: StepKind
    run: Callable[[], int]
    triplet: str | None = None
    include_dataflow_stage_gate: bool = False


def triplet_resume_checkpoint_path(triplet_name: str) -> Path:
    normalized = triplet_name.strip().lower().replace("-", "_")
    return Path("artifacts/audit_reports") / f"dataflow_resume_checkpoint_ci_{normalized}.json"


def run_obsolescence_emit(
    *,
    run_emit: Callable[..., int] = delta_state_emit.obsolescence_main,
) -> int:
    return run_emit(resume_checkpoint=triplet_resume_checkpoint_path("obsolescence"))


def run_annotation_drift_emit(
    *,
    run_emit: Callable[..., int] = delta_state_emit.annotation_drift_main,
) -> int:
    return run_emit(resume_checkpoint=triplet_resume_checkpoint_path("annotation_drift"))


def run_ambiguity_emit(
    *,
    run_emit: Callable[..., int] = delta_state_emit.ambiguity_main,
) -> int:
    return run_emit(resume_checkpoint=triplet_resume_checkpoint_path("ambiguity"))


ALL_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        id="obsolescence_delta_emit",
        label="obsolescence_delta_emit",
        kind="emit",
        run=run_obsolescence_emit,
        triplet="obsolescence",
    ),
    ToolSpec(
        id="obsolescence_delta_advisory",
        label="obsolescence_delta_advisory",
        kind="advisory",
        run=delta_advisory.obsolescence_main,
        triplet="obsolescence",
    ),
    ToolSpec(
        id="obsolescence_delta_gate",
        label="obsolescence_delta_gate",
        kind="gate",
        run=obsolescence_delta_gate.main,
        triplet="obsolescence",
        include_dataflow_stage_gate=True,
    ),
    ToolSpec(
        id="obsolescence_delta_unmapped_gate",
        label="obsolescence_delta_unmapped_gate",
        kind="gate",
        run=obsolescence_delta_unmapped_gate.main,
        triplet="obsolescence",
        include_dataflow_stage_gate=True,
    ),
    ToolSpec(
        id="annotation_drift_delta_emit",
        label="annotation_drift_delta_emit",
        kind="emit",
        run=run_annotation_drift_emit,
        triplet="annotation_drift",
    ),
    ToolSpec(
        id="annotation_drift_delta_advisory",
        label="annotation_drift_delta_advisory",
        kind="advisory",
        run=delta_advisory.annotation_drift_main,
        triplet="annotation_drift",
    ),
    ToolSpec(
        id="annotation_drift_orphaned_gate",
        label="annotation_drift_orphaned_gate",
        kind="gate",
        run=annotation_drift_orphaned_gate.main,
        triplet="annotation_drift",
        include_dataflow_stage_gate=True,
    ),
    ToolSpec(
        id="ambiguity_delta_emit",
        label="ambiguity_delta_emit",
        kind="emit",
        run=run_ambiguity_emit,
        triplet="ambiguity",
    ),
    ToolSpec(
        id="ambiguity_delta_advisory",
        label="ambiguity_delta_advisory",
        kind="advisory",
        run=delta_advisory.ambiguity_main,
        triplet="ambiguity",
    ),
    ToolSpec(
        id="ambiguity_delta_gate",
        label="ambiguity_delta_gate",
        kind="gate",
        run=ambiguity_delta_gate.main,
        triplet="ambiguity",
        include_dataflow_stage_gate=True,
    ),
    ToolSpec(
        id="docflow_delta_emit",
        label="docflow_delta_emit",
        kind="emit",
        run=docflow_delta_emit.main,
        triplet="docflow",
    ),
    ToolSpec(
        id="docflow_delta_advisory",
        label="docflow_delta_advisory",
        kind="advisory",
        run=delta_advisory.docflow_main,
        triplet="docflow",
    ),
    ToolSpec(
        id="docflow_delta_gate",
        label="docflow_delta_gate",
        kind="gate",
        run=docflow_delta_gate.main,
        triplet="docflow",
    ),
)


def triplet_specs_map() -> dict[str, tuple[ToolSpec, ...]]:
    grouped: dict[str, list[ToolSpec]] = {}
    for spec in ALL_TOOL_SPECS:
        grouped.setdefault(str(spec.triplet), []).append(spec)
    return {
        name: tuple(grouped[name])
        for name in sort_once(grouped.keys(), source = 'src/gabion/tooling/tool_specs.py:163')
    }


def dataflow_stage_gate_specs() -> tuple[ToolSpec, ...]:
    return tuple(
        spec
        for spec in ALL_TOOL_SPECS
        if spec.kind == "gate" and spec.include_dataflow_stage_gate
    )
