from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Protocol

from gabion.json_types import JSONObject

from . import aspf_resume_state
from .aspf_core import AspfOneCell, AspfTwoCellWitness
from .aspf_morphisms import CofibrationWitnessCarrier

if TYPE_CHECKING:
    from .aspf_execution_fibration import AspfExecutionTraceState


@dataclass(frozen=True)
class OneCellRecorded:
    state: AspfExecutionTraceState
    cell: AspfOneCell
    kind: str
    surface: str | None
    metadata_payload: JSONObject


@dataclass(frozen=True)
class TwoCellWitnessRecorded:
    state: AspfExecutionTraceState
    witness: AspfTwoCellWitness


@dataclass(frozen=True)
class CofibrationRecorded:
    state: AspfExecutionTraceState
    carrier: CofibrationWitnessCarrier


@dataclass(frozen=True)
class SemanticSurfaceUpdated:
    state: AspfExecutionTraceState
    surface: str
    representative: str
    normalized_value: object
    phase: str


@dataclass(frozen=True)
class RunFinalized:
    state: AspfExecutionTraceState
    trace_payload: Mapping[str, object]
    equivalence_payload: Mapping[str, object]
    opportunities_payload: Mapping[str, object]
    delta_ledger_payload: Mapping[str, object]
    state_payload: Mapping[str, object] | None


class AspfEventVisitor(Protocol):
    def visit_one_cell_recorded(self, event: OneCellRecorded) -> None: ...

    def visit_two_cell_witness_recorded(self, event: TwoCellWitnessRecorded) -> None: ...

    def visit_cofibration_recorded(self, event: CofibrationRecorded) -> None: ...

    def visit_semantic_surface_updated(self, event: SemanticSurfaceUpdated) -> None: ...

    def visit_run_finalized(self, event: RunFinalized) -> None: ...

    def on_finalize(self, event: RunFinalized) -> None: ...


class AspfInMemoryCompatibilityVisitor:
    """Reconstructs legacy in-memory side effects for transition compatibility."""

    def visit_one_cell_recorded(self, event: OneCellRecorded) -> None:
        event.state.one_cells.append(event.cell)
        raw_metadata: JSONObject = {
            "kind": event.kind,
            "surface": event.surface or "",
            "metadata": event.metadata_payload,
        }
        event.state.one_cell_metadata.append(raw_metadata)
        mutation_target = f"one_cells.{len(event.state.one_cells)}"
        phase = event.cell.basis_path[0] if event.cell.basis_path else "runtime"
        aspf_resume_state.append_delta_record(
            records=event.state.delta_records,
            event_kind=event.kind,
            phase=str(phase),
            analysis_state=(
                str(event.metadata_payload.get("analysis_state"))
                if isinstance(event.metadata_payload.get("analysis_state"), str)
                else None
            ),
            mutation_target=mutation_target,
            mutation_value={
                "source": str(event.cell.source),
                "target": str(event.cell.target),
                "representative": event.cell.representative,
                "surface": event.surface,
                "metadata": event.metadata_payload,
            },
            one_cell_ref=mutation_target,
        )

    def visit_two_cell_witness_recorded(self, event: TwoCellWitnessRecorded) -> None:
        event.state.two_cell_witnesses.append(event.witness)

    def visit_cofibration_recorded(self, event: CofibrationRecorded) -> None:
        event.state.cofibrations.append(event.carrier)

    def visit_semantic_surface_updated(self, event: SemanticSurfaceUpdated) -> None:
        event.state.surface_representatives[event.surface] = event.representative
        aspf_resume_state.append_delta_record(
            records=event.state.delta_records,
            event_kind="semantic_surface_projection",
            phase=event.phase,
            analysis_state=None,
            mutation_target=f"semantic_surfaces.{event.surface}",
            mutation_value=event.normalized_value,
            one_cell_ref=None,
        )

    def visit_run_finalized(self, event: RunFinalized) -> None:
        return None

    def on_finalize(self, event: RunFinalized) -> None:
        return None
