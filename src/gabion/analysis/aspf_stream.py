from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Mapping, Protocol, cast

from gabion.json_types import JSONObject
from gabion.order_contract import sort_once

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
    surface: str
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
    state_payload: Mapping[str, object]


class AspfEventVisitor(Protocol):
    def visit_one_cell_recorded(self, event: OneCellRecorded) -> None: ...

    def visit_two_cell_witness_recorded(self, event: TwoCellWitnessRecorded) -> None: ...

    def visit_cofibration_recorded(self, event: CofibrationRecorded) -> None: ...

    def visit_semantic_surface_updated(self, event: SemanticSurfaceUpdated) -> None: ...

    def visit_run_finalized(self, event: RunFinalized) -> None: ...

    def on_finalize(self, event: RunFinalized) -> None: ...


class AspfEventSink(Protocol):
    def write_one_cell(self, event: OneCellRecorded) -> None: ...

    def write_two_cell(self, event: TwoCellWitnessRecorded) -> None: ...

    def write_cofibration(self, event: CofibrationRecorded) -> None: ...

    def write_surface_update(self, event: SemanticSurfaceUpdated) -> None: ...

    def write_finalize(self, event: RunFinalized) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class AspfTraceSinkIndex:
    one_cell_count: int
    two_cell_witness_count: int
    cofibration_count: int
    delta_record_count: int
    surface_representatives: Mapping[str, str]
    one_cells_path: Path
    two_cell_witnesses_path: Path
    cofibrations_path: Path
    delta_records_path: Path

    def iter_one_cells(self) -> Iterator[JSONObject]:
        return _iter_jsonl(path=self.one_cells_path)

    def iter_two_cell_witnesses(self) -> Iterator[JSONObject]:
        return _iter_jsonl(path=self.two_cell_witnesses_path)

    def iter_cofibrations(self) -> Iterator[JSONObject]:
        return _iter_jsonl(path=self.cofibrations_path)

    def iter_delta_records(self) -> Iterator[JSONObject]:
        return _iter_jsonl(path=self.delta_records_path)


@dataclass
class AspfJsonlEventSink(AspfEventSink):
    sink_root: Path
    one_cells_path: Path
    two_cell_witnesses_path: Path
    cofibrations_path: Path
    delta_records_path: Path
    surface_representatives: dict[str, str]
    one_cell_count: int = 0
    two_cell_witness_count: int = 0
    cofibration_count: int = 0
    delta_record_count: int = 0

    @classmethod
    def create(cls, *, sink_root: Path) -> AspfJsonlEventSink:
        sink_root.mkdir(parents=True, exist_ok=True)
        return cls(
            sink_root=sink_root,
            one_cells_path=sink_root / "one_cells.jsonl",
            two_cell_witnesses_path=sink_root / "two_cell_witnesses.jsonl",
            cofibrations_path=sink_root / "cofibrations.jsonl",
            delta_records_path=sink_root / "delta_records.jsonl",
            surface_representatives={},
        )

    def write_one_cell(self, event: OneCellRecorded) -> None:
        payload = event.cell.as_dict()
        payload["kind"] = str(event.kind)
        payload["surface"] = str(event.surface)
        payload["metadata"] = event.metadata_payload
        _append_jsonl(self.one_cells_path, payload)
        self.one_cell_count += 1

        analysis_state_value = event.metadata_payload.get("analysis_state")
        analysis_state = (
            str(analysis_state_value) if analysis_state_value is not None else None
        )
        mutation_target = f"one_cells.{self.one_cell_count}"
        delta_record = aspf_resume_state.append_delta_record(
            records=[],
            event_kind=event.kind,
            phase=event.cell.basis_path[0] if event.cell.basis_path else "runtime",
            analysis_state=analysis_state,
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
        _append_jsonl(self.delta_records_path, delta_record)
        self.delta_record_count += 1

    def write_two_cell(self, event: TwoCellWitnessRecorded) -> None:
        _append_jsonl(self.two_cell_witnesses_path, event.witness.as_dict())
        self.two_cell_witness_count += 1

    def write_cofibration(self, event: CofibrationRecorded) -> None:
        _append_jsonl(self.cofibrations_path, event.carrier.as_dict())
        self.cofibration_count += 1

    def write_surface_update(self, event: SemanticSurfaceUpdated) -> None:
        self.surface_representatives[event.surface] = event.representative
        delta_record = aspf_resume_state.append_delta_record(
            records=[],
            event_kind="semantic_surface_projection",
            phase=event.phase,
            analysis_state=None,
            mutation_target=f"semantic_surfaces.{event.surface}",
            mutation_value=event.normalized_value,
            one_cell_ref=None,
        )
        _append_jsonl(self.delta_records_path, delta_record)
        self.delta_record_count += 1

    def write_finalize(self, event: RunFinalized) -> None:
        return None

    def close(self) -> None:
        manifest = {
            "one_cell_count": self.one_cell_count,
            "two_cell_witness_count": self.two_cell_witness_count,
            "cofibration_count": self.cofibration_count,
            "delta_record_count": self.delta_record_count,
            "surface_representatives": {
                surface: self.surface_representatives[surface]
                for surface in sort_once(
                    self.surface_representatives,
                    source="aspf_stream.AspfJsonlEventSink.close.surface_representatives",
                )
            },
        }
        (self.sink_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )

    def build_index(self) -> AspfTraceSinkIndex:
        return AspfTraceSinkIndex(
            one_cell_count=self.one_cell_count,
            two_cell_witness_count=self.two_cell_witness_count,
            cofibration_count=self.cofibration_count,
            delta_record_count=self.delta_record_count,
            surface_representatives=dict(self.surface_representatives),
            one_cells_path=self.one_cells_path,
            two_cell_witnesses_path=self.two_cell_witnesses_path,
            cofibrations_path=self.cofibrations_path,
            delta_records_path=self.delta_records_path,
        )


@dataclass
class AspfSinkVisitor:
    sink: AspfEventSink

    def visit_one_cell_recorded(self, event: OneCellRecorded) -> None:
        self.sink.write_one_cell(event)

    def visit_two_cell_witness_recorded(self, event: TwoCellWitnessRecorded) -> None:
        self.sink.write_two_cell(event)

    def visit_cofibration_recorded(self, event: CofibrationRecorded) -> None:
        self.sink.write_cofibration(event)

    def visit_semantic_surface_updated(self, event: SemanticSurfaceUpdated) -> None:
        self.sink.write_surface_update(event)

    def visit_run_finalized(self, event: RunFinalized) -> None:
        self.sink.write_finalize(event)

    def on_finalize(self, event: RunFinalized) -> None:
        return None


class AspfInMemoryCompatibilityVisitor:
    """Reconstructs legacy in-memory side effects for transition compatibility."""

    def visit_one_cell_recorded(self, event: OneCellRecorded) -> None:
        event.state.one_cells.append(event.cell)
        analysis_state_value = event.metadata_payload.get("analysis_state")
        analysis_state = (
            str(analysis_state_value) if analysis_state_value is not None else None
        )
        raw_metadata: JSONObject = {
            "kind": event.kind,
            "surface": event.surface,
            "metadata": event.metadata_payload,
        }
        event.state.one_cell_metadata.append(raw_metadata)
        mutation_target = f"one_cells.{len(event.state.one_cells)}"
        phase = event.cell.basis_path[0] if event.cell.basis_path else "runtime"
        aspf_resume_state.append_delta_record(
            records=event.state.delta_records,
            event_kind=event.kind,
            phase=str(phase),
            analysis_state=analysis_state,
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
        pass

    def on_finalize(self, event: RunFinalized) -> None:
        pass


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=False))
        handle.write("\n")


def _iter_jsonl(*, path: Path) -> Iterator[JSONObject]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            payload = cast(Mapping[str, object], json.loads(line))
            yield {str(key): payload[key] for key in payload}
