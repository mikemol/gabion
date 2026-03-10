from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Mapping, Protocol

from gabion.analysis.foundation import aspf_io_boundary, wire_text_codec
from gabion.analysis.foundation.wire_types import WireObject, WireValue
from gabion.order_contract import sort_once

from gabion.analysis.aspf import aspf_resume_state
from gabion.analysis.aspf.aspf_core import AspfOneCell, AspfTwoCellWitness
from gabion.analysis.aspf.aspf_morphisms import CofibrationWitnessCarrier
from gabion.analysis.foundation.frozen_object_map import ObjectEntry, make_object_map

AspfExecutionTraceState = object


class _SurfaceRepresentativeMap(dict[str, str]):
    pass


class _MutableObjectMap(dict[str, WireValue]):
    pass


@dataclass(frozen=True)
class OneCellRecorded:
    state: AspfExecutionTraceState
    cell: AspfOneCell
    kind: str
    surface: str
    metadata_payload: WireObject


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
    normalized_value: WireValue
    phase: str


@dataclass(frozen=True)
class RunFinalized:
    state: AspfExecutionTraceState
    trace_payload: Mapping[str, WireValue]
    equivalence_payload: Mapping[str, WireValue]
    opportunities_payload: Mapping[str, WireValue]
    delta_ledger_payload: Mapping[str, WireValue]
    state_payload: Mapping[str, WireValue]


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

    def iter_one_cells(self) -> Iterator[WireObject]:
        return _iter_event_lines(path=self.one_cells_path)

    def iter_two_cell_witnesses(self) -> Iterator[WireObject]:
        return _iter_event_lines(path=self.two_cell_witnesses_path)

    def iter_cofibrations(self) -> Iterator[WireObject]:
        return _iter_event_lines(path=self.cofibrations_path)

    def iter_delta_records(self) -> Iterator[WireObject]:
        return _iter_event_lines(path=self.delta_records_path)


@dataclass
class AspfEventJournalSink(AspfEventSink):
    sink_root: Path
    one_cells_path: Path
    two_cell_witnesses_path: Path
    cofibrations_path: Path
    delta_records_path: Path
    surface_representatives: _SurfaceRepresentativeMap
    one_cell_count: int = 0
    two_cell_witness_count: int = 0
    cofibration_count: int = 0
    delta_record_count: int = 0

    @classmethod
    def create(cls, *, sink_root: Path) -> AspfEventJournalSink:
        sink_root.mkdir(parents=True, exist_ok=True)
        return cls(
            sink_root=sink_root,
            one_cells_path=sink_root / aspf_io_boundary.ONE_CELL_STREAM_FILENAME,
            two_cell_witnesses_path=sink_root / aspf_io_boundary.TWO_CELL_STREAM_FILENAME,
            cofibrations_path=sink_root / aspf_io_boundary.COFIBRATION_STREAM_FILENAME,
            delta_records_path=sink_root / aspf_io_boundary.DELTA_RECORD_STREAM_FILENAME,
            surface_representatives=_SurfaceRepresentativeMap(),
        )

    def write_one_cell(self, event: OneCellRecorded) -> None:
        payload = _MutableObjectMap(event.cell.as_dict())
        payload["kind"] = event.kind
        payload["surface"] = event.surface
        payload["metadata"] = event.metadata_payload
        _append_event_line(self.one_cells_path, payload)
        self.one_cell_count += 1

        analysis_state_value = event.metadata_payload.get("analysis_state")
        analysis_state = _analysis_state_text(analysis_state_value)
        mutation_target = "one_cells.%s" % self.one_cell_count
        delta_record = aspf_resume_state.append_delta_record(
            records=[],
            event_kind=event.kind,
            phase=_phase_from_basis_path(event.cell.basis_path),
            analysis_state=analysis_state,
            mutation_target=mutation_target,
            mutation_value=make_object_map(
                [
                    ObjectEntry("source", event.cell.source.label),
                    ObjectEntry("target", event.cell.target.label),
                    ObjectEntry("representative", event.cell.representative),
                    ObjectEntry("surface", event.surface),
                    ObjectEntry("metadata", event.metadata_payload),
                ]
            ),
            one_cell_ref=mutation_target,
        )
        _append_event_line(self.delta_records_path, delta_record)
        self.delta_record_count += 1

    def write_two_cell(self, event: TwoCellWitnessRecorded) -> None:
        _append_event_line(self.two_cell_witnesses_path, event.witness.as_dict())
        self.two_cell_witness_count += 1

    def write_cofibration(self, event: CofibrationRecorded) -> None:
        _append_event_line(self.cofibrations_path, event.carrier.as_dict())
        self.cofibration_count += 1

    def write_surface_update(self, event: SemanticSurfaceUpdated) -> None:
        self.surface_representatives[event.surface] = event.representative
        delta_record = aspf_resume_state.append_delta_record(
            records=[],
            event_kind="semantic_surface_projection",
            phase=event.phase,
            analysis_state=None,
            mutation_target="semantic_surfaces.%s" % event.surface,
            mutation_value=event.normalized_value,
            one_cell_ref=None,
        )
        _append_event_line(self.delta_records_path, delta_record)
        self.delta_record_count += 1

    def write_finalize(self, event: RunFinalized) -> None:
        return None

    def close(self) -> None:
        manifest = make_object_map(
            [
                ObjectEntry("one_cell_count", self.one_cell_count),
                ObjectEntry("two_cell_witness_count", self.two_cell_witness_count),
                ObjectEntry("cofibration_count", self.cofibration_count),
                ObjectEntry("delta_record_count", self.delta_record_count),
                ObjectEntry(
                    "surface_representatives",
                    _sorted_surface_representatives(self.surface_representatives),
                ),
            ]
        )
        wire_text_codec.write_pretty_mapping(
            self.sink_root / aspf_io_boundary.MANIFEST_FILENAME,
            manifest,
        )
        wire_text_codec.append_trailing_newline(
            self.sink_root / aspf_io_boundary.MANIFEST_FILENAME,
        )

    def build_index(self) -> AspfTraceSinkIndex:
        return AspfTraceSinkIndex(
            one_cell_count=self.one_cell_count,
            two_cell_witness_count=self.two_cell_witness_count,
            cofibration_count=self.cofibration_count,
            delta_record_count=self.delta_record_count,
            surface_representatives=_SurfaceRepresentativeMap(self.surface_representatives),
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
        analysis_state = _analysis_state_text(analysis_state_value)
        raw_metadata: WireObject = make_object_map(
            [
                ObjectEntry("kind", event.kind),
                ObjectEntry("surface", event.surface),
                ObjectEntry("metadata", event.metadata_payload),
            ]
        )
        event.state.one_cell_metadata.append(raw_metadata)
        mutation_target = "one_cells.%s" % len(event.state.one_cells)
        phase = _phase_from_basis_path(event.cell.basis_path)
        aspf_resume_state.append_delta_record(
            records=event.state.delta_records,
            event_kind=event.kind,
            phase=phase,
            analysis_state=analysis_state,
            mutation_target=mutation_target,
            mutation_value=make_object_map(
                [
                    ObjectEntry("source", event.cell.source.label),
                    ObjectEntry("target", event.cell.target.label),
                    ObjectEntry("representative", event.cell.representative),
                    ObjectEntry("surface", event.surface),
                    ObjectEntry("metadata", event.metadata_payload),
                ]
            ),
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
            mutation_target="semantic_surfaces.%s" % event.surface,
            mutation_value=event.normalized_value,
            one_cell_ref=None,
        )

    def visit_run_finalized(self, event: RunFinalized) -> None:
        pass

    def on_finalize(self, event: RunFinalized) -> None:
        pass


def _append_event_line(path: Path, payload: Mapping[str, object]) -> None:
    wire_text_codec.append_line(path, payload)


def _iter_event_lines(*, path: Path) -> Iterator[WireObject]:
    return _EVENT_LINE_ITERATORS[path.exists()](path)


def _iter_missing_event_lines(_: Path) -> Iterator[WireObject]:
    return iter([])


def _iter_existing_event_lines(path: Path) -> Iterator[WireObject]:
    return map(_event_line_payload, wire_text_codec.iter_nonempty_lines(path))


def _strip_text(raw: str) -> str:
    return raw.strip()


def _is_nonempty_text(value: str) -> bool:
    return value != ""


def _event_line_payload(raw: str) -> WireObject:
    parsed = wire_text_codec.decode_text(raw)
    payload = _WIRE_MAPPING_COERCERS[_is_dict(parsed)](parsed)
    entries = map(
        lambda key: ObjectEntry(_TEXT_COERCERS[_is_str(key)](key), payload[key]),
        payload,
    )
    return make_object_map(entries)


def _none_text(_: WireValue) -> str:
    return ""


def _analysis_state_text(value: WireValue) -> str:
    return _ANALYSIS_STATE_TEXT_COERCERS[value is not None](value)


def _phase_from_basis_path(basis_path: tuple[str, ...]) -> str:
    return _PHASE_COERCERS[len(basis_path) > 0](basis_path)


def _phase_runtime(_: tuple[str, ...]) -> str:
    return "runtime"


def _phase_first(basis_path: tuple[str, ...]) -> str:
    return basis_path[0]


def _sorted_surface_representatives(
    surface_representatives: Mapping[str, str],
) -> Mapping[str, str]:
    ordered_surfaces = sort_once(
        surface_representatives,
        source="aspf_stream.AspfEventJournalSink.close.surface_representatives",
    )
    return _SurfaceRepresentativeMap(
        map(
            lambda surface: (surface, surface_representatives[surface]),
            ordered_surfaces,
        )
    )


_TEXT_COERCERS: list[Callable[[WireValue], str]] = [lambda _: "", lambda value: value]
_ANALYSIS_STATE_TEXT_COERCERS: list[Callable[[WireValue], str]] = [
    _none_text,
    lambda value: _TEXT_COERCERS[_is_str(value)](value),
]
_PHASE_COERCERS: list[Callable[[tuple[str, ...]], str]] = [_phase_runtime, _phase_first]
_EVENT_LINE_ITERATORS: list[Callable[[Path], Iterator[WireObject]]] = [
    _iter_missing_event_lines,
    _iter_existing_event_lines,
]


def _none_mapping(_: WireValue) -> Mapping[str, WireValue]:
    return make_object_map([])


def _identity_mapping(value: WireValue) -> Mapping[str, WireValue]:
    return value


_WIRE_MAPPING_COERCERS: list[Callable[[WireValue], Mapping[str, WireValue]]] = [
    _none_mapping,
    _identity_mapping,
]


def _is_dict(value: WireValue) -> bool:
    return type(value) is dict


def _is_str(value: WireValue) -> bool:
    return type(value) is str
