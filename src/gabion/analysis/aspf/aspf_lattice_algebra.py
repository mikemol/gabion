# gabion:decision_protocol_module
# gabion:ambiguity_boundary_module
from __future__ import annotations

import ast
import builtins
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from functools import lru_cache, reduce
import hashlib
import json
from itertools import chain, groupby
from pathlib import Path
from typing import Callable, Generic, Iterable, Iterator, Mapping, TypeVar
from gabion.analysis.kernel_vm.object_images import AugmentedRule
from gabion.invariants import never

_StreamItem = TypeVar("_StreamItem")

_LATTICE_CACHE_VERSION = "v5"
_MODULE_INGRESS_SYMBOLS = {
    "__annotations__",
    "__cached__",
    "__doc__",
    "__file__",
    "__loader__",
    "__name__",
    "__package__",
    "__spec__",
}

_AUGMENTED_RULE_OBJECT_IMAGE = AugmentedRule


@dataclass(frozen=True)
class ReplayableStream(Generic[_StreamItem]):
    factory: Callable[[], Iterator[_StreamItem]]

    def __iter__(self) -> Iterator[_StreamItem]:
        return self.factory()


def _stream_from_sequence(
    values: tuple[_StreamItem, ...],
) -> ReplayableStream[_StreamItem]:
    return ReplayableStream(factory=lambda: iter(values))


def canonical_site_identity(
    *,
    rel_path: str,
    qualname: str,
    line: int,
    column: int,
    node_kind: str,
    surface: str,
) -> str:
    return _stable_identity_hash(
        "site_identity",
        rel_path,
        qualname,
        line,
        column,
        node_kind,
        surface,
    )


def canonical_structural_identity(
    *,
    rel_path: str,
    qualname: str,
    structural_path: str,
    node_kind: str,
    surface: str,
) -> str:
    return _stable_identity_hash(
        "structural_identity",
        rel_path,
        qualname,
        structural_path,
        node_kind,
        surface,
    )


def _stable_identity_hash(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(_hash_part_bytes(part))
        digest.update(b"\x00")
    return digest.hexdigest()


def _hash_part_bytes(value: object) -> bytes:
    match value:
        case bool() as flag:
            return b"1" if flag else b"0"
        case int() as integer:
            magnitude = abs(integer)
            width = max(1, (magnitude.bit_length() + 7) // 8)
            sign = b"-" if integer < 0 else b"+"
            return sign + magnitude.to_bytes(width, byteorder="big", signed=False)
        case str() as text:
            return text.encode("utf-8")
        case bytes() as raw:
            return raw
        case _:
            return b"<unsupported>"


            never("unreachable wildcard match fall-through")
@dataclass(frozen=True)
class DataflowEvent:
    site_id: str
    site_identity: str
    rel_path: str
    qualname: str
    line: int
    column: int
    ordinal: int
    symbol: str
    event_kind: str
    node_kind: str


@dataclass(frozen=True)
class DataflowEdge:
    edge_id: str
    symbol: str
    source_site_id: str
    target_site_id: str
    source_site_identity: str
    target_site_identity: str
    source_ordinal: int
    target_ordinal: int


@dataclass(frozen=True)
class ExecutionEvent:
    site_id: str
    site_identity: str
    rel_path: str
    qualname: str
    line: int
    column: int
    ordinal: int
    node_kind: str
    event_kind: str


@dataclass(frozen=True)
class ExecutionEdge:
    edge_id: str
    source_site_id: str
    target_site_id: str
    source_site_identity: str
    target_site_identity: str
    source_ordinal: int
    target_ordinal: int


@dataclass(frozen=True)
class DataflowFiberBundle:
    rel_path: str
    qualname: str
    entry_site_id: str
    entry_site_identity: str
    events: ReplayableStream[DataflowEvent]
    edges: ReplayableStream[DataflowEdge]
    execution_events: ReplayableStream[ExecutionEvent]
    execution_edges: ReplayableStream[ExecutionEdge]
    module_bound_symbols: ReplayableStream[str] = field(
        default_factory=lambda: _stream_from_sequence(())
    )


@dataclass(frozen=True)
class DataFiber:
    rel_path: str
    qualname: str
    entry_site_id: str
    entry_site_identity: str
    events: ReplayableStream[DataflowEvent]
    edges: ReplayableStream[DataflowEdge]


@dataclass(frozen=True)
class ExecFiber:
    rel_path: str
    qualname: str
    entry_site_id: str
    entry_site_identity: str
    events: ReplayableStream[ExecutionEvent]
    edges: ReplayableStream[ExecutionEdge]


@dataclass(frozen=True)
class FiberBundle:
    rel_path: str
    qualname: str
    data: DataFiber
    exec: ExecFiber
    module_bound_symbols: ReplayableStream[str] = field(
        default_factory=lambda: _stream_from_sequence(())
    )


@dataclass(frozen=True)
class JoinWitness:
    left_ids: ReplayableStream[str]
    right_ids: ReplayableStream[str]
    result_ids: ReplayableStream[str]
    deterministic: bool


@dataclass(frozen=True)
class MeetWitness:
    left_ids: ReplayableStream[str]
    right_ids: ReplayableStream[str]
    result_ids: ReplayableStream[str]
    deterministic: bool


@dataclass(frozen=True)
class UnmappedWitness:
    source_kind: str
    source_site_id: str
    source_site_identity: str
    reason: str


@dataclass(frozen=True)
class ObligationIntro:
    obligation_id: str
    source_kind: str
    source_site_id: str
    source_site_identity: str
    reason: str
    introduced_by: str


@dataclass(frozen=True)
class ObligationErase:
    obligation_id: str
    erased_by: str
    reason: str


@dataclass(frozen=True)
class BoundaryCrossing:
    crossing_id: str
    branch_site_id: str
    branch_site_identity: str
    boundary_kind: str


@dataclass(frozen=True)
class ViolationWitness:
    violation_id: str
    boundary_crossing_id: str
    unresolved_obligation_ids: ReplayableStream[str]
    reason: str


@dataclass(frozen=True)
class BranchWitnessRequest:
    branch_line: int
    branch_column: int
    branch_node_kind: str
    required_symbols: tuple[str, ...]


@dataclass(frozen=True)
class NaturalityWitness:
    direction: str
    mapped_source_site_ids: ReplayableStream[str]
    mapped_target_site_ids: ReplayableStream[str]
    unmapped: ReplayableStream[UnmappedWitness]
    complete: bool


class ProtocolDischargeLevel(IntEnum):
    RAW_INGRESS = 0
    DECISION_TABLE = 1
    DECISION_BUNDLE = 2
    DECISION_PROTOCOL = 3
    INVARIANT_DISCHARGED = 4


class OutputCardinalityClass(IntEnum):
    CONSTANT = 0
    LINEAR = 1
    QUADRATIC = 2
    EXPONENTIAL_OR_UNBOUNDED = 3
    UNKNOWN_TOP = 4


class WorkGrowthClass(IntEnum):
    CONSTANT = 0
    LOGARITHMIC = 1
    LINEAR = 2
    N_LOG_N = 3
    QUADRATIC = 4
    EXPONENTIAL_OR_WORSE = 5
    UNKNOWN_TOP = 6


class GradeBoundaryKind(str, Enum):
    INGRESS_NORMALIZATION = "ingress_normalization"
    DECISION_PROTOCOL_BUILDER = "decision_protocol_builder"
    AGGREGATION_MATERIALIZATION = "aggregation_materialization"
    ANALYSIS_BUDGET = "analysis_budget"
    SEMANTIC_CARRIER_ADAPTER = "semantic_carrier_adapter"


@dataclass(frozen=True)
class GradeBoundaryMarker:
    kind: GradeBoundaryKind
    name: str
    source: str = ""
    line: int = 0

    def as_payload(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "name": self.name,
            "source": self.source,
            "line": self.line,
        }


@dataclass(frozen=True)
class DeterminismCostGrade:
    nullable_domain_cardinality: int
    type_domain_cardinality: int
    shape_domain_cardinality: int
    runtime_classification_count: int
    protocol_discharge_level: ProtocolDischargeLevel
    output_cardinality_class: OutputCardinalityClass
    work_growth_class: WorkGrowthClass

    @classmethod
    def unknown_top(cls) -> "DeterminismCostGrade":
        return cls(
            nullable_domain_cardinality=999,
            type_domain_cardinality=999,
            shape_domain_cardinality=999,
            runtime_classification_count=999,
            protocol_discharge_level=ProtocolDischargeLevel.RAW_INGRESS,
            output_cardinality_class=OutputCardinalityClass.UNKNOWN_TOP,
            work_growth_class=WorkGrowthClass.UNKNOWN_TOP,
        )

    def as_payload(self) -> dict[str, object]:
        return {
            "nullable_domain_cardinality": self.nullable_domain_cardinality,
            "type_domain_cardinality": self.type_domain_cardinality,
            "shape_domain_cardinality": self.shape_domain_cardinality,
            "runtime_classification_count": self.runtime_classification_count,
            "protocol_discharge_level": int(self.protocol_discharge_level),
            "protocol_discharge_label": self.protocol_discharge_level.name.lower(),
            "output_cardinality_class": int(self.output_cardinality_class),
            "output_cardinality_label": self.output_cardinality_class.name.lower(),
            "work_growth_class": int(self.work_growth_class),
            "work_growth_label": self.work_growth_class.name.lower(),
        }


@dataclass(frozen=True)
class CallEdgeGradeWitness:
    witness_id: str
    edge_site_identity: str
    edge_structural_identity: str
    caller_path: str
    caller_qualname: str
    caller_line: int
    caller_column: int
    callee_path: str
    callee_qualname: str
    callee_line: int
    callee_column: int
    call_line: int
    call_column: int
    edge_resolution_status: str
    edge_resolution_phase: str
    edge_kind: str
    boundary_marker: GradeBoundaryMarker | None
    caller_grade: DeterminismCostGrade
    callee_grade: DeterminismCostGrade
    monotone: bool
    failure_rule_ids: tuple[str, ...] = ()

    def as_payload(self) -> dict[str, object]:
        return {
            "witness_id": self.witness_id,
            "edge_site_identity": self.edge_site_identity,
            "edge_structural_identity": self.edge_structural_identity,
            "caller_path": self.caller_path,
            "caller_qualname": self.caller_qualname,
            "caller_line": self.caller_line,
            "caller_column": self.caller_column,
            "callee_path": self.callee_path,
            "callee_qualname": self.callee_qualname,
            "callee_line": self.callee_line,
            "callee_column": self.callee_column,
            "call_line": self.call_line,
            "call_column": self.call_column,
            "edge_resolution_status": self.edge_resolution_status,
            "edge_resolution_phase": self.edge_resolution_phase,
            "edge_kind": self.edge_kind,
            "boundary_marker": (
                None if self.boundary_marker is None else self.boundary_marker.as_payload()
            ),
            "caller_grade": self.caller_grade.as_payload(),
            "callee_grade": self.callee_grade.as_payload(),
            "monotone": self.monotone,
            "failure_rule_ids": list(self.failure_rule_ids),
        }


@dataclass(frozen=True)
class GradeMonotonicityViolation:
    violation_id: str
    witness_id: str
    rule_id: str
    path: str
    line: int
    column: int
    qualname: str
    callee_qualname: str
    message: str
    details: Mapping[str, object] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.rule_id}:{self.witness_id}"

    def as_payload(self) -> dict[str, object]:
        return {
            "violation_id": self.violation_id,
            "witness_id": self.witness_id,
            "rule_id": self.rule_id,
            "path": self.path,
            "line": self.line,
            "column": self.column,
            "qualname": self.qualname,
            "callee_qualname": self.callee_qualname,
            "message": self.message,
            "details": dict(self.details),
            "key": self.key,
        }


@dataclass(frozen=True)
class FrontierWitness:
    branch_site_id: str
    branch_site_identity: str
    branch_line: int
    branch_column: int
    branch_node_kind: str
    required_symbols: ReplayableStream[str]
    unresolved_symbols: ReplayableStream[str]
    data_anchor_site_id: str
    data_anchor_site_identity: str
    data_anchor_line: int
    data_anchor_column: int
    data_anchor_ordinal: int
    data_upstream_site_ids: ReplayableStream[str]
    data_upstream_site_identities: ReplayableStream[str]
    data_upstream_edge_ids: ReplayableStream[str]
    exec_frontier_site_id: str
    exec_frontier_site_identity: str
    exec_frontier_line: int
    exec_frontier_column: int
    exec_frontier_ordinal: int
    exec_upstream_site_ids: ReplayableStream[str]
    exec_upstream_site_identities: ReplayableStream[str]
    exec_upstream_edge_ids: ReplayableStream[str]
    bundle_event_count: int
    bundle_edge_count: int
    execution_event_count: int
    execution_edge_count: int
    data_exec_join: JoinWitness
    data_exec_meet: MeetWitness
    eta_data_to_exec: NaturalityWitness
    eta_exec_to_data: NaturalityWitness
    complete: bool
    obligations: ReplayableStream[ObligationIntro] = field(
        default_factory=lambda: _stream_from_sequence(())
    )
    erasures: ReplayableStream[ObligationErase] = field(
        default_factory=lambda: _stream_from_sequence(())
    )
    boundary_crossings: ReplayableStream[BoundaryCrossing] = field(
        default_factory=lambda: _stream_from_sequence(())
    )
    violation: ViolationWitness | None = None

    def as_payload(self) -> dict[str, object]:
        return {
            "branch_site_id": self.branch_site_id,
            "branch_site_identity": self.branch_site_identity,
            "branch_line": self.branch_line,
            "branch_column": self.branch_column,
            "branch_node_kind": self.branch_node_kind,
            "required_symbols": [item for item in self.required_symbols],
            "unresolved_symbols": [item for item in self.unresolved_symbols],
            "data_anchor_site_id": self.data_anchor_site_id,
            "data_anchor_site_identity": self.data_anchor_site_identity,
            "data_anchor_line": self.data_anchor_line,
            "data_anchor_column": self.data_anchor_column,
            "data_anchor_ordinal": self.data_anchor_ordinal,
            "data_upstream_site_ids": [item for item in self.data_upstream_site_ids],
            "data_upstream_site_identities": [
                item for item in self.data_upstream_site_identities
            ],
            "data_upstream_edge_ids": [item for item in self.data_upstream_edge_ids],
            "exec_frontier_site_id": self.exec_frontier_site_id,
            "exec_frontier_site_identity": self.exec_frontier_site_identity,
            "exec_frontier_line": self.exec_frontier_line,
            "exec_frontier_column": self.exec_frontier_column,
            "exec_frontier_ordinal": self.exec_frontier_ordinal,
            "exec_upstream_site_ids": [item for item in self.exec_upstream_site_ids],
            "exec_upstream_site_identities": [
                item for item in self.exec_upstream_site_identities
            ],
            "exec_upstream_edge_ids": [item for item in self.exec_upstream_edge_ids],
            "bundle_event_count": self.bundle_event_count,
            "bundle_edge_count": self.bundle_edge_count,
            "execution_event_count": self.execution_event_count,
            "execution_edge_count": self.execution_edge_count,
            "data_exec_join": {
                "left_ids": [item for item in self.data_exec_join.left_ids],
                "right_ids": [item for item in self.data_exec_join.right_ids],
                "result_ids": [item for item in self.data_exec_join.result_ids],
                "deterministic": self.data_exec_join.deterministic,
            },
            "data_exec_meet": {
                "left_ids": [item for item in self.data_exec_meet.left_ids],
                "right_ids": [item for item in self.data_exec_meet.right_ids],
                "result_ids": [item for item in self.data_exec_meet.result_ids],
                "deterministic": self.data_exec_meet.deterministic,
            },
            "eta_data_to_exec": {
                "direction": self.eta_data_to_exec.direction,
                "mapped_source_site_ids": [
                    item for item in self.eta_data_to_exec.mapped_source_site_ids
                ],
                "mapped_target_site_ids": [
                    item for item in self.eta_data_to_exec.mapped_target_site_ids
                ],
                "unmapped": [
                    {
                        "source_kind": item.source_kind,
                        "source_site_id": item.source_site_id,
                        "source_site_identity": item.source_site_identity,
                        "reason": item.reason,
                    }
                    for item in self.eta_data_to_exec.unmapped
                ],
                "complete": self.eta_data_to_exec.complete,
            },
            "eta_exec_to_data": {
                "direction": self.eta_exec_to_data.direction,
                "mapped_source_site_ids": [
                    item for item in self.eta_exec_to_data.mapped_source_site_ids
                ],
                "mapped_target_site_ids": [
                    item for item in self.eta_exec_to_data.mapped_target_site_ids
                ],
                "unmapped": [
                    {
                        "source_kind": item.source_kind,
                        "source_site_id": item.source_site_id,
                        "source_site_identity": item.source_site_identity,
                        "reason": item.reason,
                    }
                    for item in self.eta_exec_to_data.unmapped
                ],
                "complete": self.eta_exec_to_data.complete,
            },
            "complete": self.complete,
            "obligations": [
                {
                    "obligation_id": item.obligation_id,
                    "source_kind": item.source_kind,
                    "source_site_id": item.source_site_id,
                    "source_site_identity": item.source_site_identity,
                    "reason": item.reason,
                    "introduced_by": item.introduced_by,
                }
                for item in self.obligations
            ],
            "erasures": [
                {
                    "obligation_id": item.obligation_id,
                    "erased_by": item.erased_by,
                    "reason": item.reason,
                }
                for item in self.erasures
            ],
            "boundary_crossings": [
                {
                    "crossing_id": item.crossing_id,
                    "branch_site_id": item.branch_site_id,
                    "branch_site_identity": item.branch_site_identity,
                    "boundary_kind": item.boundary_kind,
                }
                for item in self.boundary_crossings
            ],
            "violation": (
                None
                if self.violation is None
                else {
                    "violation_id": self.violation.violation_id,
                    "boundary_crossing_id": self.violation.boundary_crossing_id,
                    "unresolved_obligation_ids": [
                        item for item in self.violation.unresolved_obligation_ids
                    ],
                    "reason": self.violation.reason,
                }
            ),
        }


@dataclass(frozen=True)
class _DefinitionRecord:
    symbol: str
    event: DataflowEvent


def build_dataflow_fiber_bundle_for_qualname(
    *,
    rel_path: str,
    module_tree: ast.AST,
    qualname: str,
) -> DataflowFiberBundle:
    scope = _resolve_scope(module_tree=module_tree, qualname=qualname)
    collector = _DataflowCollector(
        rel_path=rel_path,
        qualname=scope.qualname,
        entry_line=scope.start_line,
        entry_column=scope.start_column,
    )
    collector.record_argument_definitions(scope.arguments)
    list(map(collector.visit, scope.body))
    execution_collector = _ExecutionCollector(
        rel_path=rel_path,
        qualname=scope.qualname,
        entry_line=scope.start_line,
        entry_column=scope.start_column,
    )
    list(execution_collector.record_scope_statements(scope.body))
    collected_events = tuple(collector.events)
    collected_edges = tuple(
        sorted(
            collector.edges,
            key=lambda edge: (edge.source_ordinal, edge.target_ordinal, edge.symbol),
        )
    )
    execution_events = tuple(
        sorted(
            execution_collector.events,
            key=lambda event: (event.ordinal, event.line, event.column, event.node_kind),
        )
    )
    execution_edges = tuple(
        sorted(
            execution_collector.edges,
            key=lambda edge: (edge.source_ordinal, edge.target_ordinal),
        )
    )
    module_bound_symbols = _module_bound_symbols(module_tree)
    return DataflowFiberBundle(
        rel_path=rel_path,
        qualname=scope.qualname,
        entry_site_id=collector.entry_site_id,
        entry_site_identity=collector.entry_site_identity,
        events=_stream_from_sequence(collected_events),
        edges=_stream_from_sequence(collected_edges),
        execution_events=_stream_from_sequence(execution_events),
        execution_edges=_stream_from_sequence(execution_edges),
        module_bound_symbols=_stream_from_sequence(module_bound_symbols),
    )


def build_fiber_bundle_for_qualname(
    *,
    rel_path: str,
    module_tree: ast.AST,
    qualname: str,
) -> FiberBundle:
    materialized = build_dataflow_fiber_bundle_for_qualname(
        rel_path=rel_path,
        module_tree=module_tree,
        qualname=qualname,
    )
    return FiberBundle(
        rel_path=materialized.rel_path,
        qualname=materialized.qualname,
        data=DataFiber(
            rel_path=materialized.rel_path,
            qualname=materialized.qualname,
            entry_site_id=materialized.entry_site_id,
            entry_site_identity=materialized.entry_site_identity,
            events=materialized.events,
            edges=materialized.edges,
        ),
        exec=ExecFiber(
            rel_path=materialized.rel_path,
            qualname=materialized.qualname,
            entry_site_id=materialized.entry_site_id,
            entry_site_identity=materialized.entry_site_identity,
            events=materialized.execution_events,
            edges=materialized.execution_edges,
        ),
        module_bound_symbols=materialized.module_bound_symbols,
    )


def upstream_closure(
    *,
    origins: Iterable[str],
    predecessors: dict[str, list[str]],
) -> ReplayableStream[str]:
    closure = set()
    stack = list(origins)
    while stack:
        current = stack.pop()
        if current in closure:
            continue
        closure.add(current)
        stack.extend(predecessors.get(current, []))
    return _stream_from_sequence(tuple(sorted(closure)))


def join(
    *,
    left_ids: Iterable[str],
    right_ids: Iterable[str],
) -> JoinWitness:
    left = tuple(sorted(dict.fromkeys(left_ids)))
    right = tuple(sorted(dict.fromkeys(right_ids)))
    return JoinWitness(
        left_ids=_stream_from_sequence(left),
        right_ids=_stream_from_sequence(right),
        result_ids=_stream_from_sequence(tuple(sorted(dict.fromkeys([*left, *right])))),
        deterministic=True,
    )


def meet(
    *,
    left_ids: Iterable[str],
    right_ids: Iterable[str],
) -> MeetWitness:
    left = tuple(sorted(dict.fromkeys(left_ids)))
    right = tuple(sorted(dict.fromkeys(right_ids)))
    return MeetWitness(
        left_ids=_stream_from_sequence(left),
        right_ids=_stream_from_sequence(right),
        result_ids=_stream_from_sequence(tuple(sorted(set(left).intersection(right)))),
        deterministic=True,
    )


def frontier(
    *,
    rel_path: str,
    qualname: str,
    bundle: FiberBundle,
    branch_line: int,
    branch_column: int,
    branch_node_kind: str,
    required_symbols: Iterable[str],
) -> FrontierWitness:
    data_events = tuple(bundle.data.events)
    data_edges = tuple(bundle.data.edges)
    execution_events = tuple(bundle.exec.events)
    execution_edges = tuple(bundle.exec.edges)
    normalized_required = tuple(sorted(dict.fromkeys(required_symbols)))
    definition_records = list(
        _iter_latest_definitions_before_branch(
            events=data_events,
            branch_line=branch_line,
            branch_column=branch_column,
            symbols=normalized_required,
        )
    )
    defs_by_symbol = _definition_lookup(definition_records)
    symbols_defined_in_scope = _defined_symbols(data_events)
    module_bound_symbols = set(bundle.module_bound_symbols)
    unresolved = tuple(
        filter(
            _missing_local_or_ingress_definition(
                lookup=defs_by_symbol,
                symbols_defined_in_scope=symbols_defined_in_scope,
                module_bound_symbols=module_bound_symbols,
            ),
            normalized_required,
        )
    )
    upstream_events = tuple(
        sorted(
            map(lambda item: item.event, definition_records),
            key=lambda event: event.ordinal,
        )
    )
    anchor_event = _anchor_event(
        upstream_events=upstream_events,
        events=data_events,
    )
    upstream_site_ids = tuple(map(_site_id_from_event, upstream_events))
    upstream_site_identities = tuple(map(_site_identity_from_event, upstream_events))
    upstream_edge_ids = tuple(
        sorted(
            map(
                _edge_id_from_edge,
                filter(_edge_targets_in(upstream_site_ids), data_edges),
            )
        )
    )
    branch_site_identity = _branch_site_identity(
        rel_path=rel_path,
        qualname=qualname,
        line=branch_line,
        column=branch_column,
        node_kind=branch_node_kind,
    )
    execution_frontier = _execution_recombination_frontier(
        execution_events=execution_events,
        execution_edges=execution_edges,
        branch_site_identity=branch_site_identity,
        required_dataflow_events=upstream_events,
    )
    eta_forward = eta_data_to_exec(
        data_events=data_events,
        exec_events=execution_events,
    )
    eta_reverse = eta_exec_to_data(
        data_events=data_events,
        exec_events=execution_events,
    )
    data_exec_join = join(
        left_ids=upstream_site_ids,
        right_ids=execution_frontier.upstream_site_ids,
    )
    data_exec_meet = meet(
        left_ids=upstream_site_ids,
        right_ids=execution_frontier.upstream_site_ids,
    )
    return FrontierWitness(
        branch_site_id=_site_id(
            rel_path=rel_path,
            qualname=qualname,
            line=branch_line,
            column=branch_column,
            event_kind="branch",
            symbol="",
            node_kind=branch_node_kind,
        ),
        branch_site_identity=branch_site_identity,
        branch_line=branch_line,
        branch_column=branch_column,
        branch_node_kind=branch_node_kind,
        required_symbols=_stream_from_sequence(normalized_required),
        unresolved_symbols=_stream_from_sequence(unresolved),
        data_anchor_site_id=anchor_event.site_id,
        data_anchor_site_identity=anchor_event.site_identity,
        data_anchor_line=anchor_event.line,
        data_anchor_column=anchor_event.column,
        data_anchor_ordinal=anchor_event.ordinal,
        data_upstream_site_ids=_stream_from_sequence(upstream_site_ids),
        data_upstream_site_identities=_stream_from_sequence(upstream_site_identities),
        data_upstream_edge_ids=_stream_from_sequence(upstream_edge_ids),
        exec_frontier_site_id=execution_frontier.site_id,
        exec_frontier_site_identity=execution_frontier.site_identity,
        exec_frontier_line=execution_frontier.line,
        exec_frontier_column=execution_frontier.column,
        exec_frontier_ordinal=execution_frontier.ordinal,
        exec_upstream_site_ids=execution_frontier.upstream_site_ids,
        exec_upstream_site_identities=execution_frontier.upstream_site_identities,
        exec_upstream_edge_ids=execution_frontier.upstream_edge_ids,
        bundle_event_count=len(data_events),
        bundle_edge_count=len(data_edges),
        execution_event_count=len(execution_events),
        execution_edge_count=len(execution_edges),
        data_exec_join=data_exec_join,
        data_exec_meet=data_exec_meet,
        eta_data_to_exec=eta_forward,
        eta_exec_to_data=eta_reverse,
        complete=(
            not unresolved
            and eta_forward.complete
            and eta_reverse.complete
            and bool(anchor_event.site_id)
            and bool(execution_frontier.site_id)
        ),
    )


def compute_lattice_witness(
    *,
    rel_path: str,
    qualname: str,
    bundle: FiberBundle,
    branch_line: int,
    branch_column: int,
    branch_node_kind: str,
    required_symbols: Iterable[str],
) -> FrontierWitness:
    normalized_required = tuple(sorted(dict.fromkeys(required_symbols)))
    cache_key = _lattice_cache_key(
        rel_path=rel_path,
        qualname=qualname,
        bundle=bundle,
        branch_line=branch_line,
        branch_column=branch_column,
        branch_node_kind=branch_node_kind,
        required_symbols=normalized_required,
    )
    cached = _load_cached_witness(cache_key=cache_key)
    if cached is not None:
        return cached
    resolved = frontier(
        rel_path=rel_path,
        qualname=qualname,
        bundle=bundle,
        branch_line=branch_line,
        branch_column=branch_column,
        branch_node_kind=branch_node_kind,
        required_symbols=normalized_required,
    )
    _store_cached_witness(cache_key=cache_key, witness=resolved)
    return resolved


def iter_lattice_witnesses(
    *,
    rel_path: str,
    qualname: str,
    module_tree: ast.AST,
    requests: Iterable[BranchWitnessRequest],
) -> Iterator[FrontierWitness]:
    def _query() -> Iterator[FrontierWitness]:
        bundle_cache: FiberBundle | None = None
        for request in requests:
            if bundle_cache is None:
                bundle_cache = build_fiber_bundle_for_qualname(
                    rel_path=rel_path,
                    module_tree=module_tree,
                    qualname=qualname,
                )
            yield compute_lattice_witness(
                rel_path=rel_path,
                qualname=qualname,
                bundle=bundle_cache,
                branch_line=request.branch_line,
                branch_column=request.branch_column,
                branch_node_kind=request.branch_node_kind,
                required_symbols=request.required_symbols,
            )

    return _query()


def _lattice_cache_key(
    *,
    rel_path: str,
    qualname: str,
    bundle: FiberBundle,
    branch_line: int,
    branch_column: int,
    branch_node_kind: str,
    required_symbols: tuple[str, ...],
) -> str:
    return _stable_hash(
        _LATTICE_CACHE_VERSION,
        rel_path,
        qualname,
        _text_part(branch_line),
        _text_part(branch_column),
        branch_node_kind,
        *required_symbols,
        _bundle_digest(bundle),
    )


def _bundle_digest(bundle: FiberBundle) -> str:
    data_parts = tuple(
        _stable_hash(
            event.site_id,
            event.site_identity,
            _text_part(event.line),
            _text_part(event.column),
            _text_part(event.ordinal),
            event.event_kind,
            event.symbol,
            event.node_kind,
        )
        for event in bundle.data.events
    )
    exec_parts = tuple(
        _stable_hash(
            event.site_id,
            event.site_identity,
            _text_part(event.line),
            _text_part(event.column),
            _text_part(event.ordinal),
            event.node_kind,
            event.event_kind,
        )
        for event in bundle.exec.events
    )
    return _stable_hash(*data_parts, *exec_parts)

def _cache_root() -> Path:
    return Path("artifacts/out/aspf_lattice_cache")


def _load_cached_witness(*, cache_key: str) -> FrontierWitness | None:
    path = _cache_root() / f"{cache_key}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return _frontier_from_payload(payload)


def _store_cached_witness(*, cache_key: str, witness: FrontierWitness) -> None:
    path = _cache_root() / f"{cache_key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _frontier_to_payload(witness)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _frontier_to_payload(witness: FrontierWitness) -> dict[str, object]:
    return {
        "branch_site_id": witness.branch_site_id,
        "branch_site_identity": witness.branch_site_identity,
        "branch_line": witness.branch_line,
        "branch_column": witness.branch_column,
        "branch_node_kind": witness.branch_node_kind,
        "required_symbols": list(witness.required_symbols),
        "unresolved_symbols": list(witness.unresolved_symbols),
        "data_anchor_site_id": witness.data_anchor_site_id,
        "data_anchor_site_identity": witness.data_anchor_site_identity,
        "data_anchor_line": witness.data_anchor_line,
        "data_anchor_column": witness.data_anchor_column,
        "data_anchor_ordinal": witness.data_anchor_ordinal,
        "data_upstream_site_ids": list(witness.data_upstream_site_ids),
        "data_upstream_site_identities": list(witness.data_upstream_site_identities),
        "data_upstream_edge_ids": list(witness.data_upstream_edge_ids),
        "exec_frontier_site_id": witness.exec_frontier_site_id,
        "exec_frontier_site_identity": witness.exec_frontier_site_identity,
        "exec_frontier_line": witness.exec_frontier_line,
        "exec_frontier_column": witness.exec_frontier_column,
        "exec_frontier_ordinal": witness.exec_frontier_ordinal,
        "exec_upstream_site_ids": list(witness.exec_upstream_site_ids),
        "exec_upstream_site_identities": list(witness.exec_upstream_site_identities),
        "exec_upstream_edge_ids": list(witness.exec_upstream_edge_ids),
        "bundle_event_count": witness.bundle_event_count,
        "bundle_edge_count": witness.bundle_edge_count,
        "execution_event_count": witness.execution_event_count,
        "execution_edge_count": witness.execution_edge_count,
        "data_exec_join": _join_payload(witness.data_exec_join),
        "data_exec_meet": _join_payload(witness.data_exec_meet),
        "eta_data_to_exec": _naturality_payload(witness.eta_data_to_exec),
        "eta_exec_to_data": _naturality_payload(witness.eta_exec_to_data),
        "complete": witness.complete,
        "obligations": [
            {
                "obligation_id": item.obligation_id,
                "source_kind": item.source_kind,
                "source_site_id": item.source_site_id,
                "source_site_identity": item.source_site_identity,
                "reason": item.reason,
                "introduced_by": item.introduced_by,
            }
            for item in witness.obligations
        ],
        "erasures": [
            {
                "obligation_id": item.obligation_id,
                "erased_by": item.erased_by,
                "reason": item.reason,
            }
            for item in witness.erasures
        ],
        "boundary_crossings": [
            {
                "crossing_id": item.crossing_id,
                "branch_site_id": item.branch_site_id,
                "branch_site_identity": item.branch_site_identity,
                "boundary_kind": item.boundary_kind,
            }
            for item in witness.boundary_crossings
        ],
        "violation": _violation_payload(witness.violation),
    }


def _frontier_from_payload(payload: object) -> FrontierWitness | None:
    if not isinstance(payload, dict):
        return None
    try:
        data_exec_join = _join_from_payload(payload.get("data_exec_join"))
        data_exec_meet = _join_from_payload(payload.get("data_exec_meet"))
        eta_data = _naturality_from_payload(payload.get("eta_data_to_exec"))
        eta_exec = _naturality_from_payload(payload.get("eta_exec_to_data"))
        obligations = tuple(
            ObligationIntro(
                obligation_id=str(item.get("obligation_id", "")),
                source_kind=str(item.get("source_kind", "")),
                source_site_id=str(item.get("source_site_id", "")),
                source_site_identity=str(item.get("source_site_identity", "")),
                reason=str(item.get("reason", "")),
                introduced_by=str(item.get("introduced_by", "")),
            )
            for item in _as_dict_items(payload.get("obligations"))
        )
        erasures = tuple(
            ObligationErase(
                obligation_id=str(item.get("obligation_id", "")),
                erased_by=str(item.get("erased_by", "")),
                reason=str(item.get("reason", "")),
            )
            for item in _as_dict_items(payload.get("erasures"))
        )
        crossings = tuple(
            BoundaryCrossing(
                crossing_id=str(item.get("crossing_id", "")),
                branch_site_id=str(item.get("branch_site_id", "")),
                branch_site_identity=str(item.get("branch_site_identity", "")),
                boundary_kind=str(item.get("boundary_kind", "")),
            )
            for item in _as_dict_items(payload.get("boundary_crossings"))
        )
        violation = _violation_from_payload(payload.get("violation"))
        return FrontierWitness(
            branch_site_id=str(payload.get("branch_site_id", "")),
            branch_site_identity=str(payload.get("branch_site_identity", "")),
            branch_line=int(payload.get("branch_line", 0)),
            branch_column=int(payload.get("branch_column", 0)),
            branch_node_kind=str(payload.get("branch_node_kind", "")),
            required_symbols=_stream_from_sequence(
                tuple(_as_string_list(payload.get("required_symbols")))
            ),
            unresolved_symbols=_stream_from_sequence(
                tuple(_as_string_list(payload.get("unresolved_symbols")))
            ),
            data_anchor_site_id=str(payload.get("data_anchor_site_id", "")),
            data_anchor_site_identity=str(payload.get("data_anchor_site_identity", "")),
            data_anchor_line=int(payload.get("data_anchor_line", 0)),
            data_anchor_column=int(payload.get("data_anchor_column", 0)),
            data_anchor_ordinal=int(payload.get("data_anchor_ordinal", 0)),
            data_upstream_site_ids=_stream_from_sequence(
                tuple(_as_string_list(payload.get("data_upstream_site_ids")))
            ),
            data_upstream_site_identities=_stream_from_sequence(
                tuple(_as_string_list(payload.get("data_upstream_site_identities")))
            ),
            data_upstream_edge_ids=_stream_from_sequence(
                tuple(_as_string_list(payload.get("data_upstream_edge_ids")))
            ),
            exec_frontier_site_id=str(payload.get("exec_frontier_site_id", "")),
            exec_frontier_site_identity=str(payload.get("exec_frontier_site_identity", "")),
            exec_frontier_line=int(payload.get("exec_frontier_line", 0)),
            exec_frontier_column=int(payload.get("exec_frontier_column", 0)),
            exec_frontier_ordinal=int(payload.get("exec_frontier_ordinal", 0)),
            exec_upstream_site_ids=_stream_from_sequence(
                tuple(_as_string_list(payload.get("exec_upstream_site_ids")))
            ),
            exec_upstream_site_identities=_stream_from_sequence(
                tuple(_as_string_list(payload.get("exec_upstream_site_identities")))
            ),
            exec_upstream_edge_ids=_stream_from_sequence(
                tuple(_as_string_list(payload.get("exec_upstream_edge_ids")))
            ),
            bundle_event_count=int(payload.get("bundle_event_count", 0)),
            bundle_edge_count=int(payload.get("bundle_edge_count", 0)),
            execution_event_count=int(payload.get("execution_event_count", 0)),
            execution_edge_count=int(payload.get("execution_edge_count", 0)),
            data_exec_join=data_exec_join,
            data_exec_meet=data_exec_meet,
            eta_data_to_exec=eta_data,
            eta_exec_to_data=eta_exec,
            complete=bool(payload.get("complete", False)),
            obligations=_stream_from_sequence(obligations),
            erasures=_stream_from_sequence(erasures),
            boundary_crossings=_stream_from_sequence(crossings),
            violation=violation,
        )
    except (TypeError, ValueError):
        return None


def _as_dict_items(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _as_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _join_payload(witness: JoinWitness | MeetWitness) -> dict[str, object]:
    return {
        "left_ids": list(witness.left_ids),
        "right_ids": list(witness.right_ids),
        "result_ids": list(witness.result_ids),
        "deterministic": witness.deterministic,
    }


def _join_from_payload(value: object) -> JoinWitness:
    if not isinstance(value, dict):
        return join(left_ids=(), right_ids=())
    return JoinWitness(
        left_ids=_stream_from_sequence(tuple(_as_string_list(value.get("left_ids")))),
        right_ids=_stream_from_sequence(tuple(_as_string_list(value.get("right_ids")))),
        result_ids=_stream_from_sequence(tuple(_as_string_list(value.get("result_ids")))),
        deterministic=bool(value.get("deterministic", True)),
    )


def _naturality_payload(witness: NaturalityWitness) -> dict[str, object]:
    return {
        "direction": witness.direction,
        "mapped_source_site_ids": list(witness.mapped_source_site_ids),
        "mapped_target_site_ids": list(witness.mapped_target_site_ids),
        "unmapped": [
            {
                "source_kind": item.source_kind,
                "source_site_id": item.source_site_id,
                "source_site_identity": item.source_site_identity,
                "reason": item.reason,
            }
            for item in witness.unmapped
        ],
        "complete": witness.complete,
    }


def _naturality_from_payload(value: object) -> NaturalityWitness:
    if not isinstance(value, dict):
        return NaturalityWitness(
            direction="",
            mapped_source_site_ids=_stream_from_sequence(()),
            mapped_target_site_ids=_stream_from_sequence(()),
            unmapped=_stream_from_sequence(()),
            complete=False,
        )
    unmapped = tuple(
        UnmappedWitness(
            source_kind=str(item.get("source_kind", "")),
            source_site_id=str(item.get("source_site_id", "")),
            source_site_identity=str(item.get("source_site_identity", "")),
            reason=str(item.get("reason", "")),
        )
        for item in _as_dict_items(value.get("unmapped"))
    )
    return NaturalityWitness(
        direction=str(value.get("direction", "")),
        mapped_source_site_ids=_stream_from_sequence(
            tuple(_as_string_list(value.get("mapped_source_site_ids")))
        ),
        mapped_target_site_ids=_stream_from_sequence(
            tuple(_as_string_list(value.get("mapped_target_site_ids")))
        ),
        unmapped=_stream_from_sequence(unmapped),
        complete=bool(value.get("complete", False)),
    )


def _violation_payload(violation: ViolationWitness | None) -> dict[str, object] | None:
    if violation is None:
        return None
    return {
        "violation_id": violation.violation_id,
        "boundary_crossing_id": violation.boundary_crossing_id,
        "unresolved_obligation_ids": list(violation.unresolved_obligation_ids),
        "reason": violation.reason,
    }


def _violation_from_payload(value: object) -> ViolationWitness | None:
    if not isinstance(value, dict):
        return None
    return ViolationWitness(
        violation_id=str(value.get("violation_id", "")),
        boundary_crossing_id=str(value.get("boundary_crossing_id", "")),
        unresolved_obligation_ids=_stream_from_sequence(
            tuple(_as_string_list(value.get("unresolved_obligation_ids")))
        ),
        reason=str(value.get("reason", "")),
    )


def eta_data_to_exec(
    *,
    data_events: tuple[DataflowEvent, ...],
    exec_events: tuple[ExecutionEvent, ...],
) -> NaturalityWitness:
    mapped_sources: list[str] = []
    mapped_targets: list[str] = []
    unmapped: list[UnmappedWitness] = []
    for event in data_events:
        mapped = _match_execution_event_for_dataflow(
            execution_events=exec_events,
            dataflow_event=event,
        )
        if not mapped.site_id:
            unmapped.append(
                UnmappedWitness(
                    source_kind="data",
                    source_site_id=event.site_id,
                    source_site_identity=event.site_identity,
                    reason="missing execution mapping",
                )
            )
            continue
        mapped_sources.append(event.site_id)
        mapped_targets.append(mapped.site_id)
    return NaturalityWitness(
        direction="data_to_exec",
        mapped_source_site_ids=_stream_from_sequence(tuple(mapped_sources)),
        mapped_target_site_ids=_stream_from_sequence(tuple(mapped_targets)),
        unmapped=_stream_from_sequence(tuple(unmapped)),
        complete=(len(unmapped) == 0),
    )


def eta_exec_to_data(
    *,
    data_events: tuple[DataflowEvent, ...],
    exec_events: tuple[ExecutionEvent, ...],
) -> NaturalityWitness:
    data_by_identity = {event.site_identity: event for event in data_events}
    data_by_line = defaultdict(list)
    for event in data_events:
        data_by_line[event.line].append(event)
    sorted_data_events = tuple(sorted(data_events, key=lambda item: (item.line, item.column)))
    mapped_sources: list[str] = []
    mapped_targets: list[str] = []
    unmapped: list[UnmappedWitness] = []
    for event in exec_events:
        mapped = data_by_identity.get(event.site_identity)
        if mapped is None:
            candidates = [
                candidate
                for candidate in data_by_line.get(event.line, [])
                if candidate.column <= event.column
            ]
            mapped = candidates[-1] if candidates else None
        if mapped is None:
            prior_candidates = [
                candidate
                for candidate in sorted_data_events
                if (candidate.line < event.line)
                or (candidate.line == event.line and candidate.column <= event.column)
            ]
            mapped = prior_candidates[-1] if prior_candidates else None
        if mapped is None:
            unmapped.append(
                UnmappedWitness(
                    source_kind="exec",
                    source_site_id=event.site_id,
                    source_site_identity=event.site_identity,
                    reason="missing data mapping",
                )
            )
            continue
        mapped_sources.append(event.site_id)
        mapped_targets.append(mapped.site_id)
    return NaturalityWitness(
        direction="exec_to_data",
        mapped_source_site_ids=_stream_from_sequence(tuple(mapped_sources)),
        mapped_target_site_ids=_stream_from_sequence(tuple(mapped_targets)),
        unmapped=_stream_from_sequence(tuple(unmapped)),
        complete=(len(unmapped) == 0),
    )


def frontier_failure_witness(
    *,
    rel_path: str,
    qualname: str,
    line: int,
    column: int,
    node_kind: str,
    reason: str,
) -> FrontierWitness:
    branch_site_identity = canonical_site_identity(
        rel_path=rel_path,
        qualname=qualname,
        line=line,
        column=column,
        node_kind=node_kind,
        surface="pyast",
    )
    branch_site_id = _site_id(
        rel_path=rel_path,
        qualname=qualname,
        line=line,
        column=column,
        event_kind="branch",
        symbol="",
        node_kind=node_kind,
    )
    unmapped = UnmappedWitness(
        source_kind="frontier",
        source_site_id=branch_site_id,
        source_site_identity=branch_site_identity,
        reason=reason,
    )
    empty_join = join(left_ids=(), right_ids=())
    empty_meet = meet(left_ids=(), right_ids=())
    return FrontierWitness(
        branch_site_id=branch_site_id,
        branch_site_identity=branch_site_identity,
        branch_line=line,
        branch_column=column,
        branch_node_kind=node_kind,
        required_symbols=_stream_from_sequence(()),
        unresolved_symbols=_stream_from_sequence(()),
        data_anchor_site_id=branch_site_id,
        data_anchor_site_identity=branch_site_identity,
        data_anchor_line=line,
        data_anchor_column=column,
        data_anchor_ordinal=0,
        data_upstream_site_ids=_stream_from_sequence(()),
        data_upstream_site_identities=_stream_from_sequence(()),
        data_upstream_edge_ids=_stream_from_sequence(()),
        exec_frontier_site_id=branch_site_id,
        exec_frontier_site_identity=branch_site_identity,
        exec_frontier_line=line,
        exec_frontier_column=column,
        exec_frontier_ordinal=0,
        exec_upstream_site_ids=_stream_from_sequence(()),
        exec_upstream_site_identities=_stream_from_sequence(()),
        exec_upstream_edge_ids=_stream_from_sequence(()),
        bundle_event_count=0,
        bundle_edge_count=0,
        execution_event_count=0,
        execution_edge_count=0,
        data_exec_join=empty_join,
        data_exec_meet=empty_meet,
        eta_data_to_exec=NaturalityWitness(
            direction="data_to_exec",
            mapped_source_site_ids=_stream_from_sequence(()),
            mapped_target_site_ids=_stream_from_sequence(()),
            unmapped=_stream_from_sequence((unmapped,)),
            complete=False,
        ),
        eta_exec_to_data=NaturalityWitness(
            direction="exec_to_data",
            mapped_source_site_ids=_stream_from_sequence(()),
            mapped_target_site_ids=_stream_from_sequence(()),
            unmapped=_stream_from_sequence((unmapped,)),
            complete=False,
        ),
        complete=False,
    )


def branch_required_symbols(node: ast.AST) -> Iterator[str]:
    expression_nodes = list(_branch_condition_nodes(node))
    all_walked_nodes = chain.from_iterable(map(ast.walk, expression_nodes))
    name_ids = map(
        lambda item: item.id,
        filter(_is_loaded_name_node, all_walked_nodes),
    )
    for name in sorted(dict.fromkeys(name_ids)):
        yield name


@dataclass(frozen=True)
class _ResolvedScope:
    qualname: str
    start_line: int
    start_column: int
    arguments: tuple[ast.arg, ...]
    body: tuple[ast.stmt, ...]


def _resolve_scope(*, module_tree: ast.AST, qualname: str) -> _ResolvedScope:
    root_statements = list(_module_statements(module_tree))
    if qualname == "<module>":
        return _ResolvedScope(
            qualname="<module>",
            start_line=1,
            start_column=1,
            arguments=(),
            body=root_statements,
        )

    segments = list(filter(bool, qualname.split(".")))
    resolved_scope = _resolve_function_scope(segments=segments, root_statements=root_statements)
    if not resolved_scope.resolved:
        return _ResolvedScope(
            qualname=qualname,
            start_line=1,
            start_column=1,
            arguments=(),
            body=(),
        )
    return _ResolvedScope(
        qualname=qualname,
        start_line=resolved_scope.start_line,
        start_column=resolved_scope.start_column,
        arguments=resolved_scope.arguments,
        body=resolved_scope.body,
    )


@dataclass(frozen=True)
class _ResolvedFunctionScope:
    resolved: bool
    start_line: int
    start_column: int
    arguments: tuple[ast.arg, ...]
    body: tuple[ast.stmt, ...]


def _module_statements(module_tree: ast.AST) -> Iterator[ast.stmt]:
    match module_tree:
        case ast.Module(body=body):
            for statement in body:
                yield statement
        case _:
            return


            never("unreachable wildcard match fall-through")
def _resolve_function_scope(
    *,
    segments: tuple[str, ...],
    root_statements: tuple[ast.stmt, ...],
) -> _ResolvedFunctionScope:
    if not segments:
        return _empty_function_scope()
    return _resolve_function_scope_recursive(
        segments=segments,
        statements=root_statements,
    )


def _resolve_function_scope_recursive(
    *,
    segments: tuple[str, ...],
    statements: tuple[ast.stmt, ...],
) -> _ResolvedFunctionScope:
    current = segments[0]
    rest = segments[1:]
    first_match = next(filter(_statement_name_equals(current), statements), None)
    if first_match is None:
        return _empty_function_scope()
    if not rest:
        return _ResolvedFunctionScope(
            resolved=True,
            start_line=_line_value(getattr(first_match, "lineno", 1)),
            start_column=_column_value(getattr(first_match, "col_offset", 0)) + 1,
            arguments=list(_iter_function_arguments(first_match)),
            body=list(first_match.body),
        )
    return _resolve_function_scope_recursive(
        segments=rest,
        statements=list(first_match.body),
    )


def _empty_function_scope() -> _ResolvedFunctionScope:
    return _ResolvedFunctionScope(
        resolved=False,
        start_line=1,
        start_column=1,
        arguments=(),
        body=(),
    )


def _statement_name_matches(*, statement: ast.stmt, name: str) -> bool:
    match statement:
        case ast.FunctionDef(name=fn_name):
            return fn_name == name
        case ast.AsyncFunctionDef(name=fn_name):
            return fn_name == name
        case _:
            return False


            never("unreachable wildcard match fall-through")
def _statement_name_equals(name: str):
    return lambda statement: _statement_name_matches(statement=statement, name=name)


def _iter_function_arguments(
    function_node: ast.AST,
) -> Iterator[ast.arg]:
    match function_node:
        case ast.FunctionDef(args=arguments) | ast.AsyncFunctionDef(args=arguments):
            for argument in chain(arguments.posonlyargs, arguments.args, arguments.kwonlyargs):
                yield argument
            for optional in filter(_is_ast_arg, [arguments.vararg, arguments.kwarg]):
                yield optional
        case _:
            yield from ()


            never("unreachable wildcard match fall-through")
class _DataflowCollector(ast.NodeVisitor):
    def __init__(
        self,
        *,
        rel_path: str,
        qualname: str,
        entry_line: int,
        entry_column: int,
    ) -> None:
        self.rel_path = rel_path
        self.qualname = qualname
        self._next_ordinal = 1
        self.events: list[DataflowEvent] = []
        self.edges: list[DataflowEdge] = []
        self._last_def_by_symbol: dict[str, DataflowEvent] = {}
        self.entry_site_identity = _site_identity(
            rel_path=rel_path,
            qualname=qualname,
            line=entry_line,
            column=entry_column,
            node_kind="entry",
        )
        self.entry_site_id = _site_id(
            rel_path=rel_path,
            qualname=qualname,
            line=entry_line,
            column=entry_column,
            event_kind="entry",
            symbol="",
            node_kind="entry",
        )
        self.events.append(
            DataflowEvent(
                site_id=self.entry_site_id,
                site_identity=self.entry_site_identity,
                rel_path=rel_path,
                qualname=qualname,
                line=entry_line,
                column=entry_column,
                ordinal=0,
                symbol="",
                event_kind="entry",
                node_kind="entry",
            )
        )

    def record_argument_definitions(self, arguments: tuple[ast.arg, ...]) -> None:
        list(map(self._record_argument_definition, arguments))

    def _record_argument_definition(self, arg_node: ast.arg) -> None:
        self._record_definition(
            symbol=arg_node.arg,
            line=_line_value(getattr(arg_node, "lineno", 1)),
            column=_column_value(getattr(arg_node, "col_offset", 0)) + 1,
            node_kind="arg",
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        _ = node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        _ = node

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        _ = node

    def visit_Lambda(self, node: ast.Lambda) -> None:
        _ = node

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.name:
            self._record_definition(
                symbol=node.name,
                line=_line_value(getattr(node, "lineno", 1)),
                column=_column_value(getattr(node, "col_offset", 0)) + 1,
                node_kind="match_bind",
            )
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name:
            self._record_definition(
                symbol=node.name,
                line=_line_value(getattr(node, "lineno", 1)),
                column=_column_value(getattr(node, "col_offset", 0)) + 1,
                node_kind="match_bind",
            )
        self.generic_visit(node)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        if node.rest:
            self._record_definition(
                symbol=node.rest,
                line=_line_value(getattr(node, "lineno", 1)),
                column=_column_value(getattr(node, "col_offset", 0)) + 1,
                node_kind="match_bind",
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        symbol = node.id
        line = _line_value(getattr(node, "lineno", 1))
        column = _column_value(getattr(node, "col_offset", 0)) + 1
        match node.ctx:
            case ast.Store() | ast.Del():
                self._record_definition(
                    symbol=symbol,
                    line=line,
                    column=column,
                    node_kind="name_store",
                )
            case ast.Load():
                self._record_use(
                    symbol=symbol,
                    line=line,
                    column=column,
                    node_kind="name_load",
                )
            case _:
                return

                never("unreachable wildcard match fall-through")
    def _record_definition(
        self,
        *,
        symbol: str,
        line: int,
        column: int,
        node_kind: str,
    ) -> None:
        site_identity = _site_identity(
            rel_path=self.rel_path,
            qualname=self.qualname,
            line=line,
            column=column,
            node_kind=node_kind,
        )
        event = DataflowEvent(
            site_id=_site_id(
                rel_path=self.rel_path,
                qualname=self.qualname,
                line=line,
                column=column,
                event_kind="def",
                symbol=symbol,
                node_kind=node_kind,
            ),
            site_identity=site_identity,
            rel_path=self.rel_path,
            qualname=self.qualname,
            line=line,
            column=column,
            ordinal=self._next_ordinal,
            symbol=symbol,
            event_kind="def",
            node_kind=node_kind,
        )
        self.events.append(event)
        self._last_def_by_symbol[symbol] = event
        self._next_ordinal += 1

    def _record_use(
        self,
        *,
        symbol: str,
        line: int,
        column: int,
        node_kind: str,
    ) -> None:
        site_identity = _site_identity(
            rel_path=self.rel_path,
            qualname=self.qualname,
            line=line,
            column=column,
            node_kind=node_kind,
        )
        event = DataflowEvent(
            site_id=_site_id(
                rel_path=self.rel_path,
                qualname=self.qualname,
                line=line,
                column=column,
                event_kind="use",
                symbol=symbol,
                node_kind=node_kind,
            ),
            site_identity=site_identity,
            rel_path=self.rel_path,
            qualname=self.qualname,
            line=line,
            column=column,
            ordinal=self._next_ordinal,
            symbol=symbol,
            event_kind="use",
            node_kind=node_kind,
        )
        self.events.append(event)
        self._next_ordinal += 1
        source_event = self._last_def_by_symbol.get(symbol)
        if source_event is not None:
            self.edges.append(
                DataflowEdge(
                    edge_id=_edge_id(
                        symbol=symbol,
                        source_site_id=source_event.site_id,
                        target_site_id=event.site_id,
                    ),
                    symbol=symbol,
                    source_site_id=source_event.site_id,
                    target_site_id=event.site_id,
                    source_site_identity=source_event.site_identity,
                    target_site_identity=event.site_identity,
                    source_ordinal=source_event.ordinal,
                    target_ordinal=event.ordinal,
                )
            )


class _ExecutionCollector:
    def __init__(
        self,
        *,
        rel_path: str,
        qualname: str,
        entry_line: int,
        entry_column: int,
    ) -> None:
        self.rel_path = rel_path
        self.qualname = qualname
        self.events: list[ExecutionEvent] = []
        self.edges: list[ExecutionEdge] = []
        self._next_ordinal = 1
        self.entry_site_identity = _site_identity(
            rel_path=rel_path,
            qualname=qualname,
            line=entry_line,
            column=entry_column,
            node_kind="entry",
        )
        self.entry_site_id = _execution_site_id(
            rel_path=rel_path,
            qualname=qualname,
            line=entry_line,
            column=entry_column,
            node_kind="entry",
            event_kind="entry",
        )
        self.events.append(
            ExecutionEvent(
                site_id=self.entry_site_id,
                site_identity=self.entry_site_identity,
                rel_path=rel_path,
                qualname=qualname,
                line=entry_line,
                column=entry_column,
                ordinal=0,
                node_kind="entry",
                event_kind="entry",
            )
        )

    def record_scope_statements(
        self,
        statements: tuple[ast.stmt, ...],
    ) -> Iterator[ExecutionEvent]:
        predecessor = self.events[0]
        for statement in statements:
            current = self._record_statement_event(statement=statement)
            self._record_execution_edge(source=predecessor, target=current)
            self._record_nested_statements(statement=statement, branch_event=current)
            predecessor = current
            yield current

    def _record_statement_event(self, *, statement: ast.stmt) -> ExecutionEvent:
        line = _line_value(getattr(statement, "lineno", 1))
        column = _column_value(getattr(statement, "col_offset", 0)) + 1
        node_kind = _execution_node_kind(statement)
        site_identity = _site_identity(
            rel_path=self.rel_path,
            qualname=self.qualname,
            line=line,
            column=column,
            node_kind=node_kind,
        )
        event = ExecutionEvent(
            site_id=_execution_site_id(
                rel_path=self.rel_path,
                qualname=self.qualname,
                line=line,
                column=column,
                node_kind=node_kind,
                event_kind="exec",
            ),
            site_identity=site_identity,
            rel_path=self.rel_path,
            qualname=self.qualname,
            line=line,
            column=column,
            ordinal=self._next_ordinal,
            node_kind=node_kind,
            event_kind="exec",
        )
        self.events.append(event)
        self._next_ordinal += 1
        return event

    def _record_nested_statements(
        self,
        *,
        statement: ast.stmt,
        branch_event: ExecutionEvent,
    ) -> None:
        list(
            chain.from_iterable(
                map(
                    lambda group: self._iter_nested_group_events(
                        group=group,
                        branch_event=branch_event,
                    ),
                    _nested_statement_groups(statement),
                )
            )
        )

    def _iter_nested_group_events(
        self,
        *,
        group: list[ast.stmt],
        branch_event: ExecutionEvent,
    ) -> Iterator[ExecutionEvent]:
        predecessor = branch_event
        for nested_statement in group:
            nested_event = self._record_statement_event(statement=nested_statement)
            self._record_execution_edge(source=predecessor, target=nested_event)
            self._record_nested_statements(
                statement=nested_statement,
                branch_event=nested_event,
            )
            predecessor = nested_event
            yield nested_event

    def _record_execution_edge(
        self,
        *,
        source: ExecutionEvent,
        target: ExecutionEvent,
    ) -> None:
        if source.site_id == target.site_id:
            return
        self.edges.append(
            ExecutionEdge(
                edge_id=_execution_edge_id(
                    source_site_id=source.site_id,
                    target_site_id=target.site_id,
                ),
                source_site_id=source.site_id,
                target_site_id=target.site_id,
                source_site_identity=source.site_identity,
                target_site_identity=target.site_identity,
                source_ordinal=source.ordinal,
                target_ordinal=target.ordinal,
            )
        )


@dataclass(frozen=True)
class _ExecutionFrontierAnchor:
    site_id: str
    site_identity: str
    line: int
    column: int
    ordinal: int
    upstream_site_ids: ReplayableStream[str]
    upstream_site_identities: ReplayableStream[str]
    upstream_edge_ids: ReplayableStream[str]


def _iter_latest_definitions_before_branch(
    *,
    events: tuple[DataflowEvent, ...],
    branch_line: int,
    branch_column: int,
    symbols: tuple[str, ...],
) -> Iterator[_DefinitionRecord]:
    symbol_set = set(symbols)
    sorted_candidates = list(
        sorted(
            filter(
                _definition_candidate_filter(
                    symbol_set=symbol_set,
                    branch_line=branch_line,
                    branch_column=branch_column,
                ),
                events,
            ),
            key=lambda event: (event.symbol, event.ordinal),
        )
    )
    for symbol, grouped_events in groupby(
        sorted_candidates,
        key=lambda event: event.symbol,
    ):
        yield _DefinitionRecord(
            symbol=symbol,
            event=list(grouped_events)[-1],
        )


def _event_before_or_at_branch(
    *,
    event: DataflowEvent,
    branch_line: int,
    branch_column: int,
) -> bool:
    if event.line < branch_line:
        return True
    if event.line > branch_line:
        return False
    return event.column <= branch_column


def _anchor_event(
    *,
    upstream_events: tuple[DataflowEvent, ...],
    events: tuple[DataflowEvent, ...],
) -> DataflowEvent:
    if upstream_events:
        return max(upstream_events, key=lambda event: event.ordinal)
    entry_candidates = list(filter(_is_entry_event, events))
    if entry_candidates:
        return entry_candidates[0]
    return DataflowEvent(
        site_id="",
        site_identity="",
        rel_path="",
        qualname="",
        line=1,
        column=1,
        ordinal=0,
        symbol="",
        event_kind="entry",
        node_kind="entry",
    )


def _execution_recombination_frontier(
    *,
    execution_events: tuple[ExecutionEvent, ...],
    execution_edges: tuple[ExecutionEdge, ...],
    branch_site_identity: str,
    required_dataflow_events: tuple[DataflowEvent, ...],
) -> _ExecutionFrontierAnchor:
    if not execution_events:
        return _empty_execution_frontier_anchor(branch_site_identity=branch_site_identity)
    execution_by_identity = _execution_events_by_identity(execution_events)
    execution_by_site_id = _execution_events_by_site_id(execution_events)
    predecessor_map: defaultdict[str, list[str]] = defaultdict(list)
    for edge in execution_edges:
        predecessor_map.setdefault(edge.target_site_id, []).append(edge.source_site_id)
    predecessors = _dedup_predecessor_map(predecessor_map)
    branch_event = _match_execution_event_for_branch(
        execution_events=execution_events,
        branch_site_identity=branch_site_identity,
    )
    required_execution_events = tuple(
        map(
            lambda event: _match_execution_event_for_dataflow(
                execution_events=execution_events,
                dataflow_event=event,
            ),
            required_dataflow_events,
        )
    )
    origin_events = tuple(
        dict.fromkeys([branch_event, *required_execution_events])
    )
    if not origin_events:
        origin_events = (execution_events[0],)
    ancestor_sets = list(
        _execution_ancestors(site_id=origin.site_id, predecessors=predecessors)
        for origin in origin_events
    )
    common_site_ids = set.intersection(*ancestor_sets) if ancestor_sets else set()
    frontier_event = _latest_execution_event(
        event_ids=common_site_ids,
        execution_by_site_id=execution_by_site_id,
        fallback=branch_event,
    )
    upstream_events = tuple(
        sorted(
            filter(
                lambda event: event.ordinal <= frontier_event.ordinal,
                _execution_events_from_lookup(execution_by_identity),
            ),
            key=lambda event: event.ordinal,
        )
    )
    upstream_site_ids = tuple(map(_execution_site_id_from_event, upstream_events))
    upstream_site_identities = tuple(
        map(_execution_site_identity_from_event, upstream_events)
    )
    upstream_edge_ids = tuple(
        sorted(
            map(
                _execution_edge_id_from_edge,
                filter(
                    _execution_edge_targets_in(upstream_site_ids),
                    execution_edges,
                ),
            )
        )
    )
    return _ExecutionFrontierAnchor(
        site_id=frontier_event.site_id,
        site_identity=frontier_event.site_identity,
        line=frontier_event.line,
        column=frontier_event.column,
        ordinal=frontier_event.ordinal,
        upstream_site_ids=_stream_from_sequence(upstream_site_ids),
        upstream_site_identities=_stream_from_sequence(upstream_site_identities),
        upstream_edge_ids=_stream_from_sequence(upstream_edge_ids),
    )


def _empty_execution_frontier_anchor(*, branch_site_identity: str) -> _ExecutionFrontierAnchor:
    return _ExecutionFrontierAnchor(
        site_id="",
        site_identity=branch_site_identity,
        line=1,
        column=1,
        ordinal=0,
        upstream_site_ids=_stream_from_sequence(()),
        upstream_site_identities=_stream_from_sequence(()),
        upstream_edge_ids=_stream_from_sequence(()),
    )


def _execution_events_by_identity(
    events: list[ExecutionEvent],
) -> dict[str, ExecutionEvent]:
    lookup: dict[str, ExecutionEvent] = {}
    for event in events:
        lookup[event.site_identity] = event
    return lookup


def _execution_events_by_site_id(
    events: list[ExecutionEvent],
) -> dict[str, ExecutionEvent]:
    lookup: dict[str, ExecutionEvent] = {}
    for event in events:
        lookup[event.site_id] = event
    return lookup


def _execution_events_from_lookup(
    lookup: dict[str, ExecutionEvent],
) -> Iterator[ExecutionEvent]:
    for event in lookup.values():
        yield event


def _dedup_predecessor_map(
    predecessor_map: defaultdict[str, list[str]],
) -> defaultdict[str, list[str]]:
    out: defaultdict[str, list[str]] = defaultdict(list)
    for site_id, predecessor_ids in predecessor_map.items():
        seen: set[str] = set()
        for predecessor_id in predecessor_ids:
            if predecessor_id in seen:
                continue
            seen.add(predecessor_id)
            out[site_id].append(predecessor_id)
    return out


def _execution_ancestors(
    *,
    site_id: str,
    predecessors: defaultdict[str, list[str]],
) -> set[str]:
    visited: set[str] = set()
    stack = [site_id]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        stack.extend(predecessors.get(current, _empty_string_list()))
    return visited


def _latest_execution_event(
    *,
    event_ids: set[str],
    execution_by_site_id: dict[str, ExecutionEvent],
    fallback: ExecutionEvent,
) -> ExecutionEvent:
    resolved_ids = event_ids.intersection(set(execution_by_site_id))
    candidates = list(
        map(execution_by_site_id.__getitem__, resolved_ids),
    )
    if not candidates:
        return fallback
    return max(candidates, key=lambda event: event.ordinal)


def _match_execution_event_for_branch(
    *,
    execution_events: tuple[ExecutionEvent, ...],
    branch_site_identity: str,
) -> ExecutionEvent:
    direct = list(
        filter(
            lambda event: event.site_identity == branch_site_identity,
            execution_events,
        )
    )
    if direct:
        return max(direct, key=lambda event: event.ordinal)
    return execution_events[0]


def _match_execution_event_for_dataflow(
    *,
    execution_events: tuple[ExecutionEvent, ...],
    dataflow_event: DataflowEvent,
) -> ExecutionEvent:
    direct = list(
        filter(
            lambda event: event.site_identity == dataflow_event.site_identity,
            execution_events,
        )
    )
    if direct:
        return direct[0]
    same_line = list(
        filter(
            lambda event: event.line == dataflow_event.line
            and event.column <= dataflow_event.column,
            execution_events,
        )
    )
    if same_line:
        return same_line[-1]
    return execution_events[0]


def _branch_condition_nodes(node: ast.AST) -> Iterator[ast.AST]:
    match node:
        case ast.If(test=test):
            yield test
        case ast.While(test=test):
            yield test
        case ast.IfExp(test=test):
            yield test
        case ast.For(iter=iter_expr):
            yield iter_expr
        case ast.AsyncFor(iter=iter_expr):
            yield iter_expr
        case ast.Match(subject=subject):
            yield subject
        case _:
            return


            never("unreachable wildcard match fall-through")
def _is_loaded_name_node(node: ast.AST) -> bool:
    match node:
        case ast.Name(ctx=ast.Load()):
            return True
        case _:
            return False


            never("unreachable wildcard match fall-through")
def _is_ast_arg(node: object) -> bool:
    match node:
        case ast.arg():
            return True
        case _:
            return False


            never("unreachable wildcard match fall-through")
def _definition_lookup(
    definitions: list[_DefinitionRecord],
) -> dict[str, DataflowEvent]:
    lookup: dict[str, DataflowEvent] = {}
    for item in definitions:
        lookup[item.symbol] = item.event
    return lookup


def _missing_local_or_ingress_definition(
    *,
    lookup: dict[str, DataflowEvent],
    symbols_defined_in_scope: set[str],
    module_bound_symbols: set[str],
):
    builtin_symbols = _builtin_symbol_set()
    ingress_symbols = module_bound_symbols.union(_MODULE_INGRESS_SYMBOLS)
    return lambda symbol: (
        (symbol in symbols_defined_in_scope and symbol not in lookup)
        or (
            symbol not in symbols_defined_in_scope
            and symbol not in ingress_symbols
            and symbol not in builtin_symbols
        )
    )


def _defined_symbols(events: tuple[DataflowEvent, ...]) -> set[str]:
    definitions = filter(_is_definition_event, events)
    return set(map(lambda event: event.symbol, definitions))


def _is_definition_event(event: DataflowEvent) -> bool:
    return event.event_kind == "def"


def _module_bound_symbols(module_tree: ast.AST) -> tuple[str, ...]:
    collected: set[str] = set()
    for node in ast.walk(module_tree):
        match node:
            case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name) | ast.ClassDef(name=name):
                collected.add(name)
            case ast.Name(id=name, ctx=ast.Store()):
                collected.add(name)
            case ast.arg(arg=name):
                collected.add(name)
            case ast.alias(name=name, asname=asname):
                collected.add(asname if asname else name.partition(".")[0])
            case ast.ExceptHandler(name=name) if isinstance(name, str):
                collected.add(name)
            case _:
                continue
    return tuple(sorted(collected))


@lru_cache(maxsize=1)
def _builtin_symbol_set() -> set[str]:
    return set(dir(builtins))


def _edge_targets_in(targets: tuple[str, ...]):
    target_set = set(targets)
    return lambda edge: edge.target_site_id in target_set


def _edge_id_from_edge(edge: DataflowEdge) -> str:
    return edge.edge_id


def _site_id_from_event(event: DataflowEvent) -> str:
    return event.site_id


def _site_identity_from_event(event: DataflowEvent) -> str:
    return event.site_identity


def _execution_site_id_from_event(event: ExecutionEvent) -> str:
    return event.site_id


def _execution_site_identity_from_event(event: ExecutionEvent) -> str:
    return event.site_identity


def _execution_edge_id_from_edge(edge: ExecutionEdge) -> str:
    return edge.edge_id


def _execution_edge_targets_in(targets: tuple[str, ...]):
    target_set = set(targets)
    return lambda edge: edge.target_site_id in target_set


def _definition_candidate_filter(
    *,
    symbol_set: set[str],
    branch_line: int,
    branch_column: int,
):
    return lambda event: (
        event.event_kind == "def"
        and event.symbol in symbol_set
        and _event_before_or_at_branch(
            event=event,
            branch_line=branch_line,
            branch_column=branch_column,
        )
    )


def _is_entry_event(event: DataflowEvent) -> bool:
    return event.event_kind == "entry"


def _site_identity(
    *,
    rel_path: str,
    qualname: str,
    line: int,
    column: int,
    node_kind: str,
) -> str:
    return canonical_site_identity(
        rel_path=rel_path,
        qualname=qualname,
        line=line,
        column=column,
        node_kind=node_kind,
        surface="pyast",
    )


def _branch_site_identity(
    *,
    rel_path: str,
    qualname: str,
    line: int,
    column: int,
    node_kind: str,
) -> str:
    return _site_identity(
        rel_path=rel_path,
        qualname=qualname,
        line=line,
        column=column,
        node_kind=_execution_node_kind_from_branch(node_kind),
    )


def _execution_node_kind_from_branch(node_kind: str) -> str:
    if node_kind.startswith("branch:"):
        return "stmt:" + node_kind.removeprefix("branch:")
    return node_kind


def _site_id(
    *,
    rel_path: str,
    qualname: str,
    line: int,
    column: int,
    event_kind: str,
    symbol: str,
    node_kind: str,
) -> str:
    return _stable_hash(
        rel_path,
        qualname,
        _text_part(line),
        _text_part(column),
        event_kind,
        symbol,
        node_kind,
    )


def _execution_site_id(
    *,
    rel_path: str,
    qualname: str,
    line: int,
    column: int,
    node_kind: str,
    event_kind: str,
) -> str:
    return _stable_hash(
        "execution",
        rel_path,
        qualname,
        _text_part(line),
        _text_part(column),
        node_kind,
        event_kind,
    )


def _edge_id(*, symbol: str, source_site_id: str, target_site_id: str) -> str:
    return _stable_hash("edge", symbol, source_site_id, target_site_id)


def _execution_edge_id(*, source_site_id: str, target_site_id: str) -> str:
    return _stable_hash("execution_edge", source_site_id, target_site_id)


def _nested_statement_groups(statement: ast.stmt) -> Iterator[list[ast.stmt]]:
    match statement:
        case ast.If(body=body, orelse=orelse):
            yield from _non_empty_statement_groups([list(body), list(orelse)])
        case ast.For(body=body, orelse=orelse):
            yield from _non_empty_statement_groups([list(body), list(orelse)])
        case ast.AsyncFor(body=body, orelse=orelse):
            yield from _non_empty_statement_groups([list(body), list(orelse)])
        case ast.While(body=body, orelse=orelse):
            yield from _non_empty_statement_groups([list(body), list(orelse)])
        case ast.With(body=body):
            yield from _non_empty_statement_groups([list(body)])
        case ast.AsyncWith(body=body):
            yield from _non_empty_statement_groups([list(body)])
        case ast.Try(body=body, handlers=handlers, orelse=orelse, finalbody=finalbody):
            handler_groups = list(list(handler.body) for handler in handlers)
            yield from _non_empty_statement_groups(
                [list(body), *handler_groups, list(orelse), list(finalbody)]
            )
        case ast.TryStar(body=body, handlers=handlers, orelse=orelse, finalbody=finalbody):
            handler_groups = list(list(handler.body) for handler in handlers)
            yield from _non_empty_statement_groups(
                [list(body), *handler_groups, list(orelse), list(finalbody)]
            )
        case ast.Match(cases=cases):
            yield from _non_empty_statement_groups(
                list(list(case.body) for case in cases)
            )
        case _:
            return


            never("unreachable wildcard match fall-through")
def _non_empty_statement_groups(
    groups: list[list[ast.stmt]],
) -> Iterator[list[ast.stmt]]:
    for group in filter(bool, groups):
        yield group


def _execution_node_kind(node: ast.stmt) -> str:
    return "stmt:" + node.__class__.__name__.lower()


def _stable_hash(*parts: str) -> str:
    digest = reduce(_digest_with_part, parts, hashlib.sha256())
    return digest.hexdigest()


def _digest_with_part(digest: object, part: str):
    digest.update(part.encode("utf-8"))
    digest.update(b"\x00")
    return digest


def _is_execution_event(value: object) -> bool:
    match value:
        case ExecutionEvent():
            return True
        case _:
            return False


            never("unreachable wildcard match fall-through")
def _empty_string_list() -> list[str]:
    return []


def _line_value(value: object) -> int:
    match value:
        case int() as line:
            return line
        case _:
            return 1


            never("unreachable wildcard match fall-through")
def _column_value(value: object) -> int:
    match value:
        case int() as column:
            return column
        case _:
            return 0


            never("unreachable wildcard match fall-through")
def _text_part(value: object) -> str:
    match value:
        case str() as text:
            return text
        case _:
            return value.__str__()


            never("unreachable wildcard match fall-through")
__all__ = [
    "BoundaryCrossing",
    "BranchWitnessRequest",
    "CallEdgeGradeWitness",
    "DataFiber",
    "DataflowEdge",
    "DataflowEvent",
    "DeterminismCostGrade",
    "ExecFiber",
    "FiberBundle",
    "FrontierWitness",
    "GradeBoundaryKind",
    "GradeBoundaryMarker",
    "GradeMonotonicityViolation",
    "JoinWitness",
    "MeetWitness",
    "NaturalityWitness",
    "ObligationErase",
    "ObligationIntro",
    "OutputCardinalityClass",
    "ProtocolDischargeLevel",
    "DataflowFiberBundle",
    "ExecutionEdge",
    "ExecutionEvent",
    "UnmappedWitness",
    "ViolationWitness",
    "WorkGrowthClass",
    "branch_required_symbols",
    "build_fiber_bundle_for_qualname",
    "build_dataflow_fiber_bundle_for_qualname",
    "canonical_structural_identity",
    "compute_lattice_witness",
    "eta_data_to_exec",
    "eta_exec_to_data",
    "frontier",
    "frontier_failure_witness",
    "iter_lattice_witnesses",
    "join",
    "meet",
    "upstream_closure",
]
