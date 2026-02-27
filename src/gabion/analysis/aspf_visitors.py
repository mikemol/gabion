from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import hashlib
from typing import Iterable, Literal, Mapping, Protocol, cast

from gabion.analysis.aspf import Alt, Forest, Node
from gabion.json_types import JSONObject, JSONValue
from gabion.order_contract import sort_once


class AspfTraversalVisitor(Protocol):
    """Semantic-core visitor contract for ASPF traversal dispatch.

    Boundary normalization is expected to happen at ingress parsers (for example,
    ``load_trace_payload``). Traversal helpers below assume normalized payloads and
    perform deterministic visitor dispatch for semantic-core consumers.
    """

    def on_forest_node(self, *, node: Node) -> None: ...

    def on_forest_alt(self, *, alt: Alt) -> None: ...

    def on_trace_one_cell(
        self,
        *,
        index: int,
        one_cell: Mapping[str, object],
    ) -> None: ...

    def on_trace_two_cell_witness(
        self,
        *,
        index: int,
        witness: Mapping[str, object],
    ) -> None: ...

    def on_trace_cofibration(
        self,
        *,
        index: int,
        cofibration: Mapping[str, object],
    ) -> None: ...

    def on_trace_surface_representative(
        self,
        *,
        surface: str,
        representative: str,
    ) -> None: ...

    def on_equivalence_surface_row(
        self,
        *,
        index: int,
        row: Mapping[str, object],
    ) -> None: ...


@dataclass(frozen=True)
class AspfOneCellEvent:
    index: int
    payload: Mapping[str, object]

    def dispatch_to(self, visitor: AspfEventReplayVisitor) -> None:
        visitor.one_cell(self)


@dataclass(frozen=True)
class AspfTwoCellEvent:
    index: int
    payload: Mapping[str, object]

    def dispatch_to(self, visitor: AspfEventReplayVisitor) -> None:
        visitor.two_cell(self)


@dataclass(frozen=True)
class AspfCofibrationEvent:
    index: int
    payload: Mapping[str, object]

    def dispatch_to(self, visitor: AspfEventReplayVisitor) -> None:
        visitor.cofibration(self)


@dataclass(frozen=True)
class AspfSurfaceUpdateEvent:
    surface: str
    representative: str

    def dispatch_to(self, visitor: AspfEventReplayVisitor) -> None:
        visitor.surface_update(self)


@dataclass(frozen=True)
class AspfRunBoundaryEvent:
    boundary: Literal["equivalence_surface_row"]
    payload: Mapping[str, object]


class AspfTraceReplayEvent(Protocol):
    def dispatch_to(self, visitor: AspfEventReplayVisitor) -> None: ...


class AspfEventReplayVisitor(Protocol):
    """Canonical low-level ASPF event visitor protocol."""

    def one_cell(self, event: AspfOneCellEvent) -> None: ...

    def two_cell(self, event: AspfTwoCellEvent) -> None: ...

    def cofibration(self, event: AspfCofibrationEvent) -> None: ...

    def surface_update(self, event: AspfSurfaceUpdateEvent) -> None: ...

    def run_boundary(self, event: AspfRunBoundaryEvent) -> None: ...


@dataclass
class NullAspfTraversalVisitor:
    def on_forest_node(self, *, node: Node) -> None:
        pass

    def on_forest_alt(self, *, alt: Alt) -> None:
        pass

    def on_trace_one_cell(self, *, index: int, one_cell: Mapping[str, object]) -> None:
        pass

    def on_trace_two_cell_witness(
        self,
        *,
        index: int,
        witness: Mapping[str, object],
    ) -> None:
        pass

    def on_trace_cofibration(
        self,
        *,
        index: int,
        cofibration: Mapping[str, object],
    ) -> None:
        pass

    def on_trace_surface_representative(
        self,
        *,
        surface: str,
        representative: str,
    ) -> None:
        pass

    def on_equivalence_surface_row(
        self,
        *,
        index: int,
        row: Mapping[str, object],
    ) -> None:
        pass

    def one_cell(self, event: AspfOneCellEvent) -> None:
        self.on_trace_one_cell(index=event.index, one_cell=event.payload)

    def two_cell(self, event: AspfTwoCellEvent) -> None:
        self.on_trace_two_cell_witness(index=event.index, witness=event.payload)

    def cofibration(self, event: AspfCofibrationEvent) -> None:
        self.on_trace_cofibration(index=event.index, cofibration=event.payload)

    def surface_update(self, event: AspfSurfaceUpdateEvent) -> None:
        self.on_trace_surface_representative(
            surface=event.surface,
            representative=event.representative,
        )

    def run_boundary(self, event: AspfRunBoundaryEvent) -> None:
        if event.boundary == "equivalence_surface_row":
            self.on_equivalence_surface_row(index=0, row=event.payload)


def traverse_forest_to_visitor(*, forest: Forest, visitor: AspfTraversalVisitor) -> None:
    nodes = sort_once(
        forest.nodes.values(),
        key=lambda node: node.node_id.sort_key(),
        source="aspf_visitors.traverse_forest_to_visitor.nodes",
    )
    for node in nodes:
        visitor.on_forest_node(node=node)
    alts = sort_once(
        forest.alts,
        key=lambda alt: alt.sort_key(),
        source="aspf_visitors.traverse_forest_to_visitor.alts",
    )
    for alt in alts:
        visitor.on_forest_alt(alt=alt)


def adapt_live_event_stream_to_visitor(
    *,
    one_cells: Iterable[Mapping[str, object]],
    two_cell_witnesses: Iterable[Mapping[str, object]],
    cofibration_witnesses: Iterable[Mapping[str, object]],
    surface_representatives: Mapping[str, str],
    visitor: AspfEventReplayVisitor,
) -> None:
    for index, one_cell in enumerate(one_cells):
        visitor.one_cell(AspfOneCellEvent(index=index, payload=one_cell))

    for index, witness in enumerate(two_cell_witnesses):
        visitor.two_cell(AspfTwoCellEvent(index=index, payload=witness))

    for index, cofibration in enumerate(cofibration_witnesses):
        visitor.cofibration(AspfCofibrationEvent(index=index, payload=cofibration))

    ordered_surfaces = sort_once(
        [str(surface) for surface in surface_representatives],
        source="aspf_visitors.adapt_live_event_stream_to_visitor.surface_representatives",
    )
    for surface in ordered_surfaces:
        visitor.surface_update(
            AspfSurfaceUpdateEvent(
                surface=surface,
                representative=str(surface_representatives.get(surface, "")),
            )
        )


def adapt_trace_event_iterator_to_visitor(
    *,
    events: Iterable[AspfTraceReplayEvent],
    visitor: AspfEventReplayVisitor,
) -> None:
    for event in events:
        event.dispatch_to(visitor)


def adapt_event_log_reader_iterator_to_visitor(
    *,
    event_log_rows: Iterable[Mapping[str, object]],
    visitor: AspfEventReplayVisitor,
) -> None:
    for row in event_log_rows:
        visitor.run_boundary(
            AspfRunBoundaryEvent(boundary="equivalence_surface_row", payload=row)
        )


@dataclass
class TracePayloadEmitter(NullAspfTraversalVisitor):
    one_cells: list[JSONValue] = field(default_factory=list)
    two_cell_witnesses: list[JSONValue] = field(default_factory=list)
    cofibration_witnesses: list[JSONValue] = field(default_factory=list)
    surface_representatives: dict[str, str] = field(default_factory=dict)

    def one_cell(self, event: AspfOneCellEvent) -> None:
        self.on_trace_one_cell(index=event.index, one_cell=event.payload)

    def two_cell(self, event: AspfTwoCellEvent) -> None:
        self.on_trace_two_cell_witness(index=event.index, witness=event.payload)

    def cofibration(self, event: AspfCofibrationEvent) -> None:
        self.on_trace_cofibration(index=event.index, cofibration=event.payload)

    def surface_update(self, event: AspfSurfaceUpdateEvent) -> None:
        self.on_trace_surface_representative(
            surface=event.surface,
            representative=event.representative,
        )

    def run_boundary(self, event: AspfRunBoundaryEvent) -> None:
        if event.boundary == "equivalence_surface_row":
            self.on_equivalence_surface_row(index=0, row=event.payload)

    def on_trace_one_cell(self, *, index: int, one_cell: Mapping[str, object]) -> None:
        self.one_cells.append({str(key): cast(JSONValue, one_cell[key]) for key in one_cell})

    def on_trace_two_cell_witness(
        self,
        *,
        index: int,
        witness: Mapping[str, object],
    ) -> None:
        self.two_cell_witnesses.append(
            {str(key): cast(JSONValue, witness[key]) for key in witness}
        )

    def on_trace_cofibration(
        self,
        *,
        index: int,
        cofibration: Mapping[str, object],
    ) -> None:
        self.cofibration_witnesses.append(
            {str(key): cast(JSONValue, cofibration[key]) for key in cofibration}
        )

    def on_trace_surface_representative(
        self,
        *,
        surface: str,
        representative: str,
    ) -> None:
        self.surface_representatives[str(surface)] = str(representative)


class OpportunityConfidenceProvenance(StrEnum):
    MORPHISM_WITNESS = "morphism_witness"
    REPRESENTATIVE_CONFLUENCE = "representative_confluence"
    INGRESS_OBSERVATION = "ingress_observation"


class OpportunityWitnessRequirement(StrEnum):
    NONE = "none"
    REPRESENTATIVE_PAIR = "representative_pair"
    TWO_CELL_WITNESS = "two_cell_witness"


class OpportunityActionabilityState(StrEnum):
    ACTIONABLE = "actionable"
    OBSERVATIONAL = "observational"


@dataclass(frozen=True)
class OpportunityDecisionProtocol:
    opportunity_id: str
    kind: str
    affected_surfaces: tuple[str, ...]
    witness_ids: tuple[str, ...]
    reason: str
    confidence_provenance: OpportunityConfidenceProvenance
    witness_requirement: OpportunityWitnessRequirement
    actionability: OpportunityActionabilityState

    def confidence(self) -> float:
        score = 0.36
        if self.confidence_provenance is OpportunityConfidenceProvenance.INGRESS_OBSERVATION:
            score += 0.14
        if self.confidence_provenance is OpportunityConfidenceProvenance.REPRESENTATIVE_CONFLUENCE:
            score += 0.18
        if self.confidence_provenance is OpportunityConfidenceProvenance.MORPHISM_WITNESS:
            score += 0.26
        if self.witness_requirement is OpportunityWitnessRequirement.TWO_CELL_WITNESS:
            score += 0.14
        if self.witness_ids:
            score += min(0.18, 0.06 * len(self.witness_ids))
        return round(min(score, 0.99), 2)

    def as_row(self) -> JSONObject:
        return {
            "opportunity_id": self.opportunity_id,
            "kind": self.kind,
            "confidence": self.confidence(),
            "confidence_provenance": self.confidence_provenance,
            "witness_requirement": self.witness_requirement,
            "actionability": self.actionability,
            "affected_surfaces": list(self.affected_surfaces),
            "witness_ids": list(self.witness_ids),
            "reason": self.reason,
        }

    def as_rewrite_plan(self) -> JSONObject:
        return {
            "plan_id": f"rewrite:{self.opportunity_id}",
            "kind": "aspf_opportunity",
            "priority": self.confidence(),
            "actionability": self.actionability,
            "opportunity_id": self.opportunity_id,
            "affected_surfaces": list(self.affected_surfaces),
            "required_witnesses": list(self.witness_ids),
            "decision_basis": {
                "confidence_provenance": self.confidence_provenance,
                "witness_requirement": self.witness_requirement,
            },
            "summary": self.reason,
        }


@dataclass
class OpportunityPayloadEmitter(NullAspfTraversalVisitor):
    _materialize_kinds_by_resume_ref: dict[str, set[str]] = field(default_factory=dict)
    _representative_to_surfaces: dict[str, list[str]] = field(default_factory=dict)
    _representative_witness_ids: dict[str, set[str]] = field(default_factory=dict)
    _fungible_opportunities: list[OpportunityDecisionProtocol] = field(default_factory=list)

    def one_cell(self, event: AspfOneCellEvent) -> None:
        self.on_trace_one_cell(index=event.index, one_cell=event.payload)

    def two_cell(self, event: AspfTwoCellEvent) -> None:
        self.on_trace_two_cell_witness(index=event.index, witness=event.payload)

    def cofibration(self, event: AspfCofibrationEvent) -> None:
        self.on_trace_cofibration(index=event.index, cofibration=event.payload)

    def surface_update(self, event: AspfSurfaceUpdateEvent) -> None:
        self.on_trace_surface_representative(
            surface=event.surface,
            representative=event.representative,
        )

    def run_boundary(self, event: AspfRunBoundaryEvent) -> None:
        if event.boundary == "equivalence_surface_row":
            self.on_equivalence_surface_row(index=0, row=event.payload)

    def on_trace_one_cell(self, *, index: int, one_cell: Mapping[str, object]) -> None:
        kind = str(one_cell.get("kind", ""))
        metadata = cast(Mapping[str, object], one_cell.get("metadata", {}))
        resume_ref = ""
        for candidate_key in ("state_path", "import_state_path"):
            candidate = str(metadata.get(candidate_key, "")).strip()
            if candidate:
                resume_ref = candidate
                break
        if resume_ref:
            self._materialize_kinds_by_resume_ref.setdefault(resume_ref, set()).add(kind)

    def on_trace_surface_representative(
        self,
        *,
        surface: str,
        representative: str,
    ) -> None:
        self._representative_to_surfaces.setdefault(representative, []).append(surface)

    def on_trace_two_cell_witness(
        self,
        *,
        index: int,
        witness: Mapping[str, object],
    ) -> None:
        witness_id = str(witness.get("witness_id", "")).strip()
        if not witness_id:
            return
        left = str(witness.get("left_representative", "")).strip()
        right = str(witness.get("right_representative", "")).strip()
        for representative in (left, right):
            if representative:
                self._representative_witness_ids.setdefault(representative, set()).add(witness_id)

    def on_equivalence_surface_row(
        self,
        *,
        index: int,
        row: Mapping[str, object],
    ) -> None:
        surface = str(row.get("surface", "")).strip()
        witness_id = row.get("witness_id")
        if (
            str(row.get("classification")) == "non_drift"
            and bool(surface)
            and witness_id not in (None, "")
        ):
            self._fungible_opportunities.append(
                OpportunityDecisionProtocol(
                    opportunity_id=f"opp:fungible-substitution:{surface}",
                    kind="fungible_execution_path_substitution",
                    affected_surfaces=(surface,),
                    witness_ids=(str(witness_id),),
                    reason="2-cell witness links baseline/current representatives",
                    confidence_provenance=OpportunityConfidenceProvenance.MORPHISM_WITNESS,
                    witness_requirement=OpportunityWitnessRequirement.TWO_CELL_WITNESS,
                    actionability=OpportunityActionabilityState.ACTIONABLE,
                )
            )

    def _build_decisions(self) -> list[OpportunityDecisionProtocol]:
        decisions: list[OpportunityDecisionProtocol] = []
        for resume_ref in sort_once(
            self._materialize_kinds_by_resume_ref,
            source="aspf_visitors.OpportunityPayloadEmitter.materialize.resume_ref",
        ):
            kinds = self._materialize_kinds_by_resume_ref[resume_ref]
            fused = int("resume_load" in kinds and "resume_write" in kinds)
            decisions.append(
                OpportunityDecisionProtocol(
                    opportunity_id=f"opp:materialize-load-fusion:{resume_ref}",
                    kind=("materialize_load_observed", "materialize_load_fusion")[fused],
                    affected_surfaces=(),
                    witness_ids=(),
                    reason=(
                        "resume boundary observed state reference",
                        "resume load and write boundaries share state reference",
                    )[fused],
                    confidence_provenance=OpportunityConfidenceProvenance.INGRESS_OBSERVATION,
                    witness_requirement=OpportunityWitnessRequirement.NONE,
                    actionability=(
                        OpportunityActionabilityState.OBSERVATIONAL,
                        OpportunityActionabilityState.ACTIONABLE,
                    )[fused],
                )
            )

        for representative in sort_once(
            self._representative_to_surfaces,
            source="aspf_visitors.OpportunityPayloadEmitter.representative",
        ):
            surfaces = tuple(
                sort_once(
                    self._representative_to_surfaces[representative],
                    source="aspf_visitors.OpportunityPayloadEmitter.representative_surfaces",
                )
            )
            digest = hashlib.sha256(representative.encode("utf-8")).hexdigest()[:12]
            witness_ids = tuple(
                sort_once(
                    self._representative_witness_ids.get(representative, set()),
                    source="aspf_visitors.OpportunityPayloadEmitter.representative_witnesses",
                )
            )
            decisions.append(
                OpportunityDecisionProtocol(
                    opportunity_id=f"opp:reusable-boundary:{digest}",
                    kind="reusable_boundary_artifact",
                    affected_surfaces=surfaces,
                    witness_ids=witness_ids,
                    reason="multiple semantic surfaces share deterministic representative",
                    confidence_provenance=(
                        OpportunityConfidenceProvenance.REPRESENTATIVE_CONFLUENCE
                        if not witness_ids
                        else OpportunityConfidenceProvenance.MORPHISM_WITNESS
                    ),
                    witness_requirement=OpportunityWitnessRequirement.REPRESENTATIVE_PAIR,
                    actionability=(
                        OpportunityActionabilityState.OBSERVATIONAL
                        if len(surfaces) < 2
                        else OpportunityActionabilityState.ACTIONABLE
                    ),
                )
            )

        decisions.extend(self._fungible_opportunities)
        return sorted(decisions, key=lambda item: item.opportunity_id)

    def build_rows(self) -> list[JSONObject]:
        return [decision.as_row() for decision in self._build_decisions()]

    def build_rewrite_plans(self) -> list[JSONObject]:
        decisions = [
            decision
            for decision in self._build_decisions()
            if decision.actionability is OpportunityActionabilityState.ACTIONABLE
        ]
        return [decision.as_rewrite_plan() for decision in decisions]


@dataclass
class StatePayloadEmitter:
    trace: JSONObject = field(default_factory=dict)
    equivalence: JSONObject = field(default_factory=dict)
    opportunities: JSONObject = field(default_factory=dict)

    def set_trace_payload(self, payload: Mapping[str, object]) -> None:
        self.trace = {str(key): cast(JSONValue, payload[key]) for key in payload}

    def set_equivalence_payload(self, payload: Mapping[str, object]) -> None:
        self.equivalence = {str(key): cast(JSONValue, payload[key]) for key in payload}

    def set_opportunities_payload(self, payload: Mapping[str, object]) -> None:
        self.opportunities = {str(key): cast(JSONValue, payload[key]) for key in payload}
