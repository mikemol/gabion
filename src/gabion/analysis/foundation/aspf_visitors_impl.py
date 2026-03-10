from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from functools import singledispatchmethod
import hashlib
from typing import Literal, Mapping, Protocol, TypeAlias, cast

from gabion.analysis import aspf_rule_engine
from gabion.analysis.aspf.aspf import Alt, Forest, Node, NodeId
from gabion.analysis.foundation.resume_codec import mapping_default_empty, sequence_optional
from gabion.invariants import never
from gabion.analysis.foundation.wire_types import WireObject, WireValue
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



@dataclass(frozen=True)
class AspfTwoCellEvent:
    index: int
    payload: Mapping[str, object]



@dataclass(frozen=True)
class AspfCofibrationEvent:
    index: int
    payload: Mapping[str, object]



@dataclass(frozen=True)
class AspfSurfaceUpdateEvent:
    surface: str
    representative: str



@dataclass(frozen=True)
class AspfRunBoundaryEvent:
    boundary: Literal["equivalence_surface_row"]
    payload: Mapping[str, object]


AspfTraceReplayEvent: TypeAlias = (
    AspfOneCellEvent
    | AspfTwoCellEvent
    | AspfCofibrationEvent
    | AspfSurfaceUpdateEvent
    | AspfRunBoundaryEvent
)


class AspfEventReplayVisitor(Protocol):
    """Canonical low-level ASPF replay hook protocol."""

    def on_replay_event(self, event: AspfTraceReplayEvent) -> None: ...


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

    @singledispatchmethod
    def on_replay_event(self, event: AspfTraceReplayEvent) -> None:
        never("invalid aspf replay event", kind=type(event).__name__)

    @on_replay_event.register
    def _(self, event: AspfOneCellEvent) -> None:
        self.on_trace_one_cell(index=event.index, one_cell=event.payload)

    @on_replay_event.register
    def _(self, event: AspfTwoCellEvent) -> None:
        self.on_trace_two_cell_witness(index=event.index, witness=event.payload)

    @on_replay_event.register
    def _(self, event: AspfCofibrationEvent) -> None:
        self.on_trace_cofibration(index=event.index, cofibration=event.payload)

    @on_replay_event.register
    def _(self, event: AspfSurfaceUpdateEvent) -> None:
        self.on_trace_surface_representative(
            surface=event.surface,
            representative=event.representative,
        )

    @on_replay_event.register
    def _(self, event: AspfRunBoundaryEvent) -> None:
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


class TwoCellReplayNormalizationKind(StrEnum):
    VALID = "valid"
    SKIP = "skip"


@dataclass(frozen=True)
class TwoCellReplayNormalizationOutcome:
    kind: TwoCellReplayNormalizationKind
    payload: Mapping[str, WireValue] = field(default_factory=dict)


def _normalize_two_cell_witness_for_replay(
    witness: Mapping[str, WireValue],
) -> TwoCellReplayNormalizationOutcome:
    normalized = {str(key): witness[key] for key in witness}
    witness_id = str(normalized.get("witness_id", "")).strip()
    left = str(normalized.get("left_representative", "")).strip()
    right = str(normalized.get("right_representative", "")).strip()

    if not left:
        left = str(mapping_default_empty(normalized.get("left", {})).get("representative", "")).strip()
    if not right:
        right = str(mapping_default_empty(normalized.get("right", {})).get("representative", "")).strip()

    if witness_id and left and right:
        normalized["witness_id"] = witness_id
        normalized["left_representative"] = left
        normalized["right_representative"] = right
        return TwoCellReplayNormalizationOutcome(
            kind=TwoCellReplayNormalizationKind.VALID,
            payload=normalized,
        )
    return TwoCellReplayNormalizationOutcome(kind=TwoCellReplayNormalizationKind.SKIP)


def adapt_live_event_stream_to_visitor(
    *,
    one_cells: Iterable[Mapping[str, object]],
    two_cell_witnesses: Iterable[Mapping[str, object]],
    cofibration_witnesses: Iterable[Mapping[str, object]],
    surface_representatives: Mapping[str, str],
    visitor: AspfEventReplayVisitor,
) -> None:
    for index, one_cell in enumerate(one_cells):
        visitor.on_replay_event(AspfOneCellEvent(index=index, payload=one_cell))

    for index, witness in enumerate(two_cell_witnesses):
        normalized_witness = _normalize_two_cell_witness_for_replay(witness)
        if normalized_witness.kind is TwoCellReplayNormalizationKind.VALID:
            visitor.on_replay_event(
                AspfTwoCellEvent(index=index, payload=normalized_witness.payload)
            )

    for index, cofibration in enumerate(cofibration_witnesses):
        visitor.on_replay_event(AspfCofibrationEvent(index=index, payload=cofibration))

    ordered_surfaces = sort_once(
        [str(surface) for surface in surface_representatives],
        source="aspf_visitors.adapt_live_event_stream_to_visitor.surface_representatives",
    )
    for surface in ordered_surfaces:
        visitor.on_replay_event(
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
        visitor.on_replay_event(event)


def adapt_event_log_reader_iterator_to_visitor(
    *,
    event_log_rows: Iterable[Mapping[str, object]],
    visitor: AspfEventReplayVisitor,
) -> None:
    for row in event_log_rows:
        visitor.on_replay_event(
            AspfRunBoundaryEvent(boundary="equivalence_surface_row", payload=row)
        )


def replay_iterator_inputs_to_visitor(
    *,
    one_cells: Iterable[Mapping[str, object]],
    two_cell_witnesses: Iterable[Mapping[str, object]],
    cofibration_witnesses: Iterable[Mapping[str, object]],
    surface_representatives: Mapping[str, str],
    equivalence_surface_rows: Iterable[Mapping[str, object]],
    visitor: AspfEventReplayVisitor,
) -> None:
    adapt_live_event_stream_to_visitor(
        one_cells=one_cells,
        two_cell_witnesses=two_cell_witnesses,
        cofibration_witnesses=cofibration_witnesses,
        surface_representatives=surface_representatives,
        visitor=visitor,
    )
    adapt_event_log_reader_iterator_to_visitor(
        event_log_rows=equivalence_surface_rows,
        visitor=visitor,
    )


@dataclass
class TracePayloadEmitter(NullAspfTraversalVisitor):
    one_cells: list[WireValue] = field(default_factory=list)
    two_cell_witnesses: list[WireValue] = field(default_factory=list)
    cofibration_witnesses: list[WireValue] = field(default_factory=list)
    surface_representatives: dict[str, str] = field(default_factory=dict)

    def on_trace_one_cell(self, *, index: int, one_cell: Mapping[str, object]) -> None:
        self.one_cells.append({str(key): cast(WireValue, one_cell[key]) for key in one_cell})

    def on_trace_two_cell_witness(
        self,
        *,
        index: int,
        witness: Mapping[str, object],
    ) -> None:
        self.two_cell_witnesses.append(
            {str(key): cast(WireValue, witness[key]) for key in witness}
        )

    def on_trace_cofibration(
        self,
        *,
        index: int,
        cofibration: Mapping[str, object],
    ) -> None:
        self.cofibration_witnesses.append(
            {str(key): cast(WireValue, cofibration[key]) for key in cofibration}
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
    COFIBRATION_WITNESS = "cofibration_witness"


class OpportunityActionabilityState(StrEnum):
    ACTIONABLE = "actionable"
    OBSERVATIONAL = "observational"


class OpportunityStructure(StrEnum):
    ONE_CELL = "one_cell"
    TWO_CELL = "two_cell"
    COFIBRATION = "cofibration"


@dataclass(frozen=True)
class OpportunityAlgebraicObservation:
    structure: OpportunityStructure
    subject_id: str
    one_cell_count: int = 0
    two_cell_count: int = 0
    cofibration_count: int = 0
    one_cell_kinds: frozenset[str] = frozenset()
    surfaces: tuple[str, ...] = ()
    witness_ids: tuple[str, ...] = ()
    classification: str = ""
    representative: str = ""
    one_cell_carrier_observed: bool = False
    cofibration_entry_count: int = 0
    witness_chain: tuple[str, ...] = ()


@dataclass(frozen=True)
class OpportunityProofObligation:
    obligation: str
    satisfied: bool
    predicate: str
    detail: str

    def as_row(self) -> WireObject:
        return {
            "obligation": self.obligation,
            "satisfied": self.satisfied,
            "predicate": self.predicate,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class OpportunityDecisionProtocol:
    opportunity_id: str
    rule_id: str
    kind: str
    canonical_identity: WireObject
    affected_surfaces: tuple[str, ...]
    witness_ids: tuple[str, ...]
    reason: str
    confidence_provenance: OpportunityConfidenceProvenance
    witness_requirement: OpportunityWitnessRequirement
    actionability: OpportunityActionabilityState
    proof_obligations: tuple[OpportunityProofObligation, ...] = ()
    carrier_subgraph: WireObject = field(default_factory=dict)
    witness_chain: tuple[str, ...] = ()
    opportunity_hash: str = ""
    observation: WireObject = field(default_factory=dict)

    def failed_obligations(self) -> tuple[str, ...]:
        return tuple(
            obligation.obligation
            for obligation in self.proof_obligations
            if not obligation.satisfied
        )

    def confidence(self) -> float:
        if self.proof_obligations:
            satisfied = sum(1 for obligation in self.proof_obligations if obligation.satisfied)
            return round(satisfied / len(self.proof_obligations), 2)
        return 0.0

    def as_row(self) -> WireObject:
        return {
            "opportunity_id": self.opportunity_id,
            "rule_id": self.rule_id,
            "kind": self.kind,
            "confidence": self.confidence(),
            "canonical_identity": self.canonical_identity,
            "confidence_provenance": self.confidence_provenance,
            "witness_requirement": self.witness_requirement,
            "actionability": self.actionability,
            "affected_surfaces": list(self.affected_surfaces),
            "witness_ids": list(self.witness_ids),
            "reason": self.reason,
            "carrier_subgraph": self.carrier_subgraph,
            "witness_chain": list(self.witness_chain),
            "observation": self.observation,
            "proof_obligations": [
                obligation.as_row() for obligation in self.proof_obligations
            ],
            "failed_obligations": list(self.failed_obligations()),
            **({"opportunity_hash": self.opportunity_hash} if self.opportunity_hash else {}),
        }

    def as_rewrite_plan(self) -> WireObject:
        return {
            "plan_id": f"rewrite:{self.opportunity_id}",
            "kind": "aspf_opportunity",
            "rule_id": self.rule_id,
            "priority": self.confidence(),
            "actionability": self.actionability,
            "opportunity_id": self.opportunity_id,
            "canonical_identity": self.canonical_identity,
            "affected_surfaces": list(self.affected_surfaces),
            "required_witnesses": list(self.witness_ids),
            "carrier_subgraph": self.carrier_subgraph,
            "witness_chain": list(self.witness_chain),
            "decision_basis": {
                "confidence_provenance": self.confidence_provenance,
                "witness_requirement": self.witness_requirement,
                "proof_obligations": [
                    obligation.as_row() for obligation in self.proof_obligations
                ],
                "failed_obligations": list(self.failed_obligations()),
            },
            "summary": self.reason,
            **({"opportunity_hash": self.opportunity_hash} if self.opportunity_hash else {}),
        }


def _node_id_identity_payload(node_id: NodeId) -> WireObject:
    fingerprint_kind, fingerprint_parts = node_id.fingerprint()
    return {
        "node_id": node_id.as_dict(),
        "node_fingerprint": [fingerprint_kind, list(fingerprint_parts)],
    }


def _reusable_boundary_obligations(
    observation: OpportunityAlgebraicObservation,
) -> tuple[OpportunityProofObligation, ...]:
    return (
        OpportunityProofObligation(
            obligation="representative_collision",
            satisfied=len(observation.surfaces) >= 2,
            predicate="multiple semantic surfaces map to the same representative",
            detail=f"surface_count={len(observation.surfaces)}",
        ),
        OpportunityProofObligation(
            obligation="one_cell_carrier_presence",
            satisfied=observation.one_cell_carrier_observed,
            predicate="representative is realized by at least one observed 1-cell",
            detail=f"representative={observation.representative}",
        ),
        OpportunityProofObligation(
            obligation="two_cell_isomorphy_witness",
            satisfied=bool(observation.witness_ids),
            predicate="2-cell witnesses explicitly link representative to an equivalent peer",
            detail=f"witness_count={len(observation.witness_ids)}",
        ),
        OpportunityProofObligation(
            obligation="cofibration_support",
            satisfied=observation.cofibration_entry_count > 0,
            predicate="at least one cofibration entry validates domain→ASPF carrier embedding",
            detail=f"cofibration_entry_count={observation.cofibration_entry_count}",
        ),
    )


def _resume_boundary_obligations(
    observation: OpportunityAlgebraicObservation,
    *,
    fused: bool,
) -> tuple[OpportunityProofObligation, ...]:
    return (
        OpportunityProofObligation(
            obligation="resume_state_reference_observed",
            satisfied=True,
            predicate="resume load/write metadata carry an explicit state reference",
            detail=f"resume_ref={observation.subject_id}",
        ),
        OpportunityProofObligation(
            obligation="resume_bidirectional_boundary",
            satisfied=fused,
            predicate="both resume_load and resume_write occurred for the same state reference",
            detail=f"observed_kinds={sorted(observation.one_cell_kinds)}",
        ),
    )


def _build_reusable_boundary_decision(
    observation: OpportunityAlgebraicObservation,
) -> OpportunityDecisionProtocol:
    obligations = _reusable_boundary_obligations(observation)
    actionable = all(obligation.satisfied for obligation in obligations)
    return OpportunityDecisionProtocol(
        opportunity_id=f"opp:reusable-boundary:{observation.subject_id}",
        rule_id="aspf.opportunity.reusable_boundary_artifact",
        kind="reusable_boundary_artifact",
        canonical_identity=_node_id_identity_payload(
            NodeId(
                kind="Opportunity:ReusableBoundaryRepresentative",
                key=(observation.subject_id,),
            )
        ),
        affected_surfaces=observation.surfaces,
        witness_ids=observation.witness_ids,
        reason=(
            "witnessed representative isomorphy supports reusable boundary artifact"
            if actionable
            else "representative collision observed but explicit isomorphy obligations remain open"
        ),
        confidence_provenance=(
            OpportunityConfidenceProvenance.REPRESENTATIVE_CONFLUENCE
            if not observation.witness_ids
            else OpportunityConfidenceProvenance.MORPHISM_WITNESS
        ),
        witness_requirement=OpportunityWitnessRequirement.REPRESENTATIVE_PAIR,
        actionability=(
            OpportunityActionabilityState.ACTIONABLE
            if actionable
            else OpportunityActionabilityState.OBSERVATIONAL
        ),
        proof_obligations=obligations,
        carrier_subgraph={
            "representative": observation.representative,
            "surface_count": len(observation.surfaces),
            "one_cell_carrier_observed": observation.one_cell_carrier_observed,
            "cofibration_entry_count": observation.cofibration_entry_count,
        },
        witness_chain=observation.witness_chain,
        opportunity_hash=observation.subject_id,
        observation=_observation_row_payload(observation),
    )


def _observation_row_payload(observation: OpportunityAlgebraicObservation) -> WireObject:
    return {
        "structure": observation.structure.value,
        "one_cell_count": observation.one_cell_count,
        "two_cell_count": observation.two_cell_count,
        "cofibration_count": observation.cofibration_count,
        "surface_count": len(observation.surfaces),
        "classification": observation.classification,
        "has_resume_load": "resume_load" in observation.one_cell_kinds,
        "has_resume_write": "resume_write" in observation.one_cell_kinds,
    }


def _observation_policy_payload(observation: OpportunityAlgebraicObservation) -> WireObject:
    return {
        "observation": _observation_row_payload(observation),
        "witness": {
            "two_cell": observation.two_cell_count > 0,
            "cofibration": observation.cofibration_count > 0,
        },
    }


def _build_materialize_load_observed_decision(
    observation: OpportunityAlgebraicObservation,
) -> OpportunityDecisionProtocol:
    return OpportunityDecisionProtocol(
        opportunity_id=f"opp:materialize-load-observed:{observation.subject_id}",
        rule_id="aspf.opportunity.materialize_load_observed",
        kind="materialize_load_observed",
        canonical_identity=_node_id_identity_payload(
            NodeId(
                kind="Opportunity:MaterializeLoad",
                key=(observation.subject_id, 0),
            )
        ),
        affected_surfaces=(),
        witness_ids=(),
        reason="resume boundary observed state reference",
        confidence_provenance=OpportunityConfidenceProvenance.INGRESS_OBSERVATION,
        witness_requirement=OpportunityWitnessRequirement.NONE,
        actionability=OpportunityActionabilityState.OBSERVATIONAL,
        proof_obligations=_resume_boundary_obligations(
            observation,
            fused=False,
        ),
        carrier_subgraph={
            "resume_ref": observation.subject_id,
            "kinds": sorted(observation.one_cell_kinds),
        },
        observation=_observation_row_payload(observation),
    )


def _build_materialize_load_fusion_decision(
    observation: OpportunityAlgebraicObservation,
) -> OpportunityDecisionProtocol:
    return OpportunityDecisionProtocol(
        opportunity_id=f"opp:materialize-load-fusion:{observation.subject_id}",
        rule_id="aspf.opportunity.materialize_load_fusion",
        kind="materialize_load_fusion",
        canonical_identity=_node_id_identity_payload(
            NodeId(
                kind="Opportunity:MaterializeLoad",
                key=(observation.subject_id, 1),
            )
        ),
        affected_surfaces=(),
        witness_ids=(),
        reason="resume load and write boundaries share state reference",
        confidence_provenance=OpportunityConfidenceProvenance.INGRESS_OBSERVATION,
        witness_requirement=OpportunityWitnessRequirement.NONE,
        actionability=OpportunityActionabilityState.ACTIONABLE,
        proof_obligations=_resume_boundary_obligations(
            observation,
            fused=True,
        ),
        carrier_subgraph={
            "resume_ref": observation.subject_id,
            "kinds": sorted(observation.one_cell_kinds),
        },
        observation=_observation_row_payload(observation),
    )


def _build_fungible_substitution_decision(
    observation: OpportunityAlgebraicObservation,
) -> OpportunityDecisionProtocol:
    return OpportunityDecisionProtocol(
        opportunity_id=f"opp:fungible-substitution:{observation.subject_id}",
        rule_id="aspf.opportunity.fungible_execution_path_substitution",
        kind="fungible_execution_path_substitution",
        canonical_identity=_node_id_identity_payload(
            NodeId(
                kind="Opportunity:FungibleSubstitution",
                key=(
                    observation.subject_id,
                    observation.witness_ids[0] if observation.witness_ids else "",
                ),
            )
        ),
        affected_surfaces=observation.surfaces,
        witness_ids=observation.witness_ids,
        reason="2-cell witness links baseline/current representatives",
        confidence_provenance=OpportunityConfidenceProvenance.MORPHISM_WITNESS,
        witness_requirement=OpportunityWitnessRequirement.TWO_CELL_WITNESS,
        actionability=OpportunityActionabilityState.ACTIONABLE,
        proof_obligations=(
            OpportunityProofObligation(
                obligation="two_cell_equivalence_witness",
                satisfied=True,
                predicate="equivalence row classification is non_drift with witness_id",
                detail="surface row provided witnessed non-drift classification",
            ),
        ),
        carrier_subgraph={
            "surface": observation.surfaces[0] if observation.surfaces else "",
            "classification": observation.classification,
        },
        witness_chain=observation.witness_ids[:1],
        observation=_observation_row_payload(observation),
    )


def _build_cofibration_prime_embedding_decision(
    observation: OpportunityAlgebraicObservation,
) -> OpportunityDecisionProtocol:
    return OpportunityDecisionProtocol(
        opportunity_id=f"opp:cofibration-prime-embedding:{observation.subject_id}",
        rule_id="aspf.opportunity.cofibration_prime_embedding_reuse",
        kind="cofibration_prime_embedding_reuse",
        canonical_identity=_node_id_identity_payload(
            NodeId(
                kind="Opportunity:CofibrationPrimeEmbedding",
                key=(observation.subject_id,),
            )
        ),
        affected_surfaces=(),
        witness_ids=observation.witness_ids,
        reason="domain-to-ASPF cofibration witness preserves prime embedding",
        confidence_provenance=OpportunityConfidenceProvenance.MORPHISM_WITNESS,
        witness_requirement=OpportunityWitnessRequirement.COFIBRATION_WITNESS,
        actionability=OpportunityActionabilityState.OBSERVATIONAL,
        proof_obligations=(
            OpportunityProofObligation(
                obligation="cofibration_support",
                satisfied=observation.cofibration_count > 0,
                predicate="at least one cofibration entry validates domain→ASPF carrier embedding",
                detail=f"cofibration_entry_count={observation.cofibration_count}",
            ),
        ),
        carrier_subgraph={
            "canonical_identity_kind": observation.subject_id,
            "cofibration_entry_count": observation.cofibration_count,
        },
        witness_chain=observation.witness_ids,
        observation=_observation_row_payload(observation),
    )


_DSL_DECISION_BUILDERS = {
    "aspf.opportunity.materialize_load_observed": _build_materialize_load_observed_decision,
    "aspf.opportunity.materialize_load_fusion": _build_materialize_load_fusion_decision,
    "aspf.opportunity.reusable_boundary_artifact": _build_reusable_boundary_decision,
    "aspf.opportunity.fungible_execution_path_substitution": _build_fungible_substitution_decision,
    "aspf.opportunity.cofibration_prime_embedding_reuse": _build_cofibration_prime_embedding_decision,
}


@dataclass
class OpportunityPayloadEmitter(NullAspfTraversalVisitor):
    _materialize_kinds_by_resume_ref: dict[str, set[str]] = field(default_factory=dict)
    _representative_to_surfaces: dict[str, list[str]] = field(default_factory=dict)
    _representative_witness_ids: dict[str, set[str]] = field(default_factory=dict)
    _witness_chain_by_representative: dict[str, set[str]] = field(default_factory=dict)
    _representatives_observed_in_one_cells: set[str] = field(default_factory=set)
    _non_drift_witness_ids_by_surface: dict[str, set[str]] = field(default_factory=dict)
    _cofibration_entry_count_by_kind: dict[str, int] = field(default_factory=dict)
    _cofibration_entry_count_total: int = 0

    def on_trace_one_cell(self, *, index: int, one_cell: Mapping[str, object]) -> None:
        kind = str(one_cell.get("kind", ""))
        metadata = cast(Mapping[str, object], one_cell.get("metadata", {}))
        representative = str(one_cell.get("representative", "")).strip()
        if representative:
            self._representatives_observed_in_one_cells.add(representative)
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
        left = str(witness.get("left_representative", "")).strip()
        right = str(witness.get("right_representative", "")).strip()
        for representative in (left, right):
            self._representative_witness_ids.setdefault(representative, set()).add(witness_id)
            self._witness_chain_by_representative.setdefault(representative, set()).add(
                f"{left}->{right}"
            )

    def on_trace_cofibration(
        self,
        *,
        index: int,
        cofibration: Mapping[str, object],
    ) -> None:
        canonical_identity_kind = str(cofibration.get("canonical_identity_kind", "")).strip()
        normalized_cofibration = mapping_default_empty(cofibration.get("cofibration", {}))
        normalized_entries = sequence_optional(normalized_cofibration.get("entries"))
        entry_count = len(normalized_entries) if normalized_entries is not None else 0
        self._cofibration_entry_count_total += entry_count
        if canonical_identity_kind:
            self._cofibration_entry_count_by_kind[canonical_identity_kind] = entry_count

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
            self._non_drift_witness_ids_by_surface.setdefault(surface, set()).add(str(witness_id))

    def _normalize_observations(self) -> list[OpportunityAlgebraicObservation]:
        observations: list[OpportunityAlgebraicObservation] = []

        for resume_ref in sort_once(
            self._materialize_kinds_by_resume_ref,
            source="aspf_visitors.OpportunityPayloadEmitter.materialize.resume_ref",
        ):
            kinds = frozenset(self._materialize_kinds_by_resume_ref[resume_ref])
            observations.append(
                OpportunityAlgebraicObservation(
                    structure=OpportunityStructure.ONE_CELL,
                    subject_id=resume_ref,
                    one_cell_count=len(kinds),
                    one_cell_kinds=kinds,
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
            witness_chain = tuple(
                sort_once(
                    self._witness_chain_by_representative.get(representative, set()),
                    source="aspf_visitors.OpportunityPayloadEmitter.representative_witness_chain",
                )
            )
            observations.append(
                OpportunityAlgebraicObservation(
                    structure=OpportunityStructure.TWO_CELL,
                    subject_id=digest,
                    one_cell_count=len(surfaces),
                    two_cell_count=len(witness_ids),
                    surfaces=surfaces,
                    witness_ids=witness_ids,
                    representative=representative,
                    one_cell_carrier_observed=(
                        representative in self._representatives_observed_in_one_cells
                    ),
                    cofibration_entry_count=self._cofibration_entry_count_total,
                    witness_chain=witness_chain,
                )
            )

        for surface in sort_once(
            self._non_drift_witness_ids_by_surface,
            source="aspf_visitors.OpportunityPayloadEmitter.non_drift_surfaces",
        ):
            witness_ids = tuple(
                sort_once(
                    self._non_drift_witness_ids_by_surface[surface],
                    source="aspf_visitors.OpportunityPayloadEmitter.non_drift_witnesses",
                )
            )
            observations.append(
                OpportunityAlgebraicObservation(
                    structure=OpportunityStructure.TWO_CELL,
                    subject_id=surface,
                    two_cell_count=len(witness_ids),
                    surfaces=(surface,),
                    witness_ids=witness_ids,
                    classification="non_drift",
                )
            )

        for identity_kind in sort_once(
            self._cofibration_entry_count_by_kind,
            source="aspf_visitors.OpportunityPayloadEmitter.cofibration_kind",
        ):
            observations.append(
                OpportunityAlgebraicObservation(
                    structure=OpportunityStructure.COFIBRATION,
                    subject_id=identity_kind,
                    cofibration_count=self._cofibration_entry_count_by_kind[identity_kind],
                    witness_ids=(identity_kind,),
                )
            )

        return observations

    def _build_decisions(self) -> list[OpportunityDecisionProtocol]:
        decisions: list[OpportunityDecisionProtocol] = []
        for observation in self._normalize_observations():
            policy_payload = _observation_policy_payload(observation)
            policy_decision = aspf_rule_engine.classify_aspf_opportunity(policy_payload)
            builder = _DSL_DECISION_BUILDERS.get(policy_decision.rule_id)
            if builder in (None,):
                continue
            decisions.append(builder(observation))
        return sorted(decisions, key=lambda item: item.opportunity_id)

    def build_rows(self) -> list[WireObject]:
        return [decision.as_row() for decision in self._build_decisions()]

    def build_rewrite_plans(self) -> list[WireObject]:
        decisions = [
            decision
            for decision in self._build_decisions()
            if decision.actionability is OpportunityActionabilityState.ACTIONABLE
        ]
        return [decision.as_rewrite_plan() for decision in decisions]


@dataclass
class StatePayloadEmitter:
    trace: WireObject = field(default_factory=dict)
    equivalence: WireObject = field(default_factory=dict)
    opportunities: WireObject = field(default_factory=dict)

    def set_trace_payload(self, payload: Mapping[str, object]) -> None:
        self.trace = {str(key): cast(WireValue, payload[key]) for key in payload}

    def set_equivalence_payload(self, payload: Mapping[str, object]) -> None:
        self.equivalence = {str(key): cast(WireValue, payload[key]) for key in payload}

    def set_opportunities_payload(self, payload: Mapping[str, object]) -> None:
        self.opportunities = {str(key): cast(WireValue, payload[key]) for key in payload}
