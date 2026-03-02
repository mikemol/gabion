from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from gabion.analysis.aspf import Alt, Forest, Node
from gabion.analysis.aspf_visitors import (
    AspfCofibrationEvent,
    AspfOneCellEvent,
    AspfRunBoundaryEvent,
    AspfSurfaceUpdateEvent,
    AspfTwoCellEvent,
    NullAspfTraversalVisitor,
    OpportunityPayloadEmitter,
    OpportunityActionabilityState,
    OpportunityAlgebraicObservation,
    OpportunityAlgebraicPredicate,
    OpportunityStructure,
    OpportunityDecisionProtocol,
    OpportunityConfidenceProvenance,
    OpportunityWitnessRequirement,
    TwoCellReplayNormalizationKind,
    _normalize_two_cell_witness_for_replay,
    adapt_event_log_reader_iterator_to_visitor,
    adapt_live_event_stream_to_visitor,
    adapt_trace_event_iterator_to_visitor,
    traverse_forest_to_visitor,
)
from gabion.exceptions import NeverRaise


@dataclass
class _ForestOrderCaptureVisitor(NullAspfTraversalVisitor):
    node_kinds: list[str] = field(default_factory=list)
    alt_kinds: list[str] = field(default_factory=list)

    def on_forest_node(self, *, node: Node) -> None:
        self.node_kinds.append(node.kind)

    def on_forest_alt(self, *, alt: Alt) -> None:
        self.alt_kinds.append(alt.kind)


def test_traverse_forest_to_visitor_uses_deterministic_order() -> None:
    forest = Forest()
    alpha = forest.add_node("KindB", ("b",), {"name": "b"})
    beta = forest.add_node("KindA", ("a",), {"name": "a"})
    forest.add_alt("AltB", (alpha, beta))
    forest.add_alt("AltA", (beta,))

    visitor = _ForestOrderCaptureVisitor()
    traverse_forest_to_visitor(forest=forest, visitor=visitor)

    assert visitor.node_kinds == ["KindA", "KindB"]
    assert visitor.alt_kinds == ["AltA", "AltB"]


def test_replay_trace_and_equivalence_to_opportunity_visitor() -> None:
    emitter = OpportunityPayloadEmitter()
    adapt_live_event_stream_to_visitor(
        one_cells=[
            {
                "kind": "resume_load",
                "metadata": {"import_state_path": "state/a.json"},
            },
            {
                "kind": "resume_write",
                "metadata": {"state_path": "state/a.json"},
            },
        ],
        surface_representatives={
            "violation_summary": "rep:b",
            "groups_by_path": "rep:a",
        },
        two_cell_witnesses=[],
        cofibration_witnesses=[
            {
                "canonical_identity_kind": "suite_site",
                "cofibration": {
                    "entries": [
                        {
                            "domain": {"key": "domain:x", "prime": 2},
                            "aspf": {"key": "aspf:x", "prime": 2},
                        }
                    ]
                },
            }
        ],
        visitor=emitter,
    )
    adapt_event_log_reader_iterator_to_visitor(
        event_log_rows=[
            {
                "surface": "groups_by_path",
                "classification": "non_drift",
                "witness_id": "w:1",
            }
        ],
        visitor=emitter,
    )

    rows = emitter.build_rows()
    kinds = [str(row.get("kind")) for row in rows]
    assert "materialize_load_fusion" in kinds
    assert "reusable_boundary_artifact" in kinds
    assert "fungible_execution_path_substitution" in kinds
    assert "cofibration_prime_embedding_reuse" in kinds
    fungible = next(
        row for row in rows if isinstance(row, dict) and row.get("kind") == "fungible_execution_path_substitution"
    )
    assert fungible["actionability"] == "actionable"
    assert fungible["confidence_provenance"] == "morphism_witness"

    cofibration = next(
        row for row in rows if isinstance(row, dict) and row.get("kind") == "cofibration_prime_embedding_reuse"
    )
    assert cofibration["witness_requirement"] == "cofibration_witness"

    plans = emitter.build_rewrite_plans()
    assert plans
    assert plans[0]["opportunity_id"].startswith("opp:")


def test_trace_event_iterator_adapter_dispatches_without_json_batching() -> None:
    emitter = OpportunityPayloadEmitter()
    adapt_trace_event_iterator_to_visitor(
        events=[
            AspfOneCellEvent(
                index=0,
                payload={
                    "kind": "resume_load",
                    "metadata": {"import_state_path": "state/a.json"},
                },
            ),
            AspfOneCellEvent(
                index=1,
                payload={
                    "kind": "resume_write",
                    "metadata": {"state_path": "state/a.json"},
                },
            ),
            AspfTwoCellEvent(
                index=0,
                payload={
                    "witness_id": "w:1",
                    "left_representative": "rep:a",
                    "right_representative": "rep:b",
                },
            ),
            AspfCofibrationEvent(index=0, payload={"canonical_identity_kind": "k"}),
            AspfSurfaceUpdateEvent(surface="groups_by_path", representative="rep:a"),
        ],
        visitor=emitter,
    )
    adapt_event_log_reader_iterator_to_visitor(
        event_log_rows=[
            {
                "surface": "groups_by_path",
                "classification": "non_drift",
                "witness_id": "w:1",
            }
        ],
        visitor=emitter,
    )

    kinds = {str(row.get("kind")) for row in emitter.build_rows() if isinstance(row, dict)}
    assert "materialize_load_fusion" in kinds
    assert "fungible_execution_path_substitution" in kinds


def test_null_visitor_noop_methods_are_callable() -> None:
    visitor = NullAspfTraversalVisitor()
    forest = Forest()
    node = forest.add_node("Kind", ("n",), {"name": "n"})
    alt = forest.add_alt("Alt", (node,))

    visitor.on_forest_node(node=node)
    visitor.on_forest_alt(alt=alt)
    visitor.on_trace_one_cell(index=0, one_cell={})
    visitor.on_trace_two_cell_witness(index=0, witness={})
    visitor.on_trace_cofibration(index=0, cofibration={})
    visitor.on_trace_surface_representative(surface="groups_by_path", representative="rep")
    visitor.on_equivalence_surface_row(index=0, row={})


def test_two_cell_witnesses_drive_deterministic_rewrite_plan_priority() -> None:
    emitter = OpportunityPayloadEmitter()
    adapt_live_event_stream_to_visitor(
        one_cells=[
            {
                "kind": "semantic_surface",
                "representative": "rep:shared",
            }
        ],
        surface_representatives={
            "groups_by_path": "rep:shared",
            "rewrite_plans": "rep:shared",
        },
        two_cell_witnesses=[
            {
                "witness_id": "w:2",
                "left_representative": "rep:shared",
                "right_representative": "rep:baseline",
            },
            {
                "witness_id": "w:1",
                "left_representative": "rep:shared",
                "right_representative": "rep:legacy",
            },
        ],
        cofibration_witnesses=[
            {
                "cofibration": {
                    "entries": [
                        {
                            "domain": {"key": "d:int", "prime": 2},
                            "aspf": {"key": "a:int", "prime": 2},
                        }
                    ]
                }
            }
        ],
        visitor=emitter,
    )

    plans = emitter.build_rewrite_plans()
    reusable = next(
        plan
        for plan in plans
        if plan["opportunity_id"].startswith("opp:reusable-boundary:")
    )
    assert reusable["required_witnesses"] == ["w:1", "w:2"]
    assert reusable["priority"] == 1.0
    assert reusable["canonical_identity"]["node_id"]["kind"] == "Opportunity:ReusableBoundaryRepresentative"
    assert reusable["opportunity_hash"]


def test_reusable_boundary_collision_vs_witnessed_isomorphy_golden() -> None:
    collision_only = OpportunityPayloadEmitter()
    adapt_live_event_stream_to_visitor(
        one_cells=[],
        surface_representatives={
            "groups_by_path": "rep:shared",
            "violation_summary": "rep:shared",
        },
        two_cell_witnesses=[],
        cofibration_witnesses=[],
        visitor=collision_only,
    )
    collision_row = next(
        row
        for row in collision_only.build_rows()
        if row.get("kind") == "reusable_boundary_artifact"
    )

    witnessed = OpportunityPayloadEmitter()
    adapt_live_event_stream_to_visitor(
        one_cells=[{"kind": "semantic_surface", "representative": "rep:shared"}],
        surface_representatives={
            "groups_by_path": "rep:shared",
            "violation_summary": "rep:shared",
        },
        two_cell_witnesses=[
            {
                "witness_id": "w:iso",
                "left_representative": "rep:shared",
                "right_representative": "rep:baseline",
            }
        ],
        cofibration_witnesses=[{"cofibration": {"entries": [{"domain": {}, "aspf": {}}]}}],
        visitor=witnessed,
    )
    witnessed_row = next(
        row for row in witnessed.build_rows() if row.get("kind") == "reusable_boundary_artifact"
    )

    assert collision_row["actionability"] == "observational"
    assert collision_row["failed_obligations"] == [
        "one_cell_carrier_presence",
        "two_cell_isomorphy_witness",
        "cofibration_support",
    ]
    assert witnessed_row["actionability"] == "actionable"
    assert witnessed_row["failed_obligations"] == []


def test_replay_event_dispatch_rejects_unknown_event_type() -> None:
    visitor = NullAspfTraversalVisitor()
    with pytest.raises(NeverRaise):
        visitor.on_replay_event(event=object())  # type: ignore[arg-type]


def test_two_cell_replay_normalization_uses_nested_representatives_and_skip_outcome() -> None:
    normalized = _normalize_two_cell_witness_for_replay(
        {
            "witness_id": "w:nested",
            "left": {"representative": "rep:left"},
            "right": {"representative": "rep:right"},
        }
    )
    assert normalized.kind is TwoCellReplayNormalizationKind.VALID
    assert normalized.payload["left_representative"] == "rep:left"
    assert normalized.payload["right_representative"] == "rep:right"

    skipped = _normalize_two_cell_witness_for_replay({"witness_id": "w:missing"})
    assert skipped.kind is TwoCellReplayNormalizationKind.SKIP


def test_adapt_live_event_stream_skips_invalid_two_cell_payloads() -> None:
    emitter = OpportunityPayloadEmitter()
    adapt_live_event_stream_to_visitor(
        one_cells=[],
        two_cell_witnesses=[{"witness_id": "w:invalid"}],
        cofibration_witnesses=[],
        surface_representatives={},
        visitor=emitter,
    )
    assert emitter.build_rows() == []


def test_opportunity_predicate_requires_resume_kinds_and_confidence_default() -> None:
    observation = OpportunityAlgebraicObservation(
        structure=OpportunityStructure.ONE_CELL,
        subject_id="subject",
        one_cell_count=1,
        one_cell_kinds=frozenset({"resume_write"}),
    )
    requires_load = OpportunityAlgebraicPredicate(
        structure=OpportunityStructure.ONE_CELL,
        min_one_cells=1,
        requires_resume_load=True,
    )
    requires_write = OpportunityAlgebraicPredicate(
        structure=OpportunityStructure.ONE_CELL,
        min_one_cells=1,
        requires_resume_write=True,
    )
    assert requires_load.matches(observation=observation) is False
    assert requires_write.matches(observation=observation) is True
    assert requires_write.matches(
        observation=OpportunityAlgebraicObservation(
            structure=OpportunityStructure.ONE_CELL,
            subject_id="subject-no-write",
            one_cell_count=1,
            one_cell_kinds=frozenset({"resume_load"}),
        )
    ) is False

    decision = OpportunityDecisionProtocol(
        opportunity_id="opp:test",
        kind="test",
        canonical_identity={},
        affected_surfaces=(),
        witness_ids=(),
        reason="test",
        confidence_provenance=OpportunityConfidenceProvenance.INGRESS_OBSERVATION,
        witness_requirement=OpportunityWitnessRequirement.NONE,
        actionability=OpportunityActionabilityState.OBSERVATIONAL,
    )
    assert decision.confidence() == 0.0


def test_run_boundary_event_path_can_be_replayed() -> None:
    @dataclass
    class _BoundaryCaptureVisitor(NullAspfTraversalVisitor):
        rows: list[dict[str, object]] = field(default_factory=list)

        def on_equivalence_surface_row(
            self,
            *,
            index: int,
            row: dict[str, object],
        ) -> None:
            self.rows.append(row)

    visitor = _BoundaryCaptureVisitor()
    adapt_trace_event_iterator_to_visitor(
        events=[AspfRunBoundaryEvent(boundary="equivalence_surface_row", payload={})],
        visitor=visitor,
    )
    assert visitor.rows == [{}]
