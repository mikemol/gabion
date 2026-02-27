from __future__ import annotations

from dataclasses import dataclass, field

from gabion.analysis.aspf import Alt, Forest, Node
from gabion.analysis.aspf_visitors import (
    NullAspfTraversalVisitor,
    OpportunityPayloadEmitter,
    adapt_event_log_reader_iterator_to_visitor,
    adapt_live_event_stream_to_visitor,
    traverse_forest_to_visitor,
)


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
        one_cells=[],
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
        cofibration_witnesses=[],
        visitor=emitter,
    )

    plans = emitter.build_rewrite_plans()
    reusable = next(
        plan
        for plan in plans
        if plan["opportunity_id"].startswith(
            "opp:reusable-boundary:Opportunity:ReusableBoundaryRepresentative:"
        )
    )
    assert reusable["required_witnesses"] == ["w:1", "w:2"]
    assert reusable["priority"] == 0.74
    assert reusable["canonical_identity"]["node_id"]["kind"] == "Opportunity:ReusableBoundaryRepresentative"
    assert reusable["opportunity_hash"]
