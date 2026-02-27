from __future__ import annotations

from dataclasses import dataclass, field

from gabion.analysis.aspf import Alt, Forest, Node
from gabion.analysis.aspf_visitors import (
    NullAspfTraversalVisitor,
    OpportunityPayloadEmitter,
    replay_equivalence_payload_to_visitor,
    replay_trace_payload_to_visitor,
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
    replay_trace_payload_to_visitor(
        trace_payload={
            "one_cells": [
                {
                    "kind": "resume_load",
                    "metadata": {"import_state_path": "state/a.json"},
                },
                {
                    "kind": "resume_write",
                    "metadata": {"state_path": "state/a.json"},
                },
            ],
            "surface_representatives": {
                "violation_summary": "rep:b",
                "groups_by_path": "rep:a",
            },
            "two_cell_witnesses": [],
            "cofibration_witnesses": [],
        },
        visitor=emitter,
    )
    replay_equivalence_payload_to_visitor(
        equivalence_payload={
            "surface_table": [
                {
                    "surface": "groups_by_path",
                    "classification": "non_drift",
                    "witness_id": "w:1",
                }
            ]
        },
        visitor=emitter,
    )

    rows = emitter.build_rows()
    kinds = [str(row.get("kind")) for row in rows]
    assert "materialize_load_fusion" in kinds
    assert "reusable_boundary_artifact" in kinds
    assert "fungible_execution_path_substitution" in kinds
