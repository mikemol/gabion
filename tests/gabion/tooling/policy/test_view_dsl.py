from __future__ import annotations

from gabion.tooling.policy_substrate.view_dsl import (
    AddIntExpr,
    CoalesceExpr,
    PathExpr,
    WeightedSumExpr,
    WeightedTerm,
    collection_items,
    eval_int,
    eval_mapping,
    eval_text,
)


# gabion:behavior primary=desired
def test_view_dsl_supports_path_collection_and_coalesce() -> None:
    payload = {
        "workstreams": [
            {
                "object_id": "CSA-IVL",
                "next_actions": {
                    "recommended_followup": {
                        "object_id": "CSA-IVL-TP-002",
                    }
                },
            }
        ]
    }

    workstreams = collection_items(payload, "workstreams")

    assert len(workstreams) == 1
    assert eval_text(PathExpr("object_id"), workstreams[0]) == "CSA-IVL"
    assert eval_mapping(
        CoalesceExpr(
            items=(
                PathExpr("recommended_followup"),
                PathExpr("next_actions.recommended_followup"),
            )
        ),
        workstreams[0],
    ) == {"object_id": "CSA-IVL-TP-002"}


# gabion:behavior primary=desired
def test_view_dsl_supports_add_and_weighted_sums() -> None:
    payload = {
        "surviving_touchsite_count": 4,
        "policy_signal_count": 1,
        "diagnostic_count": 2,
        "doc_alignment_summary": {
            "missing_target_doc_count": 1,
            "unassigned_target_doc_count": 2,
        },
    }

    doc_alignment_expr = AddIntExpr(
        items=(
            PathExpr("doc_alignment_summary.missing_target_doc_count"),
            PathExpr("doc_alignment_summary.unassigned_target_doc_count"),
        )
    )
    pressure_expr = WeightedSumExpr(
        items=(
            WeightedTerm(weight=3, expr=PathExpr("surviving_touchsite_count")),
            WeightedTerm(weight=5, expr=PathExpr("policy_signal_count")),
            WeightedTerm(weight=8, expr=PathExpr("diagnostic_count")),
            WeightedTerm(weight=2, expr=doc_alignment_expr),
        )
    )

    assert eval_int(doc_alignment_expr, payload) == 3
    assert eval_int(pressure_expr, payload) == 39
