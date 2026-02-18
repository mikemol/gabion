from __future__ import annotations

from gabion.server import _normalize_dataflow_response


def test_server_projection_normalizes_rewrite_plan_order_and_schema_errors() -> None:
    response = {
        "exit_code": 0,
        "fingerprint_rewrite_plans": [
            {
                "plan_id": "rewrite:z.py:f:a:glossary-ambiguity:surface-canonicalize",
                "site": {"path": "z.py", "function": "f", "bundle": ["a"]},
                "rewrite": {"kind": "SURFACE_CANONICALIZE", "parameters": {"candidates": ["ctx"]}},
                "verification": {"predicates": [{"kind": "base_conservation"}]},
                "evidence": {"provenance_id": "p", "coherence_id": "c"},
            },
            {
                "plan_id": "rewrite:a.py:f:a:glossary-ambiguity:bundle-align",
                "site": {"path": "a.py", "function": "f", "bundle": ["a"]},
                "rewrite": {"kind": "BUNDLE_ALIGN", "parameters": {"candidates": ["ctx"]}},
                "verification": {
                    "predicates": [
                        {"kind": "base_conservation"},
                        {"kind": "ctor_coherence"},
                        {"kind": "match_strata"},
                        {"kind": "remainder_non_regression"},
                    ]
                },
                "evidence": {"provenance_id": "p", "coherence_id": "c"},
            },
        ],
    }
    normalized = _normalize_dataflow_response(response)
    plans = normalized["fingerprint_rewrite_plans"]
    assert plans[0]["site"]["path"] == "a.py"
    assert "rewrite_plan_schema_errors" in normalized
    assert normalized["rewrite_plan_schema_errors"][0]["plan_id"].startswith("rewrite:z.py")
