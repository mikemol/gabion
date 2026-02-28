from __future__ import annotations

from gabion.refactor import rewrite_plan as rewrite_plan_mod
from gabion.server import _normalize_dataflow_response


# gabion:evidence E:call_footprint::tests/test_server_rewrite_plan_projection.py::test_server_projection_normalizes_rewrite_plan_order_and_schema_errors::server.py::gabion.server._normalize_dataflow_response
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


# gabion:evidence E:function_site::rewrite_plan.py::gabion.refactor.rewrite_plan.attach_plan_schema
def test_rewrite_plan_schema_helpers_cover_non_dict_and_unknown_kind() -> None:
    assert rewrite_plan_mod._missing_keys("nope", ("a", "b")) == ["a", "b"]
    assert rewrite_plan_mod.attach_plan_schema({"rewrite": "bad"}) == {"rewrite": "bad"}
    unknown = {"rewrite": {"kind": "NOT_A_KIND"}}
    assert rewrite_plan_mod.attach_plan_schema(unknown) == unknown


# gabion:evidence E:function_site::tests/test_server_rewrite_plan_projection.py::test_server_projection_normalizes_lint_entry_trichotomy
def test_server_projection_normalizes_lint_entry_trichotomy() -> None:
    provided = _normalize_dataflow_response(
        {
            "lint_lines": ["x.py:1:1: X ignored"],
            "lint_entries": [{"path": "a.py", "line": 1, "col": 2, "code": "E", "message": "provided"}],
        }
    )
    assert provided["lint_entries"][0]["path"] == "a.py"

    derived = _normalize_dataflow_response({"lint_lines": ["b.py:2:3: W derived message"]})
    assert derived["lint_entries"][0]["path"] == "b.py"

    empty = _normalize_dataflow_response({})
    assert empty["lint_entries"] == []
