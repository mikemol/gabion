from __future__ import annotations

import json

from gabion.analysis.dataflow_decision_surfaces import (
    compute_fingerprint_coherence,
    compute_fingerprint_rewrite_plans,
    extract_smell_sample,
    lint_lines_from_bundle_evidence,
    lint_lines_from_constant_smells,
    lint_lines_from_type_evidence,
    lint_lines_from_unused_arg_smells,
    parse_lint_location,
    summarize_coherence_witnesses,
    summarize_deadness_witnesses,
    summarize_rewrite_plans,
)
from gabion.analysis.evidence import Site
from gabion.order_contract import ordered_or_sorted


def _check_deadline() -> None:
    return None


def _lint_line(path: str, line: int, col: int, code: str, message: str) -> str:
    return f"{path}:{line}:{col}: {code} {message}".strip()


# gabion:evidence E:call_footprint::tests/test_dataflow_decision_surfaces_module.py::test_decision_surface_summaries_and_plans_cover_edges::dataflow_decision_surfaces.py::gabion.analysis.dataflow_decision_surfaces.compute_fingerprint_coherence::dataflow_decision_surfaces.py::gabion.analysis.dataflow_decision_surfaces.compute_fingerprint_rewrite_plans::dataflow_decision_surfaces.py::gabion.analysis.dataflow_decision_surfaces.summarize_coherence_witnesses::dataflow_decision_surfaces.py::gabion.analysis.dataflow_decision_surfaces.summarize_deadness_witnesses::dataflow_decision_surfaces.py::gabion.analysis.dataflow_decision_surfaces.summarize_rewrite_plans
def test_decision_surface_summaries_and_plans_cover_edges() -> None:
    assert summarize_deadness_witnesses([], check_deadline=_check_deadline) == []
    lines = summarize_deadness_witnesses(
        [
            {
                "path": "a.py",
                "function": "f",
                "bundle": ["a"],
                "predicate": "P",
                "environment": {},
                "result": "UNKNOWN",
                "core": [],
            }
            for _ in range(12)
        ],
        max_entries=10,
        check_deadline=_check_deadline,
    )
    assert any("... 2 more" in line for line in lines)

    coherence = compute_fingerprint_coherence(
        [
            {
                "path": "a.py",
                "function": "f",
                "bundle": ["a", "b"],
                "provenance_id": "prov:a.py:f:a,b",
                "base_keys": ["int", "str"],
                "ctor_keys": [],
                "glossary_matches": ["ctx_a", "ctx_b"],
            }
        ],
        synth_version="synth@1",
        check_deadline=_check_deadline,
        ordered_or_sorted=ordered_or_sorted,
    )
    assert coherence
    assert summarize_coherence_witnesses([], check_deadline=_check_deadline) == []

    plans = compute_fingerprint_rewrite_plans(
        [
            {
                "path": "a.py",
                "function": "f",
                "bundle": ["a", "b"],
                "provenance_id": "prov:a.py:f:a,b",
                "base_keys": ["int", "str"],
                "ctor_keys": [],
                "glossary_matches": ["ctx_a", "ctx_b"],
            }
        ],
        coherence,
        synth_version="synth@1",
        exception_obligations=[
            {"site": "not-a-dict"},
            {"site": {"path": "", "function": "f", "bundle": ["a", "b"]}},
            {
                "site": {"path": "a.py", "function": "f", "bundle": ["a", "b"]},
                "status": "WEIRD",
            },
        ],
        check_deadline=_check_deadline,
        ordered_or_sorted=ordered_or_sorted,
        site_from_payload=Site.from_payload,
    )
    assert plans
    assert plans[0]["pre"]["exception_obligations_summary"]["UNKNOWN"] == 1

    assert summarize_rewrite_plans([], check_deadline=_check_deadline) == []
    rewrite_summary = summarize_rewrite_plans(
        [
            {
                "plan_id": f"plan:{i}",
                "site": {"path": "a.py", "function": "f", "bundle": ["a"]},
                "rewrite": {"kind": "BUNDLE_ALIGN"},
                "status": "UNVERIFIED",
            }
            for i in range(12)
        ],
        max_entries=10,
        check_deadline=_check_deadline,
    )
    assert any("... 2 more" in line for line in rewrite_summary)


# gabion:evidence E:call_footprint::tests/test_dataflow_decision_surfaces_module.py::test_decision_surface_lint_parsing_helpers_cover_edges::dataflow_decision_surfaces.py::gabion.analysis.dataflow_decision_surfaces.extract_smell_sample::dataflow_decision_surfaces.py::gabion.analysis.dataflow_decision_surfaces.lint_lines_from_bundle_evidence::dataflow_decision_surfaces.py::gabion.analysis.dataflow_decision_surfaces.lint_lines_from_constant_smells::dataflow_decision_surfaces.py::gabion.analysis.dataflow_decision_surfaces.lint_lines_from_type_evidence::dataflow_decision_surfaces.py::gabion.analysis.dataflow_decision_surfaces.lint_lines_from_unused_arg_smells::dataflow_decision_surfaces.py::gabion.analysis.dataflow_decision_surfaces.parse_lint_location
def test_decision_surface_lint_parsing_helpers_cover_edges() -> None:
    assert parse_lint_location("bad") is None
    assert parse_lint_location("p:1:2:-3:4: msg") == ("p", 1, 2, "msg")
    assert extract_smell_sample("no example") is None
    assert extract_smell_sample("smell (e.g. p:1:2: msg)") == "p:1:2: msg"

    assert lint_lines_from_bundle_evidence(["bad"], check_deadline=_check_deadline, lint_line=_lint_line) == []
    assert lint_lines_from_type_evidence(["bad"], check_deadline=_check_deadline, lint_line=_lint_line) == []
    assert lint_lines_from_unused_arg_smells(["bad"], check_deadline=_check_deadline, lint_line=_lint_line) == []
    assert lint_lines_from_constant_smells(["no location"], check_deadline=_check_deadline, lint_line=_lint_line) == []
    constant = "mod.py:f.a only observed constant 1 (e.g. mod.py:1:2: msg)"
    assert lint_lines_from_constant_smells([constant], check_deadline=_check_deadline, lint_line=_lint_line)


# gabion:evidence E:call_footprint::tests/test_dataflow_decision_surfaces_module.py::test_rewrite_plan_roundtrip_and_deterministic_ordering::dataflow_decision_surfaces.py::gabion.analysis.dataflow_decision_surfaces.compute_fingerprint_coherence::dataflow_decision_surfaces.py::gabion.analysis.dataflow_decision_surfaces.compute_fingerprint_rewrite_plans
def test_rewrite_plan_roundtrip_and_deterministic_ordering() -> None:
    coherence = compute_fingerprint_coherence(
        [
            {
                "path": "z.py",
                "function": "f",
                "bundle": ["a"],
                "provenance_id": "prov:z.py:f:a",
                "base_keys": ["int"],
                "ctor_keys": ["CtorZ"],
                "remainder": {"base": 1, "ctor": 1},
                "glossary_matches": ["ctx_a", "ctx_b"],
            },
            {
                "path": "a.py",
                "function": "f",
                "bundle": ["a"],
                "provenance_id": "prov:a.py:f:a",
                "base_keys": ["int"],
                "ctor_keys": ["CtorA"],
                "remainder": {"base": 1, "ctor": 1},
                "glossary_matches": ["ctx_a", "ctx_b"],
            },
        ],
        synth_version="synth@1",
        check_deadline=_check_deadline,
        ordered_or_sorted=ordered_or_sorted,
    )
    plans = compute_fingerprint_rewrite_plans(
        [
            {
                "path": "z.py",
                "function": "f",
                "bundle": ["a"],
                "provenance_id": "prov:z.py:f:a",
                "base_keys": ["int"],
                "ctor_keys": ["CtorZ"],
                "remainder": {"base": 1, "ctor": 1},
                "glossary_matches": ["ctx_a", "ctx_b"],
            },
            {
                "path": "a.py",
                "function": "f",
                "bundle": ["a"],
                "provenance_id": "prov:a.py:f:a",
                "base_keys": ["int"],
                "ctor_keys": ["CtorA"],
                "remainder": {"base": 1, "ctor": 1},
                "glossary_matches": ["ctx_a", "ctx_b"],
            },
        ],
        coherence,
        synth_version="synth@1",
        exception_obligations=None,
        check_deadline=_check_deadline,
        ordered_or_sorted=ordered_or_sorted,
        site_from_payload=Site.from_payload,
    )

    serialized = json.dumps(plans, sort_keys=True)
    reloaded = json.loads(serialized)
    assert reloaded == plans

    ordered_ids = [plan["plan_id"] for plan in plans]
    assert ordered_ids == sorted(ordered_ids) or ordered_ids[0].startswith("rewrite:a.py")
    assert all("payload_schema" in plan for plan in plans)


# gabion:evidence E:call_footprint::tests/test_dataflow_decision_surfaces_module.py::test_rewrite_plan_helpers_cover_skip_and_abstain_paths::dataflow_decision_surfaces.py::gabion.analysis.dataflow_decision_surfaces.compute_fingerprint_rewrite_plans::dataflow_decision_surfaces.py::gabion.analysis.dataflow_decision_surfaces.summarize_coherence_witnesses
def test_rewrite_plan_helpers_cover_skip_and_abstain_paths() -> None:
    coherence_summary = summarize_coherence_witnesses(
        [
            {
                "site": {"path": "a.py", "function": "f", "bundle": ["a"]},
                "result": "UNKNOWN",
                "fork_signature": "k",
                "alternatives": [],
            }
            for _ in range(12)
        ],
        max_entries=10,
        check_deadline=_check_deadline,
    )
    assert any("... 2 more" in line for line in coherence_summary)

    def _ordered(values, **kwargs):
        source = str(kwargs.get("source", ""))
        if source.endswith(".candidates"):
            return []
        return ordered_or_sorted(values, **kwargs)

    plans = compute_fingerprint_rewrite_plans(
        provenance=[
            {
                "path": "",
                "function": "f",
                "bundle": ["a"],
                "provenance_id": "prov:skip",
                "base_keys": ["int"],
                "ctor_keys": [],
                "glossary_matches": ["x", "y"],
            },
            {
                "path": "a.py",
                "function": "f",
                "bundle": ["a"],
                "provenance_id": "prov:a.py:f:a",
                "base_keys": ["int"],
                "ctor_keys": [],
                "glossary_matches": ["ctx_a", "ctx_b"],
            },
        ],
        coherence=[{"site": "not-a-mapping"}],
        synth_version="synth@1",
        exception_obligations=None,
        check_deadline=_check_deadline,
        ordered_or_sorted=_ordered,
        site_from_payload=Site.from_payload,
    )
    assert plans
    assert any(
        str(plan.get("rewrite", {}).get("kind", "")) == "SURFACE_CANONICALIZE"
        and str(plan.get("status", "")) == "ABSTAINED"
        for plan in plans
    )
    assert any(
        str(plan.get("rewrite", {}).get("kind", "")) == "AMBIENT_REWRITE"
        and str(plan.get("status", "")) == "ABSTAINED"
        for plan in plans
    )
