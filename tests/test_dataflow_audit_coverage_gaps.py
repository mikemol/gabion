from __future__ import annotations

import argparse
import ast
import os
import pytest
from pathlib import Path

from gabion.analysis import dataflow_audit as da
from gabion.exceptions import NeverThrown


def _load():
    return da


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_coherence_witnesses::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_deadness_witnesses::entries,max_entries
def test_deadness_and_coherence_summaries_cover_edges() -> None:
    assert da._summarize_deadness_witnesses([]) == []
    deadness_entries = [
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
    ]
    lines = da._summarize_deadness_witnesses(deadness_entries, max_entries=10)
    assert any("... 2 more" in line for line in lines)

    assert da._summarize_coherence_witnesses([]) == []
    coherence_entries = [
        {
            "site": {"path": "a.py", "function": "f", "bundle": ["a"]},
            "result": "UNKNOWN",
            "fork_signature": "glossary-ambiguity",
            "alternatives": ["x", "y"],
        }
        for _ in range(12)
    ]
    lines = da._summarize_coherence_witnesses(coherence_entries, max_entries=10)
    assert any("... 2 more" in line for line in lines)


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_rewrite_plans::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_rewrite_plans::exception_obligations E:decision_surface/direct::evidence.py::gabion.analysis.evidence.Site.from_payload::payload
def test_fingerprint_coherence_and_rewrite_plans_cover_edges() -> None:
    provenance_entries = [
        {
            "path": "a.py",
            "function": "f",
            "bundle": ["a", "b"],
            "provenance_id": "prov:a.py:f:a,b",
            "base_keys": ["int", "str"],
            "ctor_keys": [],
            "glossary_matches": ["ctx_a", "ctx_b"],
        }
    ]
    coherence = da._compute_fingerprint_coherence(provenance_entries, synth_version="synth@1")
    assert coherence

    coherence_with_garbage = [{"site": "nope"}] + coherence
    bad_provenance = [dict(provenance_entries[0], path="")] + provenance_entries

    plans = da._compute_fingerprint_rewrite_plans(
        bad_provenance, coherence_with_garbage, synth_version="synth@1"
    )
    assert plans
    assert plans[0]["evidence"]["coherence_id"] == coherence[0]["coherence_id"]

    plans = da._compute_fingerprint_rewrite_plans(
        provenance_entries,
        coherence,
        synth_version="synth@1",
        exception_obligations=[
            {"site": "not-a-dict"},
            {"site": {"path": "", "function": "f", "bundle": ["a", "b"]}},
            {"site": {"path": "a.py", "function": "f", "bundle": ["a", "b"]}, "status": "WEIRD"},
        ],
    )
    assert plans
    assert plans[0]["pre"]["exception_obligations_summary"]["UNKNOWN"] == 1

    assert da._summarize_rewrite_plans([]) == []
    rewrite_plans = [
        {
            "plan_id": f"plan:{i}",
            "site": {"path": "a.py", "function": "f", "bundle": ["a"]},
            "rewrite": {"kind": "BUNDLE_ALIGN"},
            "status": "UNVERIFIED",
        }
        for i in range(12)
    ]
    lines = da._summarize_rewrite_plans(rewrite_plans, max_entries=10)
    assert any("... 2 more" in line for line in lines)


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._exception_param_names::expr,params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._handler_is_broad::handler E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._handler_label::handler E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._node_in_try_body::node
def test_exception_helpers_cover_edges() -> None:
    # _enclosing_function_node returns None when no enclosing function is present.
    node = ast.parse("x = 1").body[0]
    assert da._enclosing_function_node(node, parents={}) is None

    assert da._exception_param_names(None, {"a"}) == []
    expr = ast.parse("a + b").body[0].value
    assert da._exception_param_names(expr, {"a"}) == ["a"]

    handler_any = ast.ExceptHandler(type=None, name=None, body=[])
    assert da._handler_is_broad(handler_any) is True
    assert da._handler_label(handler_any) == "except:"

    handler_attr = ast.ExceptHandler(
        type=ast.Attribute(
            value=ast.Name(id="builtins", ctx=ast.Load()),
            attr="Exception",
            ctx=ast.Load(),
        ),
        name=None,
        body=[],
    )
    assert da._handler_is_broad(handler_attr) is True

    handler_weird = ast.ExceptHandler(type=object(), name=None, body=[])
    assert da._handler_is_broad(handler_weird) is False
    assert da._handler_label(handler_weird) == "except <unknown>"

    # _node_in_try_body should find a nested node inside a try block.
    tree = ast.parse(
        "try:\n"
        "    foo(1)\n"
        "except Exception:\n"
        "    pass\n"
    )
    try_node = tree.body[0]
    assert isinstance(try_node, ast.Try)
    call_node = try_node.body[0].value
    assert da._node_in_try_body(call_node, try_node) is True
    other_call = ast.parse("bar()").body[0].value
    assert da._node_in_try_body(other_call, try_node) is False


# gabion:evidence E:decision_surface/direct::evidence.py::gabion.analysis.evidence.exception_obligation_summary_for_site::site
def test_exception_obligation_summary_helper_covers_filters_and_status_normalization() -> None:
    site = da.Site(path="a.py", function="f", bundle=("a",))
    summary = da._exception_obligation_summary_for_site(
        [
            {"site": "nope"},
            {"site": {"path": "b.py", "function": "f", "bundle": ["a"]}, "status": "UNKNOWN"},
            {"site": {"path": "a.py", "function": "f", "bundle": ["x"]}, "status": "UNKNOWN"},
            {"site": {"path": "a.py", "function": "f", "bundle": ["a"]}, "status": "WEIRD"},
            {"site": {"path": "a.py", "function": "f", "bundle": ["a"]}, "status": "DEAD"},
            {"site": {"path": "a.py", "function": "f", "bundle": ["a"]}, "status": "HANDLED"},
        ],
        site=site,
    )
    assert summary == {"UNKNOWN": 1, "DEAD": 1, "HANDLED": 1, "total": 3}


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._decorator_matches::allowlist,name E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._dead_env_map::deadness_witnesses E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._is_never_marker_raise::exception_name,never_exceptions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._exception_type_name::expr E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._exception_param_names::expr,params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_names::fn,ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_exception_obligations::handledness_witnesses E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._handler_is_broad::handler E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._handler_label::handler E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._find_handling_try::node E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_snapshot_path::root
def test_exception_collection_and_summaries_cover_edges(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("def bad(:\n    pass\n")

    narrow = tmp_path / "narrow.py"
    narrow.write_text(
        "def f(a):\n"
        "    try:\n"
        "        raise ValueError(a)\n"
        "    except ValueError:\n"
        "        return a\n"
    )

    module_level = tmp_path / "module_level.py"
    module_level.write_text(
        "import builtins\n"
        "try:\n"
        "    raise RuntimeError('boom')\n"
        "except builtins.Exception:\n"
        "    pass\n"
    )

    handledness = da._collect_handledness_witnesses(
        [bad, narrow, module_level],
        project_root=tmp_path,
        ignore_params=set(),
    )
    assert handledness

    obligations = da._collect_exception_obligations(
        [bad, module_level],
        project_root=tmp_path,
        ignore_params=set(),
        handledness_witnesses=handledness,
    )
    assert obligations


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._decorator_matches::allowlist,name E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._never_reason::call E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._never_sort_key::entry E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._is_never_marker_raise::exception_name,never_exceptions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._decorator_name::node
def test_never_helpers_and_sort_key_cover_edges() -> None:
    call = ast.parse("never('boom')").body[0].value
    assert da._is_never_call(call) is True
    assert da._never_reason(call) == "boom"
    kw_call = ast.parse("never(reason='nope')").body[0].value
    assert da._never_reason(kw_call) == "nope"
    anon_call = ast.parse("(lambda: None)()").body[0].value
    assert da._is_never_call(anon_call) is False
    other = ast.parse("other()").body[0].value
    assert da._is_never_call(other) is False
    assert da._is_never_marker_raise("never", "NeverRaise", {"NeverRaise"}) is True
    assert da._is_never_marker_raise("mod.never", "NeverRaise", {"NeverRaise"}) is True
    assert da._is_never_marker_raise("noop", "Other", {"NeverRaise"}) is False
    assert da._is_never_marker_raise("noop", "NeverRaise", {"NeverRaise"}) is False

    entry = {"status": "VIOLATION", "site": {"path": "p", "function": "f"}, "span": ["x", 0, 0, 0]}
    key = da._never_sort_key(entry)
    assert key[3] == -1


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._exception_type_name::expr E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._decorator_name::node
def test_exception_type_name_and_protocol_lint_edges() -> None:
    assert da._exception_type_name(None) is None
    assert da._exception_type_name(ast.parse("ValueError").body[0].value) == "ValueError"
    assert da._exception_type_name(ast.parse("ValueError()").body[0].value) == "ValueError"
    assert da._parse_exception_path_id("bad") is None
    assert da._parse_exception_path_id("p:f:k:1:x:raise") is None

    entries = [
        {"protocol": "other", "exception_path_id": "p:f:k:1:2:raise"},
        {"protocol": "never", "status": "DEAD", "exception_path_id": "p:f:k:1:2:raise"},
        {"protocol": "never", "status": "WEIRD", "exception_path_id": "bad"},
        {"protocol": "never", "status": "WEIRD", "exception_path_id": "p:f:k:1:2:raise", "exception_name": "Boom"},
    ]
    lines = da._exception_protocol_lint_lines(entries)
    assert any("GABION_EXC_NEVER" in line for line in lines)


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._exception_type_name::expr E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._exception_param_names::expr,params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_names::fn,ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._handler_is_broad::handler E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._handler_label::handler E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._find_handling_try::node E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_snapshot_path::root
def test_collect_handledness_system_exit(tmp_path: Path) -> None:
    path = tmp_path / "sys_exit.py"
    path.write_text("def f():\n    raise SystemExit()\n")
    witnesses = da._collect_handledness_witnesses(
        [path], project_root=tmp_path, ignore_params=set()
    )
    assert any(entry.get("handler_kind") == "convert" for entry in witnesses)


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_never_invariants::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._dead_env_map::deadness_witnesses E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._exception_param_names::expr,params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._never_reason::call E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._node_span::node E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_snapshot_path::root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_names::fn,ignore_params E:decision_surface/value_encoded::dataflow_audit.py::gabion.analysis.dataflow_audit._node_span::node
def test_collect_never_invariants_classifies(tmp_path: Path) -> None:
    da = _load()

    class FalseyDict(dict):
        def __bool__(self) -> bool:
            return False

    path = tmp_path / "never_mod.py"
    path.write_text(
        "from gabion.invariants import never\n"
        "\n"
        "def f(flag, other):\n"
        "    if flag and other:\n"
        "        never('blocked')\n"
        "\n"
        "def g(flag):\n"
        "    if flag:\n"
        "        never('boom')\n"
        "\n"
        "def h(flag, other):\n"
        "    if flag and other:\n"
        "        never()\n"
        "\n"
        "def k(zzz, aaa):\n"
        "    if zzz and aaa:\n"
        "        never('sorted')\n"
        "\n"
        "never('module')\n"
    )
    path_value = da._normalize_snapshot_path(path, tmp_path)
    deadness = [
        {
            "path": path_value,
            "function": "f",
            "bundle": ["flag"],
            "environment": FalseyDict({"flag": "False"}),
            "deadness_id": "dead:f",
        },
        {
            "path": path_value,
            "function": "g",
            "bundle": ["flag"],
            "environment": {"flag": "True"},
            "deadness_id": "dead:g",
        },
        {
            "path": path_value,
            "function": "h",
            "bundle": ["flag"],
            "environment": {"flag": "True"},
            "deadness_id": "dead:h",
        },
        {
            "path": path_value,
            "function": "k",
            "bundle": ["zzz"],
            "environment": {"zzz": "False"},
            "deadness_id": "dead:k",
        },
    ]
    invariants = da._collect_never_invariants(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        deadness_witnesses=deadness,
        forest=da.Forest(),
    )
    statuses = {entry.get("status") for entry in invariants}
    assert "PROVEN_UNREACHABLE" in statuses
    assert "VIOLATION" in statuses
    assert "OBLIGATION" in statuses
    assert any(entry.get("site", {}).get("function") == "<module>" for entry in invariants)
    assert any("depends on params" in entry.get("undecidable_reason", "") for entry in invariants)


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_never_invariants::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._dead_env_map::deadness_witnesses E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._exception_param_names::expr,params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._never_reason::call E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._node_span::node E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_snapshot_path::root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_names::fn,ignore_params E:decision_surface/value_encoded::dataflow_audit.py::gabion.analysis.dataflow_audit._node_span::node
def test_collect_never_invariants_skips_bad_syntax(tmp_path: Path) -> None:
    da = _load()
    bad = tmp_path / "bad.py"
    bad.write_text("def bad(:\n    pass\n")
    invariants = da._collect_never_invariants(
        [bad], project_root=tmp_path, ignore_params=set(), forest=da.Forest()
    )
    assert invariants == []


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_never_invariants::entries,include_proven_unreachable,max_entries E:decision_surface/direct::projection_exec.py::gabion.analysis.projection_exec.apply_spec::params_override E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_never_invariants._format_evidence::status
def test_never_invariant_lint_and_summary_formats() -> None:
    da = _load()
    assert da._summarize_never_invariants([]) == []
    entries = [
        {
            "status": "VIOLATION",
            "site": {"path": "a.py", "function": "a", "bundle": []},
            "span": ["x", 0, 0, 0],
            "reason": "bad",
            "witness_ref": "w0",
            "environment_ref": {"flag": False},
        },
        {
            "status": "VIOLATION",
            "site": {"path": "a.py", "function": "f", "bundle": []},
            "span": [0, 0, 0, 1],
            "reason": "boom",
            "witness_ref": "w1",
            "environment_ref": {"flag": True},
        },
        {
            "status": "OBLIGATION",
            "site": {"path": "b.py", "function": "g", "bundle": []},
            "span": [0, 1, 0, 2],
            "undecidable_reason": "depends on params: flag",
        },
        {
            "status": "PROVEN_UNREACHABLE",
            "site": {"path": "c.py", "function": "h", "bundle": []},
            "span": [0, 2, 0, 3],
            "witness_ref": "dead1",
        },
        {
            "status": "UNKNOWN",
            "site": {"path": "d.py", "function": "i", "bundle": []},
            "span": ["x"],
        },
    ]
    summary = da._summarize_never_invariants(entries, max_entries=1, include_proven_unreachable=False)
    assert any("VIOLATION" in line for line in summary)
    assert any("... 1 more" in line for line in summary)
    lint_entries = [entry for entry in entries if entry.get("span") != ["x", 0, 0, 0]]
    lint = da._never_invariant_lint_lines(lint_entries)
    assert any("GABION_NEVER_INVARIANT" in line for line in lint)
    assert any("witness=" in line for line in lint)
    assert any("why=depends on params" in line for line in lint)


# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit._extract_smell_sample E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit._lint_lines_from_bundle_evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit._lint_lines_from_constant_smells E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit._lint_lines_from_type_evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit._lint_lines_from_unused_arg_smells E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit._parse_lint_location
def test_parse_lint_location_and_smell_helpers() -> None:
    da = _load()
    assert da._parse_lint_location("bad") is None
    assert da._parse_lint_location("p:x:2: msg") is None
    parsed = da._parse_lint_location("p:1:2:-3:4: msg")
    assert parsed == ("p", 1, 2, "msg")
    assert da._extract_smell_sample("no example here") is None
    assert da._extract_smell_sample("smell (e.g. p:1:2: msg)") == "p:1:2: msg"

    assert da._lint_lines_from_bundle_evidence(["bad"]) == []
    assert da._lint_lines_from_type_evidence(["bad"]) == []
    assert da._lint_lines_from_unused_arg_smells(["bad"]) == []
    assert da._lint_lines_from_constant_smells(["no location here"]) == []
    constant = "mod.py:f.a only observed constant 1 across 1 non-test call(s) (e.g. mod.py:1:2: msg)"
    assert da._lint_lines_from_constant_smells([constant])


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_snapshot_path::root
def test_decision_param_lint_line_missing_span_and_transitive_callers() -> None:
    da = _load()
    info = da.FunctionInfo(
        name="f",
        qual="f",
        path=Path("mod.py"),
        params=["a"],
        annots={},
        calls=[],
        unused_params=set(),
        function_span=(0, 0, 0, 1),
    )
    assert (
        da._decision_param_lint_line(
            info, "a", project_root=None, code="CODE", message="msg"
        )
        is None
    )

    transitive = da._collect_transitive_callers(
        {"a": {"b"}, "b": {"a"}}, {"a": info, "b": info}
    )
    assert transitive["a"]


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._render_component_callsite_evidence::bundle_counts E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::bundle_sites_by_path,coherence_witnesses,constant_smells,context_suggestions,deadness_witnesses,decision_surfaces,decision_warnings,exception_obligations,fingerprint_matches,fingerprint_provenance,fingerprint_synth,fingerprint_warnings,forest,groups_by_path,handledness_witnesses,invariant_propositions,max_components,never_invariants,rewrite_plans,type_ambiguities,type_callsite_evidence,type_suggestions,unused_arg_smells,value_decision_rewrites,value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._render_mermaid_component::component,declared_global,nodes E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_never_invariants::entries,include_proven_unreachable,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_coherence_witnesses::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_deadness_witnesses::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_exception_obligations::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_handledness_witnesses::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_rewrite_plans::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_fingerprint_provenance::entries,max_examples E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._bundle_projection_from_forest::file_paths E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_dot::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_bundle_evidence_lines::forest,groups_by_path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._alt_input::kind
def test_bundle_projection_and_emitters(tmp_path: Path) -> None:
    da = _load()
    forest = da.Forest()
    site_a = forest.add_site("mod.py", "f")
    site_b = forest.add_site("mod.py", "g")
    paramset = forest.add_paramset(["x"])
    forest.add_alt("SignatureBundle", (site_a, paramset))
    forest.add_alt("SignatureBundle", (site_b, paramset))
    forest.add_alt("SignatureBundle", (site_a,))
    missing_site = da.NodeId(kind="FunctionSite", key=("missing.py", "h"))
    forest.add_alt("SignatureBundle", (missing_site, paramset))

    path = tmp_path / "mod.py"
    path.write_text("def f(x):\n    return x\n")
    groups_by_path = {path: {"f": [set(["x"])], "g": [set(["x"])]}}
    bundle_sites_by_path = {
        path: {
            "f": [
                [
                    {
                        "span": (0, 0, 0, 1),
                        "callee": "callee",
                        "params": ["x"],
                        "slots": ["x"],
                    }
                ]
            ]
        }
    }
    evidence = da._collect_bundle_evidence_lines(
        forest=forest,
        groups_by_path=groups_by_path,
        bundle_sites_by_path=bundle_sites_by_path,
    )
    assert evidence

    projection = da._bundle_projection_from_forest(
        forest, file_paths=[path]
    )
    assert projection.nodes
    assert da._alt_input(da.Alt(kind="X", inputs=(paramset,), evidence={}), "FunctionSite") is None
    assert da._paramset_key(da.Forest(), da.NodeId(kind="ParamSet", key=("a", "b"))) == ("a", "b")

    with pytest.raises(RuntimeError):
        da._emit_dot(None)

    report, _ = da._emit_report(
        {path: {"f": [set(["x"])]}},
        3,
        report=da.ReportCarrier(forest=da.Forest()),
    )
    assert report


# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit._forbid_adhoc_bundle_discovery
def test_forbid_adhoc_bundle_discovery_guard() -> None:
    da = _load()
    prev = os.environ.get("GABION_FORBID_ADHOC_BUNDLES")
    os.environ["GABION_FORBID_ADHOC_BUNDLES"] = "1"
    try:
        with pytest.raises(AssertionError):
            da._forbid_adhoc_bundle_discovery("test")
    finally:
        if prev is None:
            os.environ.pop("GABION_FORBID_ADHOC_BUNDLES", None)
        else:
            os.environ["GABION_FORBID_ADHOC_BUNDLES"] = prev


# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit._exception_protocol_warnings
def test_exception_protocol_warning_filters() -> None:
    da = _load()
    entries = [
        {"protocol": "never", "status": "DEAD", "site": {"path": "a.py", "function": "f"}},
        {"protocol": "never", "status": "FORBIDDEN", "site": {"path": "b.py", "function": "g"}, "exception_name": "Boom"},
    ]
    warnings = da._exception_protocol_warnings(entries)
    assert len(warnings) == 1
    assert "Boom" in warnings[0]


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_exception_obligations::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_handledness_witnesses::entries,max_entries
def test_exception_and_handledness_summary_edges() -> None:
    da = _load()
    assert da._summarize_exception_obligations([]) == []
    many_obligations = [
        {
            "site": {"path": "a.py", "function": "f", "bundle": ["a"]},
            "status": "UNKNOWN",
            "source_kind": "E0",
        }
        for _ in range(12)
    ]
    lines = da._summarize_exception_obligations(many_obligations, max_entries=10)
    assert any("... 2 more" in line for line in lines)

    assert da._summarize_handledness_witnesses([]) == []
    many_handledness = [
        {
            "site": {"path": "a.py", "function": "f", "bundle": ["a"]},
            "handler_boundary": "except:",
        }
        for _ in range(12)
    ]
    lines = da._summarize_handledness_witnesses(many_handledness, max_entries=10)
    assert any("... 2 more" in line for line in lines)


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_value_encoded_decisions_repo::forest,require_tiers E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._is_test_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._decision_tier_for::tier_map
def test_value_decision_surfaces_emit_forest(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text("def f(flag):\n    return min(flag, 1)\n")
    forest = da.Forest()
    da.analyze_value_encoded_decisions_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="low",
        external_filter=True,
        forest=forest,
    )
    assert any(alt.kind == "ValueDecisionSurface" for alt in forest.alts)


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_dataclass_call_bundles::dataclass_registry,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._populate_bundle_forest::groups_by_path
def test_populate_bundle_forest_dedupes(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text("def f(a):\n    return a\n")
    forest = da.Forest()
    groups_by_path = {path: {"f": [set(["a"]), set(["a"])]}}
    da._populate_bundle_forest(
        forest,
        groups_by_path=groups_by_path,
        file_paths=[path],
        project_root=tmp_path,
        parse_failure_witnesses=[],
    )
    assert any(alt.kind == "SignatureBundle" for alt in forest.alts)


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_callee::by_qual,callee_key,caller,class_index,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._infer_root::groups_by_path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._build_function_index::ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._merge_counts_by_knobs::knob_names E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::merge.py::gabion.synthesis.merge.merge_bundles::min_overlap E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_knob_param_names::strictness
def test_synthesis_plan_value_decision_counts(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text("def f(flag):\n    return min(flag, 1)\n")
    groups_by_path: dict[Path, dict[str, list[set[str]]]] = {path: {}}
    plan = da.build_synthesis_plan(
        groups_by_path,
        project_root=tmp_path,
        min_bundle_size=1,
        allow_singletons=True,
    )
    assert "protocols" in plan


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_callee::by_qual,callee_key,caller,class_index,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._infer_root::groups_by_path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._build_function_index::ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._merge_counts_by_knobs::knob_names E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::merge.py::gabion.synthesis.merge.merge_bundles::min_overlap E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_knob_param_names::strictness
def test_synthesis_plan_handles_empty_bundle_members(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text("def f(a):\n    return a\n")
    groups_by_path: dict[Path, dict[str, list[set[str]]]] = {
        path: {"f": [set(), set(["a"])]}
    }
    plan = da.build_synthesis_plan(
        groups_by_path,
        project_root=tmp_path,
        min_bundle_size=1,
        allow_singletons=True,
    )
    assert "protocols" in plan


def test_merge_counts_by_knobs_skips_larger_superset_after_first_merge() -> None:
    da = _load()
    counts = {
        ("a",): 1,
        ("a", "k1"): 1,
        ("a", "k1", "k2"): 1,
    }
    merged = da._merge_counts_by_knobs(counts, {"k1", "k2"})
    assert merged[("a", "k1")] >= 1


def test_build_synthesis_plan_ignores_non_literal_const_hints(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "class M:\n"
        "    CONST = 1\n"
        "\n"
        "def callee(p, k=None):\n"
        "    return p\n"
        "\n"
        "def caller():\n"
        "    return callee(M.CONST, k=M.CONST)\n",
        encoding="utf-8",
    )
    groups_by_path: dict[Path, dict[str, list[set[str]]]] = {
        path: {"callee": [set(["p", "k"])]}
    }
    plan = da.build_synthesis_plan(
        groups_by_path,
        project_root=tmp_path,
        min_bundle_size=1,
        allow_singletons=True,
    )
    assert "protocols" in plan


# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit.render_synthesis_section
def test_render_synthesis_section_evidence_summary() -> None:
    da = _load()
    plan = {
        "protocols": [
            {
                "name": "Bundle",
                "tier": 2,
                "fields": [
                    {"name": "a", "type_hint": "int", "source_params": ["a"]},
                ],
                "bundle": ["a"],
                "rationale": "test",
                "evidence": ["dataflow", "decision_surface"],
            }
        ],
        "warnings": [],
        "errors": [],
    }
    text = da.render_synthesis_section(plan)
    assert "evidence:" in text
    assert "Evidence summary:" in text


def test_render_synthesis_section_ignores_blank_field_names() -> None:
    da = _load()
    plan = {
        "protocols": [
            {
                "name": "Bundle",
                "tier": 2,
                "fields": [
                    {"name": "", "type_hint": "int", "source_params": []},
                    {"type_hint": "str", "source_params": []},
                ],
                "bundle": [],
                "rationale": "test",
                "evidence": [],
            }
        ],
        "warnings": [],
        "errors": [],
    }
    text = da.render_synthesis_section(plan)
    assert "(no fields)" in text


def test_invariant_proposition_and_projection_order_edges() -> None:
    da = _load()
    assert da.InvariantProposition(form="Eq", terms=("a", "b")).as_dict() == {
        "form": "Eq",
        "terms": ["a", "b"],
    }
    assert da.InvariantProposition(
        form="Eq",
        terms=("a", "b"),
        scope="mod.f",
        source="assert",
    ).as_dict() == {
        "form": "Eq",
        "terms": ["a", "b"],
        "scope": "mod.f",
        "source": "assert",
    }

    noop = lambda *_args, **_kwargs: []  # noqa: E731
    spec_a = da.ReportProjectionSpec(
        section_id="a",
        phase="collection",
        deps=(),
        build=noop,
        render=lambda _value: [],
        violation_extract=lambda _value: [],
    )
    spec_b = da.ReportProjectionSpec(
        section_id="b",
        phase="collection",
        deps=(),
        build=noop,
        render=lambda _value: [],
        violation_extract=lambda _value: [],
    )
    spec_c = da.ReportProjectionSpec(
        section_id="c",
        phase="collection",
        deps=("a", "b"),
        build=noop,
        render=lambda _value: [],
        violation_extract=lambda _value: [],
    )
    ordered = da._topologically_order_report_projection_specs((spec_a, spec_b, spec_c))
    assert tuple(spec.section_id for spec in ordered) == ("a", "b", "c")


def test_project_sections_and_invariant_helpers_misc_edges(tmp_path: Path) -> None:
    da = _load()
    report = da.ReportCarrier(forest=da.Forest())
    selected = da.project_report_sections(
        {},
        report,
        max_phase="collection",
        include_previews=True,
        preview_only=True,
    )
    assert isinstance(selected, dict)

    len_term = da._invariant_term(ast.parse("len(a)").body[0].value, {"a"})
    assert len_term == "a.length"

    fn = ast.parse(
        "def f(a):\n"
        "    assert a == a\n"
        "    assert a == a\n"
    ).body[0]
    collector = da._InvariantCollector({"a"}, "m.f")
    for stmt in fn.body:
        collector.visit(stmt)
    assert len(collector.propositions) == 1
    assert da._scope_path(Path("/outside/mod.py"), root=tmp_path) == "/outside/mod.py"


def test_param_spans_deadline_reason_and_local_info_edges() -> None:
    da = _load()
    synthetic_fn = ast.FunctionDef(
        name="f",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="a"), ast.arg(arg="b")],
            vararg=ast.arg(arg="args"),
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=ast.arg(arg="kwargs"),
            defaults=[],
        ),
        body=[],
        decorator_list=[],
    )
    assert da._param_spans(synthetic_fn) == {}

    assert da._never_reason(ast.parse("never(1)").body[0].value) is None
    assert da._never_reason(ast.parse("never(x=1)").body[0].value) is None
    assert da._never_reason(ast.parse("never(reason=1)").body[0].value) is None

    origin_call = ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="Deadline", ctx=ast.Load()),
            attr="from_timeout_ms",
            ctx=ast.Load(),
        ),
        args=[ast.Constant(value=10)],
        keywords=[],
    )
    local_info = da._collect_deadline_local_info(
        assignments=[
            ([ast.Name(id="d", ctx=ast.Store())], origin_call, (1, 0, 1, 5)),
            ([ast.Name(id="e", ctx=ast.Store())], ast.Name(id="d", ctx=ast.Load()), (2, 0, 2, 1)),
            ([ast.Name(id="skip", ctx=ast.Store())], None, None),
            ([ast.Name(id="skip", ctx=ast.Store())], ast.Name(id="deadline", ctx=ast.Load()), None),
        ],
        params={"deadline"},
    )
    assert "d" in local_info.origin_vars
    assert "e" in local_info.origin_vars
    assert local_info.alias_to_param["deadline"] == "deadline"


def test_deadline_collector_call_and_bind_args_edges() -> None:
    da = _load()
    fn = ast.parse(
        "def f(deadline):\n"
        "    obj.check_deadline(1)\n"
        "    obj.check_deadline()\n"
        "    check_deadline(1)\n"
        "    require_deadline()\n"
        "    return deadline\n"
    ).body[0]
    collector = da._DeadlineFunctionCollector(fn, {"deadline"})
    collector.visit(fn)
    assert collector.ambient_check is True

    call_node = ast.parse("fn(1, *[2], **{'x': 3}, named=4)").body[0].value
    callee = da.FunctionInfo(
        name="fn",
        qual="pkg.fn",
        path=Path("pkg/mod.py"),
        params=["a", "named"],
        annots={},
        calls=[],
        unused_params=set(),
        positional_params=("a",),
        kwonly_params=("named",),
        function_span=(1, 1, 1, 2),
    )
    bound = da._bind_call_args(call_node, callee, strictness="low")
    assert "a" in bound
    assert "named" in bound


def test_load_analysis_index_resume_payload_edge_shapes(tmp_path: Path) -> None:
    da = _load()
    file_path = tmp_path / "mod.py"
    payload = {
        "format_version": 1,
        "hydrated_paths": str(file_path),  # Sequence[str] branch with path miss hits
        "functions_by_qual": {"pkg.bad": [], 1: {}},
        "symbol_table": {"imports": []},
        "class_index": {"pkg.C": []},
    }
    hydrated, by_qual, symbol_table, class_index = da._load_analysis_index_resume_payload(
        payload=payload,
        file_paths=[file_path],
    )
    assert hydrated == set()
    assert by_qual == {}
    assert isinstance(symbol_table, da.SymbolTable)
    assert class_index == {}


def test_load_analysis_index_resume_payload_hydrates_valid_sections(tmp_path: Path) -> None:
    da = _load()
    file_path = tmp_path / "mod.py"
    file_path.write_text("def f(x):\n    return x\n", encoding="utf-8")
    payload = {
        "format_version": 1,
        "hydrated_paths": [str(file_path)],
        "functions_by_qual": {
            "mod.f": {
                "name": "f",
                "qual": "mod.f",
                "path": str(file_path),
                "params": ["x"],
                "annots": {},
                "calls": [],
                "unused_params": [],
                "transparent": True,
                "class_name": None,
                "scope": [],
                "lexical_scope": [],
                "decision_params": [],
                "value_decision_params": [],
                "value_decision_reasons": [],
                "positional_params": ["x"],
                "kwonly_params": [],
                "vararg": None,
                "kwarg": None,
                "param_spans": {},
                "function_span": [1, 0, 1, 8],
            }
        },
        "symbol_table": {
            "imports": [],
            "internal_roots": [],
            "external_filter": True,
            "star_imports": {},
            "module_exports": {},
            "module_export_map": {},
        },
        "class_index": {
            "mod.C": {
                "qual": "mod.C",
                "module": "mod",
                "bases": [],
                "methods": [],
            }
        },
    }
    hydrated, by_qual, symbol_table, class_index = da._load_analysis_index_resume_payload(
        payload=payload,
        file_paths=[file_path],
    )
    assert hydrated == {file_path}
    assert set(by_qual) == {"mod.f"}
    assert isinstance(symbol_table, da.SymbolTable)
    assert set(class_index) == {"mod.C"}


def test_scope_path_relative_and_none_root_edges(tmp_path: Path) -> None:
    da = _load()
    inside = tmp_path / "pkg" / "mod.py"
    outside = Path("/outside/mod.py")
    assert da._scope_path(inside, tmp_path) == "pkg/mod.py"
    assert da._scope_path(outside, tmp_path) == "/outside/mod.py"
    assert da._scope_path(inside, None).endswith("pkg/mod.py")


def test_resolve_local_method_in_hierarchy_recurses_to_base() -> None:
    da = _load()
    resolved = da._resolve_local_method_in_hierarchy(
        "Child",
        "act",
        class_bases={"Child": ["Base"], "Base": []},
        local_functions={"Base.act"},
        seen=set(),
    )
    assert resolved == "Base.act"


def test_fallback_deadline_arg_info_skips_vararg_kwarg_when_absent() -> None:
    da = _load()
    call = da.CallArgs(
        callee="pkg.target",
        pos_map={"1": "extra_pos"},
        kw_map={"extra": "extra_kw"},
        const_pos={"2": "1"},
        const_kw={"extra_const": "2"},
        non_const_pos={"3"},
        non_const_kw={"extra_unknown"},
        star_pos=[],
        star_kw=[],
        is_test=False,
        span=(1, 1, 1, 2),
    )
    callee = da.FunctionInfo(
        name="target",
        qual="pkg.target",
        path=Path("pkg/target.py"),
        params=["p0"],
        annots={},
        calls=[],
        unused_params=set(),
        positional_params=("p0",),
        kwonly_params=(),
        vararg=None,
        kwarg=None,
        function_span=(1, 1, 1, 2),
    )
    info_map = da._fallback_deadline_arg_info(call, callee, strictness="high")
    assert set(info_map) == set()


def test_analyze_decision_surface_indexed_lint_none_paths(tmp_path: Path) -> None:
    da = _load()
    fn = da.FunctionInfo(
        name="f",
        qual="pkg.f",
        path=tmp_path / "mod.py",
        params=["flag"],
        annots={},
        calls=[],
        unused_params=set(),
        decision_params={"flag"},
        function_span=(1, 0, 1, 10),
    )
    by_qual = {fn.qual: fn}
    index = da.AnalysisIndex(
        by_name={"f": [fn]},
        by_qual=by_qual,
        symbol_table=da.SymbolTable(),
        class_index={},
    )
    index.transitive_callers = {fn.qual: set()}
    context = da._IndexedPassContext(
        paths=[fn.path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        transparent_decorators=None,
        parse_failure_witnesses=[],
        analysis_index=index,
    )
    _surfaces, warnings, _rewrites, lint_lines = da._analyze_decision_surface_indexed(
        context,
        spec=da._DIRECT_DECISION_SURFACE_SPEC,
        decision_tiers={"other": 1},
        require_tiers=True,
        forest=da.Forest(),
    )
    assert warnings
    assert lint_lines == []

    # Internal-caller branch with tiered warning and lint=None.
    index.transitive_callers = {fn.qual: {"pkg.caller"}}
    _surfaces2, warnings2, _rewrites2, lint_lines2 = da._analyze_decision_surface_indexed(
        context,
        spec=da._DIRECT_DECISION_SURFACE_SPEC,
        decision_tiers={"flag": 2},
        require_tiers=False,
        forest=da.Forest(),
    )
    assert warnings2
    assert lint_lines2 == []


def test_collect_exception_obligations_dead_reachability_branch(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "def f(flag):\n"
        "    if flag:\n"
        "        raise ValueError(flag)\n",
        encoding="utf-8",
    )
    path_value = da._normalize_snapshot_path(path, tmp_path)
    obligations = da._collect_exception_obligations(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        handledness_witnesses=[{"exception_path_id": "", "handledness_id": "skip"}],
        deadness_witnesses=[
            {
                "path": path_value,
                "function": "f",
                "bundle": ["flag"],
                "environment": {"flag": "False"},
                "deadness_id": "dead:f",
            }
        ],
    )
    assert obligations
    assert obligations[0]["status"] == "DEAD"


def test_build_synthesis_plan_duplicate_counts_and_hint_branches(tmp_path: Path) -> None:
    da = _load()
    module = tmp_path / "mod.py"
    module.write_text(
        "def target(flag):\n"
        "    if flag:\n"
        "        return 1\n"
        "    return (flag == 1) * 2\n"
        "\n"
        "def helper(flag):\n"
        "    return flag\n"
        "\n"
        "def caller_pos():\n"
        "    return helper(1)\n"
        "\n"
        "def caller_kw():\n"
        "    return helper(flag=2)\n",
        encoding="utf-8",
    )
    groups_by_path = {module: {"target": [set(["flag"])]}}
    plan = da.build_synthesis_plan(
        groups_by_path,
        project_root=tmp_path,
        merge_overlap_threshold=1.0,
    )
    assert isinstance(plan, dict)
    assert "protocols" in plan


def test_project_report_sections_phase_and_preview_branches() -> None:
    da = _load()
    report = da.ReportCarrier(forest=da.Forest(), constant_smells=["const smell"])
    selected = da.project_report_sections(
        {},
        report,
        max_phase="post",
        include_previews=True,
        preview_only=True,
    )
    assert selected
    selected_without_max = da.project_report_sections(
        {},
        report,
        max_phase=None,
        include_previews=True,
        preview_only=True,
    )
    assert selected_without_max


def test_invariant_term_len_call_branch() -> None:
    da = _load()
    term = da._invariant_term(ast.parse("len(flag)").body[0].value, {"flag"})
    assert term == "flag.length"


def test_resolve_local_method_in_hierarchy_unresolved_branch() -> None:
    da = _load()
    resolved = da._resolve_local_method_in_hierarchy(
        "Child",
        "missing",
        class_bases={"Child": ["Base"], "Base": []},
        local_functions={"Base.other"},
        seen=set(),
    )
    assert resolved is None


def test_collect_deadline_local_info_multi_source_alias_edges() -> None:
    da = _load()
    origin_call = ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="Deadline", ctx=ast.Load()),
            attr="from_timeout_ms",
            ctx=ast.Load(),
        ),
        args=[ast.Constant(value=10)],
        keywords=[],
    )
    local_info = da._collect_deadline_local_info(
        assignments=[
            ([ast.Name(id="origin", ctx=ast.Store())], origin_call, (1, 0, 1, 8)),
            ([ast.Name(id="origin", ctx=ast.Store())], origin_call, (2, 0, 2, 8)),
            ([ast.Name(id="x", ctx=ast.Store())], ast.Name(id="a", ctx=ast.Load()), None),
            ([ast.Name(id="x", ctx=ast.Store())], ast.Name(id="b", ctx=ast.Load()), None),
            ([ast.Name(id="from_origin", ctx=ast.Store())], ast.Name(id="origin", ctx=ast.Load()), None),
        ],
        params={"a", "b"},
    )
    assert "origin" in local_info.origin_vars
    assert "x" not in local_info.alias_to_param
    assert "from_origin" in local_info.origin_vars


def test_summarize_never_invariants_evidence_edges() -> None:
    da = _load()
    entries = [
        {
            "status": "VIOLATION",
            "site": {"path": "a.py", "function": "f", "bundle": ["x"]},
            "witness_ref": None,
            "environment_ref": {"x": "1"},
            "span": [1, 1, 1, 2],
        },
        {
            "status": "PROVEN_UNREACHABLE",
            "site": {"path": "a.py", "function": "f", "bundle": ["x"]},
            "witness_ref": "dead:1",
            "environment_ref": {"x": "0"},
            "span": [2, 1, 2, 2],
        },
    ]
    lines = da._summarize_never_invariants(entries)
    assert any("env=" in line for line in lines)
    assert any("deadness=dead:1" in line for line in lines)


def test_load_analysis_index_resume_payload_non_mapping_sections(tmp_path: Path) -> None:
    da = _load()
    file_path = tmp_path / "mod.py"
    payload = {
        "format_version": 1,
        "hydrated_paths": None,
        "functions_by_qual": [],
        "symbol_table": [],
        "class_index": [],
    }
    hydrated, by_qual, symbol_table, class_index = da._load_analysis_index_resume_payload(
        payload=payload,
        file_paths=[file_path],
    )
    assert hydrated == set()
    assert by_qual == {}
    assert isinstance(symbol_table, da.SymbolTable)
    assert class_index == {}


def test_verify_rewrite_plan_verification_and_remainder_edges() -> None:
    da = _load()
    plan = {
        "plan_id": "p1",
        "site": {"path": "a.py", "function": "f", "bundle": ["x"]},
        "pre": {"base_keys": ["int"], "ctor_keys": [], "remainder": {"base": 2, "ctor": 2}},
        "rewrite": {"parameters": {"candidates": ["ctx"]}},
        "verification": [],
    }
    post = [
        {
            "path": "a.py",
            "function": "f",
            "bundle": ["x"],
            "base_keys": ["int"],
            "ctor_keys": [],
            "remainder": {"base": 1, "ctor": 1},
            "glossary_matches": ["ctx"],
        }
    ]
    result = da.verify_rewrite_plan(plan, post_provenance=post)
    assert result["accepted"] is True


def test_eval_bool_expr_or_gte_and_branch_reachability_else_edges() -> None:
    da = _load()
    or_expr = ast.parse("a or b").body[0].value
    assert da._eval_bool_expr(or_expr, {"a": False, "b": True}) is True
    gte_expr = ast.parse("x >= 1").body[0].value
    assert da._eval_bool_expr(gte_expr, {"x": 2}) is True

    tree = ast.parse(
        "if flag:\n"
        "    a = 1\n"
        "else:\n"
        "    raise ValueError(flag)\n"
    )
    parent = da.ParentAnnotator()
    parent.visit(tree)
    raise_node = tree.body[0].orelse[0]
    reach = da._branch_reachability_under_env(
        raise_node,
        parent.parents,
        {"flag": True},
    )
    assert reach is False


def test_collect_module_exports_all_assignment_none_values_edges() -> None:
    da = _load()
    tree = ast.parse(
        "__all__ = [name]\n"
        "__all__ = [\"public\"]\n"
        "public = 1\n"
    )
    exports, export_map = da._collect_module_exports(
        tree,
        module_name="pkg.mod",
        import_map={},
    )
    assert "public" in exports
    assert export_map.get("public") == "pkg.mod.public"


def test_collect_module_exports_annassign_and_augassign_edges() -> None:
    da = _load()
    tree = ast.parse(
        "__all__: list[str] = [\"first\"]\n"
        "__all__ += [\"second\"]\n"
        "first = 1\n"
        "second = 2\n",
    )
    exports, _ = da._collect_module_exports(
        tree,
        module_name="pkg.mod",
        import_map={},
    )
    assert "first" in exports
    assert "second" in exports


def test_invariant_term_len_call_branch() -> None:
    da = _load()
    expr = ast.parse("len(data)").body[0].value
    assert da._invariant_term(expr, {"data"}) == "data.length"


def test_accumulate_function_index_vararg_and_kwarg_ignored() -> None:
    da = _load()
    tree = ast.parse("def f(*skip_a, **skip_k):\n    return 1\n")
    acc = da._FunctionIndexAccumulator()
    da._accumulate_function_index_for_tree(
        acc,
        Path("mod.py"),
        tree,
        project_root=Path("."),
        ignore_params={"skip_a", "skip_k"},
        strictness="low",
        transparent_decorators=None,
    )
    info = acc.by_name["f"][0]
    assert info.vararg is None
    assert info.kwarg is None


def test_bundle_name_registry_non_empty_keys(tmp_path: Path) -> None:
    da = _load()
    (tmp_path / "mod.py").write_text(
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass\n"
        "class DemoData:\n"
        "    x: int\n"
        "\n"
        "class DemoConfig:\n"
        "    value: int\n",
        encoding="utf-8",
    )
    registry = da._bundle_name_registry(tmp_path)
    assert ("x",) in registry
    assert ("value",) in registry


def test_bundle_projection_skips_empty_evidence_paths(tmp_path: Path) -> None:
    da = _load()
    forest = da.Forest()
    site = forest.add_site("a.py", "f")
    paramset = forest.add_paramset(["p"])
    forest.add_alt("SignatureBundle", (site, paramset))
    forest.add_alt("ConfigBundle", (paramset,), evidence={"path": ""})
    forest.add_alt("MarkerBundle", (paramset,), evidence={"path": ""})
    projection = da._bundle_projection_from_forest(
        forest,
        file_paths=[tmp_path / "a.py"],
    )
    assert projection.declared_global == {("p",)}
    assert projection.declared_by_path == {}
    assert projection.documented_by_path == {}


def test_render_mermaid_component_empty_component_branch() -> None:
    da = _load()
    mermaid, summary = da._render_mermaid_component(
        nodes={},
        bundle_map={},
        bundle_counts={},
        adj={},
        component=[],
        declared_global=set(),
        declared_by_path={},
        documented_by_path={},
    )
    assert "flowchart LR" in mermaid
    assert "Observed bundles:" in summary


def test_split_top_level_empty_part_and_tail_edges() -> None:
    da = _load()
    assert da._split_top_level("a,,", ",") == ["a"]


def test_summarize_never_invariants_missing_evidence_branches() -> None:
    da = _load()
    entries = [
        {
            "status": "VIOLATION",
            "site": {"path": "a.py", "function": "f", "bundle": ["x"]},
            "witness_ref": "w:1",
            "environment_ref": None,
            "span": [1, 1, 1, 2],
        },
        {
            "status": "PROVEN_UNREACHABLE",
            "site": {"path": "a.py", "function": "f", "bundle": ["x"]},
            "witness_ref": None,
            "environment_ref": None,
            "span": [2, 1, 2, 2],
        },
    ]
    lines = da._summarize_never_invariants(entries)
    assert any("witness=w:1" in line for line in lines)


def test_resolve_callee_self_and_hierarchy_none_branches(tmp_path: Path) -> None:
    da = _load()
    caller = da.FunctionInfo(
        name="caller",
        qual="pkg.mod.caller",
        path=tmp_path / "mod.py",
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        class_name=None,
    )
    # self/cls branch with no class_name should fall through.
    assert (
        da._resolve_callee(
            "self.run",
            caller,
            {"caller": [caller]},
            {},
            da.SymbolTable(),
            tmp_path,
            {},
        )
        is None
    )

    caller_with_class = da.FunctionInfo(
        name="caller",
        qual="pkg.mod.Caller.caller",
        path=tmp_path / "mod.py",
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        class_name="Caller",
    )
    class_index = {
        "pkg.mod.Caller": da.ClassInfo(
            qual="pkg.mod.Caller",
            module="pkg.mod",
            bases=[],
            methods={"other"},
        )
    }
    # Candidate path exists but method resolution yields None.
    assert (
        da._resolve_callee(
            "pkg.mod.Caller.missing",
            caller_with_class,
            {"caller": [caller_with_class]},
            {},
            da.SymbolTable(),
            tmp_path,
            class_index,
        )
        is None
    )


def test_iter_dataclass_call_bundles_assign_and_attribute_branches(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Empty:\n"
        "    pass\n"
        "@dataclass\n"
        "class Item:\n"
        "    a: int\n"
        "    b = 1\n"
        "    c, d = (1, 2)\n"
        "def make(alias):\n"
        "    Item(1, 2)\n"
        "    alias.Item(1, 2)\n"
        "    (alias()).Item(1, 2)\n",
        encoding="utf-8",
    )
    symbol_table = da.SymbolTable(
        imports={("mod", "alias"): "external.pkg"},
        internal_roots={"mod"},
        external_filter=True,
    )
    bundles = da._iter_dataclass_call_bundles(
        path,
        project_root=tmp_path,
        dataclass_registry={"mod.Item": ["a", "b"]},
        symbol_table=symbol_table,
        parse_failure_witnesses=[],
    )
    assert ("a", "b") in bundles


def test_analyze_decision_surface_indexed_missing_tier_without_require(tmp_path: Path) -> None:
    da = _load()
    fn = da.FunctionInfo(
        name="f",
        qual="pkg.f",
        path=tmp_path / "mod.py",
        params=["flag"],
        annots={},
        calls=[],
        unused_params=set(),
        decision_params={"flag"},
        function_span=(1, 0, 1, 8),
    )
    index = da.AnalysisIndex(
        by_name={"f": [fn]},
        by_qual={fn.qual: fn},
        symbol_table=da.SymbolTable(),
        class_index={},
        transitive_callers={fn.qual: set()},
    )
    context = da._IndexedPassContext(
        paths=[fn.path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        transparent_decorators=None,
        parse_failure_witnesses=[],
        analysis_index=index,
    )
    _surfaces, warnings, _rewrites, lint_lines = da._analyze_decision_surface_indexed(
        context,
        spec=da._DIRECT_DECISION_SURFACE_SPEC,
        decision_tiers={"other": 1},
        require_tiers=False,
        forest=da.Forest(),
    )
    assert warnings == []
    assert lint_lines == []


def test_project_report_sections_preview_selects_non_empty_preview(tmp_path: Path) -> None:
    da = _load()
    report = da.ReportCarrier(
        forest=da.Forest(),
        decision_warnings=["w1"],
        parse_failure_witnesses=[],
    )
    groups_by_path = {tmp_path / "m.py": {"f": [{"x"}]}}
    projected = da.project_report_sections(
        groups_by_path,
        report,
        include_previews=True,
        preview_only=True,
        max_phase="post",
    )
    assert "violations" in projected
    assert any("known_violations" in line for line in projected["violations"])


def test_internal_broad_type_lint_lines_indexed_appends_multiple(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    info = da.FunctionInfo(
        name="f",
        qual="pkg.f",
        path=path,
        params=["a", "b", "c"],
        annots={"a": "Any", "b": "Any", "c": "Any"},
        calls=[],
        unused_params=set(),
        param_spans={"a": (0, 0, 0, 1), "b": (0, 2, 0, 3)},
    )
    index = da.AnalysisIndex(
        by_name={"f": [info]},
        by_qual={info.qual: info},
        symbol_table=da.SymbolTable(),
        class_index={},
        transitive_callers={info.qual: {"pkg.caller"}},
    )
    context = da._IndexedPassContext(
        paths=[path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        transparent_decorators=None,
        parse_failure_witnesses=[],
        analysis_index=index,
    )
    lines = da._internal_broad_type_lint_lines_indexed(context)
    assert len(lines) == 2
    assert all("GABION_BROAD_TYPE" in line for line in lines)


def test_raw_sorted_contract_violations_multi_path_exceeded_loop(tmp_path: Path) -> None:
    da = _load()
    left = tmp_path / "a.py"
    right = tmp_path / "b.py"
    left.write_text("def f(xs):\n    return sorted(xs)\n", encoding="utf-8")
    right.write_text("def g(xs):\n    return sorted(xs)\n", encoding="utf-8")
    baseline = {
        da._raw_sorted_baseline_key(left): 0,
        da._raw_sorted_baseline_key(right): 0,
    }
    lines = da._raw_sorted_contract_violations(
        [left, right],
        parse_failure_witnesses=[],
        baseline_counts=baseline,
    )
    assert len(lines) == 2
    assert all("raw_sorted exceeded baseline" in line for line in lines)


def test_fingerprint_warning_provenance_and_rewrite_verification_edges() -> None:
    da = _load()
    path = Path("pkg/mod.py")
    groups = {path: {"f": [{"x"}], "g": [{"x"}]}}
    annots = {path: {"f": {"x": "int"}, "g": {"x": "int"}}}
    registry = da.PrimeRegistry()
    index = {
        da.bundle_fingerprint_dimensional(["str"], registry, None): {"OtherCtx"},
    }
    warnings = da._compute_fingerprint_warnings(
        groups,
        annots,
        registry=registry,
        index=index,
    )
    assert warnings
    provenance = da._compute_fingerprint_provenance(
        groups,
        annots,
        registry=registry,
        index=index,
    )
    assert provenance
    summary = da._summarize_fingerprint_provenance(provenance, max_examples=1)
    assert any("base=" in line for line in summary)

    rewrite_source = [
        {
            "path": "pkg/mod.py",
            "function": "f",
            "bundle": ["x"],
            "glossary_matches": ["CtxA", "CtxB"],
            "base_keys": ["int"],
            "ctor_keys": [],
            "remainder": {"base": 1, "ctor": 1},
        }
    ]
    plans = da._compute_fingerprint_rewrite_plans(
        rewrite_source,
        coherence=[{"site": {"path": "pkg/mod.py", "function": "f", "bundle": ["x"]}}],
        synth_version="synth@1",
    )
    assert plans
    plan = dict(plans[0])
    plan["verification"] = {"predicates": "not-a-list"}
    verified = da.verify_rewrite_plan(plan, post_provenance=rewrite_source)
    assert isinstance(verified.get("accepted"), bool)


def test_invariant_term_len_with_non_param_argument_returns_none() -> None:
    da = _load()
    expr = ast.parse("len(1)").body[0].value
    assert da._invariant_term(expr, {"data"}) is None


def test_parameter_default_map_multiple_defaults_runs_single_check_once() -> None:
    da = _load()
    fn = ast.parse("def f(a=1, b=2):\n    return a + b\n").body[0]
    mapping = da._parameter_default_map(fn)
    assert set(mapping) == {"a", "b"}


def test_raw_sorted_contract_violations_mixed_baseline_paths(tmp_path: Path) -> None:
    da = _load()
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("def f(xs):\n    return sorted(xs)\n", encoding="utf-8")
    second.write_text("def g(xs):\n    return sorted(xs)\n", encoding="utf-8")
    baseline = {
        da._raw_sorted_baseline_key(first): 0,
        da._raw_sorted_baseline_key(second): 2,
    }
    lines = da._raw_sorted_contract_violations(
        [first, second],
        parse_failure_witnesses=[],
        baseline_counts=baseline,
    )
    assert len(lines) == 1
    assert "raw_sorted exceeded baseline" in lines[0]


def test_fingerprint_provenance_index_lookup_and_types_summary_branch() -> None:
    da = _load()
    path = Path("pkg/mod.py")
    groups = {path: {"f": [{"x"}], "g": [{"x"}]}}
    annots = {path: {"f": {"x": "int"}, "g": {"x": "int"}}}
    registry = da.PrimeRegistry()
    fp = da.bundle_fingerprint_dimensional(["int"], registry, None)
    provenance = da._compute_fingerprint_provenance(
        groups,
        annots,
        registry=registry,
        index={fp: {"KnownCtx"}},
    )
    assert provenance
    # No glossary matches => "types" grouping branch.
    for entry in provenance:
        entry["glossary_matches"] = []
    lines = da._summarize_fingerprint_provenance(provenance, max_examples=2)
    assert any("base=" in line for line in lines)


def test_eval_value_expr_unary_non_numeric_and_parse_range_branch() -> None:
    da = _load()
    assert da._eval_value_expr(ast.parse("-'x'").body[0].value, {}) is None
    assert da._parse_lint_location("a.py:1:2:-3:4: GABION_X message") is not None


def test_collect_exception_obligations_names_loop_without_env_match(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "m.py"
    path.write_text(
        "def f(x):\n"
        "    if 0 and missing:\n"
        "        return 1\n"
        "    raise ValueError(x)\n",
        encoding="utf-8",
    )
    deadness = [
        {
            "path": "m.py",
            "function": "f",
            "bundle": ["x"],
            "environment": {"x": "0"},
            "deadness_id": "dead:f:x",
        }
    ]
    obligations = da._collect_exception_obligations(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        deadness_witnesses=deadness,
    )
    assert obligations and obligations[0]["status"] in {"UNKNOWN", "DEAD"}


def test_collect_call_resolution_obligations_invalid_span_list_raises() -> None:
    da = _load()
    forest = da.Forest()
    call_suite = forest.add_suite_site(
        "a.py",
        "pkg.f",
        "call",
        span=(1, 0, 1, 1),
    )
    # Replace span metadata with invalid list to take coercion-failure branch.
    forest.nodes[call_suite].meta["span"] = [1, "x", 1, 1]
    forest.add_alt(
        "CallResolutionObligation",
        (call_suite,),
        evidence={"callee": "target"},
    )
    with pytest.raises(NeverThrown):
        da._collect_call_resolution_obligations_from_forest(forest)


def test_collect_recursive_nodes_singleton_self_loop_false_branch() -> None:
    da = _load()
    assert da._collect_recursive_nodes({"a": set()}) == set()


def test_analysis_index_resolved_edges_by_caller_require_transparent_branch() -> None:
    da = _load()
    index = da.AnalysisIndex(
        by_name={},
        by_qual={},
        symbol_table=da.SymbolTable(),
        class_index={},
        resolved_transparent_call_edges=(),
    )
    assert (
        da._analysis_index_resolved_call_edges_by_caller(
            index,
            project_root=None,
            require_transparent=True,
        )
        == {}
    )


def test_collect_bundle_evidence_lines_with_component_evidence() -> None:
    da = _load()
    forest = da.Forest()
    site = forest.add_site("a.py", "f")
    bundle = forest.add_paramset(["x", "y"])
    forest.add_alt("SignatureBundle", (site, bundle))
    groups = {Path("a.py"): {"f": [{"x", "y"}]}}
    bundle_sites = {
        Path("a.py"): {
            "f": [
                [
                    {
                        "span": [0, 0, 0, 1],
                        "callee": "pkg.target",
                        "params": ["x", "y"],
                        "slots": ["x", "y"],
                    }
                ]
            ]
        }
    }
    lines = da._collect_bundle_evidence_lines(
        forest=forest,
        groups_by_path=groups,
        bundle_sites_by_path=bundle_sites,
    )
    assert lines


def test_class_index_and_resolve_candidates_with_symbol_table_branches() -> None:
    da = _load()
    tree = ast.parse("class C(A, B):\n    pass\n")
    class_index: dict[str, da.ClassInfo] = {}
    da._accumulate_class_index_for_tree(
        class_index,
        Path("mod.py"),
        tree,
        project_root=Path("."),
    )
    assert class_index

    symbol_table = da.SymbolTable(imports={("pkg.mod", "pkg"): "pkg"}, internal_roots={"pkg"})
    resolved = da._resolve_class_candidates(
        "pkg.Base",
        module="pkg.mod",
        symbol_table=symbol_table,
        class_index={"pkg.Base": da.ClassInfo("pkg.Base", "pkg", [], set())},
    )
    assert "pkg.Base" in resolved


def test_collect_module_exports_augassign_initializes_explicit_all() -> None:
    da = _load()
    exports, _ = da._collect_module_exports(
        ast.parse("__all__ += ['x']\nx=1\n"),
        module_name="pkg.mod",
        import_map={},
    )
    assert "x" in exports


def test_render_reuse_stubs_and_refactor_plan_order_branches() -> None:
    da = _load()
    reuse = {
        "suggested_lemmas": [
            {
                "kind": "bundle",
                "suggested_name": "lemma_name",
                "count": 2,
                "value": ["x"],
            }
        ]
    }
    stubs = da.render_reuse_lemma_stubs(reuse)
    assert "def lemma_name" in stubs

    text = da.render_refactor_plan(
        {
            "bundles": [
                {"bundle": ["x"], "order": ["a", "b"], "cycles": [["a", "b"]]},
            ],
            "warnings": [],
        }
    )
    assert "Order (callee-first):" in text


def test_eval_expr_and_branch_reachability_else_constraint_edges() -> None:
    da = _load()
    assert da._eval_value_expr(ast.parse("+2").body[0].value, {}) == 2
    assert da._eval_bool_expr(ast.parse("a or b").body[0].value, {"a": False, "b": False}) is False
    assert da._eval_bool_expr(ast.parse("'x' >= 'a'").body[0].value, {}) is None

    tree = ast.parse(
        "if flag:\n"
        "    a = 1\n"
        "else:\n"
        "    raise ValueError(flag)\n"
    )
    parent = da.ParentAnnotator()
    parent.visit(tree)
    raise_node = tree.body[0].orelse[0]
    assert (
        da._branch_reachability_under_env(
            raise_node,
            parent.parents,
            {"flag": False},
        )
        is True
    )


def test_collect_exception_obligations_dead_env_name_filter_branches(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "dead_env.py"
    path.write_text(
        "def f(x):\n"
        "    if 0 and missing:\n"
        "        return\n"
        "    raise ValueError(x)\n"
        "def g(y):\n"
        "    if 0:\n"
        "        return\n"
        "    raise RuntimeError(y)\n",
        encoding="utf-8",
    )
    deadness = [
        {
            "path": "dead_env.py",
            "function": "f",
            "bundle": ["x"],
            "environment": {"x": "0"},
            "deadness_id": "dead:f:x",
        },
        {
            "path": "dead_env.py",
            "function": "g",
            "bundle": ["y"],
            "environment": {"y": "0"},
            "deadness_id": "dead:g:y",
        },
    ]
    obligations = da._collect_exception_obligations(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        deadness_witnesses=deadness,
    )
    assert obligations
    assert any(entry.get("status") in {"DEAD", "UNKNOWN"} for entry in obligations)


def test_deadline_local_info_call_resolution_and_recursive_node_edges() -> None:
    da = _load()
    fn = ast.parse(
        "def f(deadline):\n"
        "    token = Deadline.from_timeout_ms(1)\n"
        "    alias = token\n"
        "    check_deadline(0)\n"
    ).body[0]
    visitor = da._DeadlineFunctionCollector(fn, {"deadline"})
    visitor.visit(fn)
    local = da._collect_deadline_local_info(visitor.assignments, {"deadline"})
    assert "token" in local.origin_vars or "alias" in local.origin_vars

    forest = da.Forest()
    call_suite = forest.add_suite_site(
        "a.py",
        "pkg.f",
        "call",
        span=(1, 0, 1, 10),
    )
    forest.add_alt(
        "CallResolutionObligation",
        (call_suite,),
        evidence={"callee": "target"},
    )
    assert da._collect_call_resolution_obligations_from_forest(forest)
    assert da._collect_recursive_nodes({"a": set()}) == set()


def test_bind_call_args_classify_deadline_and_forward_edges() -> None:
    da = _load()
    call = ast.parse("fn(1, extra=2, extra2=3)").body[0].value
    callee = da.FunctionInfo(
        name="callee",
        qual="pkg.callee",
        path=Path("m.py"),
        params=["x"],
        annots={},
        calls=[],
        unused_params=set(),
        kwarg="kwargs",
    )
    mapping = da._bind_call_args(call, callee, strictness="high")
    assert "kwargs" in mapping

    assert da._classify_deadline_expr(
        ast.parse("origin_deadline").body[0].value,
        alias_to_param={},
        origin_vars={"origin_deadline"},
    ) == da._DeadlineArgInfo(kind="origin", param="origin_deadline")

    call_map = {"deadline": da._DeadlineArgInfo(kind="param", param="deadline")}
    loop_fact = da._DeadlineLoopFacts(span=(0, 0, 0, 1), kind="for")
    loop_fact.call_spans.add((0, 0, 0, 1))
    call = da.CallArgs(
        callee="callee",
        pos_map={},
        kw_map={},
        const_pos={},
        const_kw={},
        non_const_pos=set(),
        non_const_kw=set(),
        star_pos=[],
        star_kw=[],
        is_test=False,
        span=(0, 0, 0, 1),
    )
    callee = da.FunctionInfo(
        name="callee",
        qual="pkg.callee",
        path=Path("m.py"),
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
    )
    assert da._deadline_loop_forwarded_params(
        qual="pkg.caller",
        loop_fact=loop_fact,
        deadline_params={"pkg.caller": {"deadline"}, "pkg.callee": {"deadline"}},
        call_infos={"pkg.caller": [(call, callee, call_map)]},
    ) == {"deadline"}


def test_deadline_summary_parse_location_and_lint_edges() -> None:
    da = _load()
    forest = da.Forest()
    summary = da._summarize_deadline_obligations(
        [
            {
                "site": {"path": "src/a.py", "function": "f", "bundle": ["x"]},
                "span": [0, 0, 0, 1],
                "status": "UNKNOWN",
                "kind": "k",
                "detail": "d",
                "deadline_id": "id:1",
            }
        ],
        forest=forest,
    )
    assert summary

    lint = da._deadline_lint_lines(
        [{"site": {"path": "a.py"}, "span": [0, 1, 0, 2], "status": "S", "kind": "K"}]
    )
    assert lint and lint[0].startswith("a.py:1:2:")

    parsed = da._parse_lint_location("a.py:1:2:-9:10: GABION_X message")
    assert parsed is not None
    assert parsed[0] == "a.py"


def test_build_module_artifacts_and_lint_helper_edges(tmp_path: Path) -> None:
    da = _load()
    first = tmp_path / "a.py"
    second = tmp_path / "b.py"
    first.write_text("x = 1\n", encoding="utf-8")
    second.write_text("y = 2\n", encoding="utf-8")

    spec = da._ModuleArtifactSpec(
        artifact_id="count_nodes",
        stage=da._ParseModuleStage.FUNCTION_INDEX,
        init=lambda: [],
        fold=lambda acc, path, tree: acc.append((path.name, len(list(ast.walk(tree))))),
        finish=lambda acc: tuple(acc),
    )
    built = da._build_module_artifacts(
        [first, second],
        specs=(spec,),
        parse_failure_witnesses=[],
    )
    assert len(built[0]) == 2

    ambiguity = da._lint_lines_from_call_ambiguities(
        [{"site": {"path": "a.py", "span": [0, 0, 0, 1]}, "candidate_count": 3}]
    )
    const_lines = da._lint_lines_from_constant_smells(
        ["constant smell (e.g. a.py:2:3: from sample)"]
    )
    assert ambiguity and "GABION_AMBIGUITY" in ambiguity[0]
    assert const_lines and "GABION_CONST_FLOW" in const_lines[0]


def test_materialize_structured_suites_populate_runtime_and_exports_edges(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "def f(x):\n"
        "    if x:\n"
        "        return 1\n"
        "    return 0\n",
        encoding="utf-8",
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forest = da.Forest()
    da._materialize_structured_suite_sites_for_tree(
        forest=forest,
        path=path,
        tree=tree,
        project_root=tmp_path,
    )
    assert any(node.kind == "SuiteSite" for node in forest.nodes)

    groups = {path: {"f": [{"x", "y"}]}}
    da._populate_bundle_forest(
        forest,
        groups_by_path=groups,
        file_paths=[path],
        project_root=tmp_path,
        include_all_sites=True,
        parse_failure_witnesses=[],
    )
    assert forest.alts

    runtime_lines = da._summarize_runtime_obligations(
        [
            {"status": "VIOLATION", "contract": "c", "kind": "k", "detail": "d1"},
            {"status": "SATISFIED", "contract": "c", "kind": "k", "detail": "d2"},
        ],
        max_entries=1,
    )
    assert any(line.startswith("... ") for line in runtime_lines)

    exports, _ = da._collect_module_exports(
        ast.parse("__all__ += ['a']\na=1\n"),
        module_name="pkg.mod",
        import_map={},
    )
    assert "a" in exports


def test_class_resolution_type_flow_and_refactor_render_edges(tmp_path: Path) -> None:
    da = _load()
    # _resolve_class_candidates dotted + module path branch.
    class_index = {"pkg.mod.Base": da.ClassInfo("pkg.mod.Base", "pkg.mod", [], set())}
    candidates = da._resolve_class_candidates(
        "Base",
        module="pkg.mod",
        symbol_table=None,
        class_index=class_index,
    )
    assert "pkg.mod.Base" in candidates

    # _resolve_method_in_hierarchy unresolved return path.
    unresolved = da._resolve_method_in_hierarchy(
        "pkg.mod.Base",
        "missing",
        class_index=class_index,
        by_qual={},
        symbol_table=None,
        seen=set(),
    )
    assert unresolved is None

    # _resolve_callee len(parts) == 2 branch.
    caller = da.FunctionInfo(
        name="caller",
        qual="pkg.mod.caller",
        path=tmp_path / "mod.py",
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
    )
    target = da.FunctionInfo(
        name="m",
        qual="mod.Base.m",
        path=tmp_path / "mod.py",
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        class_name="Base",
    )
    resolved = da._resolve_callee(
        "Base.m",
        caller,
        by_name={"m": [target]},
        by_qual={target.qual: target},
        symbol_table=da.SymbolTable(),
        project_root=tmp_path,
        class_index={"mod.Base": da.ClassInfo("mod.Base", "mod", [], {"m"})},
    )
    assert resolved is target

    # _infer_type_flow changed=True branch and constant smell site suffix.
    callee = da.FunctionInfo(
        name="callee",
        qual="pkg.mod.callee",
        path=tmp_path / "mod.py",
        params=["p"],
        annots={"p": "int"},
        calls=[],
        unused_params=set(),
    )
    caller_flow = da.FunctionInfo(
        name="caller",
        qual="pkg.mod.caller_flow",
        path=tmp_path / "mod.py",
        params=["p"],
        annots={"p": "Any"},
        calls=[],
        unused_params=set(),
    )
    edge = da._ResolvedCallEdge(
        caller=caller_flow,
        callee=callee,
        call=da.CallArgs(
            callee="callee",
            pos_map={"0": "p"},
            kw_map={},
            const_pos={},
            const_kw={},
            non_const_pos=set(),
            non_const_kw=set(),
            star_pos=[],
            star_kw=[],
            is_test=False,
            span=(0, 0, 0, 1),
        ),
    )
    index = da.AnalysisIndex(
        by_name={"caller_flow": [caller_flow], "callee": [callee]},
        by_qual={caller_flow.qual: caller_flow, callee.qual: callee},
        symbol_table=da.SymbolTable(),
        class_index={},
        resolved_transparent_edges_by_caller={caller_flow.qual: (edge,)},
    )
    _inferred, suggestions, _ambiguities, _evidence = da._infer_type_flow(
        [tmp_path / "mod.py"],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        parse_failure_witnesses=[],
        analysis_index=index,
    )
    assert suggestions

    smells = da._constant_smells_from_details(
        [
            da.ConstantFlowDetail(
                path=tmp_path / "mod.py",
                qual="pkg.mod.caller_flow",
                name="caller_flow",
                param="p",
                value="1",
                count=2,
                sites=("a.py:1:2: loc",),
            )
        ]
    )
    assert "(e.g." in smells[0]

    # _render_type_mermaid item filtering branch.
    mermaid = da._render_type_mermaid(
        [],
        ["f downstream types conflict: ['int', '', 'str']"],
    )
    assert "int" in mermaid and "str" in mermaid


def test_scope_normalization_and_timeout_cleanup_edges(tmp_path: Path) -> None:
    da = _load()
    assert da._normalize_transparent_decorators(["  a,b  "]) == {"a", "b"}

    # _analysis_deadline_scope with default tick_limit path.
    args = argparse.Namespace(
        analysis_timeout_ticks=10,
        analysis_timeout_tick_ns=1_000_000,
        analysis_tick_limit=None,
    )
    with da._analysis_deadline_scope(args):
        da.check_deadline()

    # analyze_paths timeout cleanup should still flush best-effort emitters.
    timed_out = False
    try:
        with da.deadline_scope(da.Deadline.from_timeout_ticks(1, 1)):
            with da.deadline_clock_scope(da.GasMeter(limit=1)):
                da.analyze_paths(
                    [tmp_path / "missing.py"],
                    forest=da.Forest(),
                    recursive=True,
                    type_audit=False,
                    type_audit_report=False,
                    type_audit_max=10,
                    include_constant_smells=False,
                    include_unused_arg_smells=False,
                    include_bundle_forest=False,
                    include_lint_lines=False,
                    config=da.AuditConfig(project_root=tmp_path),
                )
    except da.TimeoutExceeded:
        timed_out = True
    assert timed_out is True


def test_additional_branch_edges_scalar_helpers() -> None:
    da = _load()
    # _invariant_term outer conditional false path.
    assert da._invariant_term(ast.parse("42").body[0].value, {"x"}) is None

    # _parameter_default_map re-enters default loop after first element.
    fn = ast.parse("def f(a=1, b=2, c=3):\n    return a\n").body[0]
    mapping = da._parameter_default_map(fn)
    assert set(mapping) == {"a", "b", "c"}

    # _parse_lint_location range marker branch with no numeric range match.
    parsed = da._parse_lint_location("a.py:1:2:-x trailing")
    assert parsed is not None

    # _normalize_transparent_decorators non-iterable branch.
    assert da._normalize_transparent_decorators(123) is None


def test_additional_branch_edges_fingerprint_and_rewrite() -> None:
    da = _load()
    path = Path("pkg/mod.py")
    groups = {path: {"f": [{"x"}]}}
    annots = {path: {"f": {"x": "int"}}}
    registry = da.PrimeRegistry()
    fp = da.bundle_fingerprint_dimensional(["int"], registry, None)

    # _compute_fingerprint_warnings: names present path skips warning.
    assert (
        da._compute_fingerprint_warnings(
            groups,
            annots,
            registry=registry,
            index={fp: {"Ctx"}},
        )
        == []
    )

    # _compute_fingerprint_provenance: branch where index is empty.
    assert da._compute_fingerprint_provenance(
        groups,
        annots,
        registry=registry,
        index={},
    )

    # _summarize_fingerprint_provenance: key/type branch.
    lines = da._summarize_fingerprint_provenance(
        [
            {
                "path": "pkg/mod.py",
                "function": "f",
                "bundle": ["x"],
                "base_keys": ["int"],
                "ctor_keys": [],
                "glossary_matches": [],
            }
        ]
    )
    assert any("base=" in line for line in lines)

    # _compute_fingerprint_rewrite_plans branch when coherence entry missing.
    plans = da._compute_fingerprint_rewrite_plans(
        [
            {
                "path": "pkg/mod.py",
                "function": "f",
                "bundle": ["x"],
                "glossary_matches": ["A", "B"],
                "base_keys": ["int"],
                "ctor_keys": [],
                "remainder": {"base": 0, "ctor": 0},
            }
        ],
        coherence=[],
        synth_version="synth@1",
    )
    assert plans

    # verify_rewrite_plan verification non-dict branch.
    plan = dict(plans[0])
    plan["verification"] = []
    assert "accepted" in da.verify_rewrite_plan(plan, post_provenance=plan["site"] and [])

    # _branch_reachability_under_env: node not in body/orelse branch.
    tree = ast.parse("if cond:\n    x = 1\nelse:\n    y = 2\n")
    parent = da.ParentAnnotator()
    parent.visit(tree)
    if_node = tree.body[0]
    assert da._branch_reachability_under_env(if_node.test, parent.parents, {"cond": True}) is None


def test_additional_branch_edges_flow_and_render(tmp_path: Path) -> None:
    da = _load()
    # _DeadlineFunctionCollector elif non-name/non-attribute branch.
    fn = ast.parse("def f(deadline):\n    (lambda x: x)(deadline)\n").body[0]
    collector = da._DeadlineFunctionCollector(fn, {"deadline"})
    collector.visit(fn)
    assert collector.check_params == set()

    # _collect_deadline_local_info unknown alias source branch.
    assignments = [
        (
            [ast.Name(id="x", ctx=ast.Store())],
            ast.Name(id="unknown", ctx=ast.Load()),
            (0, 0, 0, 1),
        )
    ]
    info = da._collect_deadline_local_info(assignments, {"deadline"})
    assert "x" not in info.alias_to_param

    # _collect_call_resolution_obligations_from_forest non-list span branch.
    forest = da.Forest()
    suite = forest.add_suite_site("a.py", "pkg.f", "call", span=(1, 0, 1, 1))
    forest.nodes[suite].meta["span"] = (1, 0, 1, 1)
    forest.add_alt("CallResolutionObligation", (suite,), evidence={"callee": "c"})
    with pytest.raises(NeverThrown):
        da._collect_call_resolution_obligations_from_forest(forest)

    # _analysis_index_resolved_call_edges_by_caller require_transparent false path.
    index = da.AnalysisIndex(
        by_name={},
        by_qual={},
        symbol_table=da.SymbolTable(),
        class_index={},
        resolved_call_edges=(),
    )
    assert (
        da._analysis_index_resolved_call_edges_by_caller(
            index,
            project_root=None,
            require_transparent=False,
        )
        == {}
    )

    # _collect_bundle_evidence_lines branch when first component yields evidence.
    site_a = forest.add_site("a.py", "f")
    bundle_a = forest.add_paramset(["x", "y"])
    forest.add_alt("SignatureBundle", (site_a, bundle_a))
    site_b = forest.add_site("b.py", "g")
    bundle_b = forest.add_paramset(["u", "v"])
    forest.add_alt("SignatureBundle", (site_b, bundle_b))
    groups = {Path("a.py"): {"f": [{"x", "y"}]}, Path("b.py"): {"g": [{"u", "v"}]}}
    bundle_sites = {
        Path("a.py"): {
            "f": [[{"span": [0, 0, 0, 1], "callee": "t", "params": ["x"], "slots": ["x"]}]]
        },
        Path("b.py"): {
            "g": [[{"span": [0, 0, 0, 1], "callee": "t", "params": ["u"], "slots": ["u"]}]]
        },
    }
    assert da._collect_bundle_evidence_lines(
        forest=forest,
        groups_by_path=groups,
        bundle_sites_by_path=bundle_sites,
    )

    # _emit_report type_suggestions branch.
    report = da.ReportCarrier(
        forest=forest,
        type_suggestions=["a.py:f.p can tighten to int"],
        type_ambiguities=[],
    )
    markdown, _ = da._emit_report({}, 1, report=report)
    assert "Type tightening candidates:" in markdown

    # _analysis_deadline_scope exceptional exit path.
    args = argparse.Namespace(
        analysis_timeout_ticks=10,
        analysis_timeout_tick_ns=1_000,
        analysis_tick_limit=5,
    )
    with pytest.raises(RuntimeError):
        with da._analysis_deadline_scope(args):
            raise RuntimeError("boom")


def test_additional_branch_edges_reporting_and_exports(tmp_path: Path) -> None:
    da = _load()
    # _summarize_runtime_obligations with empty detail branch.
    lines = da._summarize_runtime_obligations(
        [{"status": "S", "contract": "c", "kind": "k", "detail": ""}]
    )
    assert lines and "detail=" not in lines[0]

    # _collect_module_exports assignment/augassign value branches.
    tree = ast.parse(
        "__all__ = ['a']\n"
        "__all__ += ['b']\n"
        "a = 1\n"
        "b = 2\n"
    )
    exports, _ = da._collect_module_exports(tree, module_name="pkg.mod", import_map={})
    assert {"a", "b"} <= exports

    # _emit_report type_suggestions disabled branch.
    report = da.ReportCarrier(
        forest=da.Forest(),
        type_suggestions=[],
        type_ambiguities=["a.py:f.p downstream types conflict: ['int']"],
    )
    markdown, _ = da._emit_report({}, 1, report=report)
    assert "Type ambiguities" in markdown


def test_additional_branch_edges_structure_materialization(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text("def f(x):\n    return x\n", encoding="utf-8")
    forest = da.Forest()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    da._materialize_structured_suite_sites_for_tree(
        forest=forest,
        path=path,
        tree=tree,
        project_root=tmp_path,
    )

    # Function with no lineno => function_span is None branch.
    synthetic_fn = ast.FunctionDef(
        name="g",
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[ast.Pass()],
        decorator_list=[],
        returns=None,
        type_comment=None,
    )
    synthetic_tree = ast.Module(body=[synthetic_fn], type_ignores=[])
    da._materialize_structured_suite_sites_for_tree(
        forest=forest,
        path=path,
        tree=synthetic_tree,
        project_root=tmp_path,
    )
    assert any(node.kind == "SuiteSite" for node in forest.nodes.values())

    groups = {path: {"f": [{"x", "y"}]}}
    da._populate_bundle_forest(
        forest,
        groups_by_path=groups,
        file_paths=[path],
        project_root=tmp_path,
        include_all_sites=False,
        parse_failure_witnesses=[],
    )
    assert forest.alts


def test_additional_branch_edges_class_and_call_resolution(tmp_path: Path) -> None:
    da = _load()
    # _accumulate_class_index_for_tree base_name false path.
    class_index: dict[str, da.ClassInfo] = {}
    tree = ast.parse("class C(42):\n    pass\n")
    da._accumulate_class_index_for_tree(
        class_index,
        Path("mod.py"),
        tree,
        project_root=Path("."),
    )
    assert class_index

    # _resolve_class_candidates resolved_head branch.
    symbol_table = da.SymbolTable(imports={("pkg.mod", "pkg"): "pkg"}, internal_roots={"pkg"})
    candidates = da._resolve_class_candidates(
        "pkg.Base",
        module="pkg.mod",
        symbol_table=symbol_table,
        class_index={"pkg.Base": da.ClassInfo("pkg.Base", "pkg", [], set())},
    )
    assert "pkg.Base" in candidates

    # _resolve_method_in_hierarchy resolved recursion branch.
    by_qual = {
        "mod.Base.m": da.FunctionInfo(
            name="m",
            qual="mod.Base.m",
            path=tmp_path / "mod.py",
            params=[],
            annots={},
            calls=[],
            unused_params=set(),
            class_name="Base",
        )
    }
    resolved = da._resolve_method_in_hierarchy(
        "mod.Child",
        "m",
        class_index={
            "mod.Base": da.ClassInfo("mod.Base", "mod", [], {"m"}),
            "mod.Child": da.ClassInfo("mod.Child", "mod", ["Base"], set()),
        },
        by_qual=by_qual,
        symbol_table=da.SymbolTable(),
        seen=set(),
    )
    assert resolved is not None

    # _resolve_callee candidate-in-by_qual branch.
    caller = da.FunctionInfo(
        name="caller",
        qual="mod.Caller.caller",
        path=tmp_path / "mod.py",
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        class_name="Caller",
    )
    target = da.FunctionInfo(
        name="m",
        qual="mod.Caller.m",
        path=tmp_path / "mod.py",
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        class_name="Caller",
    )
    assert da._resolve_callee(
        "self.m",
        caller,
        by_name={"m": [target]},
        by_qual={target.qual: target},
        symbol_table=da.SymbolTable(),
        project_root=tmp_path,
        class_index={"mod.Caller": da.ClassInfo("mod.Caller", "mod", [], {"m"})},
    ) is target


def test_additional_branch_edges_flow_and_registry(tmp_path: Path) -> None:
    da = _load()
    # _infer_type_flow changed-loop branch.
    callee = da.FunctionInfo(
        name="callee",
        qual="pkg.mod.callee",
        path=tmp_path / "mod.py",
        params=["p"],
        annots={"p": "int"},
        calls=[],
        unused_params=set(),
    )
    caller = da.FunctionInfo(
        name="caller",
        qual="pkg.mod.caller",
        path=tmp_path / "mod.py",
        params=["p"],
        annots={"p": "Any"},
        calls=[],
        unused_params=set(),
    )
    edge = da._ResolvedCallEdge(
        caller=caller,
        callee=callee,
        call=da.CallArgs(
            callee="callee",
            pos_map={"0": "p"},
            kw_map={},
            const_pos={},
            const_kw={},
            non_const_pos=set(),
            non_const_kw=set(),
            star_pos=[],
            star_kw=[],
            is_test=False,
            span=None,
        ),
    )
    index = da.AnalysisIndex(
        by_name={"caller": [caller], "callee": [callee]},
        by_qual={caller.qual: caller, callee.qual: callee},
        symbol_table=da.SymbolTable(),
        class_index={},
        resolved_transparent_edges_by_caller={caller.qual: (edge,)},
    )
    _inferred, suggestions, _ambig, _ev = da._infer_type_flow(
        [tmp_path / "mod.py"],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        parse_failure_witnesses=[],
        analysis_index=index,
    )
    assert suggestions

    # _constant_smells_from_details site suffix false path.
    smells = da._constant_smells_from_details(
        [
            da.ConstantFlowDetail(
                path=tmp_path / "mod.py",
                qual="q",
                name="f",
                param="p",
                value="1",
                count=1,
                sites=(),
            )
        ]
    )
    assert "(e.g." not in smells[0]

    # _iter_config_fields / _dataclass_registry_for_tree non-name assign target branches.
    config_tree = ast.parse("class C:\n    x, y = (1, 2)\n")
    assert da._iter_config_fields(
        Path("m.py"),
        tree=config_tree,
        parse_failure_witnesses=[],
    ) == {}
    dataclass_tree = ast.parse(
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class D:\n"
        "    a, b = (1, 2)\n"
    )
    assert da._dataclass_registry_for_tree(
        Path("m.py"),
        dataclass_tree,
        project_root=Path("."),
    ) == {}

    # _iter_dataclass_call_bundles candidate branch.
    mod = tmp_path / "mod.py"
    mod.write_text(
        "def make(alias):\n"
        "    alias.Item(1)\n",
        encoding="utf-8",
    )
    symbol_table = da.SymbolTable(
        imports={("mod", "alias"): "pkg"},
        internal_roots={"pkg"},
        external_filter=True,
        star_imports={"mod": {"pkg"}},
    )
    assert da._iter_dataclass_call_bundles(
        mod,
        project_root=tmp_path,
        symbol_table=symbol_table,
        dataclass_registry={"pkg.Item": ["x", "y"]},
        parse_failure_witnesses=[],
    ) == set()

    # _paramset_key non-list params metadata branch.
    forest = da.Forest()
    paramset = forest.add_paramset(["x", "y"])
    forest.nodes[paramset].meta["params"] = "x,y"
    assert da._paramset_key(forest, paramset) == tuple(str(p) for p in paramset.key)


def test_additional_branch_edges_rendering_variants() -> None:
    da = _load()
    # compute_structure_reuse basic branch.
    snapshot = {
        "files": [
            {"path": "a.py", "functions": [{"name": "f", "bundles": [["x", "y"]]}]},
            {"path": "b.py", "functions": [{"name": "g", "bundles": [["x", "y"]]}]},
        ]
    }
    reuse = da.compute_structure_reuse(snapshot)
    assert reuse["suggested_lemmas"]

    # render_reuse_lemma_stubs value-none branch.
    stubs = da.render_reuse_lemma_stubs(
        {"suggested_lemmas": [{"kind": "bundle", "suggested_name": "lemma", "count": 1}]}
    )
    assert "def lemma" in stubs

    # render_refactor_plan empty order branch.
    plan_text = da.render_refactor_plan({"bundles": [{"bundle": ["x"], "order": [], "cycles": []}]})
    assert "Bundle: x" in plan_text

    # _render_type_mermaid bracket-strip branch.
    mermaid = da._render_type_mermaid([], ["f downstream types conflict: ['int', 'str']"])
    assert "int" in mermaid and "str" in mermaid


def test_branch_shifted_lint_and_projection_edges(tmp_path: Path) -> None:
    da = _load()
    # _span_line_col helper + _deadline_lint_lines invalid span shape.
    assert da._span_line_col([1, 2, 3, 4]) == (2, 3)
    assert da._span_line_col([1, 2, 3]) == (None, None)
    lint_lines = da._deadline_lint_lines(
        [
            {
                "site": {"path": "a.py"},
                "span": {"bad": 1},
                "status": "UNKNOWN",
                "kind": "k",
                "detail": "d",
            }
        ]
    )
    assert lint_lines and "a.py:1:1" in lint_lines[0]

    # _lint_lines_from_call_ambiguities invalid span shape branch.
    ambiguity_lines = da._lint_lines_from_call_ambiguities(
        [
            {
                "site": {"path": "b.py", "span": (0, 0, 0, 0)},
                "candidate_count": "x",
                "kind": "ambiguous",
            }
        ]
    )
    assert ambiguity_lines and "b.py:1:1" in ambiguity_lines[0]

    # _build_module_artifacts parse-cache hit branch for duplicate paths.
    source = tmp_path / "m.py"
    source.write_text("x = 1\n", encoding="utf-8")
    parse_calls = 0

    def _parse(path: Path) -> ast.Module:
        nonlocal parse_calls
        parse_calls += 1
        return ast.parse(path.read_text(encoding="utf-8"))

    artifacts = da._build_module_artifacts(
        [source, source],
        specs=(
            da._ModuleArtifactSpec[list[str], tuple[str, ...]](
                artifact_id="mods",
                stage="scan",
                init=list,
                fold=lambda acc, path, tree: acc.append(
                    f"{path.name}:{len(getattr(tree, 'body', []))}"
                ),
                finish=tuple,
            ),
        ),
        parse_failure_witnesses=[],
        parse_module=_parse,
    )
    assert parse_calls == 1
    assert artifacts == (("m.py:1", "m.py:1"),)


def test_branch_shifted_rewrite_and_resolution_edges(tmp_path: Path) -> None:
    da = _load()
    # verify_rewrite_plan non-dict verification payload (normalized fallback).
    plan = {
        "plan_id": "p",
        "site": {"path": "a.py", "function": "f", "bundle": ["x"]},
        "pre": {"base_keys": ["int"], "ctor_keys": [], "remainder": {"base": 0, "ctor": 0}},
        "rewrite": {"parameters": {"candidates": ["Ctx"]}},
        "post_expectation": {"match_strata": "exact"},
        "verification": [1],
    }
    post = [
        {
            "path": "a.py",
            "function": "f",
            "bundle": ["x"],
            "base_keys": ["int"],
            "ctor_keys": [],
            "glossary_matches": ["Ctx"],
            "remainder": {"base": 0, "ctor": 0},
        }
    ]
    verified = da.verify_rewrite_plan(plan, post_provenance=post)
    assert verified["accepted"] in {True, False}
    assert verified["predicate_results"]

    # _bind_call_args unknown keyword when callee has no **kwargs.
    call = ast.parse("f(x=1)").body[0].value
    assert isinstance(call, ast.Call)
    callee = da.FunctionInfo(
        name="f",
        qual="pkg.f",
        path=tmp_path / "m.py",
        params=["a"],
        annots={},
        calls=[],
        unused_params=set(),
        positional_params=("a",),
        kwonly_params=(),
        vararg=None,
        kwarg=None,
    )
    assert da._bind_call_args(call, callee, strictness="high") == {}

    # _classify_deadline_expr name unmatched in alias/origin maps.
    info = da._classify_deadline_expr(
        ast.Name(id="mystery", ctx=ast.Load()),
        alias_to_param={},
        origin_vars=set(),
    )
    assert info.kind == "unknown"

    # _deadline_loop_forwarded_params false branch for param not in caller set.
    forwarded = da._deadline_loop_forwarded_params(
        qual="pkg.caller",
        loop_fact=da._DeadlineLoopFacts(
            span=None,
            kind="loop",
            call_spans={(1, 0, 1, 1)},
        ),
        deadline_params={"pkg.caller": {"deadline"}, "pkg.callee": {"deadline"}},
        call_infos={
            "pkg.caller": [
                (
                    da.CallArgs(
                        callee="callee",
                        pos_map={},
                        kw_map={},
                        const_pos={},
                        const_kw={},
                        non_const_pos=set(),
                        non_const_kw=set(),
                        star_pos=[],
                        star_kw=[],
                        is_test=False,
                        span=(1, 0, 1, 1),
                    ),
                    da.FunctionInfo(
                        name="callee",
                        qual="pkg.callee",
                        path=tmp_path / "m.py",
                        params=["deadline"],
                        annots={},
                        calls=[],
                        unused_params=set(),
                    ),
                    {"deadline": da._DeadlineArgInfo(kind="param", param="other")},
                )
            ]
        },
    )
    assert forwarded == set()

    # _resolve_class_candidates dotted-head unresolved branch.
    candidates = da._resolve_class_candidates(
        "alias.Type",
        module="pkg.mod",
        symbol_table=da.SymbolTable(
            imports={("pkg.mod", "alias"): "external.alias"},
            internal_roots={"pkg"},
            external_filter=True,
        ),
        class_index={
            "pkg.mod.alias.Type": da.ClassInfo("pkg.mod.alias.Type", "pkg.mod", [], set())
        },
    )
    assert "pkg.mod.alias.Type" in candidates

    # _resolve_method_in_hierarchy recursion returns None branch.
    assert (
        da._resolve_method_in_hierarchy(
            "pkg.Child",
            "m",
            class_index={
                "pkg.Child": da.ClassInfo("pkg.Child", "pkg", ["Base"], set()),
                "pkg.Base": da.ClassInfo("pkg.Base", "pkg", [], set()),
            },
            by_qual={},
            symbol_table=da.SymbolTable(),
            seen=set(),
        )
        is None
    )

    # _resolve_callee self.method candidate missing in by_qual branch.
    caller = da.FunctionInfo(
        name="caller",
        qual="pkg.Mod.caller",
        path=tmp_path / "m.py",
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        class_name="Mod",
    )
    assert (
        da._resolve_callee(
            "self.missing",
            caller,
            by_name={},
            by_qual={},
            symbol_table=da.SymbolTable(),
            project_root=tmp_path,
            class_index={"pkg.Mod": da.ClassInfo("pkg.Mod", "pkg", [], set())},
        )
        is None
    )

    # _compute_fingerprint_warnings missing-glossary warning branch.
    registry = da.PrimeRegistry()
    bundle_fp = da.bundle_fingerprint_dimensional(["int"], registry, None)
    other_fp = da.bundle_fingerprint_dimensional(["str"], registry, None)
    warnings = da._compute_fingerprint_warnings(
        {Path("mod.py"): {"f": [{"x"}]}},
        {Path("mod.py"): {"f": {"x": "int"}}},
        registry=registry,
        index={other_fp: {"Ctx"}},
    )
    assert warnings and "missing glossary match" in warnings[0]


def test_branch_shifted_flow_and_obligation_edges(tmp_path: Path) -> None:
    da = _load()
    mod = tmp_path / "mod.py"
    mod.write_text(
        "def f(flag):\n"
        "    if False:\n"
        "        raise ValueError(flag)\n",
        encoding="utf-8",
    )
    path_value = da._normalize_snapshot_path(mod, tmp_path)
    obligations = da._collect_exception_obligations(
        [mod],
        project_root=tmp_path,
        ignore_params=set(),
        handledness_witnesses=[],
        deadness_witnesses=[
            {
                "path": path_value,
                "function": "f",
                "bundle": ["flag"],
                "environment": {"flag": "False"},
                "deadness_id": "dead:f",
            }
        ],
        never_exceptions=set(),
    )
    assert obligations and obligations[0]["status"] in {"UNKNOWN", "DEAD"}

    logical_path = Path("pkg/core.py")

    # _compute_knob_param_names const(None) branch.
    callee = da.FunctionInfo(
        name="callee",
        qual="pkg.callee",
        path=logical_path,
        params=["p"],
        annots={},
        calls=[],
        unused_params=set(),
        defaults={"p"},
    )
    caller = da.FunctionInfo(
        name="caller",
        qual="pkg.caller",
        path=logical_path,
        params=["x"],
        annots={},
        calls=[],
        unused_params=set(),
    )
    edge = da._ResolvedCallEdge(
        caller=caller,
        callee=callee,
        call=da.CallArgs(
            callee="callee",
            pos_map={},
            kw_map={},
            const_pos={"0": None},  # type: ignore[dict-item]
            const_kw={},
            non_const_pos=set(),
            non_const_kw=set(),
            star_pos=[],
            star_kw=[],
            is_test=False,
            span=None,
        ),
    )
    caller.calls.append(edge.call)
    index = da.AnalysisIndex(
        by_name={"caller": [caller], "callee": [callee]},
        by_qual={caller.qual: caller, callee.qual: callee},
        symbol_table=da.SymbolTable(),
        class_index={},
        resolved_transparent_edges_by_caller={caller.qual: (edge,)},
    )
    assert (
        da._compute_knob_param_names(
            by_name=index.by_name,
            by_qual=index.by_qual,
            symbol_table=index.symbol_table,
            project_root=tmp_path,
            class_index=index.class_index,
            strictness="high",
            analysis_index=index,
        )
        == set()
    )

    # _collect_constant_flow_details const event with countable=False branch.
    def _iter_events(
        _edge: da._ResolvedCallEdge, *, strictness: str, include_variadics_in_low_star: bool
    ) -> list[da._ResolvedEdgeParamEvent]:
        del strictness, include_variadics_in_low_star
        return [da._ResolvedEdgeParamEvent(kind="const", param="p", value="1", countable=False)]

    def _reduce(
        _index: da.AnalysisIndex,
        *,
        project_root: Path | None,
        require_transparent: bool,
        spec: da._ResolvedEdgeReducerSpec[da._ConstantFlowFoldAccumulator, da._ConstantFlowFoldAccumulator],
    ) -> da._ConstantFlowFoldAccumulator:
        del project_root, require_transparent
        acc = spec.init()
        spec.fold(acc, edge)
        return spec.finish(acc)

    details = da._collect_constant_flow_details(
        [mod],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        parse_failure_witnesses=[],
        analysis_index=index,
        iter_resolved_edge_param_events_fn=_iter_events,
        reduce_resolved_call_edges_fn=_reduce,
    )
    assert details and details[0].count == 0

    # _infer_type_flow no-op update branch (existing inferred annotation matches downstream).
    callee_any = da.FunctionInfo(
        name="callee_any",
        qual="pkg.callee_any",
        path=logical_path,
        params=["p"],
        annots={"p": "Any"},
        calls=[],
        unused_params=set(),
        transparent=True,
    )
    caller_any_call = da.CallArgs(
        callee="callee_any",
        pos_map={"0": "p"},
        kw_map={},
        const_pos={},
        const_kw={},
        non_const_pos=set(),
        non_const_kw=set(),
        star_pos=[],
        star_kw=[],
        is_test=False,
        span=(0, 0, 0, 1),
    )
    caller_any = da.FunctionInfo(
        name="caller_any",
        qual="pkg.caller_any",
        path=logical_path,
        params=["p"],
        annots={"p": "Any"},
        calls=[],
        unused_params=set(),
    )
    index_any = da.AnalysisIndex(
        by_name={"caller_any": [caller_any], "callee_any": [callee_any]},
        by_qual={caller_any.qual: caller_any, callee_any.qual: callee_any},
        symbol_table=da.SymbolTable(),
        class_index={},
        resolved_transparent_edges_by_caller={
            caller_any.qual: (
                da._ResolvedCallEdge(
                    caller=caller_any,
                    callee=callee_any,
                    call=caller_any_call,
                ),
            )
        },
    )
    inferred_any, suggestions_any, _amb_any, _ev_any = da._infer_type_flow(
        [logical_path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        parse_failure_witnesses=[],
        analysis_index=index_any,
    )
    assert inferred_any[caller_any.qual]["p"] == "Any"
    assert any("caller_any.p" in entry for entry in suggestions_any)

    # _analyze_unused_arg_flow_indexed call-span missing + non-const unused negative branches.
    logical_path = Path("pkg/mod.py")
    unused_call = da.CallArgs(
        callee="target",
        pos_map={"0": "x"},
        kw_map={},
        const_pos={},
        const_kw={},
        non_const_pos={"1"},
        non_const_kw={"v"},
        star_pos=[],
        star_kw=[],
        is_test=False,
        span=None,
    )
    callee_unused = da.FunctionInfo(
        name="target",
        qual="pkg.target",
        path=logical_path,
        params=["u", "v"],
        annots={},
        calls=[],
        unused_params={"u"},
        transparent=True,
    )
    caller_unused = da.FunctionInfo(
        name="caller_unused",
        qual="pkg.caller_unused",
        path=logical_path,
        params=["x"],
        annots={},
        calls=[unused_call],
        unused_params=set(),
    )
    unused_edge = da._ResolvedCallEdge(
        caller=caller_unused,
        callee=callee_unused,
        call=unused_call,
    )
    unused_index = da.AnalysisIndex(
        by_name={"caller_unused": [caller_unused], "target": [callee_unused]},
        by_qual={caller_unused.qual: caller_unused, callee_unused.qual: callee_unused},
        symbol_table=da.SymbolTable(),
        class_index={},
        resolved_transparent_edges_by_caller={caller_unused.qual: (unused_edge,)},
    )
    smells = da._analyze_unused_arg_flow_indexed(
        da._IndexedPassContext(
            paths=[logical_path],
            project_root=tmp_path,
            ignore_params=set(),
            strictness="high",
            external_filter=True,
            transparent_decorators=None,
            parse_failure_witnesses=[],
            analysis_index=unused_index,
        )
    )
    assert any(":caller_unused passes param x" in entry for entry in smells)


def test_branch_shifted_exports_refactor_and_scope_edges(tmp_path: Path) -> None:
    da = _load()
    # _collect_module_exports __all__ value parse miss branches.
    tree = ast.parse(
        "__all__: list[str] = 0\n"
        "__all__ += 0\n"
        "def keep():\n"
        "    return 1\n"
    )
    exports, export_map = da._collect_module_exports(tree, module_name="pkg.mod", import_map={})
    assert "keep" in exports
    assert export_map.get("keep") == "pkg.mod.keep"

    # _collect_config_bundles assign-target non-name branch.
    config_mod = tmp_path / "cfg.py"
    config_mod.write_text(
        "class AppConfig:\n"
        "    x, y = (1, 2)\n",
        encoding="utf-8",
    )
    assert (
        da._collect_config_bundles(
            [config_mod],
            parse_failure_witnesses=[],
        )
        == {}
    )

    # _iter_dataclass_call_bundles star-import candidate miss branch.
    call_mod = tmp_path / "calls.py"
    call_mod.write_text("def f(alias):\n    alias.Item(1)\n", encoding="utf-8")
    bundles = da._iter_dataclass_call_bundles(
        call_mod,
        project_root=tmp_path,
        symbol_table=da.SymbolTable(
            imports={("calls", "alias"): "pkg"},
            internal_roots={"pkg"},
            external_filter=True,
            star_imports={"calls": {"pkg"}},
            module_exports={"pkg": {"alias"}},
        ),
        dataclass_registry={},
        parse_failure_witnesses=[],
    )
    assert bundles == set()

    # build_refactor_plan info-miss branch.
    ref_mod = tmp_path / "ref.py"
    ref_mod.write_text("def f(x):\n    return x\n", encoding="utf-8")
    plan = da.build_refactor_plan(
        {ref_mod: {"missing": [{"x"}]}},
        [ref_mod],
        config=da.AuditConfig(project_root=tmp_path),
    )
    assert any("No bundle components" in warning for warning in plan["warnings"])

    # _render_type_mermaid rhs without bracket wrapper.
    mermaid = da._render_type_mermaid([], ["f downstream types conflict: int, str"])
    assert "int" in mermaid and "str" in mermaid

    # analyze_paths timeout before collection-progress callback is defined.
    timed_out = False
    try:
        with da.deadline_scope(da.Deadline.from_timeout_ticks(10, 1)):
            with da.deadline_clock_scope(da.GasMeter(limit=2)):
                da.analyze_paths(
                    [tmp_path / "missing.py"],
                    forest=da.Forest(),
                    recursive=True,
                    type_audit=False,
                    type_audit_report=False,
                    type_audit_max=10,
                    include_constant_smells=False,
                    include_unused_arg_smells=False,
                    include_bundle_forest=False,
                    include_lint_lines=False,
                    config=da.AuditConfig(project_root=tmp_path),
                    collection_resume={"format_version": 0},
                )
    except da.TimeoutExceeded:
        timed_out = True
    assert timed_out is True

    # _analysis_deadline_scope with explicit tick-limit normal path.
    args = argparse.Namespace(
        analysis_timeout_ticks=10,
        analysis_timeout_tick_ns=1_000_000,
        analysis_tick_limit=3,
    )
    with da._analysis_deadline_scope(args):
        da.check_deadline()


def test_write_output_helpers_cover_stdout_and_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    da = _load()
    text_path = tmp_path / "out.txt"
    json_path = tmp_path / "out.json"

    da._write_text_or_stdout("-", "hello")
    assert capsys.readouterr().out == "hello\n"

    da._write_text_or_stdout(str(text_path), "world")
    assert text_path.read_text(encoding="utf-8") == "world"

    da._write_json_or_stdout("-", {"a": 1})
    stdout = capsys.readouterr().out
    assert '"a": 1' in stdout

    da._write_json_or_stdout(str(json_path), {"b": 2})
    assert '"b": 2' in json_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    (
        "kwargs",
        "include_type_audit",
        "include_tree",
        "include_metrics",
        "include_decision",
        "expected",
    ),
    [
        ({}, True, False, False, False, False),
        ({"type_audit": True}, True, False, False, False, True),
        ({"type_audit": True}, False, False, False, False, False),
        ({}, True, True, False, False, True),
        ({}, True, False, True, False, True),
        ({}, True, False, False, True, True),
    ],
)
def test_has_followup_actions_variants(
    kwargs: dict[str, object],
    include_type_audit: bool,
    include_tree: bool,
    include_metrics: bool,
    include_decision: bool,
    expected: bool,
) -> None:
    da = _load()
    payload: dict[str, object] = {
        "type_audit": False,
        "synthesis_plan": None,
        "synthesis_report": False,
        "synthesis_protocols": None,
        "refactor_plan": False,
        "refactor_plan_json": None,
    }
    payload.update(kwargs)
    args = argparse.Namespace(**payload)
    assert (
        da._has_followup_actions(
            args,
            include_type_audit=include_type_audit,
            include_structure_tree=include_tree,
            include_structure_metrics=include_metrics,
            include_decision_snapshot=include_decision,
        )
        is expected
    )


def test_emit_sidecar_outputs_dispatches_expected_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    da = _load()
    analysis = da.AnalysisResult(
        groups_by_path={},
        param_spans_by_path={},
        bundle_sites_by_path={},
        type_suggestions=[],
        type_ambiguities=[],
        type_callsite_evidence=[],
        constant_smells=[],
        unused_arg_smells=[],
        forest=da.Forest(),
        lint_lines=["L1"],
        deadness_witnesses=[{"kind": "dead"}],
        coherence_witnesses=[{"kind": "coh"}],
        rewrite_plans=[{"kind": "plan"}],
        exception_obligations=[{"kind": "exc"}],
        handledness_witnesses=[{"kind": "handled"}],
        fingerprint_synth_registry={"registry": 1},
        fingerprint_provenance=[{"kind": "prov"}],
    )
    synth_path = tmp_path / "synth.json"
    prov_path = tmp_path / "prov.json"

    args = argparse.Namespace(
        lint=True,
        fingerprint_synth_json=str(synth_path),
        fingerprint_provenance_json=str(prov_path),
    )
    da._emit_sidecar_outputs(
        args=args,
        analysis=analysis,
        fingerprint_deadness_json="-",
        fingerprint_coherence_json=None,
        fingerprint_rewrite_plans_json=None,
        fingerprint_exception_obligations_json=None,
        fingerprint_handledness_json=None,
    )
    out = capsys.readouterr().out
    assert "L1" in out
    assert '"registry": 1' in synth_path.read_text(encoding="utf-8")
    assert '"kind": "prov"' in prov_path.read_text(encoding="utf-8")
    assert '"kind": "dead"' in out

    # require_content gate: synth/provenance are skipped when payloads are empty.
    empty_analysis = da.AnalysisResult(
        groups_by_path={},
        param_spans_by_path={},
        bundle_sites_by_path={},
        type_suggestions=[],
        type_ambiguities=[],
        type_callsite_evidence=[],
        constant_smells=[],
        unused_arg_smells=[],
        forest=da.Forest(),
    )
    synth_skip = tmp_path / "synth_skip.json"
    prov_skip = tmp_path / "prov_skip.json"
    args_skip = argparse.Namespace(
        lint=False,
        fingerprint_synth_json=str(synth_skip),
        fingerprint_provenance_json=str(prov_skip),
    )
    da._emit_sidecar_outputs(
        args=args_skip,
        analysis=empty_analysis,
        fingerprint_deadness_json=None,
        fingerprint_coherence_json=None,
        fingerprint_rewrite_plans_json=None,
        fingerprint_exception_obligations_json=None,
        fingerprint_handledness_json=None,
    )
    assert not synth_skip.exists()
    assert not prov_skip.exists()
