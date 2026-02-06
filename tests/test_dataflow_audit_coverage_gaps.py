from __future__ import annotations

import ast
import os
import pytest
from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit as da

    return da


def test_deadness_and_coherence_summaries_cover_edges() -> None:
    da = _load()

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


def test_fingerprint_coherence_and_rewrite_plans_cover_edges() -> None:
    da = _load()
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


def test_exception_helpers_cover_edges() -> None:
    da = _load()

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


def test_exception_obligation_summary_helper_covers_filters_and_status_normalization() -> None:
    da = _load()
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


def test_exception_collection_and_summaries_cover_edges(tmp_path: Path) -> None:
    da = _load()

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


def test_never_helpers_and_sort_key_cover_edges() -> None:
    da = _load()
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


def test_exception_type_name_and_protocol_lint_edges() -> None:
    da = _load()
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


def test_collect_handledness_system_exit(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "sys_exit.py"
    path.write_text("def f():\n    raise SystemExit()\n")
    witnesses = da._collect_handledness_witnesses(
        [path], project_root=tmp_path, ignore_params=set()
    )
    assert any(entry.get("handler_kind") == "convert" for entry in witnesses)


# gabion:evidence E:never/sink_classification
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


# gabion:evidence E:never/sink_classification
def test_collect_never_invariants_skips_bad_syntax(tmp_path: Path) -> None:
    da = _load()
    bad = tmp_path / "bad.py"
    bad.write_text("def bad(:\n    pass\n")
    invariants = da._collect_never_invariants(
        [bad], project_root=tmp_path, ignore_params=set()
    )
    assert invariants == []


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

    with pytest.raises(RuntimeError):
        da._emit_report({path: {"f": [set(["x"])]}}, 3, forest=None)


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


def test_exception_protocol_warning_filters() -> None:
    da = _load()
    entries = [
        {"protocol": "never", "status": "DEAD", "site": {"path": "a.py", "function": "f"}},
        {"protocol": "never", "status": "FORBIDDEN", "site": {"path": "b.py", "function": "g"}, "exception_name": "Boom"},
    ]
    warnings = da._exception_protocol_warnings(entries)
    assert len(warnings) == 1
    assert "Boom" in warnings[0]


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
    )
    assert any(alt.kind == "SignatureBundle" for alt in forest.alts)


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
