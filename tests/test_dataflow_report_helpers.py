from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit as da

    return da


def _write(path: Path, content: str) -> None:
    path.write_text(content)


def _forest_for_groups(
    da, groups_by_path: dict[Path, dict[str, list[set[str]]]], project_root: Path
):
    forest = da.Forest()
    da._populate_bundle_forest(
        forest,
        groups_by_path=groups_by_path,
        file_paths=sorted(groups_by_path),
        project_root=project_root,
        parse_failure_witnesses=[],
    )
    return forest


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._render_component_callsite_evidence::bundle_counts E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::bundle_sites_by_path,coherence_witnesses,constant_smells,context_suggestions,deadness_witnesses,decision_surfaces,decision_warnings,exception_obligations,fingerprint_matches,fingerprint_provenance,fingerprint_synth,fingerprint_warnings,forest,groups_by_path,handledness_witnesses,invariant_propositions,max_components,never_invariants,rewrite_plans,type_ambiguities,type_callsite_evidence,type_suggestions,unused_arg_smells,value_decision_rewrites,value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._render_mermaid_component::component,declared_global,nodes E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_never_invariants::entries,include_proven_unreachable,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_coherence_witnesses::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_deadness_witnesses::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_exception_obligations::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_handledness_witnesses::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_rewrite_plans::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_fingerprint_provenance::entries,max_examples E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._bundle_projection_from_forest::file_paths
def test_emit_report_empty_groups() -> None:
    da = _load()
    report, violations = da._emit_report(
        {}, 3, report=da.ReportCarrier(forest=da.Forest())
    )
    assert "No bundle components detected." in report
    assert violations == []


def test_emit_report_parse_failure_witnesses() -> None:
    da = _load()
    report, violations = da._emit_report(
        {},
        3,
        report=da.ReportCarrier(
            forest=da.Forest(),
            parse_failure_witnesses=[
                {
                    "path": "bad.py",
                    "stage": "function_index",
                    "error_type": "SyntaxError",
                    "error": "invalid syntax",
                }
            ],
        ),
    )
    assert "Parse failure witnesses" in report
    assert "bad.py stage=function_index SyntaxError: invalid syntax" in report
    assert violations == [
        "bad.py parse_failure stage=function_index SyntaxError: invalid syntax"
    ]


def test_emit_report_adds_raw_sorted_contract_violations(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    _write(path, "def f(xs):\n    return sorted(xs)\n")
    groups_by_path = {path: {}}
    report, violations = da._emit_report(
        groups_by_path,
        3,
        report=da.ReportCarrier(forest=da.Forest()),
    )
    assert "Order contract violations" in report
    assert any("raw_sorted introduced" in line for line in violations)


def test_emit_report_raw_sorted_strict_forbid_env(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    _write(path, "def f(xs):\n    return sorted(xs)\n")
    groups_by_path = {path: {}}
    previous = da.os.environ.get("GABION_FORBID_RAW_SORTED")
    try:
        da.os.environ["GABION_FORBID_RAW_SORTED"] = "1"
        _, violations = da._emit_report(
            groups_by_path,
            3,
            report=da.ReportCarrier(forest=da.Forest()),
        )
    finally:
        if previous is None:
            da.os.environ.pop("GABION_FORBID_RAW_SORTED", None)
        else:
            da.os.environ["GABION_FORBID_RAW_SORTED"] = previous
    assert any("raw sorted() forbidden" in line for line in violations)


def test_emit_report_execution_pattern_suggestions_are_non_blocking() -> None:
    da = _load()
    report, violations = da._emit_report(
        {},
        3,
        report=da.ReportCarrier(forest=da.Forest()),
        execution_pattern_suggestions=[
            "execution_pattern indexed_pass_ingress members=3"
        ],
    )
    assert "Execution pattern opportunities" in report
    assert "indexed_pass_ingress" in report
    assert all("execution_pattern" not in line for line in violations)


def test_emit_report_adds_dataflow_pattern_schema_suggestions(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    _write(path, "def f(a, b):\n    return a\n\ndef g(a, b):\n    return b\n")
    groups_by_path = {path: {"f": [set(["a", "b"])], "g": [set(["a", "b"])]}}
    report, violations = da._emit_report(
        groups_by_path,
        3,
        report=da.ReportCarrier(
            forest=_forest_for_groups(da, groups_by_path, tmp_path),
        ),
    )
    assert "pattern_schema axis=dataflow" in report
    assert "bundle=a,b" in report
    assert all("pattern_schema" not in line for line in violations)


def test_emit_report_adds_pattern_schema_residue_non_blocking(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    _write(path, "def f(a, b):\n    return a\n\ndef g(a, b):\n    return b\n")
    groups_by_path = {path: {"f": [set(["a", "b"])], "g": [set(["a", "b"])]}}
    report, violations = da._emit_report(
        groups_by_path,
        3,
        report=da.ReportCarrier(
            forest=_forest_for_groups(da, groups_by_path, tmp_path),
        ),
    )
    assert "Pattern schema residue (non-blocking)" in report
    assert "reason=unreified_protocol" in report
    assert all("unreified_protocol" not in line for line in violations)


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._render_component_callsite_evidence::bundle_counts E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::bundle_sites_by_path,coherence_witnesses,constant_smells,context_suggestions,deadness_witnesses,decision_surfaces,decision_warnings,exception_obligations,fingerprint_matches,fingerprint_provenance,fingerprint_synth,fingerprint_warnings,forest,groups_by_path,handledness_witnesses,invariant_propositions,max_components,never_invariants,rewrite_plans,type_ambiguities,type_callsite_evidence,type_suggestions,unused_arg_smells,value_decision_rewrites,value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._render_mermaid_component::component,declared_global,nodes E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_never_invariants::entries,include_proven_unreachable,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_coherence_witnesses::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_deadness_witnesses::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_exception_obligations::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_handledness_witnesses::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_rewrite_plans::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_fingerprint_provenance::entries,max_examples E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._bundle_projection_from_forest::file_paths E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._populate_bundle_forest::groups_by_path
def test_emit_report_component_summary(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    extra_path = tmp_path / "other.py"
    _write(
        path,
        "from dataclasses import dataclass\n"
        "\n"
        "# dataflow-bundle: a, b\n"
        "@dataclass\n"
        "class AppConfig:\n"
        "    c: int\n"
        "    d: int\n"
        "\n"
        "def f(a, b, x, y):\n"
        "    return a\n",
    )
    _write(extra_path, "def g():\n    return 1\n")
    groups_by_path = {
        path: {"f": [set(["a", "b"]), set(["x", "y"])]},
        extra_path: {},
    }
    forest = _forest_for_groups(da, groups_by_path, tmp_path)
    report, violations = da._emit_report(
        groups_by_path,
        3,
        report=da.ReportCarrier(
            forest=forest,
            type_suggestions=["f.a can tighten to int"],
            type_ambiguities=["f.b downstream types conflict: ['int', 'str']"],
            constant_smells=[
                "mod.py:f.a only observed constant 1 across 1 non-test call(s)"
            ],
            unused_arg_smells=["mod.py:f passes param x to unused mod.py:f.x"],
            decision_surfaces=["mod.py:f decision surface params: a"],
            value_decision_surfaces=[
                "mod.py:f value-encoded decision params: a (min/max)"
            ],
            decision_warnings=[
                "mod.py:f decision param 'a' missing decision tier metadata"
            ],
            context_suggestions=[
                "Consider contextvar for mod.py:f decision surface params: a"
            ],
        ),
    )
    assert "Observed-only bundles" in report
    assert "Documented bundles" in report
    assert "Declared Config bundles not observed" in report
    assert "Type-flow audit" in report
    assert "Constant-propagation smells" in report
    assert "Unused-argument smells" in report
    assert "Decision surface candidates" in report
    assert "Value-encoded decision surface candidates" in report
    assert "Decision tier warnings" in report
    assert "Contextvar/ambient rewrite suggestions" in report
    assert any("tier-3" in line for line in violations)


# gabion:evidence E:call_cluster::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::test_dataflow_report_helpers.py::tests.test_dataflow_report_helpers._load::test_dataflow_report_helpers.py::tests.test_dataflow_report_helpers._write
def test_emit_report_fingerprint_provenance_summary(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    _write(path, "def f(a, b):\n    return a\n")
    groups_by_path = {path: {}}
    entries = [
        {
            "path": str(path),
            "function": "f",
            "bundle": ["a", "b"],
            "base_keys": ["int", "str"],
            "ctor_keys": [],
            "glossary_matches": ["user_context"],
        },
        {
            "path": str(path),
            "function": "g",
            "bundle": ["x", "y"],
            "base_keys": ["int", "str"],
            "ctor_keys": [],
            "glossary_matches": ["user_context"],
        },
        {
            "path": str(path),
            "function": "h",
            "bundle": ["c"],
            "base_keys": ["float"],
            "ctor_keys": ["list"],
            "glossary_matches": [],
        },
    ]
    report, _ = da._emit_report(
        groups_by_path,
        3,
        report=da.ReportCarrier(
            forest=da.Forest(),
            fingerprint_provenance=entries,
        ),
    )
    assert "Packed derivation view (ASPF provenance)" in report
    assert "glossary=user_context" in report
    assert "base=['float'] ctor=['list']" in report


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._render_component_callsite_evidence::bundle_counts E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::bundle_sites_by_path,coherence_witnesses,constant_smells,context_suggestions,deadness_witnesses,decision_surfaces,decision_warnings,exception_obligations,fingerprint_matches,fingerprint_provenance,fingerprint_synth,fingerprint_warnings,forest,groups_by_path,handledness_witnesses,invariant_propositions,max_components,never_invariants,rewrite_plans,type_ambiguities,type_callsite_evidence,type_suggestions,unused_arg_smells,value_decision_rewrites,value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._render_mermaid_component::component,declared_global,nodes E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_never_invariants::entries,include_proven_unreachable,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_coherence_witnesses::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_deadness_witnesses::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_exception_obligations::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_handledness_witnesses::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_rewrite_plans::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_fingerprint_provenance::entries,max_examples E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._bundle_projection_from_forest::file_paths E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._populate_bundle_forest::groups_by_path
def test_emit_report_fingerprint_matches_and_synth(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    _write(path, "def f(a, b):\n    return a\n")
    groups_by_path = {path: {"f": [set(["a", "b"])]}}
    forest = _forest_for_groups(da, groups_by_path, tmp_path)
    report, _ = da._emit_report(
        groups_by_path,
        3,
        report=da.ReportCarrier(
            forest=forest,
            fingerprint_matches=[
                "mod.py:f bundle ['a', 'b'] fingerprint {base=2} matches: user_context"
            ],
            fingerprint_synth=["synth registry synth@1:"],
            invariant_propositions=[
                da.InvariantProposition(
                    form="Equal",
                    terms=("a", "b"),
                    scope="mod.py:f",
                    source="assert",
                )
            ],
        ),
    )
    assert "Fingerprint matches" in report
    assert "Fingerprint synthesis" in report
    assert "Invariant propositions" in report


# gabion:evidence E:call_cluster::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::test_dataflow_report_helpers.py::tests.test_dataflow_report_helpers._load::test_dataflow_report_helpers.py::tests.test_dataflow_report_helpers._write
def test_emit_report_deadness_summary(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    _write(path, "def f(a):\n    return a\n")
    groups_by_path = {path: {}}
    deadness = [
        {
            "path": str(path),
            "function": "f",
            "bundle": ["a"],
            "environment": {"a": "1"},
            "predicate": "a != 1",
            "core": ["observed constant 1"],
            "result": "UNREACHABLE",
        }
    ]
    report, _ = da._emit_report(
        groups_by_path,
        3,
        report=da.ReportCarrier(
            forest=da.Forest(),
            deadness_witnesses=deadness,
        ),
    )
    assert "Deadness evidence" in report
    assert "UNREACHABLE" in report


# gabion:evidence E:call_cluster::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::test_dataflow_report_helpers.py::tests.test_dataflow_report_helpers._load::test_dataflow_report_helpers.py::tests.test_dataflow_report_helpers._write
def test_emit_report_coherence_summary(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    _write(path, "def f(a):\n    return a\n")
    groups_by_path = {path: {}}
    coherence = [
        {
            "site": {"path": str(path), "function": "f", "bundle": ["a"]},
            "alternatives": ["ctx_a", "ctx_b"],
            "fork_signature": "glossary-ambiguity",
            "result": "UNKNOWN",
        }
    ]
    report, _ = da._emit_report(
        groups_by_path,
        3,
        report=da.ReportCarrier(
            forest=da.Forest(),
            coherence_witnesses=coherence,
        ),
    )
    assert "Coherence evidence" in report
    assert "glossary-ambiguity" in report


# gabion:evidence E:call_cluster::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::test_dataflow_report_helpers.py::tests.test_dataflow_report_helpers._load::test_dataflow_report_helpers.py::tests.test_dataflow_report_helpers._write
def test_emit_report_rewrite_plan_summary(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    _write(path, "def f(a):\n    return a\n")
    groups_by_path = {path: {}}
    plans = [
        {
            "plan_id": "rewrite:mod.py:f:a",
            "status": "UNVERIFIED",
            "site": {"path": str(path), "function": "f", "bundle": ["a"]},
            "rewrite": {"kind": "BUNDLE_ALIGN"},
        }
    ]
    report, _ = da._emit_report(
        groups_by_path,
        3,
        report=da.ReportCarrier(
            forest=da.Forest(),
            rewrite_plans=plans,
        ),
    )
    assert "Rewrite plans" in report
    assert "BUNDLE_ALIGN" in report


# gabion:evidence E:call_cluster::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::test_dataflow_report_helpers.py::tests.test_dataflow_report_helpers._load::test_dataflow_report_helpers.py::tests.test_dataflow_report_helpers._write
def test_emit_report_exception_obligation_summary(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    _write(path, "def f(a):\n    return a\n")
    groups_by_path = {path: {}}
    obligations = [
        {
            "exception_path_id": "mod.py:f:E0:1:0:raise",
            "site": {"path": str(path), "function": "f", "bundle": ["a"]},
            "source_kind": "E0",
            "status": "UNKNOWN",
        }
    ]
    report, _ = da._emit_report(
        groups_by_path,
        3,
        report=da.ReportCarrier(
            forest=da.Forest(),
            exception_obligations=obligations,
        ),
    )
    assert "Exception obligations" in report
    assert "UNKNOWN" in report


# gabion:evidence E:call_cluster::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::test_dataflow_report_helpers.py::tests.test_dataflow_report_helpers._load::test_dataflow_report_helpers.py::tests.test_dataflow_report_helpers._write
def test_emit_report_handledness_summary(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    _write(path, "def f(a):\n    return a\n")
    groups_by_path = {path: {}}
    handled = [
        {
            "site": {"path": str(path), "function": "f", "bundle": ["a"]},
            "handler_boundary": "except Exception",
        }
    ]
    report, _ = da._emit_report(
        groups_by_path,
        3,
        report=da.ReportCarrier(
            forest=da.Forest(),
            handledness_witnesses=handled,
        ),
    )
    assert "Handledness evidence" in report
    assert "except Exception" in report


# gabion:evidence E:call_cluster::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::test_dataflow_report_helpers.py::tests.test_dataflow_report_helpers._load::test_dataflow_report_helpers.py::tests.test_dataflow_report_helpers._write
def test_emit_report_anonymous_schema_surfaces(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    _write(
        path,
        "from __future__ import annotations\n"
        "from typing import Any\n"
        "\n"
        "def f(payload: dict[str, Any]) -> None:\n"
        "    return None\n",
    )
    groups_by_path = {path: {}}
    report, _ = da._emit_report(
        groups_by_path, 3, report=da.ReportCarrier(forest=da.Forest())
    )
    assert "Anonymous schema surfaces" in report
    assert "dict[str, Any]" in report


# gabion:evidence E:call_cluster::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::test_dataflow_report_helpers.py::tests.test_dataflow_report_helpers._load::test_dataflow_report_helpers.py::tests.test_dataflow_report_helpers._write
def test_emit_report_anonymous_schema_surfaces_truncates_long_list(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    lines = ["from __future__ import annotations", ""]
    for idx in range(51):
        lines.append(f"x{idx}: dict[str, object] = {{}}")
    _write(path, "\n".join(lines) + "\n")
    report, _ = da._emit_report(
        {path: {}}, 3, report=da.ReportCarrier(forest=da.Forest())
    )
    assert "Anonymous schema surfaces" in report
    assert "... 1 more" in report


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._render_component_callsite_evidence::bundle_counts E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::bundle_sites_by_path,coherence_witnesses,constant_smells,context_suggestions,deadness_witnesses,decision_surfaces,decision_warnings,exception_obligations,fingerprint_matches,fingerprint_provenance,fingerprint_synth,fingerprint_warnings,forest,groups_by_path,handledness_witnesses,invariant_propositions,max_components,never_invariants,rewrite_plans,type_ambiguities,type_callsite_evidence,type_suggestions,unused_arg_smells,value_decision_rewrites,value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._render_mermaid_component::component,declared_global,nodes E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_never_invariants::entries,include_proven_unreachable,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_coherence_witnesses::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_deadness_witnesses::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_exception_obligations::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_handledness_witnesses::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_rewrite_plans::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_fingerprint_provenance::entries,max_examples E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._bundle_projection_from_forest::file_paths E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._populate_bundle_forest::groups_by_path
def test_emit_report_max_components_cutoff(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    _write(path, "def f(a, b):\n    return a\n")
    groups_by_path = {path: {"f": [set(["a", "b"])]}}
    forest = _forest_for_groups(da, groups_by_path, tmp_path)
    report, _ = da._emit_report(
        groups_by_path, 0, report=da.ReportCarrier(forest=forest)
    )
    assert "Showing top 0 components" in report


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._render_mermaid_component::component,declared_global,nodes
def test_render_mermaid_component_declared_none_documented(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    fn_id = da.NodeId(kind="FunctionSite", key=(path.name, "f"))
    bundle_id = da.NodeId(kind="ParamSet", key=("a", "b"))
    nodes = {
        fn_id: {"kind": "fn", "label": "mod.py:f", "path": path.name, "qual": "f"},
        bundle_id: {"kind": "bundle", "label": "a, b"},
    }
    bundle_map = {bundle_id: ("a", "b")}
    bundle_counts = {("a", "b"): 2}
    adj = {fn_id: {bundle_id}, bundle_id: {fn_id}}
    component = [fn_id, bundle_id]
    declared_global = {("x", "y")}
    declared_by_path = {"other.py": {("x", "y")}}
    documented_by_path = {path.name: {("a", "b")}}
    _, summary = da._render_mermaid_component(
        nodes,
        bundle_map,
        bundle_counts,
        adj,
        component,
        declared_global,
        declared_by_path,
        documented_by_path,
    )
    assert "Declared Config bundles: none found for this component." in summary
    assert "Documented bundles" in summary
    assert "(tier-2, documented)" in summary


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._merge_counts_by_knobs::knob_names
def test_merge_counts_by_knobs_merges_subset() -> None:
    da = _load()
    counts = {("a", "b"): 1, ("a", "b", "k"): 2}
    merged = da._merge_counts_by_knobs(counts, {"k"})
    assert merged == {("a", "b", "k"): 3}
