from __future__ import annotations

import ast
from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit as da

    return da


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_names::fn,ignore_params
def test_decision_surface_params_collects_names() -> None:
    da = _load()
    tree = ast.parse(
        "def f(a, b, cfg):\n"
        "    if a > 0:\n"
        "        return b\n"
        "    while cfg.flag:\n"
        "        break\n"
        "    assert b\n"
    )
    fn = tree.body[0]
    params = da._decision_surface_params(fn, ignore_params=None)
    assert params == {"a", "b", "cfg"}


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_names::fn,ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._mark_param_roots::params
def test_value_encoded_decision_params_collects_names() -> None:
    da = _load()
    tree = ast.parse(
        "def f(a, b, mask, value):\n"
        "    x = min(a, b)\n"
        "    y = value * (a > 0)\n"
        "    z = mask & 1\n"
        "    return x + y + z\n"
    )
    fn = tree.body[0]
    params, reasons = da._value_encoded_decision_params(fn, ignore_params=None)
    assert params == {"a", "b", "mask"}
    assert reasons == {"min/max", "boolean arithmetic", "bitmask"}


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._analyze_file_internal::config,recursive E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_rewrite_plans::exception_obligations E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_synth::existing,min_occurrences E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_never_invariants::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_decision_surfaces_repo::forest,require_tiers E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_value_encoded_decisions_repo::forest,require_tiers E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._populate_bundle_forest::groups_by_path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_exception_obligations::handledness_witnesses E:decision_surface/direct::forest_spec.py::gabion.analysis.forest_spec.build_forest_spec::include_bundle_forest,include_decision_surfaces,include_never_invariants,include_value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_matches::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_provenance::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_warnings::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._is_test_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_unused_arg_flow_repo::strictness E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._decision_tier_for::tier_map
def test_analyze_decision_surfaces_repo(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "def f(a, b):\n"
        "    if b:\n"
        "        return a\n"
        "    return b\n"
    )
    surfaces, warnings, lint_lines = da.analyze_decision_surfaces_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        transparent_decorators=None,
        forest=da.Forest(),
    )
    assert surfaces == ["mod.py:mod.f decision surface params: b (boundary)"]
    assert warnings == []
    assert any("GABION_DECISION_SURFACE" in line for line in lint_lines)

    analysis = da.analyze_paths(
        forest=da.Forest(),
        paths=[path],
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_decision_surfaces=True,
        config=da.AuditConfig(project_root=tmp_path),
    )
    assert analysis.decision_surfaces == surfaces
    assert analysis.decision_warnings == warnings

    (
        value_surfaces,
        value_warnings,
        value_rewrites,
        value_lint_lines,
    ) = da.analyze_value_encoded_decisions_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        transparent_decorators=None,
        forest=da.Forest(),
    )
    assert value_surfaces == []
    assert value_warnings == []
    assert value_rewrites == []
    assert value_lint_lines == []


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_value_encoded_decisions_repo::forest,require_tiers E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._is_test_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._decision_tier_for::tier_map
def test_analyze_value_encoded_decisions_repo(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "def f(a, b, mask, value):\n"
        "    x = min(a, b)\n"
        "    y = value * (a > 0)\n"
        "    z = mask & 1\n"
        "    return x + y + z\n"
    )
    surfaces, warnings, rewrites, lint_lines = da.analyze_value_encoded_decisions_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        transparent_decorators=None,
        forest=da.Forest(),
    )
    assert surfaces == [
        "mod.py:mod.f value-encoded decision params: a, b, mask (bitmask, boolean arithmetic, min/max)"
    ]
    assert warnings == []
    assert rewrites == [
        "mod.py:mod.f consider rebranching value-encoded decision params: a, b, mask (bitmask, boolean arithmetic, min/max)"
    ]
    assert any("GABION_VALUE_DECISION_SURFACE" in line for line in lint_lines)


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._render_component_callsite_evidence::bundle_counts E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::bundle_sites_by_path,coherence_witnesses,constant_smells,context_suggestions,deadness_witnesses,decision_surfaces,decision_warnings,exception_obligations,fingerprint_matches,fingerprint_provenance,fingerprint_synth,fingerprint_warnings,forest,groups_by_path,handledness_witnesses,invariant_propositions,max_components,never_invariants,rewrite_plans,type_ambiguities,type_callsite_evidence,type_suggestions,unused_arg_smells,value_decision_rewrites,value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._render_mermaid_component::component,declared_global,nodes E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_dataclass_call_bundles::dataclass_registry,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_never_invariants::entries,include_proven_unreachable,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_coherence_witnesses::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_deadness_witnesses::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_exception_obligations::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_handledness_witnesses::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_rewrite_plans::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_fingerprint_provenance::entries,max_examples E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._bundle_projection_from_forest::file_paths E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._populate_bundle_forest::groups_by_path
def test_emit_report_includes_value_rewrites(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text("def f(a):\n    return a\n")
    groups_by_path = {path: {"f": [set(["a", "b"])]}}
    forest = da.Forest()
    da._populate_bundle_forest(
        forest,
        groups_by_path=groups_by_path,
        file_paths=[path],
        project_root=tmp_path,
        parse_failure_witnesses=[],
    )
    report, _ = da._emit_report(
        groups_by_path,
        1,
        report=da.ReportCarrier(
            forest=forest,
            value_decision_rewrites=[
                "mod.py:f consider rebranching value-encoded decision params: a (min/max)"
            ],
        ),
    )
    assert "Value-encoded decision rebranch suggestions" in report


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._analyze_file_internal::config,recursive E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_rewrite_plans::exception_obligations E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_synth::existing,min_occurrences E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_never_invariants::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_decision_surfaces_repo::forest,require_tiers E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_value_encoded_decisions_repo::forest,require_tiers E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._populate_bundle_forest::groups_by_path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_exception_obligations::handledness_witnesses E:decision_surface/direct::forest_spec.py::gabion.analysis.forest_spec.build_forest_spec::include_bundle_forest,include_decision_surfaces,include_never_invariants,include_value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_matches::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_provenance::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_warnings::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._is_test_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_unused_arg_flow_repo::strictness E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._decision_tier_for::tier_map
def test_decision_surface_internal_caller(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "def f(a, b):\n"
        "    if b:\n"
        "        return a\n"
        "    return b\n"
        "\n"
        "def g(x, y):\n"
        "    return f(x, y)\n"
    )
    surfaces, warnings, lint_lines = da.analyze_decision_surfaces_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        transparent_decorators=None,
        forest=da.Forest(),
    )
    assert surfaces == [
        "mod.py:mod.f decision surface params: b (internal callers (transitive): 1)"
    ]
    assert warnings == []
    assert lint_lines == []

    analysis = da.analyze_paths(
        forest=da.Forest(),
        paths=[path],
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_decision_surfaces=True,
        config=da.AuditConfig(project_root=tmp_path),
    )
    assert analysis.context_suggestions == [
        "Consider contextvar for mod.py:mod.f decision surface params: b (internal callers (transitive): 1)"
    ]


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._render_component_callsite_evidence::bundle_counts E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::bundle_sites_by_path,coherence_witnesses,constant_smells,context_suggestions,deadness_witnesses,decision_surfaces,decision_warnings,exception_obligations,fingerprint_matches,fingerprint_provenance,fingerprint_synth,fingerprint_warnings,forest,groups_by_path,handledness_witnesses,invariant_propositions,max_components,never_invariants,rewrite_plans,type_ambiguities,type_callsite_evidence,type_suggestions,unused_arg_smells,value_decision_rewrites,value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._render_mermaid_component::component,declared_global,nodes E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_dataclass_call_bundles::dataclass_registry,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_never_invariants::entries,include_proven_unreachable,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_coherence_witnesses::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_deadness_witnesses::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_exception_obligations::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_handledness_witnesses::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_rewrite_plans::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_fingerprint_provenance::entries,max_examples E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._bundle_projection_from_forest::file_paths E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._populate_bundle_forest::groups_by_path
def test_emit_report_includes_decision_surfaces(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text("def f(a):\n    return a\n")
    groups_by_path = {path: {"f": [set(["a", "b"])]}}
    forest = da.Forest()
    da._populate_bundle_forest(
        forest,
        groups_by_path=groups_by_path,
        file_paths=[path],
        project_root=tmp_path,
        parse_failure_witnesses=[],
    )
    report, _ = da._emit_report(
        groups_by_path,
        1,
        report=da.ReportCarrier(
            forest=forest,
            decision_surfaces=["mod.py:f decision surface params: a"],
        ),
    )
    assert "Decision surface candidates" in report


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_decision_surfaces_repo::forest,require_tiers E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._is_test_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._decision_tier_for::tier_map
def test_decision_surface_tier_warning_internal(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "def internal(user_mode):\n"
        "    if user_mode:\n"
        "        return 1\n"
        "    return 0\n"
        "\n"
        "def api(user_mode):\n"
        "    return internal(user_mode)\n"
    )
    _, warnings, lint_lines = da.analyze_decision_surfaces_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        transparent_decorators=None,
        decision_tiers={"user_mode": 3},
        forest=da.Forest(),
    )
    assert any("tier-3 decision param 'user_mode'" in warning for warning in warnings)
    assert any("GABION_DECISION_TIER" in line for line in lint_lines)


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_decision_surfaces_repo::forest,require_tiers E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._is_test_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._decision_tier_for::tier_map
def test_decision_surface_location_tier_suppresses_lint(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "def f(a):\n"
        "    if a:\n"
        "        return 1\n"
        "    return 0\n"
    )
    surfaces, warnings, lint_lines = da.analyze_decision_surfaces_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        transparent_decorators=None,
        decision_tiers={"mod.py:1:7": 1},
        forest=da.Forest(),
    )
    assert surfaces
    assert warnings == []
    assert not any("GABION_DECISION_SURFACE" in line for line in lint_lines)
