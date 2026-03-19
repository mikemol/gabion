from __future__ import annotations

from pathlib import Path
from tests.path_helpers import REPO_ROOT

from gabion.analysis.aspf.aspf import Forest

def _load():
    repo_root = REPO_ROOT
    from gabion.analysis import (
        AuditConfig, ReportCarrier, analyze_paths, render_report)

    return AuditConfig, ReportCarrier, analyze_paths, render_report

# gabion:evidence E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._emit_report::bundle_sites_by_path,coherence_witnesses,constant_smells,context_suggestions,deadness_witnesses,decision_surfaces,decision_warnings,exception_obligations,fingerprint_matches,fingerprint_provenance,fingerprint_synth,fingerprint_warnings,forest,groups_by_path,handledness_witnesses,invariant_propositions,max_components,never_invariants,rewrite_plans,type_ambiguities,type_callsite_evidence,type_suggestions,unused_arg_smells,value_decision_rewrites,value_decision_surfaces E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._iter_paths::config E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._analyze_file_internal::config,recursive E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._compute_fingerprint_rewrite_plans::exception_obligations E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._compute_fingerprint_synth::existing,min_occurrences E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._collect_never_invariants::forest E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan.analyze_decision_surfaces_repo::forest,require_tiers E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan.analyze_value_encoded_decisions_repo::forest,require_tiers E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._populate_bundle_forest::groups_by_path E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._collect_exception_obligations::handledness_witnesses E:decision_surface/direct::forest_spec.py::gabion.analysis.forest_spec.build_forest_spec::include_bundle_forest,include_decision_surfaces,include_never_invariants,include_value_decision_surfaces E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._compute_fingerprint_matches::index E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._compute_fingerprint_provenance::index E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._compute_fingerprint_warnings::index E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan.analyze_unused_arg_flow_repo::strictness E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._analyze_file_internal::stale_fc28d4eeb88d
# gabion:behavior primary=desired
def test_report_includes_callsite_evidence_for_undocumented_bundle(tmp_path: Path) -> None:
    AuditConfig, ReportCarrier, analyze_paths, render_report = _load()
    (tmp_path / "mod.py").write_text(
        "def h(x):\n"
        "    return x\n"
        "\n"
        "def g(x, y):\n"
        "    h(x)\n"
        "    h(y)\n"
        "\n"
        "def f(a, b):\n"
        "    return g(a, b)\n"
    )
    config = AuditConfig(project_root=tmp_path, external_filter=False)
    analysis = analyze_paths(
        [tmp_path],
        forest=Forest(),
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_bundle_forest=True,
        config=config,
    )
    report, _ = render_report(
        analysis.groups_by_path,
        10,
        project_root=tmp_path,
        report=ReportCarrier.from_analysis_result(analysis),
    )
    assert "Callsite evidence (undocumented bundles):" in report
    assert "mod.py" in report
    assert "f ->" in report
    assert "forwards a, b" in report

# gabion:evidence E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._emit_report::bundle_sites_by_path,coherence_witnesses,constant_smells,context_suggestions,deadness_witnesses,decision_surfaces,decision_warnings,exception_obligations,fingerprint_matches,fingerprint_provenance,fingerprint_synth,fingerprint_warnings,forest,groups_by_path,handledness_witnesses,invariant_propositions,max_components,never_invariants,rewrite_plans,type_ambiguities,type_callsite_evidence,type_suggestions,unused_arg_smells,value_decision_rewrites,value_decision_surfaces E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._iter_paths::config E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._analyze_file_internal::config,recursive E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._compute_fingerprint_rewrite_plans::exception_obligations E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._compute_fingerprint_synth::existing,min_occurrences E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._collect_never_invariants::forest E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan.analyze_decision_surfaces_repo::forest,require_tiers E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan.analyze_value_encoded_decisions_repo::forest,require_tiers E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._populate_bundle_forest::groups_by_path E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._collect_exception_obligations::handledness_witnesses E:decision_surface/direct::forest_spec.py::gabion.analysis.forest_spec.build_forest_spec::include_bundle_forest,include_decision_surfaces,include_never_invariants,include_value_decision_surfaces E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._compute_fingerprint_matches::index E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._compute_fingerprint_provenance::index E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._compute_fingerprint_warnings::index E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan.analyze_unused_arg_flow_repo::strictness E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._analyze_file_internal::stale_41de4e49e9ad
# gabion:behavior primary=desired
def test_report_omits_callsite_evidence_for_documented_bundle(tmp_path: Path) -> None:
    AuditConfig, ReportCarrier, analyze_paths, render_report = _load()
    (tmp_path / "mod.py").write_text(
        "# dataflow-bundle: a, b\n"
        "def h(x):\n"
        "    return x\n"
        "\n"
        "def g(x, y):\n"
        "    h(x)\n"
        "    h(y)\n"
        "\n"
        "def f(a, b):\n"
        "    return g(a, b)\n"
    )
    config = AuditConfig(project_root=tmp_path, external_filter=False)
    analysis = analyze_paths(
        [tmp_path],
        forest=Forest(),
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_bundle_forest=True,
        config=config,
    )
    report, _ = render_report(
        analysis.groups_by_path,
        10,
        project_root=tmp_path,
        report=ReportCarrier.from_analysis_result(analysis),
    )
    assert "Callsite evidence (undocumented bundles):" in report
    assert "undocumented bundle: a, b" not in report
