from __future__ import annotations

from pathlib import Path

def _load():
    repo_root = Path(__file__).resolve().parents[1]
    from gabion.analysis import dataflow_audit as da

    return da

def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._render_component_callsite_evidence::bundle_counts E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::bundle_sites_by_path,coherence_witnesses,constant_smells,context_suggestions,deadness_witnesses,decision_surfaces,decision_warnings,exception_obligations,fingerprint_matches,fingerprint_provenance,fingerprint_synth,fingerprint_warnings,forest,groups_by_path,handledness_witnesses,invariant_propositions,max_components,never_invariants,rewrite_plans,type_ambiguities,type_callsite_evidence,type_suggestions,unused_arg_smells,value_decision_rewrites,value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._render_mermaid_component::component,declared_global,nodes E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._analyze_file_internal::config,recursive E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_never_invariants::entries,include_proven_unreachable,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_coherence_witnesses::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_deadness_witnesses::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_exception_obligations::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_handledness_witnesses::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_rewrite_plans::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_fingerprint_provenance::entries,max_examples E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_rewrite_plans::exception_obligations E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_synth::existing,min_occurrences E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._bundle_projection_from_forest::file_paths E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_never_invariants::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_decision_surfaces_repo::forest,require_tiers E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_value_encoded_decisions_repo::forest,require_tiers E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._populate_bundle_forest::groups_by_path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_exception_obligations::handledness_witnesses E:decision_surface/direct::forest_spec.py::gabion.analysis.forest_spec.build_forest_spec::include_bundle_forest,include_decision_surfaces,include_never_invariants,include_value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_matches::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_provenance::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_warnings::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_unused_arg_flow_repo::strictness
def test_exception_protocol_never_violation(tmp_path: Path) -> None:
    da = _load()
    module_path = tmp_path / "mod.py"
    _write(
        module_path,
        "from gabion.exceptions import NeverRaise\n"
        "\n"
        "def f(a):\n"
        "    raise NeverRaise('nope')\n",
    )
    config = da.AuditConfig(
        project_root=tmp_path,
        exclude_dirs=set(),
        ignore_params=set(),
        external_filter=False,
        strictness="high",
        never_exceptions={"NeverRaise"},
    )
    analysis = da.analyze_paths(
        forest=da.Forest(),
        paths=[tmp_path],
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_exception_obligations=True,
        include_lint_lines=True,
        config=config,
    )
    obligations = analysis.exception_obligations
    assert any(entry.get("status") == "FORBIDDEN" for entry in obligations)
    assert any("GABION_EXC_NEVER" in line for line in analysis.lint_lines)
    report, violations = da._emit_report(
        analysis.groups_by_path,
        3,
        report=da.ReportCarrier(
            forest=analysis.forest,
            exception_obligations=obligations,
        ),
    )
    assert "Exception protocol violations" in report
    assert any("protocol=never" in line for line in violations)
