from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit as da

    return da


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._analyze_file_internal::config,recursive E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_exception_obligations::handledness_witnesses E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_never_invariants::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_matches::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_provenance::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_rewrite_plans::exception_obligations E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_synth::existing,min_occurrences E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_warnings::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_call_ambiguities::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::ambiguity_witnesses,bundle_sites_by_path,coherence_witnesses,constant_smells,context_suggestions,deadness_witnesses,decision_surfaces,decision_warnings,exception_obligations,fingerprint_matches,fingerprint_provenance,fingerprint_synth,fingerprint_warnings,forest,groups_by_path,handledness_witnesses,invariant_propositions,max_components,never_invariants,rewrite_plans,type_ambiguities,type_callsite_evidence,type_suggestions,unused_arg_smells,value_decision_rewrites,value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._populate_bundle_forest::groups_by_path,include_all_sites E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_decision_surfaces_repo::forest,require_tiers E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_ambiguities,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_unused_arg_flow_repo::strictness E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_value_encoded_decisions_repo::forest,require_tiers E:decision_surface/direct::forest_spec.py::gabion.analysis.forest_spec.build_forest_spec::include_ambiguities,include_bundle_forest,include_decision_surfaces,include_never_invariants,include_value_decision_surfaces
def test_never_invariants_emit_forest_and_report(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "from gabion.invariants import never\n"
        "\n"
        "def f(a):\n"
        "    if a:\n"
        "        never('boom')\n"
    )
    config = da.AuditConfig(project_root=tmp_path)
    analysis = da.analyze_paths(
        [path],
        forest=da.Forest(),
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_never_invariants=True,
        include_bundle_forest=True,
        config=config,
    )
    assert analysis.never_invariants
    assert analysis.forest is not None
    assert any(
        alt.kind == "NeverInvariantSink" for alt in analysis.forest.alts
    )
    report, _ = da.render_report(
        analysis.groups_by_path,
        max_components=3,
        report=da.ReportCarrier.from_analysis_result(analysis),
    )
    assert "Never invariants" in report


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._analyze_file_internal::config,recursive E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_exception_obligations::handledness_witnesses E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_never_invariants::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_matches::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_provenance::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_rewrite_plans::exception_obligations E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_synth::existing,min_occurrences E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_warnings::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_call_ambiguities::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._populate_bundle_forest::groups_by_path,include_all_sites E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_decision_surfaces_repo::forest,require_tiers E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_ambiguities,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_unused_arg_flow_repo::strictness E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_value_encoded_decisions_repo::forest,require_tiers E:decision_surface/direct::forest_spec.py::gabion.analysis.forest_spec.build_forest_spec::include_ambiguities,include_bundle_forest,include_decision_surfaces,include_never_invariants,include_value_decision_surfaces
def test_never_invariant_violation(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "from gabion.invariants import never\n"
        "\n"
        "def f(a):\n"
        "    if a == 1:\n"
        "        never('boom')\n"
        "\n"
        "def g():\n"
        "    f(1)\n"
    )
    config = da.AuditConfig(project_root=tmp_path)
    analysis = da.analyze_paths(
        [path],
        forest=da.Forest(),
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_deadness_witnesses=True,
        include_never_invariants=True,
        include_lint_lines=True,
        include_bundle_forest=True,
        config=config,
    )
    assert any(entry.get("status") == "VIOLATION" for entry in analysis.never_invariants)
    assert any(
        "GABION_NEVER_INVARIANT" in line and "status=VIOLATION" in line
        for line in analysis.lint_lines
    )


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._analyze_file_internal::config,recursive E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_exception_obligations::handledness_witnesses E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_never_invariants::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_matches::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_provenance::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_rewrite_plans::exception_obligations E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_synth::existing,min_occurrences E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_warnings::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_call_ambiguities::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._populate_bundle_forest::groups_by_path,include_all_sites E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_decision_surfaces_repo::forest,require_tiers E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_ambiguities,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_unused_arg_flow_repo::strictness E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_value_encoded_decisions_repo::forest,require_tiers E:decision_surface/direct::forest_spec.py::gabion.analysis.forest_spec.build_forest_spec::include_ambiguities,include_bundle_forest,include_decision_surfaces,include_never_invariants,include_value_decision_surfaces
def test_never_invariant_proven_unreachable(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "from gabion.invariants import never\n"
        "\n"
        "def f(a):\n"
        "    if a == 0:\n"
        "        never('boom')\n"
        "\n"
        "def g():\n"
        "    f(1)\n"
    )
    config = da.AuditConfig(project_root=tmp_path)
    analysis = da.analyze_paths(
        [path],
        forest=da.Forest(),
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_deadness_witnesses=True,
        include_never_invariants=True,
        include_lint_lines=True,
        include_bundle_forest=True,
        config=config,
    )
    assert any(
        entry.get("status") == "PROVEN_UNREACHABLE" for entry in analysis.never_invariants
    )
    assert not any("GABION_NEVER_INVARIANT" in line for line in analysis.lint_lines)


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._analyze_file_internal::config,recursive E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_exception_obligations::handledness_witnesses E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_never_invariants::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_matches::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_provenance::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_rewrite_plans::exception_obligations E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_synth::existing,min_occurrences E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_warnings::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_call_ambiguities::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._populate_bundle_forest::groups_by_path,include_all_sites E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_decision_surfaces_repo::forest,require_tiers E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_ambiguities,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_unused_arg_flow_repo::strictness E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_value_encoded_decisions_repo::forest,require_tiers E:decision_surface/direct::forest_spec.py::gabion.analysis.forest_spec.build_forest_spec::include_ambiguities,include_bundle_forest,include_decision_surfaces,include_never_invariants,include_value_decision_surfaces
def test_never_invariant_obligation(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "from gabion.invariants import never\n"
        "\n"
        "def f(a):\n"
        "    if a == 0:\n"
        "        never('boom')\n"
        "\n"
        "def g(x):\n"
        "    f(x)\n"
    )
    config = da.AuditConfig(project_root=tmp_path)
    analysis = da.analyze_paths(
        [path],
        forest=da.Forest(),
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_deadness_witnesses=True,
        include_never_invariants=True,
        include_lint_lines=True,
        include_bundle_forest=True,
        config=config,
    )
    assert any(
        entry.get("status") == "OBLIGATION" for entry in analysis.never_invariants
    )
    assert any(
        "GABION_NEVER_INVARIANT" in line and "status=OBLIGATION" in line
        for line in analysis.lint_lines
    )


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._analyze_file_internal::config,recursive E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_exception_obligations::handledness_witnesses E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_never_invariants::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_matches::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_provenance::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_rewrite_plans::exception_obligations E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_synth::existing,min_occurrences E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_warnings::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_call_ambiguities::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::ambiguity_witnesses,bundle_sites_by_path,coherence_witnesses,constant_smells,context_suggestions,deadness_witnesses,decision_surfaces,decision_warnings,exception_obligations,fingerprint_matches,fingerprint_provenance,fingerprint_synth,fingerprint_warnings,forest,groups_by_path,handledness_witnesses,invariant_propositions,max_components,never_invariants,rewrite_plans,type_ambiguities,type_callsite_evidence,type_suggestions,unused_arg_smells,value_decision_rewrites,value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._populate_bundle_forest::groups_by_path,include_all_sites E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_decision_surfaces_repo::forest,require_tiers E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_ambiguities,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_unused_arg_flow_repo::strictness E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_value_encoded_decisions_repo::forest,require_tiers E:decision_surface/direct::forest_spec.py::gabion.analysis.forest_spec.build_forest_spec::include_ambiguities,include_bundle_forest,include_decision_surfaces,include_never_invariants,include_value_decision_surfaces
def test_never_invariant_report_order_and_evidence(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "from gabion.invariants import never\n"
        "\n"
        "def f(a):\n"
        "    if a == 1:\n"
        "        never('violation')\n"
        "\n"
        "def g():\n"
        "    f(1)\n"
        "\n"
        "def h(a):\n"
        "    if a == 0:\n"
        "        never('unreachable')\n"
        "\n"
        "def i():\n"
        "    h(1)\n"
        "\n"
        "def j(a):\n"
        "    if a == 0:\n"
        "        never('obligation')\n"
        "\n"
        "def k(x):\n"
        "    j(x)\n"
    )
    config = da.AuditConfig(project_root=tmp_path)
    analysis = da.analyze_paths(
        [path],
        forest=da.Forest(),
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_deadness_witnesses=True,
        include_never_invariants=True,
        include_bundle_forest=True,
        config=config,
    )
    report, _ = da.render_report(
        analysis.groups_by_path,
        max_components=3,
        report=da.ReportCarrier.from_analysis_result(analysis),
    )
    assert "Never invariants:" in report
    never_section = report.split("Never invariants:")[1]
    assert "generated_by_spec_id:" in never_section
    assert "generated_by_spec:" in never_section
    order = [
        never_section.find("VIOLATION:"),
        never_section.find("OBLIGATION:"),
        never_section.find("PROVEN_UNREACHABLE:"),
    ]
    assert all(idx >= 0 for idx in order)
    assert order == sorted(order)
    assert "reason=violation" in never_section
    assert "reason=obligation" in never_section
    assert "reason=unreachable" in never_section
