from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit

    return dataflow_audit


def _write_conflict_code(path: Path) -> None:
    path.write_text(
        "def callee_int(x: int):\n"
        "    return x\n\n"
        "def callee_str(x: str):\n"
        "    return x\n\n"
        "def caller(a):\n"
        "    return callee_int(a)\n\n"
        "def caller_conflict(b):\n"
        "    callee_int(b)\n"
        "    callee_str(b)\n"
    )


def _write_bundle_code(path: Path) -> None:
    path.write_text(
        "def callee(x):\n"
        "    return x\n"
        "\n"
        "def caller(a, b):\n"
        "    callee(a)\n"
        "    callee(b)\n"
    )


def _write_typed_bundle_code(path: Path) -> None:
    path.write_text(
        "def callee(x: int):\n"
        "    return x\n"
        "\n"
        "def caller_one(a: int, b: int):\n"
        "    callee(a)\n"
        "    callee(b)\n"
        "\n"
        "def caller_two(a: int, b: int):\n"
        "    callee(a)\n"
        "    callee(b)\n"
    )


# gabion:evidence E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._apply_baseline::baseline_allowlist E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_dot::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::ambiguity_witnesses,bundle_sites_by_path,coherence_witnesses,constant_smells,context_suggestions,deadness_witnesses,decision_surfaces,decision_warnings,exception_obligations,fingerprint_matches,fingerprint_provenance,fingerprint_synth,fingerprint_warnings,forest,groups_by_path,handledness_witnesses,invariant_propositions,max_components,never_invariants,rewrite_plans,type_ambiguities,type_callsite_evidence,type_suggestions,unused_arg_smells,value_decision_rewrites,value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._load_baseline::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_transparent_decorators::value E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_baseline_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_synth_registry_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_ambiguities,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions
def test_run_baseline_write_requires_path(tmp_path: Path) -> None:
    dataflow_audit = _load()
    code = dataflow_audit.run(
        [
            str(tmp_path),
            "--root",
            str(tmp_path),
            "--baseline-write",
        ]
    )
    assert code == 2


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._apply_baseline::baseline_allowlist E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::bundle_sites_by_path,coherence_witnesses,constant_smells,context_suggestions,deadness_witnesses,decision_surfaces,decision_warnings,exception_obligations,fingerprint_matches,fingerprint_provenance,fingerprint_synth,fingerprint_warnings,forest,groups_by_path,handledness_witnesses,invariant_propositions,max_components,never_invariants,rewrite_plans,type_ambiguities,type_callsite_evidence,type_suggestions,unused_arg_smells,value_decision_rewrites,value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_dot::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._load_baseline::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_baseline_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_synth_registry_path::path E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_transparent_decorators::value
def test_run_dot_only_returns_early(tmp_path: Path) -> None:
    dataflow_audit = _load()
    sample = tmp_path / "sample.py"
    sample.write_text("def f(a, b):\n    return a + b\n")
    dot_path = tmp_path / "graph.dot"
    code = dataflow_audit.run(
        [
            str(tmp_path),
            "--root",
            str(tmp_path),
            "--dot",
            str(dot_path),
        ]
    )
    assert code == 0
    assert dot_path.exists()


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._apply_baseline::baseline_allowlist E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::bundle_sites_by_path,coherence_witnesses,constant_smells,context_suggestions,deadness_witnesses,decision_surfaces,decision_warnings,exception_obligations,fingerprint_matches,fingerprint_provenance,fingerprint_synth,fingerprint_warnings,forest,groups_by_path,handledness_witnesses,invariant_propositions,max_components,never_invariants,rewrite_plans,type_ambiguities,type_callsite_evidence,type_suggestions,unused_arg_smells,value_decision_rewrites,value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_dot::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._load_baseline::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_baseline_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_synth_registry_path::path E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_transparent_decorators::value
def test_run_fail_on_type_ambiguities_with_synthesis_plan(tmp_path: Path) -> None:
    dataflow_audit = _load()
    sample = tmp_path / "mod.py"
    _write_conflict_code(sample)
    plan_path = tmp_path / "plan.json"
    code = dataflow_audit.run(
        [
            str(tmp_path),
            "--root",
            str(tmp_path),
            "--synthesis-plan",
            str(plan_path),
            "--fail-on-type-ambiguities",
        ]
    )
    assert plan_path.exists()
    assert code == 1


# gabion:evidence E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._apply_baseline::baseline_allowlist E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_dot::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::ambiguity_witnesses,bundle_sites_by_path,coherence_witnesses,constant_smells,context_suggestions,deadness_witnesses,decision_surfaces,decision_warnings,exception_obligations,fingerprint_matches,fingerprint_provenance,fingerprint_synth,fingerprint_warnings,forest,groups_by_path,handledness_witnesses,invariant_propositions,max_components,never_invariants,rewrite_plans,type_ambiguities,type_callsite_evidence,type_suggestions,unused_arg_smells,value_decision_rewrites,value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._load_baseline::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_transparent_decorators::value E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_baseline_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_synth_registry_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_ambiguities,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions
def test_run_report_with_baseline_write(tmp_path: Path) -> None:
    dataflow_audit = _load()
    sample = tmp_path / "sample.py"
    _write_bundle_code(sample)
    report_path = tmp_path / "report.md"
    baseline_path = tmp_path / "baseline.txt"
    code = dataflow_audit.run(
        [
            str(sample),
            "--root",
            str(tmp_path),
            "--report",
            str(report_path),
            "--baseline",
            str(baseline_path),
            "--baseline-write",
            "--fail-on-violations",
        ]
    )
    assert code == 0
    assert report_path.exists()
    assert baseline_path.exists()


# gabion:evidence E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._apply_baseline::baseline_allowlist E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_dot::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::ambiguity_witnesses,bundle_sites_by_path,coherence_witnesses,constant_smells,context_suggestions,deadness_witnesses,decision_surfaces,decision_warnings,exception_obligations,fingerprint_matches,fingerprint_provenance,fingerprint_synth,fingerprint_warnings,forest,groups_by_path,handledness_witnesses,invariant_propositions,max_components,never_invariants,rewrite_plans,type_ambiguities,type_callsite_evidence,type_suggestions,unused_arg_smells,value_decision_rewrites,value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._load_baseline::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_transparent_decorators::value E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_baseline_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_synth_registry_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_ambiguities,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions
def test_run_report_with_baseline_apply(tmp_path: Path) -> None:
    dataflow_audit = _load()
    sample = tmp_path / "sample.py"
    _write_bundle_code(sample)
    report_path = tmp_path / "report.md"
    baseline_path = tmp_path / "baseline.txt"
    baseline_path.write_text("preexisting\n")
    code = dataflow_audit.run(
        [
            str(sample),
            "--root",
            str(tmp_path),
            "--report",
            str(report_path),
            "--baseline",
            str(baseline_path),
            "--fail-on-violations",
        ]
    )
    assert report_path.exists()
    assert code == 1


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._apply_baseline::baseline_allowlist E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::bundle_sites_by_path,coherence_witnesses,constant_smells,context_suggestions,deadness_witnesses,decision_surfaces,decision_warnings,exception_obligations,fingerprint_matches,fingerprint_provenance,fingerprint_synth,fingerprint_warnings,forest,groups_by_path,handledness_witnesses,invariant_propositions,max_components,never_invariants,rewrite_plans,type_ambiguities,type_callsite_evidence,type_suggestions,unused_arg_smells,value_decision_rewrites,value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_dot::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._load_baseline::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_baseline_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_synth_registry_path::path E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_transparent_decorators::value
def test_run_type_audit_prints_findings(tmp_path: Path, capsys) -> None:
    dataflow_audit = _load()
    sample = tmp_path / "types.py"
    _write_conflict_code(sample)
    code = dataflow_audit.run(
        [
            str(sample),
            "--root",
            str(tmp_path),
            "--type-audit",
            "--type-audit-max",
            "5",
            "--transparent-decorators",
            "wrap, deco",
        ]
    )
    captured = capsys.readouterr().out
    assert "Type tightening candidates" in captured
    assert "Type ambiguities" in captured
    assert code == 0


# gabion:evidence E:call_cluster::dataflow_audit.py::gabion.analysis.dataflow_audit.run::test_dataflow_run_edges.py::tests.test_dataflow_run_edges._load::test_dataflow_run_edges.py::tests.test_dataflow_run_edges._write_bundle_code
def test_run_synthesis_protocols_only(tmp_path: Path) -> None:
    dataflow_audit = _load()
    sample = tmp_path / "sample.py"
    _write_bundle_code(sample)
    stubs_path = tmp_path / "stubs.py"
    code = dataflow_audit.run(
        [
            str(sample),
            "--root",
            str(tmp_path),
            "--synthesis-protocols",
            str(stubs_path),
            "--synthesis-merge-overlap",
            "1.5",
        ]
    )
    assert stubs_path.exists()
    assert code == 0


# gabion:evidence E:call_cluster::dataflow_audit.py::gabion.analysis.dataflow_audit.run::test_dataflow_run_edges.py::tests.test_dataflow_run_edges._load::test_dataflow_run_edges.py::tests.test_dataflow_run_edges._write_bundle_code
def test_run_synthesis_outputs_to_stdout(tmp_path: Path, capsys) -> None:
    dataflow_audit = _load()
    sample = tmp_path / "sample.py"
    _write_bundle_code(sample)
    code = dataflow_audit.run(
        [
            str(sample),
            "--root",
            str(tmp_path),
            "--synthesis-plan",
            "-",
            "--synthesis-protocols",
            "-",
        ]
    )
    captured = capsys.readouterr().out
    assert "class" in captured
    assert code == 0


# gabion:evidence E:call_cluster::dataflow_audit.py::gabion.analysis.dataflow_audit.run::test_dataflow_run_edges.py::tests.test_dataflow_run_edges._load::test_dataflow_run_edges.py::tests.test_dataflow_run_edges._write_bundle_code
def test_run_synthesis_uses_config_overlap(tmp_path: Path) -> None:
    dataflow_audit = _load()
    sample = tmp_path / "sample.py"
    _write_bundle_code(sample)
    config_path = tmp_path / "gabion.toml"
    config_path.write_text("[synthesis]\nmerge_overlap_threshold = 0.5\n")
    plan_path = tmp_path / "plan.json"
    code = dataflow_audit.run(
        [
            str(sample),
            "--root",
            str(tmp_path),
            "--config",
            str(config_path),
            "--synthesis-plan",
            str(plan_path),
        ]
    )
    assert plan_path.exists()
    assert code == 0


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._apply_baseline::baseline_allowlist E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::bundle_sites_by_path,coherence_witnesses,constant_smells,context_suggestions,deadness_witnesses,decision_surfaces,decision_warnings,exception_obligations,fingerprint_matches,fingerprint_provenance,fingerprint_synth,fingerprint_warnings,forest,groups_by_path,handledness_witnesses,invariant_propositions,max_components,never_invariants,rewrite_plans,type_ambiguities,type_callsite_evidence,type_suggestions,unused_arg_smells,value_decision_rewrites,value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_dot::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._load_baseline::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_baseline_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_synth_registry_path::path E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_transparent_decorators::value
def test_run_fingerprint_outputs_and_decision_snapshot(tmp_path: Path, capsys) -> None:
    dataflow_audit = _load()
    sample = tmp_path / "typed.py"
    _write_typed_bundle_code(sample)
    config_path = tmp_path / "gabion.toml"
    config_path.write_text(
        "[fingerprints]\n"
        "user_context = [\"int\"]\n"
        "synth_min_occurrences = 2\n"
        "\n"
        "[decision]\n"
        "tier2 = [\"a\"]\n"
    )
    code = dataflow_audit.run(
        [
            str(sample),
            "--root",
            str(tmp_path),
            "--config",
            str(config_path),
            "--fingerprint-synth-json",
            "-",
            "--fingerprint-provenance-json",
            "-",
            "--fingerprint-deadness-json",
            "-",
            "--fingerprint-coherence-json",
            "-",
            "--fingerprint-rewrite-plans-json",
            "-",
            "--fingerprint-exception-obligations-json",
            "-",
            "--fingerprint-handledness-json",
            "-",
            "--emit-decision-snapshot",
            "-",
        ]
    )
    output = capsys.readouterr().out
    assert "fingerprint" in output
    assert "decision_surfaces" in output
    assert code == 0


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._apply_baseline::baseline_allowlist E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::bundle_sites_by_path,coherence_witnesses,constant_smells,context_suggestions,deadness_witnesses,decision_surfaces,decision_warnings,exception_obligations,fingerprint_matches,fingerprint_provenance,fingerprint_synth,fingerprint_warnings,forest,groups_by_path,handledness_witnesses,invariant_propositions,max_components,never_invariants,rewrite_plans,type_ambiguities,type_callsite_evidence,type_suggestions,unused_arg_smells,value_decision_rewrites,value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_dot::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._load_baseline::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_baseline_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_synth_registry_path::path E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_transparent_decorators::value
def test_run_fingerprint_outputs_write_files(tmp_path: Path) -> None:
    dataflow_audit = _load()
    sample = tmp_path / "typed.py"
    _write_typed_bundle_code(sample)
    synth_registry_path = tmp_path / "synth_registry.json"
    synth_registry_path.write_text(
        "{\"version\": \"synth@1\", \"min_occurrences\": 2, \"entries\": []}"
    )
    config_path = tmp_path / "gabion.toml"
    config_path.write_text(
        "[fingerprints]\n"
        "user_context = [\"int\"]\n"
        "synth_min_occurrences = 2\n"
        f"synth_registry_path = \"{synth_registry_path}\"\n"
    )
    synth_path = tmp_path / "fingerprint_synth.json"
    provenance_path = tmp_path / "fingerprint_provenance.json"
    deadness_path = tmp_path / "fingerprint_deadness.json"
    coherence_path = tmp_path / "fingerprint_coherence.json"
    rewrite_plans_path = tmp_path / "fingerprint_rewrite_plans.json"
    exception_obligations_path = tmp_path / "fingerprint_exception_obligations.json"
    handledness_path = tmp_path / "fingerprint_handledness.json"
    code = dataflow_audit.run(
        [
            str(sample),
            "--root",
            str(tmp_path),
            "--config",
            str(config_path),
            "--fingerprint-synth-json",
            str(synth_path),
            "--fingerprint-provenance-json",
            str(provenance_path),
            "--fingerprint-deadness-json",
            str(deadness_path),
            "--fingerprint-coherence-json",
            str(coherence_path),
            "--fingerprint-rewrite-plans-json",
            str(rewrite_plans_path),
            "--fingerprint-exception-obligations-json",
            str(exception_obligations_path),
            "--fingerprint-handledness-json",
            str(handledness_path),
        ]
    )
    assert code == 0
    assert synth_path.exists()
    assert provenance_path.exists()
    assert deadness_path.exists()
    assert coherence_path.exists()
    assert rewrite_plans_path.exists()
    assert exception_obligations_path.exists()
    assert handledness_path.exists()


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._apply_baseline::baseline_allowlist E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::bundle_sites_by_path,coherence_witnesses,constant_smells,context_suggestions,deadness_witnesses,decision_surfaces,decision_warnings,exception_obligations,fingerprint_matches,fingerprint_provenance,fingerprint_synth,fingerprint_warnings,forest,groups_by_path,handledness_witnesses,invariant_propositions,max_components,never_invariants,rewrite_plans,type_ambiguities,type_callsite_evidence,type_suggestions,unused_arg_smells,value_decision_rewrites,value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_dot::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._load_baseline::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_baseline_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_synth_registry_path::path E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_transparent_decorators::value
def test_run_lint_outputs(capsys, tmp_path: Path) -> None:
    dataflow_audit = _load()
    sample = tmp_path / "never_mod.py"
    sample.write_text(
        "from gabion.invariants import never\n"
        "\n"
        "def f(flag):\n"
        "    if flag:\n"
        "        never('boom')\n"
    )
    report_path = tmp_path / "report.md"
    code = dataflow_audit.run(
        [
            str(sample),
            "--root",
            str(tmp_path),
            "--report",
            str(report_path),
            "--lint",
        ]
    )
    out = capsys.readouterr().out
    assert "GABION_NEVER_INVARIANT" in out
    assert code == 0


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._apply_baseline::baseline_allowlist E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::bundle_sites_by_path,coherence_witnesses,constant_smells,context_suggestions,deadness_witnesses,decision_surfaces,decision_warnings,exception_obligations,fingerprint_matches,fingerprint_provenance,fingerprint_synth,fingerprint_warnings,forest,groups_by_path,handledness_witnesses,invariant_propositions,max_components,never_invariants,rewrite_plans,type_ambiguities,type_callsite_evidence,type_suggestions,unused_arg_smells,value_decision_rewrites,value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_dot::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._load_baseline::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_baseline_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_synth_registry_path::path E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_transparent_decorators::value
def test_run_decision_snapshot_writes_file(tmp_path: Path) -> None:
    dataflow_audit = _load()
    sample = tmp_path / "typed.py"
    _write_typed_bundle_code(sample)
    decision_path = tmp_path / "decision.json"
    code = dataflow_audit.run(
        [
            str(sample),
            "--root",
            str(tmp_path),
            "--emit-decision-snapshot",
            str(decision_path),
        ]
    )
    assert code == 0
    assert decision_path.exists()


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._apply_baseline::baseline_allowlist E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::bundle_sites_by_path,coherence_witnesses,constant_smells,context_suggestions,deadness_witnesses,decision_surfaces,decision_warnings,exception_obligations,fingerprint_matches,fingerprint_provenance,fingerprint_synth,fingerprint_warnings,forest,groups_by_path,handledness_witnesses,invariant_propositions,max_components,never_invariants,rewrite_plans,type_ambiguities,type_callsite_evidence,type_suggestions,unused_arg_smells,value_decision_rewrites,value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_dot::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._load_baseline::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_baseline_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_synth_registry_path::path E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_transparent_decorators::value
def test_run_synth_registry_path_invalid_json(tmp_path: Path) -> None:
    dataflow_audit = _load()
    sample = tmp_path / "typed.py"
    _write_typed_bundle_code(sample)
    synth_path = tmp_path / "synth.json"
    synth_path.write_text("{invalid")
    config_path = tmp_path / "gabion.toml"
    config_path.write_text(
        "[fingerprints]\n"
        "user_context = [\"int\"]\n"
        "synth_min_occurrences = \"bad\"\n"
        f"synth_registry_path = \"{synth_path}\"\n"
    )
    code = dataflow_audit.run(
        [
            str(sample),
            "--root",
            str(tmp_path),
            "--config",
            str(config_path),
        ]
    )
    assert code == 0


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._apply_baseline::baseline_allowlist E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::bundle_sites_by_path,coherence_witnesses,constant_smells,context_suggestions,deadness_witnesses,decision_surfaces,decision_warnings,exception_obligations,fingerprint_matches,fingerprint_provenance,fingerprint_synth,fingerprint_warnings,forest,groups_by_path,handledness_witnesses,invariant_propositions,max_components,never_invariants,rewrite_plans,type_ambiguities,type_callsite_evidence,type_suggestions,unused_arg_smells,value_decision_rewrites,value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_dot::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._load_baseline::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_baseline_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_synth_registry_path::path E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_transparent_decorators::value
def test_run_synth_registry_path_missing_latest(tmp_path: Path) -> None:
    dataflow_audit = _load()
    sample = tmp_path / "typed.py"
    _write_typed_bundle_code(sample)
    config_path = tmp_path / "gabion.toml"
    config_path.write_text(
        "[fingerprints]\n"
        "user_context = [\"int\"]\n"
        "synth_registry_path = \"out/LATEST/fingerprint_synth.json\"\n"
    )
    code = dataflow_audit.run(
        [
            str(sample),
            "--root",
            str(tmp_path),
            "--config",
            str(config_path),
        ]
    )
    assert code == 0


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._apply_baseline::baseline_allowlist E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::bundle_sites_by_path,coherence_witnesses,constant_smells,context_suggestions,deadness_witnesses,decision_surfaces,decision_warnings,exception_obligations,fingerprint_matches,fingerprint_provenance,fingerprint_synth,fingerprint_warnings,forest,groups_by_path,handledness_witnesses,invariant_propositions,max_components,never_invariants,rewrite_plans,type_ambiguities,type_callsite_evidence,type_suggestions,unused_arg_smells,value_decision_rewrites,value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_dot::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._load_baseline::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_baseline_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_synth_registry_path::path E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_transparent_decorators::value
def test_run_synth_registry_path_valid_json(tmp_path: Path) -> None:
    dataflow_audit = _load()
    sample = tmp_path / "typed.py"
    _write_typed_bundle_code(sample)
    synth_path = tmp_path / "synth.json"
    synth_path.write_text(
        "{\"version\": \"synth@1\", \"min_occurrences\": 2, \"entries\": []}"
    )
    config_path = tmp_path / "gabion.toml"
    config_path.write_text(
        "[fingerprints]\n"
        "user_context = [\"int\"]\n"
        f"synth_registry_path = \"{synth_path}\"\n"
    )
    code = dataflow_audit.run(
        [
            str(sample),
            "--root",
            str(tmp_path),
            "--config",
            str(config_path),
        ]
    )
    assert code == 0


# gabion:evidence E:call_cluster::dataflow_audit.py::gabion.analysis.dataflow_audit.run::test_dataflow_run_edges.py::tests.test_dataflow_run_edges._load::test_dataflow_run_edges.py::tests.test_dataflow_run_edges._write_bundle_code
def test_run_refactor_plan_json_stdout(tmp_path: Path, capsys) -> None:
    dataflow_audit = _load()
    sample = tmp_path / "sample.py"
    _write_bundle_code(sample)
    code = dataflow_audit.run(
        [
            str(sample),
            "--root",
            str(tmp_path),
            "--refactor-plan-json",
            "-",
        ]
    )
    captured = capsys.readouterr().out
    assert captured.strip()
    assert code == 0


# gabion:evidence E:call_cluster::dataflow_audit.py::gabion.analysis.dataflow_audit.run::test_dataflow_run_edges.py::tests.test_dataflow_run_edges._load::test_dataflow_run_edges.py::tests.test_dataflow_run_edges._write_bundle_code
def test_run_refactor_plan_without_json(tmp_path: Path, capsys) -> None:
    dataflow_audit = _load()
    sample = tmp_path / "sample.py"
    _write_bundle_code(sample)
    code = dataflow_audit.run(
        [
            str(sample),
            "--root",
            str(tmp_path),
            "--refactor-plan",
            "--exclude",
            "in, .venv",
            "--ignore-params",
            "x, y",
        ]
    )
    captured = capsys.readouterr().out
    assert "bundle" in captured
    assert code == 0


# gabion:evidence E:call_cluster::dataflow_audit.py::gabion.analysis.dataflow_audit.run::test_dataflow_run_edges.py::tests.test_dataflow_run_edges._load::test_dataflow_run_edges.py::tests.test_dataflow_run_edges._write_bundle_code
def test_run_dot_stdout_returns_early(tmp_path: Path, capsys) -> None:
    dataflow_audit = _load()
    sample = tmp_path / "sample.py"
    _write_bundle_code(sample)
    code = dataflow_audit.run(
        [
            str(sample),
            "--root",
            str(tmp_path),
            "--dot",
            "-",
        ]
    )
    captured = capsys.readouterr().out
    assert "digraph" in captured
    assert code == 0


# gabion:evidence E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._apply_baseline::baseline_allowlist E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_dot::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::ambiguity_witnesses,bundle_sites_by_path,coherence_witnesses,constant_smells,context_suggestions,deadness_witnesses,decision_surfaces,decision_warnings,exception_obligations,fingerprint_matches,fingerprint_provenance,fingerprint_synth,fingerprint_warnings,forest,groups_by_path,handledness_witnesses,invariant_propositions,max_components,never_invariants,rewrite_plans,type_ambiguities,type_callsite_evidence,type_suggestions,unused_arg_smells,value_decision_rewrites,value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._load_baseline::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_transparent_decorators::value E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_baseline_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_synth_registry_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_ambiguities,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions
def test_run_fail_on_violations_with_baseline(tmp_path: Path, capsys) -> None:
    dataflow_audit = _load()
    sample = tmp_path / "sample.py"
    _write_bundle_code(sample)
    baseline_path = tmp_path / "baseline.txt"
    baseline_path.write_text("preexisting\n")
    code = dataflow_audit.run(
        [
            str(sample),
            "--root",
            str(tmp_path),
            "--baseline",
            str(baseline_path),
            "--fail-on-violations",
        ]
    )
    captured = capsys.readouterr().out
    assert "bundle" in captured
    assert code == 1


# gabion:evidence E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._apply_baseline::baseline_allowlist E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_dot::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::ambiguity_witnesses,bundle_sites_by_path,coherence_witnesses,constant_smells,context_suggestions,deadness_witnesses,decision_surfaces,decision_warnings,exception_obligations,fingerprint_matches,fingerprint_provenance,fingerprint_synth,fingerprint_warnings,forest,groups_by_path,handledness_witnesses,invariant_propositions,max_components,never_invariants,rewrite_plans,type_ambiguities,type_callsite_evidence,type_suggestions,unused_arg_smells,value_decision_rewrites,value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._load_baseline::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_transparent_decorators::value E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_baseline_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_synth_registry_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_ambiguities,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions
def test_run_fail_on_violations_baseline_write(tmp_path: Path) -> None:
    dataflow_audit = _load()
    sample = tmp_path / "sample.py"
    _write_bundle_code(sample)
    baseline_path = tmp_path / "baseline.txt"
    code = dataflow_audit.run(
        [
            str(sample),
            "--root",
            str(tmp_path),
            "--baseline",
            str(baseline_path),
            "--baseline-write",
            "--fail-on-violations",
        ]
    )
    assert baseline_path.exists()
    assert code == 0
