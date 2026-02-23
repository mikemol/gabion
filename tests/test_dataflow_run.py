from __future__ import annotations

from pathlib import Path

from gabion.analysis import dataflow_audit


def _write_sample_code(path: Path) -> None:
    path.write_text(
        "def helper(x, y):\n"
        "    return x + y\n\n"
        "def alpha(a, b, *args, **kwargs):\n"
        "    c = a\n"
        "    d, e = (b, b)\n"
        "    return helper(c, d)\n"
    )


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._apply_baseline::baseline_allowlist E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::bundle_sites_by_path,coherence_witnesses,constant_smells,context_suggestions,deadness_witnesses,decision_surfaces,decision_warnings,exception_obligations,fingerprint_matches,fingerprint_provenance,fingerprint_synth,fingerprint_warnings,forest,groups_by_path,handledness_witnesses,invariant_propositions,max_components,never_invariants,rewrite_plans,type_ambiguities,type_callsite_evidence,type_suggestions,unused_arg_smells,value_decision_rewrites,value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_dot::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._load_baseline::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_baseline_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_synth_registry_path::path E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_transparent_decorators::value
def test_run_generates_outputs(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    _write_sample_code(sample)
    report = tmp_path / "report.md"
    dot = tmp_path / "graph.dot"
    plan = tmp_path / "plan.json"
    protocols = tmp_path / "protocols.py"
    refactor = tmp_path / "refactor.json"
    snapshot = tmp_path / "structure.json"
    metrics = tmp_path / "metrics.json"
    argv = [
        str(tmp_path),
        "--root",
        str(tmp_path),
        "--report",
        str(report),
        "--dot",
        str(dot),
        "--synthesis-plan",
        str(plan),
        "--synthesis-protocols",
        str(protocols),
        "--synthesis-protocols-kind",
        "protocol",
        "--synthesis-min-bundle-size",
        "1",
        "--synthesis-allow-singletons",
        "--refactor-plan",
        "--refactor-plan-json",
        str(refactor),
        "--emit-structure-tree",
        str(snapshot),
        "--emit-structure-metrics",
        str(metrics),
        "--type-audit-report",
        "--type-audit-max",
        "5",
        "--transparent-decorators",
        "decorator_a,decorator_b",
    ]
    code = dataflow_audit.run(argv)
    assert code == 0
    assert report.exists()
    assert dot.exists()
    assert plan.exists()
    assert protocols.exists()
    assert refactor.exists()
    assert snapshot.exists()
    assert metrics.exists()


# gabion:evidence E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._apply_baseline::baseline_allowlist E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_dot::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::ambiguity_witnesses,bundle_sites_by_path,coherence_witnesses,constant_smells,context_suggestions,deadness_witnesses,decision_surfaces,decision_warnings,exception_obligations,fingerprint_matches,fingerprint_provenance,fingerprint_synth,fingerprint_warnings,forest,groups_by_path,handledness_witnesses,invariant_propositions,max_components,never_invariants,rewrite_plans,type_ambiguities,type_callsite_evidence,type_suggestions,unused_arg_smells,value_decision_rewrites,value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._load_baseline::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_transparent_decorators::value E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_baseline_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_synth_registry_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_ambiguities,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions
def test_run_baseline_write_and_apply(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    _write_sample_code(sample)
    baseline = tmp_path / "baseline.txt"
    report = tmp_path / "report.md"
    code = dataflow_audit.run(
        [
            str(tmp_path),
            "--root",
            str(tmp_path),
            "--baseline",
            str(baseline),
            "--baseline-write",
            "--report",
            str(report),
            "--fail-on-violations",
        ]
    )
    assert code == 0
    assert baseline.exists()

    code = dataflow_audit.run(
        [
            str(tmp_path),
            "--root",
            str(tmp_path),
            "--baseline",
            str(baseline),
            "--report",
            str(report),
            "--fail-on-violations",
        ]
    )
    assert code == 0


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._apply_baseline::baseline_allowlist E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::bundle_sites_by_path,coherence_witnesses,constant_smells,context_suggestions,deadness_witnesses,decision_surfaces,decision_warnings,exception_obligations,fingerprint_matches,fingerprint_provenance,fingerprint_synth,fingerprint_warnings,forest,groups_by_path,handledness_witnesses,invariant_propositions,max_components,never_invariants,rewrite_plans,type_ambiguities,type_callsite_evidence,type_suggestions,unused_arg_smells,value_decision_rewrites,value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_dot::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._load_baseline::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_baseline_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_synth_registry_path::path E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_transparent_decorators::value
def test_run_invalid_strictness_from_config(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    _write_sample_code(sample)
    config = tmp_path / "gabion.toml"
    config.write_text("[dataflow]\nstrictness = \"weird\"\n")
    code = dataflow_audit.run(
        [
            str(tmp_path),
            "--root",
            str(tmp_path),
            "--config",
            str(config),
            "--report",
            str(tmp_path / "report.md"),
        ]
    )
    assert code == 0


def _empty_analysis(dataflow_audit, tmp_path: Path):
    return dataflow_audit.AnalysisResult(
        groups_by_path={},
        param_spans_by_path={},
        bundle_sites_by_path={},
        type_suggestions=[],
        type_ambiguities=[],
        type_callsite_evidence=[],
        constant_smells=[],
        unused_arg_smells=[],
        forest=dataflow_audit.Forest(),
    )


# gabion:evidence E:call_footprint::tests/test_dataflow_run.py::test_run_impl_type_audit_branches_via_di::dataflow_audit.py::gabion.analysis.dataflow_audit._build_parser::dataflow_audit.py::gabion.analysis.dataflow_audit._run_impl::test_dataflow_run.py::tests.test_dataflow_run._empty_analysis
def test_run_impl_type_audit_branches_via_di(tmp_path: Path) -> None:
    parser = dataflow_audit._build_parser()
    args = parser.parse_args(
        [
            str(tmp_path),
            "--root",
            str(tmp_path),
            "--type-audit",
            "--exclude",
            ",,keep",
        ]
    )
    analysis = _empty_analysis(dataflow_audit, tmp_path)
    analysis.type_suggestions = ["use Optional[int]"]
    analysis.type_ambiguities = ["x receives conflicting constraints"]
    code = dataflow_audit._run_impl(
        args,
        analyze_paths_fn=lambda *_args, **_kwargs: analysis,
    )
    assert code == 0


# gabion:evidence E:call_footprint::tests/test_dataflow_run.py::test_run_impl_decision_snapshot_non_terminal_and_baseline_no_new_violations::dataflow_audit.py::gabion.analysis.dataflow_audit._build_parser::dataflow_audit.py::gabion.analysis.dataflow_audit._run_impl::test_dataflow_run.py::tests.test_dataflow_run._empty_analysis
def test_run_impl_decision_snapshot_non_terminal_and_baseline_no_new_violations(
    tmp_path: Path,
) -> None:
    parser = dataflow_audit._build_parser()
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("same-violation\n", encoding="utf-8")
    snapshot = tmp_path / "decision.json"
    dot = tmp_path / "graph.dot"
    args = parser.parse_args(
        [
            str(tmp_path),
            "--root",
            str(tmp_path),
            "--emit-decision-snapshot",
            str(snapshot),
            "--dot",
            str(dot),
            "--fail-on-violations",
            "--baseline",
            str(baseline),
        ]
    )

    config = tmp_path / "gabion.toml"
    config.write_text("[fingerprints]\nsynth_min_occurrences = 1\n", encoding="utf-8")
    args.config = str(config)

    analysis = _empty_analysis(dataflow_audit, tmp_path)
    code = dataflow_audit._run_impl(
        args,
        analyze_paths_fn=lambda *_args, **_kwargs: analysis,
        compute_violations_fn=lambda *_args, **_kwargs: ["same-violation"],
    )
    assert code == 0
    assert snapshot.exists()
    assert dot.exists()


# gabion:evidence E:call_footprint::tests/test_dataflow_run.py::test_run_impl_type_audit_empty_lists_and_terminal_return::dataflow_audit.py::gabion.analysis.dataflow_audit._build_parser::dataflow_audit.py::gabion.analysis.dataflow_audit._run_impl::test_dataflow_run.py::tests.test_dataflow_run._empty_analysis
def test_run_impl_type_audit_empty_lists_and_terminal_return(tmp_path: Path) -> None:
    parser = dataflow_audit._build_parser()
    args = parser.parse_args(
        [
            str(tmp_path),
            "--root",
            str(tmp_path),
            "--type-audit",
        ]
    )
    analysis = _empty_analysis(dataflow_audit, tmp_path)
    code = dataflow_audit._run_impl(
        args,
        analyze_paths_fn=lambda *_args, **_kwargs: analysis,
    )
    assert code == 0


# gabion:evidence E:call_footprint::tests/test_dataflow_run.py::test_run_impl_fail_on_violations_baseline_and_no_baseline_edges::dataflow_audit.py::gabion.analysis.dataflow_audit._build_parser::dataflow_audit.py::gabion.analysis.dataflow_audit._run_impl::test_dataflow_run.py::tests.test_dataflow_run._empty_analysis
def test_run_impl_fail_on_violations_baseline_and_no_baseline_edges(tmp_path: Path) -> None:
    parser = dataflow_audit._build_parser()
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("known\n", encoding="utf-8")

    args_with_baseline = parser.parse_args(
        [
            str(tmp_path),
            "--root",
            str(tmp_path),
            "--fail-on-violations",
            "--baseline",
            str(baseline),
        ]
    )
    analysis = _empty_analysis(dataflow_audit, tmp_path)
    code = dataflow_audit._run_impl(
        args_with_baseline,
        analyze_paths_fn=lambda *_args, **_kwargs: analysis,
        compute_violations_fn=lambda *_args, **_kwargs: ["known"],
    )
    assert code == 0

    args_without_baseline = parser.parse_args(
        [
            str(tmp_path),
            "--root",
            str(tmp_path),
            "--fail-on-violations",
        ]
    )
    code = dataflow_audit._run_impl(
        args_without_baseline,
        analyze_paths_fn=lambda *_args, **_kwargs: analysis,
        compute_violations_fn=lambda *_args, **_kwargs: [],
    )
    assert code == 0


# gabion:evidence E:call_footprint::tests/test_dataflow_run.py::test_run_impl_fingerprint_spec_with_empty_index_branch::dataflow_audit.py::gabion.analysis.dataflow_audit._build_parser::dataflow_audit.py::gabion.analysis.dataflow_audit._run_impl::test_dataflow_run.py::tests.test_dataflow_run._empty_analysis
def test_run_impl_fingerprint_spec_with_empty_index_branch(tmp_path: Path) -> None:
    parser = dataflow_audit._build_parser()
    config = tmp_path / "gabion.toml"
    config.write_text(
        "[fingerprints]\n"
        "custom = []\n"
        "synth_registry_path = \"missing.json\"\n",
        encoding="utf-8",
    )
    args = parser.parse_args(
        [
            str(tmp_path),
            "--root",
            str(tmp_path),
            "--config",
            str(config),
        ]
    )
    analysis = _empty_analysis(dataflow_audit, tmp_path)
    code = dataflow_audit._run_impl(
        args,
        analyze_paths_fn=lambda *_args, **_kwargs: analysis,
    )
    assert code == 0
