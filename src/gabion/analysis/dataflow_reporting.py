# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass

_BOUND = False


def _bind_audit_symbols() -> None:
    global _BOUND
    if _BOUND:
        return
    from gabion.analysis import dataflow_audit as _audit

    module_globals = globals()
    for name, value in _audit.__dict__.items():
        module_globals.setdefault(name, value)
    _BOUND = True


@dataclass
class _ReportEmitState:
    lines: list[str]
    violations: list[str]


def _default_parse_witness_contract_violations() -> list[str]:
    _bind_audit_symbols()
    return _parse_witness_contract_violations()


def _append_report_tail_sections(
    *,
    state: _ReportEmitState,
    report: ReportCarrier,
    root: Path,
    file_paths: list[Path],
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    execution_pattern_suggestions: list[str],
    parse_witness_contract_violations_fn: Callable[[], list[str]],
    projected: Callable[[str, Iterable[str]], list[str]],
    start_section: Callable[[str], None],
) -> None:
    type_suggestions = report.type_suggestions
    type_ambiguities = report.type_ambiguities
    type_callsite_evidence = report.type_callsite_evidence
    constant_smells = report.constant_smells
    unused_arg_smells = report.unused_arg_smells
    deadness_witnesses = report.deadness_witnesses
    coherence_witnesses = report.coherence_witnesses
    rewrite_plans = report.rewrite_plans
    exception_obligations = report.exception_obligations
    never_invariants = report.never_invariants
    ambiguity_witnesses = report.ambiguity_witnesses
    handledness_witnesses = report.handledness_witnesses
    decision_surfaces = report.decision_surfaces
    value_decision_surfaces = report.value_decision_surfaces
    decision_warnings = report.decision_warnings
    fingerprint_warnings = report.fingerprint_warnings
    fingerprint_matches = report.fingerprint_matches
    fingerprint_synth = report.fingerprint_synth
    fingerprint_provenance = report.fingerprint_provenance
    context_suggestions = report.context_suggestions
    invariant_propositions = report.invariant_propositions
    value_decision_rewrites = report.value_decision_rewrites
    deadline_obligations = report.deadline_obligations
    parse_failure_witnesses = report.parse_failure_witnesses
    resumability_obligations = report.resumability_obligations
    incremental_report_obligations = report.incremental_report_obligations

    if state.violations:
        start_section("violations")
        state.lines.append("Violations:")
        state.lines.append("```")
        state.lines.extend(projected("violations", state.violations))
        state.lines.append("```")
    if type_suggestions or type_ambiguities:
        start_section("type_flow")
        state.lines.append("Type-flow audit:")
        type_mermaid = _render_type_mermaid(type_suggestions or [], type_ambiguities or [])
        state.lines.extend(projected("type_flow_mermaid", type_mermaid.splitlines()))
        if type_suggestions:
            state.lines.append("Type tightening candidates:")
            state.lines.append("```")
            state.lines.extend(projected("type_suggestions", type_suggestions))
            state.lines.append("```")
        if type_ambiguities:
            state.lines.append("Type ambiguities (conflicting downstream expectations):")
            state.lines.append("```")
            state.lines.extend(projected("type_ambiguities", type_ambiguities))
            state.lines.append("```")
        if type_callsite_evidence:
            state.lines.append("Type-flow callsite evidence:")
            state.lines.append("```")
            state.lines.extend(projected("type_callsite_evidence", type_callsite_evidence))
            state.lines.append("```")
    if constant_smells:
        start_section("constant_smells")
        state.lines.append("Constant-propagation smells (non-test call sites):")
        state.lines.append("```")
        state.lines.extend(projected("constant_smells", constant_smells))
        state.lines.append("```")
    if unused_arg_smells:
        start_section("unused_arg_smells")
        state.lines.append("Unused-argument smells (non-test call sites):")
        state.lines.append("```")
        state.lines.extend(projected("unused_arg_smells", unused_arg_smells))
        state.lines.append("```")
    if deadness_witnesses:
        summary = _summarize_deadness_witnesses(deadness_witnesses)
        start_section("deadness_summary")
        state.lines.append("Deadness evidence:")
        state.lines.append("```")
        state.lines.extend(projected("deadness_summary", summary))
        state.lines.append("```")
    if coherence_witnesses:
        summary = _summarize_coherence_witnesses(coherence_witnesses)
        start_section("coherence_summary")
        state.lines.append("Coherence evidence:")
        state.lines.append("```")
        state.lines.extend(projected("coherence_summary", summary))
        state.lines.append("```")
    if rewrite_plans:
        summary = _summarize_rewrite_plans(rewrite_plans)
        start_section("rewrite_plans_summary")
        state.lines.append("Rewrite plans:")
        state.lines.append("```")
        state.lines.extend(projected("rewrite_plans_summary", summary))
        state.lines.append("```")
    if never_invariants:
        summary = _summarize_never_invariants(never_invariants)
        start_section("never_invariants_summary")
        state.lines.append("Never invariants:")
        state.lines.append("```")
        state.lines.extend(projected("never_invariants_summary", summary))
        state.lines.append("```")
    if ambiguity_witnesses:
        summary = _summarize_call_ambiguities(ambiguity_witnesses)
        start_section("ambiguity_summary")
        state.lines.append("Ambiguities:")
        state.lines.append("```")
        state.lines.extend(projected("ambiguity_summary", summary))
        state.lines.append("```")
    if exception_obligations:
        summary = _summarize_exception_obligations(exception_obligations)
        start_section("exception_obligations_summary")
        state.lines.append("Exception obligations:")
        state.lines.append("```")
        state.lines.extend(projected("exception_obligations_summary", summary))
        state.lines.append("```")
        protocol_evidence = _exception_protocol_evidence(exception_obligations)
        if protocol_evidence:
            start_section("exception_protocol_evidence")
            state.lines.append("Exception protocol evidence:")
            state.lines.append("```")
            state.lines.extend(projected("exception_protocol_evidence", protocol_evidence))
            state.lines.append("```")
        protocol_warnings = _exception_protocol_warnings(exception_obligations)
        if protocol_warnings:
            start_section("exception_protocol_warnings")
            state.lines.append("Exception protocol violations:")
            state.lines.append("```")
            state.lines.extend(projected("exception_protocol_warnings", protocol_warnings))
            state.lines.append("```")
            state.violations.extend(protocol_warnings)
    if handledness_witnesses:
        summary = _summarize_handledness_witnesses(handledness_witnesses)
        start_section("handledness_summary")
        state.lines.append("Handledness evidence:")
        state.lines.append("```")
        state.lines.extend(projected("handledness_summary", summary))
        state.lines.append("```")
    if deadline_obligations:
        summary = _summarize_deadline_obligations(
            deadline_obligations,
            forest=report.forest,
        )
        start_section("deadline_summary")
        state.lines.append("Deadline propagation:")
        state.lines.append("```")
        state.lines.extend(projected("deadline_summary", summary))
        state.lines.append("```")
    if resumability_obligations:
        summary = _summarize_runtime_obligations(resumability_obligations)
        start_section("resumability_obligations")
        state.lines.append("Resumability obligations:")
        state.lines.append("```")
        state.lines.extend(projected("resumability_obligations", summary))
        state.lines.append("```")
        state.violations.extend(_runtime_obligation_violation_lines(resumability_obligations))
    if incremental_report_obligations:
        summary = _summarize_runtime_obligations(incremental_report_obligations)
        start_section("incremental_report_obligations")
        state.lines.append("Incremental report obligations:")
        state.lines.append("```")
        state.lines.extend(projected("incremental_report_obligations", summary))
        state.lines.append("```")
        state.violations.extend(
            _runtime_obligation_violation_lines(incremental_report_obligations)
        )
    if parse_failure_witnesses:
        summary = _summarize_parse_failure_witnesses(parse_failure_witnesses)
        start_section("parse_failure_witnesses")
        state.lines.append("Parse failure witnesses:")
        state.lines.append("```")
        state.lines.extend(projected("parse_failure_witnesses", summary))
        state.lines.append("```")
        state.violations.extend(_parse_failure_violation_lines(parse_failure_witnesses))
    contract_violations = parse_witness_contract_violations_fn()
    if contract_violations:
        start_section("parse_witness_contract_violations")
        state.lines.append("Parse witness contract violations:")
        state.lines.append("```")
        state.lines.extend(projected("parse_witness_contract_violations", contract_violations))
        state.lines.append("```")
        state.violations.extend(contract_violations)
    raw_sorted_violations = _raw_sorted_contract_violations(
        file_paths,
        parse_failure_witnesses=parse_failure_witnesses,
    )
    if raw_sorted_violations:
        start_section("order_contract_violations")
        state.lines.append("Order contract violations:")
        state.lines.append("```")
        state.lines.extend(projected("order_contract_violations", raw_sorted_violations))
        state.lines.append("```")
        state.violations.extend(raw_sorted_violations)
    pattern_instances = _pattern_schema_matches(
        groups_by_path=groups_by_path,
        include_execution=True,
    )
    if not execution_pattern_suggestions:
        execution_pattern_suggestions = _pattern_schema_suggestions_from_instances(
            pattern_instances
        )
    if execution_pattern_suggestions:
        start_section("execution_pattern_suggestions")
        state.lines.append("Execution pattern opportunities:")
        state.lines.append("```")
        state.lines.extend(
            projected("execution_pattern_suggestions", execution_pattern_suggestions)
        )
        state.lines.append("```")
    pattern_residue = _pattern_schema_residue_entries(pattern_instances)
    if pattern_residue:
        start_section("pattern_schema_residue")
        state.lines.append("Pattern schema residue (non-blocking):")
        state.lines.append("```")
        state.lines.extend(
            projected(
                "pattern_schema_residue",
                _pattern_schema_residue_lines(pattern_residue),
            )
        )
        state.lines.append("```")
    if decision_surfaces:
        start_section("decision_surfaces")
        state.lines.append("Decision surface candidates (direct param use in conditionals):")
        state.lines.append("```")
        state.lines.extend(projected("decision_surfaces", decision_surfaces))
        state.lines.append("```")
    if value_decision_surfaces:
        start_section("value_decision_surfaces")
        state.lines.append("Value-encoded decision surface candidates (branchless control):")
        state.lines.append("```")
        state.lines.extend(projected("value_decision_surfaces", value_decision_surfaces))
        state.lines.append("```")
    if value_decision_rewrites:
        start_section("value_decision_rewrites")
        state.lines.append("Value-encoded decision rebranch suggestions:")
        state.lines.append("```")
        state.lines.extend(projected("value_decision_rewrites", value_decision_rewrites))
        state.lines.append("```")
    if decision_warnings:
        start_section("decision_warnings")
        state.lines.append("Decision tier warnings:")
        state.lines.append("```")
        state.lines.extend(projected("decision_warnings", decision_warnings))
        state.lines.append("```")
        state.violations.extend(decision_warnings)
    if fingerprint_warnings:
        start_section("fingerprint_warnings")
        state.lines.append("Fingerprint warnings:")
        state.lines.append("```")
        state.lines.extend(projected("fingerprint_warnings", fingerprint_warnings))
        state.lines.append("```")
    if fingerprint_matches:
        start_section("fingerprint_matches")
        state.lines.append("Fingerprint matches:")
        state.lines.append("```")
        state.lines.extend(projected("fingerprint_matches", fingerprint_matches))
        state.lines.append("```")
    if fingerprint_synth:
        start_section("fingerprint_synthesis")
        state.lines.append("Fingerprint synthesis:")
        state.lines.append("```")
        state.lines.extend(projected("fingerprint_synthesis", fingerprint_synth))
        state.lines.append("```")
    if fingerprint_provenance:
        provenance_summary = _summarize_fingerprint_provenance(fingerprint_provenance)
        start_section("fingerprint_provenance_summary")
        state.lines.append("Packed derivation view (ASPF provenance):")
        state.lines.append("```")
        state.lines.extend(projected("fingerprint_provenance_summary", provenance_summary))
        state.lines.append("```")
    if invariant_propositions:
        start_section("invariant_propositions")
        state.lines.append("Invariant propositions:")
        state.lines.append("```")
        state.lines.extend(
            projected(
                "invariant_propositions",
                _format_invariant_propositions(invariant_propositions),
            )
        )
        state.lines.append("```")
    if context_suggestions:
        start_section("context_suggestions")
        state.lines.append("Contextvar/ambient rewrite suggestions:")
        state.lines.append("```")
        state.lines.extend(projected("context_suggestions", context_suggestions))
        state.lines.append("```")
    schema_surfaces = find_anonymous_schema_surfaces(file_paths, project_root=root)
    if schema_surfaces:
        start_section("schema_surfaces")
        state.lines.append("Anonymous schema surfaces (dict[str, object] payloads):")
        state.lines.append("```")
        schema_lines = [surface.format() for surface in schema_surfaces[:50]]
        state.lines.extend(projected("schema_surfaces", schema_lines))
        if len(schema_surfaces) > 50:
            state.lines.append(f"... {len(schema_surfaces) - 50} more")
        state.lines.append("```")


def emit_report(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    max_components: int,
    *,
    report: ReportCarrier,
    execution_pattern_suggestions: tuple[str, ...] = (),
    parse_witness_contract_violations_fn: Callable[[], list[str]] = _default_parse_witness_contract_violations,
) -> tuple[str, list[str]]:
    _bind_audit_symbols()
    check_deadline()
    forest = report.forest
    bundle_sites_by_path = report.bundle_sites_by_path
    type_suggestions = report.type_suggestions
    type_ambiguities = report.type_ambiguities
    type_callsite_evidence = report.type_callsite_evidence
    constant_smells = report.constant_smells
    unused_arg_smells = report.unused_arg_smells
    deadness_witnesses = report.deadness_witnesses
    coherence_witnesses = report.coherence_witnesses
    rewrite_plans = report.rewrite_plans
    exception_obligations = report.exception_obligations
    never_invariants = report.never_invariants
    ambiguity_witnesses = report.ambiguity_witnesses
    handledness_witnesses = report.handledness_witnesses
    decision_surfaces = report.decision_surfaces
    value_decision_surfaces = report.value_decision_surfaces
    decision_warnings = report.decision_warnings
    fingerprint_warnings = report.fingerprint_warnings
    fingerprint_matches = report.fingerprint_matches
    fingerprint_synth = report.fingerprint_synth
    fingerprint_provenance = report.fingerprint_provenance
    context_suggestions = report.context_suggestions
    invariant_propositions = report.invariant_propositions
    value_decision_rewrites = report.value_decision_rewrites
    deadline_obligations = report.deadline_obligations
    parse_failure_witnesses = report.parse_failure_witnesses
    resumability_obligations = report.resumability_obligations
    incremental_report_obligations = report.incremental_report_obligations
    has_bundles = _has_bundles(groups_by_path)
    if groups_by_path:
        common = os.path.commonpath([str(p) for p in groups_by_path])
        root = Path(common)
    else:
        root = Path(".")
    # Use the analyzed file set (not a repo-wide rglob) so reports and schema
    # audits don't accidentally ingest virtualenvs or unrelated files.
    file_paths = (
        sort_once(
            groups_by_path.keys(),
            source="_emit_report.file_paths",
            key=lambda path: str(path),
        )
        if groups_by_path
        else []
    )
    projection = _bundle_projection_from_forest(forest, file_paths=file_paths) if has_bundles else None
    components = (
        _connected_components(projection.nodes, projection.adj)
        if projection is not None
        else []
    )
    bundle_site_index = (
        _bundle_site_index(groups_by_path, bundle_sites_by_path)
        if bundle_sites_by_path
        else {}
    )
    lines = [
        _report_section_marker("intro"),
        "<!-- dataflow-grammar -->",
        "Dataflow grammar audit (observed forwarding bundles).",
        "",
    ]
    report_run_id = f"report_{len(forest.nodes)}_{len(forest.alts)}"

    def _projected(section_id: str, values: Iterable[str]) -> list[str]:
        return _project_report_section_lines(
            forest=forest,
            section_key=_ReportSectionKey(run_id=report_run_id, section=section_id),
            lines=values,
        )

    def _start_section(section_id: str) -> None:
        lines.append(_report_section_marker(section_id))

    violations: list[str] = []
    _start_section("components")
    if not components:
        lines.append("No bundle components detected.")
    else:
        if len(components) > max_components:
            lines.append(
                f"Showing top {max_components} components of {len(components)}."
            )
        for idx, comp in enumerate(components[:max_components], start=1):
            check_deadline()
            lines.append(f"### Component {idx}")
            mermaid, summary = _render_mermaid_component(
                projection.nodes,
                projection.bundle_map,
                projection.bundle_counts,
                projection.adj,
                comp,
                projection.declared_global,
                projection.declared_by_path,
                projection.documented_by_path,
            )
            lines.extend(_projected(f"component_{idx}_mermaid", mermaid.splitlines()))
            lines.append("")
            lines.append("Summary:")
            lines.append("```")
            lines.extend(_projected(f"component_{idx}_summary", summary.splitlines()))
            lines.append("```")
            lines.append("")
            if bundle_sites_by_path:
                evidence = _render_component_callsite_evidence(
                    component=comp,
                    nodes=projection.nodes,
                    bundle_map=projection.bundle_map,
                    bundle_counts=projection.bundle_counts,
                    adj=projection.adj,
                    documented_by_path=projection.documented_by_path,
                    declared_global=projection.declared_global,
                    bundle_site_index=bundle_site_index,
                    root=projection.root,
                    path_lookup=projection.path_lookup,
                )
                if evidence:
                    lines.append("Callsite evidence (undocumented bundles):")
                    lines.append("```")
                    lines.extend(_projected(f"component_{idx}_callsite_evidence", evidence))
                    lines.append("```")
                    lines.append("")
            for line in summary.splitlines():
                # Violation strings are semantic objects; avoid leaking markdown
                # bullets into baseline keys.
                check_deadline()
                candidate = line.strip()
                if candidate.startswith("- "):
                    candidate = candidate[2:].strip()
                tier12 = "(tier-1," in candidate or "(tier-2," in candidate
                if "(tier-3, undocumented)" in candidate or (
                    tier12 and "undocumented" in candidate
                ):
                    violations.append(candidate)
    if deadline_obligations:
        deadline_violations: list[str] = []
        for entry in deadline_obligations:
            check_deadline()
            if entry.get("status") != "VIOLATION":
                continue
            site = entry.get("site", {}) or {}
            path = site.get("path", "?")
            function = site.get("function", "?")
            bundle = site.get("bundle", [])
            status = entry.get("status", "UNKNOWN")
            kind = entry.get("kind", "?")
            detail = entry.get("detail", "")
            deadline_violations.append(
                f"{path}:{function} bundle={bundle} status={status} kind={kind} {detail}".strip()
            )
        violations.extend(deadline_violations)
    state = _ReportEmitState(lines=lines, violations=violations)
    _append_report_tail_sections(
        state=state,
        report=report,
        root=root,
        file_paths=file_paths,
        groups_by_path=groups_by_path,
        execution_pattern_suggestions=list(execution_pattern_suggestions),
        parse_witness_contract_violations_fn=parse_witness_contract_violations_fn,
        projected=_projected,
        start_section=_start_section,
    )
    return "\n".join(state.lines), state.violations
