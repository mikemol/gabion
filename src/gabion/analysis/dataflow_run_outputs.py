# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Mapping

from gabion.analysis.json_types import JSONObject, JSONValue

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


@dataclass(frozen=True)
class DataflowRunOutputContext:
    args: argparse.Namespace
    analysis: AnalysisResult
    paths: list[Path]
    config: AuditConfig
    synth_defaults: Mapping[str, object]
    baseline_path: Path | None
    baseline_write: bool
    decision_snapshot_path: str | None
    fingerprint_deadness_json: str | None
    fingerprint_coherence_json: str | None
    fingerprint_rewrite_plans_json: str | None
    fingerprint_exception_obligations_json: str | None
    fingerprint_handledness_json: str | None


@dataclass(frozen=True)
class DataflowRunOutputOutcome:
    exit_code: int
    terminal_phase: str


@dataclass(frozen=True)
class RunOutputOp:
    op_id: str


@dataclass(frozen=True)
class _ProjectionOutputEffect:
    effect_id: str
    output_path: str
    payload: JSONValue | object


@dataclass(frozen=True)
class _RunOutputState:
    synthesis_plan: JSONObject | None = None
    refactor_plan: JSONObject | None = None


def plan_run_output_ops(_: DataflowRunOutputContext) -> tuple[RunOutputOp, ...]:
    check_deadline()
    return (
        RunOutputOp(op_id="projection_outputs"),
        RunOutputOp(op_id="synthesis_outputs"),
        RunOutputOp(op_id="refactor_outputs"),
        RunOutputOp(op_id="dot_output"),
        RunOutputOp(op_id="type_audit_output"),
        RunOutputOp(op_id="report_output"),
        RunOutputOp(op_id="console_and_violations"),
    )


def apply_run_output_ops(
    *,
    context: DataflowRunOutputContext,
    ops: tuple[RunOutputOp, ...],
    emit_report_fn: Callable[..., tuple[str, list[str]]],
    compute_violations_fn: Callable[..., list[str]],
) -> DataflowRunOutputOutcome:
    check_deadline()
    state = _RunOutputState()
    for op in ops:
        check_deadline()
        if op.op_id == "projection_outputs":
            if emit_projection_outputs(context):
                return DataflowRunOutputOutcome(exit_code=0, terminal_phase=op.op_id)
            continue
        if op.op_id == "synthesis_outputs":
            state = replace(
                state,
                synthesis_plan=emit_optional_synthesis_outputs(context),
            )
            continue
        if op.op_id == "refactor_outputs":
            state = replace(
                state,
                refactor_plan=emit_optional_refactor_outputs(context),
            )
            continue
        if op.op_id == "dot_output":
            if emit_optional_dot_output(context):
                return DataflowRunOutputOutcome(exit_code=0, terminal_phase=op.op_id)
            continue
        if op.op_id == "type_audit_output":
            if emit_optional_type_audit_output(context):
                return DataflowRunOutputOutcome(exit_code=0, terminal_phase=op.op_id)
            continue
        if op.op_id == "report_output":
            report_exit = emit_report_output(
                context=context,
                synthesis_plan=state.synthesis_plan,
                refactor_plan=state.refactor_plan,
                emit_report_fn=emit_report_fn,
            )
            if report_exit is not None:
                return DataflowRunOutputOutcome(
                    exit_code=report_exit,
                    terminal_phase=op.op_id,
                )
            continue
        if op.op_id == "console_and_violations":
            return DataflowRunOutputOutcome(
                exit_code=emit_console_output_and_violation_gate(
                    context=context,
                    compute_violations_fn=compute_violations_fn,
                ),
                terminal_phase=op.op_id,
            )
    never(  # pragma: no cover - defensive invariant for malformed op plans.
        "run-output operation pipeline did not terminate",
        operation_count=len(ops),
    )


def finalize_run_outputs(
    *,
    context: DataflowRunOutputContext,
    emit_report_fn: Callable[..., tuple[str, list[str]]],
    compute_violations_fn: Callable[..., list[str]],
) -> DataflowRunOutputOutcome:
    _bind_audit_symbols()
    ops = plan_run_output_ops(context)
    return apply_run_output_ops(
        context=context,
        ops=ops,
        emit_report_fn=emit_report_fn,
        compute_violations_fn=compute_violations_fn,
    )


def emit_projection_outputs(context: DataflowRunOutputContext) -> bool:
    args = context.args
    _emit_sidecar_outputs(
        args=args,
        analysis=context.analysis,
        fingerprint_deadness_json=context.fingerprint_deadness_json,
        fingerprint_coherence_json=context.fingerprint_coherence_json,
        fingerprint_rewrite_plans_json=context.fingerprint_rewrite_plans_json,
        fingerprint_exception_obligations_json=(
            context.fingerprint_exception_obligations_json
        ),
        fingerprint_handledness_json=context.fingerprint_handledness_json,
    )
    effects = plan_projection_output_effects(context)
    for effect in effects:
        check_deadline()
        _write_json_or_stdout(effect.output_path, effect.payload)
    if not effects:
        return False
    return (
        args.report is None
        and args.dot is None
        and not _has_followup_actions(args)
    )


def plan_projection_output_effects(
    context: DataflowRunOutputContext,
) -> list[_ProjectionOutputEffect]:
    args = context.args
    analysis = context.analysis
    effects: list[_ProjectionOutputEffect] = []
    if args.emit_structure_tree:
        effects.append(
            _ProjectionOutputEffect(
                effect_id="structure_tree",
                output_path=args.emit_structure_tree,
                payload=render_structure_snapshot(
                    analysis.groups_by_path,
                    project_root=context.config.project_root,
                    forest=analysis.forest,
                    forest_spec=analysis.forest_spec,
                    invariant_propositions=analysis.invariant_propositions,
                ),
            )
        )
    if args.emit_structure_metrics:
        effects.append(
            _ProjectionOutputEffect(
                effect_id="structure_metrics",
                output_path=args.emit_structure_metrics,
                payload=compute_structure_metrics(
                    analysis.groups_by_path,
                    forest=analysis.forest,
                ),
            )
        )
    if context.decision_snapshot_path:
        effects.append(
            _ProjectionOutputEffect(
                effect_id="decision_snapshot",
                output_path=context.decision_snapshot_path,
                payload=render_decision_snapshot(
                    surfaces=DecisionSnapshotSurfaces(
                        decision_surfaces=analysis.decision_surfaces,
                        value_decision_surfaces=analysis.value_decision_surfaces,
                    ),
                    project_root=context.config.project_root,
                    forest=analysis.forest,
                    forest_spec=analysis.forest_spec,
                    groups_by_path=analysis.groups_by_path,
                ),
            )
        )
    return effects


def merge_overlap_threshold(
    args: argparse.Namespace,
    synth_defaults: Mapping[str, object],
) -> float | None:
    if args.synthesis_merge_overlap is not None:
        threshold: float | None = args.synthesis_merge_overlap
    else:
        threshold = None
        value = synth_defaults.get("merge_overlap_threshold")
        if isinstance(value, (int, float)):
            threshold = float(value)
    if threshold is None:
        return None
    return max(0.0, min(1.0, threshold))


def emit_optional_synthesis_outputs(
    context: DataflowRunOutputContext,
) -> JSONObject | None:
    args = context.args
    analysis = context.analysis
    if not (args.synthesis_plan or args.synthesis_report or args.synthesis_protocols):
        return None
    synthesis_plan = build_synthesis_plan(
        analysis.groups_by_path,
        project_root=context.config.project_root,
        max_tier=args.synthesis_max_tier,
        min_bundle_size=args.synthesis_min_bundle_size,
        allow_singletons=args.synthesis_allow_singletons,
        merge_overlap_threshold=merge_overlap_threshold(args, context.synth_defaults),
        config=context.config,
        invariant_propositions=analysis.invariant_propositions,
        property_hook_min_confidence=args.synthesis_property_hook_min_confidence,
        emit_hypothesis_templates=bool(args.synthesis_property_hook_hypothesis),
    )
    if args.synthesis_plan:
        _write_json_or_stdout(args.synthesis_plan, synthesis_plan)
    if args.synthesis_protocols:
        stubs = render_protocol_stubs(synthesis_plan, kind=args.synthesis_protocols_kind)
        _write_text_or_stdout(args.synthesis_protocols, stubs)
    return synthesis_plan


def emit_optional_refactor_outputs(
    context: DataflowRunOutputContext,
) -> JSONObject | None:
    args = context.args
    if not (args.refactor_plan or args.refactor_plan_json):
        return None
    refactor_plan = build_refactor_plan(
        context.analysis.groups_by_path,
        context.paths,
        config=context.config,
    )
    if args.refactor_plan_json:
        _write_json_or_stdout(args.refactor_plan_json, refactor_plan)
    return refactor_plan


def emit_optional_dot_output(context: DataflowRunOutputContext) -> bool:
    args = context.args
    if args.dot is None:
        return False
    dot = _emit_dot(context.analysis.forest)
    _write_text_or_stdout(args.dot, dot)
    return args.report is None and not _has_followup_actions(
        args,
        include_structure_tree=bool(args.emit_structure_tree),
        include_structure_metrics=bool(args.emit_structure_metrics),
        include_decision_snapshot=bool(context.decision_snapshot_path),
    )


def emit_optional_type_audit_output(context: DataflowRunOutputContext) -> bool:
    args = context.args
    analysis = context.analysis
    if not args.type_audit:
        return False
    if analysis.type_suggestions:
        print("Type tightening candidates:")
        for line in analysis.type_suggestions[: args.type_audit_max]:
            check_deadline()
            print(f"- {line}")
    if analysis.type_ambiguities:
        print("Type ambiguities (conflicting downstream expectations):")
        for line in analysis.type_ambiguities[: args.type_audit_max]:
            check_deadline()
            print(f"- {line}")
    return args.report is None and not _has_followup_actions(args, include_type_audit=False)


def emit_report_output(
    *,
    context: DataflowRunOutputContext,
    synthesis_plan: JSONObject | None,
    refactor_plan: JSONObject | None,
    emit_report_fn: Callable[..., tuple[str, list[str]]],
) -> int | None:
    args = context.args
    if args.report is None:
        return None
    report_carrier = ReportCarrier.from_analysis_result(
        context.analysis,
        include_type_audit=args.type_audit_report,
    )
    report, violations = emit_report_fn(
        context.analysis.groups_by_path,
        args.max_components,
        report=report_carrier,
    )
    suppressed: list[str] = []
    new_violations = violations
    if context.baseline_path is not None:
        baseline_entries = _load_baseline(context.baseline_path)
        if context.baseline_write:
            _write_baseline(context.baseline_path, violations)
            baseline_entries = set(violations)
            new_violations = []
        else:
            new_violations, suppressed = _apply_baseline(violations, baseline_entries)
        report = (
            report
            + "\n\nBaseline/Ratchet:\n```\n"
            + f"Baseline: {context.baseline_path}\n"
            + f"Baseline entries: {len(baseline_entries)}\n"
            + f"Suppressed: {len(suppressed)}\n"
            + f"New violations: {len(new_violations)}\n"
            + "```\n"
        )
    if synthesis_plan and (args.synthesis_report or args.synthesis_plan or args.synthesis_protocols):
        report = report + render_synthesis_section(synthesis_plan)
    if refactor_plan and (args.refactor_plan or args.refactor_plan_json):
        report = report + render_refactor_plan(refactor_plan)
    _write_text_or_stdout(args.report, report)
    if args.fail_on_violations and violations:
        if context.baseline_write:
            return 0
        if new_violations:
            return 1
    return 0


def emit_console_output_and_violation_gate(
    *,
    context: DataflowRunOutputContext,
    compute_violations_fn: Callable[..., list[str]],
) -> int:
    args = context.args
    analysis = context.analysis
    for path, groups in analysis.groups_by_path.items():
        check_deadline()
        print(f"# {path}")
        for fn, bundles in groups.items():
            check_deadline()
            if not bundles:
                continue
            print(f"{fn}:")
            for bundle in bundles:
                check_deadline()
                print(
                    f"  bundle: {sort_once(bundle, source = 'src/gabion/analysis/dataflow_run_outputs.py:383')}"
                )
        print()
    if args.fail_on_type_ambiguities and analysis.type_ambiguities:
        return 1
    if args.fail_on_violations:
        violation_carrier = ReportCarrier(
            forest=analysis.forest,
            type_suggestions=analysis.type_suggestions if args.type_audit_report else [],
            type_ambiguities=analysis.type_ambiguities if args.type_audit_report else [],
            decision_warnings=analysis.decision_warnings,
            fingerprint_warnings=analysis.fingerprint_warnings,
            parse_failure_witnesses=analysis.parse_failure_witnesses,
        )
        violations = compute_violations_fn(
            analysis.groups_by_path,
            args.max_components,
            report=violation_carrier,
        )
        if context.baseline_path is not None:
            baseline_entries = _load_baseline(context.baseline_path)
            if context.baseline_write:
                _write_baseline(context.baseline_path, violations)
                return 0
            new_violations, _ = _apply_baseline(violations, baseline_entries)
            if new_violations:
                return 1
        elif violations:
            return 1
    return 0
