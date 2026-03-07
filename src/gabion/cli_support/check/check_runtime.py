# gabion:decision_protocol_module
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from gabion.commands import check_contract
from gabion.json_types import JSONObject

CheckArtifactFlags = check_contract.CheckArtifactFlags
CheckPolicyFlags = check_contract.CheckPolicyFlags
DataflowFilterBundle = check_contract.DataflowFilterBundle
CheckDeltaOptions = check_contract.CheckDeltaOptions
CheckAuxOperation = check_contract.CheckAuxOperation

Runner = Callable[..., JSONObject]
ResolveCheckReportPathFn = Callable[[Path | None], Path]
BuildCheckPayloadFn = Callable[..., JSONObject]
BuildCheckExecutionPlanRequestFn = Callable[..., object]
DispatchCommandFn = Callable[..., JSONObject]


def run_check(
    *,
    paths: Optional[list[Path]],
    report: Optional[Path],
    policy: CheckPolicyFlags,
    root: Path,
    config: Optional[Path],
    baseline: Path | None,
    baseline_write: bool,
    decision_snapshot: Path | None,
    artifact_flags: CheckArtifactFlags,
    delta_options: CheckDeltaOptions,
    exclude: Optional[list[str]],
    filter_bundle: DataflowFilterBundle | None,
    allow_external: Optional[bool],
    strictness: Optional[str],
    analysis_tick_limit: int | None = None,
    aux_operation: CheckAuxOperation | None = None,
    aspf_trace_json: Path | None = None,
    aspf_import_trace: Optional[list[Path]] = None,
    aspf_equivalence_against: Optional[list[Path]] = None,
    aspf_opportunities_json: Path | None = None,
    aspf_state_json: Path | None = None,
    aspf_import_state: Optional[list[Path]] = None,
    aspf_delta_jsonl: Path | None = None,
    aspf_semantic_surface: Optional[list[str]] = None,
    runner: Runner,
    notification_callback: Callable[[JSONObject], None] | None = None,
    resolve_check_report_path_fn: ResolveCheckReportPathFn,
    build_check_payload_fn: BuildCheckPayloadFn,
    build_check_execution_plan_request_fn: BuildCheckExecutionPlanRequestFn,
    dispatch_command_fn: DispatchCommandFn,
    dataflow_command: str,
) -> JSONObject:
    if filter_bundle is None:
        filter_bundle = DataflowFilterBundle(None, None)
    # dataflow-bundle: filter_bundle
    resolved_report = resolve_check_report_path_fn(report, root=root)
    resolved_report.parent.mkdir(parents=True, exist_ok=True)
    payload = build_check_payload_fn(
        paths=paths,
        report=resolved_report,
        fail_on_violations=policy.fail_on_violations,
        root=root,
        config=config,
        baseline=baseline,
        baseline_write=baseline_write if baseline is not None else False,
        decision_snapshot=decision_snapshot,
        artifact_flags=artifact_flags,
        delta_options=delta_options,
        exclude=exclude,
        filter_bundle=filter_bundle,
        allow_external=allow_external,
        strictness=strictness,
        fail_on_type_ambiguities=policy.fail_on_type_ambiguities,
        lint=policy.lint,
        analysis_tick_limit=analysis_tick_limit,
        aux_operation=aux_operation,
        aspf_trace_json=aspf_trace_json,
        aspf_import_trace=aspf_import_trace,
        aspf_equivalence_against=aspf_equivalence_against,
        aspf_opportunities_json=aspf_opportunities_json,
        aspf_state_json=aspf_state_json,
        aspf_import_state=aspf_import_state,
        aspf_delta_jsonl=aspf_delta_jsonl,
        aspf_semantic_surface=aspf_semantic_surface,
    )
    execution_plan_request = build_check_execution_plan_request_fn(
        payload=payload,
        report=resolved_report,
        decision_snapshot=decision_snapshot,
        baseline=baseline,
        baseline_write=baseline_write,
        policy=policy,
        profile="strict",
        artifact_flags=artifact_flags,
        emit_test_obsolescence_state=delta_options.emit_test_obsolescence_state,
        emit_test_obsolescence_delta=delta_options.emit_test_obsolescence_delta,
        emit_test_annotation_drift_delta=delta_options.emit_test_annotation_drift_delta,
        emit_ambiguity_delta=delta_options.emit_ambiguity_delta,
        emit_ambiguity_state=delta_options.emit_ambiguity_state,
        aspf_trace_json=aspf_trace_json,
        aspf_opportunities_json=aspf_opportunities_json,
        aspf_state_json=aspf_state_json,
        aspf_delta_jsonl=aspf_delta_jsonl,
        aspf_equivalence_enabled=bool(
            aspf_trace_json
            or aspf_import_trace
            or aspf_equivalence_against
            or aspf_opportunities_json
            or aspf_state_json
            or aspf_import_state
            or aspf_delta_jsonl
            or aspf_semantic_surface
        ),
    )
    return dispatch_command_fn(
        command=dataflow_command,
        payload=payload,
        root=root,
        runner=runner,
        execution_plan_request=execution_plan_request,
        notification_callback=notification_callback,
    )
