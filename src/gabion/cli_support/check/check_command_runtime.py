from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import typer

from gabion.commands import check_contract
from gabion.cli_support.shared.runtime_deps import CliRuntimeDeps
from gabion.json_types import JSONObject

CheckArtifactFlags = check_contract.CheckArtifactFlags
DataflowFilterBundle = check_contract.DataflowFilterBundle
CheckAuxOperation = check_contract.CheckAuxOperation

ContextCliDepsFn = Callable[[typer.Context], CliRuntimeDeps]
RunWithTimeoutRetriesFn = Callable[..., JSONObject]
PhaseProgressFromNotificationFn = Callable[[Mapping[str, object]], dict[str, object] | None]
PhaseProgressSignatureFn = Callable[[Mapping[str, object]], tuple[object, ...]]
PhaseTimelineFromPhaseProgressFn = Callable[[Mapping[str, object]], dict[str, str]]
EmitPhaseTimelineProgressFn = Callable[..., None]
DeadlineScopeFactory = Callable[[], AbstractContextManager[object]]
EmitLintOutputsFn = Callable[..., None]
EmitAnalysisResumeSummaryFn = Callable[[JSONObject], None]
EmitNonzeroExitCausesFn = Callable[[JSONObject], None]
EmitStatusWatchOutcomeFn = Callable[..., None]
CheckPolicyFlagsCtor = Callable[..., object]
RunCheckCommandFn = Callable[..., None]
DefaultCheckArtifactFlagsFn = Callable[[], CheckArtifactFlags]
DefaultCheckDeltaOptionsFn = Callable[[], check_contract.CheckDeltaOptions]
DataflowFilterBundleCtor = Callable[..., DataflowFilterBundle]
ParamIsCommandLineFn = Callable[[typer.Context, str], bool]
DeadlineLoopIterFn = Callable[[list[str]], list[str] | tuple[str, ...] | object]
RawProfileUnsupportedFlagsFn = Callable[[typer.Context], list[str]]
CheckRawProfileArgsFn = Callable[..., list[str]]
RunDataflowRawArgvFn = Callable[[list[str]], None]


@dataclass(frozen=True)
class CheckRunForwardingPayload:
    ctx: typer.Context
    paths: list[Path] | None
    report: Path | None
    root: Path
    config: Path | None
    decision_snapshot: Path | None
    allow_external: bool | None
    strictness: object
    analysis_budget_checks: int | None
    aspf_state_json: Path | None
    aspf_import_state: list[Path] | None
    aspf_delta_jsonl: Path | None
    aux_operation: CheckAuxOperation

    def validate(self) -> None:
        if not self.aux_operation.domain:
            raise typer.BadParameter("aux operation domain is required")
        if not self.aux_operation.action:
            raise typer.BadParameter("aux operation action is required")


@dataclass(frozen=True)
class CheckAuxRuntimePayload:
    forwarding: CheckRunForwardingPayload

    def validate(self) -> None:
        self.forwarding.validate()

    def run(
        self,
        *,
        run_check_command_fn: RunCheckCommandFn,
        default_check_artifact_flags_fn: DefaultCheckArtifactFlagsFn,
        default_check_delta_options_fn: DefaultCheckDeltaOptionsFn,
        dataflow_filter_bundle_ctor: DataflowFilterBundleCtor,
        gate_none: object,
        lint_mode_none: object,
    ) -> None:
        self.validate()
        strictness_value = getattr(self.forwarding.strictness, "value", self.forwarding.strictness)
        run_check_command_fn(
            ctx=self.forwarding.ctx,
            paths=self.forwarding.paths,
            report=self.forwarding.report,
            root=self.forwarding.root,
            config=self.forwarding.config,
            baseline=None,
            baseline_write=False,
            decision_snapshot=self.forwarding.decision_snapshot,
            artifact_flags=default_check_artifact_flags_fn(),
            delta_options=default_check_delta_options_fn(),
            exclude=None,
            filter_bundle=dataflow_filter_bundle_ctor(
                ignore_params_csv=None,
                transparent_decorators_csv=None,
            ),
            allow_external=self.forwarding.allow_external,
            strictness=str(strictness_value),
            analysis_budget_checks=self.forwarding.analysis_budget_checks,
            gate=gate_none,
            lint_mode=lint_mode_none,
            lint_jsonl_out=None,
            lint_sarif_out=None,
            aspf_trace_json=None,
            aspf_import_trace=None,
            aspf_equivalence_against=None,
            aspf_opportunities_json=None,
            aspf_state_json=self.forwarding.aspf_state_json,
            aspf_import_state=self.forwarding.aspf_import_state,
            aspf_delta_jsonl=self.forwarding.aspf_delta_jsonl,
            aspf_semantic_surface=None,
            aux_operation=self.forwarding.aux_operation,
        )


def run_check_command(
    *,
    ctx: typer.Context,
    paths: list[Path] | None,
    report: Path | None,
    root: Path,
    config: Path | None,
    baseline: Path | None,
    baseline_write: bool,
    decision_snapshot: Path | None,
    artifact_flags: CheckArtifactFlags,
    delta_options: check_contract.CheckDeltaOptions,
    exclude: list[str] | None,
    filter_bundle: DataflowFilterBundle | None,
    allow_external: bool | None,
    strictness: str | None,
    analysis_budget_checks: int | None,
    gate: object,
    lint_mode: object,
    lint_jsonl_out: Path | None,
    lint_sarif_out: Path | None,
    aspf_trace_json: Path | None,
    aspf_import_trace: list[Path] | None,
    aspf_equivalence_against: list[Path] | None,
    aspf_opportunities_json: Path | None,
    aspf_state_json: Path | None,
    aspf_import_state: list[Path] | None,
    aspf_delta_jsonl: Path | None = None,
    aspf_semantic_surface: list[str] | None = None,
    aux_operation: CheckAuxOperation | None = None,
    status_watch_options: object | None = None,
    check_gate_policy_fn: Callable[[object], tuple[bool, bool]],
    check_lint_mode_fn: Callable[..., tuple[bool, bool]],
    context_cli_deps_fn: ContextCliDepsFn,
    phase_progress_from_progress_notification_fn: PhaseProgressFromNotificationFn,
    phase_progress_signature_fn: PhaseProgressSignatureFn,
    phase_timeline_from_phase_progress_fn: PhaseTimelineFromPhaseProgressFn,
    emit_phase_timeline_progress_fn: EmitPhaseTimelineProgressFn,
    run_with_timeout_retries_fn: RunWithTimeoutRetriesFn,
    cli_deadline_scope_factory: DeadlineScopeFactory,
    emit_lint_outputs_fn: EmitLintOutputsFn,
    emit_analysis_resume_summary_fn: EmitAnalysisResumeSummaryFn,
    emit_nonzero_exit_causes_fn: EmitNonzeroExitCausesFn,
    emit_status_watch_outcome_fn: EmitStatusWatchOutcomeFn,
    check_policy_flags_ctor: CheckPolicyFlagsCtor,
    path_ctor: Callable[[Path], Path] = Path,
) -> None:
    fail_on_violations, fail_on_type_ambiguities = check_gate_policy_fn(gate)
    lint_enabled, lint_line = check_lint_mode_fn(
        lint_mode=lint_mode,
        lint_jsonl_out=lint_jsonl_out,
        lint_sarif_out=lint_sarif_out,
    )
    deps = context_cli_deps_fn(ctx)
    timeline_header_emitted = False
    last_phase_progress_signature: tuple[object, ...] | None = None
    last_phase_event_seq: int | None = None

    def _on_notification(notification: JSONObject) -> None:
        nonlocal timeline_header_emitted
        nonlocal last_phase_progress_signature
        nonlocal last_phase_event_seq
        phase_progress = phase_progress_from_progress_notification_fn(notification)
        match phase_progress:
            case {**phase_progress_payload}:
                event_seq = phase_progress_payload.get("event_seq")
                match event_seq:
                    case int() as event_sequence:
                        if last_phase_event_seq == event_sequence:
                            return
                        last_phase_event_seq = event_sequence
                    case _:
                        pass
                signature = phase_progress_signature_fn(phase_progress_payload)
                if signature == last_phase_progress_signature:
                    return
                last_phase_progress_signature = signature
                timeline_update = phase_timeline_from_phase_progress_fn(
                    phase_progress_payload
                )
                row = str(timeline_update.get("row") or "")
                header_value = timeline_update.get("header")
                match header_value:
                    case str() as header_text if not timeline_header_emitted and header_text:
                        header = header_text
                    case _:
                        header = None
                emit_phase_timeline_progress_fn(header=header, row=row)
                if header is not None:
                    timeline_header_emitted = True
            case _:
                return

    result = run_with_timeout_retries_fn(
        run_once=lambda: deps.run_check_fn(
            paths=paths,
            report=report,
            policy=check_policy_flags_ctor(
                fail_on_violations=fail_on_violations,
                fail_on_type_ambiguities=fail_on_type_ambiguities,
                lint=lint_enabled,
            ),
            root=root,
            config=config,
            baseline=baseline,
            baseline_write=baseline_write,
            decision_snapshot=decision_snapshot,
            artifact_flags=artifact_flags,
            delta_options=delta_options,
            exclude=exclude,
            filter_bundle=filter_bundle,
            allow_external=allow_external,
            strictness=strictness,
            analysis_tick_limit=analysis_budget_checks,
            aspf_trace_json=aspf_trace_json,
            aspf_import_trace=aspf_import_trace,
            aspf_equivalence_against=aspf_equivalence_against,
            aspf_opportunities_json=aspf_opportunities_json,
            aspf_state_json=aspf_state_json,
            aspf_import_state=aspf_import_state,
            aspf_delta_jsonl=aspf_delta_jsonl,
            aspf_semantic_surface=aspf_semantic_surface,
            aux_operation=aux_operation,
            notification_callback=_on_notification,
        ),
        root=path_ctor(root),
    )
    with cli_deadline_scope_factory():
        lint_lines = result.get("lint_lines", []) or []
        emit_lint_outputs_fn(
            lint_lines,
            lint=lint_line,
            lint_jsonl=lint_jsonl_out,
            lint_sarif=lint_sarif_out,
        )
    emit_analysis_resume_summary_fn(result)
    emit_nonzero_exit_causes_fn(result)
    local_exit_code = int(result.get("exit_code", 0))
    if local_exit_code != 0:
        raise typer.Exit(code=local_exit_code)
    if status_watch_options is None:
        raise typer.Exit(code=local_exit_code)
    try:
        watch_result = deps.run_ci_watch_fn(status_watch_options)
    except SystemExit as exc:
        match exc.code:
            case str() as message if message:
                typer.secho(message, err=True, fg=typer.colors.RED)
            case _:
                pass
        match exc.code:
            case int() as exit_code:
                raise typer.Exit(code=exit_code) from None
            case _:
                raise typer.Exit(code=1) from None
    emit_status_watch_outcome_fn(result=watch_result, options=status_watch_options)
    raise typer.Exit(code=watch_result.exit_code)


def run_check_aux_operation(
    *,
    ctx: typer.Context,
    domain: str,
    action: str,
    paths: list[Path] | None,
    root: Path,
    config: Path | None,
    strictness: object,
    allow_external: bool | None,
    baseline: Path | None,
    state_in: Path | None,
    out_json: Path | None,
    out_md: Path | None,
    report: Path | None,
    decision_snapshot: Path | None,
    analysis_budget_checks: int | None,
    aspf_state_json: Path | None,
    aspf_import_state: list[Path] | None,
    aspf_delta_jsonl: Path | None = None,
    run_check_command_fn: RunCheckCommandFn,
    default_check_artifact_flags_fn: DefaultCheckArtifactFlagsFn,
    default_check_delta_options_fn: DefaultCheckDeltaOptionsFn,
    dataflow_filter_bundle_ctor: DataflowFilterBundleCtor,
    gate_none: object,
    lint_mode_none: object,
) -> None:
    payload = CheckAuxRuntimePayload(
        forwarding=CheckRunForwardingPayload(
            ctx=ctx,
            paths=paths,
            report=report,
            root=root,
            config=config,
            decision_snapshot=decision_snapshot,
            allow_external=allow_external,
            strictness=strictness,
            analysis_budget_checks=analysis_budget_checks,
            aspf_state_json=aspf_state_json,
            aspf_import_state=aspf_import_state,
            aspf_delta_jsonl=aspf_delta_jsonl,
            aux_operation=CheckAuxOperation(
                domain=domain,
                action=action,
                baseline_path=baseline,
                state_in_path=state_in,
                out_json=out_json,
                out_md=out_md,
            ),
        )
    )
    payload.run(
        run_check_command_fn=run_check_command_fn,
        default_check_artifact_flags_fn=default_check_artifact_flags_fn,
        default_check_delta_options_fn=default_check_delta_options_fn,
        dataflow_filter_bundle_ctor=dataflow_filter_bundle_ctor,
        gate_none=gate_none,
        lint_mode_none=lint_mode_none,
    )


def check_raw_profile_args(
    *,
    ctx: typer.Context,
    paths: list[Path] | None,
    report: Path | None,
    fail_on_violations: bool,
    root: Path,
    config: Path | None,
    decision_snapshot: Path | None,
    baseline: Path | None,
    baseline_write: bool,
    exclude: list[str] | None,
    filter_bundle: DataflowFilterBundle | None,
    allow_external: bool | None,
    strictness: str | None,
    fail_on_type_ambiguities: bool,
    lint: bool,
    lint_jsonl: Path | None,
    lint_sarif: Path | None,
    param_is_command_line_fn: ParamIsCommandLineFn,
    deadline_loop_iter_fn: Callable[[list[str]], object],
    dataflow_filter_bundle_ctor: DataflowFilterBundleCtor,
) -> list[str]:
    resolved_filter_bundle = filter_bundle or dataflow_filter_bundle_ctor(None, None)
    argv = [str(path) for path in (paths or [])]
    if param_is_command_line_fn(ctx, "root"):
        argv.extend(["--root", str(root)])
    if param_is_command_line_fn(ctx, "config") and config is not None:
        argv.extend(["--config", str(config)])
    if param_is_command_line_fn(ctx, "report") and report is not None:
        argv.extend(["--report", str(report)])
    if param_is_command_line_fn(ctx, "decision_snapshot") and decision_snapshot is not None:
        argv.extend(["--emit-decision-snapshot", str(decision_snapshot)])
    if param_is_command_line_fn(ctx, "baseline") and baseline is not None:
        argv.extend(["--baseline", str(baseline)])
    if param_is_command_line_fn(ctx, "baseline_write") and baseline_write:
        argv.append("--baseline-write")
    if param_is_command_line_fn(ctx, "exclude"):
        iter_exclude = deadline_loop_iter_fn(exclude or [])
        for entry in iter_exclude:
            argv.extend(["--exclude", entry])
    if (
        param_is_command_line_fn(ctx, "ignore_params_csv")
        and resolved_filter_bundle.ignore_params_csv is not None
    ):
        argv.extend(["--ignore-params", resolved_filter_bundle.ignore_params_csv])
    if (
        param_is_command_line_fn(ctx, "transparent_decorators_csv")
        and resolved_filter_bundle.transparent_decorators_csv is not None
    ):
        argv.extend(["--transparent-decorators", resolved_filter_bundle.transparent_decorators_csv])
    if param_is_command_line_fn(ctx, "allow_external") and allow_external is not None:
        argv.append("--allow-external" if allow_external else "--no-allow-external")
    if param_is_command_line_fn(ctx, "strictness") and strictness is not None:
        argv.extend(["--strictness", strictness])
    if param_is_command_line_fn(ctx, "fail_on_violations") and fail_on_violations:
        argv.append("--fail-on-violations")
    if param_is_command_line_fn(ctx, "fail_on_type_ambiguities") and fail_on_type_ambiguities:
        argv.append("--fail-on-type-ambiguities")
    if param_is_command_line_fn(ctx, "lint") and lint:
        argv.append("--lint")
    if param_is_command_line_fn(ctx, "lint_jsonl") and lint_jsonl is not None:
        argv.extend(["--lint-jsonl", str(lint_jsonl)])
    if param_is_command_line_fn(ctx, "lint_sarif") and lint_sarif is not None:
        argv.extend(["--lint-sarif", str(lint_sarif)])
    return argv


def run_check_raw_profile(
    *,
    ctx: typer.Context,
    paths: list[Path] | None,
    report: Path | None,
    fail_on_violations: bool,
    root: Path,
    config: Path | None,
    decision_snapshot: Path | None,
    baseline: Path | None,
    baseline_write: bool,
    exclude: list[str] | None,
    filter_bundle: DataflowFilterBundle | None,
    allow_external: bool | None,
    strictness: str | None,
    fail_on_type_ambiguities: bool,
    lint: bool,
    lint_jsonl: Path | None,
    lint_sarif: Path | None,
    run_dataflow_raw_argv_fn: RunDataflowRawArgvFn | None = None,
    raw_profile_unsupported_flags_fn: RawProfileUnsupportedFlagsFn,
    check_raw_profile_args_fn: CheckRawProfileArgsFn,
    default_run_dataflow_raw_argv_fn: RunDataflowRawArgvFn,
) -> None:
    unsupported = raw_profile_unsupported_flags_fn(ctx)
    if unsupported:
        rendered = ", ".join(unsupported)
        raise typer.BadParameter(
            f"--profile raw does not support check-only options: {rendered}"
        )
    raw_args = check_raw_profile_args_fn(
        ctx=ctx,
        paths=paths,
        report=report,
        fail_on_violations=fail_on_violations,
        root=root,
        config=config,
        decision_snapshot=decision_snapshot,
        baseline=baseline,
        baseline_write=baseline_write,
        exclude=exclude,
        filter_bundle=filter_bundle,
        allow_external=allow_external,
        strictness=strictness,
        fail_on_type_ambiguities=fail_on_type_ambiguities,
        lint=lint,
        lint_jsonl=lint_jsonl,
        lint_sarif=lint_sarif,
    )
    resolved_run = run_dataflow_raw_argv_fn or default_run_dataflow_raw_argv_fn
    resolved_run(raw_args + list(ctx.args))
