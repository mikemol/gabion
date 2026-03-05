# gabion:boundary_normalization_module
# gabion:decision_protocol_module

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import typer

BuildStatusWatchOptionsFn = Callable[..., object | None]
RunCheckCommandFn = Callable[..., None]
DataflowFilterBundleCtor = Callable[..., object]
CheckFlagsFactory = Callable[[], object]
CheckHelpOrExitFn = Callable[[typer.Context], None]
DeltaOptionsFactory = Callable[[], object]
RunCheckAuxOperationFn = Callable[..., None]


@dataclass(frozen=True)
class CheckAuxCommandRegistration:
    domain: str
    action: str
    option_profile: "CheckAuxOptionProfile"

    @property
    def identity(self) -> str:
        return f"{self.domain}:{self.action}"


@dataclass(frozen=True)
class CheckAuxOptionProfile:
    name: str
    baseline_required: bool
    out_md_allowed: bool


@dataclass(frozen=True)
class CheckAuxCommandSpec:
    registration: CheckAuxCommandRegistration

    def validate(self) -> None:
        profile = self.registration.option_profile
        if profile.baseline_required and self.registration.action not in {"delta", "baseline-write"}:
            raise typer.BadParameter("baseline_required profile only supports delta/baseline-write actions")
        if not profile.out_md_allowed and self.registration.action == "report":
            raise typer.BadParameter("report actions require out_md_allowed profile")

    @property
    def option_profile(self) -> CheckAuxOptionProfile:
        return self.registration.option_profile

    @property
    def domain(self) -> str:
        return self.registration.domain

    @property
    def action(self) -> str:
        return self.registration.action

    @property
    def identity(self) -> str:
        return self.registration.identity


@dataclass(frozen=True)
class CheckAuxOperationPayload:
    ctx: typer.Context
    spec: CheckAuxCommandSpec
    paths: list[Path]
    root: Path
    config: Path | None
    strictness: object
    allow_external: bool | None
    baseline: Path | None
    state_in: Path | None
    out_json: Path | None
    out_md: Path | None
    report: Path | None
    decision_snapshot: Path | None
    analysis_budget_checks: int | None
    aspf_state_json: Path | None
    aspf_import_state: list[Path] | None

    def validate(self) -> None:
        self.spec.validate()
        profile = self.spec.option_profile
        if profile.baseline_required and self.baseline is None:
            raise typer.BadParameter("--baseline is required for this command profile")
        if not profile.out_md_allowed and self.out_md is not None:
            raise typer.BadParameter("--out-md is not allowed for this command profile")

    def to_runtime_kwargs(self) -> dict[str, object]:
        self.validate()
        return {
            "ctx": self.ctx,
            "domain": self.spec.domain,
            "action": self.spec.action,
            "paths": self.paths,
            "root": self.root,
            "config": self.config,
            "strictness": self.strictness,
            "allow_external": self.allow_external,
            "baseline": self.baseline,
            "state_in": self.state_in,
            "out_json": self.out_json,
            "out_md": self.out_md,
            "report": self.report,
            "decision_snapshot": self.decision_snapshot,
            "analysis_budget_checks": self.analysis_budget_checks,
            "aspf_state_json": self.aspf_state_json,
            "aspf_import_state": self.aspf_import_state,
        }


CHECK_AUX_OPTION_PROFILE_REPORT = CheckAuxOptionProfile(
    name="report",
    baseline_required=False,
    out_md_allowed=True,
)
CHECK_AUX_OPTION_PROFILE_STATE = CheckAuxOptionProfile(
    name="state",
    baseline_required=False,
    out_md_allowed=False,
)
CHECK_AUX_OPTION_PROFILE_DELTA = CheckAuxOptionProfile(
    name="delta",
    baseline_required=True,
    out_md_allowed=True,
)


CHECK_AUX_COMMAND_REGISTRATIONS: tuple[CheckAuxCommandRegistration, ...] = (
    CheckAuxCommandRegistration("obsolescence", "report", CHECK_AUX_OPTION_PROFILE_REPORT),
    CheckAuxCommandRegistration("obsolescence", "state", CHECK_AUX_OPTION_PROFILE_STATE),
    CheckAuxCommandRegistration("obsolescence", "delta", CHECK_AUX_OPTION_PROFILE_DELTA),
    CheckAuxCommandRegistration("obsolescence", "baseline-write", CHECK_AUX_OPTION_PROFILE_DELTA),
    CheckAuxCommandRegistration("annotation-drift", "report", CHECK_AUX_OPTION_PROFILE_REPORT),
    CheckAuxCommandRegistration("annotation-drift", "state", CHECK_AUX_OPTION_PROFILE_STATE),
    CheckAuxCommandRegistration("annotation-drift", "delta", CHECK_AUX_OPTION_PROFILE_DELTA),
    CheckAuxCommandRegistration("annotation-drift", "baseline-write", CHECK_AUX_OPTION_PROFILE_DELTA),
    CheckAuxCommandRegistration("ambiguity", "state", CHECK_AUX_OPTION_PROFILE_STATE),
    CheckAuxCommandRegistration("ambiguity", "delta", CHECK_AUX_OPTION_PROFILE_DELTA),
    CheckAuxCommandRegistration("ambiguity", "baseline-write", CHECK_AUX_OPTION_PROFILE_DELTA),
    CheckAuxCommandRegistration("taint", "state", CHECK_AUX_OPTION_PROFILE_STATE),
    CheckAuxCommandRegistration("taint", "delta", CHECK_AUX_OPTION_PROFILE_DELTA),
    CheckAuxCommandRegistration("taint", "baseline-write", CHECK_AUX_OPTION_PROFILE_DELTA),
    CheckAuxCommandRegistration("taint", "lifecycle", CHECK_AUX_OPTION_PROFILE_STATE),
)


def register_check_aux_commands(
    *,
    command_domains: dict[str, typer.Typer],
    check_strictness_mode: type,
    run_check_aux_operation_fn: RunCheckAuxOperationFn,
    command_registrations: tuple[CheckAuxCommandRegistration, ...] = CHECK_AUX_COMMAND_REGISTRATIONS,
) -> dict[str, Callable[..., None]]:
    commands: dict[str, Callable[..., None]] = {}
    for registration in command_registrations:
        spec = CheckAuxCommandSpec(registration=registration)
        spec.validate()
        command_app = command_domains[registration.domain]
        command = _build_check_aux_command(
            command_app=command_app,
            command_name=registration.action,
            spec=spec,
            check_strictness_mode=check_strictness_mode,
            run_check_aux_operation_fn=run_check_aux_operation_fn,
        )
        commands[spec.identity] = command
    return commands


def _build_check_aux_operation_payload(
    *,
    ctx: typer.Context,
    spec: CheckAuxCommandSpec,
    paths: list[Path],
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
) -> CheckAuxOperationPayload:
    payload = CheckAuxOperationPayload(
        ctx=ctx,
        spec=spec,
        paths=paths,
        root=root,
        config=config,
        strictness=strictness,
        allow_external=allow_external,
        baseline=baseline,
        state_in=state_in,
        out_json=out_json,
        out_md=out_md,
        report=report,
        decision_snapshot=decision_snapshot,
        analysis_budget_checks=analysis_budget_checks,
        aspf_state_json=aspf_state_json,
        aspf_import_state=aspf_import_state,
    )
    payload.validate()
    return payload


def _build_check_aux_command(
    *,
    command_app: typer.Typer,
    command_name: str,
    spec: CheckAuxCommandSpec,
    check_strictness_mode: type,
    run_check_aux_operation_fn: RunCheckAuxOperationFn,
) -> Callable[..., None]:
    option_profile = spec.option_profile

    @command_app.command(command_name)
    def command(
        ctx: typer.Context,
        paths: list[Path] = typer.Argument(None),
        root: Path = typer.Option(Path("."), "--root"),
        config: Path | None = typer.Option(None, "--config"),
        strictness: check_strictness_mode = typer.Option(check_strictness_mode.high, "--strictness"),
        allow_external: bool | None = typer.Option(
            None, "--allow-external/--no-allow-external"
        ),
        baseline: Path | None = typer.Option(
            ..., "--baseline"
        )
        if option_profile.baseline_required
        else typer.Option(None, "--baseline"),
        state_in: Path | None = typer.Option(None, "--state-in"),
        out_json: Path | None = typer.Option(None, "--out-json"),
        report: Path | None = typer.Option(None, "--report"),
        decision_snapshot: Path | None = typer.Option(None, "--decision-snapshot"),
        analysis_budget_checks: int | None = typer.Option(
            None,
            "--analysis-budget-checks",
            min=1,
        ),
        out_md: Path | None = typer.Option(None, "--out-md", hidden=not option_profile.out_md_allowed),
        aspf_state_json: Path | None = typer.Option(None, "--aspf-state-json"),
        aspf_import_state: list[Path] | None = typer.Option(
            None,
            "--aspf-import-state",
        ),
    ) -> None:
        payload = _build_check_aux_operation_payload(
            ctx=ctx,
            spec=spec,
            paths=paths,
            root=root,
            config=config,
            strictness=strictness,
            allow_external=allow_external,
            baseline=baseline,
            state_in=state_in,
            out_json=out_json,
            out_md=out_md,
            report=report,
            decision_snapshot=decision_snapshot,
            analysis_budget_checks=analysis_budget_checks,
            aspf_state_json=aspf_state_json,
            aspf_import_state=aspf_import_state,
        )
        run_check_aux_operation_fn(
            **payload.to_runtime_kwargs()
        )

    return command


def register_check_run_command(
    *,
    check_app: typer.Typer,
    check_strictness_mode: type,
    check_baseline_mode: type,
    check_gate_mode: type,
    check_lint_mode: type,
    dataflow_filter_bundle_ctor: DataflowFilterBundleCtor,
    build_status_watch_options_fn: BuildStatusWatchOptionsFn,
    run_check_command_fn: RunCheckCommandFn,
    default_check_artifact_flags_fn: CheckFlagsFactory,
    default_check_delta_options_fn: CheckFlagsFactory,
) -> Callable[..., None]:
    @check_app.command("run")
    def check_run(
        ctx: typer.Context,
        paths: list[Path] = typer.Argument(None),
        root: Path = typer.Option(Path("."), "--root"),
        config: Path | None = typer.Option(None, "--config"),
        report: Path | None = typer.Option(None, "--report"),
        strictness: check_strictness_mode = typer.Option(
            check_strictness_mode.high,
            "--strictness",
        ),
        allow_external: bool | None = typer.Option(
            None,
            "--allow-external/--no-allow-external",
        ),
        baseline: Path | None = typer.Option(None, "--baseline"),
        baseline_mode: check_baseline_mode = typer.Option(
            check_baseline_mode.off,
            "--baseline-mode",
        ),
        gate: check_gate_mode = typer.Option(check_gate_mode.all, "--gate"),
        analysis_budget_checks: int | None = typer.Option(
            None,
            "--analysis-budget-checks",
            min=1,
        ),
        removed_analysis_tick_limit: int | None = typer.Option(
            None,
            "--analysis-tick-limit",
            hidden=True,
        ),
        decision_snapshot: Path | None = typer.Option(
            None,
            "--decision-snapshot",
        ),
        lint: check_lint_mode = typer.Option(check_lint_mode.none, "--lint"),
        lint_jsonl_out: Path | None = typer.Option(
            None,
            "--lint-jsonl-out",
        ),
        lint_sarif_out: Path | None = typer.Option(
            None,
            "--lint-sarif-out",
        ),
        aspf_trace_json: Path | None = typer.Option(
            None,
            "--aspf-trace-json",
        ),
        aspf_import_trace: list[Path] | None = typer.Option(
            None,
            "--aspf-import-trace",
        ),
        aspf_equivalence_against: list[Path] | None = typer.Option(
            None,
            "--aspf-equivalence-against",
        ),
        aspf_opportunities_json: Path | None = typer.Option(
            None,
            "--aspf-opportunities-json",
        ),
        aspf_state_json: Path | None = typer.Option(
            None,
            "--aspf-state-json",
        ),
        aspf_delta_jsonl: Path | None = typer.Option(
            None,
            "--aspf-delta-jsonl",
        ),
        aspf_import_state: list[Path] | None = typer.Option(
            None,
            "--aspf-import-state",
        ),
        aspf_semantic_surface: list[str] | None = typer.Option(
            None,
            "--aspf-semantic-surface",
        ),
        status_watch: bool = typer.Option(
            False,
            "--status-watch/--no-status-watch",
            help="Watch GitHub status checks after a successful local check run.",
        ),
        status_watch_run_id: str | None = typer.Option(
            None,
            "--status-watch-run-id",
            help="Specific run id to watch (skips lookup).",
        ),
        status_watch_branch: str | None = typer.Option(
            None,
            "--status-watch-branch",
            help="Branch to watch (default: stage).",
        ),
        status_watch_workflow: str | None = typer.Option(
            None,
            "--status-watch-workflow",
            help="Optional workflow name or file filter.",
        ),
        status_watch_status: str | None = typer.Option(
            None,
            "--status-watch-status",
            help="Optional status filter for fallback run lookup.",
        ),
        status_watch_prefer_active: bool | None = typer.Option(
            None,
            "--status-watch-prefer-active/--no-status-watch-prefer-active",
            help="Prefer queued/in-progress runs when selecting run id.",
        ),
        status_watch_download_artifacts_on_failure: bool | None = typer.Option(
            None,
            "--status-watch-download-artifacts-on-failure/--no-status-watch-download-artifacts-on-failure",
            help="Collect failed-run logs/artifacts when watch fails.",
        ),
        status_watch_artifact_output_root: Path | None = typer.Option(
            None,
            "--status-watch-artifact-output-root",
            help="Output root for failure bundle collection.",
        ),
        status_watch_artifact_name: list[str] | None = typer.Option(
            None,
            "--status-watch-artifact-name",
            help="Artifact name filter for failure downloads (repeatable).",
        ),
        status_watch_collect_failed_logs: bool | None = typer.Option(
            None,
            "--status-watch-collect-failed-logs/--no-status-watch-collect-failed-logs",
            help="Collect `gh run view --log-failed` output.",
        ),
        status_watch_summary_json: Path | None = typer.Option(
            None,
            "--status-watch-summary-json",
            help="Write status-watch summary JSON to this path.",
        ),
    ) -> None:
        if removed_analysis_tick_limit is not None:
            raise typer.BadParameter(
                "Removed --analysis-tick-limit. Use --analysis-budget-checks."
            )
        if (
            baseline_mode in {check_baseline_mode.enforce, check_baseline_mode.write}
            and baseline is None
        ):
            raise typer.BadParameter(
                "--baseline is required when --baseline-mode is enforce or write."
            )
        if baseline_mode is check_baseline_mode.off and baseline is not None:
            raise typer.BadParameter(
                "--baseline is only valid when --baseline-mode is enforce or write."
            )
        baseline_path = baseline if baseline_mode is not check_baseline_mode.off else None
        baseline_write = baseline_mode is check_baseline_mode.write
        status_watch_options = build_status_watch_options_fn(
            status_watch=status_watch,
            run_id=status_watch_run_id,
            branch=status_watch_branch,
            workflow=status_watch_workflow,
            status=status_watch_status,
            prefer_active=status_watch_prefer_active,
            download_artifacts_on_failure=status_watch_download_artifacts_on_failure,
            artifact_output_root=status_watch_artifact_output_root,
            artifact_name=status_watch_artifact_name,
            collect_failed_logs=status_watch_collect_failed_logs,
            summary_json=status_watch_summary_json,
        )
        run_check_command_fn(
            ctx=ctx,
            paths=paths,
            report=report,
            root=root,
            config=config,
            baseline=baseline_path,
            baseline_write=baseline_write,
            decision_snapshot=decision_snapshot,
            artifact_flags=default_check_artifact_flags_fn(),
            delta_options=default_check_delta_options_fn(),
            exclude=None,
            filter_bundle=dataflow_filter_bundle_ctor(
                ignore_params_csv=None,
                transparent_decorators_csv=None,
            ),
            allow_external=allow_external,
            strictness=str(strictness.value),
            analysis_budget_checks=analysis_budget_checks,
            gate=gate,
            lint_mode=lint,
            lint_jsonl_out=lint_jsonl_out,
            lint_sarif_out=lint_sarif_out,
            aspf_trace_json=aspf_trace_json,
            aspf_import_trace=aspf_import_trace,
            aspf_equivalence_against=aspf_equivalence_against,
            aspf_opportunities_json=aspf_opportunities_json,
            aspf_state_json=aspf_state_json,
            aspf_delta_jsonl=aspf_delta_jsonl,
            aspf_import_state=aspf_import_state,
            aspf_semantic_surface=aspf_semantic_surface,
            status_watch_options=status_watch_options,
        )

    return check_run


def register_check_group_callback(
    *,
    check_app: typer.Typer,
    check_help_or_exit_fn: CheckHelpOrExitFn,
) -> Callable[..., None]:
    @check_app.callback()
    def check_group(
        ctx: typer.Context,
        removed_profile: str | None = typer.Option(None, "--profile", hidden=True),
        removed_emit_test_obsolescence: bool = typer.Option(
            False,
            "--emit-test-obsolescence",
            hidden=True,
        ),
        removed_emit_test_obsolescence_state: bool = typer.Option(
            False,
            "--emit-test-obsolescence-state",
            hidden=True,
        ),
        removed_emit_test_obsolescence_delta: bool = typer.Option(
            False,
            "--emit-test-obsolescence-delta",
            hidden=True,
        ),
        removed_test_obsolescence_state: Path | None = typer.Option(
            None,
            "--test-obsolescence-state",
            hidden=True,
        ),
        removed_emit_test_annotation_drift: bool = typer.Option(
            False,
            "--emit-test-annotation-drift",
            hidden=True,
        ),
        removed_emit_test_annotation_drift_delta: bool = typer.Option(
            False,
            "--emit-test-annotation-drift-delta",
            hidden=True,
        ),
        removed_write_test_annotation_drift_baseline: bool = typer.Option(
            False,
            "--write-test-annotation-drift-baseline",
            hidden=True,
        ),
        removed_test_annotation_drift_state: Path | None = typer.Option(
            None,
            "--test-annotation-drift-state",
            hidden=True,
        ),
        removed_emit_ambiguity_delta: bool = typer.Option(
            False,
            "--emit-ambiguity-delta",
            hidden=True,
        ),
        removed_emit_ambiguity_state: bool = typer.Option(
            False,
            "--emit-ambiguity-state",
            hidden=True,
        ),
        removed_ambiguity_state: Path | None = typer.Option(
            None,
            "--ambiguity-state",
            hidden=True,
        ),
        removed_write_ambiguity_baseline: bool = typer.Option(
            False,
            "--write-ambiguity-baseline",
            hidden=True,
        ),
        removed_write_test_obsolescence_baseline: bool = typer.Option(
            False,
            "--write-test-obsolescence-baseline",
            hidden=True,
        ),
    ) -> None:
        if removed_profile is not None:
            raise typer.BadParameter(
                "Removed --profile flag. Use `gabion check run` or `gabion check raw -- ...`."
            )
        if (
            removed_emit_test_obsolescence
            or removed_emit_test_obsolescence_state
            or removed_emit_test_obsolescence_delta
            or removed_test_obsolescence_state is not None
            or removed_emit_test_annotation_drift
            or removed_emit_test_annotation_drift_delta
            or removed_write_test_annotation_drift_baseline
            or removed_test_annotation_drift_state is not None
            or removed_emit_ambiguity_delta
            or removed_emit_ambiguity_state
            or removed_ambiguity_state is not None
            or removed_write_ambiguity_baseline
            or removed_write_test_obsolescence_baseline
        ):
            raise typer.BadParameter(
                "Removed legacy check modality flags. Use `gabion check obsolescence|annotation-drift|ambiguity|taint` subcommands."
            )
        check_help_or_exit_fn(ctx)

    return check_group


def register_check_delta_bundle_command(
    *,
    check_app: typer.Typer,
    check_strictness_mode: type,
    check_gate_mode: type,
    check_lint_mode: type,
    run_check_command_fn: RunCheckCommandFn,
    dataflow_filter_bundle_ctor: DataflowFilterBundleCtor,
    delta_bundle_artifact_flags_fn: CheckFlagsFactory,
    delta_bundle_delta_options_fn: DeltaOptionsFactory,
) -> Callable[..., None]:
    @check_app.command("delta-bundle")
    def check_delta_bundle(
        ctx: typer.Context,
        paths: list[Path] = typer.Argument(None),
        root: Path = typer.Option(Path("."), "--root"),
        config: Path | None = typer.Option(None, "--config"),
        report: Path | None = typer.Option(None, "--report"),
        strictness: check_strictness_mode = typer.Option(
            check_strictness_mode.high,
            "--strictness",
        ),
        allow_external: bool | None = typer.Option(
            None,
            "--allow-external/--no-allow-external",
        ),
        decision_snapshot: Path | None = typer.Option(
            None,
            "--decision-snapshot",
        ),
        analysis_budget_checks: int | None = typer.Option(
            None,
            "--analysis-budget-checks",
            min=1,
        ),
        aspf_trace_json: Path | None = typer.Option(
            None,
            "--aspf-trace-json",
        ),
        aspf_import_trace: list[Path] | None = typer.Option(
            None,
            "--aspf-import-trace",
        ),
        aspf_equivalence_against: list[Path] | None = typer.Option(
            None,
            "--aspf-equivalence-against",
        ),
        aspf_opportunities_json: Path | None = typer.Option(
            None,
            "--aspf-opportunities-json",
        ),
        aspf_state_json: Path | None = typer.Option(
            None,
            "--aspf-state-json",
        ),
        aspf_delta_jsonl: Path | None = typer.Option(
            None,
            "--aspf-delta-jsonl",
        ),
        aspf_import_state: list[Path] | None = typer.Option(
            None,
            "--aspf-import-state",
        ),
        aspf_semantic_surface: list[str] | None = typer.Option(
            None,
            "--aspf-semantic-surface",
        ),
    ) -> None:
        run_check_command_fn(
            ctx=ctx,
            paths=paths,
            report=report,
            root=root,
            config=config,
            baseline=None,
            baseline_write=False,
            decision_snapshot=decision_snapshot,
            artifact_flags=delta_bundle_artifact_flags_fn(),
            delta_options=delta_bundle_delta_options_fn(),
            exclude=None,
            filter_bundle=dataflow_filter_bundle_ctor(
                ignore_params_csv=None,
                transparent_decorators_csv=None,
            ),
            allow_external=allow_external,
            strictness=str(strictness.value),
            analysis_budget_checks=analysis_budget_checks,
            gate=check_gate_mode.none,
            lint_mode=check_lint_mode.none,
            lint_jsonl_out=None,
            lint_sarif_out=None,
            aspf_trace_json=aspf_trace_json,
            aspf_import_trace=aspf_import_trace,
            aspf_equivalence_against=aspf_equivalence_against,
            aspf_opportunities_json=aspf_opportunities_json,
            aspf_state_json=aspf_state_json,
            aspf_delta_jsonl=aspf_delta_jsonl,
            aspf_import_state=aspf_import_state,
            aspf_semantic_surface=aspf_semantic_surface,
        )

    return check_delta_bundle
