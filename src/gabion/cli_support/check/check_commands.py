# gabion:boundary_normalization_module
# gabion:decision_protocol_module

from dataclasses import dataclass, replace
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
class CheckRunDecisionResolution:
    baseline: Path | None
    baseline_write: bool
    status_watch_options: object | None


@dataclass(frozen=True)
class CheckRunOptionBundle:
    ctx: typer.Context
    paths: list[Path]
    root: Path
    config: Path | None
    report: Path | None
    strictness: object
    allow_external: bool | None
    baseline: Path | None
    baseline_mode: object | None
    gate: object
    analysis_budget_checks: int | None
    decision_snapshot: Path | None
    lint_mode: object
    lint_jsonl_out: Path | None
    lint_sarif_out: Path | None
    aspf_trace_json: Path | None
    aspf_import_trace: list[Path] | None
    aspf_equivalence_against: list[Path] | None
    aspf_opportunities_json: Path | None
    aspf_state_json: Path | None
    aspf_delta_jsonl: Path | None
    aspf_import_state: list[Path] | None
    aspf_semantic_surface: list[str] | None
    status_watch: bool = False
    status_watch_run_id: str | None = None
    status_watch_branch: str | None = None
    status_watch_workflow: str | None = None
    status_watch_status: str | None = None
    status_watch_prefer_active: bool | None = None
    status_watch_download_artifacts_on_failure: bool | None = None
    status_watch_artifact_output_root: Path | None = None
    status_watch_artifact_name: list[str] | None = None
    status_watch_collect_failed_logs: bool | None = None
    status_watch_summary_json: Path | None = None

    def with_overrides(self, **overrides: object) -> "CheckRunOptionBundle":
        return replace(self, **overrides)


@dataclass(frozen=True)
class CheckRunDecisionProtocol:
    option_bundle: CheckRunOptionBundle
    check_baseline_mode: type | None
    build_status_watch_options_fn: BuildStatusWatchOptionsFn | None

    def resolve(self) -> CheckRunDecisionResolution:
        baseline = self.option_bundle.baseline
        baseline_write = False
        baseline_mode = self.option_bundle.baseline_mode
        if self.check_baseline_mode is not None:
            if baseline_mode is None:
                raise typer.BadParameter("--baseline-mode is required.")
            baseline_enum = self.check_baseline_mode
            if baseline_mode in {baseline_enum.enforce, baseline_enum.write} and baseline is None:
                raise typer.BadParameter(
                    "--baseline is required when --baseline-mode is enforce or write."
                )
            if baseline_mode is baseline_enum.off and baseline is not None:
                raise typer.BadParameter(
                    "--baseline is only valid when --baseline-mode is enforce or write."
                )
            baseline = baseline if baseline_mode is not baseline_enum.off else None
            baseline_write = baseline_mode is baseline_enum.write

        status_watch_options: object | None = None
        if self.build_status_watch_options_fn is not None:
            status_watch_options = self.build_status_watch_options_fn(
                status_watch=self.option_bundle.status_watch,
                run_id=self.option_bundle.status_watch_run_id,
                branch=self.option_bundle.status_watch_branch,
                workflow=self.option_bundle.status_watch_workflow,
                status=self.option_bundle.status_watch_status,
                prefer_active=self.option_bundle.status_watch_prefer_active,
                download_artifacts_on_failure=self.option_bundle.status_watch_download_artifacts_on_failure,
                artifact_output_root=self.option_bundle.status_watch_artifact_output_root,
                artifact_name=self.option_bundle.status_watch_artifact_name,
                collect_failed_logs=self.option_bundle.status_watch_collect_failed_logs,
                summary_json=self.option_bundle.status_watch_summary_json,
            )

        return CheckRunDecisionResolution(
            baseline=baseline,
            baseline_write=baseline_write,
            status_watch_options=status_watch_options,
        )


def _check_run_runtime_kwargs_from_bundle(
    *,
    option_bundle: CheckRunOptionBundle,
    check_baseline_mode: type | None,
    build_status_watch_options_fn: BuildStatusWatchOptionsFn | None,
    artifact_flags: object,
    delta_options: object,
    dataflow_filter_bundle_ctor: DataflowFilterBundleCtor,
) -> dict[str, object]:
    decision = CheckRunDecisionProtocol(
        option_bundle=option_bundle,
        check_baseline_mode=check_baseline_mode,
        build_status_watch_options_fn=build_status_watch_options_fn,
    ).resolve()
    return {
        "ctx": option_bundle.ctx,
        "paths": option_bundle.paths,
        "report": option_bundle.report,
        "root": option_bundle.root,
        "config": option_bundle.config,
        "baseline": decision.baseline,
        "baseline_write": decision.baseline_write,
        "decision_snapshot": option_bundle.decision_snapshot,
        "artifact_flags": artifact_flags,
        "delta_options": delta_options,
        "exclude": None,
        "filter_bundle": dataflow_filter_bundle_ctor(
            ignore_params_csv=None,
            transparent_decorators_csv=None,
        ),
        "allow_external": option_bundle.allow_external,
        "strictness": str(option_bundle.strictness.value),
        "analysis_budget_checks": option_bundle.analysis_budget_checks,
        "gate": option_bundle.gate,
        "lint_mode": option_bundle.lint_mode,
        "lint_jsonl_out": option_bundle.lint_jsonl_out,
        "lint_sarif_out": option_bundle.lint_sarif_out,
        "aspf_trace_json": option_bundle.aspf_trace_json,
        "aspf_import_trace": option_bundle.aspf_import_trace,
        "aspf_equivalence_against": option_bundle.aspf_equivalence_against,
        "aspf_opportunities_json": option_bundle.aspf_opportunities_json,
        "aspf_state_json": option_bundle.aspf_state_json,
        "aspf_delta_jsonl": option_bundle.aspf_delta_jsonl,
        "aspf_import_state": option_bundle.aspf_import_state,
        "aspf_semantic_surface": option_bundle.aspf_semantic_surface,
        "status_watch_options": decision.status_watch_options,
    }


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
        option_bundle = CheckRunOptionBundle(
            ctx=ctx,
            paths=paths,
            root=root,
            config=config,
            report=report,
            strictness=strictness,
            allow_external=allow_external,
            baseline=baseline,
            baseline_mode=baseline_mode,
            gate=gate,
            analysis_budget_checks=analysis_budget_checks,
            decision_snapshot=decision_snapshot,
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
            status_watch=status_watch,
            status_watch_run_id=status_watch_run_id,
            status_watch_branch=status_watch_branch,
            status_watch_workflow=status_watch_workflow,
            status_watch_status=status_watch_status,
            status_watch_prefer_active=status_watch_prefer_active,
            status_watch_download_artifacts_on_failure=status_watch_download_artifacts_on_failure,
            status_watch_artifact_output_root=status_watch_artifact_output_root,
            status_watch_artifact_name=status_watch_artifact_name,
            status_watch_collect_failed_logs=status_watch_collect_failed_logs,
            status_watch_summary_json=status_watch_summary_json,
        )
        runtime_kwargs = _check_run_runtime_kwargs_from_bundle(
            option_bundle=option_bundle,
            check_baseline_mode=check_baseline_mode,
            build_status_watch_options_fn=build_status_watch_options_fn,
            artifact_flags=default_check_artifact_flags_fn(),
            delta_options=default_check_delta_options_fn(),
            dataflow_filter_bundle_ctor=dataflow_filter_bundle_ctor,
        )
        run_check_command_fn(**runtime_kwargs)

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
        option_bundle = CheckRunOptionBundle(
            ctx=ctx,
            paths=paths,
            root=root,
            config=config,
            report=report,
            strictness=strictness,
            allow_external=allow_external,
            baseline=None,
            baseline_mode=None,
            gate=check_gate_mode.all,
            analysis_budget_checks=analysis_budget_checks,
            decision_snapshot=decision_snapshot,
            lint_mode=check_lint_mode.all,
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
        runtime_kwargs = _check_run_runtime_kwargs_from_bundle(
            option_bundle=option_bundle.with_overrides(
                baseline=None,
                baseline_mode=None,
                gate=check_gate_mode.none,
                lint_mode=check_lint_mode.none,
            ),
            check_baseline_mode=None,
            build_status_watch_options_fn=None,
            artifact_flags=delta_bundle_artifact_flags_fn(),
            delta_options=delta_bundle_delta_options_fn(),
            dataflow_filter_bundle_ctor=dataflow_filter_bundle_ctor,
        )
        run_check_command_fn(**runtime_kwargs)

    return check_delta_bundle
