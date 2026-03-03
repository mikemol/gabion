# gabion:boundary_normalization_module
# gabion:decision_protocol_module

from pathlib import Path
from typing import Callable

import typer

BuildStatusWatchOptionsFn = Callable[..., object | None]
RunCheckCommandFn = Callable[..., None]
DataflowFilterBundleCtor = Callable[..., object]
CheckFlagsFactory = Callable[[], object]
CheckHelpOrExitFn = Callable[[typer.Context], None]
DeltaOptionsFactory = Callable[[], object]


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
