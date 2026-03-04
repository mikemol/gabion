# gabion:boundary_normalization_module
# gabion:decision_protocol_module

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Protocol

import typer


class _StatusWatchOptions(Protocol):
    def to_argv(self) -> list[str]: ...


StatusWatchOptionsCtor = Callable[..., _StatusWatchOptions]
RunToolingWithArgvFn = Callable[[str, list[str]], int]
RunToolingNoArgFn = Callable[[str], int]


def invoke_argparse_command(
    main_fn: Callable[[list[str] | None], int],
    argv: list[str],
) -> int:
    try:
        return int(main_fn(argv))
    except SystemExit as exc:
        code = exc.code
        if isinstance(code, int):
            return int(code)
        return 1


@contextmanager
def tooling_runner_override(
    *,
    no_arg_runners: dict[str, Callable[[], int]],
    with_argv_runners: dict[str, Callable[[list[str] | None], int]],
    no_arg: Mapping[str, Callable[[], int]] | None = None,
    with_argv: Mapping[str, Callable[[list[str] | None], int]] | None = None,
) -> Generator[None, None, None]:
    previous_no_arg = dict(no_arg_runners)
    previous_with_argv = dict(with_argv_runners)
    if isinstance(no_arg, Mapping):
        no_arg_runners.update(
            {str(key): value for key, value in no_arg.items() if callable(value)}
        )
    if isinstance(with_argv, Mapping):
        with_argv_runners.update(
            {str(key): value for key, value in with_argv.items() if callable(value)}
        )
    try:
        yield
    finally:
        no_arg_runners.clear()
        no_arg_runners.update(previous_no_arg)
        with_argv_runners.clear()
        with_argv_runners.update(previous_with_argv)


def run_tooling_no_arg(
    *,
    command_name: str,
    no_arg_runners: dict[str, Callable[[], int]],
    cli_deadline_scope_factory: Callable[[], object],
) -> int:
    runner = no_arg_runners[command_name]
    with cli_deadline_scope_factory():
        return int(runner())


def run_tooling_with_argv(
    *,
    command_name: str,
    argv: list[str],
    with_argv_runners: dict[str, Callable[[list[str] | None], int]],
    cli_deadline_scope_factory: Callable[[], object],
) -> int:
    runner = with_argv_runners[command_name]
    with cli_deadline_scope_factory():
        return invoke_argparse_command(runner, argv)


def register_tooling_passthrough_commands(
    *,
    app: typer.Typer,
    run_tooling_no_arg_fn: RunToolingNoArgFn,
    run_tooling_with_argv_fn: RunToolingWithArgvFn,
) -> dict[str, Callable[..., None]]:
    @app.command("delta-advisory-telemetry")
    def delta_advisory_telemetry() -> None:
        """Emit non-blocking advisory telemetry artifacts."""
        raise typer.Exit(code=run_tooling_no_arg_fn("delta-advisory-telemetry"))

    @app.command("docflow-delta-emit")
    def docflow_delta_emit() -> None:
        """Emit docflow compliance delta through the gabion CLI."""
        raise typer.Exit(code=run_tooling_no_arg_fn("docflow-delta-emit"))

    @app.command(
        "impact-select-tests",
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    )
    def impact_select_tests(ctx: typer.Context) -> None:
        """Select impacted tests from diffs and evidence index."""
        raise typer.Exit(
            code=run_tooling_with_argv_fn(
                "impact-select-tests",
                list(ctx.args),
            )
        )

    @app.command(
        "run-dataflow-stage",
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    )
    def run_dataflow_stage(ctx: typer.Context) -> None:
        """Run a single dataflow stage with CI-aligned outputs."""
        raise typer.Exit(
            code=run_tooling_with_argv_fn(
                "run-dataflow-stage",
                list(ctx.args),
            )
        )

    @app.command(
        "ambiguity-contract-gate",
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    )
    def ambiguity_contract_gate(ctx: typer.Context) -> None:
        """Run ambiguity-contract policy gate for deterministic-core surfaces."""
        raise typer.Exit(
            code=run_tooling_with_argv_fn(
                "ambiguity-contract-gate",
                list(ctx.args),
            )
        )

    @app.command(
        "normative-symdiff",
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    )
    def normative_symdiff(ctx: typer.Context) -> None:
        """Compute a normative-docs ↔ code/tooling symmetric-difference report."""
        raise typer.Exit(
            code=run_tooling_with_argv_fn(
                "normative-symdiff",
                list(ctx.args),
            )
        )

    return {
        "delta_advisory_telemetry": delta_advisory_telemetry,
        "docflow_delta_emit": docflow_delta_emit,
        "impact_select_tests": impact_select_tests,
        "run_dataflow_stage": run_dataflow_stage,
        "ambiguity_contract_gate": ambiguity_contract_gate,
        "normative_symdiff": normative_symdiff,
    }


def build_status_watch_options(
    *,
    status_watch: bool,
    run_id: str | None,
    branch: str | None,
    workflow: str | None,
    status: str | None,
    prefer_active: bool | None,
    download_artifacts_on_failure: bool | None,
    artifact_output_root: Path | None,
    artifact_name: list[str] | None,
    collect_failed_logs: bool | None,
    summary_json: Path | None,
    default_status_watch_artifact_root: Path,
    status_watch_options_ctor: StatusWatchOptionsCtor,
) -> _StatusWatchOptions | None:
    has_prefixed_options = any(
        (
            run_id is not None,
            branch is not None,
            workflow is not None,
            status is not None,
            prefer_active is not None,
            download_artifacts_on_failure is not None,
            artifact_output_root is not None,
            bool(artifact_name),
            collect_failed_logs is not None,
            summary_json is not None,
        )
    )
    if not status_watch:
        if has_prefixed_options:
            raise typer.BadParameter("--status-watch-* options require --status-watch.")
        return None
    return status_watch_options_ctor(
        branch=str(branch or "stage"),
        run_id=str(run_id) if run_id else None,
        status=str(status) if status else None,
        workflow=str(workflow) if workflow else None,
        prefer_active=True if prefer_active is None else bool(prefer_active),
        download_artifacts_on_failure=(
            True
            if download_artifacts_on_failure is None
            else bool(download_artifacts_on_failure)
        ),
        artifact_output_root=artifact_output_root or default_status_watch_artifact_root,
        artifact_names=tuple(str(name) for name in (artifact_name or [])),
        collect_failed_logs=(True if collect_failed_logs is None else bool(collect_failed_logs)),
        summary_json=summary_json,
    )


def register_ci_watch_command(
    *,
    app: typer.Typer,
    default_status_watch_artifact_root: Path,
    status_watch_options_ctor: StatusWatchOptionsCtor,
    run_tooling_with_argv_fn: RunToolingWithArgvFn,
) -> Callable[..., None]:
    @app.command("ci-watch")
    def ci_watch(
        run_id: str | None = typer.Option(
            None,
            "--run-id",
            help="Specific run id to watch (skips lookup).",
        ),
        branch: str = typer.Option(
            "stage",
            "--branch",
            help="Branch to watch (default: stage).",
        ),
        workflow: str | None = typer.Option(
            None,
            "--workflow",
            help="Optional workflow name or file filter.",
        ),
        status: str | None = typer.Option(
            None,
            "--status",
            help="Optional status filter for fallback run lookup.",
        ),
        prefer_active: bool = typer.Option(
            True,
            "--prefer-active/--no-prefer-active",
            help="Prefer in-progress/queued runs during run lookup.",
        ),
        download_artifacts_on_failure: bool = typer.Option(
            True,
            "--download-artifacts-on-failure/--no-download-artifacts-on-failure",
            help="Collect failure metadata/logs/artifacts when watched run fails.",
        ),
        artifact_output_root: Path = typer.Option(
            default_status_watch_artifact_root,
            "--artifact-output-root",
            help="Root path for failure bundle collection.",
        ),
        artifact_name: list[str] | None = typer.Option(
            None,
            "--artifact-name",
            help="Artifact name filter (repeatable).",
        ),
        collect_failed_logs: bool = typer.Option(
            True,
            "--collect-failed-logs/--no-collect-failed-logs",
            help="Collect `gh run view --log-failed` output on failure.",
        ),
        summary_json: Path | None = typer.Option(
            None,
            "--summary-json",
            help="Write watch/collection summary JSON to this path.",
        ),
    ) -> None:
        options = status_watch_options_ctor(
            branch=branch,
            run_id=run_id,
            status=status,
            workflow=workflow,
            prefer_active=prefer_active,
            download_artifacts_on_failure=download_artifacts_on_failure,
            artifact_output_root=artifact_output_root,
            artifact_names=tuple(artifact_name or []),
            collect_failed_logs=collect_failed_logs,
            summary_json=summary_json,
        )
        raise typer.Exit(code=run_tooling_with_argv_fn("ci-watch", options.to_argv()))

    return ci_watch
