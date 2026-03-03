# gabion:boundary_normalization_module
# gabion:decision_protocol_module

from pathlib import Path
from typing import Callable, Protocol

import typer


class _StatusWatchOptions(Protocol):
    def to_argv(self) -> list[str]: ...


StatusWatchOptionsCtor = Callable[..., _StatusWatchOptions]
RunToolingWithArgvFn = Callable[[str, list[str]], int]


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
