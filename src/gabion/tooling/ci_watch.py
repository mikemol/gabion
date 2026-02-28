# gabion:decision_protocol_module
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

from gabion.analysis.timeout_context import check_deadline
from gabion.tooling.deadline_runtime import DeadlineBudget, deadline_scope_from_lsp_env

_DEFAULT_TIMEOUT_TICKS = 120_000
_DEFAULT_TIMEOUT_TICK_NS = 1_000_000
_DEFAULT_TIMEOUT_BUDGET = DeadlineBudget(
    ticks=_DEFAULT_TIMEOUT_TICKS,
    tick_ns=_DEFAULT_TIMEOUT_TICK_NS,
)
_DEFAULT_FAILURE_ARTIFACT_ROOT = Path("artifacts/out/ci_watch")
_COLLECTION_FAILURE_EXIT = 2

RunCommand = Callable[..., subprocess.CompletedProcess[str]]
PrintErr = Callable[[str], None]


@dataclass(frozen=True)
class CiWatchDeps:
    run: RunCommand
    print_err: PrintErr


@dataclass(frozen=True)
class CollectionStatus:
    run_view_json_rc: int
    log_failed_rc: int | None
    download_rc: int
    failed_job_count: int
    failed_step_count: int
    artifact_file_count: int

    def mandatory_failures(self, *, collect_failed_logs: bool) -> list[str]:
        failures: list[str] = []
        if self.run_view_json_rc != 0:
            failures.append("run_view_json")
        if collect_failed_logs and self.log_failed_rc not in (None, 0):
            failures.append("log_failed")
        if self.download_rc != 0:
            failures.append("download")
        return failures

    def to_payload(self, *, collect_failed_logs: bool) -> dict[str, Any]:
        mandatory_failures = self.mandatory_failures(
            collect_failed_logs=collect_failed_logs
        )
        return {
            "run_view_json_rc": self.run_view_json_rc,
            "log_failed_rc": self.log_failed_rc,
            "download_rc": self.download_rc,
            "failed_job_count": self.failed_job_count,
            "failed_step_count": self.failed_step_count,
            "artifact_file_count": self.artifact_file_count,
            "collect_failed_logs": collect_failed_logs,
            "mandatory_failures": mandatory_failures,
            "collection_success": not mandatory_failures,
        }


@dataclass(frozen=True)
class FailureCollectionResult:
    run_root: Path
    status: CollectionStatus


@dataclass(frozen=True)
class StatusWatchOptions:
    branch: str = "stage"
    run_id: str | None = None
    status: str | None = None
    workflow: str | None = None
    prefer_active: bool = True
    download_artifacts_on_failure: bool = True
    artifact_output_root: Path = _DEFAULT_FAILURE_ARTIFACT_ROOT
    artifact_names: tuple[str, ...] = ()
    collect_failed_logs: bool = True
    summary_json: Path | None = None

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "StatusWatchOptions":
        return cls(
            branch=str(args.branch),
            run_id=str(args.run_id) if args.run_id else None,
            status=str(args.status) if args.status else None,
            workflow=str(args.workflow) if args.workflow else None,
            prefer_active=bool(args.prefer_active),
            download_artifacts_on_failure=bool(args.download_artifacts_on_failure),
            artifact_output_root=Path(str(args.artifact_output_root)),
            artifact_names=tuple(str(name) for name in list(args.artifact_name or [])),
            collect_failed_logs=bool(args.collect_failed_logs),
            summary_json=Path(str(args.summary_json)) if args.summary_json else None,
        )

    def to_argv(self) -> list[str]:
        argv: list[str] = ["--branch", self.branch]
        if self.run_id:
            argv.extend(["--run-id", self.run_id])
        if self.status:
            argv.extend(["--status", self.status])
        if self.workflow:
            argv.extend(["--workflow", self.workflow])
        argv.append("--prefer-active" if self.prefer_active else "--no-prefer-active")
        argv.append(
            "--download-artifacts-on-failure"
            if self.download_artifacts_on_failure
            else "--no-download-artifacts-on-failure"
        )
        argv.extend(["--artifact-output-root", str(self.artifact_output_root)])
        for name in self.artifact_names:
            argv.extend(["--artifact-name", str(name)])
        argv.append(
            "--collect-failed-logs"
            if self.collect_failed_logs
            else "--no-collect-failed-logs"
        )
        if self.summary_json is not None:
            argv.extend(["--summary-json", str(self.summary_json)])
        return argv


@dataclass(frozen=True)
class StatusWatchResult:
    run_id: str
    watch_exit_code: int
    exit_code: int
    artifact_output_root: Path
    collection: FailureCollectionResult | None

    def summary_payload(
        self,
        *,
        options: StatusWatchOptions,
    ) -> dict[str, Any]:
        collection_payload: dict[str, Any] | None = None
        if self.collection is not None:
            status_payload = self.collection.status.to_payload(
                collect_failed_logs=options.collect_failed_logs
            )
            collection_payload = {
                "run_root": str(self.collection.run_root),
                "status": status_payload,
            }
        return {
            "run_id": self.run_id,
            "watch_exit_code": self.watch_exit_code,
            "exit_code": self.exit_code,
            "artifact_output_root": str(self.artifact_output_root),
            "download_artifacts_on_failure": options.download_artifacts_on_failure,
            "collect_failed_logs": options.collect_failed_logs,
            "collection": collection_payload,
        }


def _deadline_scope():
    return deadline_scope_from_lsp_env(
        default_budget=_DEFAULT_TIMEOUT_BUDGET,
    )


def _default_print_err(message: str) -> None:
    print(message, file=sys.stderr)


def _default_deps() -> CiWatchDeps:
    return CiWatchDeps(run=subprocess.run, print_err=_default_print_err)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch the latest GitHub Actions run for a branch.",
    )
    parser.add_argument(
        "--branch",
        default="stage",
        help="Branch to watch (default: stage)",
    )
    parser.add_argument(
        "--run-id",
        help="Specific run id to watch (skips lookup).",
    )
    parser.add_argument(
        "--status",
        help="Optional status filter for the fallback run lookup.",
    )
    parser.add_argument(
        "--workflow",
        help="Optional workflow name or file to filter runs.",
    )
    parser.add_argument(
        "--prefer-active",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer in-progress/queued runs when choosing the latest run.",
    )
    parser.add_argument(
        "--download-artifacts-on-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When the watched run fails, collect metadata/logs and download artifacts "
            "(default: true)."
        ),
    )
    parser.add_argument(
        "--artifact-output-root",
        default=str(_DEFAULT_FAILURE_ARTIFACT_ROOT),
        help=(
            "Root path for failure collections "
            "(default: artifacts/out/ci_watch)."
        ),
    )
    parser.add_argument(
        "--artifact-name",
        action="append",
        default=[],
        help=(
            "Artifact name to download on failure. Repeat for multiple names. "
            "If omitted, download all available artifacts."
        ),
    )
    parser.add_argument(
        "--collect-failed-logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collect `gh run view --log-failed` output on failure (default: true).",
    )
    parser.add_argument(
        "--summary-json",
        help="Write watch/collection summary JSON to this path.",
    )
    return parser.parse_args(argv)


# gabion:boundary_normalization
def _decode_json_dict(payload: str) -> dict[str, Any]:
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


# gabion:boundary_normalization
def _failed_jobs(run_payload: dict[str, Any]) -> list[dict[str, Any]]:
    check_deadline()
    raw_jobs = run_payload.get("jobs")
    if not isinstance(raw_jobs, list):
        return []
    failed: list[dict[str, Any]] = []
    for raw_job in raw_jobs:
        check_deadline()
        if not isinstance(raw_job, dict):
            continue
        conclusion = raw_job.get("conclusion")
        if conclusion != "failure":
            continue
        failed.append(
            {
                "databaseId": raw_job.get("databaseId"),
                "name": raw_job.get("name"),
                "status": raw_job.get("status"),
                "conclusion": conclusion,
                "url": raw_job.get("url"),
            }
        )
    return failed


# gabion:boundary_normalization
def _failed_steps(run_payload: dict[str, Any]) -> list[dict[str, Any]]:
    check_deadline()
    raw_jobs = run_payload.get("jobs")
    if not isinstance(raw_jobs, list):
        return []
    failed_steps: list[dict[str, Any]] = []
    for raw_job in raw_jobs:
        check_deadline()
        if not isinstance(raw_job, dict):
            continue
        job_name = raw_job.get("name")
        raw_steps = raw_job.get("steps")
        if not isinstance(raw_steps, list):
            continue
        for raw_step in raw_steps:
            check_deadline()
            if not isinstance(raw_step, dict):
                continue
            if raw_step.get("conclusion") != "failure":
                continue
            failed_steps.append(
                {
                    "job_name": job_name,
                    "job_databaseId": raw_job.get("databaseId"),
                    "step_number": raw_step.get("number"),
                    "step_name": raw_step.get("name"),
                    "status": raw_step.get("status"),
                    "conclusion": raw_step.get("conclusion"),
                    "job_url": raw_job.get("url"),
                }
            )
    return failed_steps


def _artifacts_manifest(artifacts_root: Path) -> dict[str, Any]:
    if not artifacts_root.exists():
        return {
            "root": str(artifacts_root),
            "artifact_dirs": [],
            "files": [],
        }
    artifact_dirs = sorted(
        path.name for path in artifacts_root.iterdir() if path.is_dir()
    )
    files = sorted(
        str(path.relative_to(artifacts_root))
        for path in artifacts_root.rglob("*")
        if path.is_file()
    )
    return {
        "root": str(artifacts_root),
        "artifact_dirs": artifact_dirs,
        "files": files,
    }


def _find_run_id(
    deps: CiWatchDeps,
    branch: str,
    status: str | None,
    workflow: str | None,
) -> str | None:
    cmd = [
        "gh",
        "run",
        "list",
        "--branch",
        branch,
        "--limit",
        "1",
        "--json",
        "databaseId,status,conclusion,headSha,displayTitle",
    ]
    if workflow:
        cmd.extend(["--workflow", workflow])
    if status:
        cmd.extend(["--status", status])
    payload = deps.run(cmd, check=True, capture_output=True, text=True).stdout.strip()
    data: list[dict[str, Any]] = json.loads(payload)
    if not data:
        return None
    return str(data[0]["databaseId"])


def _collect_failure_artifacts(
    deps: CiWatchDeps,
    *,
    run_id: str,
    output_root: Path,
    artifact_names: list[str],
    collect_failed_logs: bool,
) -> FailureCollectionResult:
    check_deadline()
    run_root = output_root / f"run_{run_id}"
    run_root.mkdir(parents=True, exist_ok=True)

    check_deadline()
    run_view = deps.run(
        [
            "gh",
            "run",
            "view",
            run_id,
            "--json",
            "name,url,status,conclusion,headSha,jobs",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    _write_text(run_root / "run.json", run_view.stdout)
    _write_text(run_root / "run_view.stderr.log", run_view.stderr)

    run_payload = _decode_json_dict(run_view.stdout)
    failed_jobs = _failed_jobs(run_payload)
    failed_steps = _failed_steps(run_payload)
    _write_json(run_root / "failed_jobs.json", failed_jobs)
    _write_json(run_root / "failed_steps.json", failed_steps)

    failed_logs_stdout = ""
    failed_logs_stderr = ""
    log_failed_rc: int | None = None

    if collect_failed_logs:
        check_deadline()
        failed_logs = deps.run(
            ["gh", "run", "view", run_id, "--log-failed"],
            check=False,
            capture_output=True,
            text=True,
        )
        failed_logs_stdout = failed_logs.stdout
        failed_logs_stderr = failed_logs.stderr
        log_failed_rc = failed_logs.returncode
    _write_text(run_root / "failed.log", failed_logs_stdout)
    _write_text(run_root / "failed.stderr.log", failed_logs_stderr)

    download_cmd = [
        "gh",
        "run",
        "download",
        run_id,
        "--dir",
        str(run_root / "artifacts"),
    ]
    for artifact_name in artifact_names:
        check_deadline()
        download_cmd.extend(["--name", artifact_name])
    check_deadline()
    artifact_download = deps.run(
        download_cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    _write_text(run_root / "download.stdout.log", artifact_download.stdout)
    _write_text(run_root / "download.stderr.log", artifact_download.stderr)

    manifest = _artifacts_manifest(run_root / "artifacts")
    _write_json(run_root / "artifacts_manifest.json", manifest)

    status = CollectionStatus(
        run_view_json_rc=run_view.returncode,
        log_failed_rc=log_failed_rc,
        download_rc=artifact_download.returncode,
        failed_job_count=len(failed_jobs),
        failed_step_count=len(failed_steps),
        artifact_file_count=len(manifest["files"]),
    )
    _write_json(
        run_root / "collection_status.json",
        status.to_payload(collect_failed_logs=collect_failed_logs),
    )
    return FailureCollectionResult(run_root=run_root, status=status)


def run_watch(
    *,
    options: StatusWatchOptions,
    deps: CiWatchDeps | None = None,
) -> StatusWatchResult:
    with _deadline_scope():
        active_deps = deps or _default_deps()
        run_id = options.run_id
        if not run_id:
            if options.prefer_active:
                for status in (
                    "in_progress",
                    "queued",
                    "requested",
                    "waiting",
                    "pending",
                ):
                    check_deadline()
                    run_id = _find_run_id(
                        active_deps,
                        options.branch,
                        status,
                        options.workflow,
                    )
                    if run_id:
                        break
            if not run_id:
                run_id = _find_run_id(
                    active_deps,
                    options.branch,
                    options.status,
                    options.workflow,
                )
        if not run_id:
            raise SystemExit(f"No runs found for branch {options.branch}")
        watch_proc = active_deps.run(
            ["gh", "run", "watch", run_id, "--exit-status"],
            check=False,
        )
        watch_rc = watch_proc.returncode
        collection: FailureCollectionResult | None = None
        exit_code = watch_rc
        if watch_rc != 0 and options.download_artifacts_on_failure:
            collection = _collect_failure_artifacts(
                active_deps,
                run_id=run_id,
                output_root=options.artifact_output_root,
                artifact_names=list(options.artifact_names),
                collect_failed_logs=options.collect_failed_logs,
            )
            failures = collection.status.mandatory_failures(
                collect_failed_logs=options.collect_failed_logs
            )
            if failures:
                active_deps.print_err(
                    "ci_watch: run "
                    f"{run_id} failed; collection had failures ({', '.join(failures)}) "
                    f"under {collection.run_root}"
                )
                exit_code = _COLLECTION_FAILURE_EXIT
            else:
                active_deps.print_err(
                    f"ci_watch: run {run_id} failed; collected artifacts under {collection.run_root}"
                )
        result = StatusWatchResult(
            run_id=run_id,
            watch_exit_code=watch_rc,
            exit_code=exit_code,
            artifact_output_root=options.artifact_output_root,
            collection=collection,
        )
        if options.summary_json is not None:
            _write_json(options.summary_json, result.summary_payload(options=options))
        return result


def main(argv: list[str] | None = None, deps: CiWatchDeps | None = None) -> int:
    args = _parse_args(argv)
    options = StatusWatchOptions.from_namespace(args)
    result = run_watch(options=options, deps=deps)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
