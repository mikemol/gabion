from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:  # pragma: no cover - import form depends on invocation mode
    from scripts.deadline_runtime import DeadlineBudget, deadline_scope_from_lsp_env
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from deadline_runtime import DeadlineBudget, deadline_scope_from_lsp_env
from gabion.analysis.timeout_context import check_deadline

_DEFAULT_TIMEOUT_TICKS = 120_000
_DEFAULT_TIMEOUT_TICK_NS = 1_000_000
_DEFAULT_TIMEOUT_BUDGET = DeadlineBudget(
    ticks=_DEFAULT_TIMEOUT_TICKS,
    tick_ns=_DEFAULT_TIMEOUT_TICK_NS,
)
_DEFAULT_FAILURE_ARTIFACT_ROOT = Path("artifacts/out/ci_watch")


def _deadline_scope():
    return deadline_scope_from_lsp_env(
        default_budget=_DEFAULT_TIMEOUT_BUDGET,
    )


def _run(*cmd: str) -> str:
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return proc.stdout.strip()


def _write_output(path: Path, content: str) -> None:
    if not content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _find_run_id(
    branch: str, status: str | None, workflow: str | None
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
    payload = _run(*cmd)
    data: list[dict[str, Any]] = json.loads(payload)
    if not data:
        return None
    return str(data[0]["databaseId"])


def _collect_failure_artifacts(
    *,
    run_id: str,
    output_root: Path,
    artifact_names: list[str],
    collect_failed_logs: bool,
) -> Path:
    check_deadline()
    run_root = output_root / f"run_{run_id}"
    run_root.mkdir(parents=True, exist_ok=True)

    check_deadline()
    run_view = subprocess.run(
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
    _write_output(run_root / "run.json", run_view.stdout)
    _write_output(run_root / "run_view.stderr.log", run_view.stderr)

    if collect_failed_logs:
        check_deadline()
        failed_logs = subprocess.run(
            ["gh", "run", "view", run_id, "--log-failed"],
            check=False,
            capture_output=True,
            text=True,
        )
        _write_output(run_root / "failed.log", failed_logs.stdout)
        _write_output(run_root / "failed.stderr.log", failed_logs.stderr)

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
    artifact_download = subprocess.run(
        download_cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    _write_output(run_root / "download.stdout.log", artifact_download.stdout)
    _write_output(run_root / "download.stderr.log", artifact_download.stderr)
    return run_root


def main() -> int:
    with _deadline_scope():
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
        args = parser.parse_args()

        run_id = args.run_id
        if not run_id:
            if args.prefer_active:
                for status in ("in_progress", "queued", "requested", "waiting", "pending"):
                    check_deadline()
                    run_id = _find_run_id(args.branch, status, args.workflow)
                    if run_id:
                        break
            if not run_id:
                run_id = _find_run_id(args.branch, args.status, args.workflow)
        if not run_id:
            raise SystemExit(f"No runs found for branch {args.branch}")
        watch_proc = subprocess.run(
            ["gh", "run", "watch", run_id, "--exit-status"],
            check=False,
        )
        if watch_proc.returncode != 0 and args.download_artifacts_on_failure:
            artifact_root = Path(args.artifact_output_root)
            collection_root = _collect_failure_artifacts(
                run_id=run_id,
                output_root=artifact_root,
                artifact_names=args.artifact_name,
                collect_failed_logs=args.collect_failed_logs,
            )
            print(
                f"ci_watch: run {run_id} failed; collected artifacts under {collection_root}",
                file=sys.stderr,
            )
        return watch_proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
