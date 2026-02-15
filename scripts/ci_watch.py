from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import subprocess
import sys
from typing import Any

from gabion.analysis.aspf import Forest
from gabion.analysis.timeout_context import (
    Deadline,
    check_deadline,
    deadline_clock_scope,
    deadline_scope,
    forest_scope,
)
from gabion.deadline_clock import GasMeter
from gabion.lsp_client import _env_timeout_ticks, _has_env_timeout

_DEFAULT_TIMEOUT_TICKS = 120_000
_DEFAULT_TIMEOUT_TICK_NS = 1_000_000


@contextmanager
def _deadline_scope():
    if _has_env_timeout():
        ticks, tick_ns = _env_timeout_ticks()
    else:
        ticks, tick_ns = _DEFAULT_TIMEOUT_TICKS, _DEFAULT_TIMEOUT_TICK_NS
    with forest_scope(Forest()):
        with deadline_scope(Deadline.from_timeout_ticks(ticks, tick_ns)):
            with deadline_clock_scope(GasMeter(limit=int(ticks))):
                yield


def _run(*cmd: str) -> str:
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return proc.stdout.strip()


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
        subprocess.run(
            ["gh", "run", "watch", run_id, "--exit-status"],
            check=True,
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
