from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any


def _run(*cmd: str) -> str:
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return proc.stdout.strip()


def _latest_run_id(branch: str) -> str:
    payload = _run(
        "gh",
        "run",
        "list",
        "--branch",
        branch,
        "--limit",
        "1",
        "--json",
        "databaseId,status,conclusion,headSha,displayTitle",
    )
    data: list[dict[str, Any]] = json.loads(payload)
    if not data:
        raise SystemExit(f"No runs found for branch {branch}")
    return str(data[0]["databaseId"])


def main() -> int:
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
    args = parser.parse_args()

    run_id = args.run_id or _latest_run_id(args.branch)
    subprocess.run(
        ["gh", "run", "watch", run_id, "--exit-status"],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
