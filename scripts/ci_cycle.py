from __future__ import annotations

import argparse
import subprocess
import sys


def _run(*cmd: str) -> None:
    subprocess.run(cmd, check=True)


def _working_tree_clean() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() == ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a no-op commit and optionally push/watch CI.",
    )
    parser.add_argument(
        "--message",
        default="chore: ci cycle",
        help="Commit message to use (default: chore: ci cycle).",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow committing with a dirty working tree.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push the no-op commit to stage.",
    )
    parser.add_argument(
        "--refspec",
        default="stage-merge:stage",
        help="Refspec to push (default: stage-merge:stage).",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch the stage CI run after push.",
    )
    parser.add_argument(
        "--branch",
        default="stage",
        help="Branch to watch (default: stage).",
    )
    parser.add_argument(
        "--workflow",
        default="ci",
        help="Workflow name or file to watch (default: ci).",
    )
    args = parser.parse_args()

    if not args.allow_dirty and not _working_tree_clean():
        print("Working tree is not clean; aborting no-op commit.")
        return 1

    _run("git", "commit", "--allow-empty", "-m", args.message)

    if args.push:
        _run("git", "push", "origin", args.refspec)

    if args.watch:
        _run(
            sys.executable,
            "scripts/ci_watch.py",
            "--branch",
            args.branch,
            "--workflow",
            args.workflow,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
