#!/usr/bin/env python3
"""Sync SPPF checklist-linked issues from commit messages.

This is a local helper to avoid CI write scopes. It parses commit messages for
references like "GH-17" or "Closes #17" and comments/closes/labels issues
via the GitHub CLI.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass

from deadline_runtime import deadline_scope_from_lsp_env
from gabion.analysis.timeout_context import check_deadline


GH_REF_RE = re.compile(r"\bGH-(\d+)\b", re.IGNORECASE)
KEYWORD_REF_RE = re.compile(r"\b(?:Closes|Fixes|Resolves|Refs)\s+#(\d+)\b", re.IGNORECASE)
_DEFAULT_TIMEOUT_TICKS = 120_000
_DEFAULT_TIMEOUT_TICK_NS = 1_000_000


def _deadline_scope():
    return deadline_scope_from_lsp_env(
        default_ticks=_DEFAULT_TIMEOUT_TICKS,
        default_tick_ns=_DEFAULT_TIMEOUT_TICK_NS,
    )


@dataclass(frozen=True)
class CommitInfo:
    sha: str
    subject: str
    body: str


def _run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _default_range() -> str:
    try:
        _run_git(["rev-parse", "origin/stage"])
        return "origin/stage..HEAD"
    except Exception:
        return "HEAD~20..HEAD"


def _collect_commits(rev_range: str) -> list[CommitInfo]:
    try:
        raw = subprocess.check_output(
            [
                "git",
                "log",
                "--format=%H%x1f%s%x1f%B%x1e",
                rev_range,
            ],
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"git log failed for range {rev_range}: {exc}")

    commits: list[CommitInfo] = []
    for record in raw.split("\x1e"):
        check_deadline()
        if not record.strip():
            continue
        parts = record.split("\x1f")
        if len(parts) < 3:
            continue
        sha, subject, body = parts[0].strip(), parts[1].strip(), parts[2].strip()
        commits.append(CommitInfo(sha=sha, subject=subject, body=body))
    return commits


def _extract_issue_ids(text: str) -> set[str]:
    issues = set(match.group(1) for match in GH_REF_RE.finditer(text))
    issues.update(match.group(1) for match in KEYWORD_REF_RE.finditer(text))
    return issues


def _issue_ids_from_commits(commits: list[CommitInfo]) -> set[str]:
    issues: set[str] = set()
    for commit in commits:
        check_deadline()
        issues.update(_extract_issue_ids(commit.subject))
        issues.update(_extract_issue_ids(commit.body))
    return issues


def _build_comment(rev_range: str, commits: list[CommitInfo]) -> str:
    lines = [f"SPPF sync from `{rev_range}`:"]
    for commit in commits:
        check_deadline()
        lines.append(f"- {commit.sha[:8]} {commit.subject}")
    return "\n".join(lines)


def _run_gh(args: list[str], dry_run: bool) -> None:
    if dry_run:
        print("DRY RUN:", " ".join(["gh", *args]))
        return
    subprocess.run(["gh", *args], check=True)


def main() -> int:
    with _deadline_scope():
        parser = argparse.ArgumentParser(description="Sync SPPF-linked issues from commit messages.")
        parser.add_argument(
            "--range",
            dest="rev_range",
            default=None,
            help="Git revision range (default: origin/stage..HEAD if available).")
        parser.add_argument(
            "--comment",
            action="store_true",
            help="Comment on each referenced issue with commit summary.")
        parser.add_argument(
            "--close",
            action="store_true",
            help="Close each referenced issue with a summary comment.")
        parser.add_argument(
            "--label",
            default=None,
            help="Apply a label to each referenced issue.")
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Print gh commands without executing.")
        args = parser.parse_args()

        rev_range = args.rev_range or _default_range()
        commits = _collect_commits(rev_range)
        if not commits:
            print("No commits in range; nothing to sync.")
            return 0

        issue_ids = sorted(_issue_ids_from_commits(commits))
        if not issue_ids:
            print("No issue references found in commit messages.")
            return 0

        comment = _build_comment(rev_range, commits)
        for issue_id in issue_ids:
            check_deadline()
            if args.close:
                _run_gh(["issue", "close", issue_id, "-c", comment], args.dry_run)
            elif args.comment:
                _run_gh(["issue", "comment", issue_id, "-b", comment], args.dry_run)
            if args.label:
                _run_gh(["issue", "edit", issue_id, "--add-label", args.label], args.dry_run)
        return 0


if __name__ == "__main__":
    sys.exit(main())
