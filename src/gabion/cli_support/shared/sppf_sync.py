from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import re
import subprocess

import typer

from gabion.order_contract import sort_once


_SPPF_GH_REF_RE = re.compile(r"\bGH-(\d+)\b", re.IGNORECASE)
_SPPF_KEYWORD_REF_RE = re.compile(
    r"\b(?:Closes|Fixes|Resolves|Refs)\s+#(\d+)\b", re.IGNORECASE
)
_SPPF_PLACEHOLDER_ISSUE_BY_COMMIT: dict[str, str] = {
    "683da24bd121524dc48c218d9771dfbdf181d6f0": "214",
    "61c5d617e7b1d4e734a476adf69bc92c19f35e0f": "214",
}


@dataclass(frozen=True)
class SppfSyncCommitInfo:
    sha: str
    subject: str
    body: str


def run_sppf_git(
    args: list[str],
    *,
    check_output_fn: Callable[..., str] = subprocess.check_output,
) -> str:
    return check_output_fn(["git", *args], text=True).strip()


def default_sppf_rev_range(
    *,
    run_sppf_git_fn: Callable[[list[str]], str] = run_sppf_git,
) -> str:
    try:
        run_sppf_git_fn(["rev-parse", "origin/stage"])
        return "origin/stage..HEAD"
    except Exception:
        return "HEAD~20..HEAD"


def collect_sppf_commits(
    rev_range: str,
    *,
    check_deadline_fn: Callable[[], None],
    check_output_fn: Callable[..., str] = subprocess.check_output,
) -> list[SppfSyncCommitInfo]:
    try:
        raw = check_output_fn(
            [
                "git",
                "log",
                "--format=%H%x1f%s%x1f%B%x1e",
                rev_range,
            ],
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise typer.BadParameter(f"git log failed for range {rev_range}: {exc}") from exc

    commits: list[SppfSyncCommitInfo] = []
    for record in raw.split("\x1e"):
        check_deadline_fn()
        if not record.strip():
            continue
        parts = record.split("\x1f")
        if len(parts) < 3:
            continue
        sha, subject, body = parts[0].strip(), parts[1].strip(), parts[2].strip()
        commits.append(SppfSyncCommitInfo(sha=sha, subject=subject, body=body))
    return commits


def extract_sppf_issue_ids(text: str) -> set[str]:
    def _canonical(issue_id: str) -> str:
        normalized = issue_id.lstrip("0")
        return normalized or "0"

    issues = set(_canonical(match.group(1)) for match in _SPPF_GH_REF_RE.finditer(text))
    issues.update(_canonical(match.group(1)) for match in _SPPF_KEYWORD_REF_RE.finditer(text))
    return issues


def normalize_sppf_issue_ids_for_commit(
    commit: SppfSyncCommitInfo,
    issue_ids: set[str],
) -> set[str]:
    normalized = set(issue_ids)
    if "0" not in normalized:
        return normalized
    normalized.discard("0")
    replacement = _SPPF_PLACEHOLDER_ISSUE_BY_COMMIT.get(commit.sha)
    if replacement is not None:
        normalized.add(replacement)
    else:
        normalized.add("0")
    return normalized


def issue_ids_from_sppf_commits(
    commits: list[SppfSyncCommitInfo],
    *,
    check_deadline_fn: Callable[[], None],
) -> set[str]:
    issues: set[str] = set()
    for commit in commits:
        check_deadline_fn()
        commit_issue_ids = extract_sppf_issue_ids(commit.subject)
        commit_issue_ids.update(extract_sppf_issue_ids(commit.body))
        issues.update(normalize_sppf_issue_ids_for_commit(commit, commit_issue_ids))
    return issues


def build_sppf_comment(rev_range: str, commits: list[SppfSyncCommitInfo], *, check_deadline_fn: Callable[[], None]) -> str:
    lines = [f"SPPF sync from `{rev_range}`:"]
    for commit in commits:
        check_deadline_fn()
        lines.append(f"- {commit.sha[:8]} {commit.subject}")
    return "\n".join(lines)


def run_sppf_gh(
    args: list[str],
    *,
    dry_run: bool,
    echo_fn: Callable[[str], None] = typer.echo,
    run_fn: Callable[..., object] = subprocess.run,
) -> None:
    if dry_run:
        echo_fn("DRY RUN: " + " ".join(["gh", *args]))
        return
    run_fn(["gh", *args], check=True)


def run_sppf_sync(
    *,
    rev_range: str | None,
    comment: bool,
    close: bool,
    label: str | None,
    dry_run: bool,
    check_deadline_fn: Callable[[], None],
    default_rev_range_fn: Callable[[], str] = default_sppf_rev_range,
    collect_sppf_commits_fn: Callable[[str], list[SppfSyncCommitInfo]],
    run_sppf_gh_fn: Callable[[list[str]], None] | None = None,
    sort_once_fn: Callable[..., list[str]] = sort_once,
    echo_fn: Callable[[str], None] = typer.echo,
) -> int:
    resolved_range = rev_range or default_rev_range_fn()
    commits = collect_sppf_commits_fn(resolved_range)
    if not commits:
        echo_fn("No commits in range; nothing to sync.")
        return 0

    issue_ids = sort_once_fn(
        issue_ids_from_sppf_commits(commits, check_deadline_fn=check_deadline_fn),
        source="gabion.cli.sppf_sync.issue_ids",
    )
    if not issue_ids:
        echo_fn("No issue references found in commit messages.")
        return 0

    summary_comment = build_sppf_comment(
        resolved_range,
        commits,
        check_deadline_fn=check_deadline_fn,
    )
    gh_runner = run_sppf_gh_fn or (
        lambda args: run_sppf_gh(args, dry_run=dry_run, echo_fn=echo_fn)
    )
    for issue_id in issue_ids:
        check_deadline_fn()
        if close:
            gh_runner(["issue", "close", issue_id, "-c", summary_comment])
        elif comment:
            gh_runner(["issue", "comment", issue_id, "-b", summary_comment])
        if label:
            gh_runner(["issue", "edit", issue_id, "--add-label", label])
    return 0
