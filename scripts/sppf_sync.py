#!/usr/bin/env python3
"""SPPF sync helpers plus compatibility shim for `gabion sppf-sync`."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass

try:  # pragma: no cover - import form depends on invocation mode
    from scripts.deadline_runtime import DeadlineBudget, deadline_scope_from_lsp_env
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from deadline_runtime import DeadlineBudget, deadline_scope_from_lsp_env

from gabion.analysis.timeout_context import check_deadline
from gabion.cli import run_sppf_sync_compat
from gabion.execution_plan import IssueLinkFacet
from gabion.order_contract import ordered_or_sorted

GH_REF_RE = re.compile(r"\bGH-(\d+)\b", re.IGNORECASE)
KEYWORD_REF_RE = re.compile(r"\b(?:Closes|Fixes|Resolves|Refs)\s+#(\d+)\b", re.IGNORECASE)
SPPF_RELEVANT_PATHS = ("src/", "in/", "docs/sppf_checklist.md")
_DEFAULT_TIMEOUT_TICKS = 120_000
_DEFAULT_TIMEOUT_TICK_NS = 1_000_000
_DEFAULT_TIMEOUT_BUDGET = DeadlineBudget(
    ticks=_DEFAULT_TIMEOUT_TICKS,
    tick_ns=_DEFAULT_TIMEOUT_TICK_NS,
)


def _deadline_scope():
    return deadline_scope_from_lsp_env(
        default_budget=_DEFAULT_TIMEOUT_BUDGET,
    )


@dataclass(frozen=True)
class CommitInfo:
    sha: str
    subject: str
    body: str


@dataclass(frozen=True)
class IssueLifecycle:
    issue_id: str
    state: str
    labels: tuple[str, ...]


def _run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _default_range() -> str:
    try:
        _run_git(["rev-parse", "origin/stage"])
        return "origin/stage..HEAD"
    except Exception:
        return "HEAD~20..HEAD"


def _is_sppf_relevant_push(rev_range: str) -> bool:
    try:
        changed = _run_git(["diff", "--name-only", rev_range])
    except Exception:
        return True
    for path in changed.splitlines():
        normalized = path.strip()
        if not normalized:
            continue
        if normalized == "docs/sppf_checklist.md":
            return True
        if any(normalized.startswith(prefix) for prefix in SPPF_RELEVANT_PATHS[:2]):
            return True
    return False


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


def _build_issue_link_facet(commits: list[CommitInfo]) -> IssueLinkFacet:
    issue_counts: dict[str, int] = {}
    for commit in commits:
        check_deadline()
        refs = _extract_issue_ids(f"{commit.subject}\n{commit.body}")
        for issue_id in refs:
            check_deadline()
            issue_counts[issue_id] = issue_counts.get(issue_id, 0) + 1
    issue_ids = tuple(
        ordered_or_sorted(
            issue_counts,
            source="scripts.sppf_sync.issue_link.issue_ids",
            key=str,
        )
    )
    checklist_impact = tuple((issue_id, issue_counts[issue_id]) for issue_id in issue_ids)
    return IssueLinkFacet(issue_ids=issue_ids, checklist_impact=checklist_impact)


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


def _fetch_issue(issue_id: str) -> IssueLifecycle:
    try:
        raw = subprocess.check_output(
            ["gh", "issue", "view", issue_id, "--json", "state,labels"],
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"GH-{issue_id}: issue lookup failed. Ensure the issue exists and GH auth is configured (gh auth status)."
        ) from exc
    payload = json.loads(raw)
    labels_data = payload.get("labels") if isinstance(payload, dict) else []
    labels: list[str] = []
    if isinstance(labels_data, list):
        for label in labels_data:
            if isinstance(label, dict) and isinstance(label.get("name"), str):
                labels.append(label["name"])
    state = str(payload.get("state", "")).lower() if isinstance(payload, dict) else ""
    return IssueLifecycle(issue_id=issue_id, state=state, labels=tuple(labels))


def _validate_issue_lifecycle(
    issue_ids: list[str],
    *,
    required_labels: list[str],
    expected_state: str,
) -> list[str]:
    violations: list[str] = []
    for issue_id in issue_ids:
        check_deadline()
        try:
            issue = _fetch_issue(issue_id)
        except RuntimeError as err:
            violations.append(str(err))
            continue

        if expected_state != "any" and issue.state != expected_state:
            violations.append(
                " ".join(
                    [
                        f"GH-{issue_id}: expected state '{expected_state}' but found '{issue.state}'.",
                        "Remediation: update lifecycle status before pushing",
                        f"(e.g. gh issue reopen {issue_id} or gh issue close {issue_id}).",
                    ]
                )
            )

        missing = [label for label in required_labels if label not in issue.labels]
        if missing:
            add_labels = " ".join(f"--label {label}" for label in missing)
            violations.append(
                " ".join(
                    [
                        f"GH-{issue_id}: missing required label(s): {', '.join(missing)}.",
                        "Remediation: run locally:",
                        f"scripts/sppf_sync.py --range <rev-range> {add_labels}",
                    ]
                )
            )
    return violations


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync SPPF-linked issues from commit messages.")
    parser.add_argument(
        "--range",
        dest="rev_range",
        default=None,
        help="Git revision range (default: origin/stage..HEAD if available).",
    )
    parser.add_argument("--comment", action="store_true", help="Comment on each referenced issue.")
    parser.add_argument("--close", action="store_true", help="Close each referenced issue.")
    parser.add_argument("--label", action="append", default=[], help="Apply label (repeatable).")
    parser.add_argument("--dry-run", action="store_true", help="Print gh commands without executing.")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Non-mutating verification mode for referenced issue lifecycle.",
    )
    parser.add_argument(
        "--only-when-relevant",
        action="store_true",
        help="Skip when the range does not touch SPPF-relevant paths.",
    )
    parser.add_argument(
        "--require-label",
        action="append",
        default=[],
        help="Required label in --validate mode (repeatable).",
    )
    parser.add_argument(
        "--require-state",
        choices=("open", "closed", "any"),
        default="open",
        help="Expected state in --validate mode.",
    )
    return parser.parse_args(argv)


def _run_validate_mode(args: argparse.Namespace) -> int:
    rev_range = args.rev_range or _default_range()
    if args.only_when_relevant and not _is_sppf_relevant_push(rev_range):
        print(f"Range {rev_range} does not touch SPPF-relevant paths; skipping SPPF sync/check.")
        return 0

    commits = _collect_commits(rev_range)
    if not commits:
        print("No commits in range; nothing to sync.")
        return 0

    issue_ids = list(
        ordered_or_sorted(
            _issue_ids_from_commits(commits),
            source="scripts.sppf_sync.issue_ids",
            key=str,
        )
    )
    if not issue_ids:
        print("No issue references found in commit messages.")
        return 0

    violations = _validate_issue_lifecycle(
        issue_ids,
        required_labels=list(args.require_label),
        expected_state=str(args.require_state),
    )
    if violations:
        print("SPPF issue lifecycle validation failed:")
        for item in violations:
            print(f"- {item}")
        return 1
    print(
        "SPPF issue lifecycle validation passed for "
        + ", ".join(f"GH-{issue_id}" for issue_id in issue_ids)
    )
    return 0


def main() -> int:
    with _deadline_scope():
        argv = sys.argv[1:]
        args = _parse_args(argv)

        if args.validate and (args.comment or args.close or args.label):
            raise SystemExit(
                "--validate cannot be combined with mutating options (--comment/--close/--label)."
            )

        # Preserve compatibility with the typed CLI command for standard sync operations.
        if args.validate or args.only_when_relevant or args.require_label or args.require_state != "open":
            return _run_validate_mode(args)

        return run_sppf_sync_compat(argv)


if __name__ == "__main__":
    sys.exit(main())
