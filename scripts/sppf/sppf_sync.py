#!/usr/bin/env python3
from __future__ import annotations

import sys

from gabion.tooling.sppf import sync_core


# Re-export helpers consumed by governance audits.
CommitInfo = sync_core.CommitInfo
IssueLifecycle = sync_core.IssueLifecycle
_default_range = sync_core._default_range
_is_sppf_relevant_push = sync_core._is_sppf_relevant_push
_collect_commits = sync_core._collect_commits
_issue_ids_from_commits = sync_core._issue_ids_from_commits
_build_issue_link_facet = sync_core._build_issue_link_facet
_fetch_issue = sync_core._fetch_issue
_validate_issue_lifecycle = sync_core._validate_issue_lifecycle
_run_validate_mode = sync_core._run_validate_mode
_deadline_scope = sync_core._deadline_scope
main = sync_core.main


if __name__ == "__main__":
    raise SystemExit(
        "Removed direct script entrypoint: scripts/sppf/sppf_sync.py. "
        "Use `gabion sppf sync`. "
        "See docs/user_workflows.md#user_workflows and "
        "docs/normative_clause_index.md#clause-command-maturity-parity."
    )
