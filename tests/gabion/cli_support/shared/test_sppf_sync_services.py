from __future__ import annotations

from gabion.cli_support.shared import sppf_sync


def test_extract_sppf_issue_ids_normalizes_patterns() -> None:
    text = "GH-0011 Refs #12 closes #0013 and fixes #000"
    assert sppf_sync.extract_sppf_issue_ids(text) == {"11", "12", "13", "0"}


def test_normalize_sppf_issue_ids_for_commit_rewrites_placeholder_issue() -> None:
    commit = sppf_sync.SppfSyncCommitInfo(
        sha="683da24bd121524dc48c218d9771dfbdf181d6f0",
        subject="GH-000",
        body="",
    )
    assert sppf_sync.normalize_sppf_issue_ids_for_commit(commit, {"0", "99"}) == {
        "214",
        "99",
    }


def test_issue_ids_from_sppf_commits_applies_extraction_and_normalization() -> None:
    commits = [
        sppf_sync.SppfSyncCommitInfo(
            sha="61c5d617e7b1d4e734a476adf69bc92c19f35e0f",
            subject="Refs #000",
            body="GH-12",
        ),
        sppf_sync.SppfSyncCommitInfo(sha="x", subject="GH-9", body=""),
    ]
    assert sppf_sync.issue_ids_from_sppf_commits(
        commits,
        check_deadline_fn=lambda: None,
    ) == {"214", "12", "9"}
