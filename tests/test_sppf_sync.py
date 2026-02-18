from __future__ import annotations

import subprocess

import pytest

from scripts import sppf_sync


def test_is_sppf_relevant_push_detects_relevant_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sppf_sync, "_run_git", lambda args: "src/gabion/cli.py\nREADME.md")
    assert sppf_sync._is_sppf_relevant_push("origin/stage..HEAD") is True


def test_validate_issue_lifecycle_reports_missing_labels_and_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sppf_sync,
        "_fetch_issue",
        lambda issue_id: sppf_sync.IssueLifecycle(issue_id=issue_id, state="closed", labels=("done-on-stage",)),
    )
    violations = sppf_sync._validate_issue_lifecycle(
        ["123"],
        required_labels=["done-on-stage", "status/pending-release"],
        expected_state="open",
    )
    assert len(violations) == 2
    assert "expected state 'open'" in violations[0]
    assert "missing required label(s): status/pending-release" in violations[1]
    assert "scripts/sppf_sync.py --range <rev-range> --label status/pending-release" in violations[1]


def test_main_validate_mode_fails_with_clear_remediation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sppf_sync, "_default_range", lambda: "origin/stage..HEAD")
    monkeypatch.setattr(sppf_sync, "_is_sppf_relevant_push", lambda rev_range: True)
    monkeypatch.setattr(
        sppf_sync,
        "_collect_commits",
        lambda rev_range: [sppf_sync.CommitInfo(sha="abc", subject="SPPF: GH-123", body="")],
    )
    monkeypatch.setattr(sppf_sync, "_issue_ids_from_commits", lambda commits: {"123"})
    monkeypatch.setattr(
        sppf_sync,
        "_fetch_issue",
        lambda issue_id: sppf_sync.IssueLifecycle(issue_id=issue_id, state="open", labels=()),
    )

    monkeypatch.setattr(
        sppf_sync.sys,
        "argv",
        [
            "sppf_sync.py",
            "--validate",
            "--only-when-relevant",
            "--require-label",
            "done-on-stage",
            "--require-label",
            "status/pending-release",
        ],
    )
    exit_code = sppf_sync.main()
    captured = capsys.readouterr().out

    assert exit_code == 1
    assert "SPPF issue lifecycle validation failed" in captured
    assert "GH-123: missing required label(s): done-on-stage, status/pending-release" in captured


def test_fetch_issue_reports_missing_issue(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*_args, **_kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=["gh"])

    monkeypatch.setattr(subprocess, "check_output", _raise)
    with pytest.raises(RuntimeError) as exc:
        sppf_sync._fetch_issue("999")
    assert "GH-999: issue lookup failed" in str(exc.value)
