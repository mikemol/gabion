from __future__ import annotations

import subprocess

import pytest

from scripts import sppf_sync


# gabion:evidence E:call_footprint::tests/test_sppf_sync.py::test_is_sppf_relevant_push_detects_relevant_paths::sppf_sync.py::scripts.sppf_sync._is_sppf_relevant_push
def test_is_sppf_relevant_push_detects_relevant_paths() -> None:
    assert (
        sppf_sync._is_sppf_relevant_push(
            "origin/stage..HEAD",
            run_git_fn=lambda _args: "src/gabion/cli.py\nREADME.md",
        )
        is True
    )


# gabion:evidence E:call_footprint::tests/test_sppf_sync.py::test_validate_issue_lifecycle_reports_missing_labels_and_state::sppf_sync.py::scripts.sppf_sync._validate_issue_lifecycle
def test_validate_issue_lifecycle_reports_missing_labels_and_state() -> None:
    violations = sppf_sync._validate_issue_lifecycle(
        ["123"],
        required_labels=["done-on-stage", "status/pending-release"],
        expected_state="open",
        fetch_issue_fn=lambda issue_id: sppf_sync.IssueLifecycle(
            issue_id=issue_id,
            state="closed",
            labels=("done-on-stage",),
        ),
    )
    assert len(violations) == 2
    assert "expected state 'open'" in violations[0]
    assert "missing required label(s): status/pending-release" in violations[1]
    assert "python -m scripts.sppf_sync --range <rev-range> --label status/pending-release" in violations[1]


# gabion:evidence E:call_footprint::tests/test_sppf_sync.py::test_issue_ids_from_commits_normalizes_known_gh_0000_placeholders::sppf_sync.py::scripts.sppf_sync._issue_ids_from_commits
def test_issue_ids_from_commits_normalizes_known_gh_0000_placeholders() -> None:
    commits = [
        sppf_sync.CommitInfo(
            sha="683da24bd121524dc48c218d9771dfbdf181d6f0",
            subject="SPPF: GH-0000",
            body="",
        ),
        sppf_sync.CommitInfo(
            sha="61c5d617e7b1d4e734a476adf69bc92c19f35e0f",
            subject="GH-0000",
            body="Refs #42",
        ),
    ]
    assert sppf_sync._issue_ids_from_commits(commits) == {"42", "214"}


# gabion:evidence E:call_footprint::tests/test_sppf_sync.py::test_build_issue_link_facet_normalizes_known_gh_0000_placeholders::sppf_sync.py::scripts.sppf_sync._build_issue_link_facet
def test_build_issue_link_facet_normalizes_known_gh_0000_placeholders() -> None:
    commits = [
        sppf_sync.CommitInfo(
            sha="683da24bd121524dc48c218d9771dfbdf181d6f0",
            subject="SPPF: GH-0000",
            body="",
        ),
        sppf_sync.CommitInfo(
            sha="61c5d617e7b1d4e734a476adf69bc92c19f35e0f",
            subject="GH-0000",
            body="",
        ),
    ]
    facet = sppf_sync._build_issue_link_facet(commits)
    assert facet.issue_ids == ("214",)
    assert facet.checklist_impact == (("214", 2),)


# gabion:evidence E:call_footprint::tests/test_sppf_sync.py::test_main_validate_mode_fails_with_clear_remediation::sppf_sync.py::scripts.sppf_sync.main
def test_main_validate_mode_fails_with_clear_remediation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _run_validate_mode(args):
        return sppf_sync._run_validate_mode(
            args,
            default_range_fn=lambda: "origin/stage..HEAD",
            is_sppf_relevant_push_fn=lambda _rev_range: True,
            collect_commits_fn=lambda _rev_range: [
                sppf_sync.CommitInfo(sha="abc", subject="SPPF: GH-123", body="")
            ],
            issue_ids_from_commits_fn=lambda _commits: {"123"},
            validate_issue_lifecycle_fn=lambda issue_ids, *, required_labels, expected_state: sppf_sync._validate_issue_lifecycle(
                issue_ids,
                required_labels=required_labels,
                expected_state=expected_state,
                fetch_issue_fn=lambda issue_id: sppf_sync.IssueLifecycle(
                    issue_id=issue_id,
                    state="open",
                    labels=(),
                ),
            ),
        )

    exit_code = sppf_sync.main(
        [
            "--validate",
            "--only-when-relevant",
            "--require-label",
            "done-on-stage",
            "--require-label",
            "status/pending-release",
        ],
        run_validate_mode_fn=_run_validate_mode,
    )
    captured = capsys.readouterr().out

    assert exit_code == 1
    assert "SPPF issue lifecycle validation failed" in captured
    assert "GH-123: missing required label(s): done-on-stage, status/pending-release" in captured


# gabion:evidence E:call_footprint::tests/test_sppf_sync.py::test_fetch_issue_reports_missing_issue::sppf_sync.py::scripts.sppf_sync._fetch_issue
def test_fetch_issue_reports_missing_issue() -> None:
    def _raise(*_args, **_kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=["gh"])

    with pytest.raises(RuntimeError) as exc:
        sppf_sync._fetch_issue("999", check_output_fn=_raise)
    assert "GH-999: issue lookup failed" in str(exc.value)
