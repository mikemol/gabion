from __future__ import annotations

import subprocess

import pytest
import typer
from typer.testing import CliRunner

from gabion import cli
from scripts import sppf_sync


# gabion:evidence E:call_footprint::tests/test_sppf_sync_cli.py::test_sppf_sync_cli_maps_flags_to_runner::cli.py::gabion.cli::cli.py::gabion.cli.app
def test_sppf_sync_cli_maps_flags_to_runner() -> None:
    captured: dict[str, object] = {}

    def _fake_run_sppf_sync(**kwargs):
        captured.update(kwargs)
        return 0

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "sppf-sync",
            "--range",
            "HEAD~3..HEAD",
            "--comment",
            "--close",
            "--label",
            "done-on-stage",
            "--dry-run",
        ],
        obj={"run_sppf_sync": _fake_run_sppf_sync},
    )

    assert result.exit_code == 0
    assert captured == {
        "rev_range": "HEAD~3..HEAD",
        "comment": True,
        "close": True,
        "label": "done-on-stage",
        "dry_run": True,
    }


# gabion:evidence E:call_footprint::tests/test_sppf_sync_cli.py::test_sppf_sync_dry_run_routes_comment_and_label::cli.py::gabion.cli._run_sppf_sync
def test_sppf_sync_dry_run_routes_comment_and_label() -> None:
    commits = [
        cli.SppfSyncCommitInfo(
            sha="abcdef0123456789",
            subject="Link GH-22",
            body="Refs #22",
        )
    ]
    calls: list[list[str]] = []

    def _fake_collect(_rev_range: str) -> list[cli.SppfSyncCommitInfo]:
        return commits

    def _fake_gh(args: list[str]) -> None:
        calls.append(list(args))

    exit_code = cli._run_sppf_sync(
        rev_range="HEAD~1..HEAD",
        comment=True,
        close=False,
        label="done-on-stage",
        dry_run=True,
        collect_sppf_commits_fn=_fake_collect,
        run_sppf_gh_fn=_fake_gh,
    )

    assert exit_code == 0
    assert calls == [
        [
            "issue",
            "comment",
            "22",
            "-b",
            "SPPF sync from `HEAD~1..HEAD`:\n- abcdef01 Link GH-22",
        ],
        ["issue", "edit", "22", "--add-label", "done-on-stage"],
    ]


# gabion:evidence E:call_footprint::tests/test_sppf_sync_cli.py::test_sppf_issue_id_extraction_matches_previous_patterns::cli.py::gabion.cli._extract_sppf_issue_ids
def test_sppf_issue_id_extraction_matches_previous_patterns() -> None:
    text = "GH-11 closes #12; resolves #13; refs #12 and Fixes #14"
    assert cli._extract_sppf_issue_ids(text) == {"11", "12", "13", "14"}


# gabion:evidence E:call_footprint::tests/test_sppf_sync_cli.py::test_script_sppf_sync_delegates_to_cli_compat::sppf_sync.py::scripts.sppf_sync.main
def test_script_sppf_sync_delegates_to_cli_compat() -> None:
    captured: list[str] = []

    def _fake_run(argv: list[str]) -> int:
        captured.extend(argv)
        return 0

    assert sppf_sync.main(
        argv=["--dry-run", "--comment"],
        run_sppf_sync_compat_fn=_fake_run,
    ) == 0
    assert captured == ["--dry-run", "--comment"]


# gabion:evidence E:call_footprint::tests/test_sppf_sync_cli.py::test_sppf_sync_git_helpers_and_commit_collection_edges::cli.py::gabion.cli._default_sppf_rev_range::cli.py::gabion.cli._collect_sppf_commits::cli.py::gabion.cli._run_sppf_git
def test_sppf_sync_git_helpers_and_commit_collection_edges() -> None:
    assert cli._run_sppf_git(
        ["status"],
        check_output_fn=lambda *_a, **_k: "ok\n",
    ) == "ok"
    assert cli._default_sppf_rev_range(run_sppf_git_fn=lambda _args: "ok") == "origin/stage..HEAD"

    def _raise_git(_args):
        raise RuntimeError("missing remote")

    assert cli._default_sppf_rev_range(run_sppf_git_fn=_raise_git) == "HEAD~20..HEAD"

    raw = (
        "abc123\x1fsubject one\x1fbody one\x1e"
        "malformed\x1fonly-two\x1e"
        "def456\x1fsubject two\x1fbody two\x1e"
    )
    commits = cli._collect_sppf_commits(
        "HEAD~2..HEAD",
        check_output_fn=lambda *_a, **_k: raw,
    )
    assert [commit.sha for commit in commits] == ["abc123", "def456"]

    def _raise_called(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, ["git", "log"])

    with pytest.raises(typer.BadParameter):
        cli._collect_sppf_commits("bad-range", check_output_fn=_raise_called)


# gabion:evidence E:call_footprint::tests/test_sppf_sync_cli.py::test_sppf_sync_gh_runner_and_empty_sync_paths::cli.py::gabion.cli._run_sppf_gh::cli.py::gabion.cli._run_sppf_sync
def test_sppf_sync_gh_runner_and_empty_sync_paths(
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[list[str]] = []

    cli._run_sppf_gh(
        ["issue", "comment", "1"],
        dry_run=False,
        run_fn=lambda args, check: calls.append(list(args)),
    )
    assert calls == [["gh", "issue", "comment", "1"]]

    cli._run_sppf_gh(["issue", "close", "2"], dry_run=True)
    assert "DRY RUN: gh issue close 2" in capsys.readouterr().out

    assert cli._run_sppf_sync(
        rev_range="HEAD~1..HEAD",
        comment=True,
        close=False,
        label=None,
        dry_run=True,
        collect_sppf_commits_fn=lambda _range: [],
    ) == 0

    commits = [cli.SppfSyncCommitInfo(sha="abc", subject="no refs", body="none")]
    assert cli._run_sppf_sync(
        rev_range="HEAD~1..HEAD",
        comment=True,
        close=False,
        label=None,
        dry_run=True,
        collect_sppf_commits_fn=lambda _range: commits,
    ) == 0

    called: list[list[str]] = []
    assert cli._run_sppf_sync(
        rev_range="HEAD~1..HEAD",
        comment=False,
        close=True,
        label=None,
        dry_run=True,
        collect_sppf_commits_fn=lambda _range: [
            cli.SppfSyncCommitInfo(sha="abc12345", subject="GH-9", body="refs #9")
        ],
        run_sppf_gh_fn=lambda args: called.append(list(args)),
    ) == 0
    assert called and called[0][:3] == ["issue", "close", "9"]


# gabion:evidence E:call_footprint::tests/test_sppf_sync_cli.py::test_run_sppf_sync_compat_parses_and_forwards_flags::cli.py::gabion.cli.run_sppf_sync_compat
def test_run_sppf_sync_compat_parses_and_forwards_flags() -> None:
    captured: dict[str, object] = {}

    def _fake_run_sppf_sync(**kwargs):
        captured.update(kwargs)
        return 7

    exit_code = cli.run_sppf_sync_compat(
        [
            "--range",
            "HEAD~3..HEAD",
            "--comment",
            "--close",
            "--label",
            "done-on-stage",
            "--dry-run",
        ],
        run_sppf_sync_fn=_fake_run_sppf_sync,
    )
    assert exit_code == 7
    assert captured == {
        "rev_range": "HEAD~3..HEAD",
        "comment": True,
        "close": True,
        "label": "done-on-stage",
        "dry_run": True,
    }
