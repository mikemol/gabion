from __future__ import annotations

import sys

from typer.testing import CliRunner

from gabion import cli
from scripts import sppf_sync


def test_sppf_sync_cli_maps_flags_to_runner(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run_sppf_sync(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "_run_sppf_sync", _fake_run_sppf_sync)

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
    )

    assert result.exit_code == 0
    assert captured == {
        "rev_range": "HEAD~3..HEAD",
        "comment": True,
        "close": True,
        "label": "done-on-stage",
        "dry_run": True,
    }


def test_sppf_sync_dry_run_routes_comment_and_label(monkeypatch) -> None:
    commits = [
        cli.SppfSyncCommitInfo(
            sha="abcdef0123456789",
            subject="Link GH-22",
            body="Refs #22",
        )
    ]
    calls: list[tuple[list[str], bool]] = []

    def _fake_collect(_rev_range: str):
        return commits

    def _fake_gh(args: list[str], *, dry_run: bool):
        calls.append((args, dry_run))

    monkeypatch.setattr(cli, "_collect_sppf_commits", _fake_collect)
    monkeypatch.setattr(cli, "_run_sppf_gh", _fake_gh)

    exit_code = cli._run_sppf_sync(
        rev_range="HEAD~1..HEAD",
        comment=True,
        close=False,
        label="done-on-stage",
        dry_run=True,
    )

    assert exit_code == 0
    assert calls == [
        (
            [
                "issue",
                "comment",
                "22",
                "-b",
                "SPPF sync from `HEAD~1..HEAD`:\n- abcdef01 Link GH-22",
            ],
            True,
        ),
        (["issue", "edit", "22", "--add-label", "done-on-stage"], True),
    ]


def test_sppf_issue_id_extraction_matches_previous_patterns() -> None:
    text = "GH-11 closes #12; resolves #13; refs #12 and Fixes #14"
    assert cli._extract_sppf_issue_ids(text) == {"11", "12", "13", "14"}


def test_script_sppf_sync_delegates_to_cli_compat(monkeypatch) -> None:
    captured: list[str] = []

    def _fake_run(argv: list[str]):
        captured.extend(argv)
        return 0

    monkeypatch.setattr(sppf_sync, "run_sppf_sync_compat", _fake_run)
    monkeypatch.setattr(sys, "argv", ["scripts/sppf_sync.py", "--dry-run", "--comment"])

    assert sppf_sync.main() == 0
    assert captured == ["--dry-run", "--comment"]
