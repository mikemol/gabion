from __future__ import annotations

import sys
import types
from pathlib import Path


def _load_audit_tools():
    from scripts import audit_tools

    return audit_tools


def _install_fake_sppf_sync(monkeypatch, *, rev_range: str, commits: list[str], issue_ids: set[str]) -> None:
    module = types.SimpleNamespace(
        _default_range=lambda: rev_range,
        _collect_commits=lambda _range: commits,
        _issue_ids_from_commits=lambda _commits: issue_ids,
    )
    monkeypatch.setitem(sys.modules, "scripts.sppf_sync", module)


def test_sppf_gh_refs_required_mode_reports_violation(monkeypatch, tmp_path: Path) -> None:
    audit_tools = _load_audit_tools()
    _install_fake_sppf_sync(
        monkeypatch,
        rev_range="HEAD~1..HEAD",
        commits=["abc123"],
        issue_ids=set(),
    )
    monkeypatch.setattr(audit_tools, "_git_diff_paths", lambda _rev: ["src/gabion/cli.py"])

    violations, warnings = audit_tools._sppf_sync_check(tmp_path, mode="required")

    assert len(violations) == 1
    assert "no GH references found" in violations[0]
    assert warnings == []


def test_sppf_gh_refs_advisory_mode_reports_warning(monkeypatch, tmp_path: Path) -> None:
    audit_tools = _load_audit_tools()
    _install_fake_sppf_sync(
        monkeypatch,
        rev_range="HEAD~1..HEAD",
        commits=["abc123"],
        issue_ids=set(),
    )
    monkeypatch.setattr(audit_tools, "_git_diff_paths", lambda _rev: ["in/in-30.md"])

    violations, warnings = audit_tools._sppf_sync_check(tmp_path, mode="advisory")

    assert violations == []
    assert len(warnings) == 1
    assert "no GH references found" in warnings[0]


def test_sppf_gh_refs_required_mode_ignores_irrelevant_paths(monkeypatch, tmp_path: Path) -> None:
    audit_tools = _load_audit_tools()
    _install_fake_sppf_sync(
        monkeypatch,
        rev_range="HEAD~1..HEAD",
        commits=["abc123"],
        issue_ids=set(),
    )
    monkeypatch.setattr(audit_tools, "_git_diff_paths", lambda _rev: ["docs/notes.md"])

    violations, warnings = audit_tools._sppf_sync_check(tmp_path, mode="required")

    assert violations == []
    assert warnings == []


def test_sppf_gh_refs_required_mode_passes_when_refs_present(monkeypatch, tmp_path: Path) -> None:
    audit_tools = _load_audit_tools()
    _install_fake_sppf_sync(
        monkeypatch,
        rev_range="HEAD~1..HEAD",
        commits=["abc123"],
        issue_ids={"GH-123"},
    )
    monkeypatch.setattr(audit_tools, "_git_diff_paths", lambda _rev: ["docs/sppf_checklist.md"])

    violations, warnings = audit_tools._sppf_sync_check(tmp_path, mode="required")

    assert violations == []
    assert warnings == []
