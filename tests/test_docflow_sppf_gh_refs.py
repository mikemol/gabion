from __future__ import annotations

from pathlib import Path


def _load_audit_tools():
    from gabion.tooling import governance_audit as audit_tools

    return audit_tools


class _FakeSppfSync:
    def __init__(self, *, rev_range: str, commits: list[str], issue_ids: set[str]) -> None:
        self._rev_range = rev_range
        self._commits = commits
        self._issue_ids = issue_ids

    def _default_range(self) -> str:
        return self._rev_range

    def _collect_commits(self, _range: str) -> list[str]:
        return self._commits

    def _issue_ids_from_commits(self, _commits: list[str]) -> set[str]:
        return self._issue_ids


# gabion:evidence E:call_footprint::tests/test_docflow_sppf_gh_refs.py::test_sppf_gh_refs_required_mode_reports_violation::governance_audit.py::gabion.tooling.governance_audit._sppf_sync_check::test_docflow_sppf_gh_refs.py::tests.test_docflow_sppf_gh_refs._load_audit_tools
def test_sppf_gh_refs_required_mode_reports_violation(tmp_path: Path) -> None:
    audit_tools = _load_audit_tools()
    sppf_sync = _FakeSppfSync(
        rev_range="HEAD~1..HEAD",
        commits=["abc123"],
        issue_ids=set(),
    )

    violations, warnings = audit_tools._sppf_sync_check(
        tmp_path,
        mode="required",
        load_sppf_sync_module_fn=lambda: sppf_sync,
        git_diff_paths_fn=lambda _rev: ["src/gabion/cli.py"],
    )

    assert len(violations) == 1
    assert "no GH references found" in violations[0]
    assert warnings == []


# gabion:evidence E:call_footprint::tests/test_docflow_sppf_gh_refs.py::test_sppf_gh_refs_advisory_mode_reports_warning::governance_audit.py::gabion.tooling.governance_audit._sppf_sync_check::test_docflow_sppf_gh_refs.py::tests.test_docflow_sppf_gh_refs._load_audit_tools
def test_sppf_gh_refs_advisory_mode_reports_warning(tmp_path: Path) -> None:
    audit_tools = _load_audit_tools()
    sppf_sync = _FakeSppfSync(
        rev_range="HEAD~1..HEAD",
        commits=["abc123"],
        issue_ids=set(),
    )

    violations, warnings = audit_tools._sppf_sync_check(
        tmp_path,
        mode="advisory",
        load_sppf_sync_module_fn=lambda: sppf_sync,
        git_diff_paths_fn=lambda _rev: ["in/in-30.md"],
    )

    assert violations == []
    assert len(warnings) == 1
    assert "no GH references found" in warnings[0]


# gabion:evidence E:call_footprint::tests/test_docflow_sppf_gh_refs.py::test_sppf_gh_refs_required_mode_ignores_irrelevant_paths::governance_audit.py::gabion.tooling.governance_audit._sppf_sync_check::test_docflow_sppf_gh_refs.py::tests.test_docflow_sppf_gh_refs._load_audit_tools
def test_sppf_gh_refs_required_mode_ignores_irrelevant_paths(tmp_path: Path) -> None:
    audit_tools = _load_audit_tools()
    sppf_sync = _FakeSppfSync(
        rev_range="HEAD~1..HEAD",
        commits=["abc123"],
        issue_ids=set(),
    )

    violations, warnings = audit_tools._sppf_sync_check(
        tmp_path,
        mode="required",
        load_sppf_sync_module_fn=lambda: sppf_sync,
        git_diff_paths_fn=lambda _rev: ["docs/notes.md"],
    )

    assert violations == []
    assert warnings == []


# gabion:evidence E:call_footprint::tests/test_docflow_sppf_gh_refs.py::test_sppf_gh_refs_required_mode_passes_when_refs_present::governance_audit.py::gabion.tooling.governance_audit._sppf_sync_check::test_docflow_sppf_gh_refs.py::tests.test_docflow_sppf_gh_refs._load_audit_tools
def test_sppf_gh_refs_required_mode_passes_when_refs_present(tmp_path: Path) -> None:
    audit_tools = _load_audit_tools()
    sppf_sync = _FakeSppfSync(
        rev_range="HEAD~1..HEAD",
        commits=["abc123"],
        issue_ids={"GH-123"},
    )

    violations, warnings = audit_tools._sppf_sync_check(
        tmp_path,
        mode="required",
        load_sppf_sync_module_fn=lambda: sppf_sync,
        git_diff_paths_fn=lambda _rev: ["docs/sppf_checklist.md"],
    )

    assert violations == []
    assert warnings == []
