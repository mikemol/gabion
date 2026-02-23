from __future__ import annotations

from pathlib import Path

from gabion.tooling import governance_audit as audit_tools


def _write_triplet(
    root: Path,
    *,
    in_status: str,
    checklist_state: str,
    influence_status: str,
    override_on: str | None = None,
) -> None:
    (root / "in").mkdir(parents=True, exist_ok=True)
    (root / "docs").mkdir(parents=True, exist_ok=True)

    status_line = in_status
    if override_on == "in":
        status_line = f"{status_line}  <!-- {audit_tools.STATUS_TRIPLET_OVERRIDE_MARKER} -->"
    (root / "in" / "in-99.md").write_text(
        "---\n"
        "doc_revision: 1\n"
        "---\n"
        "# in/in-99.md\n"
        "\n"
        "### Status\n"
        f"{status_line}\n",
        encoding="utf-8",
    )

    checklist_line = f"- [{checklist_state}] Node text. (in-99) sppf{{doc=partial; impl=partial; doc_ref=in-99@1}}"
    if override_on == "checklist":
        checklist_line = f"{checklist_line} <!-- {audit_tools.STATUS_TRIPLET_OVERRIDE_MARKER} -->"
    (root / "docs" / "sppf_checklist.md").write_text(checklist_line + "\n", encoding="utf-8")

    index_line = f"- in/in-99.md — **{influence_status}** (summary row)."
    if override_on == "index":
        index_line = f"{index_line} <!-- {audit_tools.STATUS_TRIPLET_OVERRIDE_MARKER} -->"
    (root / "docs" / "influence_index.md").write_text(index_line + "\n", encoding="utf-8")


# gabion:evidence E:call_footprint::tests/test_docflow_status_triplets.py::test_status_triplets_accept_mapped_statuses::governance_audit.py::gabion.tooling.governance_audit._sppf_status_triplet_violations::test_docflow_status_triplets.py::tests.test_docflow_status_triplets._write_triplet
def test_status_triplets_accept_mapped_statuses(tmp_path: Path) -> None:
    _write_triplet(
        tmp_path,
        in_status="Planned refinement.",
        checklist_state=" ",
        influence_status="queued",
    )

    violations = audit_tools._sppf_status_triplet_violations(tmp_path)

    assert violations == []


# gabion:evidence E:call_footprint::tests/test_docflow_status_triplets.py::test_status_triplets_report_all_three_records_on_conflict::governance_audit.py::gabion.tooling.governance_audit._sppf_status_triplet_violations::test_docflow_status_triplets.py::tests.test_docflow_status_triplets._write_triplet
def test_status_triplets_report_all_three_records_on_conflict(tmp_path: Path) -> None:
    _write_triplet(
        tmp_path,
        in_status="Planned refinement.",
        checklist_state="x",
        influence_status="adopted",
    )

    violations = audit_tools._sppf_status_triplet_violations(tmp_path)

    assert len(violations) == 1
    message = violations[0]
    assert "in/in-99.md" in message
    assert "docs/sppf_checklist.md" in message
    assert "docs/influence_index.md" in message


# gabion:evidence E:call_footprint::tests/test_docflow_status_triplets.py::test_status_triplets_honor_override_marker::governance_audit.py::gabion.tooling.governance_audit._sppf_status_triplet_violations::test_docflow_status_triplets.py::tests.test_docflow_status_triplets._write_triplet
def test_status_triplets_honor_override_marker(tmp_path: Path) -> None:
    _write_triplet(
        tmp_path,
        in_status="Planned refinement.",
        checklist_state="x",
        influence_status="adopted",
        override_on="index",
    )

    violations = audit_tools._sppf_status_triplet_violations(tmp_path)

    assert violations == []
