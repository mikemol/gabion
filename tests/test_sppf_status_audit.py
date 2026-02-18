from __future__ import annotations

from pathlib import Path

from scripts.sppf_status_audit import run_audit


def _write_fixture(root: Path, *, in_status: str, checklist_status: str, influence_status: str) -> None:
    (root / "in").mkdir(parents=True, exist_ok=True)
    (root / "docs").mkdir(parents=True, exist_ok=True)

    (root / "in" / "in-1.md").write_text(
        f"""---
doc_revision: 1
---

### Status
{in_status}
""",
        encoding="utf-8",
    )
    (root / "docs" / "sppf_checklist.md").write_text(
        f"""- <a id=\"in-1-test\"></a>[x] Example node. sppf{{doc={checklist_status}; impl={checklist_status}; doc_ref=in-1@1}}
""",
        encoding="utf-8",
    )
    (root / "docs" / "influence_index.md").write_text(
        f"""## Inbox entries

- in/in-1.md — **{influence_status}**
""",
        encoding="utf-8",
    )


def test_sppf_status_audit_passes_for_matching_statuses(tmp_path: Path) -> None:
    _write_fixture(
        tmp_path,
        in_status="Done and adopted.",
        checklist_status="done",
        influence_status="adopted",
    )

    code, lines = run_audit(tmp_path)

    assert code == 0
    assert lines == ["sppf-status-audit: no drift detected"]


def test_sppf_status_audit_fails_for_stale_in_doc_status(tmp_path: Path) -> None:
    _write_fixture(
        tmp_path,
        in_status="Planned refinement.",
        checklist_status="done",
        influence_status="adopted",
    )

    code, lines = run_audit(tmp_path)

    assert code == 1
    assert any("in-1: status drift detected" in line for line in lines)
    assert any("in: planned" in line for line in lines)


def test_sppf_status_audit_fails_for_stale_influence_row(tmp_path: Path) -> None:
    _write_fixture(
        tmp_path,
        in_status="Done and adopted.",
        checklist_status="done",
        influence_status="partial",
    )

    code, lines = run_audit(tmp_path)

    assert code == 1
    assert any("in-1: status drift detected" in line for line in lines)
    assert any("influence_index: implemented-in-part" in line for line in lines)
