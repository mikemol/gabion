from __future__ import annotations

from pathlib import Path

from pytest import CaptureFixture

from scripts.deadline_runtime import DeadlineBudget, deadline_scope_from_lsp_env
from scripts.sppf_status_audit import (
    _collect_overrides,
    _extract_in_status,
    _normalize_checklist_pair,
    _normalize_in_status,
    _normalize_influence_status,
    main,
    run_audit,
)

_TEST_BUDGET = DeadlineBudget(ticks=120_000, tick_ns=1_000_000)


def _run_with_deadline(path: Path) -> tuple[int, list[str]]:
    with deadline_scope_from_lsp_env(default_budget=_TEST_BUDGET):
        return run_audit(path)


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

    code, lines = _run_with_deadline(tmp_path)

    assert code == 0
    assert lines == ["sppf-status-audit: no drift detected"]


def test_sppf_status_audit_fails_for_stale_in_doc_status(tmp_path: Path) -> None:
    _write_fixture(
        tmp_path,
        in_status="Planned refinement.",
        checklist_status="done",
        influence_status="adopted",
    )

    code, lines = _run_with_deadline(tmp_path)

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

    code, lines = _run_with_deadline(tmp_path)

    assert code == 1
    assert any("in-1: status drift detected" in line for line in lines)
    assert any("influence_index: implemented-in-part" in line for line in lines)


def test_normalization_helpers_cover_status_variants() -> None:
    assert _normalize_in_status("Implemented in part") == "implemented-in-part"
    assert _normalize_in_status("Partially implemented") == "implemented-in-part"
    assert _normalize_in_status("partial") == "implemented-in-part"
    assert _normalize_in_status("IN PROGRESS") == "implemented-in-part"
    assert _normalize_in_status("queued") == "planned"
    assert _normalize_in_status("Done and adopted") == "done"
    assert _normalize_in_status("unknown") is None

    assert _normalize_influence_status("adopted") == "done"
    assert _normalize_influence_status("partial") == "implemented-in-part"
    assert _normalize_influence_status("queued") == "planned"
    assert _normalize_influence_status("untriaged") == "planned"
    assert _normalize_influence_status("rejected") == "rejected"
    assert _normalize_influence_status("bogus") is None

    assert _normalize_checklist_pair("done", "done") == "done"
    assert _normalize_checklist_pair("done", "planned") == "planned"
    assert _normalize_checklist_pair("partial", "done") == "implemented-in-part"


def test_extract_in_status_handles_missing_or_empty_sections(tmp_path: Path) -> None:
    file_no_heading = tmp_path / "no-heading.md"
    file_no_heading.write_text("No status section", encoding="utf-8")
    with deadline_scope_from_lsp_env(default_budget=_TEST_BUDGET):
        assert _extract_in_status(file_no_heading) is None

    file_comment_heading = tmp_path / "comment-heading.md"
    file_comment_heading.write_text("### Status\n# next heading", encoding="utf-8")
    with deadline_scope_from_lsp_env(default_budget=_TEST_BUDGET):
        assert _extract_in_status(file_comment_heading) is None

    file_empty_status = tmp_path / "empty-status.md"
    file_empty_status.write_text("### Status\n\n\n", encoding="utf-8")
    with deadline_scope_from_lsp_env(default_budget=_TEST_BUDGET):
        assert _extract_in_status(file_empty_status) is None


def test_collect_overrides_is_case_insensitive() -> None:
    with deadline_scope_from_lsp_env(default_budget=_TEST_BUDGET):
        overrides = _collect_overrides(
            "sppf-status-override: in-2\nsppf-status-override: ALL",
            "sppf-status-override: in-9",
        )
    assert overrides == {"in-2", "all", "in-9"}


def test_run_audit_handles_unknown_rows_and_untracked_in_file_names(tmp_path: Path) -> None:
    (tmp_path / "in").mkdir(parents=True, exist_ok=True)
    (tmp_path / "docs").mkdir(parents=True, exist_ok=True)

    (tmp_path / "in" / "in-1.md").write_text(
        "---\ndoc_revision: 1\n---\n\n### Status\nmystery\n",
        encoding="utf-8",
    )
    (tmp_path / "in" / "in-2.md").write_text(
        "---\ndoc_revision: 1\n---\n\n### Status\nmystery\n",
        encoding="utf-8",
    )
    (tmp_path / "in" / "in-foo.md").write_text(
        "---\ndoc_revision: 1\n---\n\n### Status\ndone\n",
        encoding="utf-8",
    )

    (tmp_path / "docs" / "sppf_checklist.md").write_text(
        "\n".join(
            [
                "- [x] Ignored row. sppf{doc=done; impl=done; doc_ref=in-1@1}",
                (
                    "- <a id=\"in-1-test\"></a>[ ] Planned row. "
                    "sppf-status-node sppf{doc=queued; impl=planned; doc_ref=in-1@1}"
                ),
                (
                    "- <a id=\"in-1-test-2\"></a>[ ] Mixed row. "
                    "sppf-status-node sppf{doc=done; impl=partial; doc_ref=in-1@1}"
                ),
                (
                    "- <a id=\"in-2-test\"></a>[ ] Planned-only row. "
                    "sppf-status-node sppf{doc=queued; impl=planned; doc_ref=in-2@1}"
                ),
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "docs" / "influence_index.md").write_text(
        "## Inbox entries\n\n- in/in-1.md — **mystery**\n",
        encoding="utf-8",
    )

    code, lines = _run_with_deadline(tmp_path)

    assert code == 0
    assert lines == ["sppf-status-audit: no drift detected"]


def test_run_audit_allows_override_marker(tmp_path: Path) -> None:
    _write_fixture(
        tmp_path,
        in_status="Planned",
        checklist_status="done",
        influence_status="adopted",
    )
    checklist = tmp_path / "docs" / "sppf_checklist.md"
    checklist.write_text(
        checklist.read_text(encoding="utf-8") + "\nsppf-status-override: in-1\n",
        encoding="utf-8",
    )

    code, lines = _run_with_deadline(tmp_path)

    assert code == 0
    assert lines == ["sppf-status-audit: no drift detected"]


def test_main_reports_to_stdout_and_stderr(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    _write_fixture(
        tmp_path,
        in_status="Done",
        checklist_status="done",
        influence_status="adopted",
    )

    assert main(["--root", str(tmp_path)]) == 0
    ok_capture = capsys.readouterr()
    assert "no drift detected" in ok_capture.out
    assert ok_capture.err == ""

    _write_fixture(
        tmp_path,
        in_status="Planned",
        checklist_status="done",
        influence_status="adopted",
    )

    assert main(["--root", str(tmp_path)]) == 1
    err_capture = capsys.readouterr()
    assert "status drift detected" in err_capture.err
