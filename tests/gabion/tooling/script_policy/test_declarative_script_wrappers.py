from __future__ import annotations

import json
from pathlib import Path

from scripts.audit import audit_in_step_structure
from scripts.governance import docflow_promote_sections
from scripts.misc import extract_test_evidence


# gabion:behavior primary=desired
def test_audit_in_step_structure_main_accepts_positional_paths(
    tmp_path: Path,
    capsys,
) -> None:
    doc = tmp_path / "in-999.md"
    doc.write_text(
        "---\n"
        "doc_role: in_step\n"
        "---\n"
        "# purpose\n",
        encoding="utf-8",
    )

    rc = audit_in_step_structure.main([str(doc)])

    assert rc == 2
    assert "missing frontmatter field 'doc_id'" in capsys.readouterr().out


# gabion:behavior primary=desired
def test_extract_test_evidence_main_writes_output_from_hosted_runtime(
    tmp_path: Path,
) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_sample.py").write_text(
        "def test_sample():\n"
        "    assert True\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "out/test_evidence.json"

    rc = extract_test_evidence.main(
        [
            "--tests",
            str(tests_dir),
            "--out",
            str(out_path),
            "--root",
            str(tmp_path),
        ]
    )

    assert rc == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["scope"]["root"] == "."
    assert len(payload["tests"]) == 1
    assert payload["tests"][0]["test_id"].endswith("tests/test_sample.py::test_sample")


# gabion:behavior primary=desired
def test_docflow_promote_sections_main_adds_declared_sections(
    tmp_path: Path,
    capsys,
) -> None:
    doc = tmp_path / "guide.md"
    doc.write_text(
        "---\n"
        "doc_id: guide\n"
        "doc_requires: []\n"
        "doc_reviewed_as_of: {}\n"
        "doc_review_notes: {}\n"
        "---\n"
        "# Guide\n"
        "Body.\n",
        encoding="utf-8",
    )

    rc = docflow_promote_sections.main([str(doc), "--add-sections"])

    assert rc == 0
    text = doc.read_text(encoding="utf-8")
    assert "doc_sections:" in text
    assert '<a id="guide"></a>' in text
    assert "Updated 1 document(s)." in capsys.readouterr().out
