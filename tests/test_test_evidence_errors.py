from __future__ import annotations

from pathlib import Path

from gabion.analysis import test_evidence


def test_extract_file_evidence_handles_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.py"
    assert test_evidence._extract_file_evidence(missing, tmp_path) == []


def test_extract_file_evidence_handles_syntax_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("def oops(:\n", encoding="utf-8")
    assert test_evidence._extract_file_evidence(bad, tmp_path) == []


def test_normalize_evidence_items_skips_empty_and_duplicates() -> None:
    items = test_evidence._normalize_evidence_items(
        ["", "E:paramset::a", "E:paramset::a"]
    )
    assert len(items) == 1
