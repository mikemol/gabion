from __future__ import annotations

from pathlib import Path
import json
import sys

import pytest


def _load_audit_tools():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "scripts"))
    import audit_tools

    return audit_tools


def _write_decision_snapshot(path: Path, *, include_forest: bool = False) -> None:
    payload = {
        "format_version": 1,
        "root": str(path.parent),
        "decision_surfaces": ["mod.py:mod.fn decision surface params: a (boundary)"],
        "value_decision_surfaces": [],
        "summary": {"decision_surfaces": 1, "value_decision_surfaces": 0},
    }
    if include_forest:
        payload["forest"] = {"format_version": 1, "nodes": [], "alts": []}
    path.write_text(json.dumps(payload))


def test_consolidation_requires_forest_in_strict_mode(tmp_path: Path) -> None:
    audit_tools = _load_audit_tools()
    (tmp_path / "gabion.toml").write_text("[consolidation]\nrequire_forest = true\n")
    decision_path = tmp_path / "decision_snapshot.json"
    lint_path = tmp_path / "lint.txt"
    output_path = tmp_path / "consolidation_report.md"
    _write_decision_snapshot(decision_path, include_forest=False)
    lint_path.write_text("")

    with pytest.raises(SystemExit) as exc:
        audit_tools.run_consolidation_cli(
            [
                "--root",
                str(tmp_path),
                "--decision",
                str(decision_path),
                "--lint",
                str(lint_path),
                "--output",
                str(output_path),
            ]
        )
    assert "forest-only mode enabled" in str(exc.value)


def test_consolidation_allows_fallback_in_permissive_mode(tmp_path: Path) -> None:
    audit_tools = _load_audit_tools()
    (tmp_path / "gabion.toml").write_text("[consolidation]\nrequire_forest = false\n")
    decision_path = tmp_path / "decision_snapshot.json"
    lint_path = tmp_path / "lint.txt"
    output_path = tmp_path / "consolidation_report.md"
    _write_decision_snapshot(decision_path, include_forest=False)
    lint_path.write_text("")

    result = audit_tools.run_consolidation_cli(
        [
            "--root",
            str(tmp_path),
            "--decision",
            str(decision_path),
            "--lint",
            str(lint_path),
            "--output",
            str(output_path),
        ]
    )
    assert result == 0
    report_text = output_path.read_text()
    assert "FOREST_FALLBACK_USED" in report_text
