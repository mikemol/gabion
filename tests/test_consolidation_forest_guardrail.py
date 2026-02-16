from __future__ import annotations

from pathlib import Path
import json

import pytest

from gabion.exceptions import NeverThrown
from tests.env_helpers import env_scope

def _load_audit_tools():
    from scripts import audit_tools

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

# gabion:evidence E:function_site::test_consolidation_forest_guardrail.py::tests.test_consolidation_forest_guardrail._load_audit_tools E:function_site::test_consolidation_forest_guardrail.py::tests.test_consolidation_forest_guardrail._write_decision_snapshot
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

# gabion:evidence E:function_site::test_consolidation_forest_guardrail.py::tests.test_consolidation_forest_guardrail._load_audit_tools E:function_site::test_consolidation_forest_guardrail.py::tests.test_consolidation_forest_guardrail._write_decision_snapshot
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

def test_audit_tools_gas_limit_env_override() -> None:
    audit_tools = _load_audit_tools()
    with env_scope({"GABION_AUDIT_GAS_LIMIT": "12345"}):
        assert audit_tools._audit_gas_limit() == 12345

@pytest.mark.parametrize("value", ["", "0", "-1", "bad"])
def test_audit_tools_gas_limit_env_rejects_invalid(value: str) -> None:
    audit_tools = _load_audit_tools()
    with env_scope({"GABION_AUDIT_GAS_LIMIT": value}):
        if value == "":
            assert audit_tools._audit_gas_limit() == audit_tools._DEFAULT_AUDIT_GAS_LIMIT
        else:
            with pytest.raises(NeverThrown):
                audit_tools._audit_gas_limit()
