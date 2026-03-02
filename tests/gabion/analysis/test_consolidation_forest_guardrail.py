from __future__ import annotations

from pathlib import Path
import json

import pytest

from gabion.exceptions import NeverThrown
from tests.env_helpers import env_scope

def _load_audit_tools():
    from gabion.tooling import governance_audit as audit_tools

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
        payload["forest"] = {
            "format_version": 1,
            "nodes": [
                {
                    "kind": "FunctionSite",
                    "key": ["mod.py", "mod.fn"],
                    "meta": {"path": "mod.py", "qual": "mod.fn"},
                },
                {
                    "kind": "ParamSet",
                    "key": ["a"],
                    "meta": {"params": ["a"]},
                },
            ],
            "alts": [
                {
                    "kind": "DecisionSurface",
                    "inputs": [
                        {"kind": "FunctionSite", "key": ["mod.py", "mod.fn"]},
                        {"kind": "ParamSet", "key": ["a"]},
                    ],
                    "evidence": {"meta": "boundary"},
                }
            ],
        }
    path.write_text(json.dumps(payload))

# gabion:evidence E:function_site::test_consolidation_forest_guardrail.py::tests.test_consolidation_forest_guardrail._load_audit_tools E:function_site::test_consolidation_forest_guardrail.py::tests.test_consolidation_forest_guardrail._write_decision_snapshot E:decision_surface/direct::test_consolidation_forest_guardrail.py::tests.test_consolidation_forest_guardrail._load_audit_tools::stale_df4a2ad0492a
def test_consolidation_requires_forest_in_strict_mode(tmp_path: Path) -> None:
    audit_tools = _load_audit_tools()
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

# gabion:evidence E:function_site::test_consolidation_forest_guardrail.py::tests.test_consolidation_forest_guardrail._load_audit_tools E:function_site::test_consolidation_forest_guardrail.py::tests.test_consolidation_forest_guardrail._write_decision_snapshot E:decision_surface/direct::test_consolidation_forest_guardrail.py::tests.test_consolidation_forest_guardrail._load_audit_tools::stale_649d8b273257_3f84b302
def test_consolidation_uses_forest_when_present(tmp_path: Path) -> None:
    audit_tools = _load_audit_tools()
    decision_path = tmp_path / "decision_snapshot.json"
    lint_path = tmp_path / "lint.txt"
    output_path = tmp_path / "consolidation_report.md"
    _write_decision_snapshot(decision_path, include_forest=True)
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
    assert "Consolidation source mode: forest-native" in report_text
    assert "FOREST_FALLBACK_USED" not in report_text


# gabion:evidence E:function_site::test_consolidation_forest_guardrail.py::tests.test_consolidation_forest_guardrail._load_audit_tools E:function_site::test_consolidation_forest_guardrail.py::tests.test_consolidation_forest_guardrail._write_decision_snapshot
def test_consolidation_explicit_fallback_mode_emits_warning_payload(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    audit_tools = _load_audit_tools()
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
            "--allow-fallback",
        ]
    )

    assert result == 0
    captured = capsys.readouterr()
    assert '"warning_class": "consolidation.forest_fallback"' in captured.out
    report_text = output_path.read_text()
    assert "Consolidation source mode: fallback-derived" in report_text
    assert "FOREST_FALLBACK_USED" in report_text

# gabion:evidence E:call_footprint::tests/test_consolidation_forest_guardrail.py::test_audit_tools_gas_limit_env_override::governance_audit.py::gabion.tooling.governance_audit._audit_gas_limit::env_helpers.py::tests.env_helpers.env_scope::test_consolidation_forest_guardrail.py::tests.test_consolidation_forest_guardrail._load_audit_tools
def test_audit_tools_gas_limit_env_override() -> None:
    audit_tools = _load_audit_tools()
    with env_scope({"GABION_AUDIT_GAS_LIMIT": "12345"}):
        assert audit_tools._audit_gas_limit() == 12345

# gabion:evidence E:call_footprint::tests/test_consolidation_forest_guardrail.py::test_audit_tools_gas_limit_env_rejects_invalid::governance_audit.py::gabion.tooling.governance_audit._audit_gas_limit::env_helpers.py::tests.env_helpers.env_scope::test_consolidation_forest_guardrail.py::tests.test_consolidation_forest_guardrail._load_audit_tools
@pytest.mark.parametrize("value", ["", "0", "-1", "bad"])
def test_audit_tools_gas_limit_env_rejects_invalid(value: str) -> None:
    audit_tools = _load_audit_tools()
    with env_scope({"GABION_AUDIT_GAS_LIMIT": value}):
        if value == "":
            assert audit_tools._audit_gas_limit() == audit_tools._DEFAULT_AUDIT_GAS_LIMIT
        else:
            with pytest.raises(NeverThrown):
                audit_tools._audit_gas_limit()
