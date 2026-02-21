from __future__ import annotations

import json
from pathlib import Path


def _load_audit_tools():
    from scripts import audit_tools

    return audit_tools


# gabion:evidence E:call_footprint::tests/test_audit_tools_obligations.py::test_emit_docflow_compliance_includes_obligations::audit_tools.py::scripts.audit_tools._emit_docflow_compliance::test_audit_tools_obligations.py::tests.test_audit_tools_obligations._load_audit_tools
def test_emit_docflow_compliance_includes_obligations(tmp_path: Path) -> None:
    audit_tools = _load_audit_tools()
    json_output = tmp_path / "compliance.json"
    md_output = tmp_path / "compliance.md"

    obligations = audit_tools.DocflowObligationResult(
        entries=[
            {
                "obligation_id": "sppf_gh_reference_validation",
                "triggered": True,
                "status": "unmet",
                "enforcement": "fail",
                "description": "SPPF-relevant path changes require GH-reference validation.",
            }
        ],
        summary={"total": 1, "triggered": 1, "met": 0, "unmet_fail": 1, "unmet_warn": 0},
        warnings=[],
        violations=["obligation unmet"],
    )

    audit_tools._emit_docflow_compliance(
        rows=[],
        invariants=[],
        json_output=json_output,
        md_output=md_output,
        obligations=obligations,
    )

    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert payload["obligations"]["summary"]["unmet_fail"] == 1
    assert payload["obligations"]["entries"][0]["obligation_id"] == "sppf_gh_reference_validation"

    report = md_output.read_text(encoding="utf-8")
    assert "Obligations:" in report
    assert "sppf_gh_reference_validation: unmet (fail)" in report
