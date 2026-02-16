from __future__ import annotations

from gabion.analysis.dataflow_report_rendering import render_synthesis_section


def _check_deadline() -> None:
    return None


def test_render_synthesis_section_module_evidence_summary_and_blank_fields() -> None:
    plan = {
        "protocols": [
            {
                "name": "Bundle",
                "tier": 2,
                "fields": [
                    {"name": "a", "type_hint": "int", "source_params": ["a"]},
                    {"name": "", "type_hint": "str", "source_params": []},
                ],
                "bundle": ["a"],
                "rationale": "test",
                "evidence": ["dataflow", "decision_surface"],
            }
        ],
        "warnings": [],
        "errors": [],
    }
    text = render_synthesis_section(plan, check_deadline=_check_deadline)
    assert "evidence:" in text
    assert "Evidence summary:" in text

    blank_fields_plan = {
        "protocols": [
            {
                "name": "Bundle",
                "tier": 2,
                "fields": [
                    {"name": "", "type_hint": "int", "source_params": []},
                    {"type_hint": "str", "source_params": []},
                ],
                "bundle": [],
                "rationale": "test",
                "evidence": [],
            }
        ],
        "warnings": [],
        "errors": [],
    }
    assert "(no fields)" in render_synthesis_section(blank_fields_plan, check_deadline=_check_deadline)
