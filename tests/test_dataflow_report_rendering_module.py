from __future__ import annotations

import random

from gabion.analysis.dataflow_report_rendering import (
    render_synthesis_section,
    render_unsupported_by_adapter_section,
)


def _check_deadline() -> None:
    return None


# gabion:evidence E:call_footprint::tests/test_dataflow_report_rendering_module.py::test_render_synthesis_section_module_evidence_summary_and_blank_fields::dataflow_report_rendering.py::gabion.analysis.dataflow_report_rendering.render_synthesis_section
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


# gabion:evidence E:call_footprint::tests/test_dataflow_report_rendering_module.py::test_render_synthesis_section_is_byte_stable_for_shuffled_inputs::dataflow_report_rendering.py::gabion.analysis.dataflow_report_rendering.render_synthesis_section
def test_render_synthesis_section_is_byte_stable_for_shuffled_inputs() -> None:
    baseline = None
    base_protocols = [
        {
            "name": "Beta",
            "tier": 2,
            "fields": [{"name": "b", "type_hint": "int"}],
            "evidence": ["decision_surface", "dataflow"],
        },
        {
            "name": "Alpha",
            "tier": 3,
            "fields": [{"name": "a", "type_hint": "str"}],
            "evidence": ["dataflow"],
        },
    ]
    for seed in range(15):
        rng = random.Random(seed)
        protocols = [dict(spec) for spec in base_protocols]
        rng.shuffle(protocols)
        for spec in protocols:
            evidence = list(spec.get("evidence", []))
            rng.shuffle(evidence)
            spec["evidence"] = evidence
        plan = {"protocols": protocols, "warnings": [], "errors": []}
        rendered = render_synthesis_section(plan, check_deadline=_check_deadline)
        if baseline is None:
            baseline = rendered
            continue
        assert rendered == baseline


# gabion:evidence E:call_footprint::tests/test_dataflow_report_rendering_module.py::test_render_synthesis_section_renders_error_block::dataflow_report_rendering.py::gabion.analysis.dataflow_report_rendering.render_synthesis_section
def test_render_synthesis_section_renders_error_block() -> None:
    text = render_synthesis_section(
        {"protocols": [], "warnings": [], "errors": ["first", "second"]},
        check_deadline=_check_deadline,
    )
    assert "Errors:" in text
    assert "first" in text
    assert "second" in text



# gabion:evidence E:call_footprint::tests/test_dataflow_report_rendering_module.py::test_render_unsupported_by_adapter_section_marks_required::dataflow_report_rendering.py::gabion.analysis.dataflow_report_rendering.render_unsupported_by_adapter_section
def test_render_unsupported_by_adapter_section_marks_required() -> None:
    lines = render_unsupported_by_adapter_section(
        [
            {"surface": "type-flow", "adapter": "limited", "required_by_policy": False},
            {"surface": "decision-surfaces", "adapter": "limited", "required_by_policy": True},
        ],
        check_deadline=_check_deadline,
    )
    assert "type-flow: unsupported_by_adapter (limited)" in lines
    assert "decision-surfaces: unsupported_by_adapter (limited) [required]" in lines
