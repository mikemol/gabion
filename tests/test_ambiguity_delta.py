from __future__ import annotations

import pytest

from gabion.analysis import ambiguity_delta


# gabion:evidence E:function_site::ambiguity_delta.py::gabion.analysis.ambiguity_delta.build_delta_payload
def test_ambiguity_delta_payload_and_render() -> None:
    baseline_payload = ambiguity_delta.build_baseline_payload(
        [
            {"kind": "local_resolution_ambiguous"},
            {"kind": "local_resolution_ambiguous"},
            {"kind": "other"},
        ]
    )
    baseline = ambiguity_delta.parse_baseline_payload(baseline_payload)
    current_payload = ambiguity_delta.build_baseline_payload(
        [
            {"kind": "local_resolution_ambiguous"},
            {"kind": "new_kind"},
        ]
    )
    current = ambiguity_delta.parse_baseline_payload(current_payload)
    delta_payload = ambiguity_delta.build_delta_payload(baseline, current)
    summary = delta_payload.get("summary", {})
    total = summary.get("total", {})
    assert total.get("delta") == current.total - baseline.total
    md = ambiguity_delta.render_markdown(delta_payload)
    assert "generated_by_spec_id" in md
    assert "total:" in md


# gabion:evidence E:function_site::ambiguity_delta.py::gabion.analysis.ambiguity_delta.parse_baseline_payload
def test_ambiguity_baseline_rejects_invalid_version() -> None:
    with pytest.raises(ValueError):
        ambiguity_delta.parse_baseline_payload({"version": "bad", "summary": {}})


# gabion:evidence E:function_site::ambiguity_delta.py::gabion.analysis.ambiguity_delta.load_baseline
def test_ambiguity_baseline_load_rejects_non_object(tmp_path) -> None:
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text("[]\n")
    with pytest.raises(ValueError):
        ambiguity_delta.load_baseline(str(baseline_path))


# gabion:evidence E:function_site::ambiguity_delta.py::gabion.analysis.ambiguity_delta.render_markdown
def test_ambiguity_delta_render_handles_invalid_numbers() -> None:
    payload = {
        "summary": {
            "total": {"baseline": 1, "current": 1, "delta": "bad"},
            "by_kind": {
                "baseline": {"x": 1},
                "current": {"x": 1},
                "delta": {"x": "bad"},
            },
        }
    }
    rendered = ambiguity_delta.render_markdown(payload)
    assert "total:" in rendered


# gabion:evidence E:call_footprint::tests/test_ambiguity_delta.py::test_ambiguity_baseline_parse_ignores_non_mapping_summary_shapes::ambiguity_delta.py::gabion.analysis.ambiguity_delta.parse_baseline_payload
def test_ambiguity_baseline_parse_ignores_non_mapping_summary_shapes() -> None:
    parsed = ambiguity_delta.parse_baseline_payload(
        {
            "version": 1,
            "summary": [],
            "generated_by_spec_id": "spec",
            "generated_by_spec": {},
        }
    )
    assert parsed.total == 0
    assert parsed.by_kind == {}

    parsed = ambiguity_delta.parse_baseline_payload(
        {
            "version": 1,
            "summary": {"total": 3, "by_kind": []},
            "generated_by_spec_id": "spec",
            "generated_by_spec": {},
        }
    )
    assert parsed.total == 3
    assert parsed.by_kind == {}


# gabion:evidence E:call_footprint::tests/test_ambiguity_delta.py::test_ambiguity_delta_render_ignores_non_mapping_by_kind_summary::ambiguity_delta.py::gabion.analysis.ambiguity_delta.render_markdown
def test_ambiguity_delta_render_ignores_non_mapping_by_kind_summary() -> None:
    rendered = ambiguity_delta.render_markdown(
        {
            "summary": {
                "total": {"baseline": 1, "current": 2, "delta": 1},
                "by_kind": ["invalid"],
            }
        }
    )
    assert "- total: 1 -> 2 (+1)" in rendered
