from __future__ import annotations

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
