from __future__ import annotations

from gabion.analysis import test_annotation_drift_delta


# gabion:evidence E:function_site::test_annotation_drift_delta.py::gabion.analysis.test_annotation_drift_delta.build_delta_payload
def test_annotation_drift_delta_payload_and_render() -> None:
    baseline_payload = test_annotation_drift_delta.build_baseline_payload(
        {"ok": 2, "orphaned": 1}
    )
    baseline = test_annotation_drift_delta.parse_baseline_payload(baseline_payload)
    current_payload = test_annotation_drift_delta.build_baseline_payload(
        {"ok": 3, "orphaned": 1, "legacy_tag": 2}
    )
    current = test_annotation_drift_delta.parse_baseline_payload(current_payload)
    delta_payload = test_annotation_drift_delta.build_delta_payload(baseline, current)
    summary = delta_payload.get("summary", {})
    delta = summary.get("delta", {})
    assert delta.get("ok") == 1
    md = test_annotation_drift_delta.render_markdown(delta_payload)
    assert "generated_by_spec_id" in md
