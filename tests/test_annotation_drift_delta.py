from __future__ import annotations

import pytest

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


# gabion:evidence E:function_site::test_annotation_drift_delta.py::gabion.analysis.test_annotation_drift_delta.parse_baseline_payload
def test_annotation_drift_delta_rejects_bad_baseline(tmp_path) -> None:
    with pytest.raises(ValueError):
        test_annotation_drift_delta.parse_baseline_payload(
            {"version": "bad", "summary": {}}
        )

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text("[]\n")
    with pytest.raises(ValueError):
        test_annotation_drift_delta.load_baseline(str(baseline_path))


# gabion:evidence E:function_site::test_annotation_drift_delta.py::gabion.analysis.test_annotation_drift_delta.render_markdown
def test_annotation_drift_delta_handles_invalid_numbers() -> None:
    payload = test_annotation_drift_delta.build_baseline_payload({"ok": "bad"})
    assert payload["summary"]["ok"] == 0

    rendered = test_annotation_drift_delta.render_markdown(
        {
            "summary": {
                "baseline": {"ok": 1},
                "current": {"ok": 1},
                "delta": {"ok": "bad"},
            }
        }
    )
    assert "- ok:" in rendered
