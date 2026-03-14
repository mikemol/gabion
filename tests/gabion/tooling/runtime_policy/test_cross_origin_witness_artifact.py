from __future__ import annotations

from pathlib import Path

from gabion.tooling.runtime.cross_origin_witness_artifact import (
    build_cross_origin_witness_contract_artifact_payload,
)


def test_build_cross_origin_witness_contract_artifact_payload_captures_remap_and_overlap_rows(
    tmp_path: Path,
) -> None:
    payload = build_cross_origin_witness_contract_artifact_payload(root=tmp_path)

    assert payload["artifact_kind"] == "cross_origin_witness_contract"
    assert payload["summary"]["case_count"] == 2
    assert payload["summary"]["failing_case_count"] == 0
    assert payload["summary"]["witness_row_count"] >= 3
    assert {item["case_key"] for item in payload["cases"]} == {
        "analysis_union_path_remap",
        "condition_overlap_ledger",
    }
    remap_case = next(
        item for item in payload["cases"] if item["case_key"] == "analysis_union_path_remap"
    )
    overlap_case = next(
        item for item in payload["cases"] if item["case_key"] == "condition_overlap_ledger"
    )
    assert remap_case["status"] == "pass"
    assert overlap_case["status"] == "pass"
    assert remap_case["row_keys"]
    assert overlap_case["row_keys"]
    assert any(
        item["row_kind"] == "path_remap"
        and item["left_origin_kind"] == "analysis_input_witness.file"
        and item["right_origin_kind"] == "aspf_union_view.module"
        for item in payload["witness_rows"]
    )
    assert any(
        item["row_kind"] == "condition_overlap"
        and item["left_origin_kind"] == "taint_interval"
        and item["right_origin_kind"] == "condition_event"
        for item in payload["witness_rows"]
    )
