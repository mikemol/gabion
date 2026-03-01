from __future__ import annotations

from pathlib import Path

import pytest

from gabion.exceptions import NeverThrown
from gabion.tooling.execution_envelope import ExecutionEnvelope


def test_execution_envelope_for_delta_bundle_and_raw() -> None:
    delta = ExecutionEnvelope.for_delta_bundle(
        root=Path("."),
        report_path=Path("artifacts/audit_reports/dataflow_report.md"),
        strictness="high",
        allow_external=True,
        aspf_state_json=Path("out/state.json"),
        aspf_delta_jsonl=Path("out/delta.jsonl"),
        aspf_import_state=(Path("out/import.json"),),
    )
    assert delta.operation == "delta_bundle"
    assert delta.report_path is not None
    assert delta.strictness == "high"

    raw = ExecutionEnvelope.for_raw(
        root=Path("."),
        aspf_state_json=None,
        aspf_delta_jsonl=None,
    )
    assert raw.operation == "raw"
    assert raw.report_path is None


def test_execution_envelope_validate_rejects_invalid_shapes() -> None:
    with pytest.raises(NeverThrown):
        ExecutionEnvelope(
            operation="unexpected",  # type: ignore[arg-type]
            root=Path("."),
            report_path=None,
            strictness=None,
            allow_external=None,
            aspf_state_json=None,
            aspf_delta_jsonl=None,
            aspf_import_state=(),
        ).validate()

    with pytest.raises(NeverThrown):
        ExecutionEnvelope(
            operation="delta_bundle",  # type: ignore[arg-type]
            root=Path("."),
            report_path=None,
            strictness=None,
            allow_external=None,
            aspf_state_json=None,
            aspf_delta_jsonl=None,
            aspf_import_state=(),
        ).validate()

    with pytest.raises(NeverThrown):
        ExecutionEnvelope(
            operation="delta_bundle",
            root=Path("."),
            report_path=Path("report.md"),
            strictness="medium",
            allow_external=None,
            aspf_state_json=None,
            aspf_delta_jsonl=None,
            aspf_import_state=(),
        ).validate()

    with pytest.raises(NeverThrown):
        ExecutionEnvelope(
            operation="raw",
            root=Path("."),
            report_path=None,
            strictness=None,
            allow_external=None,
            aspf_state_json=Path("state.json"),
            aspf_delta_jsonl=None,
            aspf_import_state=(),
        ).validate()
