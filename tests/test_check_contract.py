from __future__ import annotations

from pathlib import Path

import pytest
import typer

from gabion.commands import check_contract


def _artifact_flags() -> check_contract.CheckArtifactFlags:
    return check_contract.CheckArtifactFlags(
        emit_test_obsolescence=False,
        emit_test_evidence_suggestions=False,
        emit_call_clusters=False,
        emit_call_cluster_consolidation=False,
        emit_test_annotation_drift=False,
        emit_semantic_coverage_map=False,
    )


def _delta_options() -> check_contract.CheckDeltaOptions:
    return check_contract.CheckDeltaOptions(
        obsolescence_mode=check_contract.CheckAuxMode(kind="off"),
        annotation_drift_mode=check_contract.CheckAuxMode(kind="off"),
        ambiguity_mode=check_contract.CheckAuxMode(kind="off"),
    )


# gabion:evidence E:function_site::tests/test_check_contract.py::test_check_aux_operation_validation_errors_cover_domain_action_and_baseline
def test_check_aux_operation_validation_errors_cover_domain_action_and_baseline() -> None:
    with pytest.raises(typer.BadParameter):
        check_contract.CheckAuxOperation(domain="invalid", action="report").validate()

    with pytest.raises(typer.BadParameter):
        check_contract.CheckAuxOperation(
            domain="ambiguity",
            action="report",
        ).validate()

    with pytest.raises(typer.BadParameter):
        check_contract.CheckAuxOperation(
            domain="obsolescence",
            action="delta",
        ).validate()


# gabion:evidence E:function_site::tests/test_check_contract.py::test_check_aux_operation_to_payload_and_build_payload_aux_surface
def test_check_aux_operation_to_payload_and_build_payload_aux_surface() -> None:
    aux = check_contract.CheckAuxOperation(
        domain="obsolescence",
        action="baseline-write",
        baseline_path=Path("baselines/obsolescence.json"),
        state_in_path=Path("state.json"),
        out_json=Path("delta.json"),
        out_md=Path("delta.md"),
    )
    payload = aux.to_payload()
    assert payload["domain"] == "obsolescence"
    assert payload["action"] == "baseline-write"
    assert payload["baseline_path"] == "baselines/obsolescence.json"
    assert payload["state_in"] == "state.json"

    check_payload = check_contract.build_check_payload(
        paths=[Path("sample.py")],
        report=None,
        fail_on_violations=False,
        root=Path("."),
        config=None,
        baseline=Path("baseline.txt"),
        baseline_write=True,
        decision_snapshot=None,
        artifact_flags=_artifact_flags(),
        delta_options=_delta_options(),
        exclude=None,
        filter_bundle=check_contract.DataflowFilterBundle(
            ignore_params_csv=None,
            transparent_decorators_csv=None,
        ),
        allow_external=None,
        strictness="high",
        fail_on_type_ambiguities=False,
        lint=False,
        aux_operation=aux,
    )
    assert check_payload["aux_operation"] == payload


def test_check_aux_operation_accepts_taint_lifecycle_without_baseline() -> None:
    aux = check_contract.CheckAuxOperation(
        domain="taint",
        action="lifecycle",
    )
    payload = aux.to_payload()
    assert payload["domain"] == "taint"
    assert payload["action"] == "lifecycle"
    assert payload["baseline_path"] is None


# gabion:evidence E:function_site::check_contract.py::gabion.commands.check_contract.build_check_payload
def test_build_check_payload_includes_aspf_controls() -> None:
    payload = check_contract.build_check_payload(
        paths=[Path("sample.py")],
        report=None,
        fail_on_violations=False,
        root=Path("."),
        config=None,
        baseline=None,
        baseline_write=False,
        decision_snapshot=None,
        artifact_flags=_artifact_flags(),
        delta_options=_delta_options(),
        exclude=None,
        filter_bundle=check_contract.DataflowFilterBundle(
            ignore_params_csv=None,
            transparent_decorators_csv=None,
        ),
        allow_external=None,
        strictness="high",
        fail_on_type_ambiguities=False,
        lint=False,
        aspf_trace_json=Path("artifacts/out/aspf_trace.json"),
        aspf_import_trace=[Path("artifacts/out/prev_trace.json")],
        aspf_equivalence_against=[Path("artifacts/out/baseline_trace.json")],
        aspf_opportunities_json=Path("artifacts/out/aspf_opportunities.json"),
        aspf_state_json=Path("artifacts/out/aspf_state/session/0001_step.json"),
        aspf_import_state=[Path("artifacts/out/aspf_state/session/0000_prev.json")],
        aspf_semantic_surface=["groups_by_path", "violation_summary"],
    )
    assert payload["aspf_trace_json"] == "artifacts/out/aspf_trace.json"
    assert payload["aspf_import_trace"] == ["artifacts/out/prev_trace.json"]
    assert payload["aspf_equivalence_against"] == [
        "artifacts/out/baseline_trace.json"
    ]
    assert payload["aspf_opportunities_json"] == "artifacts/out/aspf_opportunities.json"
    assert payload["aspf_state_json"] == "artifacts/out/aspf_state/session/0001_step.json"
    assert payload["aspf_import_state"] == [
        "artifacts/out/aspf_state/session/0000_prev.json"
    ]
    assert payload["aspf_semantic_surface"] == ["groups_by_path", "violation_summary"]


# gabion:evidence E:function_site::tests/test_check_contract.py::test_lint_entries_decision_protocol_trichotomy
def test_lint_entries_decision_protocol_trichotomy() -> None:
    provided = check_contract.LintEntriesDecision.from_response(
        {"lint_entries": [{"path": "a.py", "line": 1, "col": 1, "code": "X", "message": "m"}], "lint_lines": ["ignored"]}
    )
    assert provided.kind == "provided_entries"
    assert len(provided.normalize_entries(parse_lint_entry_fn=lambda _line: None)) == 1

    derived = check_contract.LintEntriesDecision.from_response(
        {"lint_lines": ["a.py:1:2: X detail", "invalid"]}
    )
    parsed = derived.normalize_entries(
        parse_lint_entry_fn=(
            lambda line: {"line": line} if line.startswith("a.py:") else None
        )
    )
    assert derived.kind == "derive_from_lines"
    assert parsed == [{"line": "a.py:1:2: X detail"}]

    empty = check_contract.LintEntriesDecision.from_response({})
    assert empty.kind == "empty"
    assert empty.normalize_entries(parse_lint_entry_fn=lambda _line: {"unused": True}) == []


def test_check_aux_mode_validate_and_delta_option_properties_cover_edges() -> None:
    with pytest.raises(typer.BadParameter):
        check_contract.CheckAuxMode(kind="report").validate(
            domain="ambiguity",
            allow_report=False,
        )

    options = check_contract.CheckDeltaOptions(
        obsolescence_mode=check_contract.CheckAuxMode(
            kind="baseline-write",
            state_path=Path("obsolescence_state.json"),
        ),
        annotation_drift_mode=check_contract.CheckAuxMode(
            kind="baseline-write",
            state_path=Path("annotation_state.json"),
        ),
        ambiguity_mode=check_contract.CheckAuxMode(
            kind="baseline-write",
            state_path=Path("ambiguity_state.json"),
        ),
    )
    assert options.obsolescence_mode.emit_report is False
    assert options.obsolescence_mode.write_baseline is True
    assert options.emit_test_obsolescence_state is False
    assert options.test_obsolescence_state == Path("obsolescence_state.json")
    assert options.write_test_obsolescence_baseline is True
    assert options.test_annotation_drift_state == Path("annotation_state.json")
    assert options.write_test_annotation_drift_baseline is True
    assert options.ambiguity_state == Path("ambiguity_state.json")
    assert options.write_ambiguity_baseline is True
