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
        emit_test_obsolescence_state=False,
        test_obsolescence_state=None,
        emit_test_obsolescence_delta=False,
        test_annotation_drift_state=None,
        emit_test_annotation_drift_delta=False,
        write_test_annotation_drift_baseline=False,
        write_test_obsolescence_baseline=False,
        emit_ambiguity_delta=False,
        emit_ambiguity_state=False,
        ambiguity_state=None,
        write_ambiguity_baseline=False,
    )


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
