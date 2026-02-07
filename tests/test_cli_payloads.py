from __future__ import annotations

from pathlib import Path

import pytest
import typer

from gabion import cli


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli.build_check_payload::baseline,config,decision_snapshot,emit_test_obsolescence_delta,fail_on_type_ambiguities,paths,report,strictness,write_test_obsolescence_baseline E:decision_surface/direct::cli.py::gabion.cli._split_csv_entries::entries E:decision_surface/direct::cli.py::gabion.cli._split_csv::value
def test_check_builds_payload() -> None:
    payload = cli.build_check_payload(
        paths=None,
        report=None,
        fail_on_violations=True,
        root=Path("."),
        config=None,
        baseline=None,
        baseline_write=False,
        decision_snapshot=None,
        emit_test_obsolescence=False,
        emit_test_obsolescence_delta=False,
        emit_test_evidence_suggestions=False,
        emit_test_annotation_drift=False,
        emit_test_annotation_drift_delta=False,
        write_test_annotation_drift_baseline=False,
        write_test_obsolescence_baseline=False,
        emit_ambiguity_delta=False,
        write_ambiguity_baseline=False,
        exclude=None,
        ignore_params_csv=None,
        transparent_decorators_csv=None,
        allow_external=None,
        strictness=None,
        fail_on_type_ambiguities=True,
        lint=False,
    )
    assert payload["paths"] == ["."]
    assert payload["fail_on_violations"] is True
    assert payload["fail_on_type_ambiguities"] is True
    assert payload["type_audit"] is True
    assert payload["emit_test_obsolescence"] is False
    assert payload["emit_test_obsolescence_delta"] is False
    assert payload["emit_test_evidence_suggestions"] is False
    assert payload["emit_test_annotation_drift"] is False
    assert payload["emit_test_annotation_drift_delta"] is False
    assert payload["write_test_annotation_drift_baseline"] is False
    assert payload["write_test_obsolescence_baseline"] is False
    assert payload["emit_ambiguity_delta"] is False
    assert payload["write_ambiguity_baseline"] is False


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli.build_check_payload::baseline,config,decision_snapshot,emit_test_obsolescence_delta,fail_on_type_ambiguities,paths,report,strictness,write_test_obsolescence_baseline E:decision_surface/direct::cli.py::gabion.cli._split_csv_entries::entries E:decision_surface/direct::cli.py::gabion.cli._split_csv::value
def test_check_payload_strictness_validation() -> None:
    with pytest.raises(typer.BadParameter):
        cli.build_check_payload(
            paths=[Path(".")],
            report=None,
            fail_on_violations=True,
            root=Path("."),
            config=None,
            baseline=None,
            baseline_write=False,
            decision_snapshot=None,
            emit_test_obsolescence=False,
            emit_test_obsolescence_delta=False,
            emit_test_evidence_suggestions=False,
            emit_test_annotation_drift=False,
            emit_test_annotation_drift_delta=False,
            write_test_annotation_drift_baseline=False,
            write_test_obsolescence_baseline=False,
            emit_ambiguity_delta=False,
            write_ambiguity_baseline=False,
            exclude=None,
            ignore_params_csv=None,
            transparent_decorators_csv=None,
            allow_external=None,
            strictness="medium",
            fail_on_type_ambiguities=False,
            lint=False,
        )


# gabion:evidence E:baseline/ratchet_monotonicity
def test_check_payload_baseline_write_requires_baseline() -> None:
    payload = cli.build_check_payload(
        paths=[Path(".")],
        report=None,
        fail_on_violations=True,
        root=Path("."),
        config=None,
        baseline=None,
        baseline_write=True,
        decision_snapshot=None,
        emit_test_obsolescence=False,
        emit_test_obsolescence_delta=False,
        emit_test_evidence_suggestions=False,
        emit_test_annotation_drift=False,
        emit_test_annotation_drift_delta=False,
        write_test_annotation_drift_baseline=False,
        write_test_obsolescence_baseline=False,
        emit_ambiguity_delta=False,
        write_ambiguity_baseline=False,
        exclude=None,
        ignore_params_csv=None,
        transparent_decorators_csv=None,
        allow_external=None,
        strictness=None,
        fail_on_type_ambiguities=False,
        lint=False,
    )
    assert payload["baseline_write"] is None


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._split_csv_entries::entries E:decision_surface/direct::cli.py::gabion.cli.build_dataflow_payload::opts E:decision_surface/direct::cli.py::gabion.cli._split_csv::value
def test_dataflow_audit_payload_parsing() -> None:
    opts = cli.parse_dataflow_args(
        [
            ".",
            "--strictness",
            "low",
            "--exclude",
            "a,b",
            "--ignore-params",
            "x,y",
            "--no-recursive",
            "--fail-on-violations",
            "--emit-structure-tree",
            "snapshot.json",
            "--emit-structure-metrics",
            "metrics.json",
            "--emit-decision-snapshot",
            "decisions.json",
        ]
    )
    payload = cli.build_dataflow_payload(opts)
    assert payload["paths"] == ["."]
    assert payload["strictness"] == "low"
    assert payload["exclude"] == ["a", "b"]
    assert payload["ignore_params"] == ["x", "y"]
    assert payload["no_recursive"] is True
    assert payload["fail_on_violations"] is True
    assert payload["structure_tree"] == "snapshot.json"
    assert payload["structure_metrics"] == "metrics.json"
    assert payload["decision_snapshot"] == "decisions.json"


# gabion:evidence E:baseline/ratchet_monotonicity
def test_dataflow_payload_baseline_and_transparent() -> None:
    opts = cli.parse_dataflow_args(
        [
            ".",
            "--baseline",
            "baseline.txt",
            "--baseline-write",
            "--transparent-decorators",
            "foo,bar",
            "--fail-on-type-ambiguities",
        ]
    )
    payload = cli.build_dataflow_payload(opts)
    assert payload["baseline"] == "baseline.txt"
    assert payload["baseline_write"] is True
    assert payload["transparent_decorators"] == ["foo", "bar"]
    assert payload["fail_on_type_ambiguities"] is True


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli.build_refactor_payload::bundle,input_payload,protocol_name,target_path
def test_refactor_protocol_payload(tmp_path: Path) -> None:
    payload = cli.build_refactor_payload(
        protocol_name="Bundle",
        bundle=["a", "b"],
        field=["a:int", "b:str"],
        target_path=tmp_path / "sample.py",
        target_functions=["alpha"],
        compatibility_shim=True,
        rationale="use bundle",
    )
    assert payload["protocol_name"] == "Bundle"
    assert payload["bundle"] == ["a", "b"]
    assert payload["target_functions"] == ["alpha"]
    assert payload["compatibility_shim"] is True


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli.build_refactor_payload::bundle,input_payload,protocol_name,target_path
def test_refactor_payload_infers_bundle(tmp_path: Path) -> None:
    payload = cli.build_refactor_payload(
        protocol_name="Bundle",
        bundle=None,
        field=["a:int", "b:str"],
        target_path=tmp_path / "sample.py",
        target_functions=[],
        compatibility_shim=False,
        rationale=None,
    )
    assert payload["bundle"] == ["a", "b"]


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli.run_check::baseline E:decision_surface/direct::cli.py::gabion.cli.build_check_payload::baseline,config,decision_snapshot,emit_test_obsolescence_delta,fail_on_type_ambiguities,paths,report,strictness,write_test_obsolescence_baseline
def test_run_check_uses_runner_dispatch(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def runner(request, *, root=None):
        captured["command"] = request.command
        captured["payload"] = request.arguments[0]
        captured["root"] = root
        return {"exit_code": 0}

    result = cli.run_check(
        paths=[tmp_path],
        report=None,
        fail_on_violations=True,
        root=tmp_path,
        config=None,
        baseline=None,
        baseline_write=False,
        decision_snapshot=None,
        emit_test_obsolescence=False,
        emit_test_obsolescence_delta=False,
        emit_test_evidence_suggestions=False,
        emit_test_annotation_drift=False,
        emit_test_annotation_drift_delta=False,
        write_test_annotation_drift_baseline=False,
        write_test_obsolescence_baseline=False,
        emit_ambiguity_delta=False,
        write_ambiguity_baseline=False,
        exclude=None,
        ignore_params_csv=None,
        transparent_decorators_csv=None,
        allow_external=None,
        strictness=None,
        fail_on_type_ambiguities=True,
        lint=False,
        runner=runner,
    )
    assert result["exit_code"] == 0
    assert captured["command"] == cli.DATAFLOW_COMMAND
    assert captured["payload"]["paths"] == [str(tmp_path)]
    assert captured["payload"]["emit_test_obsolescence"] is False
    assert captured["payload"]["emit_test_obsolescence_delta"] is False
    assert captured["payload"]["emit_test_evidence_suggestions"] is False
    assert captured["payload"]["emit_test_annotation_drift"] is False
    assert captured["payload"]["emit_test_annotation_drift_delta"] is False
    assert captured["payload"]["write_test_annotation_drift_baseline"] is False
    assert captured["payload"]["write_test_obsolescence_baseline"] is False
    assert captured["payload"]["emit_ambiguity_delta"] is False
    assert captured["payload"]["write_ambiguity_baseline"] is False
    assert captured["root"] == tmp_path


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli.build_check_payload::baseline,config,decision_snapshot,emit_test_obsolescence_delta,fail_on_type_ambiguities,paths,report,strictness,write_test_obsolescence_baseline E:decision_surface/direct::cli.py::gabion.cli._split_csv_entries::entries E:decision_surface/direct::cli.py::gabion.cli._split_csv::value
def test_check_payload_rejects_delta_and_baseline_write() -> None:
    with pytest.raises(typer.BadParameter):
        cli.build_check_payload(
            paths=[Path(".")],
            report=None,
            fail_on_violations=True,
            root=Path("."),
            config=None,
            baseline=None,
            baseline_write=False,
            decision_snapshot=None,
            emit_test_obsolescence=False,
            emit_test_obsolescence_delta=True,
            emit_test_evidence_suggestions=False,
            emit_test_annotation_drift=False,
            emit_test_annotation_drift_delta=False,
            write_test_annotation_drift_baseline=False,
            write_test_obsolescence_baseline=True,
            emit_ambiguity_delta=False,
            write_ambiguity_baseline=False,
            exclude=None,
            ignore_params_csv=None,
            transparent_decorators_csv=None,
            allow_external=None,
            strictness=None,
            fail_on_type_ambiguities=False,
            lint=False,
        )


# gabion:evidence E:function_site::cli.py::gabion.cli.build_check_payload
def test_check_payload_rejects_annotation_drift_delta_and_baseline_write() -> None:
    with pytest.raises(typer.BadParameter):
        cli.build_check_payload(
            paths=[Path(".")],
            report=None,
            fail_on_violations=True,
            root=Path("."),
            config=None,
            baseline=None,
            baseline_write=False,
            decision_snapshot=None,
            emit_test_obsolescence=False,
            emit_test_obsolescence_delta=False,
            emit_test_evidence_suggestions=False,
            emit_test_annotation_drift=False,
            emit_test_annotation_drift_delta=True,
            write_test_annotation_drift_baseline=True,
            write_test_obsolescence_baseline=False,
            emit_ambiguity_delta=False,
            write_ambiguity_baseline=False,
            exclude=None,
            ignore_params_csv=None,
            transparent_decorators_csv=None,
            allow_external=None,
            strictness=None,
            fail_on_type_ambiguities=False,
            lint=False,
        )


# gabion:evidence E:function_site::cli.py::gabion.cli.build_check_payload
def test_check_payload_rejects_ambiguity_delta_and_baseline_write() -> None:
    with pytest.raises(typer.BadParameter):
        cli.build_check_payload(
            paths=[Path(".")],
            report=None,
            fail_on_violations=True,
            root=Path("."),
            config=None,
            baseline=None,
            baseline_write=False,
            decision_snapshot=None,
            emit_test_obsolescence=False,
            emit_test_obsolescence_delta=False,
            emit_test_evidence_suggestions=False,
            emit_test_annotation_drift=False,
            emit_test_annotation_drift_delta=False,
            write_test_annotation_drift_baseline=False,
            write_test_obsolescence_baseline=False,
            emit_ambiguity_delta=True,
            write_ambiguity_baseline=True,
            exclude=None,
            ignore_params_csv=None,
            transparent_decorators_csv=None,
            allow_external=None,
            strictness=None,
            fail_on_type_ambiguities=False,
            lint=False,
        )
