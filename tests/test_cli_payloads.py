from __future__ import annotations

from pathlib import Path

import pytest
import typer

from gabion import cli

_DEFAULT_CHECK_ARTIFACT_FLAGS = cli.CheckArtifactFlags(
    emit_test_obsolescence=False,
    emit_test_evidence_suggestions=False,
    emit_call_clusters=False,
    emit_call_cluster_consolidation=False,
    emit_test_annotation_drift=False,
)


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli.build_check_payload::ambiguity_state,baseline,config,decision_snapshot,emit_ambiguity_delta,emit_ambiguity_state,emit_test_annotation_drift_delta,emit_test_obsolescence_delta,emit_test_obsolescence_state,fail_on_type_ambiguities,paths,report,strictness,test_annotation_drift_state,test_obsolescence_state,write_ambiguity_baseline,write_test_annotation_drift_baseline,write_test_obsolescence_baseline E:decision_surface/direct::cli.py::gabion.cli._split_csv_entries::entries E:decision_surface/direct::cli.py::gabion.cli._split_csv::value
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
        artifact_flags=_DEFAULT_CHECK_ARTIFACT_FLAGS,
        delta_options=cli.CheckDeltaOptions(
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
        ),
        exclude=None,
        filter_bundle=cli.DataflowFilterBundle(
            ignore_params_csv=None,
            transparent_decorators_csv=None,
        ),
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
    assert payload["emit_test_obsolescence_state"] is False
    assert payload["test_obsolescence_state"] is None
    assert payload["emit_test_obsolescence_delta"] is False
    assert payload["emit_test_evidence_suggestions"] is False
    assert payload["emit_call_clusters"] is False
    assert payload["emit_call_cluster_consolidation"] is False
    assert payload["emit_test_annotation_drift"] is False
    assert payload["test_annotation_drift_state"] is None
    assert payload["emit_test_annotation_drift_delta"] is False
    assert payload["write_test_annotation_drift_baseline"] is False
    assert payload["write_test_obsolescence_baseline"] is False
    assert payload["emit_ambiguity_delta"] is False
    assert payload["emit_ambiguity_state"] is False
    assert payload["ambiguity_state"] is None
    assert payload["write_ambiguity_baseline"] is False
    assert payload["exclude"] is None
    assert payload["ignore_params"] is None
    assert payload["transparent_decorators"] is None
    assert payload["emit_timeout_progress_report"] is False
    assert payload["resume_on_timeout"] == 0


# gabion:evidence E:call_footprint::tests/test_cli_payloads.py::test_check_builds_payload_with_none_filter_bundle::cli.py::gabion.cli.build_check_payload
def test_check_builds_payload_with_none_filter_bundle() -> None:
    payload = cli.build_check_payload(
        paths=[Path(".")],
        report=None,
        fail_on_violations=True,
        root=Path("."),
        config=None,
        baseline=None,
        baseline_write=False,
        decision_snapshot=None,
        artifact_flags=_DEFAULT_CHECK_ARTIFACT_FLAGS,
        delta_options=cli.CheckDeltaOptions(
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
        ),
        exclude=None,
        filter_bundle=None,
        allow_external=None,
        strictness=None,
        fail_on_type_ambiguities=False,
        lint=False,
    )
    assert payload["ignore_params"] is None
    assert payload["transparent_decorators"] is None


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli.build_check_payload::ambiguity_state,baseline,config,decision_snapshot,emit_ambiguity_delta,emit_ambiguity_state,emit_test_annotation_drift_delta,emit_test_obsolescence_delta,emit_test_obsolescence_state,fail_on_type_ambiguities,paths,report,strictness,test_annotation_drift_state,test_obsolescence_state,write_ambiguity_baseline,write_test_annotation_drift_baseline,write_test_obsolescence_baseline E:decision_surface/direct::cli.py::gabion.cli._split_csv_entries::entries E:decision_surface/direct::cli.py::gabion.cli._split_csv::value
def test_check_payload_preserves_strictness_for_server_validation() -> None:
    payload = cli.build_check_payload(
        paths=[Path(".")],
        report=None,
        fail_on_violations=True,
        root=Path("."),
        config=None,
        baseline=None,
        baseline_write=False,
        decision_snapshot=None,
        artifact_flags=_DEFAULT_CHECK_ARTIFACT_FLAGS,
        delta_options=cli.CheckDeltaOptions(
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
        ),
        exclude=None,
        filter_bundle=cli.DataflowFilterBundle(
            ignore_params_csv=None,
            transparent_decorators_csv=None,
        ),
        allow_external=None,
        strictness="medium",
        fail_on_type_ambiguities=False,
        lint=False,
    )
    assert payload["strictness"] == "medium"


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._split_csv::value E:decision_surface/direct::cli.py::gabion.cli._split_csv_entries::entries E:decision_surface/direct::cli.py::gabion.cli.build_check_payload::ambiguity_state,baseline,config,decision_snapshot,emit_ambiguity_delta,emit_ambiguity_state,emit_test_annotation_drift_delta,emit_test_obsolescence_delta,emit_test_obsolescence_state,fail_on_type_ambiguities,paths,report,strictness,test_annotation_drift_state,test_obsolescence_state,write_ambiguity_baseline,write_test_annotation_drift_baseline,write_test_obsolescence_baseline
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
        artifact_flags=_DEFAULT_CHECK_ARTIFACT_FLAGS,
        delta_options=cli.CheckDeltaOptions(
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
        ),
        exclude=None,
        filter_bundle=cli.DataflowFilterBundle(
            ignore_params_csv=None,
            transparent_decorators_csv=None,
        ),
        allow_external=None,
        strictness=None,
        fail_on_type_ambiguities=False,
        lint=False,
    )
    assert payload["baseline_write"] is False


# gabion:evidence E:call_footprint::tests/test_cli_payloads.py::test_build_check_execution_plan_request_sets_write_baseline_mode::cli.py::gabion.cli.build_check_execution_plan_request
def test_build_check_execution_plan_request_sets_write_baseline_mode() -> None:
    baseline = Path("baselines/dataflow_baseline.txt")
    request = cli.build_check_execution_plan_request(
        payload={"analysis_timeout_ticks": 11, "analysis_timeout_tick_ns": 22},
        report=Path("artifacts/audit_reports/dataflow_report.md"),
        decision_snapshot=None,
        baseline=baseline,
        baseline_write=True,
        policy=cli.CheckPolicyFlags(
            fail_on_violations=False,
            fail_on_type_ambiguities=False,
            lint=False,
        ),
        profile="raw",
        artifact_flags=_DEFAULT_CHECK_ARTIFACT_FLAGS,
        emit_test_obsolescence_state=False,
        emit_test_obsolescence_delta=False,
        emit_test_annotation_drift_delta=False,
        emit_ambiguity_delta=False,
        emit_ambiguity_state=False,
    )
    payload = request.to_payload()
    policy_metadata = payload["policy_metadata"]
    assert isinstance(policy_metadata, dict)
    assert policy_metadata["baseline_mode"] == "write"


# gabion:evidence E:call_footprint::tests/test_cli_payloads.py::test_build_check_execution_plan_request_sets_read_baseline_mode::cli.py::gabion.cli.build_check_execution_plan_request
def test_build_check_execution_plan_request_sets_read_baseline_mode() -> None:
    baseline = Path("baselines/dataflow_baseline.txt")
    request = cli.build_check_execution_plan_request(
        payload={"analysis_timeout_ticks": 11, "analysis_timeout_tick_ns": 22},
        report=Path("artifacts/audit_reports/dataflow_report.md"),
        decision_snapshot=None,
        baseline=baseline,
        baseline_write=False,
        policy=cli.CheckPolicyFlags(
            fail_on_violations=False,
            fail_on_type_ambiguities=False,
            lint=False,
        ),
        profile="raw",
        artifact_flags=_DEFAULT_CHECK_ARTIFACT_FLAGS,
        emit_test_obsolescence_state=False,
        emit_test_obsolescence_delta=False,
        emit_test_annotation_drift_delta=False,
        emit_ambiguity_delta=False,
        emit_ambiguity_state=False,
    )
    payload = request.to_payload()
    policy_metadata = payload["policy_metadata"]
    assert isinstance(policy_metadata, dict)
    assert policy_metadata["baseline_mode"] == "read"


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._split_csv_entries::entries E:decision_surface/direct::cli.py::gabion.cli.build_dataflow_payload::opts E:decision_surface/direct::cli.py::gabion.cli._split_csv::value
def test_dataflow_audit_payload_parsing() -> None:
    opts = cli.parse_dataflow_args_or_exit(
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


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._split_csv::value E:decision_surface/direct::cli.py::gabion.cli._split_csv_entries::entries E:decision_surface/direct::cli.py::gabion.cli.build_dataflow_payload::opts
def test_dataflow_payload_baseline_and_transparent() -> None:
    opts = cli.parse_dataflow_args_or_exit(
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


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli.build_check_payload::ambiguity_state,baseline,config,decision_snapshot,emit_ambiguity_delta,emit_ambiguity_state,emit_test_annotation_drift_delta,emit_test_obsolescence_delta,emit_test_obsolescence_state,fail_on_type_ambiguities,paths,report,strictness,test_annotation_drift_state,test_obsolescence_state,write_ambiguity_baseline,write_test_annotation_drift_baseline,write_test_obsolescence_baseline E:decision_surface/direct::cli.py::gabion.cli.build_dataflow_payload::opts
def test_check_and_raw_payloads_match_common_fields() -> None:
    check_payload = cli.build_check_payload(
        paths=[Path("sample.py")],
        report=Path("report.md"),
        fail_on_violations=True,
        root=Path("."),
        config=Path("cfg.toml"),
        baseline=Path("baseline.txt"),
        baseline_write=True,
        decision_snapshot=Path("decision.json"),
        artifact_flags=_DEFAULT_CHECK_ARTIFACT_FLAGS,
        delta_options=cli.CheckDeltaOptions(
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
        ),
        exclude=["a,b"],
        filter_bundle=cli.DataflowFilterBundle(
            ignore_params_csv="x,y",
            transparent_decorators_csv="d1,d2",
        ),
        allow_external=True,
        strictness="low",
        fail_on_type_ambiguities=True,
        lint=True,
        resume_checkpoint=Path("resume.json"),
        emit_timeout_progress_report=True,
        resume_on_timeout=2,
    )
    raw_opts = cli.parse_dataflow_args_or_exit(
        [
            "sample.py",
            "--root",
            ".",
            "--config",
            "cfg.toml",
            "--report",
            "report.md",
            "--fail-on-violations",
            "--fail-on-type-ambiguities",
            "--baseline",
            "baseline.txt",
            "--baseline-write",
            "--emit-decision-snapshot",
            "decision.json",
            "--exclude",
            "a,b",
            "--ignore-params",
            "x,y",
            "--transparent-decorators",
            "d1,d2",
            "--allow-external",
            "--strictness",
            "low",
            "--lint",
            "--resume-checkpoint",
            "resume.json",
            "--emit-timeout-progress-report",
            "--resume-on-timeout",
            "2",
        ]
    )
    raw_payload = cli.build_dataflow_payload(raw_opts)
    common_keys = [
        "paths",
        "root",
        "config",
        "report",
        "fail_on_violations",
        "fail_on_type_ambiguities",
        "baseline",
        "baseline_write",
        "decision_snapshot",
        "exclude",
        "ignore_params",
        "transparent_decorators",
        "allow_external",
        "strictness",
        "lint",
        "resume_checkpoint",
        "emit_timeout_progress_report",
        "resume_on_timeout",
        "deadline_profile",
    ]
    assert {key: check_payload[key] for key in common_keys} == {
        key: raw_payload[key] for key in common_keys
    }


# gabion:evidence E:function_site::cli.py::gabion.cli.build_dataflow_payload
def test_dataflow_payload_resume_checkpoint_and_timeout_flags() -> None:
    opts = cli.parse_dataflow_args_or_exit(
        [
            ".",
            "--resume-checkpoint",
            "resume.json",
            "--emit-timeout-progress-report",
            "--resume-on-timeout",
            "2",
        ]
    )
    payload = cli.build_dataflow_payload(opts)
    assert payload["resume_checkpoint"] == "resume.json"
    assert payload["emit_timeout_progress_report"] is True
    assert payload["resume_on_timeout"] == 2
    assert opts.emit_timeout_progress_report is True
    assert opts.resume_on_timeout == 2


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli.build_refactor_payload::bundle,input_payload,protocol_name,target_path
def test_refactor_protocol_payload(tmp_path: Path) -> None:
    payload = cli.build_refactor_payload(
        protocol_name="Bundle",
        bundle=["a", "b"],
        field=["a:int", "b:str"],
        target_path=tmp_path / "sample.py",
        target_functions=["alpha"],
        compatibility_shim=True,
        compatibility_shim_warnings=True,
        compatibility_shim_overloads=True,
        ambient_rewrite=False,
        rationale="use bundle",
    )
    assert payload["protocol_name"] == "Bundle"
    assert payload["bundle"] == ["a", "b"]
    assert payload["target_functions"] == ["alpha"]
    assert payload["compatibility_shim"] == {
        "enabled": True,
        "emit_deprecation_warning": True,
        "emit_overload_stubs": True,
    }


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli.build_refactor_payload::bundle,input_payload,protocol_name,target_path
def test_refactor_payload_infers_bundle(tmp_path: Path) -> None:
    payload = cli.build_refactor_payload(
        protocol_name="Bundle",
        bundle=None,
        field=["a:int", "b:str"],
        target_path=tmp_path / "sample.py",
        target_functions=[],
        compatibility_shim=False,
        compatibility_shim_warnings=True,
        compatibility_shim_overloads=True,
        ambient_rewrite=False,
        rationale=None,
    )
    assert payload["bundle"] == ["a", "b"]


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli.run_check::baseline E:decision_surface/direct::cli.py::gabion.cli.build_check_payload::ambiguity_state,baseline,config,decision_snapshot,emit_ambiguity_delta,emit_ambiguity_state,emit_test_annotation_drift_delta,emit_test_obsolescence_delta,emit_test_obsolescence_state,fail_on_type_ambiguities,paths,report,strictness,test_annotation_drift_state,test_obsolescence_state,write_ambiguity_baseline,write_test_annotation_drift_baseline,write_test_obsolescence_baseline
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
        policy=cli.CheckPolicyFlags(
            fail_on_violations=True,
            fail_on_type_ambiguities=True,
            lint=False,
        ),
        root=tmp_path,
        config=None,
        baseline=None,
        baseline_write=False,
        decision_snapshot=None,
        artifact_flags=_DEFAULT_CHECK_ARTIFACT_FLAGS,
        delta_options=cli.CheckDeltaOptions(
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
        ),
        exclude=None,
        filter_bundle=cli.DataflowFilterBundle(
            ignore_params_csv=None,
            transparent_decorators_csv=None,
        ),
        allow_external=None,
        strictness=None,
        runner=runner,
    )
    assert result["exit_code"] == 0
    assert captured["command"] == cli.DATAFLOW_COMMAND
    assert captured["payload"]["paths"] == [str(tmp_path)]
    assert captured["payload"]["report"] == str(
        tmp_path / "artifacts" / "audit_reports" / "dataflow_report.md"
    )
    assert (tmp_path / "artifacts" / "audit_reports").is_dir()
    assert captured["payload"]["emit_test_obsolescence"] is False
    assert captured["payload"]["emit_test_obsolescence_state"] is False
    assert captured["payload"]["test_obsolescence_state"] is None
    assert captured["payload"]["emit_test_obsolescence_delta"] is False
    assert captured["payload"]["emit_test_evidence_suggestions"] is False
    assert captured["payload"]["emit_call_clusters"] is False
    assert captured["payload"]["emit_call_cluster_consolidation"] is False
    assert captured["payload"]["emit_test_annotation_drift"] is False
    assert captured["payload"]["test_annotation_drift_state"] is None
    assert captured["payload"]["emit_test_annotation_drift_delta"] is False
    assert captured["payload"]["write_test_annotation_drift_baseline"] is False
    assert captured["payload"]["write_test_obsolescence_baseline"] is False
    assert captured["payload"]["emit_ambiguity_delta"] is False
    assert captured["payload"]["emit_ambiguity_state"] is False
    assert captured["payload"]["ambiguity_state"] is None
    assert captured["payload"]["write_ambiguity_baseline"] is False
    execution_plan_request = captured["payload"].get("execution_plan_request")
    assert isinstance(execution_plan_request, dict)
    assert execution_plan_request["requested_operations"] == sorted(
        [
            cli.DATAFLOW_COMMAND,
            "gabion.check",
        ]
    )
    assert execution_plan_request["derived_artifacts"]
    assert captured["root"] == tmp_path


# gabion:evidence E:call_footprint::tests/test_cli_payloads.py::test_run_check_uses_explicit_report_path::cli.py::gabion.cli.run_check
def test_run_check_uses_explicit_report_path(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def runner(request, *, root=None):
        captured["payload"] = request.arguments[0]
        captured["root"] = root
        return {"exit_code": 0}

    explicit_report = tmp_path / "custom.md"
    result = cli.run_check(
        paths=[tmp_path],
        report=explicit_report,
        policy=cli.CheckPolicyFlags(
            fail_on_violations=True,
            fail_on_type_ambiguities=True,
            lint=False,
        ),
        root=tmp_path,
        config=None,
        baseline=None,
        baseline_write=False,
        decision_snapshot=None,
        artifact_flags=_DEFAULT_CHECK_ARTIFACT_FLAGS,
        delta_options=cli.CheckDeltaOptions(
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
        ),
        exclude=None,
        filter_bundle=cli.DataflowFilterBundle(
            ignore_params_csv=None,
            transparent_decorators_csv=None,
        ),
        allow_external=None,
        strictness=None,
        runner=runner,
    )
    assert result["exit_code"] == 0
    assert captured["payload"]["report"] == str(explicit_report)
    assert captured["root"] == tmp_path


# gabion:evidence E:call_footprint::tests/test_cli_payloads.py::test_run_check_with_none_filter_bundle::cli.py::gabion.cli.run_check
def test_run_check_with_none_filter_bundle(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def runner(request, *, root=None):
        captured["payload"] = request.arguments[0]
        captured["root"] = root
        return {"exit_code": 0}

    result = cli.run_check(
        paths=[tmp_path],
        report=None,
        policy=cli.CheckPolicyFlags(
            fail_on_violations=True,
            fail_on_type_ambiguities=False,
            lint=False,
        ),
        root=tmp_path,
        config=None,
        baseline=None,
        baseline_write=False,
        decision_snapshot=None,
        artifact_flags=_DEFAULT_CHECK_ARTIFACT_FLAGS,
        delta_options=cli.CheckDeltaOptions(
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
        ),
        exclude=None,
        filter_bundle=None,
        allow_external=None,
        strictness=None,
        runner=runner,
    )
    assert result["exit_code"] == 0
    assert captured["payload"]["ignore_params"] is None
    assert captured["payload"]["transparent_decorators"] is None
    assert captured["root"] == tmp_path


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli.build_check_payload::ambiguity_state,baseline,config,decision_snapshot,emit_ambiguity_delta,emit_ambiguity_state,emit_test_annotation_drift_delta,emit_test_obsolescence_delta,emit_test_obsolescence_state,fail_on_type_ambiguities,paths,report,strictness,test_annotation_drift_state,test_obsolescence_state,write_ambiguity_baseline,write_test_annotation_drift_baseline,write_test_obsolescence_baseline E:decision_surface/direct::cli.py::gabion.cli._split_csv_entries::entries E:decision_surface/direct::cli.py::gabion.cli._split_csv::value
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
            artifact_flags=_DEFAULT_CHECK_ARTIFACT_FLAGS,
            delta_options=cli.CheckDeltaOptions(
                emit_test_obsolescence_state=False,
                test_obsolescence_state=None,
                emit_test_obsolescence_delta=True,
                test_annotation_drift_state=None,
                emit_test_annotation_drift_delta=False,
                write_test_annotation_drift_baseline=False,
                write_test_obsolescence_baseline=True,
                emit_ambiguity_delta=False,
                emit_ambiguity_state=False,
                ambiguity_state=None,
                write_ambiguity_baseline=False,
            ),
            exclude=None,
            filter_bundle=cli.DataflowFilterBundle(
                ignore_params_csv=None,
                transparent_decorators_csv=None,
            ),
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
            artifact_flags=_DEFAULT_CHECK_ARTIFACT_FLAGS,
            delta_options=cli.CheckDeltaOptions(
                emit_test_obsolescence_state=False,
                test_obsolescence_state=None,
                emit_test_obsolescence_delta=False,
                test_annotation_drift_state=None,
                emit_test_annotation_drift_delta=True,
                write_test_annotation_drift_baseline=True,
                write_test_obsolescence_baseline=False,
                emit_ambiguity_delta=False,
                emit_ambiguity_state=False,
                ambiguity_state=None,
                write_ambiguity_baseline=False,
            ),
            exclude=None,
            filter_bundle=cli.DataflowFilterBundle(
                ignore_params_csv=None,
                transparent_decorators_csv=None,
            ),
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
            artifact_flags=_DEFAULT_CHECK_ARTIFACT_FLAGS,
            delta_options=cli.CheckDeltaOptions(
                emit_test_obsolescence_state=False,
                test_obsolescence_state=None,
                emit_test_obsolescence_delta=False,
                test_annotation_drift_state=None,
                emit_test_annotation_drift_delta=False,
                write_test_annotation_drift_baseline=False,
                write_test_obsolescence_baseline=False,
                emit_ambiguity_delta=True,
                emit_ambiguity_state=False,
                ambiguity_state=None,
                write_ambiguity_baseline=True,
            ),
            exclude=None,
            filter_bundle=cli.DataflowFilterBundle(
                ignore_params_csv=None,
                transparent_decorators_csv=None,
            ),
            allow_external=None,
            strictness=None,
            fail_on_type_ambiguities=False,
            lint=False,
        )


# gabion:evidence E:function_site::cli.py::gabion.cli.build_check_payload
def test_check_payload_rejects_obsolescence_state_and_path() -> None:
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
            artifact_flags=_DEFAULT_CHECK_ARTIFACT_FLAGS,
            delta_options=cli.CheckDeltaOptions(
                emit_test_obsolescence_state=True,
                test_obsolescence_state=Path(
                    "artifacts/out/test_obsolescence_state.json"
                ),
                emit_test_obsolescence_delta=False,
                test_annotation_drift_state=None,
                emit_test_annotation_drift_delta=False,
                write_test_annotation_drift_baseline=False,
                write_test_obsolescence_baseline=False,
                emit_ambiguity_delta=False,
                emit_ambiguity_state=False,
                ambiguity_state=None,
                write_ambiguity_baseline=False,
            ),
            exclude=None,
            filter_bundle=cli.DataflowFilterBundle(
                ignore_params_csv=None,
                transparent_decorators_csv=None,
            ),
            allow_external=None,
            strictness=None,
            fail_on_type_ambiguities=False,
            lint=False,
        )


# gabion:evidence E:function_site::cli.py::gabion.cli.build_check_payload
def test_check_payload_rejects_ambiguity_state_and_path() -> None:
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
            artifact_flags=_DEFAULT_CHECK_ARTIFACT_FLAGS,
            delta_options=cli.CheckDeltaOptions(
                emit_test_obsolescence_state=False,
                test_obsolescence_state=None,
                emit_test_obsolescence_delta=False,
                test_annotation_drift_state=None,
                emit_test_annotation_drift_delta=False,
                write_test_annotation_drift_baseline=False,
                write_test_obsolescence_baseline=False,
                emit_ambiguity_delta=False,
                emit_ambiguity_state=True,
                ambiguity_state=Path("artifacts/out/ambiguity_state.json"),
                write_ambiguity_baseline=False,
            ),
            exclude=None,
            filter_bundle=cli.DataflowFilterBundle(
                ignore_params_csv=None,
                transparent_decorators_csv=None,
            ),
            allow_external=None,
            strictness=None,
            fail_on_type_ambiguities=False,
            lint=False,
        )
