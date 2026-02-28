from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from gabion import cli, server
from gabion.commands import check_contract


def _has_pygls() -> bool:
    return importlib.util.find_spec("pygls") is not None


def _cli_env() -> dict[str, str]:
    return {
        **os.environ,
        "GABION_DIRECT_RUN": "1",
        "GABION_LSP_TIMEOUT_TICKS": "5000",
        "GABION_LSP_TIMEOUT_TICK_NS": "1000000",
    }


def _dummy_ls(root: Path | None = None) -> SimpleNamespace:
    return SimpleNamespace(workspace=SimpleNamespace(root_path=str(root) if root else None))


def _with_timeout(payload: dict[str, object]) -> dict[str, object]:
    ticks, tick_ns = cli._cli_timeout_ticks()
    return {
        **payload,
        "analysis_timeout_ticks": int(ticks),
        "analysis_timeout_tick_ns": int(tick_ns),
    }


def _default_delta_options() -> cli.CheckDeltaOptions:
    return cli.CheckDeltaOptions(
        obsolescence_mode=check_contract.CheckAuxMode(kind="off"),
        annotation_drift_mode=check_contract.CheckAuxMode(kind="off"),
        ambiguity_mode=check_contract.CheckAuxMode(kind="off"),
    )


# gabion:evidence E:call_footprint::tests/test_cli_server_parity.py::test_synthesis_plan_cli_matches_server::server.py::gabion.server.execute_synthesis::test_cli_server_parity.py::tests.test_cli_server_parity._cli_env::test_cli_server_parity.py::tests.test_cli_server_parity._dummy_ls::test_cli_server_parity.py::tests.test_cli_server_parity._has_pygls::test_cli_server_parity.py::tests.test_cli_server_parity._with_timeout
@pytest.mark.skipif(not _has_pygls(), reason="pygls not installed")
def test_synthesis_plan_cli_matches_server(tmp_path: Path) -> None:
    payload = {
        "bundles": [{"bundle": ["a", "b"], "tier": 2}],
        "allow_singletons": True,
        "min_bundle_size": 1,
    }
    payload_path = tmp_path / "synth.json"
    output_path = tmp_path / "synth-out.json"
    payload_path.write_text(json.dumps(payload))

    runner = CliRunner()
    cli_result = runner.invoke(
        cli.app,
        ["synthesis-plan", "--input", str(payload_path), "--output", str(output_path)],
        env=_cli_env(),
    )
    assert cli_result.exit_code == 0
    cli_response = json.loads(output_path.read_text())

    server_response = server.execute_synthesis(_dummy_ls(tmp_path), _with_timeout(payload))
    assert cli_response == server_response


# gabion:evidence E:call_footprint::tests/test_cli_server_parity.py::test_refactor_protocol_cli_matches_server::server.py::gabion.server.execute_refactor::test_cli_server_parity.py::tests.test_cli_server_parity._cli_env::test_cli_server_parity.py::tests.test_cli_server_parity._dummy_ls::test_cli_server_parity.py::tests.test_cli_server_parity._has_pygls::test_cli_server_parity.py::tests.test_cli_server_parity._with_timeout
@pytest.mark.skipif(not _has_pygls(), reason="pygls not installed")
def test_refactor_protocol_cli_matches_server(tmp_path: Path) -> None:
    target = tmp_path / "module.py"
    target.write_text("def f(a, b):\n    return a + b\n")
    payload = {
        "protocol_name": "Bundle",
        "bundle": [],
        "fields": [{"name": "a", "type_hint": "int"}, {"name": "b", "type_hint": None}],
        "target_path": str(target),
        "target_functions": [],
        "compatibility_shim": False,
    }
    payload_path = tmp_path / "refactor.json"
    output_path = tmp_path / "refactor-out.json"
    payload_path.write_text(json.dumps(payload))

    runner = CliRunner()
    cli_result = runner.invoke(
        cli.app,
        ["refactor-protocol", "--input", str(payload_path), "--output", str(output_path)],
        env=_cli_env(),
    )
    assert cli_result.exit_code == 0
    cli_response = json.loads(output_path.read_text())

    server_response = server.execute_refactor(_dummy_ls(tmp_path), _with_timeout(payload))
    normalized_server = cli.RefactorProtocolResponseDTO.model_validate(server_response).model_dump()
    assert cli_response == json.loads(json.dumps(normalized_server))


# gabion:evidence E:call_footprint::tests/test_cli_server_parity.py::test_structure_and_decision_diff_cli_match_server::server.py::gabion.server.execute_decision_diff::server.py::gabion.server.execute_structure_diff::test_cli_server_parity.py::tests.test_cli_server_parity._cli_env::test_cli_server_parity.py::tests.test_cli_server_parity._dummy_ls::test_cli_server_parity.py::tests.test_cli_server_parity._has_pygls::test_cli_server_parity.py::tests.test_cli_server_parity._with_timeout
@pytest.mark.skipif(not _has_pygls(), reason="pygls not installed")
def test_structure_and_decision_diff_cli_match_server(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    baseline.write_text(json.dumps({"files": []}))
    current.write_text(json.dumps({"files": [{"functions": [{"bundles": [["x"]]}]}]}))
    runner = CliRunner()

    structure_cli = runner.invoke(
        cli.app,
        [
            "structure-diff",
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--root",
            str(tmp_path),
        ],
        env=_cli_env(),
    )
    assert structure_cli.exit_code == 0
    structure_cli_payload = json.loads(structure_cli.output)
    structure_server_payload = server.execute_structure_diff(
        _dummy_ls(tmp_path),
        _with_timeout({"baseline": str(baseline), "current": str(current)}),
    )
    assert structure_cli_payload == structure_server_payload

    decision_cli = runner.invoke(
        cli.app,
        [
            "decision-diff",
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--root",
            str(tmp_path),
        ],
        env=_cli_env(),
    )
    assert decision_cli.exit_code == 0
    decision_cli_payload = json.loads(decision_cli.output)
    decision_server_payload = server.execute_decision_diff(
        _dummy_ls(tmp_path),
        _with_timeout({"baseline": str(baseline), "current": str(current)}),
    )
    assert decision_cli_payload == decision_server_payload


# gabion:evidence E:call_footprint::tests/test_cli_server_parity.py::test_structure_reuse_cli_matches_server::server.py::gabion.server.execute_structure_reuse::test_cli_server_parity.py::tests.test_cli_server_parity._cli_env::test_cli_server_parity.py::tests.test_cli_server_parity._dummy_ls::test_cli_server_parity.py::tests.test_cli_server_parity._has_pygls::test_cli_server_parity.py::tests.test_cli_server_parity._with_timeout
@pytest.mark.skipif(not _has_pygls(), reason="pygls not installed")
def test_structure_reuse_cli_matches_server(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(
        json.dumps(
            {
                "files": [
                    {"functions": [{"bundles": [["a", "b"]], "body_calls": []}]},
                    {"functions": [{"bundles": [["a", "b"]], "body_calls": []}]},
                ]
            }
        )
    )
    runner = CliRunner()
    cli_result = runner.invoke(
        cli.app,
        [
            "structure-reuse",
            "--snapshot",
            str(snapshot),
            "--min-count",
            "2",
            "--lemma-stubs",
            "-",
            "--root",
            str(tmp_path),
        ],
        env=_cli_env(),
    )
    assert cli_result.exit_code == 0
    cli_payload = json.loads(cli_result.output)

    server_payload = server.execute_structure_reuse(
        _dummy_ls(tmp_path),
        _with_timeout({"snapshot": str(snapshot), "min_count": 2, "lemma_stubs": "-"}),
    )
    assert cli_payload == server_payload


# gabion:evidence E:call_footprint::tests/test_cli_server_parity.py::test_dataflow_run_check_payload_semantics_match_direct_server::cli.py::gabion.cli.build_check_payload::cli.py::gabion.cli.run_check::server.py::gabion.server.execute_command::test_cli_server_parity.py::tests.test_cli_server_parity._dummy_ls::test_cli_server_parity.py::tests.test_cli_server_parity._has_pygls::test_cli_server_parity.py::tests.test_cli_server_parity._with_timeout
@pytest.mark.skipif(not _has_pygls(), reason="pygls not installed")
def test_dataflow_run_check_payload_semantics_match_direct_server(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.write_text("def f(x):\n    return x\n")

    artifact_flags = cli.CheckArtifactFlags(
        emit_test_obsolescence=False,
        emit_test_evidence_suggestions=False,
        emit_call_clusters=False,
        emit_call_cluster_consolidation=False,
        emit_test_annotation_drift=False,
    )
    delta_options = _default_delta_options()

    cli_result = cli.run_check(
        paths=[module],
        report=tmp_path / "report.md",
        policy=cli.CheckPolicyFlags(
            fail_on_violations=False,
            fail_on_type_ambiguities=False,
            lint=False,
        ),
        root=tmp_path,
        config=None,
        baseline=tmp_path / "baseline.json",
        baseline_write=True,
        decision_snapshot=None,
        artifact_flags=artifact_flags,
        delta_options=delta_options,
        exclude=["venv,__pycache__"],
        filter_bundle=cli.DataflowFilterBundle("self,cls", "cache"),
        allow_external=False,
        strictness="high",
        runner=cli.run_command_direct,
    )

    payload = cli.build_check_payload(
        paths=[module],
        report=tmp_path / "report.md",
        fail_on_violations=False,
        root=tmp_path,
        config=None,
        baseline=tmp_path / "baseline.json",
        baseline_write=True,
        decision_snapshot=None,
        artifact_flags=artifact_flags,
        delta_options=delta_options,
        exclude=["venv,__pycache__"],
        filter_bundle=cli.DataflowFilterBundle("self,cls", "cache"),
        allow_external=False,
        strictness="high",
        fail_on_type_ambiguities=False,
        lint=False,
    )
    server_result = server.execute_command(_dummy_ls(tmp_path), _with_timeout(payload))

    keys = ["exit_code", "analysis_state", "errors", "baseline_path"]
    assert {key: cli_result.get(key) for key in keys} == {
        key: server_result.get(key) for key in keys
    }

# gabion:evidence E:call_footprint::tests/test_cli_server_parity.py::test_dataflow_run_check_matches_server_fields::cli.py::gabion.cli.build_check_payload::cli.py::gabion.cli.run_check::server.py::gabion.server.execute_command::test_cli_server_parity.py::tests.test_cli_server_parity._dummy_ls::test_cli_server_parity.py::tests.test_cli_server_parity._with_timeout
def test_dataflow_run_check_matches_server_fields(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.write_text(
        "def callee_int(x: int):\n"
        "    return x\n"
        "def callee_str(x: str):\n"
        "    return x\n"
        "def caller(a):\n"
        "    callee_int(a)\n"
        "    callee_str(a)\n"
    )
    artifact_flags = cli.CheckArtifactFlags(
        emit_test_obsolescence=False,
        emit_test_evidence_suggestions=False,
        emit_call_clusters=False,
        emit_call_cluster_consolidation=False,
        emit_test_annotation_drift=False,
    )
    cli_result = cli.run_check(
        paths=[module],
        report=None,
        policy=cli.CheckPolicyFlags(
            fail_on_violations=False,
            fail_on_type_ambiguities=False,
            lint=True,
        ),
        root=tmp_path,
        config=None,
        baseline=None,
        baseline_write=False,
        decision_snapshot=None,
        artifact_flags=artifact_flags,
        delta_options=_default_delta_options(),
        exclude=None,
        filter_bundle=cli.DataflowFilterBundle(None, None),
        allow_external=None,
        strictness="high",
        runner=cli.run_command_direct,
    )
    payload = cli.build_check_payload(
        paths=[module],
        report=tmp_path / "artifacts/audit_reports/dataflow_report.md",
        fail_on_violations=False,
        root=tmp_path,
        config=None,
        baseline=None,
        baseline_write=False,
        decision_snapshot=None,
        artifact_flags=artifact_flags,
        delta_options=_default_delta_options(),
        exclude=None,
        filter_bundle=cli.DataflowFilterBundle(None, None),
        allow_external=None,
        strictness="high",
        fail_on_type_ambiguities=False,
        lint=True,
    )
    server_result = server.execute_command(_dummy_ls(tmp_path), _with_timeout(payload))
    keys = ["exit_code", "analysis_state", "lint_lines", "lint_entries"]
    assert {key: cli_result.get(key) for key in keys} == {
        key: server_result.get(key) for key in keys
    }
