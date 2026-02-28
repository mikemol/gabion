from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from gabion.commands import transport_policy
from gabion import cli
from gabion.runtime import env_policy


def _has_pygls() -> bool:
    return importlib.util.find_spec("pygls") is not None


def _cli_env() -> dict[str, str]:
    return {
        **os.environ,
        "GABION_DIRECT_RUN": "1",
    }


def _invoke(runner: CliRunner, args: list[str], *, input_text: str | None = None):
    argv = list(args)
    if "--timeout" not in argv:
        argv = ["--timeout", "50000000000ns", *argv]
    return runner.invoke(cli.app, argv, env=_cli_env(), input=input_text)


# gabion:evidence E:call_footprint::tests/test_cli_commands.py::test_cli_help_lists_tooling_subcommands::cli.py::gabion.cli.app
def test_cli_help_lists_tooling_subcommands() -> None:
    runner = CliRunner()
    result = _invoke(runner, ["--help"])
    assert result.exit_code == 0
    assert "docflow-delta-emit" in result.output
    assert "ambiguity-contract-gate" in result.output
    assert "normative-symdiff" in result.output
    assert "impact-select-tests" in result.output
    assert "run-dataflow-stage" in result.output
    assert "ci-watch" in result.output


# gabion:evidence E:call_footprint::tests/test_cli_commands.py::test_cli_tooling_subcommand_help_invocations::cli.py::gabion.cli.app
def test_cli_tooling_subcommand_help_invocations() -> None:
    runner = CliRunner()
    for command_name in (
        "docflow-delta-emit",
        "ambiguity-contract-gate",
        "normative-symdiff",
        "impact-select-tests",
        "run-dataflow-stage",
        "ci-watch",
    ):
        result = _invoke(runner, [command_name, "--help"])
        assert result.exit_code == 0, command_name


# gabion:evidence E:call_footprint::tests/test_cli_commands.py::test_cli_tooling_wrappers_and_argparse_exit_handling::cli.py::gabion.cli._invoke_argparse_command::cli.py::gabion.cli.docflow_delta_emit::cli.py::gabion.cli.ambiguity_contract_gate::cli.py::gabion.cli.impact_select_tests::cli.py::gabion.cli.run_dataflow_stage
def test_cli_tooling_wrappers_and_argparse_exit_handling() -> None:
    assert cli._invoke_argparse_command(lambda _argv: 3, []) == 3
    assert (
        cli._invoke_argparse_command(
            lambda _argv: (_ for _ in ()).throw(SystemExit(4)),
            [],
        )
        == 4
    )
    assert (
        cli._invoke_argparse_command(
            lambda _argv: (_ for _ in ()).throw(SystemExit("bad")),
            [],
        )
        == 1
    )

    class _Ctx:
        def __init__(self, args: list[str]) -> None:
            self.args = args

    argv_seen: list[list[str]] = []
    ambiguity_args: list[list[str]] = []
    symdiff_args: list[list[str]] = []
    ci_watch_args: list[list[str]] = []
    with cli._tooling_runner_override(
        no_arg={
            "docflow-delta-emit": lambda: 13,
        },
        with_argv={
            "ci-watch": lambda argv: (ci_watch_args.append(list(argv or [])) or 18),
            "ambiguity-contract-gate": lambda argv: (ambiguity_args.append(list(argv or [])) or 16),
            "normative-symdiff": lambda argv: (symdiff_args.append(list(argv or [])) or 17),
            "impact-select-tests": lambda argv: (argv_seen.append(list(argv or [])) or 14),
            "run-dataflow-stage": lambda argv: (_ for _ in ()).throw(SystemExit(15)),
        },
    ):
        with pytest.raises(typer.Exit) as exc:
            cli.docflow_delta_emit()
        assert exc.value.exit_code == 13
        with pytest.raises(typer.Exit) as exc:
            cli.ambiguity_contract_gate(_Ctx(["--root", ".", "--baseline", "b.json"]))  # type: ignore[arg-type]
        assert exc.value.exit_code == 16
        with pytest.raises(typer.Exit) as exc:
            cli.normative_symdiff(_Ctx(["--root", ".", "--json-out", "out.json"]))  # type: ignore[arg-type]
        assert exc.value.exit_code == 17
        with pytest.raises(typer.Exit) as exc:
            cli.impact_select_tests(_Ctx(["--root", "."]))  # type: ignore[arg-type]
        assert exc.value.exit_code == 14
        with pytest.raises(typer.Exit) as exc:
            cli.run_dataflow_stage(_Ctx(["--stage-id", "run"]))  # type: ignore[arg-type]
        assert exc.value.exit_code == 15
        with pytest.raises(typer.Exit) as exc:
            cli.ci_watch(
                run_id="123",
                branch="stage",
                workflow="ci",
                status=None,
                prefer_active=False,
                download_artifacts_on_failure=True,
                artifact_output_root=Path("artifacts/out/ci_watch"),
                artifact_name=["test-runs"],
                collect_failed_logs=False,
                summary_json=Path("artifacts/out/ci_watch_summary.json"),
            )
        assert exc.value.exit_code == 18

    assert argv_seen == [["--root", "."]]
    assert ambiguity_args == [["--root", ".", "--baseline", "b.json"]]
    assert symdiff_args == [["--root", ".", "--json-out", "out.json"]]
    assert ci_watch_args == [
        [
            "--branch",
            "stage",
            "--run-id",
            "123",
            "--workflow",
            "ci",
            "--no-prefer-active",
            "--download-artifacts-on-failure",
            "--artifact-output-root",
            "artifacts/out/ci_watch",
            "--artifact-name",
            "test-runs",
            "--no-collect-failed-logs",
            "--summary-json",
            "artifacts/out/ci_watch_summary.json",
        ]
    ]


# gabion:evidence E:function_site::tests/test_cli_commands.py::test_removed_delta_wrapper_commands_emit_migration_errors
def test_removed_delta_wrapper_commands_emit_migration_errors() -> None:
    runner = CliRunner()
    removed_emit = _invoke(runner, ["delta-state-emit"])
    assert removed_emit.exit_code != 0
    assert "delta-bundle" in removed_emit.output
    removed_triplets = _invoke(runner, ["delta-triplets"])
    assert removed_triplets.exit_code != 0
    assert "delta-gates" in removed_triplets.output


def test_delta_advisory_telemetry_command_forwards_exit_code() -> None:
    with cli._tooling_runner_override(
        no_arg={"delta-advisory-telemetry": lambda: 7},
    ):
        with pytest.raises(typer.Exit) as exc:
            cli.delta_advisory_telemetry()
    assert exc.value.exit_code == 7


# gabion:evidence E:call_footprint::tests/test_cli_commands.py::test_tooling_runner_override_ignores_non_mapping_overrides::cli.py::gabion.cli._tooling_runner_override
def test_tooling_runner_override_ignores_non_mapping_overrides() -> None:
    no_arg_before = dict(cli._TOOLING_NO_ARG_RUNNERS)
    with_argv_before = dict(cli._TOOLING_ARGV_RUNNERS)
    with cli._tooling_runner_override(
        no_arg=[],  # type: ignore[arg-type]
        with_argv=[],  # type: ignore[arg-type]
    ):
        assert dict(cli._TOOLING_NO_ARG_RUNNERS) == no_arg_before
        assert dict(cli._TOOLING_ARGV_RUNNERS) == with_argv_before
    assert dict(cli._TOOLING_NO_ARG_RUNNERS) == no_arg_before
    assert dict(cli._TOOLING_ARGV_RUNNERS) == with_argv_before


# gabion:evidence E:function_site::test_cli_commands.py::tests.test_cli_commands.test_configure_runtime_flags_maps_transport_mode_to_direct_requested
def test_configure_runtime_flags_maps_transport_mode_to_direct_requested() -> None:
    timeout_before = env_policy.lsp_timeout_override()
    transport_before = transport_policy.transport_override()
    try:
        cli.configure_runtime_flags(
            timeout="200ns",
            carrier=cli.CliTransportMode.direct,
            carrier_override_record=Path("overrides/record.json"),
            removed_lsp_timeout_ticks=None,
            removed_lsp_timeout_tick_ns=None,
            removed_lsp_timeout_ms=None,
            removed_lsp_timeout_seconds=None,
            removed_transport=None,
            removed_direct_run_override_evidence=None,
            removed_override_record_json=None,
        )
        timeout_override = env_policy.lsp_timeout_override()
        transport_override = transport_policy.transport_override()
        assert timeout_override is not None
        assert timeout_override.ticks == 1
        assert timeout_override.tick_ns == 1_000_000
        assert transport_override is not None
        assert transport_override.direct_requested is True
        assert transport_override.override_record_path == "overrides/record.json"
        assert transport_override.override_record_json is None
    finally:
        env_policy.set_lsp_timeout_override(timeout_before)
        transport_policy.set_transport_override(transport_before)


# gabion:evidence E:function_site::test_cli_commands.py::tests.test_cli_commands._has_pygls E:decision_surface/direct::test_cli_commands.py::tests.test_cli_commands._has_pygls::stale_fe77309ae8a6_eaff81bd
@pytest.mark.skipif(not _has_pygls(), reason="pygls not installed")
def test_cli_check_and_dataflow_audit(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.write_text(
        "def callee_int(x: int):\n"
        "    return x\n"
        "def callee_str(x: str):\n"
        "    return x\n"
        "def caller_single(a):\n"
        "    return callee_int(a)\n"
        "def caller_multi(b):\n"
        "    callee_int(b)\n"
        "    callee_str(b)\n"
    )
    runner = CliRunner()
    result = _invoke(
        runner,
        [
            "--carrier",
            "lsp",
            "check",
            "run",
            str(module),
            "--root",
            str(tmp_path),
            "--gate",
            "none",
        ],
    )
    assert result.exit_code == 0

    result = _invoke(
        runner,
        [
            "--carrier",
            "lsp",
            "check",
            "raw",
            "--",
            str(module),
            "--root",
            str(tmp_path),
            "--type-audit",
            "--type-audit-max",
            "10",
            "--dot",
            "-",
            "--synthesis-plan",
            "-",
            "--synthesis-protocols",
            "-",
            "--synthesis-min-bundle-size",
            "1",
            "--synthesis-allow-singletons",
            "--refactor-plan-json",
            "-",
        ],
    )
    assert result.exit_code == 0
    assert "Type tightening candidates" in result.output
    assert "Type ambiguities" in result.output


# gabion:evidence E:function_site::test_cli_commands.py::tests.test_cli_commands._has_pygls E:decision_surface/direct::test_cli_commands.py::tests.test_cli_commands._has_pygls::stale_0f3883f99dda
@pytest.mark.skipif(not _has_pygls(), reason="pygls not installed")
def test_cli_impact_json(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (src / "sample.py").write_text(
        "def target(v):\n"
        "    return v\n\n"
        "def bridge(v):\n"
        "    return target(v)\n"
    )
    (tests_dir / "test_sample.py").write_text(
        "from src.sample import bridge\n\n"
        "def test_bridge():\n"
        "    assert bridge(1) == 1\n"
    )
    runner = CliRunner()
    result = _invoke(
        runner,
        [
            "impact",
            "--change",
            "src/sample.py:1-2",
            "--root",
            str(tmp_path),
            "--json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert "must_run_tests" in payload


# gabion:evidence E:call_footprint::tests/test_cli_commands.py::test_cli_impact_reads_git_diff_file::test_cli_commands.py::tests.test_cli_commands._has_pygls::test_cli_commands.py::tests.test_cli_commands._invoke
@pytest.mark.skipif(not _has_pygls(), reason="pygls not installed")
def test_cli_impact_reads_git_diff_file(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.write_text("def f():\n    return 1\n")
    diff_path = tmp_path / "changes.diff"
    diff_path.write_text(
        "diff --git a/module.py b/module.py\n"
        "+++ b/module.py\n"
        "@@ -1,1 +1,2 @@\n"
        " def f():\n"
        "+    return 2\n"
    )

    runner = CliRunner()
    result = _invoke(
        runner,
        [
            "impact",
            "--git-diff",
            str(diff_path),
            "--root",
            str(tmp_path),
            "--json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["changes"][0]["path"] == "module.py"


# gabion:evidence E:call_footprint::tests/test_cli_commands.py::test_cli_impact_reads_git_diff_stdin::test_cli_commands.py::tests.test_cli_commands._has_pygls::test_cli_commands.py::tests.test_cli_commands._invoke
@pytest.mark.skipif(not _has_pygls(), reason="pygls not installed")
def test_cli_impact_reads_git_diff_stdin(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.write_text("def f():\n    return 1\n")
    diff_text = (
        "diff --git a/module.py b/module.py\n"
        "+++ b/module.py\n"
        "@@ -1,1 +1,2 @@\n"
        " def f():\n"
        "+    return 2\n"
    )

    runner = CliRunner()
    result = _invoke(
        runner,
        [
            "impact",
            "--git-diff",
            "-",
            "--root",
            str(tmp_path),
            "--json",
        ],
        input_text=diff_text,
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["changes"][0]["path"] == "module.py"


# gabion:evidence E:call_footprint::tests/test_cli_commands.py::test_cli_docflow_audit::cli.py::gabion.cli.app
def test_cli_docflow() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runner = CliRunner()
    result = _invoke(
        runner,
        [
            "docflow",
            "--root",
            str(repo_root),
            "--no-fail-on-violations",
        ],
    )
    assert result.exit_code == 0


# gabion:evidence E:call_footprint::tests/test_cli_commands.py::test_cli_dataflow_audit_requires_paths::cli.py::gabion.cli.app


# gabion:evidence E:call_footprint::tests/test_cli_commands.py::test_cli_sppf_graph_and_status_consistency::cli.py::gabion.cli.app
def test_cli_sppf_graph_and_status_consistency(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    graph_json = tmp_path / "graph.json"
    status_json = tmp_path / "status.json"

    runner = CliRunner()
    graph_result = _invoke(
        runner,
        [
            "sppf-graph",
            "--root",
            str(repo_root),
            "--json-output",
            str(graph_json),
        ],
    )
    assert graph_result.exit_code == 0
    assert graph_json.exists()

    status_result = _invoke(
        runner,
        [
            "status-consistency",
            "--root",
            str(repo_root),
            "--json-output",
            str(status_json),
            "--no-fail-on-violations",
        ],
    )
    assert status_result.exit_code == 0
    assert status_json.exists()


# gabion:evidence E:call_footprint::tests/test_cli_commands.py::test_cli_dataflow_audit_command_is_removed::test_cli_commands.py::tests.test_cli_commands._invoke
def test_cli_dataflow_audit_command_is_removed() -> None:
    runner = CliRunner()
    result = _invoke(runner, ["dataflow-audit", "--help"])
    assert result.exit_code != 0


# gabion:evidence E:function_site::test_cli_commands.py::tests.test_cli_commands._has_pygls E:decision_surface/direct::test_cli_commands.py::tests.test_cli_commands._has_pygls::stale_eb2b6007df89
@pytest.mark.skipif(not _has_pygls(), reason="pygls not installed")
def test_cli_synth_and_synthesis_plan(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.write_text(
        "def callee(x, y):\n"
        "    return x, y\n"
        "def caller(a, b):\n"
        "    return callee(a, b)\n"
    )
    out_dir = tmp_path / "out"
    runner = CliRunner()
    result = _invoke(
        runner,
        [
            "synth",
            str(module),
            "--root",
            str(tmp_path),
            "--out-dir",
            str(out_dir),
            "--synthesis-min-bundle-size",
            "1",
            "--synthesis-allow-singletons",
        ],
    )
    assert result.exit_code == 0
    assert "Snapshot:" in result.output


    result = _invoke(
        runner,
        [
            "synth",
            str(module),
            "--root",
            str(tmp_path),
            "--no-timestamp",
            "--no-refactor-plan",
            "--synthesis-min-bundle-size",
            "1",
            "--synthesis-allow-singletons",
        ],
    )
    assert result.exit_code == 0
    assert "Snapshot:" not in result.output

    payload_path = tmp_path / "synth.json"
    payload_path.write_text(
        '{"bundles":[{"bundle":["x"],"tier":2}],"allow_singletons":true,"min_bundle_size":1}'
    )
    output_path = tmp_path / "synth_out.json"
    result = _invoke(
        runner,
        [
            "synthesis-plan",
            "--input",
            str(payload_path),
            "--output",
            str(output_path),
        ],
    )
    assert result.exit_code == 0
    assert output_path.exists()


# gabion:evidence E:function_site::test_cli_commands.py::tests.test_cli_commands._has_pygls E:decision_surface/direct::test_cli_commands.py::tests.test_cli_commands._has_pygls::stale_daabf20e679c
@pytest.mark.skipif(not _has_pygls(), reason="pygls not installed")
def test_cli_structure_diff(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    baseline.write_text(json.dumps({"files": []}))
    current.write_text(json.dumps({"files": [{"functions": [{"bundles": [["a"]]}]}]}))
    runner = CliRunner()
    result = _invoke(
        runner,
        [
            "structure-diff",
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--root",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0
    assert "\"diff\"" in result.output


# gabion:evidence E:function_site::test_cli_commands.py::tests.test_cli_commands._has_pygls E:decision_surface/direct::test_cli_commands.py::tests.test_cli_commands._has_pygls::stale_71868ab0baee
@pytest.mark.skipif(not _has_pygls(), reason="pygls not installed")
def test_cli_refactor_protocol(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.write_text("def f(a, b):\n    return a + b\n")
    runner = CliRunner()
    result = _invoke(
        runner,
        [
            "refactor-protocol",
            "--protocol-name",
            "Bundle",
            "--bundle",
            "a",
            "--bundle",
            "b",
            "--target-path",
            str(module),
        ],
    )
    assert result.exit_code == 0


# gabion:evidence E:call_footprint::tests/test_cli_commands.py::test_cli_synthesis_plan_invalid_json::cli.py::gabion.cli.app
def test_cli_synthesis_plan_invalid_json(tmp_path: Path) -> None:
    payload_path = tmp_path / "bad.json"
    payload_path.write_text("{bad")
    runner = CliRunner()
    result = _invoke(runner, ["synthesis-plan", "--input", str(payload_path)])
    assert result.exit_code != 0
    assert "Invalid JSON payload" in result.output


# gabion:evidence E:call_footprint::tests/test_cli_commands.py::test_cli_refactor_protocol_invalid_json::cli.py::gabion.cli.app
def test_cli_refactor_protocol_invalid_json(tmp_path: Path) -> None:
    payload_path = tmp_path / "bad.json"
    payload_path.write_text("{bad")
    runner = CliRunner()
    result = _invoke(
        runner,
        [
            "refactor-protocol",
            "--input",
            str(payload_path),
        ],
    )
    assert result.exit_code != 0
    assert "Invalid JSON payload" in result.output


# gabion:evidence E:function_site::test_cli_commands.py::tests.test_cli_commands._has_pygls E:decision_surface/direct::test_cli_commands.py::tests.test_cli_commands._has_pygls::stale_b8d0dddbecae
@pytest.mark.skipif(not _has_pygls(), reason="pygls not installed")
def test_cli_synthesis_plan_stdout(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.json"
    payload_path.write_text('{"bundles":[{"bundle":["x"],"tier":2}]}')
    runner = CliRunner()
    result = _invoke(
        runner,
        [
            "synthesis-plan",
            "--input",
            str(payload_path),
        ],
    )
    assert result.exit_code == 0
    assert result.output.strip().startswith("{")


# gabion:evidence E:function_site::test_cli_commands.py::tests.test_cli_commands._has_pygls E:decision_surface/direct::test_cli_commands.py::tests.test_cli_commands._has_pygls::stale_530c5ee02284
@pytest.mark.skipif(not _has_pygls(), reason="pygls not installed")
def test_cli_refactor_protocol_output_file(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.write_text("def f(a, b):\n    return a + b\n")
    out_path = tmp_path / "out.json"
    runner = CliRunner()
    result = _invoke(
        runner,
        [
            "refactor-protocol",
            "--protocol-name",
            "Bundle",
            "--bundle",
            "a",
            "--bundle",
            "b",
            "--target-path",
            str(module),
            "--output",
            str(out_path),
        ],
    )
    assert result.exit_code == 0
    assert out_path.exists()


# gabion:evidence E:call_footprint::tests/test_cli_commands.py::test_cli_synth_invalid_strictness::cli.py::gabion.cli.app
def test_cli_synth_invalid_strictness(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.write_text("def f(a, b):\n    return a\n")
    runner = CliRunner()
    result = _invoke(
        runner,
        ["synth", str(module), "--root", str(tmp_path), "--strictness", "weird"],
    )
    assert result.exit_code != 0
    assert "strictness" in result.output


# gabion:evidence E:call_footprint::tests/test_cli_commands.py::test_cli_synth_invalid_protocols_kind::cli.py::gabion.cli.app
def test_cli_synth_invalid_protocols_kind(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.write_text("def f(a, b):\n    return a\n")
    runner = CliRunner()
    result = _invoke(
        runner,
        [
            "synth",
            str(module),
            "--root",
            str(tmp_path),
            "--synthesis-protocols-kind",
            "nope",
        ],
    )
    assert result.exit_code != 0
    assert "synthesis-protocols-kind" in result.output

# gabion:evidence E:call_footprint::tests/test_cli_commands.py::test_cli_refactor_protocol_emits_rewrite_plan_metadata::cli.py::gabion.cli.app
@pytest.mark.skipif(not _has_pygls(), reason="pygls not installed")
def test_cli_refactor_protocol_emits_rewrite_plan_metadata(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.write_text(
        "def sink(ctx):\n"
        "    return ctx\n\n"
        "def caller(ctx):\n"
        "    return sink(ctx)\n"
    )
    runner = CliRunner()
    result = _invoke(
        runner,
        [
            "refactor-protocol",
            "--protocol-name",
            "CtxBundle",
            "--bundle",
            "ctx",
            "--target-path",
            str(module),
            "--target-function",
            "sink",
            "--target-function",
            "caller",
            "--ambient-rewrite",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["rewrite_plans"]
    assert payload["rewrite_plans"][0]["kind"] == "AMBIENT_REWRITE"
