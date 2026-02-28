from __future__ import annotations

from pathlib import Path
import re

import pytest
import typer
from typer.testing import CliRunner

from gabion import cli
from gabion.exceptions import NeverThrown


_ANSI_ESCAPE = re.compile(r"\x1B\[[0-9;]*m")


def _normalize_output(text: str) -> str:
    return " ".join(_ANSI_ESCAPE.sub("", text).split())


def _check_obj(
    captured: list[dict[str, object]],
    *,
    run_check_delta_gates: object | None = None,
) -> dict[str, object]:
    def _run_check(**kwargs: object) -> dict[str, object]:
        captured.append({str(key): kwargs[key] for key in kwargs})
        return {"exit_code": 0, "lint_lines": []}

    obj: dict[str, object] = {
        "run_check": _run_check,
    }
    if run_check_delta_gates is not None:
        obj["run_check_delta_gates"] = run_check_delta_gates
    return obj


# gabion:evidence E:function_site::tests/test_cli_check_surface_edges.py::test_configure_runtime_flags_rejects_removed_timeout_and_transport_flags
def test_configure_runtime_flags_rejects_removed_timeout_and_transport_flags() -> None:
    with pytest.raises(typer.BadParameter):
        cli.configure_runtime_flags(
            timeout=None,
            carrier=None,
            carrier_override_record=None,
            removed_lsp_timeout_ticks=1,
            removed_lsp_timeout_tick_ns=None,
            removed_lsp_timeout_ms=None,
            removed_lsp_timeout_seconds=None,
            removed_transport=None,
            removed_direct_run_override_evidence=None,
            removed_override_record_json=None,
        )
    with pytest.raises(typer.BadParameter):
        cli.configure_runtime_flags(
            timeout=None,
            carrier=None,
            carrier_override_record=None,
            removed_lsp_timeout_ticks=None,
            removed_lsp_timeout_tick_ns=None,
            removed_lsp_timeout_ms=None,
            removed_lsp_timeout_seconds=None,
            removed_transport="direct",
            removed_direct_run_override_evidence=None,
            removed_override_record_json=None,
        )


# gabion:evidence E:function_site::tests/test_cli_check_surface_edges.py::test_check_group_and_subgroups_require_explicit_subcommand
def test_check_group_and_subgroups_require_explicit_subcommand() -> None:
    runner = CliRunner()
    for argv in (
        ["check"],
        ["check", "obsolescence"],
        ["check", "annotation-drift"],
        ["check", "ambiguity"],
        ["check", "taint"],
    ):
        result = runner.invoke(cli.app, argv)
        assert result.exit_code == 2
        assert "Usage:" in result.output


# gabion:evidence E:function_site::tests/test_cli_check_surface_edges.py::test_check_raw_usage_and_passthrough_branches
def test_check_raw_usage_and_passthrough_branches() -> None:
    runner = CliRunner()
    bad = runner.invoke(cli.app, ["check", "raw"])
    assert bad.exit_code != 0
    normalized = _normalize_output(bad.output)
    assert "Usage: gabion check raw -- [raw-args...]" in normalized

    captured: list[list[str]] = []

    def _fake_raw(argv: list[str]) -> None:
        captured.append(list(argv))

    ok = runner.invoke(
        cli.app,
        ["check", "raw", "sample.py", "--dot", "-"],
        obj={"run_dataflow_raw_argv": _fake_raw},
    )
    assert ok.exit_code == 0
    assert captured == [["sample.py", "--dot", "-"]]


# gabion:evidence E:function_site::tests/test_cli_check_surface_edges.py::test_check_raw_direct_helper_strips_leading_double_dash
def test_check_raw_direct_helper_strips_leading_double_dash() -> None:
    captured: list[list[str]] = []

    class _Ctx:
        args = ["--", "sample.py", "--dot", "-"]
        obj = {"run_dataflow_raw_argv": lambda argv: captured.append(list(argv))}

    cli.check_raw(_Ctx())  # type: ignore[arg-type]
    assert captured == [["sample.py", "--dot", "-"]]


# gabion:evidence E:function_site::tests/test_cli_check_surface_edges.py::test_check_run_removed_and_invalid_baseline_mode_paths
def test_check_run_removed_and_invalid_baseline_mode_paths() -> None:
    runner = CliRunner()
    removed = runner.invoke(
        cli.app,
        ["check", "run", "sample.py", "--analysis-tick-limit", "10"],
    )
    assert removed.exit_code != 0
    assert "Removed --analysis-tick-limit" in _normalize_output(removed.output)

    missing = runner.invoke(
        cli.app,
        ["check", "run", "sample.py", "--baseline-mode", "enforce"],
    )
    assert missing.exit_code != 0
    assert "--baseline is required" in _normalize_output(missing.output)

    invalid = runner.invoke(
        cli.app,
        [
            "check",
            "run",
            "sample.py",
            "--baseline-mode",
            "off",
            "--baseline",
            "baseline.json",
        ],
    )
    assert invalid.exit_code != 0
    assert "--baseline is only valid" in _normalize_output(invalid.output)

    removed_resume = runner.invoke(
        cli.app,
        ["check", "run", "sample.py", "--resume-checkpoint", "resume.json"],
    )
    assert removed_resume.exit_code != 0
    assert "No such option: --resume-checkpoint" in _normalize_output(
        removed_resume.output
    )

    removed_progress = runner.invoke(
        cli.app,
        ["check", "run", "sample.py", "--emit-timeout-progress-report"],
    )
    assert removed_progress.exit_code != 0
    assert "No such option: --emit-timeout-progress-report" in _normalize_output(
        removed_progress.output
    )

    removed_retry = runner.invoke(
        cli.app,
        ["check", "run", "sample.py", "--resume-on-timeout", "2"],
    )
    assert removed_retry.exit_code != 0
    assert "No such option: --resume-on-timeout" in _normalize_output(
        removed_retry.output
    )


# gabion:evidence E:function_site::tests/test_cli_check_surface_edges.py::test_check_delta_bundle_dispatches_single_pass_delta_options
def test_check_delta_bundle_dispatches_single_pass_delta_options() -> None:
    runner = CliRunner()
    captured: list[dict[str, object]] = []
    result = runner.invoke(
        cli.app,
        ["check", "delta-bundle", "sample.py"],
        obj=_check_obj(captured),
    )
    assert result.exit_code == 0
    assert len(captured) == 1
    kwargs = captured[0]
    delta_options = kwargs["delta_options"]
    assert isinstance(delta_options, cli.CheckDeltaOptions)
    assert delta_options.emit_test_obsolescence_state is True
    assert delta_options.emit_test_obsolescence_delta is True
    assert delta_options.emit_test_annotation_drift_delta is True
    assert delta_options.emit_ambiguity_state is True
    assert delta_options.emit_ambiguity_delta is True


# gabion:evidence E:function_site::tests/test_cli_check_surface_edges.py::test_check_delta_gates_uses_gate_runner_exit_code
def test_check_delta_gates_uses_gate_runner_exit_code() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["check", "delta-gates"],
        obj=_check_obj([], run_check_delta_gates=lambda: 7),
    )
    assert result.exit_code == 7


# gabion:evidence E:function_site::tests/test_cli_check_surface_edges.py::test_check_lint_mode_validation_errors
def test_check_lint_mode_validation_errors() -> None:
    runner = CliRunner()
    jsonl_missing = runner.invoke(
        cli.app,
        ["check", "run", "sample.py", "--gate", "none", "--lint", "jsonl"],
    )
    assert jsonl_missing.exit_code != 0
    assert "--lint-jsonl-out is required" in _normalize_output(jsonl_missing.output)

    sarif_missing = runner.invoke(
        cli.app,
        ["check", "run", "sample.py", "--gate", "none", "--lint", "sarif"],
    )
    assert sarif_missing.exit_code != 0
    assert "--lint-sarif-out is required" in _normalize_output(sarif_missing.output)

    jsonl_invalid = runner.invoke(
        cli.app,
        [
            "check",
            "run",
            "sample.py",
            "--gate",
            "none",
            "--lint",
            "none",
            "--lint-jsonl-out",
            "lint.jsonl",
        ],
    )
    assert jsonl_invalid.exit_code != 0
    assert "--lint-jsonl-out is only valid" in _normalize_output(jsonl_invalid.output)

    sarif_invalid = runner.invoke(
        cli.app,
        [
            "check",
            "run",
            "sample.py",
            "--gate",
            "none",
            "--lint",
            "line",
            "--lint-sarif-out",
            "lint.sarif",
        ],
    )
    assert sarif_invalid.exit_code != 0
    assert "--lint-sarif-out is only valid" in _normalize_output(sarif_invalid.output)


# gabion:evidence E:function_site::tests/test_cli_check_surface_edges.py::test_check_gate_policy_all_modes_and_invalid_value
def test_check_gate_policy_all_modes_and_invalid_value() -> None:
    assert cli._check_gate_policy(cli.CheckGateMode.all) == (True, True)
    assert cli._check_gate_policy(cli.CheckGateMode.none) == (False, False)
    assert cli._check_gate_policy(cli.CheckGateMode.violations) == (True, False)
    assert cli._check_gate_policy(cli.CheckGateMode.type_ambiguities) == (False, True)
    with pytest.raises(NeverThrown):
        cli._check_gate_policy("invalid")  # type: ignore[arg-type]


# gabion:evidence E:function_site::tests/test_cli_check_surface_edges.py::test_raw_profile_helper_functions_cover_commandline_source_branches
def test_raw_profile_helper_functions_cover_commandline_source_branches() -> None:
    class _Ctx:
        def __init__(self, *, commandline_params: set[str], args: list[str]) -> None:
            self._commandline_params = commandline_params
            self.args = args

        def get_parameter_source(self, param: str):
            if param in self._commandline_params:
                return cli.ParameterSource.COMMANDLINE
            return cli.ParameterSource.DEFAULT

    ctx = _Ctx(
        commandline_params={
            "root",
            "config",
            "report",
            "decision_snapshot",
            "baseline",
            "baseline_write",
            "exclude",
            "ignore_params_csv",
            "transparent_decorators_csv",
            "allow_external",
            "strictness",
            "fail_on_violations",
            "fail_on_type_ambiguities",
            "lint",
            "lint_jsonl",
            "lint_sarif",
            "emit_test_obsolescence",
        },
        args=["--dot", "-"],
    )
    assert cli._param_is_command_line(ctx, "root") is True  # type: ignore[arg-type]
    assert cli._param_is_command_line(ctx, "missing") is False  # type: ignore[arg-type]
    unsupported = cli._raw_profile_unsupported_flags(ctx)  # type: ignore[arg-type]
    assert "--emit-test-obsolescence" in unsupported

    argv = cli._check_raw_profile_args(
        ctx=ctx,  # type: ignore[arg-type]
        paths=[Path("sample.py")],
        report=Path("report.md"),
        fail_on_violations=True,
        root=Path("."),
        config=Path("cfg.toml"),
        decision_snapshot=Path("decision.json"),
        baseline=Path("baseline.json"),
        baseline_write=True,
        exclude=["a,b"],
        filter_bundle=cli.DataflowFilterBundle(
            ignore_params_csv="x,y",
            transparent_decorators_csv="deco",
        ),
        allow_external=False,
        strictness="low",
        fail_on_type_ambiguities=True,
        lint=True,
        lint_jsonl=Path("lint.jsonl"),
        lint_sarif=Path("lint.sarif"),
    )
    assert argv[:2] == ["sample.py", "--root"]
    assert "--lint-sarif" in argv

    with pytest.raises(typer.BadParameter):
        cli._run_check_raw_profile(
            ctx=ctx,  # type: ignore[arg-type]
            paths=[Path("sample.py")],
            report=Path("report.md"),
            fail_on_violations=True,
            root=Path("."),
            config=Path("cfg.toml"),
            decision_snapshot=Path("decision.json"),
            baseline=Path("baseline.json"),
            baseline_write=True,
            exclude=["a,b"],
            filter_bundle=cli.DataflowFilterBundle(
                ignore_params_csv="x,y",
                transparent_decorators_csv="deco",
            ),
            allow_external=False,
            strictness="low",
            fail_on_type_ambiguities=True,
            lint=True,
            lint_jsonl=Path("lint.jsonl"),
            lint_sarif=Path("lint.sarif"),
            run_dataflow_raw_argv_fn=lambda _argv: None,
        )

    ctx_ok = _Ctx(commandline_params=set(), args=["--type-audit"])
    captured: list[list[str]] = []
    cli._run_check_raw_profile(
        ctx=ctx_ok,  # type: ignore[arg-type]
        paths=[Path("sample.py")],
        report=None,
        fail_on_violations=False,
        root=Path("."),
        config=None,
        decision_snapshot=None,
        baseline=None,
        baseline_write=False,
        exclude=None,
        filter_bundle=None,
        allow_external=None,
        strictness=None,
        fail_on_type_ambiguities=False,
        lint=False,
        lint_jsonl=None,
        lint_sarif=None,
        run_dataflow_raw_argv_fn=lambda raw_argv: captured.append(list(raw_argv)),
    )
    assert captured == [["sample.py", "--type-audit"]]

# gabion:evidence E:function_site::tests/test_cli_check_surface_edges.py::test_check_aux_subcommands_forward_domain_and_action
@pytest.mark.parametrize(
    ("argv_suffix", "domain", "action"),
    [
        (["obsolescence", "report", "sample.py"], "obsolescence", "report"),
        (["obsolescence", "state", "sample.py"], "obsolescence", "state"),
        (
            ["obsolescence", "delta", "sample.py", "--baseline", "obs.json"],
            "obsolescence",
            "delta",
        ),
        (
            ["obsolescence", "baseline-write", "sample.py", "--baseline", "obs.json"],
            "obsolescence",
            "baseline-write",
        ),
        (
            ["annotation-drift", "report", "sample.py"],
            "annotation-drift",
            "report",
        ),
        (
            ["annotation-drift", "state", "sample.py"],
            "annotation-drift",
            "state",
        ),
        (
            ["annotation-drift", "delta", "sample.py", "--baseline", "drift.json"],
            "annotation-drift",
            "delta",
        ),
        (
            [
                "annotation-drift",
                "baseline-write",
                "sample.py",
                "--baseline",
                "drift.json",
            ],
            "annotation-drift",
            "baseline-write",
        ),
        (["ambiguity", "state", "sample.py"], "ambiguity", "state"),
        (["ambiguity", "delta", "sample.py", "--baseline", "amb.json"], "ambiguity", "delta"),
        (
            ["ambiguity", "baseline-write", "sample.py", "--baseline", "amb.json"],
            "ambiguity",
            "baseline-write",
        ),
        (["taint", "state", "sample.py"], "taint", "state"),
        (["taint", "delta", "sample.py", "--baseline", "taint.json"], "taint", "delta"),
        (
            ["taint", "baseline-write", "sample.py", "--baseline", "taint.json"],
            "taint",
            "baseline-write",
        ),
        (["taint", "lifecycle", "sample.py"], "taint", "lifecycle"),
    ],
)
def test_check_aux_subcommands_forward_domain_and_action(
    argv_suffix: list[str],
    domain: str,
    action: str,
) -> None:
    runner = CliRunner()
    captured: list[dict[str, object]] = []
    result = runner.invoke(cli.app, ["check", *argv_suffix], obj=_check_obj(captured))
    assert result.exit_code == 0
    assert captured
    aux_operation = captured[-1].get("aux_operation")
    assert isinstance(aux_operation, cli.CheckAuxOperation)
    assert aux_operation.domain == domain
    assert aux_operation.action == action
