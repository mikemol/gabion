from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from gabion.exceptions import NeverThrown
from gabion import cli, server
from gabion.tooling.governance_rules import CommandPolicy, ControllerDriftPolicy, GovernanceRules


def _default_controller_drift_policy() -> ControllerDriftPolicy:
    return ControllerDriftPolicy(
        severity_classes=("low", "medium", "high", "critical"),
        enforce_at_or_above="high",
        remediation_by_severity={"high": "override_or_fix"},
        consecutive_passes_required=3,
    )


def _rules_for_check_command(
    *,
    require_lsp_carrier: bool = True,
    parity_required: bool = True,
    probe_payload: dict[str, object] | None,
) -> GovernanceRules:
    return GovernanceRules(
        override_token_env="TOKEN",
        gates={},
        command_policies={
            "gabion.check": CommandPolicy(
                command_id="gabion.check",
                maturity="beta",
                require_lsp_carrier=require_lsp_carrier,
                parity_required=parity_required,
                probe_payload=probe_payload,
                parity_ignore_keys=(),
            )
        },
        controller_drift=_default_controller_drift_policy(),
    )


# gabion:evidence E:call_footprint::tests/test_lsp_parity_gate.py::test_beta_command_fails_gate_when_lsp_validation_missing::server.py::gabion.server._execute_lsp_parity_gate_total
def test_beta_command_fails_gate_when_lsp_validation_missing() -> None:
    rules = _rules_for_check_command(probe_payload=None)
    ls = SimpleNamespace(workspace=SimpleNamespace(root_path="."))
    result = server._execute_lsp_parity_gate_total(
        ls,
        {},
        load_rules=lambda: rules,
    )
    assert result["exit_code"] == 1
    assert "requires LSP validation" in result["errors"][0]


# gabion:evidence E:call_footprint::tests/test_lsp_parity_gate.py::test_parity_gate_normalizes_payload_and_is_deterministic::server.py::gabion.server._execute_lsp_parity_gate_total
def test_parity_gate_normalizes_payload_and_is_deterministic() -> None:
    rules = _rules_for_check_command(probe_payload={"z": 1, "a": 2})

    def _lsp(_ls, payload):
        return {"k": payload.get("a"), "m": payload.get("z")}

    def _direct(_ls, payload):
        return {"m": payload.get("z"), "k": payload.get("a")}

    ls = SimpleNamespace(workspace=SimpleNamespace(root_path="."))
    first = server._execute_lsp_parity_gate_total(
        ls,
        {},
        load_rules=lambda: rules,
        lsp_executor_for_command=lambda _command: _lsp,
        direct_executor_for_command=lambda _command: _direct,
    )
    second = server._execute_lsp_parity_gate_total(
        ls,
        {},
        load_rules=lambda: rules,
        lsp_executor_for_command=lambda _command: _lsp,
        direct_executor_for_command=lambda _command: _direct,
    )
    assert first == second
    assert first["exit_code"] == 0
    assert first["checked_commands"][0]["parity_ok"] is True


# gabion:evidence E:call_footprint::tests/test_lsp_parity_gate.py::test_lsp_command_executor_map_covers_known_and_unknown::server.py::gabion.server._lsp_command_executor
def test_lsp_command_executor_map_covers_known_and_unknown() -> None:
    assert server._lsp_command_executor(server.CHECK_COMMAND) is server.execute_command
    assert server._lsp_command_executor(server.IMPACT_COMMAND) is server.execute_impact
    assert server._lsp_command_executor("gabion.unknown") is None


# gabion:evidence E:call_footprint::tests/test_lsp_parity_gate.py::test_strip_parity_ignored_keys_filters_requested_fields::server.py::gabion.server._strip_parity_ignored_keys
def test_strip_parity_ignored_keys_filters_requested_fields() -> None:
    assert server._strip_parity_ignored_keys(
        {"a": 1, "b": 2},
        ignored_keys=("b",),
    ) == {"a": 1}


# gabion:evidence E:call_footprint::tests/test_lsp_parity_gate.py::test_normalize_probe_payload_preserves_existing_root_and_timeout::server.py::gabion.server._normalize_probe_payload
def test_normalize_probe_payload_preserves_existing_root_and_timeout() -> None:
    normalized = server._normalize_probe_payload(
        {
            "root": "/tmp/custom",
            "analysis_timeout_ticks": 7,
            "analysis_timeout_tick_ns": 5,
            "b": 2,
            "a": 1,
        },
        root=Path("."),
        command="gabion.check",
    )
    assert normalized == {
        "a": 1,
        "analysis_timeout_tick_ns": 5,
        "analysis_timeout_ticks": 7,
        "b": 2,
        "root": "/tmp/custom",
    }


# gabion:evidence E:call_footprint::tests/test_lsp_parity_gate.py::test_execute_lsp_parity_gate_reports_missing_command_policy::server.py::gabion.server._execute_lsp_parity_gate_total
def test_execute_lsp_parity_gate_reports_missing_command_policy() -> None:
    ls = SimpleNamespace(workspace=SimpleNamespace(root_path="."))
    result = server._execute_lsp_parity_gate_total(
        ls,
        {"commands": ["gabion.unknown"]},
        load_rules=lambda: GovernanceRules(
            override_token_env="TOKEN",
            gates={},
            command_policies={},
            controller_drift=_default_controller_drift_policy(),
        ),
    )
    assert result["exit_code"] == 1
    assert result["errors"] == ["missing command policy for gabion.unknown"]


# gabion:evidence E:call_footprint::tests/test_lsp_parity_gate.py::test_execute_lsp_parity_gate_reports_missing_executors::server.py::gabion.server._execute_lsp_parity_gate_total
def test_execute_lsp_parity_gate_reports_missing_executors() -> None:
    rules = _rules_for_check_command(probe_payload={"k": "v"})
    ls = SimpleNamespace(workspace=SimpleNamespace(root_path="."))
    no_lsp = server._execute_lsp_parity_gate_total(
        ls,
        {"commands": ["gabion.check"]},
        load_rules=lambda: rules,
        lsp_executor_for_command=lambda _command: None,
        direct_executor_for_command=lambda _command: (lambda _ls, _payload: {}),
    )
    no_direct = server._execute_lsp_parity_gate_total(
        ls,
        {"commands": ["gabion.check"]},
        load_rules=lambda: rules,
        lsp_executor_for_command=lambda _command: (lambda _ls, _payload: {}),
        direct_executor_for_command=lambda _command: None,
    )
    assert no_lsp["errors"] == ["no LSP executor registered for gabion.check"]
    assert no_direct["errors"] == ["no direct executor registered for gabion.check"]


# gabion:evidence E:call_footprint::tests/test_lsp_parity_gate.py::test_execute_lsp_parity_gate_reports_parity_mismatch::server.py::gabion.server._execute_lsp_parity_gate_total
def test_execute_lsp_parity_gate_reports_parity_mismatch() -> None:
    rules = _rules_for_check_command(probe_payload={"k": "v"})
    ls = SimpleNamespace(workspace=SimpleNamespace(root_path="."))
    result = server._execute_lsp_parity_gate_total(
        ls,
        {"commands": ["gabion.check"]},
        load_rules=lambda: rules,
        lsp_executor_for_command=lambda _command: (lambda _ls, _payload: {"x": 1}),
        direct_executor_for_command=lambda _command: (lambda _ls, _payload: {"x": 2}),
    )
    assert result["errors"] == ["parity mismatch for gabion.check"]


# gabion:evidence E:call_footprint::tests/test_lsp_parity_gate.py::test_execute_lsp_parity_gate_converts_never_thrown_to_error::server.py::gabion.server._execute_lsp_parity_gate_total
def test_execute_lsp_parity_gate_converts_never_thrown_to_error() -> None:
    rules = _rules_for_check_command(probe_payload={"k": "v"})
    ls = SimpleNamespace(workspace=SimpleNamespace(root_path="."))

    def _raise_never(_ls, _payload):
        raise NeverThrown("invariant trip")

    result = server._execute_lsp_parity_gate_total(
        ls,
        {"commands": ["gabion.check"]},
        load_rules=lambda: rules,
        lsp_executor_for_command=lambda _command: _raise_never,
        direct_executor_for_command=lambda _command: (lambda _ls, _payload: {}),
    )
    assert result["errors"] == ["invariant trip"]


# gabion:evidence E:call_footprint::tests/test_lsp_parity_gate.py::test_execute_lsp_parity_gate_command_wrapper_returns_ordered_response::server.py::gabion.server.execute_lsp_parity_gate
def test_execute_lsp_parity_gate_command_wrapper_returns_ordered_response() -> None:
    ls = SimpleNamespace(workspace=SimpleNamespace(root_path="."))
    result = server.execute_lsp_parity_gate(ls, {"commands": ["gabion.unknown"]})
    assert result["exit_code"] == 1
    assert result["errors"] == ["missing command policy for gabion.unknown"]


# gabion:evidence E:call_footprint::tests/test_lsp_parity_gate.py::test_cli_run_lsp_parity_gate_is_thin_dispatcher::cli.py::gabion.cli.run_lsp_parity_gate
def test_cli_run_lsp_parity_gate_is_thin_dispatcher() -> None:
    calls: list[tuple[str, list[object], Path]] = []

    def _runner(request, *, root, notification_callback=None):
        _ = notification_callback
        calls.append((request.command, list(request.arguments), root))
        return {"exit_code": 0, "checked_commands": [], "errors": []}

    result = cli.run_lsp_parity_gate(commands=["gabion.check"], runner=_runner)
    assert result["exit_code"] == 0
    assert len(calls) == 1
    command, arguments, root = calls[0]
    assert command == cli.LSP_PARITY_GATE_COMMAND
    assert root == Path(".")
    assert len(arguments) == 1
    payload = arguments[0]
    assert isinstance(payload, dict)
    assert payload["commands"] == ["gabion.check"]
    assert payload["root"] == "."
    assert int(payload["analysis_timeout_ticks"]) > 0
    assert int(payload["analysis_timeout_tick_ns"]) > 0


# gabion:evidence E:call_footprint::tests/test_lsp_parity_gate.py::test_cli_run_lsp_parity_gate_without_commands_uses_root_only_payload::cli.py::gabion.cli.run_lsp_parity_gate
def test_cli_run_lsp_parity_gate_without_commands_uses_root_only_payload() -> None:
    calls: list[list[object]] = []

    def _runner(request, *, root, notification_callback=None):
        _ = root, notification_callback
        calls.append(list(request.arguments))
        return {"exit_code": 0, "checked_commands": [], "errors": []}

    result = cli.run_lsp_parity_gate(commands=None, runner=_runner)
    assert result["exit_code"] == 0
    assert len(calls) == 1
    assert len(calls[0]) == 1
    payload = calls[0][0]
    assert isinstance(payload, dict)
    assert payload["root"] == "."
    assert int(payload["analysis_timeout_ticks"]) > 0
    assert int(payload["analysis_timeout_tick_ns"]) > 0


# gabion:evidence E:call_footprint::tests/test_lsp_parity_gate.py::test_cli_lsp_parity_gate_command_reports_nonzero_exit::cli.py::gabion.cli.lsp_parity_gate
def test_cli_lsp_parity_gate_command_reports_nonzero_exit() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["lsp-parity-gate", "--command", "gabion.unknown", "--root", "."],
        env={
            "GABION_DIRECT_RUN": "1",
            "GABION_LSP_TIMEOUT_TICKS": "100000",
            "GABION_LSP_TIMEOUT_TICK_NS": "1000000",
        },
    )
    assert result.exit_code == 1
    assert "missing command policy for gabion.unknown" in result.stdout


# gabion:evidence E:call_footprint::tests/test_lsp_parity_gate.py::test_cli_lsp_parity_gate_command_allows_zero_exit::cli.py::gabion.cli.lsp_parity_gate
def test_cli_lsp_parity_gate_command_allows_zero_exit() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["lsp-parity-gate", "--command", "gabion.check", "--root", "."],
        env={
            "GABION_DIRECT_RUN": "1",
            "GABION_LSP_TIMEOUT_TICKS": "100000",
            "GABION_LSP_TIMEOUT_TICK_NS": "1000000",
        },
    )
    assert result.exit_code == 0

# gabion:evidence E:call_footprint::tests/test_lsp_parity_gate.py::test_execute_lsp_parity_gate_uses_typed_probe_error_channel::server.py::gabion.server._execute_lsp_parity_gate_total
def test_execute_lsp_parity_gate_uses_typed_probe_error_channel() -> None:
    rules = _rules_for_check_command(probe_payload={"k": "v"})
    ls = SimpleNamespace(workspace=SimpleNamespace(root_path="."))

    def _raise_runtime(_ls, _payload):
        raise RuntimeError("executor failed")

    result = server._execute_lsp_parity_gate_total(
        ls,
        {"commands": ["gabion.check"]},
        load_rules=lambda: rules,
        lsp_executor_for_command=lambda _command: _raise_runtime,
        direct_executor_for_command=lambda _command: (lambda _ls, _payload: {}),
    )
    assert result["exit_code"] == 1
    assert result["errors"] == ["executor failed"]
