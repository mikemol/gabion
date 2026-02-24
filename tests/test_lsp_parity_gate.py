from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

from gabion import cli, server
from gabion.tooling.governance_rules import CommandPolicy, GovernanceRules


# gabion:evidence E:call_footprint::tests/test_lsp_parity_gate.py::test_beta_command_fails_gate_when_lsp_validation_missing::server.py::gabion.server._execute_lsp_parity_gate_total
def test_beta_command_fails_gate_when_lsp_validation_missing() -> None:
    rules = GovernanceRules(
        override_token_env="TOKEN",
        gates={},
        command_policies={
            "gabion.check": CommandPolicy(
                command_id="gabion.check",
                maturity="beta",
                require_lsp_carrier=True,
                parity_required=True,
                probe_payload=None,
                parity_ignore_keys=(),
            )
        },
    )
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
    rules = GovernanceRules(
        override_token_env="TOKEN",
        gates={},
        command_policies={
            "gabion.check": CommandPolicy(
                command_id="gabion.check",
                maturity="beta",
                require_lsp_carrier=True,
                parity_required=True,
                probe_payload={"z": 1, "a": 2},
                parity_ignore_keys=(),
            )
        },
    )

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
    assert arguments == [
        {
            "commands": ["gabion.check"],
            "root": ".",
            "analysis_timeout_ticks": 100,
            "analysis_timeout_tick_ns": 1_000_000,
        }
    ]
