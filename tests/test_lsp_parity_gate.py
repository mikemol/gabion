from __future__ import annotations

from types import SimpleNamespace

from typer.testing import CliRunner

from gabion import cli, server
from gabion.tooling.governance_rules import CommandPolicy, GovernanceRules


# gabion:evidence E:call_footprint::tests/test_lsp_parity_gate.py::test_beta_command_fails_gate_when_lsp_validation_missing::server.py::gabion.server._execute_lsp_parity_gate_total
def test_beta_command_fails_gate_when_lsp_validation_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        server,
        "load_governance_rules",
        lambda: GovernanceRules(
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
        ),
    )
    ls = SimpleNamespace(workspace=SimpleNamespace(root_path="."))
    result = server._execute_lsp_parity_gate_total(ls, {})
    assert result["exit_code"] == 1
    assert "requires LSP validation" in result["errors"][0]


# gabion:evidence E:call_footprint::tests/test_lsp_parity_gate.py::test_parity_gate_normalizes_payload_and_is_deterministic::server.py::gabion.server._execute_lsp_parity_gate_total
def test_parity_gate_normalizes_payload_and_is_deterministic(monkeypatch) -> None:
    monkeypatch.setattr(
        server,
        "load_governance_rules",
        lambda: GovernanceRules(
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
        ),
    )

    def _lsp(_ls, payload):
        return {"k": payload.get("a"), "m": payload.get("z")}

    def _direct(_ls, payload):
        return {"m": payload.get("z"), "k": payload.get("a")}

    monkeypatch.setattr(server, "_lsp_command_executor", lambda _command: _lsp)
    monkeypatch.setattr(server.direct_dispatch, "direct_executor", lambda _command: _direct)

    ls = SimpleNamespace(workspace=SimpleNamespace(root_path="."))
    first = server._execute_lsp_parity_gate_total(ls, {})
    second = server._execute_lsp_parity_gate_total(ls, {})
    assert first == second
    assert first["exit_code"] == 0
    assert first["checked_commands"][0]["parity_ok"] is True


# gabion:evidence E:call_footprint::tests/test_lsp_parity_gate.py::test_cli_lsp_parity_gate_is_thin_dispatcher::cli.py::gabion.cli.lsp_parity_gate
def test_cli_lsp_parity_gate_is_thin_dispatcher(monkeypatch) -> None:
    calls: list[tuple[list[str] | None, object]] = []

    def _run(*, commands=None, root=None, runner=None):
        calls.append((commands, root))
        return {"exit_code": 0, "checked_commands": [], "errors": []}

    monkeypatch.setattr(cli, "run_lsp_parity_gate", _run)
    runner = CliRunner()
    result = runner.invoke(cli.app, ["lsp-parity-gate", "--command", "gabion.check"])
    assert result.exit_code == 0
    assert calls == [(["gabion.check"], None)]
