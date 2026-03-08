from __future__ import annotations

from gabion.server_core import command_orchestrator_primitives
from gabion.server_core import dataflow_runtime_contract


def test_command_orchestrator_public_response_contract_matches_private_impl() -> None:
    payload = {
        "exit_code": 0,
        "lint_lines": ["pkg/mod.py:1:2: DF001 bad"],
        "lint_entries": "malformed",
    }
    public_envelope = command_orchestrator_primitives.ingress_normalize_dataflow_response_envelope(payload)
    private_envelope = command_orchestrator_primitives._normalize_dataflow_response(payload)
    assert public_envelope.model_dump() == private_envelope.model_dump()

    public_serialized = command_orchestrator_primitives.report_serialize_dataflow_response(public_envelope)
    private_serialized = command_orchestrator_primitives._serialize_dataflow_response(private_envelope)
    assert public_serialized == private_serialized


def test_runtime_contract_public_aliases_match_core_contract_values() -> None:
    assert (
        dataflow_runtime_contract.PROGRESS_LSP_NOTIFICATION_METHOD
        == dataflow_runtime_contract.LSP_PROGRESS_NOTIFICATION_METHOD
    )
    assert dataflow_runtime_contract.INGRESS_STDOUT_ALIAS == dataflow_runtime_contract.STDOUT_ALIAS
    assert (
        dataflow_runtime_contract.TIMEOUT_DEADLINE_TICK_BUDGET_ALLOWS_CHECK
        is dataflow_runtime_contract.deadline_tick_budget_allows_check
    )
