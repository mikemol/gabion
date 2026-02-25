from __future__ import annotations

from typing import Callable

from gabion.commands import command_ids
from gabion.order_contract import sort_once

DirectExecutor = Callable[[object, dict[str, object]], dict]


def _server_direct_executor(name: str) -> DirectExecutor:
    def _executor(ls: object, params: dict[str, object]) -> dict:
        from gabion import server

        candidate = getattr(server, name)
        return candidate(ls, params)

    return _executor


_UNORDERED_DIRECT_EXECUTORS: dict[str, DirectExecutor] = {
    command_ids.CHECK_COMMAND: _server_direct_executor("execute_command"),
    command_ids.DATAFLOW_COMMAND: _server_direct_executor("execute_command"),
    command_ids.STRUCTURE_DIFF_COMMAND: _server_direct_executor("execute_structure_diff"),
    command_ids.STRUCTURE_REUSE_COMMAND: _server_direct_executor("execute_structure_reuse"),
    command_ids.DECISION_DIFF_COMMAND: _server_direct_executor("execute_decision_diff"),
    command_ids.SYNTHESIS_COMMAND: _server_direct_executor("execute_synthesis"),
    command_ids.REFACTOR_COMMAND: _server_direct_executor("execute_refactor"),
    command_ids.IMPACT_COMMAND: _server_direct_executor("execute_impact"),
    command_ids.LSP_PARITY_GATE_COMMAND: _server_direct_executor(
        "execute_lsp_parity_gate"
    ),
}


def direct_executor_registry() -> dict[str, DirectExecutor]:
    return {
        command: _UNORDERED_DIRECT_EXECUTORS[command]
        for command in command_ids.SEMANTIC_COMMAND_IDS
        if command in _UNORDERED_DIRECT_EXECUTORS
    }


DIRECT_EXECUTOR_REGISTRY: dict[str, DirectExecutor] = direct_executor_registry()


def missing_semantic_command_ids() -> tuple[str, ...]:
    missing = [
        command
        for command in command_ids.SEMANTIC_COMMAND_IDS
        if command not in DIRECT_EXECUTOR_REGISTRY
    ]
    return tuple(missing)


def extra_direct_command_ids() -> tuple[str, ...]:
    extras = (
        command
        for command in DIRECT_EXECUTOR_REGISTRY
        if command not in command_ids.SEMANTIC_COMMAND_IDS
    )
    # Sort key is lexical command-id text for deterministic diagnostics.
    return tuple(sort_once(extras, source="src/gabion/commands/direct_dispatch.py:50"))


def is_registry_complete() -> bool:
    return not missing_semantic_command_ids() and not extra_direct_command_ids()


def direct_executor(command: str) -> DirectExecutor | None:
    return DIRECT_EXECUTOR_REGISTRY.get(command)
