from __future__ import annotations

from typing import Callable

from gabion.commands import command_ids
from gabion.commands.dispatch_registry import (
    CommandExecutorRefs,
    build_command_dispatch_registry,
    direct_executor_registry as build_direct_executor_registry,
)
from gabion.order_contract import sort_once

DirectExecutor = Callable[[object, dict[str, object] | None], dict]


def _server_command_registry() -> dict[str, DirectExecutor]:
    from gabion import server

    registry = build_command_dispatch_registry(
        CommandExecutorRefs(
            execute_command=server.execute_command,
            execute_structure_diff=server.execute_structure_diff,
            execute_structure_reuse=server.execute_structure_reuse,
            execute_decision_diff=server.execute_decision_diff,
            execute_synthesis=server.execute_synthesis,
            execute_refactor=server.execute_refactor,
            execute_impact=server.execute_impact,
            execute_lsp_parity_gate=server.execute_lsp_parity_gate,
        )
    )
    return build_direct_executor_registry(registry)


def direct_executor_registry() -> dict[str, DirectExecutor]:
    return _server_command_registry()


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
