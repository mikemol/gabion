# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from gabion.commands import command_ids
from gabion.invariants import never

CommandExecutor = Callable[[object, dict[str, object] | None], dict]


@dataclass(frozen=True)
class CommandExecutorRefs:
    execute_command: CommandExecutor
    execute_structure_diff: CommandExecutor
    execute_structure_reuse: CommandExecutor
    execute_decision_diff: CommandExecutor
    execute_synthesis: CommandExecutor
    execute_refactor: CommandExecutor
    execute_impact: CommandExecutor
    execute_lsp_parity_gate: CommandExecutor


@dataclass(frozen=True)
class CommandDispatchRegistration:
    executor: CommandExecutor
    transport_lsp: bool
    transport_direct: bool


def _validate_registry_coverage(registry: dict[str, CommandDispatchRegistration]) -> None:
    semantic = set(command_ids.SEMANTIC_COMMAND_IDS)
    registered = set(registry)
    missing = tuple(command for command in command_ids.SEMANTIC_COMMAND_IDS if command not in registered)
    extras = tuple(command for command in registry if command not in semantic)
    if missing:
        never(
            "command dispatch registry missing semantic command ids",
            missing=missing,
        )
    if extras:
        never(
            "command dispatch registry has non-semantic command ids",
            extras=extras,
        )


def build_command_dispatch_registry(
    refs: CommandExecutorRefs,
) -> dict[str, CommandDispatchRegistration]:
    registry: dict[str, CommandDispatchRegistration] = {
        command_ids.CHECK_COMMAND: CommandDispatchRegistration(
            executor=refs.execute_command,
            transport_lsp=True,
            transport_direct=True,
        ),
        command_ids.DATAFLOW_COMMAND: CommandDispatchRegistration(
            executor=refs.execute_command,
            transport_lsp=True,
            transport_direct=True,
        ),
        command_ids.STRUCTURE_DIFF_COMMAND: CommandDispatchRegistration(
            executor=refs.execute_structure_diff,
            transport_lsp=True,
            transport_direct=True,
        ),
        command_ids.STRUCTURE_REUSE_COMMAND: CommandDispatchRegistration(
            executor=refs.execute_structure_reuse,
            transport_lsp=True,
            transport_direct=True,
        ),
        command_ids.DECISION_DIFF_COMMAND: CommandDispatchRegistration(
            executor=refs.execute_decision_diff,
            transport_lsp=True,
            transport_direct=True,
        ),
        command_ids.SYNTHESIS_COMMAND: CommandDispatchRegistration(
            executor=refs.execute_synthesis,
            transport_lsp=True,
            transport_direct=True,
        ),
        command_ids.REFACTOR_COMMAND: CommandDispatchRegistration(
            executor=refs.execute_refactor,
            transport_lsp=True,
            transport_direct=True,
        ),
        command_ids.IMPACT_COMMAND: CommandDispatchRegistration(
            executor=refs.execute_impact,
            transport_lsp=True,
            transport_direct=True,
        ),
        command_ids.LSP_PARITY_GATE_COMMAND: CommandDispatchRegistration(
            executor=refs.execute_lsp_parity_gate,
            transport_lsp=False,
            transport_direct=True,
        ),
    }
    _validate_registry_coverage(registry)
    return registry


def executor_for_transport(
    *,
    registry: dict[str, CommandDispatchRegistration],
    command: str,
    transport: Literal["lsp", "direct"],
) -> CommandExecutor | None:
    registration = registry.get(command)
    if registration is None:
        return None
    if transport == "lsp" and not registration.transport_lsp:
        return None
    if transport == "direct" and not registration.transport_direct:
        return None
    return registration.executor


def direct_executor_registry(
    registry: dict[str, CommandDispatchRegistration],
) -> dict[str, CommandExecutor]:
    return {
        command: registry[command].executor
        for command in command_ids.SEMANTIC_COMMAND_IDS
        if command in registry and registry[command].transport_direct
    }
