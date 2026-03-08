from __future__ import annotations

from gabion import server
from gabion.commands import command_ids, direct_dispatch
from gabion.commands.dispatch_registry import (
    CommandDispatchRegistration,
    executor_for_transport,
)


# gabion:evidence E:call_footprint::tests/test_command_dispatch_registry.py::test_semantic_command_ids_sorted::command_ids.py::gabion.commands.command_ids
# gabion:behavior primary=desired
def test_semantic_command_ids_sorted() -> None:
    assert command_ids.SEMANTIC_COMMAND_IDS == tuple(
        sorted(command_ids.SEMANTIC_COMMAND_IDS)
    )


# gabion:evidence E:call_footprint::tests/test_command_dispatch_registry.py::test_direct_dispatch_registry_sorted_and_complete::direct_dispatch.py::gabion.commands.direct_dispatch.direct_executor_registry
# gabion:behavior primary=desired
def test_direct_dispatch_registry_sorted_and_complete() -> None:
    keys = tuple(direct_dispatch.DIRECT_EXECUTOR_REGISTRY.keys())
    assert keys == tuple(sorted(keys))
    assert direct_dispatch.missing_semantic_command_ids() == ()
    assert direct_dispatch.extra_direct_command_ids() == ()
    assert direct_dispatch.is_registry_complete() is True
    for command in command_ids.SEMANTIC_COMMAND_IDS:
        assert callable(direct_dispatch.direct_executor(command))
    assert direct_dispatch.direct_executor(command_ids.CHECK_COMMAND) is not None
    assert direct_dispatch.direct_executor(command_ids.DATAFLOW_COMMAND) is not None


# gabion:evidence E:call_footprint::tests/test_command_dispatch_registry.py::test_semantic_command_transport_behavior_is_consistent::dispatch_registry.py::gabion.commands.dispatch_registry.executor_for_transport
# gabion:behavior primary=desired
def test_semantic_command_transport_behavior_is_consistent() -> None:
    registry = server._command_dispatch_registry()
    for command in command_ids.SEMANTIC_COMMAND_IDS:
        registration = registry[command]
        assert isinstance(registration, CommandDispatchRegistration)
        lsp_executor = executor_for_transport(
            registry=registry,
            command=command,
            transport="lsp",
        )
        direct_executor = executor_for_transport(
            registry=registry,
            command=command,
            transport="direct",
        )
        assert (lsp_executor is not None) is registration.transport_lsp
        assert (direct_executor is not None) is registration.transport_direct
        assert (direct_dispatch.direct_executor(command) is not None) is registration.transport_direct
