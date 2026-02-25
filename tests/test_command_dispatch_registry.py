from __future__ import annotations

from gabion.commands import command_ids, direct_dispatch


# gabion:evidence E:call_footprint::tests/test_command_dispatch_registry.py::test_semantic_command_ids_sorted::command_ids.py::gabion.commands.command_ids
def test_semantic_command_ids_sorted() -> None:
    assert command_ids.SEMANTIC_COMMAND_IDS == tuple(
        sorted(command_ids.SEMANTIC_COMMAND_IDS)
    )


# gabion:evidence E:call_footprint::tests/test_command_dispatch_registry.py::test_direct_dispatch_registry_sorted_and_complete::direct_dispatch.py::gabion.commands.direct_dispatch.direct_executor_registry
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
