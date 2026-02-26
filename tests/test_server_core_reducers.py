from __future__ import annotations

from pathlib import Path

import pytest

from gabion.server import _default_execute_command_deps
from gabion.server_core.command_contract import CommandRuntimeInput, CommandRuntimeState
from gabion.server_core.command_effects import CommandEffects
from gabion.server_core.command_reducers import (
    initial_collection_progress,
    initial_paths_count,
    normalize_paths,
    normalize_timeout_total_ticks,
)


# gabion:evidence E:function_site::command_reducers.py::gabion.server_core.command_reducers.normalize_timeout_total_ticks E:decision_surface/direct::command_reducers.py::gabion.server_core.command_reducers.normalize_timeout_total_ticks::stale_c07f84f81743
def test_normalize_timeout_total_ticks_uses_default_when_unset() -> None:
    calls: list[tuple[str, object]] = []

    def _never(message: str, **kwargs: object) -> None:
        calls.append((message, kwargs.get("tick_limit")))

    actual = normalize_timeout_total_ticks(
        {},
        default_ticks=42,
        never_fn=_never,
    )
    assert actual == 42
    assert calls == []


# gabion:evidence E:function_site::command_reducers.py::gabion.server_core.command_reducers.normalize_timeout_total_ticks E:decision_surface/direct::command_reducers.py::gabion.server_core.command_reducers.normalize_timeout_total_ticks::stale_38e1b63ac9c5_0f457f57
def test_normalize_timeout_total_ticks_applies_explicit_limit() -> None:
    actual = normalize_timeout_total_ticks(
        {"analysis_tick_limit": "10"},
        default_ticks=42,
        never_fn=lambda *_args, **_kwargs: None,
    )
    assert actual == 10


# gabion:evidence E:function_site::command_reducers.py::gabion.server_core.command_reducers.normalize_timeout_total_ticks E:decision_surface/direct::command_reducers.py::gabion.server_core.command_reducers.normalize_timeout_total_ticks::stale_93f0a21e701c
@pytest.mark.parametrize("bad_value", ["0", "-1", "abc"])
def test_normalize_timeout_total_ticks_rejects_bad_limits(bad_value: str) -> None:
    calls: list[object] = []

    def _never(message: str, **kwargs: object) -> None:
        calls.append((message, kwargs.get("tick_limit")))

    actual = normalize_timeout_total_ticks(
        {"analysis_tick_limit": bad_value},
        default_ticks=64,
        never_fn=_never,
    )
    assert actual == 64
    assert calls


# gabion:evidence E:function_site::command_reducers.py::gabion.server_core.command_reducers.initial_collection_progress
def test_initial_collection_progress_shape() -> None:
    assert initial_collection_progress(total_files=7) == {
        "completed_files": 0,
        "in_progress_files": 0,
        "remaining_files": 0,
        "total_files": 7,
    }


# gabion:evidence E:function_site::command_reducers.py::gabion.server_core.command_reducers.initial_paths_count
def test_initial_paths_count_shape() -> None:
    assert initial_paths_count(["a", "b"]) == 2
    assert initial_paths_count(None) == 1


# gabion:evidence E:function_site::command_reducers.py::gabion.server_core.command_reducers.normalize_paths E:decision_surface/direct::command_reducers.py::gabion.server_core.command_reducers.normalize_paths::stale_8d1ddc9e98bb
def test_normalize_paths_defaults_to_root() -> None:
    root = Path("/tmp/demo")
    assert normalize_paths(None, root=root) == [root]


# gabion:evidence E:function_site::command_reducers.py::gabion.server_core.command_reducers.normalize_paths E:decision_surface/direct::command_reducers.py::gabion.server_core.command_reducers.normalize_paths::stale_20e03d310681_9b64d164
def test_normalize_paths_converts_entries() -> None:
    root = Path("/tmp/demo")
    actual = normalize_paths(["a.py", Path("b.py")], root=root)
    assert actual == [Path("a.py"), Path("b.py")]


# gabion:evidence E:function_site::command_contract.py::gabion.server_core.command_contract.CommandRuntimeInput
def test_command_contract_dataclasses() -> None:
    runtime_input = CommandRuntimeInput(
        payload={"root": "."},
        root=Path("."),
        report_path_text=None,
        timeout_total_ticks=100,
    )
    runtime_state = CommandRuntimeState(
        latest_collection_progress=initial_collection_progress(total_files=0)
    )
    assert runtime_input.timeout_total_ticks == 100
    runtime_state.latest_collection_progress["remaining_files"] = 1
    assert runtime_state.latest_collection_progress["remaining_files"] == 1


# gabion:evidence E:function_site::server.py::gabion.server._default_execute_command_deps
def test_default_execute_deps_implements_command_effects_protocol() -> None:
    deps = _default_execute_command_deps()
    assert isinstance(deps, CommandEffects)
