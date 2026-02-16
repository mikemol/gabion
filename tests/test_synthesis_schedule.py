from __future__ import annotations

from gabion.synthesis.schedule import (
    _strongly_connected_components,
    topological_schedule,
)


# gabion:evidence E:function_site::schedule.py::gabion.synthesis.schedule.topological_schedule
def test_topological_schedule_orders_dependencies() -> None:
    graph = {"a": {"b"}, "b": set()}
    result = topological_schedule(graph)
    assert result.order == ["b", "a"]
    assert result.cycles == []


# gabion:evidence E:function_site::schedule.py::gabion.synthesis.schedule.topological_schedule
def test_topological_schedule_reports_cycles() -> None:
    graph = {"a": {"b"}, "b": {"a"}}
    result = topological_schedule(graph)
    assert result.cycles
    assert set().union(*result.cycles) == {"a", "b"}


def test_topological_schedule_handles_duplicate_followers_without_requeue() -> None:
    graph = {
        "a": {"root"},
        "b": {"root"},
        "c": {"a", "b"},
        "root": set(),
    }
    result = topological_schedule(graph)
    assert result.cycles == []
    assert result.order[0] == "root"
    assert result.order[-1] == "c"


def test_topological_schedule_reports_self_cycle() -> None:
    graph = {"a": {"a"}}
    result = topological_schedule(graph)
    assert result.order == []
    assert result.cycles == [{"a"}]


def test_strongly_connected_components_handles_back_edges() -> None:
    components = _strongly_connected_components({"a": {"b"}, "b": {"a"}})
    assert {"a", "b"} in components
