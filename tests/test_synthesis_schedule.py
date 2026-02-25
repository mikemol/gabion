from __future__ import annotations

from gabion.synthesis.schedule import (
    _strongly_connected_components,
    topological_schedule,
)


# gabion:evidence E:function_site::schedule.py::gabion.synthesis.schedule.topological_schedule E:decision_surface/direct::schedule.py::gabion.synthesis.schedule.topological_schedule::stale_00571c55270f_2b9a839b
def test_topological_schedule_orders_dependencies() -> None:
    graph = {"a": {"b"}, "b": set()}
    result = topological_schedule(graph)
    assert result.order == ["b", "a"]
    assert result.cycles == []


# gabion:evidence E:function_site::schedule.py::gabion.synthesis.schedule.topological_schedule E:decision_surface/direct::schedule.py::gabion.synthesis.schedule.topological_schedule::stale_678a3f92c068
def test_topological_schedule_reports_cycles() -> None:
    graph = {"a": {"b"}, "b": {"a"}}
    result = topological_schedule(graph)
    assert result.cycles
    assert set().union(*result.cycles) == {"a", "b"}


# gabion:evidence E:call_footprint::tests/test_synthesis_schedule.py::test_topological_schedule_handles_duplicate_followers_without_requeue::schedule.py::gabion.synthesis.schedule.topological_schedule
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


# gabion:evidence E:call_footprint::tests/test_synthesis_schedule.py::test_topological_schedule_reports_self_cycle::schedule.py::gabion.synthesis.schedule.topological_schedule
def test_topological_schedule_reports_self_cycle() -> None:
    graph = {"a": {"a"}}
    result = topological_schedule(graph)
    assert result.order == []
    assert result.cycles == [{"a"}]


# gabion:evidence E:call_footprint::tests/test_synthesis_schedule.py::test_strongly_connected_components_handles_back_edges::schedule.py::gabion.synthesis.schedule._strongly_connected_components
def test_strongly_connected_components_handles_back_edges() -> None:
    components = _strongly_connected_components({"a": {"b"}, "b": {"a"}})
    assert {"a", "b"} in components
