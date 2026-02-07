from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.synthesis.schedule import topological_schedule

    return topological_schedule


# gabion:evidence E:function_site::schedule.py::gabion.synthesis.schedule.topological_schedule
def test_topological_schedule_orders_dependencies() -> None:
    topological_schedule = _load()
    graph = {"a": {"b"}, "b": set()}
    result = topological_schedule(graph)
    assert result.order == ["b", "a"]
    assert result.cycles == []


# gabion:evidence E:function_site::schedule.py::gabion.synthesis.schedule.topological_schedule
def test_topological_schedule_reports_cycles() -> None:
    topological_schedule = _load()
    graph = {"a": {"b"}, "b": {"a"}}
    result = topological_schedule(graph)
    assert result.cycles
    assert set().union(*result.cycles) == {"a", "b"}
