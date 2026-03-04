from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CallGraphAnalyzerInput:
    infos: dict[str, object]
    project_root: object


@dataclass(frozen=True)
class CallGraphAnalyzerOutput:
    call_graph: dict[str, set[str]]


def analyze_call_graph(*, data: CallGraphAnalyzerInput, runner) -> CallGraphAnalyzerOutput:
    graph = runner(data.infos, project_root=data.project_root)
    return CallGraphAnalyzerOutput(call_graph=dict(graph))
