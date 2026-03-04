from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeadlineAnalyzerInput:
    entries: list[dict[str, object]]
    forest: object
    max_entries: int


@dataclass(frozen=True)
class DeadlineAnalyzerOutput:
    summary_lines: list[str]


def analyze_deadline_obligations(*, data: DeadlineAnalyzerInput, runner) -> DeadlineAnalyzerOutput:
    summary_lines = runner(entries=data.entries, max_entries=data.max_entries, forest=data.forest)
    return DeadlineAnalyzerOutput(summary_lines=list(summary_lines))
