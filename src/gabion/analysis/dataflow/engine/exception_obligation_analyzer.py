from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExceptionObligationAnalyzerInput:
    infos: dict[str, object]
    never_exceptions: set[str]
    markers: set[str]


@dataclass(frozen=True)
class ExceptionObligationAnalyzerOutput:
    obligations: list[str]


def analyze_exception_obligations(*, data: ExceptionObligationAnalyzerInput, runner) -> ExceptionObligationAnalyzerOutput:
    obligations = runner(
        infos=data.infos,
        never_exceptions=data.never_exceptions,
        markers=data.markers,
    )
    return ExceptionObligationAnalyzerOutput(obligations=list(obligations))
