# gabion:ambiguity_boundary_module
# gabion:boundary_normalization_module
# gabion:grade_boundary kind=semantic_carrier_adapter name=dataflow_report_section_contracts
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ReportSectionState:
    section_id: str
    _line_iterator_factory: Callable[[], Iterator[str]] = field(
        default_factory=lambda: (lambda: iter(()))
    )


@dataclass(frozen=True)
class PendingReportSectionState:
    section_id: str
    phase: str
    deps: tuple[str, ...] = ()
    reason: str = "policy"


@dataclass(frozen=True)
class ReportSectionsState:
    _resolved_section_iterator_factory: Callable[[], Iterator[ReportSectionState]] = field(
        default_factory=lambda: (lambda: iter(()))
    )
    _pending_section_iterator_factory: Callable[
        [], Iterator[PendingReportSectionState]
    ] = field(default_factory=lambda: (lambda: iter(())))


__all__ = [
    "PendingReportSectionState",
    "ReportSectionsState",
    "ReportSectionState",
]
