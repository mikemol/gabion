# gabion:ambiguity_boundary_module
# gabion:boundary_normalization_module
# gabion:grade_boundary kind=semantic_carrier_adapter name=dataflow_report_section_contracts
from __future__ import annotations

from dataclasses import dataclass, field

from gabion.foundation.replayable_stream import ReplayableStream, empty_stream


@dataclass(frozen=True)
class ReportSectionState:
    section_id: str
    lines: ReplayableStream[str] = field(default_factory=empty_stream)


@dataclass(frozen=True)
class PendingReportSectionState:
    section_id: str
    phase: str
    deps: tuple[str, ...] = ()
    reason: str = "policy"


@dataclass(frozen=True)
class ReportSectionsState:
    resolved_sections: ReplayableStream[ReportSectionState] = field(
        default_factory=empty_stream
    )
    pending_sections: ReplayableStream[PendingReportSectionState] = field(
        default_factory=empty_stream
    )


__all__ = [
    "PendingReportSectionState",
    "ReportSectionsState",
    "ReportSectionState",
]
