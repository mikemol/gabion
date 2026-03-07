#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class FiberTraceEvent:
    ordinal: int
    line: int
    column: int
    event_kind: str
    normalization_class: str
    input_slot: str
    phase_hint: str
    pre_core: bool


@dataclass(frozen=True)
class FiberApplicabilityBounds:
    current_boundary_before_ordinal: int
    violation_applies_when_boundary_before_ordinal_gt: int
    violation_clears_when_boundary_before_ordinal_lte: int
    boundary_domain_max_before_ordinal: int
    core_entry_before_ordinal: int | None


@dataclass(frozen=True)
class FiberCounterfactualBoundary:
    suggested_boundary_before_ordinal: int
    boundary_event_kind: str
    boundary_line: int
    boundary_column: int
    eliminates_violation_without_other_changes: bool
    preserves_prior_normalization: bool
    rationale: str


def to_payload_trace(
    events: Sequence[FiberTraceEvent],
) -> tuple[dict[str, object], ...]:
    return tuple(asdict(event) for event in events)


def to_payload_bounds(bounds: FiberApplicabilityBounds) -> dict[str, object]:
    return asdict(bounds)


# gabion:decision_protocol
def to_payload_counterfactual(
    counterfactual: FiberCounterfactualBoundary | None,
) -> dict[str, object]:
    match counterfactual:
        case None:
            return {"status": "absent"}
        case _:
            payload = asdict(counterfactual)
            payload["status"] = "present"
            return payload


__all__ = [
    "FiberApplicabilityBounds",
    "FiberCounterfactualBoundary",
    "FiberTraceEvent",
    "to_payload_bounds",
    "to_payload_counterfactual",
    "to_payload_trace",
]
