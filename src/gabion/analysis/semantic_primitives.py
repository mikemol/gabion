# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from gabion.invariants import never


@dataclass(frozen=True)
class CallableId:
    value: str

    @classmethod
    def from_raw(cls, value: str) -> "CallableId":
        normalized = value.strip()
        if not normalized:
            never("callable id must be non-empty")
        return cls(value=normalized)


@dataclass(frozen=True)
class ParameterId:
    value: str

    @classmethod
    def from_raw(cls, value: str) -> "ParameterId":
        normalized = value.strip()
        if not normalized:
            never("parameter id must be non-empty")
        return cls(value=normalized)


@dataclass(frozen=True)
class SpanIdentity:
    start_line: int
    start_col: int
    end_line: int
    end_col: int

    @classmethod
    def from_tuple(cls, value: tuple[int, int, int, int]) -> "SpanIdentity":
        start_line, start_col, end_line, end_col = value
        if min(value) < 0:
            never("span identity cannot contain negative offsets", span=value)
        if end_line < start_line:
            never("span identity must be monotonic", span=value)
        if end_line == start_line and end_col < start_col:
            never("span identity columns must be monotonic", span=value)
        return cls(start_line=start_line, start_col=start_col, end_line=end_line, end_col=end_col)


@dataclass(frozen=True)
class CallArgumentMapping:
    positional: Mapping[int, ParameterId]
    keywords: Mapping[str, ParameterId]
    star_positional: tuple[tuple[int, ParameterId], ...]
    star_keywords: tuple[ParameterId, ...]


@dataclass(frozen=True)
class DecisionPredicateEvidence:
    parameter: ParameterId
    reasons: tuple[str, ...]
    spans: tuple[SpanIdentity, ...]


@dataclass(frozen=True)
class AnalysisPassPrerequisites:
    bundle_inference: bool
    call_propagation: bool
    decision_surfaces: bool
    type_flow: bool
    lint_evidence: bool

    def validate(self, *, pass_id: str) -> None:
        if not self.bundle_inference:
            never("bundle inference prerequisite missing", pass_id=pass_id)
        if not self.call_propagation:
            never("call propagation prerequisite missing", pass_id=pass_id)
        if not self.decision_surfaces:
            never("decision surfaces prerequisite missing", pass_id=pass_id)
        if not self.type_flow:
            never("type-flow prerequisite missing", pass_id=pass_id)
        if not self.lint_evidence:
            never("lint evidence prerequisite missing", pass_id=pass_id)
