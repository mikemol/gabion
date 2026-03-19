from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol, TypeVar

from gabion.json_types import JSONObject
from gabion.server_core.command_contract import (
    AnalysisOutcomeContract,
    CollectionResumeProgressState,
    ExecutionPayloadOptionsContract,
    IngressStageMode,
)


class AnalysisContextContract(Protocol):
    payload: Mapping[str, object]


class AnalysisStateContract(Protocol):
    latest_collection_progress: Mapping[str, int]


@dataclass(frozen=True)
class StageIngressResult:
    payload: dict[str, object]
    options: ExecutionPayloadOptionsContract
    mode: IngressStageMode


@dataclass(frozen=True)
class StageAnalysisResult:
    analysis_outcome: AnalysisOutcomeContract
    collection_resume_progress_state: CollectionResumeProgressState
    latest_collection_progress: JSONObject


@dataclass(frozen=True)
class StageOutputResult:
    response: dict[str, object]
    phase_checkpoint_state: JSONObject


@dataclass(frozen=True)
class StageTimeoutResult:
    response: dict[str, object]


class PayloadNormalizer(Protocol):
    def __call__(self, *, payload: dict[str, object]) -> dict[str, object]: ...


class PayloadOptionsParser(Protocol):
    def __call__(
        self, *, payload: Mapping[str, object], root: Path
    ) -> ExecutionPayloadOptionsContract: ...


class IngressModeSelector(Protocol):
    def __call__(
        self,
        *,
        payload: Mapping[str, object],
        options: ExecutionPayloadOptionsContract,
    ) -> IngressStageMode: ...


class AnalysisRunner(Protocol):
    def __call__(
        self,
        *,
        context: AnalysisContextContract,
        state: AnalysisStateContract,
        collection_resume_payload: JSONObject | None,
    ) -> AnalysisOutcomeContract: ...


@dataclass(frozen=True)
class PrimaryOutputRequest:
    emit: Callable[[], dict[str, object]]


@dataclass(frozen=True)
class AuxiliaryOutputRequest:
    emit: Callable[[], None]


_TimeoutCleanupContextT = TypeVar("_TimeoutCleanupContextT", contravariant=True)


class TimeoutCleanupHandler(Protocol[_TimeoutCleanupContextT]):
    def __call__(
        self,
        *,
        exc: BaseException,
        context: _TimeoutCleanupContextT,
    ) -> dict[str, object]: ...


@dataclass(frozen=True)
class _ExecutionPayloadOptions:
    """Shared execution options contract extracted for stage boundaries."""

    emit_phase_timeline: bool
    progress_heartbeat_seconds: float


@dataclass(frozen=True)
class _AnalysisInclusionFlags:
    """Shared analysis inclusion contract extracted for stage boundaries."""

    type_audit: bool
    include_decisions: bool
    include_rewrite_plans: bool
    include_exception_obligations: bool
    include_handledness_witnesses: bool
    include_never_invariants: bool
    include_wl_refinement: bool
    include_ambiguities: bool
    include_coherence: bool
    needs_analysis: bool
