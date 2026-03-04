from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol


JSONValue = object
JSONObject = dict[str, JSONValue]


@dataclass(frozen=True)
class StageIngressResult:
    payload: dict[str, object]
    options: Any
    mode: str


@dataclass(frozen=True)
class StageAnalysisResult:
    analysis_outcome: Any
    semantic_progress_cumulative: JSONObject | None
    latest_collection_progress: JSONObject
    last_collection_resume_payload: JSONObject | None


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
    def __call__(self, *, payload: Mapping[str, object], root: Path) -> Any: ...


class IngressModeSelector(Protocol):
    def __call__(self, *, payload: Mapping[str, object], options: Any) -> str: ...


class AnalysisRunner(Protocol):
    def __call__(self, *, context: Any, state: Any, collection_resume_payload: JSONObject | None) -> Any: ...


class PrimaryOutputEmitter(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, object]: ...


class AuxiliaryOutputEmitter(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> None: ...


class TimeoutCleanupHandler(Protocol):
    def __call__(self, *, exc: BaseException, context: Any) -> dict[str, object]: ...


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
