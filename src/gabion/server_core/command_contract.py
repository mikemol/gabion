from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping, Protocol

from gabion.json_types import JSONObject, JSONValue


class IngressStageMode(str, Enum):
    ANALYSIS = "analysis"
    AUX_OPERATION = "aux_operation"


class ExecutionPayloadOptionsContract(Protocol):
    emit_phase_timeline: bool
    progress_heartbeat_seconds: float


class AnalysisOutcomeContract(Protocol):
    collection_progress_runtime_state: CollectionProgressRuntimeState


@dataclass(frozen=True)
class CollectionResumeProgressState:
    last_collection_resume_payload: JSONObject | None = None
    semantic_progress_cumulative: JSONObject = field(default_factory=dict)


@dataclass(frozen=True)
class CollectionProgressRuntimeState:
    collection_resume_progress_state: CollectionResumeProgressState = field(
        default_factory=CollectionResumeProgressState
    )
    latest_collection_progress: JSONObject = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisResumeRuntimeState:
    state_path: Path | None = None
    state_status: str | None = None
    reused_files: int = 0
    total_files: int = 0


@dataclass(frozen=True)
class AnalysisResumeProjectionState:
    runtime_state: AnalysisResumeRuntimeState = field(
        default_factory=AnalysisResumeRuntimeState
    )
    source: str = "cold_start"
    compatibility_status: str | None = None


class ProgressTraceStateContract(Protocol):
    """Opaque progress trace state transported across progress hooks."""


@dataclass(frozen=True)
class CommandRuntimeInput:
    """Normalized command boundary inputs for server-core orchestration."""

    payload: Mapping[str, JSONValue]
    root: Path
    report_path_text: str | None
    timeout_total_ticks: int


@dataclass
class CommandRuntimeState:
    """Mutable runtime state carried through command execution."""

    latest_collection_progress: dict[str, int]
    semantic_progress_cumulative: JSONObject = field(default_factory=dict)


@dataclass(frozen=True)
class ProgressEvent:
    event_kind: str
    phase: str
    dimensions: Mapping[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True)
class CommandRuntimeOutcome:
    response: JSONObject
    terminal_phase: str


@dataclass(frozen=True)
class LspParityCommandResult:
    command_id: str
    maturity: str
    require_lsp_carrier: bool
    parity_required: bool
    lsp_validated: bool
    parity_ok: bool
    error: str | None = None


@dataclass(frozen=True)
class LspParityGateOutcome:
    exit_code: int
    checked_commands: tuple[LspParityCommandResult, ...]
    errors: tuple[str, ...]
