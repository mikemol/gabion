from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class CommandRuntimeInput:
    """Normalized command boundary inputs for server-core orchestration."""

    payload: Mapping[str, object]
    root: Path
    report_path_text: str | None
    timeout_total_ticks: int


@dataclass
class CommandRuntimeState:
    """Mutable runtime state carried through command execution."""

    latest_collection_progress: dict[str, int]
    semantic_progress_cumulative: dict[str, object] | None = None


@dataclass(frozen=True)
class ProgressEvent:
    event_kind: str
    phase: str
    dimensions: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CommandRuntimeOutcome:
    response: dict[str, object]
    terminal_phase: str
