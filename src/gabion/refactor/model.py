from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

Position = Tuple[int, int]


@dataclass(frozen=True)
class TextEdit:
    path: str
    start: Position
    end: Position
    replacement: str


@dataclass(frozen=True)
class RefactorRequest:
    protocol_name: str
    bundle: List[str]
    target_path: str
    target_functions: List[str] = field(default_factory=list)
    rationale: Optional[str] = None


@dataclass
class RefactorPlan:
    edits: List[TextEdit] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
