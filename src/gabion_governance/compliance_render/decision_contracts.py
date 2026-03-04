from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionSurface:
    path: str
    qual: str
    params: tuple[str, ...]
    meta: str

    @property
    def is_boundary(self) -> bool:
        return "boundary" in self.meta


@dataclass(frozen=True)
class LintEntry:
    path: str
    line: int
    col: int
    code: str
    message: str
    param: str | None


@dataclass(frozen=True)
class ConsolidationConfig:
    min_functions: int = 3
    min_files: int = 2
    max_examples: int = 5
    require_forest: bool = True
