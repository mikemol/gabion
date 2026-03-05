from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SurfaceParseResult:
    path: str
    qual: str
    params: tuple[str, ...]
    meta: str
