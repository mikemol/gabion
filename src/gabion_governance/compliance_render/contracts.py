from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RenderedArtifact:
    markdown: str


@dataclass(frozen=True)
class ComplianceRenderResult:
    status_consistency: RenderedArtifact
