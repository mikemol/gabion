from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BaselineFacet:
    """Facet that captures baseline refresh delta risk state."""

    risks: tuple[tuple[str, int], ...] = ()

    def risk(self, key: str) -> int:
        for name, value in self.risks:
            if name == key:
                return value
        return 0


@dataclass(frozen=True)
class DocflowFacet:
    """Facet that captures docflow context from analysis."""

    changed_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class IssueLinkFacet:
    """Facet for SPPF linkage and checklist impact."""

    issue_ids: tuple[str, ...] = ()
    checklist_impact: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class DeadlineFacet:
    """Facet for deadline/budget controls."""

    timeout_seconds: int | None = None


@dataclass
class ExecutionPlan:
    """Execution plan with typed facets and deterministic decoration order."""

    baseline: BaselineFacet = field(default_factory=BaselineFacet)
    docflow: DocflowFacet = field(default_factory=DocflowFacet)
    issue_link: IssueLinkFacet = field(default_factory=IssueLinkFacet)
    deadline: DeadlineFacet = field(default_factory=DeadlineFacet)
    _decorations: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

    def with_baseline(self, facet: BaselineFacet) -> ExecutionPlan:
        self.baseline = facet
        return self

    def with_docflow(self, facet: DocflowFacet) -> ExecutionPlan:
        self.docflow = facet
        return self

    def with_issue_link(self, facet: IssueLinkFacet) -> ExecutionPlan:
        self.issue_link = facet
        return self

    def with_deadline(self, facet: DeadlineFacet) -> ExecutionPlan:
        self.deadline = facet
        return self

    def decorate(self, key: str, payload: dict[str, Any]) -> None:
        self._decorations[key] = dict(payload)

    def decorations(self) -> list[tuple[str, dict[str, Any]]]:
        return [(key, self._decorations[key]) for key in sorted(self._decorations)]
