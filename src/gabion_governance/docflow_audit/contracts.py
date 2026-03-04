from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, TypeAlias

if TYPE_CHECKING:
    from gabion.analysis.projection.projection_spec import ProjectionSpec

FrontmatterScalar: TypeAlias = str | int
FrontmatterValue: TypeAlias = FrontmatterScalar | list[str] | dict[str, FrontmatterScalar]
Frontmatter: TypeAlias = dict[str, FrontmatterValue]
JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True)
class Doc:
    frontmatter: Frontmatter
    body: str


@dataclass(frozen=True)
class DocflowInvariant:
    name: str
    kind: str
    spec: ProjectionSpec
    status: str = "active"


@dataclass(frozen=True)
class DocflowAuditContext:
    docs: dict[str, Doc]
    revisions: dict[str, int]
    invariant_rows: list[dict[str, object]]
    invariants: list[DocflowInvariant]
    warnings: list[str]
    violations: list[str]


@dataclass(frozen=True)
class DocflowObligationResult:
    entries: list[dict[str, JSONValue]]
    summary: dict[str, int]
    warnings: list[str]
    violations: list[str]


@dataclass(frozen=True)
class AgentDirective:
    source: str
    scope_root: str
    line: int
    text: str
    normalized: str
    mandatory: bool
    delta_marked: bool


@dataclass(frozen=True)
class DocflowDomainResult:
    context: DocflowAuditContext
    obligations: DocflowObligationResult
    warnings: list[str]
    violations: list[str]
