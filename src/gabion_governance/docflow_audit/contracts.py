from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, TypeAlias

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
class DocflowPredicateMatcher:
    predicates: tuple[str, ...]
    params: Mapping[str, JSONValue]


DocflowInvariantKind: TypeAlias = Literal["cover", "never", "require"]
DocflowInvariantStatus: TypeAlias = Literal["active", "proposed"]


@dataclass(frozen=True)
class DocflowInvariant:
    name: str
    kind: DocflowInvariantKind
    matcher: DocflowPredicateMatcher
    status: DocflowInvariantStatus = "active"


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
    context: dict[str, JSONValue] = field(default_factory=dict)


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
