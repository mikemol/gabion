# gabion:decision_protocol_module
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
class FieldSpec:
    name: str
    type_hint: Optional[str] = None


@dataclass(frozen=True)
class CompatibilityShimConfig:
    enabled: bool = True
    emit_deprecation_warning: bool = True
    emit_overload_stubs: bool = True


def normalize_compatibility_shim(
    compatibility_shim: bool | CompatibilityShimConfig,
) -> CompatibilityShimConfig:
    if isinstance(compatibility_shim, CompatibilityShimConfig):
        return compatibility_shim
    if compatibility_shim:
        return CompatibilityShimConfig(enabled=True)
    return CompatibilityShimConfig(enabled=False)


@dataclass(frozen=True)
class RefactorRequest:
    protocol_name: str
    bundle: List[str]
    target_path: str
    fields: List[FieldSpec] = field(default_factory=list)
    target_functions: List[str] = field(default_factory=list)
    compatibility_shim: bool | CompatibilityShimConfig = False
    ambient_rewrite: bool = False
    rationale: Optional[str] = None


@dataclass(frozen=True)
class RewritePlanEntry:
    kind: str
    status: str
    target: str
    summary: str
    non_rewrite_reasons: List[str] = field(default_factory=list)


@dataclass
class RefactorPlan:
    edits: List[TextEdit] = field(default_factory=list)
    rewrite_plans: List[RewritePlanEntry] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
