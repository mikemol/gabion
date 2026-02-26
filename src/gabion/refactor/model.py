# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

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
    type_hint: str = ""


@dataclass(frozen=True)
class CompatibilityShimConfig:
    enabled: bool = True
    emit_deprecation_warning: bool = True
    emit_overload_stubs: bool = True


def normalize_compatibility_shim(
    compatibility_shim: object,
) -> CompatibilityShimConfig:
    if type(compatibility_shim) is CompatibilityShimConfig:
        return compatibility_shim
    if type(compatibility_shim) is bool and compatibility_shim:
        return CompatibilityShimConfig(enabled=True)
    if type(compatibility_shim) is bool:
        return CompatibilityShimConfig(enabled=False)
    raise TypeError("compatibility_shim must be bool or CompatibilityShimConfig")


@dataclass(frozen=True)
class RefactorRequest:
    protocol_name: str
    bundle: List[str]
    target_path: str
    fields: List[FieldSpec] = field(default_factory=list)
    target_functions: List[str] = field(default_factory=list)
    compatibility_shim: CompatibilityShimConfig = field(
        default_factory=lambda: CompatibilityShimConfig(enabled=False)
    )
    ambient_rewrite: bool = False
    rationale: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "compatibility_shim",
            normalize_compatibility_shim(self.compatibility_shim),
        )


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
