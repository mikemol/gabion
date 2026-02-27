from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass(frozen=True)
class FieldSpec:
    name: str
    type_hint: str = ""
    source_params: Set[str] = field(default_factory=set)


@dataclass(frozen=True)
class ProtocolSpec:
    name: str
    fields: List[FieldSpec]
    bundle: Set[str]
    tier: int
    rationale: str = ""


@dataclass(frozen=True)
class SynthesisPlan:
    protocols: List[ProtocolSpec] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class NamingContext:
    existing_names: Set[str] = field(default_factory=set)
    frequency: Dict[str, int] = field(default_factory=dict)
    fallback_prefix: str = "Bundle"


@dataclass(frozen=True)
class SynthesisConfig:
    max_tier: int = 2
    min_bundle_size: int = 2
    allow_singletons: bool = False
    merge_overlap_threshold: float = 0.75
