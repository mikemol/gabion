from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from gabion.analysis.aspf import NodeId


StructuralAtom = (
    None
    | bool
    | int
    | float
    | str
    | bytes
    | tuple["StructuralAtom", ...]
)
DerivationNodeId = NodeId
ValueT = TypeVar("ValueT")


@dataclass(frozen=True)
class DerivationOp:
    name: str
    version: int = 1
    scope: str = "analysis"

    def as_key(self) -> tuple[str, int, str]:
        return (self.name, int(self.version), self.scope)


@dataclass(frozen=True)
class DerivationKey:
    op: DerivationOp
    input_nodes: tuple[DerivationNodeId, ...]
    params: StructuralAtom
    dependencies: StructuralAtom


@dataclass(frozen=True)
class DerivationNode:
    node_id: DerivationNodeId
    key: DerivationKey
    source: str


@dataclass(frozen=True)
class DerivationEdge:
    input_node_id: DerivationNodeId
    output_node_id: DerivationNodeId
    op_label: str


@dataclass(frozen=True)
class DerivationValue(Generic[ValueT]):
    node_id: DerivationNodeId
    value: ValueT
    lineage: tuple[DerivationNodeId, ...]
    generated_at_ns: int


@dataclass(frozen=True)
class DerivationCacheStats:
    hits: int
    misses: int
    evictions: int
    invalidations: int
    regenerations: int

