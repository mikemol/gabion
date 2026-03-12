from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionProjectionOp:
    source_index: int
    op_name: str


@dataclass(frozen=True)
class SelectExecutionOp(ExecutionProjectionOp):
    predicates: tuple[str, ...]


@dataclass(frozen=True)
class ProjectExecutionOp(ExecutionProjectionOp):
    fields: tuple[str, ...]


@dataclass(frozen=True)
class CountByExecutionOp(ExecutionProjectionOp):
    fields: tuple[str, ...]


@dataclass(frozen=True)
class TraverseExecutionOp(ExecutionProjectionOp):
    field: str
    merge: bool = True
    keep: bool = False
    prefix: str = ""
    as_field: str = ""
    index_field: str = ""


@dataclass(frozen=True)
class SortKey:
    field: str
    order: str = "asc"


@dataclass(frozen=True)
class SortExecutionOp(ExecutionProjectionOp):
    keys: tuple[SortKey, ...]


@dataclass(frozen=True)
class LimitExecutionOp(ExecutionProjectionOp):
    count: int
