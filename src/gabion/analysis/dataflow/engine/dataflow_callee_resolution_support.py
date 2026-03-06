# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from gabion.analysis.dataflow.engine.dataflow_contracts import ClassInfo, FunctionInfo
from gabion.analysis.foundation.timeout_context import check_deadline


def _callee_key(name: str) -> str:
    if not name:
        return name
    return name.split(".")[-1]


def _resolve_class_candidates(
    base: str,
    *,
    module: str,
    symbol_table,
    class_index: dict[str, ClassInfo],
) -> list[str]:
    check_deadline()
    if not base:
        return []
    candidates: list[str] = []
    if "." in base:
        parts = base.split(".")
        head = parts[0]
        tail = ".".join(parts[1:])
        if symbol_table is not None:
            resolved_head = symbol_table.resolve(module, head)
            if resolved_head:
                candidates.append(f"{resolved_head}.{tail}")
        if module:
            candidates.append(f"{module}.{base}")
        candidates.append(base)
    else:
        if symbol_table is not None:
            resolved = symbol_table.resolve(module, base)
            if resolved:
                candidates.append(resolved)
            resolved_star = symbol_table.resolve_star(module, base)
            if resolved_star:
                candidates.append(resolved_star)
        if module:
            candidates.append(f"{module}.{base}")
        candidates.append(base)
    seen: set[str] = set()
    resolved_candidates: list[str] = []
    for candidate in candidates:
        check_deadline()
        if candidate not in seen:
            seen.add(candidate)
            if candidate in class_index:
                resolved_candidates.append(candidate)
    return resolved_candidates


@dataclass(frozen=True)
class _MethodHierarchyResolutionFound:
    kind: Literal["found"]
    resolved: FunctionInfo


@dataclass(frozen=True)
class _MethodHierarchyResolutionMissing:
    kind: Literal["not_found"]


MethodHierarchyResolution = (
    _MethodHierarchyResolutionFound | _MethodHierarchyResolutionMissing
)


def _resolve_method_in_hierarchy_outcome(
    class_qual: str,
    method: str,
    *,
    class_index: dict[str, ClassInfo],
    by_qual: Mapping[str, FunctionInfo],
    symbol_table,
    seen: set[str],
) -> MethodHierarchyResolution:
    check_deadline()
    if class_qual in seen:
        return _MethodHierarchyResolutionMissing(kind="not_found")
    seen.add(class_qual)
    candidate = f"{class_qual}.{method}"
    resolved = by_qual.get(candidate)
    if resolved is not None:
        return _MethodHierarchyResolutionFound(kind="found", resolved=resolved)
    info = class_index.get(class_qual)
    if info is not None:
        for base in info.bases:
            check_deadline()
            for base_qual in _resolve_class_candidates(
                base,
                module=info.module,
                symbol_table=symbol_table,
                class_index=class_index,
            ):
                check_deadline()
                resolution = _resolve_method_in_hierarchy_outcome(
                    base_qual,
                    method,
                    class_index=class_index,
                    by_qual=by_qual,
                    symbol_table=symbol_table,
                    seen=seen,
                )
                if type(resolution) is _MethodHierarchyResolutionFound:
                    return resolution
    return _MethodHierarchyResolutionMissing(kind="not_found")


def _resolve_method_in_hierarchy(
    class_qual: str,
    method: str,
    *,
    class_index: dict[str, ClassInfo],
    by_qual: Mapping[str, FunctionInfo],
    symbol_table,
    seen: set[str],
) -> MethodHierarchyResolution:
    return _resolve_method_in_hierarchy_outcome(
        class_qual,
        method,
        class_index=class_index,
        by_qual=by_qual,
        symbol_table=symbol_table,
        seen=seen,
    )


__all__ = [
    "MethodHierarchyResolution",
    "_callee_key",
    "_resolve_class_candidates",
    "_resolve_method_in_hierarchy",
]
