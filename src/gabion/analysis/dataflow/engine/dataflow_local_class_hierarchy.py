from __future__ import annotations

"""Canonical local-class hierarchy helpers for ingest analysis."""

import ast
from typing import cast

from gabion.analysis.dataflow.engine.dataflow_evidence_helpers import (
    _base_identifier,
    _enclosing_class_scopes,
)
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.analysis.indexed_scan.calls.callee_resolution_helpers import (
    resolve_local_method_in_hierarchy as _resolve_local_method_in_hierarchy_impl,
)


def _collect_local_class_bases(
    tree: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> dict[str, list[str]]:
    check_deadline()
    class_bases: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        check_deadline()
        if type(node) is not ast.ClassDef:
            continue
        class_node = cast(ast.ClassDef, node)
        scopes = _enclosing_class_scopes(class_node, parents)
        qual_parts = list(scopes)
        qual_parts.append(class_node.name)
        qual = ".".join(qual_parts)
        bases: list[str] = []
        for base in class_node.bases:
            check_deadline()
            base_name = _base_identifier(base)
            if base_name:
                bases.append(base_name)
        class_bases[qual] = bases
    return class_bases


def _local_class_name(
    base: str,
    class_bases: dict[str, list[str]],
):
    if base in class_bases:
        return base
    if "." in base:
        tail = base.split(".")[-1]
        if tail in class_bases:
            return tail
    return None


def _resolve_local_method_in_hierarchy(
    class_name: str,
    method: str,
    *,
    class_bases: dict[str, list[str]],
    local_functions: set[str],
    seen: set[str],
):
    return _resolve_local_method_in_hierarchy_impl(
        class_name,
        method,
        class_bases=class_bases,
        local_functions=local_functions,
        seen=seen,
        check_deadline_fn=check_deadline,
        local_class_name_fn=_local_class_name,
    )


__all__ = [
    "_collect_local_class_bases",
    "_local_class_name",
    "_resolve_local_method_in_hierarchy",
]
