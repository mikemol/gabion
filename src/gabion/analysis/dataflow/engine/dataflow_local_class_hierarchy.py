from __future__ import annotations

"""Canonical local-class hierarchy helpers for ingest analysis."""

import ast
from collections.abc import Sequence

from gabion.analysis.dataflow.engine.dataflow_evidence_helpers import (
    _base_identifier,
    _enclosing_class_scopes,
)
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.analysis.indexed_scan.calls.callee_resolution_helpers import (
    resolve_local_method_in_hierarchy as _resolve_local_method_in_hierarchy_impl,
)


def _collect_local_class_bases(
    class_nodes: Sequence[ast.ClassDef],
    parents: dict[ast.AST, ast.AST],
) -> dict[str, list[str]]:
    check_deadline()
    return {
        ".".join([*_enclosing_class_scopes(class_node, parents), class_node.name]): [
            base_name
            for base in class_node.bases
            if (base_name := _base_identifier(base))
        ]
        for class_node in class_nodes
    }


def _local_class_name(
    base: str,
    class_bases: dict[str, list[str]],
):
    return next(
        (candidate for candidate in (base, base.rsplit(".", 1)[-1]) if candidate in class_bases),
        None,
    )


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
