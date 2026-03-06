# gabion:decision_protocol_module
from __future__ import annotations

"""Ingest/index helpers hoisted from legacy runtime for adapter consumers."""

import ast
from collections.abc import Iterable
from pathlib import Path
from typing import cast

from gabion.ingest.python_ingest import iter_python_paths
from gabion.order_contract import sort_once

from gabion.analysis.dataflow.engine.dataflow_contracts import AuditConfig, FunctionNode
from gabion.analysis.foundation.timeout_context import check_deadline


def _iter_paths(paths: Iterable[str], config: AuditConfig) -> list[Path]:
    return iter_python_paths(
        paths,
        config=config,
        check_deadline=check_deadline,
        sort_once=sort_once,
    )


def resolve_analysis_paths(paths: Iterable[object], *, config: AuditConfig) -> list[Path]:
    check_deadline()
    return _iter_paths([str(path) for path in paths], config)


def _collect_functions(tree: ast.AST) -> list[FunctionNode]:
    check_deadline()
    funcs: list[FunctionNode] = []
    for idx, node in enumerate(ast.walk(tree), start=1):
        if (idx & 63) == 0:
            check_deadline()
        node_type = type(node)
        if node_type is ast.FunctionDef or node_type is ast.AsyncFunctionDef:
            funcs.append(cast(FunctionNode, node))
    return funcs

__all__ = [
    "_collect_functions",
    "_iter_paths",
    "resolve_analysis_paths",
]
