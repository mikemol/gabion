# gabion:boundary_normalization_module
from __future__ import annotations

"""Canonical lambda-runtime helpers for function-index accumulation."""

import ast
import hashlib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import cast

from gabion.analysis.dataflow.engine.dataflow_contracts import FunctionInfo
from gabion.analysis.dataflow.engine.dataflow_evidence_helpers import _target_names
from gabion.analysis.dataflow.engine.dataflow_function_index_helpers import (
    _enclosing_class,
    _enclosing_function_scopes,
    _enclosing_scopes,
    _node_span,
)
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.analysis.indexed_scan.ast.lambda_bindings import (
    ClosureLambdaFactoriesDeps as _ClosureLambdaFactoriesDeps,
    CollectLambdaFunctionInfosDeps as _CollectLambdaFunctionInfosDeps,
    LambdaBindingsByCallerDeps as _LambdaBindingsByCallerDeps,
    collect_closure_lambda_factories as _collect_closure_lambda_factories_impl,
    collect_lambda_bindings_by_caller as _collect_lambda_bindings_by_caller_impl,
    collect_lambda_function_infos as _collect_lambda_function_infos_impl,
)
from gabion.invariants import require_not_none
from gabion.order_contract import sort_once


def _function_key(scope: Iterable[str], name: str) -> str:
    scope_text = [str(item).strip() for item in scope if str(item).strip()]
    if not scope_text:
        return str(name)
    return ".".join([*scope_text, str(name)])


def _synthetic_lambda_name(
    *,
    module: str,
    lexical_scope: Sequence[str],
    span: tuple[int, int, int, int],
) -> str:
    check_deadline()
    lexical = ".".join(lexical_scope) if lexical_scope else "<module>"
    stable_payload = f"{module}|{lexical}|{span[0]}:{span[1]}:{span[2]}:{span[3]}"
    digest = hashlib.sha1(stable_payload.encode("utf-8")).hexdigest()[:12]
    return f"<lambda:{digest}>"


def _collect_lambda_function_infos(
    tree: ast.AST,
    *,
    path: Path,
    module: str,
    parent_map: Mapping[ast.AST, ast.AST],
    ignore_params,
) -> list[FunctionInfo]:
    return cast(
        list[FunctionInfo],
        _collect_lambda_function_infos_impl(
            tree,
            path=path,
            module=module,
            parent_map=parent_map,
            ignore_params=ignore_params,
            deps=_CollectLambdaFunctionInfosDeps(
                check_deadline_fn=check_deadline,
                node_span_fn=_node_span,
                enclosing_function_scopes_fn=_enclosing_function_scopes,
                enclosing_scopes_fn=_enclosing_scopes,
                enclosing_class_fn=_enclosing_class,
                synthetic_lambda_name_fn=_synthetic_lambda_name,
                function_info_ctor=FunctionInfo,
            ),
        ),
    )


def _collect_closure_lambda_factories(
    tree: ast.AST,
    *,
    module: str,
    parent_map: dict[ast.AST, ast.AST],
    lambda_qual_by_span: Mapping[tuple[int, int, int, int], str],
) -> dict[str, set[str]]:
    return _collect_closure_lambda_factories_impl(
        tree,
        module=module,
        parent_map=parent_map,
        lambda_qual_by_span=lambda_qual_by_span,
        deps=_ClosureLambdaFactoriesDeps(
            check_deadline_fn=check_deadline,
            node_span_fn=_node_span,
            target_names_fn=_target_names,
            enclosing_scopes_fn=_enclosing_scopes,
            function_key_fn=_function_key,
        ),
    )


def _collect_lambda_bindings_by_caller(
    tree: ast.AST,
    *,
    module: str,
    parent_map: dict[ast.AST, ast.AST],
    lambda_infos: Sequence[FunctionInfo],
) -> dict[str, dict[str, tuple[str, ...]]]:
    return _collect_lambda_bindings_by_caller_impl(
        tree,
        module=module,
        parent_map=parent_map,
        lambda_infos=cast(Sequence[object], lambda_infos),
        deps=_LambdaBindingsByCallerDeps(
            check_deadline_fn=check_deadline,
            require_not_none_fn=require_not_none,
            collect_closure_lambda_factories_fn=_collect_closure_lambda_factories,
            node_span_fn=_node_span,
            enclosing_scopes_fn=_enclosing_scopes,
            target_names_fn=_target_names,
            sort_once_fn=sort_once,
        ),
    )


__all__ = [
    "_collect_lambda_bindings_by_caller",
    "_collect_lambda_function_infos",
    "_function_key",
]
