# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path
from typing import cast

from gabion.analysis.foundation.json_types import JSONObject, JSONValue

from gabion.analysis.indexed_scan.ast.ast_context import (
    enclosing_function_context)
from gabion.analysis.indexed_scan.ast.context_walkers import (
    empty_param_annotations, iter_nodes_of_types, iter_parsed_path_contexts)
from gabion.analysis.indexed_scan.obligations.obligation_decision import decide_exception_obligation


def dead_env_map(
    deadness_witnesses,
    *,
    check_deadline_fn: Callable[[], None],
    sequence_or_none_fn: Callable[[JSONValue], object],
    mapping_or_none_fn: Callable[[JSONValue], object],
    literal_eval_error_types: tuple[type[BaseException], ...],
) -> dict[tuple[str, str], dict[str, tuple[JSONValue, JSONObject]]]:
    check_deadline_fn()
    out: dict[tuple[str, str], dict[str, tuple[JSONValue, JSONObject]]] = {}
    if not deadness_witnesses:
        return out
    for entry in deadness_witnesses:
        check_deadline_fn()
        path_value = str(entry.get("path", ""))
        function_value = str(entry.get("function", ""))
        bundle_values = sequence_or_none_fn(cast(JSONValue, entry.get("bundle", []) or []))
        if bundle_values is not None and bundle_values:
            param = str(bundle_values[0])
            environment = mapping_or_none_fn(entry.get("environment", {}))
            if environment is not None:
                value_str = environment.get(param)
                literal_value = None
                if type(value_str) is str:
                    try:
                        literal_value = ast.literal_eval(value_str)
                    except literal_eval_error_types:
                        literal_value = None
                if literal_value is not None:
                    out.setdefault((path_value, function_value), {})[param] = (
                        literal_value,
                        entry,
                    )
    return out


def collect_exception_obligations(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    handledness_witnesses,
    deadness_witnesses,
    never_exceptions,
    check_deadline_fn: Callable[[], None],
    parent_annotator_factory: Callable[[], object],
    collect_functions_fn: Callable[[ast.AST], list[ast.AST]],
    param_names_fn: Callable[..., list[str]],
    normalize_snapshot_path_fn: Callable[..., str],
    enclosing_function_node_fn: Callable[..., object],
    enclosing_scopes_fn: Callable[..., list[str]],
    function_key_fn: Callable[..., str],
    exception_type_name_fn: Callable[..., object],
    decorator_matches_fn: Callable[..., bool],
    is_never_marker_raise_fn: Callable[..., bool],
    exception_param_names_fn: Callable[..., list[str]],
    exception_path_id_fn: Callable[..., str],
    sequence_or_none_fn: Callable[[JSONValue], object],
    branch_reachability_under_env_fn: Callable[..., object],
    is_reachability_false_fn: Callable[..., bool],
    is_reachability_true_fn: Callable[..., bool],
    names_in_expr_fn: Callable[..., set[str]],
    sort_once_fn: Callable[..., object],
    order_policy_sort,
    order_policy_enforce,
    mapping_or_none_fn: Callable[[JSONValue], object],
    literal_eval_error_types: tuple[type[BaseException], ...],
) -> list[JSONObject]:
    check_deadline_fn()
    obligations: list[JSONObject] = []
    never_exceptions_set = set(never_exceptions or [])

    handled_map: dict[str, JSONObject] = {}
    if handledness_witnesses:
        for entry in handledness_witnesses:
            check_deadline_fn()
            exception_id = str(entry.get("exception_path_id", ""))
            if exception_id:
                handled_map[exception_id] = entry

    env_by_site = dead_env_map(
        deadness_witnesses,
        check_deadline_fn=check_deadline_fn,
        sequence_or_none_fn=sequence_or_none_fn,
        mapping_or_none_fn=mapping_or_none_fn,
        literal_eval_error_types=literal_eval_error_types,
    )
    raise_or_assert_types = (ast.Raise, ast.Assert)

    for context in iter_parsed_path_contexts(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        check_deadline_fn=check_deadline_fn,
        parent_annotator_factory=parent_annotator_factory,
        collect_functions_fn=collect_functions_fn,
        param_names_fn=param_names_fn,
        normalize_snapshot_path_fn=normalize_snapshot_path_fn,
        param_annotations_fn=empty_param_annotations,
    ):
        for node in iter_nodes_of_types(
            context.tree,
            raise_or_assert_types,
            check_deadline_fn=check_deadline_fn,
        ):
            raise_node = cast(ast.Raise | ast.Assert, node)
            source_kind = "E0"
            kind = "raise" if type(raise_node) is ast.Raise else "assert"

            function, params, _ = enclosing_function_context(
                raise_node,
                parents=context.parents,
                params_by_fn=context.params_by_fn,
                param_annotations_by_fn=context.param_annotations_by_fn,
                enclosing_function_node_fn=enclosing_function_node_fn,
                enclosing_scopes_fn=enclosing_scopes_fn,
                function_key_fn=function_key_fn,
            )

            expr = (
                cast(ast.Raise, raise_node).exc
                if type(raise_node) is ast.Raise
                else cast(ast.Assert, raise_node).test
            )
            exception_name = exception_type_name_fn(expr)
            protocol = None
            if (
                exception_name
                and never_exceptions_set
                and decorator_matches_fn(exception_name, never_exceptions_set)
            ):
                protocol = "never"
            if not is_never_marker_raise_fn(function, exception_name, never_exceptions_set):
                bundle = exception_param_names_fn(expr, params)
                lineno = getattr(raise_node, "lineno", 0)
                col = getattr(raise_node, "col_offset", 0)
                exception_id = exception_path_id_fn(
                    path=context.path_value,
                    function=function,
                    source_kind=source_kind,
                    lineno=lineno,
                    col=col,
                    kind=kind,
                )

                decision = decide_exception_obligation(
                    kind=kind,
                    handled=handled_map.get(exception_id, {}),
                    has_handledness=exception_id in handled_map,
                    node=raise_node,
                    parents=context.parents,
                    env_entries=env_by_site.get((context.path_value, function), {}),
                    sequence_or_none_fn=sequence_or_none_fn,
                    branch_reachability_under_env_fn=branch_reachability_under_env_fn,
                    is_reachability_false_fn=is_reachability_false_fn,
                    names_in_expr_fn=names_in_expr_fn,
                    sort_once_fn=sort_once_fn,
                    order_policy_sort=order_policy_sort,
                    order_policy_enforce=order_policy_enforce,
                    check_deadline_fn=check_deadline_fn,
                )
                status = decision.status
                if protocol == "never" and status != "DEAD":
                    status = "FORBIDDEN"

                obligations.append(
                    {
                        "exception_path_id": exception_id,
                        "site": {
                            "path": context.path_value,
                            "function": function,
                            "bundle": bundle,
                        },
                        "source_kind": source_kind,
                        "status": status,
                        "handledness_reason_code": decision.handledness_reason_code,
                        "handledness_reason": decision.handledness_reason,
                        "exception_type_source": decision.exception_type_source,
                        "exception_type_candidates": decision.exception_type_candidates,
                        "type_refinement_opportunity": decision.type_refinement_opportunity,
                        "witness_ref": decision.witness_ref,
                        "remainder": decision.remainder,
                        "environment_ref": decision.environment_ref,
                        "exception_name": exception_name,
                        "protocol": protocol,
                    }
                )

    return sort_once_fn(
        obligations,
        key=lambda entry: (
            str(entry.get("site", {}).get("path", "")),
            str(entry.get("site", {}).get("function", "")),
            ",".join(entry.get("site", {}).get("bundle", []) or []),
            str(entry.get("source_kind", "")),
            str(entry.get("exception_path_id", "")),
        ),
        source="indexed_scan.exception_obligations.collect_exception_obligations",
    )


def collect_exception_obligations_from_runtime_module(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    handledness_witnesses=None,
    deadness_witnesses=None,
    never_exceptions=None,
    runtime_module,
) -> list[JSONObject]:
    return collect_exception_obligations(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        handledness_witnesses=handledness_witnesses,
        deadness_witnesses=deadness_witnesses,
        never_exceptions=never_exceptions,
        check_deadline_fn=runtime_module.check_deadline,
        parent_annotator_factory=runtime_module.ParentAnnotator,
        collect_functions_fn=runtime_module._collect_functions,
        param_names_fn=runtime_module._param_names,
        normalize_snapshot_path_fn=runtime_module._normalize_snapshot_path,
        enclosing_function_node_fn=runtime_module._enclosing_function_node,
        enclosing_scopes_fn=runtime_module._enclosing_scopes,
        function_key_fn=runtime_module._function_key,
        exception_type_name_fn=runtime_module._exception_type_name,
        decorator_matches_fn=runtime_module._decorator_matches,
        is_never_marker_raise_fn=runtime_module._is_never_marker_raise,
        exception_param_names_fn=runtime_module._exception_param_names,
        exception_path_id_fn=runtime_module._exception_path_id,
        sequence_or_none_fn=runtime_module.sequence_or_none,
        branch_reachability_under_env_fn=runtime_module._branch_reachability_under_env,
        is_reachability_false_fn=runtime_module._is_reachability_false,
        is_reachability_true_fn=runtime_module._is_reachability_true,
        names_in_expr_fn=runtime_module._names_in_expr,
        sort_once_fn=runtime_module.sort_once,
        order_policy_sort=runtime_module.OrderPolicy.SORT,
        order_policy_enforce=runtime_module.OrderPolicy.ENFORCE,
        mapping_or_none_fn=runtime_module.mapping_or_none,
        literal_eval_error_types=runtime_module._LITERAL_EVAL_ERROR_TYPES,
    )


__all__ = [
    "collect_exception_obligations",
    "collect_exception_obligations_from_runtime_module",
    "dead_env_map",
]
