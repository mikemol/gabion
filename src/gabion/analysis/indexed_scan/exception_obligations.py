# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path
from typing import cast

from gabion.analysis.json_types import JSONObject, JSONValue


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
        bundle = entry.get("bundle", []) or []
        bundle_values = sequence_or_none_fn(cast(JSONValue, bundle))
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
    raise_or_assert_types = {ast.Raise, ast.Assert}
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
    for path in paths:
        check_deadline_fn()
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        parent_annotator = parent_annotator_factory()
        parent_annotator.visit(tree)
        parents = parent_annotator.parents
        params_by_fn: dict[ast.AST, set[str]] = {}
        for fn in collect_functions_fn(tree):
            check_deadline_fn()
            params_by_fn[fn] = set(param_names_fn(fn, ignore_params))
        path_value = normalize_snapshot_path_fn(path, project_root)
        for node in ast.walk(tree):
            check_deadline_fn()
            if type(node) not in raise_or_assert_types:
                continue
            raise_node = cast(ast.Raise | ast.Assert, node)
            source_kind = "E0"
            kind = "raise" if type(raise_node) is ast.Raise else "assert"
            fn_node = enclosing_function_node_fn(raise_node, parents)
            if fn_node is None:
                function = "<module>"
                params = set()
            else:
                scopes = enclosing_scopes_fn(fn_node, parents)
                function = function_key_fn(scopes, fn_node.name)
                params = params_by_fn.get(fn_node, set())
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
            if is_never_marker_raise_fn(function, exception_name, never_exceptions_set):
                continue
            bundle = exception_param_names_fn(expr, params)
            lineno = getattr(raise_node, "lineno", 0)
            col = getattr(raise_node, "col_offset", 0)
            exception_id = exception_path_id_fn(
                path=path_value,
                function=function,
                source_kind=source_kind,
                lineno=lineno,
                col=col,
                kind=kind,
            )
            handled = handled_map.get(exception_id)
            status = "UNKNOWN"
            witness_ref = None
            remainder: dict[str, object] = {"exception_kind": kind}
            environment_ref: JSONValue = None
            handledness_reason_code = "NO_HANDLER"
            handledness_reason = "no handledness witness"
            exception_type_source: JSONValue = None
            exception_type_candidates: list[str] = []
            type_refinement_opportunity = ""
            if handled:
                witness_result = str(handled.get("result", ""))
                handledness_reason_code = str(
                    handled.get("handledness_reason_code", "UNKNOWN_REASON")
                )
                handledness_reason = str(handled.get("handledness_reason", ""))
                exception_type_source = handled.get("exception_type_source")
                raw_candidates = sequence_or_none_fn(
                    handled.get("exception_type_candidates") or []
                )
                if raw_candidates is not None:
                    exception_type_candidates = [str(v) for v in raw_candidates]
                type_refinement_opportunity = str(
                    handled.get("type_refinement_opportunity", "")
                )
                if witness_result == "HANDLED":
                    status = "HANDLED"
                    remainder = {}
                else:
                    remainder["handledness_result"] = witness_result or "UNKNOWN"
                    remainder["type_compatibility"] = str(
                        handled.get("type_compatibility", "unknown")
                    )
                    remainder["handledness_reason_code"] = handledness_reason_code
                    remainder["handledness_reason"] = handledness_reason
                    if exception_type_source:
                        remainder["exception_type_source"] = exception_type_source
                    if exception_type_candidates:
                        remainder["exception_type_candidates"] = exception_type_candidates
                    if type_refinement_opportunity:
                        remainder["type_refinement_opportunity"] = type_refinement_opportunity
                witness_ref = handled.get("handledness_id")
                environment_ref = handled.get("environment") or {}
            if status != "HANDLED":
                env_entries = env_by_site.get((path_value, function), {})
                if env_entries:
                    env = {name: value for name, (value, _) in env_entries.items()}
                    reachability = branch_reachability_under_env_fn(raise_node, parents, env)
                    if is_reachability_false_fn(reachability):
                        names: set[str] = set()
                        current = parents.get(raise_node)
                        while current is not None:
                            check_deadline_fn()
                            if type(current) is ast.If:
                                names.update(names_in_expr_fn(cast(ast.If, current).test))
                            current = parents.get(current)
                        ordered_names = sort_once_fn(
                            names,
                            source="indexed_scan.exception_obligations.names.dead",
                            policy=order_policy_sort,
                        )
                        for name in sort_once_fn(
                            ordered_names,
                            source="indexed_scan.exception_obligations.names.dead.enforce",
                            policy=order_policy_enforce,
                        ):
                            check_deadline_fn()
                            if name not in env_entries:
                                continue
                            _, witness = env_entries[name]
                            status = "DEAD"
                            witness_ref = witness.get("deadness_id")
                            remainder = {}
                            environment_ref = witness.get("environment") or {}
                            break
            if protocol == "never" and status != "DEAD":
                status = "FORBIDDEN"
            obligations.append(
                {
                    "exception_path_id": exception_id,
                    "site": {
                        "path": path_value,
                        "function": function,
                        "bundle": bundle,
                    },
                    "source_kind": source_kind,
                    "status": status,
                    "handledness_reason_code": handledness_reason_code,
                    "handledness_reason": handledness_reason,
                    "exception_type_source": exception_type_source,
                    "exception_type_candidates": exception_type_candidates,
                    "type_refinement_opportunity": type_refinement_opportunity,
                    "witness_ref": witness_ref,
                    "remainder": remainder,
                    "environment_ref": environment_ref,
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


__all__ = [
    "collect_exception_obligations",
    "dead_env_map",
]
