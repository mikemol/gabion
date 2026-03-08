from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path
from typing import cast

from gabion.analysis.foundation.json_types import JSONObject, JSONValue
from gabion.order_contract import sort_once

from gabion.analysis.indexed_scan.ast.ast_context import (
    enclosing_function_context)
from gabion.analysis.indexed_scan.ast.context_walkers import iter_nodes_of_types, iter_parsed_path_contexts
from gabion.analysis.indexed_scan.obligations.handledness_decision import decide_handledness


def collect_handledness_witnesses(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    check_deadline_fn: Callable[[], None],
    parent_annotator_factory: Callable[[], object],
    collect_functions_fn: Callable[[ast.AST], list[ast.AST]],
    param_names_fn: Callable[..., list[str]],
    param_annotations_fn: Callable[..., dict[str, JSONValue]],
    normalize_snapshot_path_fn: Callable[..., str],
    find_handling_try_fn: Callable[..., object],
    enclosing_function_node_fn: Callable[..., object],
    enclosing_scopes_fn: Callable[..., list[str]],
    function_key_fn: Callable[..., str],
    refine_exception_name_from_annotations_fn: Callable[..., tuple[str, str, tuple[str, ...]]],
    exception_param_names_fn: Callable[..., list[str]],
    exception_path_id_fn: Callable[..., str],
    exception_handler_compatibility_fn: Callable[..., str],
    handler_label_fn: Callable[..., str],
    handler_type_names_fn: Callable[..., tuple[str, ...]],
) -> list[JSONObject]:
    check_deadline_fn()
    witnesses: list[JSONObject] = []
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
        param_annotations_fn=param_annotations_fn,
    ):
        for node in iter_nodes_of_types(
            context.tree,
            raise_or_assert_types,
            check_deadline_fn=check_deadline_fn,
        ):
            raise_node = cast(ast.Raise | ast.Assert, node)
            try_node = find_handling_try_fn(raise_node, context.parents)
            source_kind = "E0"
            match raise_node:
                case ast.Raise() as raise_stmt:
                    kind = "raise"
                    expr = raise_stmt.exc
                case ast.Assert() as assert_stmt:
                    kind = "assert"
                    expr = assert_stmt.test
                case _:
                    raise AssertionError("node must be ast.Raise or ast.Assert")

            function, params, param_annotations = enclosing_function_context(
                raise_node,
                parents=context.parents,
                params_by_fn=context.params_by_fn,
                param_annotations_by_fn=context.param_annotations_by_fn,
                enclosing_function_node_fn=enclosing_function_node_fn,
                enclosing_scopes_fn=enclosing_scopes_fn,
                function_key_fn=function_key_fn,
            )
            (
                exception_name,
                exception_type_source,
                exception_type_candidates,
            ) = refine_exception_name_from_annotations_fn(
                expr,
                param_annotations=param_annotations,
            )

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
            handledness_id = f"handled:{exception_id}"

            decision = decide_handledness(
                try_node,
                exception_name=exception_name,
                exception_type_candidates=exception_type_candidates,
                exception_handler_compatibility_fn=exception_handler_compatibility_fn,
                handler_label_fn=handler_label_fn,
                handler_type_names_fn=handler_type_names_fn,
                check_deadline_fn=check_deadline_fn,
            )
            if decision.handler_kind is not None and decision.result is not None:
                witnesses.append(
                    {
                        "handledness_id": handledness_id,
                        "exception_path_id": exception_id,
                        "site": {
                            "path": context.path_value,
                            "function": function,
                            "bundle": bundle,
                        },
                        "handler_kind": decision.handler_kind,
                        "handler_boundary": decision.handler_boundary,
                        "handler_types": list(decision.handler_type_names),
                        "type_compatibility": decision.compatibility,
                        "exception_type_source": exception_type_source,
                        "exception_type_candidates": list(exception_type_candidates),
                        "type_refinement_opportunity": decision.type_refinement_opportunity,
                        "handledness_reason_code": decision.handledness_reason_code,
                        "handledness_reason": decision.handledness_reason,
                        "environment": {},
                        "core": (
                            [f"enclosed by {decision.handler_boundary}"]
                            if decision.handler_kind == "catch"
                            else ["converted to process exit"]
                        ),
                        "result": decision.result,
                    }
                )

    return sort_once(
        witnesses,
        key=lambda entry: (
            str(entry.get("site", {}).get("path", "")),
            str(entry.get("site", {}).get("function", "")),
            ",".join(entry.get("site", {}).get("bundle", []) or []),
            str(entry.get("exception_path_id", "")),
        ),
        source="indexed_scan.handledness.collect_handledness_witnesses",
    )


__all__ = ["collect_handledness_witnesses"]
