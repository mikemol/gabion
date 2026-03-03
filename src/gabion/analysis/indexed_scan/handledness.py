# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path
from typing import cast

from gabion.analysis.json_types import JSONObject, JSONValue
from gabion.order_contract import sort_once


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
    raise_or_assert_types = {ast.Raise, ast.Assert}
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
        param_annotations_by_fn: dict[ast.AST, dict[str, JSONValue]] = {}
        for fn in collect_functions_fn(tree):
            check_deadline_fn()
            params_by_fn[fn] = set(param_names_fn(fn, ignore_params))
            param_annotations_by_fn[fn] = param_annotations_fn(fn, ignore_params)
        path_value = normalize_snapshot_path_fn(path, project_root)
        for node in ast.walk(tree):
            check_deadline_fn()
            if type(node) not in raise_or_assert_types:
                continue
            raise_node = cast(ast.Raise | ast.Assert, node)
            try_node = find_handling_try_fn(raise_node, parents)
            source_kind = "E0"
            kind = "raise" if type(raise_node) is ast.Raise else "assert"
            fn_node = enclosing_function_node_fn(raise_node, parents)
            if fn_node is None:
                function = "<module>"
                params = set()
                param_annotations: dict[str, JSONValue] = {}
            else:
                scopes = enclosing_scopes_fn(fn_node, parents)
                function = function_key_fn(scopes, fn_node.name)
                params = params_by_fn.get(fn_node, set())
                param_annotations = param_annotations_by_fn.get(fn_node, {})
            expr = (
                cast(ast.Raise, raise_node).exc
                if type(raise_node) is ast.Raise
                else cast(ast.Assert, raise_node).test
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
                path=path_value,
                function=function,
                source_kind=source_kind,
                lineno=lineno,
                col=col,
                kind=kind,
            )
            handledness_id = f"handled:{exception_id}"
            handler_kind = None
            handler_boundary = None
            compatibility = "incompatible"
            handledness_reason_code = "NO_HANDLER"
            handledness_reason = "no enclosing handler discharges this exception path"
            type_refinement_opportunity = ""
            if try_node is not None:
                unknown_handler = None
                first_incompatible_handler = None
                for handler in try_node.handlers:
                    check_deadline_fn()
                    compatibility = exception_handler_compatibility_fn(
                        exception_name,
                        handler.type,
                    )
                    if compatibility == "compatible":
                        handler_kind = "catch"
                        handler_boundary = handler_label_fn(handler)
                        if handler.type is None:
                            handledness_reason_code = "BROAD_EXCEPT"
                            handledness_reason = (
                                "handled by broad except: without a typed match proof"
                            )
                        else:
                            handledness_reason_code = "TYPED_MATCH"
                            handledness_reason = (
                                "raised exception type matches an explicit except clause"
                            )
                        break
                    if compatibility == "unknown" and unknown_handler is None:
                        unknown_handler = handler
                    if (
                        compatibility == "incompatible"
                        and first_incompatible_handler is None
                    ):
                        first_incompatible_handler = handler
                if handler_kind is None and unknown_handler is not None:
                    handler_kind = "catch"
                    handler_boundary = handler_label_fn(unknown_handler)
                    compatibility = "unknown"
                    handledness_reason_code = "TYPE_UNRESOLVED"
                    handledness_reason = (
                        "exception or handler types are dynamic/unresolved; handledness is unknown"
                    )
                    if exception_type_candidates:
                        type_refinement_opportunity = (
                            "narrow raised exception type to a single concrete exception"
                        )
                elif handler_kind is None and first_incompatible_handler is not None:
                    handler_kind = "catch"
                    handler_boundary = handler_label_fn(first_incompatible_handler)
                    compatibility = "incompatible"
                    handledness_reason_code = "TYPED_MISMATCH"
                    handledness_reason = (
                        "explicit except clauses do not match the raised exception type"
                    )
                    type_refinement_opportunity = (
                        f"consider except {exception_name} (or a supertype) to dominate this raise path"
                        if exception_name
                        else "consider a typed except clause to dominate this raise path"
                    )
            if handler_kind is None and exception_name == "SystemExit":
                handler_kind = "convert"
                handler_boundary = "process exit"
                compatibility = "compatible"
                handledness_reason_code = "SYSTEM_EXIT_CONVERT"
                handledness_reason = "SystemExit is converted to process exit"
            if handler_kind is not None:
                witness_result = "HANDLED" if compatibility == "compatible" else "UNKNOWN"
                handler_type_names: tuple[str, ...] = ()
                if try_node is not None and handler_kind == "catch":
                    handler_types_by_label: dict[str, tuple[str, ...]] = {}
                    for handler in try_node.handlers:
                        check_deadline_fn()
                        handler_types_by_label[handler_label_fn(handler)] = handler_type_names_fn(
                            handler.type
                        )
                    handler_type_names = handler_types_by_label.get(
                        str(handler_boundary), ()
                    )
                witnesses.append(
                    {
                        "handledness_id": handledness_id,
                        "exception_path_id": exception_id,
                        "site": {
                            "path": path_value,
                            "function": function,
                            "bundle": bundle,
                        },
                        "handler_kind": handler_kind,
                        "handler_boundary": handler_boundary,
                        "handler_types": list(handler_type_names),
                        "type_compatibility": compatibility,
                        "exception_type_source": exception_type_source,
                        "exception_type_candidates": list(exception_type_candidates),
                        "type_refinement_opportunity": type_refinement_opportunity,
                        "handledness_reason_code": handledness_reason_code,
                        "handledness_reason": handledness_reason,
                        "environment": {},
                        "core": (
                            [f"enclosed by {handler_boundary}"]
                            if handler_kind == "catch"
                            else ["converted to process exit"]
                        ),
                        "result": witness_result,
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
