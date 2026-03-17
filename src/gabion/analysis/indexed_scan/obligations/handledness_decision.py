from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast
from gabion.invariants import never


@dataclass(frozen=True)
class HandlednessDecision:
    handler_kind: object
    handler_boundary: object
    handler_type_names: tuple[str, ...]
    compatibility: str
    handledness_reason_code: str
    handledness_reason: str
    type_refinement_opportunity: str
    result: object


def decide_handledness(
    try_node: object,
    *,
    exception_name: str,
    exception_type_candidates: tuple[str, ...],
    exception_handler_compatibility_fn: Callable[..., str],
    handler_label_fn: Callable[..., str],
    handler_type_names_fn: Callable[..., tuple[str, ...]],
    check_deadline_fn: Callable[[], None],
) -> HandlednessDecision:
    handler_kind = None
    handler_boundary = None
    chosen_handler = None
    compatibility = "incompatible"
    handledness_reason_code = "NO_HANDLER"
    handledness_reason = "no enclosing handler discharges this exception path"
    type_refinement_opportunity = ""

    match try_node:
        case ast.Try() as typed_try_node:
            unknown_handler = None
            first_incompatible_handler = None
            for handler in typed_try_node.handlers:
                check_deadline_fn()
                compatibility = exception_handler_compatibility_fn(exception_name, handler.type)
                if compatibility == "compatible":
                    handler_kind = "catch"
                    chosen_handler = handler
                    handler_boundary = handler_label_fn(handler)
                    if handler.type is None:
                        handledness_reason_code = "BROAD_EXCEPT"
                        handledness_reason = "handled by broad except: without a typed match proof"
                    else:
                        handledness_reason_code = "TYPED_MATCH"
                        handledness_reason = "raised exception type matches an explicit except clause"
                    break
                if compatibility == "unknown" and unknown_handler is None:
                    unknown_handler = handler
                if compatibility == "incompatible" and first_incompatible_handler is None:
                    first_incompatible_handler = handler

            if handler_kind is None and unknown_handler is not None:
                handler_kind = "catch"
                chosen_handler = unknown_handler
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
                chosen_handler = first_incompatible_handler
                handler_boundary = handler_label_fn(first_incompatible_handler)
                compatibility = "incompatible"
                handledness_reason_code = "TYPED_MISMATCH"
                handledness_reason = "explicit except clauses do not match the raised exception type"
                type_refinement_opportunity = (
                    f"consider except {exception_name} (or a supertype) to dominate this raise path"
                    if exception_name
                    else "consider a typed except clause to dominate this raise path"
                )
        case object():
            pass
    if handler_kind is None and exception_name == "SystemExit":
        handler_kind = "convert"
        handler_boundary = "process exit"
        compatibility = "compatible"
        handledness_reason_code = "SYSTEM_EXIT_CONVERT"
        handledness_reason = "SystemExit is converted to process exit"

    result = None
    handler_type_names: tuple[str, ...] = ()
    if handler_kind is not None:
        result = "HANDLED" if compatibility == "compatible" else "UNKNOWN"
        if handler_kind == "catch" and chosen_handler is not None:
            handler_type_names = handler_type_names_fn(chosen_handler.type)

    return HandlednessDecision(
        handler_kind=handler_kind,
        handler_boundary=handler_boundary,
        handler_type_names=handler_type_names,
        compatibility=compatibility,
        handledness_reason_code=handledness_reason_code,
        handledness_reason=handledness_reason,
        type_refinement_opportunity=type_refinement_opportunity,
        result=result,
    )


__all__ = ["HandlednessDecision", "decide_handledness"]
