from __future__ import annotations

import ast
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import cast
from gabion.analysis.aspf.aspf import Forest, NodeId
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.invariants import never


class _SuiteSpanStatus(StrEnum):
    PRESENT = "present"
    MISSING = "missing"


@dataclass(frozen=True)
class _SuiteSpanOutcome:
    status: _SuiteSpanStatus
    span: tuple[int, int, int, int]


def _suite_span_from_statements_outcome(
    statements: Sequence[ast.stmt],
    *,
    node_span_fn: Callable[[ast.AST], object],
    check_deadline_fn: Callable[[], None],
) -> _SuiteSpanOutcome:
    check_deadline_fn()
    missing_span = (0, 0, 0, 0)
    if not statements:
        return _SuiteSpanOutcome(_SuiteSpanStatus.MISSING, missing_span)
    first_span_raw = node_span_fn(statements[0])
    first_span = _int_span4_optional(first_span_raw)
    if first_span is None:
        return _SuiteSpanOutcome(_SuiteSpanStatus.MISSING, missing_span)
    last_span = first_span
    for stmt in statements[1:]:
        check_deadline_fn()
        candidate_raw = node_span_fn(stmt)
        candidate_span = _int_span4_optional(candidate_raw)
        if candidate_span is not None:
            last_span = candidate_span
    return _SuiteSpanOutcome(
        _SuiteSpanStatus.PRESENT,
        (first_span[0], first_span[1], last_span[2], last_span[3]),
    )


def _int_span4_optional(value):
    match value:
        case tuple() as span_candidate if len(span_candidate) == 4:
            try:
                return (
                    int(span_candidate[0]),
                    int(span_candidate[1]),
                    int(span_candidate[2]),
                    int(span_candidate[3]),
                )
            except (TypeError, ValueError):
                return None
        case _:
            return None


            never("unreachable wildcard match fall-through")
def materialize_statement_suite_contains(
    *,
    forest: Forest,
    path_name: str,
    qual: str,
    statements: Sequence[ast.stmt],
    parent_suite: NodeId,
    node_span_fn: Callable[[ast.AST], object],
    check_deadline_fn: Callable[[], None] = check_deadline,
) -> None:
    check_deadline_fn()

    def _emit_body_suite(
        suite_kind: str,
        body: Sequence[ast.stmt],
    ) -> object:
        check_deadline_fn()
        span_outcome = _suite_span_from_statements_outcome(
            body,
            node_span_fn=node_span_fn,
            check_deadline_fn=check_deadline_fn,
        )
        if span_outcome.status is _SuiteSpanStatus.PRESENT:
            return forest.add_suite_site(
                path_name,
                qual,
                suite_kind,
                span=span_outcome.span,
                parent=parent_suite,
            )
        return None

    for stmt in statements:
        check_deadline_fn()
        stmt_type = type(stmt)
        if stmt_type is ast.If:
            if_stmt = cast(ast.If, stmt)
            if_suite = _emit_body_suite("if_body", if_stmt.body)
            if if_suite is not None:
                materialize_statement_suite_contains(
                    forest=forest,
                    path_name=path_name,
                    qual=qual,
                    statements=if_stmt.body,
                    parent_suite=if_suite,
                    node_span_fn=node_span_fn,
                    check_deadline_fn=check_deadline_fn,
                )
            if if_stmt.orelse:
                else_suite = _emit_body_suite("if_else", if_stmt.orelse)
                if else_suite is not None:
                    materialize_statement_suite_contains(
                        forest=forest,
                        path_name=path_name,
                        qual=qual,
                        statements=if_stmt.orelse,
                        parent_suite=else_suite,
                        node_span_fn=node_span_fn,
                        check_deadline_fn=check_deadline_fn,
                    )
            continue
        if stmt_type is ast.For:
            for_stmt = cast(ast.For, stmt)
            for_suite = _emit_body_suite("for_body", for_stmt.body)
            if for_suite is not None:
                materialize_statement_suite_contains(
                    forest=forest,
                    path_name=path_name,
                    qual=qual,
                    statements=for_stmt.body,
                    parent_suite=for_suite,
                    node_span_fn=node_span_fn,
                    check_deadline_fn=check_deadline_fn,
                )
            if for_stmt.orelse:
                for_else_suite = _emit_body_suite("for_else", for_stmt.orelse)
                if for_else_suite is not None:
                    materialize_statement_suite_contains(
                        forest=forest,
                        path_name=path_name,
                        qual=qual,
                        statements=for_stmt.orelse,
                        parent_suite=for_else_suite,
                        node_span_fn=node_span_fn,
                        check_deadline_fn=check_deadline_fn,
                    )
            continue
        if stmt_type is ast.AsyncFor:
            async_for_stmt = cast(ast.AsyncFor, stmt)
            async_for_suite = _emit_body_suite("async_for_body", async_for_stmt.body)
            if async_for_suite is not None:
                materialize_statement_suite_contains(
                    forest=forest,
                    path_name=path_name,
                    qual=qual,
                    statements=async_for_stmt.body,
                    parent_suite=async_for_suite,
                    node_span_fn=node_span_fn,
                    check_deadline_fn=check_deadline_fn,
                )
            if async_for_stmt.orelse:
                async_for_else_suite = _emit_body_suite("async_for_else", async_for_stmt.orelse)
                if async_for_else_suite is not None:
                    materialize_statement_suite_contains(
                        forest=forest,
                        path_name=path_name,
                        qual=qual,
                        statements=async_for_stmt.orelse,
                        parent_suite=async_for_else_suite,
                        node_span_fn=node_span_fn,
                        check_deadline_fn=check_deadline_fn,
                    )
            continue
        if stmt_type is ast.While:
            while_stmt = cast(ast.While, stmt)
            while_suite = _emit_body_suite("while_body", while_stmt.body)
            if while_suite is not None:
                materialize_statement_suite_contains(
                    forest=forest,
                    path_name=path_name,
                    qual=qual,
                    statements=while_stmt.body,
                    parent_suite=while_suite,
                    node_span_fn=node_span_fn,
                    check_deadline_fn=check_deadline_fn,
                )
            if while_stmt.orelse:
                while_else_suite = _emit_body_suite("while_else", while_stmt.orelse)
                if while_else_suite is not None:
                    materialize_statement_suite_contains(
                        forest=forest,
                        path_name=path_name,
                        qual=qual,
                        statements=while_stmt.orelse,
                        parent_suite=while_else_suite,
                        node_span_fn=node_span_fn,
                        check_deadline_fn=check_deadline_fn,
                    )
            continue
        if stmt_type is ast.Try:
            try_stmt = cast(ast.Try, stmt)
            try_suite = _emit_body_suite("try_body", try_stmt.body)
            if try_suite is not None:
                materialize_statement_suite_contains(
                    forest=forest,
                    path_name=path_name,
                    qual=qual,
                    statements=try_stmt.body,
                    parent_suite=try_suite,
                    node_span_fn=node_span_fn,
                    check_deadline_fn=check_deadline_fn,
                )
            for handler in try_stmt.handlers:
                check_deadline_fn()
                handler_suite = _emit_body_suite("except_body", handler.body)
                if handler_suite is not None:
                    materialize_statement_suite_contains(
                        forest=forest,
                        path_name=path_name,
                        qual=qual,
                        statements=handler.body,
                        parent_suite=handler_suite,
                        node_span_fn=node_span_fn,
                        check_deadline_fn=check_deadline_fn,
                    )
            if try_stmt.orelse:
                else_suite = _emit_body_suite("try_else", try_stmt.orelse)
                if else_suite is not None:
                    materialize_statement_suite_contains(
                        forest=forest,
                        path_name=path_name,
                        qual=qual,
                        statements=try_stmt.orelse,
                        parent_suite=else_suite,
                        node_span_fn=node_span_fn,
                        check_deadline_fn=check_deadline_fn,
                    )
            if try_stmt.finalbody:
                finally_suite = _emit_body_suite("try_finally", try_stmt.finalbody)
                if finally_suite is not None:
                    materialize_statement_suite_contains(
                        forest=forest,
                        path_name=path_name,
                        qual=qual,
                        statements=try_stmt.finalbody,
                        parent_suite=finally_suite,
                        node_span_fn=node_span_fn,
                        check_deadline_fn=check_deadline_fn,
                    )


__all__ = ["materialize_statement_suite_contains"]
