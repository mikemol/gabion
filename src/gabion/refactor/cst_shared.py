from __future__ import annotations

from typing import Callable, Sequence, cast

import libcst as cst


def _noop_check_deadline() -> None:
    return


def _maybe_check_deadline(check_deadline_fn: Callable[[], None]) -> None:
    check_deadline_fn()


def is_docstring_statement(stmt: object) -> bool:
    if type(stmt) is not cst.SimpleStatementLine:
        return False
    line = cast(cst.SimpleStatementLine, stmt)
    if not line.body:
        return False
    expr = line.body[0]
    if type(expr) is not cst.Expr:
        return False
    value = cast(cst.Expr, expr).value
    return type(value) is cst.SimpleString


def find_import_insert_index(
    body: Sequence[cst.CSTNode],
    *,
    check_deadline_fn: Callable[[], None] = _noop_check_deadline,
) -> int:
    _maybe_check_deadline(check_deadline_fn)
    insert_idx = 0
    if body and is_docstring_statement(body[0]):
        insert_idx = 1
    while insert_idx < len(body):
        _maybe_check_deadline(check_deadline_fn)
        stmt = body[insert_idx]
        if type(stmt) is not cst.SimpleStatementLine:
            break
        line = cast(cst.SimpleStatementLine, stmt)
        if not any(type(item) in {cst.Import, cst.ImportFrom} for item in line.body):
            break
        insert_idx += 1
    return insert_idx


def module_expr_to_str(
    expr: object,
    *,
    check_deadline_fn: Callable[[], None] = _noop_check_deadline,
) -> object:
    _maybe_check_deadline(check_deadline_fn)
    if type(expr) is cst.Name:
        return cast(cst.Name, expr).value
    if type(expr) is cst.Attribute:
        parts: list[str] = []
        current = expr
        while type(current) is cst.Attribute:
            _maybe_check_deadline(check_deadline_fn)
            current_attr = cast(cst.Attribute, current)
            parts.append(current_attr.attr.value)
            current = current_attr.value
        if type(current) is cst.Name:
            parts.append(cast(cst.Name, current).value)
        return ".".join(reversed(parts))
    return None


def has_import_from(
    body: Sequence[cst.CSTNode],
    *,
    module_name: str,
    symbol: str,
    check_deadline_fn: Callable[[], None] = _noop_check_deadline,
) -> bool:
    for stmt in body:
        _maybe_check_deadline(check_deadline_fn)
        if type(stmt) is not cst.SimpleStatementLine:
            continue
        line = cast(cst.SimpleStatementLine, stmt)
        for item in line.body:
            _maybe_check_deadline(check_deadline_fn)
            if type(item) is not cst.ImportFrom:
                continue
            import_from = cast(cst.ImportFrom, item)
            module_text = module_expr_to_str(
                import_from.module,
                check_deadline_fn=check_deadline_fn,
            )
            names = import_from.names
            if module_text != module_name or type(names) not in {tuple, list}:
                continue
            for alias in cast(tuple[cst.ImportAlias, ...], tuple(names)):
                _maybe_check_deadline(check_deadline_fn)
                if type(alias.name) is cst.Name and alias.name.value == symbol:
                    return True
    return False
