from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch
from typing import Callable, Sequence

import libcst as cst

from gabion.invariants import never

_NONE_TYPE = type(None)


def _leaf_cst_subclasses(base_type: type[cst.CSTNode]) -> tuple[type[cst.CSTNode], ...]:
    node_types: list[type[cst.CSTNode]] = []
    for candidate in vars(cst).values():
        try:
            if issubclass(candidate, base_type) and candidate is not base_type:
                node_types.append(candidate)
        except TypeError:
            continue
    node_types_tuple = tuple(node_types)
    return tuple(
        node_type
        for node_type in node_types_tuple
        if not any(
            candidate is not node_type and issubclass(candidate, node_type)
            for candidate in node_types_tuple
        )
    )


_CST_LEAF_EXPRESSION_TYPES = _leaf_cst_subclasses(cst.BaseExpression)
_CST_LEAF_NODE_TYPES = _leaf_cst_subclasses(cst.CSTNode)
_CST_LEAF_STATEMENT_TYPES = _leaf_cst_subclasses(cst.BaseStatement)
_CST_LEAF_SMALL_STATEMENT_TYPES = _leaf_cst_subclasses(cst.BaseSmallStatement)


@dataclass(frozen=True)
class ModuleExprTextOutcome:
    valid: bool
    text: str


def _noop_check_deadline() -> None:
    return


def _maybe_check_deadline(check_deadline_fn: Callable[[], None]) -> None:
    check_deadline_fn()


@singledispatch
def _is_simple_statement_line(value: object) -> bool:
    never("unregistered runtime type", value_type=type(value).__name__)


@_is_simple_statement_line.register(cst.SimpleStatementLine)
def _(value: cst.SimpleStatementLine) -> bool:
    _ = value
    return True


def _is_not_simple_statement_line(value: cst.CSTNode) -> bool:
    _ = value
    return False


for _runtime_type in _CST_LEAF_NODE_TYPES:
    if _runtime_type is cst.SimpleStatementLine:
        continue
    _is_simple_statement_line.register(_runtime_type)(_is_not_simple_statement_line)


@singledispatch
def _is_expr_statement(value: object) -> bool:
    never("unregistered runtime type", value_type=type(value).__name__)


@_is_expr_statement.register(cst.Expr)
def _(value: cst.Expr) -> bool:
    _ = value
    return True


def _is_not_expr_statement(value: cst.BaseSmallStatement) -> bool:
    _ = value
    return False


for _runtime_type in _CST_LEAF_SMALL_STATEMENT_TYPES:
    if _runtime_type is cst.Expr:
        continue
    _is_expr_statement.register(_runtime_type)(_is_not_expr_statement)


@singledispatch
def _is_simple_string_expression(value: object) -> bool:
    never("unregistered runtime type", value_type=type(value).__name__)


@_is_simple_string_expression.register(cst.SimpleString)
def _(value: cst.SimpleString) -> bool:
    _ = value
    return True


def _is_not_simple_string_expression(value: cst.BaseExpression) -> bool:
    _ = value
    return False


for _runtime_type in _CST_LEAF_EXPRESSION_TYPES:
    if _runtime_type is cst.SimpleString:
        continue
    _is_simple_string_expression.register(_runtime_type)(_is_not_simple_string_expression)


@singledispatch
def _line_body_items(value: object) -> tuple[object, ...]:
    never("unregistered runtime type", value_type=type(value).__name__)


@_line_body_items.register(cst.SimpleStatementLine)
def _(value: cst.SimpleStatementLine) -> tuple[object, ...]:
    return tuple(value.body)


def _empty_line_body_items(value: cst.CSTNode) -> tuple[object, ...]:
    _ = value
    return ()


for _runtime_type in _CST_LEAF_NODE_TYPES:
    if _runtime_type is cst.SimpleStatementLine:
        continue
    _line_body_items.register(_runtime_type)(_empty_line_body_items)


@singledispatch
def _expr_statement_value(value: object) -> object:
    never("unregistered runtime type", value_type=type(value).__name__)


@_expr_statement_value.register(cst.Expr)
def _(value: cst.Expr) -> object:
    return value.value


def is_docstring_statement(stmt: object) -> bool:
    line_items = _line_body_items(stmt)
    if not line_items:
        return False
    first_item = line_items[0]
    if not _is_expr_statement(first_item):
        return False
    return _is_simple_string_expression(_expr_statement_value(first_item))


@singledispatch
def _is_import_statement_item(value: object) -> bool:
    never("unregistered runtime type", value_type=type(value).__name__)


@_is_import_statement_item.register(cst.Import)
def _(value: cst.Import) -> bool:
    _ = value
    return True


@_is_import_statement_item.register(cst.ImportFrom)
def _(value: cst.ImportFrom) -> bool:
    _ = value
    return True


def _is_not_import_statement_item(value: cst.BaseSmallStatement) -> bool:
    _ = value
    return False


for _runtime_type in _CST_LEAF_SMALL_STATEMENT_TYPES:
    if _runtime_type in {cst.Import, cst.ImportFrom}:
        continue
    _is_import_statement_item.register(_runtime_type)(_is_not_import_statement_item)


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
        line_items = _line_body_items(stmt)
        if not line_items:
            break
        if not any(_is_import_statement_item(item) for item in line_items):
            break
        insert_idx += 1
    return insert_idx


@singledispatch
def _module_expr_text(
    expr: object,
    *,
    check_deadline_fn: Callable[[], None] = _noop_check_deadline,
) -> ModuleExprTextOutcome:
    _ = check_deadline_fn
    never("unregistered runtime type", value_type=type(expr).__name__)


@_module_expr_text.register(cst.Name)
def _(
    expr: cst.Name,
    *,
    check_deadline_fn: Callable[[], None] = _noop_check_deadline,
) -> ModuleExprTextOutcome:
    _maybe_check_deadline(check_deadline_fn)
    return ModuleExprTextOutcome(valid=True, text=expr.value)


@_module_expr_text.register(cst.Attribute)
def _(
    expr: cst.Attribute,
    *,
    check_deadline_fn: Callable[[], None] = _noop_check_deadline,
) -> ModuleExprTextOutcome:
    _maybe_check_deadline(check_deadline_fn)
    parent = _module_expr_text(expr.value, check_deadline_fn=check_deadline_fn)
    if not parent.valid:
        return ModuleExprTextOutcome(valid=True, text=expr.attr.value)
    return ModuleExprTextOutcome(valid=True, text=f"{parent.text}.{expr.attr.value}")


@_module_expr_text.register(_NONE_TYPE)
def _(
    expr: None,
    *,
    check_deadline_fn: Callable[[], None] = _noop_check_deadline,
) -> ModuleExprTextOutcome:
    _ = expr
    _maybe_check_deadline(check_deadline_fn)
    return ModuleExprTextOutcome(valid=False, text="")


def _invalid_module_expr_text(
    expr: cst.BaseExpression,
    *,
    check_deadline_fn: Callable[[], None] = _noop_check_deadline,
) -> ModuleExprTextOutcome:
    _ = expr
    _maybe_check_deadline(check_deadline_fn)
    return ModuleExprTextOutcome(valid=False, text="")


for _runtime_type in _CST_LEAF_EXPRESSION_TYPES:
    if _runtime_type in {cst.Name, cst.Attribute}:
        continue
    _module_expr_text.register(_runtime_type)(_invalid_module_expr_text)


def module_expr_to_str(
    expr: object,
    *,
    check_deadline_fn: Callable[[], None] = _noop_check_deadline,
) -> object:
    outcome = _module_expr_text(expr, check_deadline_fn=check_deadline_fn)
    if outcome.valid:
        return outcome.text
    return None


@singledispatch
def _is_import_from_statement(value: object) -> bool:
    never("unregistered runtime type", value_type=type(value).__name__)


@_is_import_from_statement.register(cst.ImportFrom)
def _(value: cst.ImportFrom) -> bool:
    _ = value
    return True


def _is_not_import_from_statement(value: cst.BaseSmallStatement) -> bool:
    _ = value
    return False


for _runtime_type in _CST_LEAF_SMALL_STATEMENT_TYPES:
    if _runtime_type is cst.ImportFrom:
        continue
    _is_import_from_statement.register(_runtime_type)(_is_not_import_from_statement)


@singledispatch
def _import_from_module_expr(value: object) -> object:
    never("unregistered runtime type", value_type=type(value).__name__)


@_import_from_module_expr.register(cst.ImportFrom)
def _(value: cst.ImportFrom) -> object:
    return value.module


@singledispatch
def _import_from_names_value(value: object) -> object:
    never("unregistered runtime type", value_type=type(value).__name__)


@_import_from_names_value.register(cst.ImportFrom)
def _(value: cst.ImportFrom) -> object:
    return value.names


@singledispatch
def _is_alias_sequence(value: object) -> bool:
    never("unregistered runtime type", value_type=type(value).__name__)


@_is_alias_sequence.register(tuple)
def _(value: tuple[object, ...]) -> bool:
    _ = value
    return True


@_is_alias_sequence.register(list)
def _(value: list[object]) -> bool:
    _ = value
    return True


@_is_alias_sequence.register(cst.ImportStar)
def _(value: cst.ImportStar) -> bool:
    _ = value
    return False


@singledispatch
def _alias_sequence_items(value: object) -> tuple[object, ...]:
    never("unregistered runtime type", value_type=type(value).__name__)


@_alias_sequence_items.register(tuple)
def _(value: tuple[object, ...]) -> tuple[object, ...]:
    return value


@_alias_sequence_items.register(list)
def _(value: list[object]) -> tuple[object, ...]:
    return tuple(value)


@_alias_sequence_items.register(cst.ImportStar)
def _(value: cst.ImportStar) -> tuple[object, ...]:
    _ = value
    return ()


@singledispatch
def _is_import_alias(value: object) -> bool:
    never("unregistered runtime type", value_type=type(value).__name__)


@_is_import_alias.register(cst.ImportAlias)
def _(value: cst.ImportAlias) -> bool:
    _ = value
    return True


@_is_import_alias.register(cst.ImportStar)
def _(value: cst.ImportStar) -> bool:
    _ = value
    return False


@singledispatch
def _import_alias_name_node(value: object) -> object:
    never("unregistered runtime type", value_type=type(value).__name__)


@_import_alias_name_node.register(cst.ImportAlias)
def _(value: cst.ImportAlias) -> object:
    return value.name


@singledispatch
def _is_name_node(value: object) -> bool:
    never("unregistered runtime type", value_type=type(value).__name__)


@_is_name_node.register(cst.Name)
def _(value: cst.Name) -> bool:
    _ = value
    return True


@_is_name_node.register(cst.Attribute)
def _(value: cst.Attribute) -> bool:
    _ = value
    return False


@singledispatch
def _name_node_value(value: object) -> str:
    never("unregistered runtime type", value_type=type(value).__name__)


@_name_node_value.register(cst.Name)
def _(value: cst.Name) -> str:
    return value.value


def has_import_from(
    body: Sequence[cst.CSTNode],
    *,
    module_name: str,
    symbol: str,
    check_deadline_fn: Callable[[], None] = _noop_check_deadline,
) -> bool:
    for stmt in body:
        _maybe_check_deadline(check_deadline_fn)
        line_items = _line_body_items(stmt)
        if not line_items:
            continue
        for item in line_items:
            _maybe_check_deadline(check_deadline_fn)
            if not _is_import_from_statement(item):
                continue
            module_outcome = _module_expr_text(
                _import_from_module_expr(item),
                check_deadline_fn=check_deadline_fn,
            )
            if not module_outcome.valid or module_outcome.text != module_name:
                continue
            names = _import_from_names_value(item)
            if not _is_alias_sequence(names):
                continue
            for alias in _alias_sequence_items(names):
                _maybe_check_deadline(check_deadline_fn)
                if not _is_import_alias(alias):
                    continue
                alias_name = _import_alias_name_node(alias)
                if _is_name_node(alias_name) and _name_node_value(alias_name) == symbol:
                    return True
    return False
