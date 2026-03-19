from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce, singledispatch, singledispatchmethod
from itertools import chain
from pathlib import Path
import re
from typing import cast

import libcst as cst
from libcst import metadata as cst_metadata
from libcst import matchers as cst_matchers

from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.invariants import grade_boundary, never
import gabion.refactor.cst_shared as cst_shared
from gabion.refactor.model import (
    LoopGeneratorRequest,
    RefactorPlan,
    RefactorPlanOutcome,
    RewritePlanEntry,
    TextEdit,
)

_LOOP_HELPER_PATTERN = re.compile(r"^_iter_(?P<name>[A-Za-z_]\w*)_loop_(?P<line>\d+)$")
_MAX_HELPER_CHASE_DEPTH = 128


@singledispatch
def _operator_token(operator: object) -> str:
    never("unregistered runtime type", value_type=type(operator).__name__)


def _register_operator_tokens(operator_types: tuple[type[object], ...], token: str) -> None:
    for operator_type in operator_types:
        @_operator_token.register(operator_type)
        def _(_: object, _token: str = token) -> str:
            return _token


_register_operator_tokens((cst.AddAssign, cst.Add), "+")
_register_operator_tokens((cst.SubtractAssign, cst.Subtract), "-")
_register_operator_tokens((cst.MultiplyAssign, cst.Multiply), "*")
_register_operator_tokens((cst.DivideAssign, cst.Divide), "/")
_register_operator_tokens((cst.FloorDivideAssign, cst.FloorDivide), "//")
_register_operator_tokens((cst.ModuloAssign, cst.Modulo), "%")
_register_operator_tokens((cst.PowerAssign, cst.Power), "**")
_register_operator_tokens((cst.BitOrAssign, cst.BitOr), "|")
_register_operator_tokens((cst.BitAndAssign, cst.BitAnd), "&")
_register_operator_tokens((cst.BitXorAssign, cst.BitXor), "^")
_register_operator_tokens((cst.LeftShiftAssign, cst.LeftShift), "<<")
_register_operator_tokens((cst.RightShiftAssign, cst.RightShift), ">>")


def _empty_operator_token(value: object) -> str:
    _ = value
    return ""


for _operator_type in (cst.MatrixMultiplyAssign, cst.MatrixMultiply):
    _operator_token.register(_operator_type)(_empty_operator_token)

def _loop_dataclass_decorator() -> cst.Decorator:
    return cst.Decorator(
        decorator=cst.Call(
            func=cst.Name("dataclass"),
            args=[
                cst.Arg(
                    keyword=cst.Name("frozen"),
                    value=cst.Name("True"),
                )
            ],
        )
    )


def _carrier_field(name: str, type_name: str) -> cst.SimpleStatementLine:
    return cst.SimpleStatementLine(
        body=[
            cst.AnnAssign(
                target=cst.Name(name),
                annotation=cst.Annotation(annotation=cst.Name(type_name)),
                value=None,
            )
        ]
    )


def _loop_carrier_class(
    class_name: str,
    fields: tuple[tuple[str, str], ...],
) -> cst.ClassDef:
    return cst.ClassDef(
        name=cst.Name(class_name),
        decorators=[_loop_dataclass_decorator()],
        body=cst.IndentedBlock(
            body=[_carrier_field(field_name, field_type) for field_name, field_type in fields]
        ),
    )


def _loop_transform_union_assignment(
    carrier_names: tuple[str, ...],
) -> cst.SimpleStatementLine:
    expr = reduce(
        _union_binary_expression,
        carrier_names[1:],
        cst.Name(carrier_names[0]),
    )
    return cst.SimpleStatementLine(
        body=[
            cst.Assign(
                targets=[cst.AssignTarget(target=cst.Name("_LoopTransformOp"))],
                value=expr,
            )
        ]
    )


def _union_binary_expression(
    left: cst.BaseExpression,
    carrier_name: str,
) -> cst.BaseExpression:
    check_deadline()
    return cst.BinaryOperation(
        left=left,
        operator=cst.BitOr(),
        right=cst.Name(carrier_name),
    )


def _loop_carrier_nodes() -> tuple[cst.CSTNode, ...]:
    carrier_shapes: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
        (
            "_LoopListAppendOp",
            (
                ("item", "object"),
                ("value", "object"),
                ("target", "str"),
            ),
        ),
        (
            "_LoopSetAddOp",
            (
                ("item", "object"),
                ("value", "object"),
                ("target", "str"),
            ),
        ),
        (
            "_LoopDictSetOp",
            (
                ("item", "object"),
                ("key", "object"),
                ("value", "object"),
                ("target", "str"),
            ),
        ),
        (
            "_LoopReduceOp",
            (
                ("item", "object"),
                ("target", "str"),
                ("operator", "str"),
                ("operand", "object"),
            ),
        ),
    )
    classes = tuple(
        _loop_carrier_class(class_name, fields)
        for class_name, fields in carrier_shapes
    )
    union_assignment = _loop_transform_union_assignment(
        tuple(class_name for class_name, _ in carrier_shapes)
    )
    return (*classes, union_assignment)


@dataclass(frozen=True)
class _LoopOperation:
    kind: str
    target: str
    value_expr: cst.BaseExpression = field(
        default_factory=lambda: cst.Name("_loop_unused_expr")
    )
    key_expr: cst.BaseExpression = field(
        default_factory=lambda: cst.Name("_loop_unused_expr")
    )
    operator: str = ""


@dataclass(frozen=True)
class _LoopRewriteSpec:
    function_name: str
    qualname: str
    loop_line: int
    loop_var: str
    iter_expr: cst.BaseExpression
    guard_exprs: tuple[cst.BaseExpression, ...]
    operations: tuple[_LoopOperation, ...]
    helper_name: str
    params: cst.Parameters
    call_args: tuple[cst.Arg, ...]


@dataclass(frozen=True)
class _LoopCandidate:
    loop_node: cst.CSTNode
    line: int


@dataclass(frozen=True)
class _FunctionIndexEntry:
    qualname: str
    node: cst.FunctionDef


@dataclass(frozen=True)
class _TargetResolution:
    effective_targets: tuple[str, ...]
    root_to_effective: tuple[tuple[str, tuple[str, ...]], ...]
    effective_to_roots: tuple[tuple[str, tuple[str, ...]], ...]
    chase_issues: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _FunctionAnalysis:
    pass


@dataclass(frozen=True)
class _FunctionAnalysisError(_FunctionAnalysis):
    target: str
    reason: str


@dataclass(frozen=True)
class _FunctionAnalysisNoop(_FunctionAnalysis):
    target: str
    reason: str


@dataclass(frozen=True)
class _FunctionAnalysisSuccess(_FunctionAnalysis):
    spec: _LoopRewriteSpec


@dataclass(frozen=True)
class _LoopBodyOutcome:
    kind: str
    guard_expr: object = None
    operation: object = None
    reason: str = ""


def _loop_body_error_reason(outcome: _LoopBodyOutcome) -> str:
    if outcome.kind == "error":
        return outcome.reason
    return ""


def _loop_body_guard_expr_optional(outcome: _LoopBodyOutcome):
    if outcome.kind == "guard":
        return outcome.guard_expr
    return None


def _loop_body_operation_optional(outcome: _LoopBodyOutcome):
    if outcome.kind == "operation":
        return outcome.operation
    return None


def _is_not_none_expression(value: object) -> bool:
    return value is not None


def _is_not_none_loop_operation(value: object) -> bool:
    return value is not None


@singledispatch
def _indented_block_optional(suite: object):
    never("unregistered runtime type", value_type=type(suite).__name__)


@_indented_block_optional.register(cst.IndentedBlock)
def _sd_reg_1(suite: cst.IndentedBlock):
    return suite


@_indented_block_optional.register(cst.SimpleStatementSuite)
def _sd_reg_2(suite: cst.SimpleStatementSuite):
    _ = suite
    return None


@singledispatch
def _star_param_optional(value: object):
    never("unregistered runtime type", value_type=type(value).__name__)


@_star_param_optional.register(cst.Param)
def _sd_reg_3(value: cst.Param):
    return value


@_star_param_optional.register(cst.MaybeSentinel)
def _sd_reg_4(value: cst.MaybeSentinel):
    _ = value
    return None


@singledispatch
def _simple_statement_line_optional(stmt: object):
    never("unregistered runtime type", value_type=type(stmt).__name__)


@_simple_statement_line_optional.register(cst.SimpleStatementLine)
def _sd_reg_5(stmt: cst.SimpleStatementLine):
    return stmt


def _simple_statement_line_none(value: object):
    _ = value
    return None


for _statement_type in (
    cst.For,
    cst.While,
    cst.If,
    cst.With,
    cst.Try,
    cst.TryStar,
    cst.Match,
    cst.FunctionDef,
    cst.ClassDef,
):
    _simple_statement_line_optional.register(_statement_type)(_simple_statement_line_none)


def _single_small_statement_optional(
    line: cst.SimpleStatementLine,
):
    if len(line.body) != 1:
        return None
    return line.body[0]


@singledispatch
def _is_continue_statement(stmt: object) -> bool:
    never("unregistered runtime type", value_type=type(stmt).__name__)


@_is_continue_statement.register(cst.Continue)
def _sd_reg_6(stmt: cst.Continue) -> bool:
    _ = stmt
    return True


def _never_continue_statement(value: object) -> bool:
    _ = value
    return False


for _small_statement_type in (
    type(None),
    cst.AnnAssign,
    cst.Assert,
    cst.Assign,
    cst.AugAssign,
    cst.Break,
    cst.Del,
    cst.Expr,
    cst.Global,
    cst.Import,
    cst.ImportFrom,
    cst.Nonlocal,
    cst.Pass,
    cst.Raise,
    cst.Return,
    cst.TypeAlias,
):
    _is_continue_statement.register(_small_statement_type)(_never_continue_statement)


@singledispatch
def _name_optional(node: object):
    never("unregistered runtime type", value_type=type(node).__name__)


@_name_optional.register(cst.Name)
def _sd_reg_7(node: cst.Name):
    return node


def _name_none(node: object):
    _ = node
    return None


for _assign_target_type in (cst.Attribute, cst.List, cst.Subscript, cst.Tuple):
    _name_optional.register(_assign_target_type)(_name_none)


@singledispatch
def _call_optional(node: object):
    never("unregistered runtime type", value_type=type(node).__name__)


@_call_optional.register(cst.Call)
def _sd_reg_8(node: cst.Call):
    return node


def _call_none(node: object):
    _ = node
    return None


for _call_nonmatch_type in (
    cst.Attribute,
    cst.Await,
    cst.BinaryOperation,
    cst.BooleanOperation,
    cst.Comparison,
    cst.Dict,
    cst.FormattedString,
    cst.GeneratorExp,
    cst.IfExp,
    cst.Integer,
    cst.List,
    cst.Name,
    cst.Set,
    cst.SimpleString,
    cst.Subscript,
    cst.Tuple,
):
    _call_optional.register(_call_nonmatch_type)(_call_none)


@singledispatch
def _attribute_optional(node: object):
    never("unregistered runtime type", value_type=type(node).__name__)


@_attribute_optional.register(cst.Attribute)
def _sd_reg_9(node: cst.Attribute):
    return node


def _attribute_none(node: object):
    _ = node
    return None


for _attribute_nonmatch_type in (
    cst.Call,
    cst.Name,
    cst.Subscript,
    cst.Tuple,
):
    _attribute_optional.register(_attribute_nonmatch_type)(_attribute_none)


@singledispatch
def _return_optional(stmt: object):
    never("unregistered runtime type", value_type=type(stmt).__name__)


@_return_optional.register(cst.Return)
def _sd_reg_10(stmt: cst.Return):
    return stmt


for _small_stmt_type in (
    cst.AnnAssign,
    cst.Assert,
    cst.Assign,
    cst.AugAssign,
    cst.Break,
    cst.Continue,
    cst.Del,
    cst.Expr,
    cst.Global,
    cst.Import,
    cst.ImportFrom,
    cst.Nonlocal,
    cst.Pass,
    cst.Raise,
    cst.TypeAlias,
):
    _return_optional.register(_small_stmt_type)(_name_none)


@singledispatch
def _subscript_optional(node: object):
    never("unregistered runtime type", value_type=type(node).__name__)


@_subscript_optional.register(cst.Subscript)
def _sd_reg_11(node: cst.Subscript):
    return node


for _subscript_nonmatch_type in (cst.Attribute, cst.Call, cst.Name, cst.Tuple):
    _subscript_optional.register(_subscript_nonmatch_type)(_name_none)


@singledispatch
def _binary_operation_optional(node: object):
    never("unregistered runtime type", value_type=type(node).__name__)


@_binary_operation_optional.register(cst.BinaryOperation)
def _sd_reg_12(node: cst.BinaryOperation):
    return node


for _binary_nonmatch_type in (
    cst.Attribute,
    cst.Call,
    cst.Comparison,
    cst.Dict,
    cst.GeneratorExp,
    cst.Name,
    cst.Subscript,
    cst.Tuple,
):
    _binary_operation_optional.register(_binary_nonmatch_type)(_name_none)


@singledispatch
def _subscript_index_value_optional(slice_value: object):
    never("unregistered runtime type", value_type=type(slice_value).__name__)


@_subscript_index_value_optional.register(cst.Index)
def _sd_reg_13(slice_value: cst.Index):
    return slice_value.value


@_subscript_index_value_optional.register(cst.Slice)
def _sd_reg_14(slice_value: cst.Slice):
    _ = slice_value
    return None


@singledispatch
def _for_loop_optional(node: object):
    never("unregistered runtime type", value_type=type(node).__name__)


@_for_loop_optional.register(cst.For)
def _sd_reg_15(node: cst.For):
    return node


@_for_loop_optional.register(cst.While)
def _sd_reg_16(node: cst.While):
    _ = node
    return None


@singledispatch
def _analysis_success_optional(analysis: object):
    never("unregistered runtime type", value_type=type(analysis).__name__)


@_analysis_success_optional.register(_FunctionAnalysisSuccess)
def _sd_reg_17(analysis: _FunctionAnalysisSuccess):
    return analysis


def _analysis_success_none(value: object):
    _ = value
    return None


for _analysis_type in (_FunctionAnalysisError, _FunctionAnalysisNoop):
    _analysis_success_optional.register(_analysis_type)(_analysis_success_none)


def _is_not_none_analysis_success(analysis: object) -> bool:
    return analysis is not None


@singledispatch
def _analysis_error_optional(analysis: object):
    never("unregistered runtime type", value_type=type(analysis).__name__)


@_analysis_error_optional.register(_FunctionAnalysisError)
def _sd_reg_18(analysis: _FunctionAnalysisError):
    return analysis


def _analysis_error_none(value: object):
    _ = value
    return None


for _analysis_type in (_FunctionAnalysisSuccess, _FunctionAnalysisNoop):
    _analysis_error_optional.register(_analysis_type)(_analysis_error_none)


class _SideEffectSafetyVisitor(cst.CSTVisitor):
    def __init__(self) -> None:
        self.reason = ""

    def _mark(self, reason: str) -> None:
        if not self.reason:
            self.reason = reason

    def visit_Call(self, node: cst.Call) -> None:
        self._mark("calls are not side-effect-safe")

    def visit_ListComp(self, node: cst.ListComp) -> None:
        self._mark("list comprehensions are not side-effect-safe")

    def visit_SetComp(self, node: cst.SetComp) -> None:
        self._mark("set comprehensions are not side-effect-safe")

    def visit_DictComp(self, node: cst.DictComp) -> None:
        self._mark("dict comprehensions are not side-effect-safe")

    def visit_GeneratorExp(self, node: cst.GeneratorExp) -> None:
        self._mark("generator comprehensions are not side-effect-safe")

    def visit_NamedExpr(self, node: cst.NamedExpr) -> None:
        self._mark("assignment expressions are not side-effect-safe")

    def visit_Await(self, node: cst.Await) -> None:
        self._mark("await expressions are not side-effect-safe")

    def visit_Yield(self, node: cst.Yield) -> None:
        self._mark("yield expressions are not side-effect-safe")

    def visit_From(self, node: cst.From) -> None:
        self._mark("yield from expressions are not side-effect-safe")


class _LoopHazardVisitor(cst.CSTVisitor):
    def __init__(self) -> None:
        self.reason = ""

    def _mark(self, reason: str) -> None:
        if not self.reason:
            self.reason = reason

    def visit_For(self, node: cst.For) -> None:
        self._mark("nested loops are not supported")

    def visit_While(self, node: cst.While) -> None:
        self._mark("nested loops are not supported")

    def visit_Break(self, node: cst.Break) -> None:
        self._mark("break is not supported in loop_generator mode")

    def visit_Return(self, node: cst.Return) -> None:
        self._mark("return is not supported inside targeted loops")

    def visit_Raise(self, node: cst.Raise) -> None:
        self._mark("raise is not supported inside targeted loops")

    def visit_Try(self, node: cst.Try) -> None:
        self._mark("try is not supported inside targeted loops")

    def visit_With(self, node: cst.With) -> None:
        self._mark("with is not supported inside targeted loops")

    def visit_Yield(self, node: cst.Yield) -> None:
        self._mark("yield is not supported inside targeted loops")

    def visit_From(self, node: cst.From) -> None:
        self._mark("yield from is not supported inside targeted loops")


def _code_for_node(module: cst.Module, node: cst.CSTNode) -> str:
    return module.code_for_node(node).strip()


def _is_docstring_statement(stmt: cst.BaseStatement) -> bool:
    return cst_shared.is_docstring_statement(stmt)


def _is_side_effect_safe_expression(
    expr: cst.BaseExpression,
) -> tuple[bool, str]:
    visitor = _SideEffectSafetyVisitor()
    expr.visit(visitor)
    return not visitor.reason, visitor.reason


def _extract_subscript_key(target: cst.Subscript):
    if len(target.slice) != 1:
        return None
    first_slice = target.slice[0]
    return _subscript_index_value_optional(first_slice.slice)


def _contains_loop_hazards(loop: cst.For) -> str:
    loop_body = _indented_block_optional(loop.body)
    if loop_body is None:
        return "loop body must be a block"
    return next(
        filter(
            bool,
            map(_loop_hazard_reason, loop_body.body),
        ),
        "",
    )


def _loop_hazard_reason(stmt: cst.BaseStatement) -> str:
    check_deadline()
    visitor = _LoopHazardVisitor()
    stmt.visit(visitor)
    return visitor.reason


def _parameter_call_args(params: cst.Parameters) -> tuple[cst.Arg, ...]:
    args: list[cst.Arg] = list(
        map(
            _parameter_positional_arg,
            chain(params.posonly_params, params.params),
        )
    )
    star_param = _star_param_optional(params.star_arg)
    if star_param is not None:
        args.append(cst.Arg(star="*", value=cst.Name(star_param.name.value)))
    args.extend(map(_parameter_keyword_arg, params.kwonly_params))
    if params.star_kwarg is not None:
        args.append(
            cst.Arg(
                star="**",
                value=cst.Name(params.star_kwarg.name.value),
            )
        )
    return tuple(args)


def _parameter_positional_arg(param: cst.Param) -> cst.Arg:
    check_deadline()
    return cst.Arg(value=cst.Name(param.name.value))


def _parameter_keyword_arg(param: cst.Param) -> cst.Arg:
    check_deadline()
    name = param.name.value
    return cst.Arg(
        keyword=cst.Name(name),
        value=cst.Name(name),
    )


def _is_simple_continue_guard(stmt: cst.If) -> bool:
    branch_body = _indented_block_optional(stmt.body)
    body = tuple(branch_body.body) if branch_body is not None else ()
    line = _simple_statement_line_optional(body[0]) if len(body) == 1 else None
    only = _single_small_statement_optional(line) if line is not None else None
    return stmt.orelse is None and _is_continue_statement(only)


def _clone_expression(expr: cst.BaseExpression) -> cst.BaseExpression:
    return cast(cst.BaseExpression, expr.deep_clone())


def _clone_arg(arg: cst.Arg) -> cst.Arg:
    return cast(cst.Arg, arg.deep_clone())


def _string_literal(value: str) -> cst.SimpleString:
    return cst.SimpleString(repr(value))


def _loop_transform_annotation() -> cst.Annotation:
    return cst.Annotation(
        annotation=cst.Subscript(
            value=cst.Name("Iterator"),
            slice=[
                cst.SubscriptElement(
                    slice=cst.Index(value=cst.Name("_LoopTransformOp"))
                )
            ],
        )
    )


def _join_guard_expressions(guards: tuple[cst.BaseExpression, ...]) -> cst.BaseExpression:
    return reduce(
        _joined_guard_expression,
        guards[1:],
        _clone_expression(guards[0]),
    )


def _joined_guard_expression(
    left: cst.BaseExpression,
    right: cst.BaseExpression,
) -> cst.BaseExpression:
    check_deadline()
    return cst.BooleanOperation(
        left=left,
        operator=cst.Or(),
        right=_clone_expression(right),
    )


def _yield_operation_statement(
    *,
    loop_var: str,
    operation: _LoopOperation,
) -> cst.SimpleStatementLine:
    args: list[cst.Arg] = [
        cst.Arg(keyword=cst.Name("item"), value=cst.Name(loop_var)),
    ]
    op_ctor = ""
    if operation.kind == "LIST_APPEND":
        op_ctor = "_LoopListAppendOp"
        args.extend(
            [
                cst.Arg(
                    keyword=cst.Name("value"),
                    value=_clone_expression(operation.value_expr),
                ),
                cst.Arg(
                    keyword=cst.Name("target"),
                    value=_string_literal(operation.target),
                ),
            ]
        )
    elif operation.kind == "SET_ADD":
        op_ctor = "_LoopSetAddOp"
        args.extend(
            [
                cst.Arg(
                    keyword=cst.Name("value"),
                    value=_clone_expression(operation.value_expr),
                ),
                cst.Arg(
                    keyword=cst.Name("target"),
                    value=_string_literal(operation.target),
                ),
            ]
        )
    elif operation.kind == "DICT_SET":
        op_ctor = "_LoopDictSetOp"
        args.extend(
            [
                cst.Arg(
                    keyword=cst.Name("key"),
                    value=_clone_expression(operation.key_expr),
                ),
                cst.Arg(
                    keyword=cst.Name("value"),
                    value=_clone_expression(operation.value_expr),
                ),
                cst.Arg(
                    keyword=cst.Name("target"),
                    value=_string_literal(operation.target),
                ),
            ]
        )
    else:
        op_ctor = "_LoopReduceOp"
        args.extend(
            [
                cst.Arg(
                    keyword=cst.Name("target"),
                    value=_string_literal(operation.target),
                ),
                cst.Arg(
                    keyword=cst.Name("operator"),
                    value=_string_literal(operation.operator),
                ),
                cst.Arg(
                    keyword=cst.Name("operand"),
                    value=_clone_expression(operation.value_expr),
                ),
            ]
        )
    return cst.SimpleStatementLine(
        body=[
            cst.Expr(
                value=cst.Yield(
                    value=cst.Call(
                        func=cst.Name(op_ctor),
                        args=args,
                    )
                )
            )
        ]
    )


def _build_helper_function(spec: _LoopRewriteSpec) -> cst.FunctionDef:
    helper_body: list[cst.BaseStatement] = []
    loop_iter: cst.BaseExpression = _clone_expression(spec.iter_expr)
    if spec.guard_exprs:
        joined_guards = _join_guard_expressions(spec.guard_exprs)
        filter_call = cst.Call(
            func=cst.Name("filter"),
            args=[
                cst.Arg(
                    value=cst.Lambda(
                        params=cst.Parameters(
                            params=[cst.Param(name=cst.Name(spec.loop_var))]
                        ),
                        body=cst.UnaryOperation(
                            operator=cst.Not(),
                            expression=joined_guards,
                        ),
                    )
                ),
                cst.Arg(value=_clone_expression(spec.iter_expr)),
            ],
        )
        helper_body.append(
            cst.SimpleStatementLine(
                body=[
                    cst.Assign(
                        targets=[cst.AssignTarget(target=cst.Name("_filtered_iter"))],
                        value=filter_call,
                    )
                ]
            )
        )
        loop_iter = cst.Name("_filtered_iter")

    loop_body = [
        _yield_operation_statement(loop_var=spec.loop_var, operation=operation)
        for operation in spec.operations
    ]
    helper_body.append(
        cst.For(
            target=cst.Name(spec.loop_var),
            iter=loop_iter,
            body=cst.IndentedBlock(body=loop_body),
        )
    )
    return cst.FunctionDef(
        name=cst.Name(spec.helper_name),
        params=cast(cst.Parameters, spec.params.deep_clone()),
        returns=_loop_transform_annotation(),
        body=cst.IndentedBlock(body=helper_body),
    )


def _find_import_insert_index(body: list[cst.CSTNode]) -> int:
    return cst_shared.find_import_insert_index(
        body,
        check_deadline_fn=check_deadline,
    )


def _has_import_from(body: list[cst.CSTNode], *, module_name: str, symbol: str) -> bool:
    return cst_shared.has_import_from(
        body,
        module_name=module_name,
        symbol=symbol,
        check_deadline_fn=check_deadline,
    )


@singledispatch
def _assigned_target_name_optional(stmt: object):
    never("unregistered runtime type", value_type=type(stmt).__name__)


@_assigned_target_name_optional.register(cst.Assign)
def _sd_reg_19(stmt: cst.Assign):
    if len(stmt.targets) != 1:
        return ""
    name_target = _name_optional(stmt.targets[0].target)
    if not name_target:
        return ""
    return name_target.value


def _assigned_target_name_none(value: object):
    _ = value
    return ""


for _small_statement_type in (
    cst.AnnAssign,
    cst.Assert,
    cst.AugAssign,
    cst.Break,
    cst.Continue,
    cst.Del,
    cst.Expr,
    cst.Global,
    cst.Import,
    cst.ImportFrom,
    cst.Nonlocal,
    cst.Pass,
    cst.Raise,
    cst.Return,
    cst.TypeAlias,
):
    _assigned_target_name_optional.register(_small_statement_type)(_assigned_target_name_none)


@singledispatch
def _defined_top_level_name(stmt: object) -> str:
    never("unregistered runtime type", value_type=type(stmt).__name__)


@_defined_top_level_name.register(cst.ClassDef)
def _sd_reg_20(stmt: cst.ClassDef) -> str:
    return stmt.name.value


@_defined_top_level_name.register(cst.FunctionDef)
def _sd_reg_21(stmt: cst.FunctionDef) -> str:
    return stmt.name.value


@_defined_top_level_name.register(cst.SimpleStatementLine)
def _sd_reg_22(stmt: cst.SimpleStatementLine) -> str:
    only_stmt = _single_small_statement_optional(stmt)
    assigned_name = (
        _assigned_target_name_optional(only_stmt)
        if only_stmt is not None
        else ""
    )
    return assigned_name or ""


def _empty_defined_top_level_name(value: object) -> str:
    _ = value
    return ""


for _statement_type in (
    cst.For,
    cst.While,
    cst.If,
    cst.With,
    cst.Try,
    cst.TryStar,
    cst.Match,
    cst.EmptyLine,
):
    _defined_top_level_name.register(_statement_type)(_empty_defined_top_level_name)


def _ensure_loop_generator_scaffolding(module: cst.Module) -> cst.Module:
    body = list(module.body)
    insert_idx = _find_import_insert_index(body)

    if not _has_import_from(body, module_name="dataclasses", symbol="dataclass"):
        body.insert(
            insert_idx,
            cst.SimpleStatementLine(
                [
                    cst.ImportFrom(
                        module=cst.Name("dataclasses"),
                        names=[cst.ImportAlias(name=cst.Name("dataclass"))],
                    )
                ]
            ),
        )
        insert_idx += 1
    if not _has_import_from(body, module_name="typing", symbol="Iterator"):
        body.insert(
            insert_idx,
            cst.SimpleStatementLine(
                [
                    cst.ImportFrom(
                        module=cst.Name("typing"),
                        names=[cst.ImportAlias(name=cst.Name("Iterator"))],
                    )
                ]
            ),
        )
        insert_idx += 1

    existing_names = set(
        filter(
            bool,
            map(_defined_top_level_name, body),
        )
    )
    scaffold_nodes = _loop_carrier_nodes()
    name_node_pairs = tuple(map(_node_name_pair, scaffold_nodes))
    nodes_to_insert = list(
        map(
            lambda item: item[1],
            filter(
                lambda item: item[0] and item[0] not in existing_names,
                name_node_pairs,
            ),
        )
    )
    existing_names.update(
        map(
            lambda item: item[0],
            filter(
                lambda item: item[0] and item[0] not in existing_names,
                name_node_pairs,
            ),
        )
    )
    if not nodes_to_insert:
        return module.with_changes(body=body)
    body[insert_idx:insert_idx] = [cst.EmptyLine(), *nodes_to_insert, cst.EmptyLine()]
    return module.with_changes(body=body)


def _node_name_pair(node: cst.CSTNode) -> tuple[str, cst.CSTNode]:
    check_deadline()
    return _defined_top_level_name(node), node


class _FunctionIndexVisitor(cst.CSTVisitor):
    def __init__(self) -> None:
        self._stack: list[str] = []
        self.entries: list[_FunctionIndexEntry] = []

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        self._stack.append(node.name.value)
        return True

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        self._stack.pop()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        qualname = ".".join([*self._stack, node.name.value])
        self.entries.append(_FunctionIndexEntry(qualname=qualname, node=node))
        self._stack.append(node.name.value)
        return True

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        self._stack.pop()


def _function_non_doc_body(node: cst.FunctionDef) -> tuple[cst.BaseStatement, ...]:
    body_block = _indented_block_optional(node.body)
    body = list(body_block.body) if body_block is not None else []
    if body and _is_docstring_statement(body[0]):
        return tuple(body[1:])
    return tuple(body)


def _trampoline_helper_name(node: cst.FunctionDef) -> str:
    body = _function_non_doc_body(node)
    line = _simple_statement_line_optional(body[0]) if len(body) == 1 else None
    only_stmt = _single_small_statement_optional(line) if line is not None else None
    matches_return_call = bool(
        only_stmt is not None
        and cst_matchers.matches(
            only_stmt,
            cst_matchers.Return(value=cst_matchers.Call(func=cst_matchers.Name())),
        )
    )
    ret = _return_optional(only_stmt) if only_stmt is not None else None
    ret_call = _call_optional(ret.value) if ret is not None else None
    helper_name_node = _name_optional(ret_call.func) if ret_call is not None else None
    helper_name = helper_name_node.value if helper_name_node is not None else ""
    match = _LOOP_HELPER_PATTERN.match(helper_name)
    source_name_match = _LOOP_HELPER_PATTERN.match(node.name.value)
    source_name = (
        source_name_match.group("name")
        if source_name_match
        else node.name.value
    )
    matches_source = bool(match and match.group("name") == source_name)
    return (
        helper_name
        if line is not None
        and matches_return_call
        and ret is not None
        and ret_call is not None
        and helper_name_node is not None
        and matches_source
        else ""
    )


def _build_function_index(
    module: cst.Module,
) -> tuple[dict[str, _FunctionIndexEntry], dict[str, tuple[str, ...]]]:
    visitor = _FunctionIndexVisitor()
    module.visit(visitor)
    by_qualname = {entry.qualname: entry for entry in visitor.entries}
    by_name: dict[str, list[str]] = {}
    tuple(map(lambda entry: _append_function_name_entry(by_name, entry), visitor.entries))
    return by_qualname, {name: tuple(values) for name, values in by_name.items()}


def _append_function_name_entry(
    by_name: dict[str, list[str]],
    entry: _FunctionIndexEntry,
) -> None:
    check_deadline()
    by_name.setdefault(entry.node.name.value, []).append(entry.qualname)


def _follow_trampoline_chain(
    *,
    start_qualname: str,
    by_qualname: dict[str, _FunctionIndexEntry],
) -> tuple[str, str]:
    return _follow_trampoline_chain_recursive(
        current=start_qualname,
        seen={start_qualname},
        depth=0,
        by_qualname=by_qualname,
    )


def _follow_trampoline_chain_recursive(
    *,
    current: str,
    seen: set[str],
    depth: int,
    by_qualname: dict[str, _FunctionIndexEntry],
) -> tuple[str, str]:
    if depth >= _MAX_HELPER_CHASE_DEPTH:
        return current, f"helper chase exceeded max depth {_MAX_HELPER_CHASE_DEPTH}"
    check_deadline()
    next_target = _next_trampoline_target(current=current, by_qualname=by_qualname)
    if not next_target:
        return current, ""
    if next_target in seen:
        return current, f"helper chase cycle detected at `{next_target}`"
    return _follow_trampoline_chain_recursive(
        current=next_target,
        seen={*seen, next_target},
        depth=depth + 1,
        by_qualname=by_qualname,
    )


def _next_trampoline_target(
    *,
    current: str,
    by_qualname: dict[str, _FunctionIndexEntry],
) -> str:
    entry = by_qualname.get(current)
    if entry is None:
        return ""
    helper_name = _trampoline_helper_name(entry.node)
    if not helper_name:
        return ""
    parent = current.rpartition(".")[0]
    scoped_target = f"{parent}.{helper_name}" if parent else helper_name
    if scoped_target in by_qualname:
        return scoped_target
    if helper_name in by_qualname:
        return helper_name
    return scoped_target


def _resolve_loop_generator_targets(
    *,
    module: cst.Module,
    requested_targets: set[str],
) -> _TargetResolution:
    by_qualname, by_name = _build_function_index(module)
    effective_targets: set[str] = set()
    root_to_effective: list[tuple[str, tuple[str, ...]]] = []
    effective_to_roots: dict[str, set[str]] = {}
    chase_issues: list[tuple[str, str]] = []

    for root in sorted(requested_targets):
        check_deadline()
        candidates = _resolution_root_candidates(
            root=root,
            by_qualname=by_qualname,
            by_name=by_name,
        )
        outcomes = tuple(
            map(
                lambda candidate: _resolution_candidate_outcome(
                    root=root,
                    candidate=candidate,
                    by_qualname=by_qualname,
                ),
                candidates,
            )
        )
        root_chase_issues = tuple(
            filter(
                bool,
                map(_resolution_issue_optional, outcomes),
            )
        )
        chase_issues.extend(map(lambda issue: (root, issue), root_chase_issues))
        resolved = tuple(
            sorted(
                set(
                    filter(
                        bool,
                        map(_resolution_terminal_optional, outcomes),
                    )
                )
            )
        )
        effective_targets.update(resolved)
        tuple(map(lambda target: effective_to_roots.setdefault(target, set()).add(root), resolved))
        root_to_effective.append((root, resolved))

    effective_to_roots_items = tuple(
        (target, tuple(sorted(roots)))
        for target, roots in sorted(effective_to_roots.items())
    )
    return _TargetResolution(
        effective_targets=tuple(sorted(effective_targets)),
        root_to_effective=tuple(root_to_effective),
        effective_to_roots=effective_to_roots_items,
        chase_issues=tuple(chase_issues),
    )


def _resolution_root_candidates(
    *,
    root: str,
    by_qualname: dict[str, _FunctionIndexEntry],
    by_name: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    if root in by_qualname:
        return (root,)
    return by_name.get(root, ())


def _resolution_candidate_outcome(
    *,
    root: str,
    candidate: str,
    by_qualname: dict[str, _FunctionIndexEntry],
) -> tuple[str, str]:
    check_deadline()
    terminal, issue = _follow_trampoline_chain(
        start_qualname=candidate,
        by_qualname=by_qualname,
    )
    if issue:
        return "", f"{root}: {issue}"
    return terminal, ""


def _resolution_terminal_optional(outcome: tuple[str, str]) -> str:
    terminal, _issue = outcome
    return terminal


def _resolution_issue_optional(outcome: tuple[str, str]) -> str:
    _terminal, issue = outcome
    return issue


def _rewrite_plan_with_chase_context(
    plan: RewritePlanEntry,
    *,
    effective_to_roots: dict[str, tuple[str, ...]],
) -> RewritePlanEntry:
    forwarded = tuple(
        filter(
            lambda root: root != plan.target,
            effective_to_roots.get(plan.target, ()),
        )
    )
    if not forwarded:
        return plan
    summary = f"{plan.summary} (helper chase from: {', '.join(forwarded)})"
    return RewritePlanEntry(
        kind=plan.kind,
        status=plan.status,
        target=plan.target,
        summary=summary,
        non_rewrite_reasons=list(plan.non_rewrite_reasons),
    )


def _handler_body_statements(
    handler: cst.ExceptHandler,
) -> tuple[cst.BaseStatement, ...]:
    check_deadline()
    body_block = _indented_block_optional(handler.body)
    return tuple(body_block.body) if body_block is not None else ()


def _optional_suite_statements(suite: object) -> tuple[tuple[cst.BaseStatement, ...], ...]:
    if suite is None:
        return ()
    body = getattr(suite, "body")
    body_block = _indented_block_optional(body)
    return ((tuple(body_block.body),) if body_block is not None else ())


class _LoopGeneratorTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (cst_metadata.PositionProvider,)

    def __init__(
        self,
        *,
        module: cst.Module,
        targets: set[str],
        target_loop_lines: set[int],
    ) -> None:
        self._module = module
        self._targets = targets
        self._target_loop_lines = target_loop_lines
        self._stack: list[str] = []
        self.changed = False
        self.errors: list[str] = []
        self.rewrite_plans: list[RewritePlanEntry] = []
        self.matched_targets: set[str] = set()

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        self._stack.append(node.name.value)
        return True

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.CSTNode:
        if self._stack:
            self._stack.pop()
        return updated_node

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        self._stack.append(node.name.value)
        return True

    def leave_FunctionDef(
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef,
    ) -> cst.CSTNode:
        qualname = ".".join(self._stack)
        target_key = qualname if qualname in self._targets else ""

        output: cst.CSTNode = updated_node
        if target_key:
            self.matched_targets.add(target_key)
            if original_node.asynchronous is not None:
                reason = f"{qualname}: async functions are not supported in loop_generator mode"
                self.errors.append(reason)
                self.rewrite_plans.append(
                    RewritePlanEntry(
                        kind="LOOP_GENERATOR",
                        status="ABSTAINED",
                        target=qualname,
                        summary="Loop generator rewrite was not applied.",
                        non_rewrite_reasons=[reason],
                    )
                )
            else:
                analysis = self._analyze_function(original_node, qualname)
                output = self._apply_function_analysis(
                    analysis,
                    updated_node=updated_node,
                )

        if self._stack:
            self._stack.pop()
        return output

    @singledispatchmethod
    def _apply_function_analysis(
        self,
        analysis: _FunctionAnalysis,
        *,
        updated_node: cst.FunctionDef,
    ) -> cst.CSTNode:
        never("unregistered runtime type", value_type=type(analysis).__name__)

    @_apply_function_analysis.register
    def _(
        self,
        analysis: _FunctionAnalysisError,
        *,
        updated_node: cst.FunctionDef,
    ) -> cst.CSTNode:
        _ = updated_node
        reason = analysis.reason
        self.errors.append(reason)
        self.rewrite_plans.append(
            RewritePlanEntry(
                kind="LOOP_GENERATOR",
                status="ABSTAINED",
                target=analysis.target,
                summary="Loop generator rewrite was not applied.",
                non_rewrite_reasons=[reason],
            )
        )
        return updated_node

    @_apply_function_analysis.register
    def _(
        self,
        analysis: _FunctionAnalysisNoop,
        *,
        updated_node: cst.FunctionDef,
    ) -> cst.CSTNode:
        _ = updated_node
        self.rewrite_plans.append(
            RewritePlanEntry(
                kind="LOOP_GENERATOR",
                status="noop",
                target=analysis.target,
                summary=analysis.reason,
            )
        )
        return updated_node

    @_apply_function_analysis.register
    def _(
        self,
        analysis: _FunctionAnalysisSuccess,
        *,
        updated_node: cst.FunctionDef,
    ) -> cst.CSTNode:
        spec = analysis.spec
        helper_node = _build_helper_function(spec)
        replacement = self._rewrite_target_function(updated_node, spec)
        self.changed = True
        self.rewrite_plans.append(
            RewritePlanEntry(
                kind="LOOP_GENERATOR",
                status="applied",
                target=spec.qualname,
                summary=(
                    f"Rewrote loop at line {spec.loop_line} into "
                    f"{spec.helper_name}."
                ),
            )
        )
        return cst.FlattenSentinel([helper_node, replacement])

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.CSTNode:
        if not self.changed:
            return updated_node
        return _ensure_loop_generator_scaffolding(updated_node)

    def _rewrite_target_function(
        self,
        node: cst.FunctionDef,
        spec: _LoopRewriteSpec,
    ) -> cst.FunctionDef:
        body_block = _indented_block_optional(node.body)
        if body_block is None:
            return node
        existing = list(body_block.body)
        new_body: list[cst.BaseStatement] = []
        if existing and _is_docstring_statement(existing[0]):
            new_body.append(existing[0])
        new_body.append(
            cst.SimpleStatementLine(
                body=[
                    cst.Return(
                        value=cst.Call(
                            func=cst.Name(spec.helper_name),
                            args=[_clone_arg(arg) for arg in spec.call_args],
                        )
                    )
                ]
            )
        )
        return node.with_changes(body=node.body.with_changes(body=new_body))

    def _analyze_function(self, node: cst.FunctionDef, qualname: str) -> _FunctionAnalysis:
        body_block = _indented_block_optional(node.body)
        if body_block is None:
            return _FunctionAnalysisError(
                target=qualname,
                reason=f"{qualname}: function body must be a block",
            )
        body = list(body_block.body)
        non_doc_body = body[1:] if body and _is_docstring_statement(body[0]) else body
        if not non_doc_body:
            return _FunctionAnalysisError(
                target=qualname,
                reason=f"{qualname}: function body is empty",
            )
        loop_candidates = self._collect_loop_candidates(tuple(non_doc_body))
        if not loop_candidates:
            if self._is_already_rewritten(non_doc_body, function_name=node.name.value):
                return _FunctionAnalysisNoop(
                    target=qualname,
                    reason=f"{qualname}: already rewritten for loop_generator mode",
                )
            if _LOOP_HELPER_PATTERN.match(node.name.value):
                return _FunctionAnalysisNoop(
                    target=qualname,
                    reason=f"{qualname}: helper chain terminal has no remaining eligible loops",
                )
            return _FunctionAnalysisError(
                target=qualname,
                reason=f"{qualname}: no loop found",
            )

        if self._target_loop_lines:
            by_line = tuple(
                filter(
                    lambda candidate: candidate.line in self._target_loop_lines,
                    loop_candidates,
                )
            )
            if not by_line:
                return _FunctionAnalysisError(
                    target=qualname,
                    reason=f"{qualname}: no loop matched requested target_loop_lines",
                )
            if len(by_line) > 1:
                return _FunctionAnalysisError(
                    target=qualname,
                    reason=f"{qualname}: multiple loops matched target_loop_lines",
                )
            return self._analyze_loop_candidate(
                by_line[0],
                qualname=qualname,
                function_name=node.name.value,
                params=node.params,
            )

        first_outcome = self._analyze_loop_candidate(
            loop_candidates[0],
            qualname=qualname,
            function_name=node.name.value,
            params=node.params,
        )
        first_success = _analysis_success_optional(first_outcome)
        if first_success is not None:
            return first_success
        first_error = _analysis_error_optional(first_outcome)
        if first_error is None:
            return first_outcome
        fallback_successes = tuple(
            filter(
                _is_not_none_analysis_success,
                map(
                    _analysis_success_optional,
                    map(
                        lambda candidate: self._analyze_loop_candidate(
                            candidate,
                            qualname=qualname,
                            function_name=node.name.value,
                            params=node.params,
                        ),
                        loop_candidates[1:],
                    ),
                ),
            )
        )
        if fallback_successes:
            return fallback_successes[0]

        if _LOOP_HELPER_PATTERN.match(node.name.value):
            return _FunctionAnalysisNoop(
                target=qualname,
                reason=f"{qualname}: helper chain terminal has no remaining eligible loops",
            )
        return first_error

    def _suite_statements(self, suite: cst.BaseSuite) -> tuple[cst.BaseStatement, ...]:
        block = _indented_block_optional(suite)
        body = block.body if block is not None else ()
        return tuple(body)

    @singledispatchmethod
    def _child_statement_blocks(
        self,
        stmt: cst.BaseStatement,
    ) -> tuple[tuple[cst.BaseStatement, ...], ...]:
        never("unregistered runtime type", value_type=type(stmt).__name__)

    @_child_statement_blocks.register
    def _(self, stmt: cst.For) -> tuple[tuple[cst.BaseStatement, ...], ...]:
        blocks = [self._suite_statements(stmt.body)]
        if stmt.orelse is not None:
            blocks.append(self._suite_statements(stmt.orelse.body))
        return tuple(blocks)

    @_child_statement_blocks.register
    def _(self, stmt: cst.While) -> tuple[tuple[cst.BaseStatement, ...], ...]:
        blocks = [self._suite_statements(stmt.body)]
        if stmt.orelse is not None:
            blocks.append(self._suite_statements(stmt.orelse.body))
        return tuple(blocks)

    @_child_statement_blocks.register
    def _(self, stmt: cst.If) -> tuple[tuple[cst.BaseStatement, ...], ...]:
        blocks = [self._suite_statements(stmt.body)]
        if stmt.orelse is not None:
            blocks.append(self._suite_statements(stmt.orelse.body))
        return tuple(blocks)

    @_child_statement_blocks.register
    def _(self, stmt: cst.With) -> tuple[tuple[cst.BaseStatement, ...], ...]:
        return (self._suite_statements(stmt.body),)

    @_child_statement_blocks.register
    def _(self, stmt: cst.Try) -> tuple[tuple[cst.BaseStatement, ...], ...]:
        return (
            self._suite_statements(stmt.body),
            *tuple(map(_handler_body_statements, stmt.handlers)),
            *_optional_suite_statements(stmt.orelse),
            *_optional_suite_statements(stmt.finalbody),
        )

    @_child_statement_blocks.register
    def _(self, stmt: cst.Match) -> tuple[tuple[cst.BaseStatement, ...], ...]:
        return tuple(self._suite_statements(case.body) for case in stmt.cases)

    @_child_statement_blocks.register
    def _(self, stmt: cst.TryStar) -> tuple[tuple[cst.BaseStatement, ...], ...]:
        _ = stmt
        return ()

    @_child_statement_blocks.register
    def _(self, stmt: cst.SimpleStatementLine) -> tuple[tuple[cst.BaseStatement, ...], ...]:
        _ = stmt
        return ()

    @_child_statement_blocks.register
    def _(self, stmt: cst.FunctionDef) -> tuple[tuple[cst.BaseStatement, ...], ...]:
        _ = stmt
        return ()

    @_child_statement_blocks.register
    def _(self, stmt: cst.ClassDef) -> tuple[tuple[cst.BaseStatement, ...], ...]:
        return ()

    @singledispatchmethod
    def _loop_candidate_statement_optional(
        self,
        stmt: cst.BaseStatement,
    ):
        never("unregistered runtime type", value_type=type(stmt).__name__)

    @_loop_candidate_statement_optional.register
    def _(self, stmt: cst.For):
        return stmt

    @_loop_candidate_statement_optional.register
    def _(self, stmt: cst.While):
        return stmt

    @_loop_candidate_statement_optional.register
    def _(self, stmt: cst.If):
        _ = stmt
        return None

    @_loop_candidate_statement_optional.register
    def _(self, stmt: cst.With):
        _ = stmt
        return None

    @_loop_candidate_statement_optional.register
    def _(self, stmt: cst.Try):
        _ = stmt
        return None

    @_loop_candidate_statement_optional.register
    def _(self, stmt: cst.TryStar):
        _ = stmt
        return None

    @_loop_candidate_statement_optional.register
    def _(self, stmt: cst.Match):
        _ = stmt
        return None

    @_loop_candidate_statement_optional.register
    def _(self, stmt: cst.SimpleStatementLine):
        _ = stmt
        return None

    @_loop_candidate_statement_optional.register
    def _(self, stmt: cst.FunctionDef):
        _ = stmt
        return None

    @_loop_candidate_statement_optional.register
    def _(self, stmt: cst.ClassDef):
        _ = stmt
        return None

    def _collect_loop_candidates_from_statement(
        self,
        stmt: cst.BaseStatement,
        out: list[_LoopCandidate],
    ) -> None:
        loop_stmt = self._loop_candidate_statement_optional(stmt)
        if loop_stmt is not None:
            pos = self.get_metadata(
                cst_metadata.PositionProvider,
                loop_stmt,
            )
            out.append(
                _LoopCandidate(
                    loop_node=loop_stmt,
                    line=pos.start.line,
                )
            )
        for child_stmt in chain.from_iterable(self._child_statement_blocks(stmt)):
            check_deadline()
            self._collect_loop_candidates_from_statement(child_stmt, out)

    def _collect_loop_candidates(
        self,
        statements: tuple[cst.BaseStatement, ...],
    ) -> tuple[_LoopCandidate, ...]:
        out: list[_LoopCandidate] = []
        tuple(map(lambda stmt: self._collect_loop_candidates_from_statement(stmt, out), statements))
        return tuple(out)

    def _analyze_loop_candidate(
        self,
        candidate: _LoopCandidate,
        *,
        qualname: str,
        function_name: str,
        params: cst.Parameters,
    ) -> _FunctionAnalysis:
        for_loop = _for_loop_optional(candidate.loop_node)
        if for_loop is None:
            return _FunctionAnalysisError(
                target=qualname,
                reason=f"{qualname}: only for-loops are supported",
            )
        return self._analyze_for_loop(
            for_loop,
            qualname=qualname,
            function_name=function_name,
            loop_line=candidate.line,
            params=params,
        )

    def _analyze_for_loop(
        self,
        loop: cst.For,
        *,
        qualname: str,
        function_name: str,
        loop_line: int,
        params: cst.Parameters,
    ) -> _FunctionAnalysis:
        if loop.orelse is not None:
            return _FunctionAnalysisError(
                target=qualname,
                reason=f"{qualname}: for-else loops are not supported",
            )
        if loop.asynchronous is not None:
            return _FunctionAnalysisError(
                target=qualname,
                reason=f"{qualname}: async-for loops are not supported",
            )
        loop_target = _name_optional(loop.target)
        if loop_target is None:
            return _FunctionAnalysisError(
                target=qualname,
                reason=f"{qualname}: loop target must be a simple name",
            )
        hazard = _contains_loop_hazards(loop)
        if hazard:
            return _FunctionAnalysisError(
                target=qualname,
                reason=f"{qualname}: {hazard}",
            )

        loop_var = loop_target.value
        guard_exprs: list[cst.BaseExpression] = []
        operations: list[_LoopOperation] = []
        loop_body = _indented_block_optional(loop.body)
        if loop_body is None:
            return _FunctionAnalysisError(
                target=qualname,
                reason=f"{qualname}: loop body must be a block",
            )
        outcomes = tuple(
            map(
                lambda stmt: self._analyze_for_body_statement(stmt, qualname=qualname),
                loop_body.body,
            )
        )
        first_error_reason = next(
            filter(
                bool,
                map(_loop_body_error_reason, outcomes),
            ),
            "",
        )
        if first_error_reason:
            return _FunctionAnalysisError(
                target=qualname,
                reason=first_error_reason,
            )
        guard_exprs.extend(
            filter(
                _is_not_none_expression,
                map(_loop_body_guard_expr_optional, outcomes),
            )
        )
        operations.extend(
            filter(
                _is_not_none_loop_operation,
                map(_loop_body_operation_optional, outcomes),
            )
        )

        if not operations:
            return _FunctionAnalysisError(
                target=qualname,
                reason=f"{qualname}: loop has no supported mutation operations",
            )
        helper_name = f"_iter_{function_name}_loop_{loop_line}"
        return _FunctionAnalysisSuccess(
            spec=_LoopRewriteSpec(
                function_name=function_name,
                qualname=qualname,
                loop_line=loop_line,
                loop_var=loop_var,
                iter_expr=cast(cst.BaseExpression, loop.iter.deep_clone()),
                guard_exprs=tuple(guard_exprs),
                operations=tuple(operations),
                helper_name=helper_name,
                params=cast(cst.Parameters, params.deep_clone()),
                call_args=_parameter_call_args(params),
            )
        )

    @singledispatchmethod
    def _analyze_for_body_statement(
        self,
        stmt: cst.BaseStatement,
        *,
        qualname: str,
    ) -> _LoopBodyOutcome:
        never("unregistered runtime type", value_type=type(stmt).__name__)

    @_analyze_for_body_statement.register
    def _(self, stmt: cst.If, *, qualname: str) -> _LoopBodyOutcome:
        if not _is_simple_continue_guard(stmt):
            return _LoopBodyOutcome(
                kind="error",
                reason=f"{qualname}: only `if <predicate>: continue` guards are allowed",
            )
        safe, reason = _is_side_effect_safe_expression(stmt.test)
        if not safe:
            return _LoopBodyOutcome(
                kind="error",
                reason=f"{qualname}: guard predicate is unsafe ({reason})",
            )
        return _LoopBodyOutcome(
            kind="guard",
            guard_expr=stmt.test,
        )

    @_analyze_for_body_statement.register
    def _(self, stmt: cst.SimpleStatementLine, *, qualname: str) -> _LoopBodyOutcome:
        only = _single_small_statement_optional(stmt)
        if only is None:
            return _LoopBodyOutcome(
                kind="error",
                reason=f"{qualname}: compound simple statements are not supported",
            )
        return self._analyze_for_small_statement(only, qualname=qualname)

    def _unsupported_for_body_statement(
        self,
        stmt: cst.BaseStatement,
        *,
        qualname: str,
    ) -> _LoopBodyOutcome:
        return _LoopBodyOutcome(
            kind="error",
            reason=f"{qualname}: unsupported statement type {type(stmt).__name__}",
        )

    @_analyze_for_body_statement.register
    def _(self, stmt: cst.For, *, qualname: str) -> _LoopBodyOutcome:
        return self._unsupported_for_body_statement(stmt, qualname=qualname)

    @_analyze_for_body_statement.register
    def _(self, stmt: cst.While, *, qualname: str) -> _LoopBodyOutcome:
        return self._unsupported_for_body_statement(stmt, qualname=qualname)

    @_analyze_for_body_statement.register
    def _(self, stmt: cst.With, *, qualname: str) -> _LoopBodyOutcome:
        return self._unsupported_for_body_statement(stmt, qualname=qualname)

    @_analyze_for_body_statement.register
    def _(self, stmt: cst.Try, *, qualname: str) -> _LoopBodyOutcome:
        return self._unsupported_for_body_statement(stmt, qualname=qualname)

    @_analyze_for_body_statement.register
    def _(self, stmt: cst.TryStar, *, qualname: str) -> _LoopBodyOutcome:
        return self._unsupported_for_body_statement(stmt, qualname=qualname)

    @_analyze_for_body_statement.register
    def _(self, stmt: cst.Match, *, qualname: str) -> _LoopBodyOutcome:
        return self._unsupported_for_body_statement(stmt, qualname=qualname)

    @_analyze_for_body_statement.register
    def _(self, stmt: cst.FunctionDef, *, qualname: str) -> _LoopBodyOutcome:
        return self._unsupported_for_body_statement(stmt, qualname=qualname)

    @_analyze_for_body_statement.register
    def _(self, stmt: cst.ClassDef, *, qualname: str) -> _LoopBodyOutcome:
        return self._unsupported_for_body_statement(stmt, qualname=qualname)

    @singledispatchmethod
    def _analyze_for_small_statement(
        self,
        stmt: cst.BaseSmallStatement,
        *,
        qualname: str,
    ) -> _LoopBodyOutcome:
        never("unregistered runtime type", value_type=type(stmt).__name__)

    @_analyze_for_small_statement.register
    def _(self, stmt: cst.Expr, *, qualname: str) -> _LoopBodyOutcome:
        if not cst_matchers.matches(stmt.value, cst_matchers.Call()):
            return _LoopBodyOutcome(
                kind="error",
                reason=f"{qualname}: statement `{type(stmt).__name__}` is unsupported",
            )
        call = _call_optional(stmt.value)
        if call is None:
            return _LoopBodyOutcome(
                kind="error",
                reason=f"{qualname}: statement `{type(stmt).__name__}` is unsupported",
            )
        call_func = _attribute_optional(call.func)
        if call_func is None:
            return _LoopBodyOutcome(
                kind="error",
                reason=f"{qualname}: only list.append/set.add calls are supported",
            )
        target_name_node = _name_optional(call_func.value)
        if target_name_node is None:
            return _LoopBodyOutcome(
                kind="error",
                reason=f"{qualname}: only simple-name receivers are supported",
            )
        target_name = target_name_node.value
        method = call_func.attr.value
        if method not in {"append", "add"}:
            return _LoopBodyOutcome(
                kind="error",
                reason=f"{qualname}: unsupported mutation method `{method}`",
            )
        if len(call.args) != 1:
            return _LoopBodyOutcome(
                kind="error",
                reason=f"{qualname}: mutation calls must have exactly one argument",
            )
        argument = call.args[0]
        if argument.keyword is not None or argument.star:
            return _LoopBodyOutcome(
                kind="error",
                reason=f"{qualname}: mutation calls may not use keyword/star arguments",
            )
        safe, reason = _is_side_effect_safe_expression(argument.value)
        if not safe:
            return _LoopBodyOutcome(
                kind="error",
                reason=f"{qualname}: mutation operand is unsafe ({reason})",
            )
        return _LoopBodyOutcome(
            kind="operation",
            operation=_LoopOperation(
                kind="LIST_APPEND" if method == "append" else "SET_ADD",
                target=target_name,
                value_expr=argument.value,
            ),
        )

    @_analyze_for_small_statement.register
    def _(self, stmt: cst.Assign, *, qualname: str) -> _LoopBodyOutcome:
        if len(stmt.targets) != 1:
            return _LoopBodyOutcome(
                kind="error",
                reason=f"{qualname}: only single-target assignment is supported",
            )
        target_expr = stmt.targets[0].target
        if cst_matchers.matches(
            target_expr,
            cst_matchers.Subscript(value=cst_matchers.Name()),
        ):
            target_subscript = _subscript_optional(target_expr)
            if target_subscript is None:
                return _LoopBodyOutcome(
                    kind="error",
                    reason=f"{qualname}: dict assignment form is unsupported",
                )
            key_expr_obj = _extract_subscript_key(target_subscript)
            if key_expr_obj is None:
                return _LoopBodyOutcome(
                    kind="error",
                    reason=f"{qualname}: dict assignment must use a single index key",
                )
            key_expr = key_expr_obj
            safe_key, key_reason = _is_side_effect_safe_expression(key_expr)
            if not safe_key:
                return _LoopBodyOutcome(
                    kind="error",
                    reason=f"{qualname}: dict key expression is unsafe ({key_reason})",
                )
            safe_value, value_reason = _is_side_effect_safe_expression(stmt.value)
            if not safe_value:
                return _LoopBodyOutcome(
                    kind="error",
                    reason=f"{qualname}: dict value expression is unsafe ({value_reason})",
                )
            target_name_node = _name_optional(target_subscript.value)
            if target_name_node is None:
                return _LoopBodyOutcome(
                    kind="error",
                    reason=f"{qualname}: dict target must be a simple name",
                )
            target_name = target_name_node.value
            return _LoopBodyOutcome(
                kind="operation",
                operation=_LoopOperation(
                    kind="DICT_SET",
                    target=target_name,
                    key_expr=key_expr,
                    value_expr=stmt.value,
                ),
            )
        target_name_node = _name_optional(target_expr)
        if target_name_node is None:
            return _LoopBodyOutcome(
                kind="error",
                reason=f"{qualname}: assignment form is unsupported",
            )
        if not cst_matchers.matches(
            stmt.value,
            cst_matchers.BinaryOperation(left=cst_matchers.Name()),
        ):
            return _LoopBodyOutcome(
                kind="error",
                reason=f"{qualname}: assignment form is unsupported",
            )
        binary = _binary_operation_optional(stmt.value)
        if binary is None:
            return _LoopBodyOutcome(
                kind="error",
                reason=f"{qualname}: assignment form is unsupported",
            )
        left_name = _name_optional(binary.left)
        op_token = _operator_token(binary.operator)
        if not op_token or left_name is None or left_name.value != target_name_node.value:
            return _LoopBodyOutcome(
                kind="error",
                reason=f"{qualname}: unsupported reducer assignment form",
            )
        safe_operand, operand_reason = _is_side_effect_safe_expression(binary.right)
        if not safe_operand:
            return _LoopBodyOutcome(
                kind="error",
                reason=f"{qualname}: reducer operand is unsafe ({operand_reason})",
            )
        return _LoopBodyOutcome(
            kind="operation",
            operation=_LoopOperation(
                kind="REDUCE",
                target=target_name_node.value,
                operator=op_token,
                value_expr=binary.right,
            ),
        )

    @_analyze_for_small_statement.register
    def _(self, stmt: cst.AugAssign, *, qualname: str) -> _LoopBodyOutcome:
        target_name = _name_optional(stmt.target)
        if target_name is None:
            return _LoopBodyOutcome(
                kind="error",
                reason=f"{qualname}: reducer target must be a simple name",
            )
        op_token = _operator_token(stmt.operator)
        if not op_token:
            return _LoopBodyOutcome(
                kind="error",
                reason=f"{qualname}: reducer operator is not in the safe subset",
            )
        safe_operand, operand_reason = _is_side_effect_safe_expression(stmt.value)
        if not safe_operand:
            return _LoopBodyOutcome(
                kind="error",
                reason=f"{qualname}: reducer operand is unsafe ({operand_reason})",
            )
        return _LoopBodyOutcome(
            kind="operation",
            operation=_LoopOperation(
                kind="REDUCE",
                target=target_name.value,
                operator=op_token,
                value_expr=stmt.value,
            ),
        )

    def _unsupported_for_small_statement(
        self,
        stmt: cst.BaseSmallStatement,
        *,
        qualname: str,
    ) -> _LoopBodyOutcome:
        return _LoopBodyOutcome(
            kind="error",
            reason=f"{qualname}: statement `{type(stmt).__name__}` is unsupported",
        )

    @_analyze_for_small_statement.register
    def _(self, stmt: cst.AnnAssign, *, qualname: str) -> _LoopBodyOutcome:
        return self._unsupported_for_small_statement(stmt, qualname=qualname)

    @_analyze_for_small_statement.register
    def _(self, stmt: cst.Assert, *, qualname: str) -> _LoopBodyOutcome:
        return self._unsupported_for_small_statement(stmt, qualname=qualname)

    @_analyze_for_small_statement.register
    def _(self, stmt: cst.Break, *, qualname: str) -> _LoopBodyOutcome:
        return self._unsupported_for_small_statement(stmt, qualname=qualname)

    @_analyze_for_small_statement.register
    def _(self, stmt: cst.Continue, *, qualname: str) -> _LoopBodyOutcome:
        return self._unsupported_for_small_statement(stmt, qualname=qualname)

    @_analyze_for_small_statement.register
    def _(self, stmt: cst.Del, *, qualname: str) -> _LoopBodyOutcome:
        return self._unsupported_for_small_statement(stmt, qualname=qualname)

    @_analyze_for_small_statement.register
    def _(self, stmt: cst.Global, *, qualname: str) -> _LoopBodyOutcome:
        return self._unsupported_for_small_statement(stmt, qualname=qualname)

    @_analyze_for_small_statement.register
    def _(self, stmt: cst.Import, *, qualname: str) -> _LoopBodyOutcome:
        return self._unsupported_for_small_statement(stmt, qualname=qualname)

    @_analyze_for_small_statement.register
    def _(self, stmt: cst.ImportFrom, *, qualname: str) -> _LoopBodyOutcome:
        return self._unsupported_for_small_statement(stmt, qualname=qualname)

    @_analyze_for_small_statement.register
    def _(self, stmt: cst.Nonlocal, *, qualname: str) -> _LoopBodyOutcome:
        return self._unsupported_for_small_statement(stmt, qualname=qualname)

    @_analyze_for_small_statement.register
    def _(self, stmt: cst.Pass, *, qualname: str) -> _LoopBodyOutcome:
        return self._unsupported_for_small_statement(stmt, qualname=qualname)

    @_analyze_for_small_statement.register
    def _(self, stmt: cst.Raise, *, qualname: str) -> _LoopBodyOutcome:
        return self._unsupported_for_small_statement(stmt, qualname=qualname)

    @_analyze_for_small_statement.register
    def _(self, stmt: cst.Return, *, qualname: str) -> _LoopBodyOutcome:
        return self._unsupported_for_small_statement(stmt, qualname=qualname)

    @_analyze_for_small_statement.register
    def _(self, stmt: cst.TypeAlias, *, qualname: str) -> _LoopBodyOutcome:
        return self._unsupported_for_small_statement(stmt, qualname=qualname)

    def _is_already_rewritten(
        self,
        non_doc_body: list[cst.BaseStatement],
        *,
        function_name: str,
    ) -> bool:
        line = _simple_statement_line_optional(non_doc_body[0]) if len(non_doc_body) == 1 else None
        only_stmt = _single_small_statement_optional(line) if line is not None else None
        matches_return_call = bool(
            only_stmt is not None
            and cst_matchers.matches(
                only_stmt,
                cst_matchers.Return(value=cst_matchers.Call(func=cst_matchers.Name())),
            )
        )
        ret = _return_optional(only_stmt) if only_stmt is not None else None
        ret_call = _call_optional(ret.value) if ret is not None else None
        helper_name_node = _name_optional(ret_call.func) if ret_call is not None else None
        helper_name = helper_name_node.value if helper_name_node is not None else ""
        match = _LOOP_HELPER_PATTERN.match(helper_name)
        return bool(
            line is not None
            and matches_return_call
            and ret is not None
            and ret_call is not None
            and helper_name_node is not None
            and match
            and match.group("name") == function_name
        )


def _root_not_found_after_rewrite(
    *,
    root: str,
    root_to_effective: dict[str, tuple[str, ...]],
    matched_targets: set[str],
    roots_with_chase_issue: set[str],
) -> bool:
    effective = root_to_effective.get(root, ())
    matched = any(map(lambda target: target in matched_targets, effective))
    chased_only = root in roots_with_chase_issue and not effective
    return not matched and not chased_only


def _append_not_found_rewrite(
    transformer: _LoopGeneratorTransformer,
    root: str,
) -> None:
    reason = f"{root}: target function was not found"
    transformer.errors.append(reason)
    transformer.rewrite_plans.append(
        RewritePlanEntry(
            kind="LOOP_GENERATOR",
            status="ABSTAINED",
            target=root,
            summary="Loop generator rewrite was not applied.",
            non_rewrite_reasons=[reason],
        )
    )


@grade_boundary(
    kind="semantic_carrier_adapter",
    name="refactor.loop_generator_request_ingress",
)
def plan_loop_generator_rewrite(
    *,
    request: LoopGeneratorRequest,
) -> RefactorPlan:
    check_deadline()
    path = request.target_path
    targets = set(
        filter(
            bool,
            map(str.strip, request.target_functions),
        )
    )
    if not targets:
        return RefactorPlan(
            outcome=RefactorPlanOutcome.ERROR,
            errors=["loop_generator mode requires non-empty target_functions"],
        )
    loop_lines = set(map(int, request.target_loop_lines))
    invalid_loop_lines = tuple(filter(lambda line: line <= 0, loop_lines))
    if invalid_loop_lines:
        invalid_line = invalid_loop_lines[0]
        return RefactorPlan(
            outcome=RefactorPlanOutcome.ERROR,
            errors=[f"target_loop_lines must be 1-based positive integers (got {invalid_line})"],
        )
    try:
        source = path.read_text()
    except Exception as exc:
        return RefactorPlan(
            outcome=RefactorPlanOutcome.ERROR,
            errors=[f"Failed to read {path}: {exc}"],
        )
    try:
        module = cst.parse_module(source)
    except Exception as exc:
        return RefactorPlan(
            outcome=RefactorPlanOutcome.ERROR,
            errors=[f"LibCST parse failed for {path}: {exc}"],
        )

    resolution = _resolve_loop_generator_targets(
        module=module,
        requested_targets=targets,
    )
    effective_targets = set(resolution.effective_targets)
    root_to_effective = dict(resolution.root_to_effective)
    effective_to_roots = dict(resolution.effective_to_roots)

    transformer = _LoopGeneratorTransformer(
        module=module,
        targets=effective_targets,
        target_loop_lines=loop_lines,
    )
    wrapper = cst_metadata.MetadataWrapper(module)
    rewritten = cast(cst.Module, wrapper.visit(transformer))

    roots_with_chase_issue = {root for root, _ in resolution.chase_issues}
    for root, reason in resolution.chase_issues:
        check_deadline()
        transformer.errors.append(reason)
        transformer.rewrite_plans.append(
            RewritePlanEntry(
                kind="LOOP_GENERATOR",
                status="ABSTAINED",
                target=root,
                summary="Loop generator rewrite was not applied.",
                non_rewrite_reasons=[reason],
            )
        )

    roots_not_found = tuple(
        filter(
            lambda root: _root_not_found_after_rewrite(
                root=root,
                root_to_effective=root_to_effective,
                matched_targets=transformer.matched_targets,
                roots_with_chase_issue=roots_with_chase_issue,
            ),
            sorted(targets),
        )
    )
    tuple(map(lambda root: _append_not_found_rewrite(transformer, root), roots_not_found))

    rewrite_plans = [
        _rewrite_plan_with_chase_context(
            plan_entry,
            effective_to_roots=effective_to_roots,
        )
        for plan_entry in transformer.rewrite_plans
    ]

    if transformer.changed:
        new_source = rewritten.code
        end_line = len(source.splitlines())
        return RefactorPlan(
            outcome=RefactorPlanOutcome.APPLIED,
            edits=[
                TextEdit(
                    path=str(path),
                    start=(0, 0),
                    end=(end_line, 0),
                    replacement=new_source,
                )
            ],
            rewrite_plans=rewrite_plans,
            errors=transformer.errors,
        )

    if transformer.errors:
        return RefactorPlan(
            outcome=RefactorPlanOutcome.ERROR,
            rewrite_plans=rewrite_plans,
            errors=transformer.errors,
        )
    return RefactorPlan(
        outcome=RefactorPlanOutcome.NO_CHANGES,
        rewrite_plans=rewrite_plans,
    )
