# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import cast

import libcst as cst
from libcst import metadata as cst_metadata

from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.refactor.model import (
    LoopGeneratorRequest,
    RefactorPlan,
    RefactorPlanOutcome,
    RewritePlanEntry,
    TextEdit,
)

_LOOP_HELPER_PATTERN = re.compile(r"^_iter_(?P<name>[A-Za-z_]\w*)_loop_(?P<line>\d+)$")
_MAX_HELPER_CHASE_DEPTH = 128
_ALLOWED_ASSIGN_OPERATORS: dict[type[object], str] = {
    cst.AddAssign: "+",
    cst.SubtractAssign: "-",
    cst.MultiplyAssign: "*",
    cst.DivideAssign: "/",
    cst.FloorDivideAssign: "//",
    cst.ModuloAssign: "%",
    cst.PowerAssign: "**",
    cst.BitOrAssign: "|",
    cst.BitAndAssign: "&",
    cst.BitXorAssign: "^",
    cst.LeftShiftAssign: "<<",
    cst.RightShiftAssign: ">>",
}
_ALLOWED_BINARY_OPERATORS: dict[type[object], str] = {
    cst.Add: "+",
    cst.Subtract: "-",
    cst.Multiply: "*",
    cst.Divide: "/",
    cst.FloorDivide: "//",
    cst.Modulo: "%",
    cst.Power: "**",
    cst.BitOr: "|",
    cst.BitAnd: "&",
    cst.BitXor: "^",
    cst.LeftShift: "<<",
    cst.RightShift: ">>",
}

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
    expr: cst.BaseExpression = cst.Name(carrier_names[0])
    for carrier_name in carrier_names[1:]:
        check_deadline()
        expr = cst.BinaryOperation(
            left=expr,
            operator=cst.BitOr(),
            right=cst.Name(carrier_name),
        )
    return cst.SimpleStatementLine(
        body=[
            cst.Assign(
                targets=[cst.AssignTarget(target=cst.Name("_LoopTransformOp"))],
                value=expr,
            )
        ]
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
    if type(stmt) is not cst.SimpleStatementLine:
        return False
    line = cast(cst.SimpleStatementLine, stmt)
    if not line.body:
        return False
    expr = line.body[0]
    if type(expr) is not cst.Expr:
        return False
    return type(cast(cst.Expr, expr).value) is cst.SimpleString


def _is_side_effect_safe_expression(
    expr: cst.BaseExpression,
) -> tuple[bool, str]:
    visitor = _SideEffectSafetyVisitor()
    expr.visit(visitor)
    return not visitor.reason, visitor.reason


def _operator_token(operator: object) -> str:
    token = _ALLOWED_ASSIGN_OPERATORS.get(type(operator))
    if token:
        return token
    return _ALLOWED_BINARY_OPERATORS.get(type(operator), "")


def _extract_subscript_key(target: cst.Subscript) -> object:
    if len(target.slice) != 1:
        return None
    first_slice = target.slice[0]
    if type(first_slice.slice) is not cst.Index:
        return None
    return cast(cst.Index, first_slice.slice).value


def _contains_loop_hazards(loop: cst.For) -> str:
    if type(loop.body) is not cst.IndentedBlock:
        return "loop body must be a block"
    visitor = _LoopHazardVisitor()
    for stmt in cast(cst.IndentedBlock, loop.body).body:
        check_deadline()
        stmt.visit(visitor)
        if visitor.reason:
            return visitor.reason
    return ""


def _parameter_call_args(params: cst.Parameters) -> tuple[cst.Arg, ...]:
    args: list[cst.Arg] = []
    for param in params.posonly_params:
        check_deadline()
        args.append(cst.Arg(value=cst.Name(param.name.value)))
    for param in params.params:
        check_deadline()
        args.append(cst.Arg(value=cst.Name(param.name.value)))
    if type(params.star_arg) is cst.Param:
        args.append(cst.Arg(star="*", value=cst.Name(params.star_arg.name.value)))
    for param in params.kwonly_params:
        check_deadline()
        name = param.name.value
        args.append(
            cst.Arg(
                keyword=cst.Name(name),
                value=cst.Name(name),
            )
        )
    if params.star_kwarg is not None:
        args.append(
            cst.Arg(
                star="**",
                value=cst.Name(params.star_kwarg.name.value),
            )
        )
    return tuple(args)


def _is_simple_continue_guard(stmt: cst.If) -> bool:
    if stmt.orelse is not None:
        return False
    if type(stmt.body) is not cst.IndentedBlock:
        return False
    body = cast(cst.IndentedBlock, stmt.body).body
    if len(body) != 1:
        return False
    only = body[0]
    if type(only) is not cst.SimpleStatementLine:
        return False
    line = cast(cst.SimpleStatementLine, only)
    if len(line.body) != 1:
        return False
    return type(line.body[0]) is cst.Continue


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
    joined = _clone_expression(guards[0])
    for guard in guards[1:]:
        check_deadline()
        joined = cst.BooleanOperation(
            left=joined,
            operator=cst.Or(),
            right=_clone_expression(guard),
        )
    return joined


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
    insert_idx = 0
    if body and _is_docstring_statement(cast(cst.BaseStatement, body[0])):
        insert_idx = 1
    while insert_idx < len(body):
        check_deadline()
        node = body[insert_idx]
        if type(node) is not cst.SimpleStatementLine:
            break
        line = cast(cst.SimpleStatementLine, node)
        if not any(type(item) in {cst.Import, cst.ImportFrom} for item in line.body):
            break
        insert_idx += 1
    return insert_idx


def _has_import_from(body: list[cst.CSTNode], *, module_name: str, symbol: str) -> bool:
    for stmt in body:
        check_deadline()
        if type(stmt) is cst.SimpleStatementLine:
            line = cast(cst.SimpleStatementLine, stmt)
            for item in line.body:
                check_deadline()
                if type(item) is cst.ImportFrom:
                    import_from = cast(cst.ImportFrom, item)
                    has_module = import_from.module is not None
                    module_matches = has_module and (
                        _code_for_node(cst.Module([]), cast(cst.CSTNode, import_from.module))
                        == module_name
                    )
                    names = import_from.names
                    names_are_aliases = type(names) in {tuple, list}
                    if module_matches and names_are_aliases:
                        for alias in cast(tuple[cst.ImportAlias, ...], tuple(names)):
                            check_deadline()
                            if type(alias.name) is cst.Name and alias.name.value == symbol:
                                return True
    return False


def _defined_top_level_name(stmt: cst.CSTNode) -> str:
    if type(stmt) is cst.ClassDef:
        return cast(cst.ClassDef, stmt).name.value
    if type(stmt) is cst.FunctionDef:
        return cast(cst.FunctionDef, stmt).name.value
    if type(stmt) is cst.SimpleStatementLine:
        line = cast(cst.SimpleStatementLine, stmt)
        if len(line.body) != 1:
            return ""
        only = line.body[0]
        if type(only) is cst.Assign and len(cast(cst.Assign, only).targets) == 1:
            target = cast(cst.Assign, only).targets[0].target
            if type(target) is cst.Name:
                return target.value
    return ""


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

    existing_names = {
        name
        for name in (_defined_top_level_name(stmt) for stmt in body)
        if name
    }
    scaffold_nodes = _loop_carrier_nodes()
    nodes_to_insert: list[cst.CSTNode] = []
    for node in scaffold_nodes:
        check_deadline()
        name = _defined_top_level_name(node)
        if name and name not in existing_names:
            existing_names.add(name)
            nodes_to_insert.append(node)
    if not nodes_to_insert:
        return module.with_changes(body=body)
    body[insert_idx:insert_idx] = [cst.EmptyLine(), *nodes_to_insert, cst.EmptyLine()]
    return module.with_changes(body=body)


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
    if type(node.body) is not cst.IndentedBlock:
        return ()
    body = list(cast(cst.IndentedBlock, node.body).body)
    if body and _is_docstring_statement(body[0]):
        return tuple(body[1:])
    return tuple(body)


def _trampoline_helper_name(node: cst.FunctionDef) -> str:
    body = _function_non_doc_body(node)
    if len(body) != 1:
        return ""
    stmt = body[0]
    if type(stmt) is not cst.SimpleStatementLine:
        return ""
    line = cast(cst.SimpleStatementLine, stmt)
    if len(line.body) != 1 or type(line.body[0]) is not cst.Return:
        return ""
    ret = cast(cst.Return, line.body[0])
    if type(ret.value) is not cst.Call or type(ret.value.func) is not cst.Name:
        return ""
    helper_name = cast(cst.Name, ret.value.func).value
    match = _LOOP_HELPER_PATTERN.match(helper_name)
    if not match:
        return ""
    source_name_match = _LOOP_HELPER_PATTERN.match(node.name.value)
    source_name = (
        source_name_match.group("name")
        if source_name_match
        else node.name.value
    )
    if match.group("name") != source_name:
        return ""
    return helper_name


def _build_function_index(
    module: cst.Module,
) -> tuple[dict[str, _FunctionIndexEntry], dict[str, tuple[str, ...]]]:
    visitor = _FunctionIndexVisitor()
    module.visit(visitor)
    by_qualname = {entry.qualname: entry for entry in visitor.entries}
    by_name: dict[str, list[str]] = {}
    for entry in visitor.entries:
        check_deadline()
        by_name.setdefault(entry.node.name.value, []).append(entry.qualname)
    return by_qualname, {name: tuple(values) for name, values in by_name.items()}


def _follow_trampoline_chain(
    *,
    start_qualname: str,
    by_qualname: dict[str, _FunctionIndexEntry],
) -> tuple[str, str]:
    current = start_qualname
    seen = {current}
    for _ in range(_MAX_HELPER_CHASE_DEPTH):
        check_deadline()
        entry = by_qualname.get(current)
        if entry is None:
            return current, ""
        helper_name = _trampoline_helper_name(entry.node)
        if not helper_name:
            return current, ""
        parent = current.rpartition(".")[0]
        scoped_target = f"{parent}.{helper_name}" if parent else helper_name
        if scoped_target in by_qualname:
            next_target = scoped_target
        elif helper_name in by_qualname:
            next_target = helper_name
        else:
            next_target = scoped_target
        if next_target in seen:
            return current, f"helper chase cycle detected at `{next_target}`"
        seen.add(next_target)
        current = next_target
    return current, f"helper chase exceeded max depth {_MAX_HELPER_CHASE_DEPTH}"


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
        candidates: tuple[str, ...]
        if root in by_qualname:
            candidates = (root,)
        else:
            candidates = by_name.get(root, ())
        resolved: list[str] = []
        for candidate in candidates:
            check_deadline()
            terminal, issue = _follow_trampoline_chain(
                start_qualname=candidate,
                by_qualname=by_qualname,
            )
            if issue:
                chase_issues.append((root, f"{root}: {issue}"))
                continue
            resolved.append(terminal)
            effective_targets.add(terminal)
            effective_to_roots.setdefault(terminal, set()).add(root)
        root_to_effective.append((root, tuple(sorted(set(resolved)))))

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


def _rewrite_plan_with_chase_context(
    plan: RewritePlanEntry,
    *,
    effective_to_roots: dict[str, tuple[str, ...]],
) -> RewritePlanEntry:
    forwarded = tuple(
        root
        for root in effective_to_roots.get(plan.target, ())
        if root != plan.target
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
        target_key = qualname if qualname in self._targets else None

        output: cst.CSTNode = updated_node
        if target_key is not None:
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
                if type(analysis) is _FunctionAnalysisError:
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
                elif type(analysis) is _FunctionAnalysisNoop:
                    self.rewrite_plans.append(
                        RewritePlanEntry(
                            kind="LOOP_GENERATOR",
                            status="noop",
                            target=analysis.target,
                            summary=analysis.reason,
                        )
                    )
                else:
                    spec = analysis.spec
                    helper_node = _build_helper_function(spec)
                    replacement = self._rewrite_target_function(updated_node, spec)
                    output = cst.FlattenSentinel([helper_node, replacement])
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

        if self._stack:
            self._stack.pop()
        return output

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.CSTNode:
        if not self.changed:
            return updated_node
        return _ensure_loop_generator_scaffolding(updated_node)

    def _rewrite_target_function(
        self,
        node: cst.FunctionDef,
        spec: _LoopRewriteSpec,
    ) -> cst.FunctionDef:
        if type(node.body) is not cst.IndentedBlock:
            return node
        existing = list(cast(cst.IndentedBlock, node.body).body)
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
        if type(node.body) is not cst.IndentedBlock:
            return _FunctionAnalysisError(
                target=qualname,
                reason=f"{qualname}: function body must be a block",
            )
        body = list(cast(cst.IndentedBlock, node.body).body)
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
            by_line = [
                candidate
                for candidate in loop_candidates
                if candidate.line in self._target_loop_lines
            ]
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
        if type(first_outcome) is _FunctionAnalysisSuccess:
            return first_outcome
        first_error = cast(_FunctionAnalysisError, first_outcome)
        for candidate in loop_candidates[1:]:
            check_deadline()
            outcome = self._analyze_loop_candidate(
                candidate,
                qualname=qualname,
                function_name=node.name.value,
                params=node.params,
            )
            if type(outcome) is _FunctionAnalysisSuccess:
                return outcome

        if _LOOP_HELPER_PATTERN.match(node.name.value):
            return _FunctionAnalysisNoop(
                target=qualname,
                reason=f"{qualname}: helper chain terminal has no remaining eligible loops",
            )
        return first_error

    def _suite_statements(self, suite: cst.BaseSuite) -> tuple[cst.BaseStatement, ...]:
        if type(suite) is cst.IndentedBlock:
            return tuple(cast(cst.IndentedBlock, suite).body)
        return ()

    def _child_statement_blocks(
        self,
        stmt: cst.BaseStatement,
    ) -> tuple[tuple[cst.BaseStatement, ...], ...]:
        if type(stmt) is cst.For:
            loop = cast(cst.For, stmt)
            blocks = [self._suite_statements(loop.body)]
            if loop.orelse is not None:
                blocks.append(self._suite_statements(loop.orelse.body))
            return tuple(blocks)
        if type(stmt) is cst.While:
            loop = cast(cst.While, stmt)
            blocks = [self._suite_statements(loop.body)]
            if loop.orelse is not None:
                blocks.append(self._suite_statements(loop.orelse.body))
            return tuple(blocks)
        if type(stmt) is cst.If:
            branch = cast(cst.If, stmt)
            blocks = [self._suite_statements(branch.body)]
            if branch.orelse is not None:
                blocks.append(self._suite_statements(branch.orelse.body))
            return tuple(blocks)
        if type(stmt) is cst.With:
            return (self._suite_statements(cast(cst.With, stmt).body),)
        if type(stmt) is cst.Try:
            trial = cast(cst.Try, stmt)
            blocks = [self._suite_statements(trial.body)]
            for handler in trial.handlers:
                check_deadline()
                blocks.append(self._suite_statements(handler.body))
            if trial.orelse is not None:
                blocks.append(self._suite_statements(trial.orelse.body))
            if trial.finalbody is not None:
                blocks.append(self._suite_statements(trial.finalbody.body))
            return tuple(blocks)
        if type(stmt) is cst.Match:
            return tuple(
                self._suite_statements(case.body)
                for case in cast(cst.Match, stmt).cases
            )
        return ()

    def _collect_loop_candidates_from_statement(
        self,
        stmt: cst.BaseStatement,
        out: list[_LoopCandidate],
    ) -> None:
        if type(stmt) in {cst.For, cst.While}:
            pos = self.get_metadata(
                cst_metadata.PositionProvider,
                cast(cst.CSTNode, stmt),
            )
            out.append(
                _LoopCandidate(
                    loop_node=cast(cst.CSTNode, stmt),
                    line=pos.start.line,
                )
            )
        for block in self._child_statement_blocks(stmt):
            check_deadline()
            for child_stmt in block:
                check_deadline()
                self._collect_loop_candidates_from_statement(child_stmt, out)

    def _collect_loop_candidates(
        self,
        statements: tuple[cst.BaseStatement, ...],
    ) -> tuple[_LoopCandidate, ...]:
        out: list[_LoopCandidate] = []
        for stmt in statements:
            check_deadline()
            self._collect_loop_candidates_from_statement(stmt, out)
        return tuple(out)

    def _analyze_loop_candidate(
        self,
        candidate: _LoopCandidate,
        *,
        qualname: str,
        function_name: str,
        params: cst.Parameters,
    ) -> _FunctionAnalysis:
        if type(candidate.loop_node) is not cst.For:
            return _FunctionAnalysisError(
                target=qualname,
                reason=f"{qualname}: only for-loops are supported",
            )
        return self._analyze_for_loop(
            cast(cst.For, candidate.loop_node),
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
        if type(loop.target) is not cst.Name:
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

        loop_var = cast(cst.Name, loop.target).value
        guard_exprs: list[cst.BaseExpression] = []
        operations: list[_LoopOperation] = []
        for stmt in cast(cst.IndentedBlock, loop.body).body:
            check_deadline()
            if type(stmt) is cst.If:
                guard_stmt = cast(cst.If, stmt)
                if not _is_simple_continue_guard(guard_stmt):
                    return _FunctionAnalysisError(
                        target=qualname,
                        reason=f"{qualname}: only `if <predicate>: continue` guards are allowed",
                    )
                safe, reason = _is_side_effect_safe_expression(guard_stmt.test)
                if not safe:
                    return _FunctionAnalysisError(
                        target=qualname,
                        reason=f"{qualname}: guard predicate is unsafe ({reason})",
                    )
                guard_exprs.append(
                    cast(cst.BaseExpression, guard_stmt.test.deep_clone())
                )
                continue
            if type(stmt) is not cst.SimpleStatementLine:
                return _FunctionAnalysisError(
                    target=qualname,
                    reason=f"{qualname}: unsupported statement type {type(stmt).__name__}",
                )
            line = cast(cst.SimpleStatementLine, stmt)
            if len(line.body) != 1:
                return _FunctionAnalysisError(
                    target=qualname,
                    reason=f"{qualname}: compound simple statements are not supported",
                )
            only = line.body[0]
            if type(only) is cst.Expr and type(cast(cst.Expr, only).value) is cst.Call:
                call = cast(cst.Call, cast(cst.Expr, only).value)
                if type(call.func) is not cst.Attribute or type(call.func.value) is not cst.Name:
                    return _FunctionAnalysisError(
                        target=qualname,
                        reason=f"{qualname}: only list.append/set.add calls are supported",
                    )
                target_name = cast(cst.Name, call.func.value).value
                method = call.func.attr.value
                if method not in {"append", "add"}:
                    return _FunctionAnalysisError(
                        target=qualname,
                        reason=f"{qualname}: unsupported mutation method `{method}`",
                    )
                if len(call.args) != 1:
                    return _FunctionAnalysisError(
                        target=qualname,
                        reason=f"{qualname}: mutation calls must have exactly one argument",
                    )
                argument = call.args[0]
                if argument.keyword is not None or argument.star:
                    return _FunctionAnalysisError(
                        target=qualname,
                        reason=f"{qualname}: mutation calls may not use keyword/star arguments",
                    )
                safe, reason = _is_side_effect_safe_expression(argument.value)
                if not safe:
                    return _FunctionAnalysisError(
                        target=qualname,
                        reason=f"{qualname}: mutation operand is unsafe ({reason})",
                    )
                operations.append(
                    _LoopOperation(
                        kind="LIST_APPEND" if method == "append" else "SET_ADD",
                        target=target_name,
                        value_expr=cast(
                            cst.BaseExpression,
                            argument.value.deep_clone(),
                        ),
                    )
                )
                continue
            if type(only) is cst.Assign:
                assign = cast(cst.Assign, only)
                if len(assign.targets) != 1:
                    return _FunctionAnalysisError(
                        target=qualname,
                        reason=f"{qualname}: only single-target assignment is supported",
                    )
                target_expr = assign.targets[0].target
                if type(target_expr) is cst.Subscript and type(target_expr.value) is cst.Name:
                    key_expr_obj = _extract_subscript_key(target_expr)
                    if key_expr_obj is None:
                        return _FunctionAnalysisError(
                            target=qualname,
                            reason=f"{qualname}: dict assignment must use a single index key",
                        )
                    key_expr = cast(cst.BaseExpression, key_expr_obj)
                    safe_key, key_reason = _is_side_effect_safe_expression(key_expr)
                    if not safe_key:
                        return _FunctionAnalysisError(
                            target=qualname,
                            reason=f"{qualname}: dict key expression is unsafe ({key_reason})",
                        )
                    safe_value, value_reason = _is_side_effect_safe_expression(assign.value)
                    if not safe_value:
                        return _FunctionAnalysisError(
                            target=qualname,
                            reason=f"{qualname}: dict value expression is unsafe ({value_reason})",
                        )
                    operations.append(
                        _LoopOperation(
                            kind="DICT_SET",
                            target=cast(cst.Name, target_expr.value).value,
                            key_expr=cast(cst.BaseExpression, key_expr.deep_clone()),
                            value_expr=cast(cst.BaseExpression, assign.value.deep_clone()),
                        )
                    )
                    continue
                if type(target_expr) is cst.Name and type(assign.value) is cst.BinaryOperation:
                    binary = cast(cst.BinaryOperation, assign.value)
                    op_token = _operator_token(binary.operator)
                    if (
                        not op_token
                        or type(binary.left) is not cst.Name
                        or cast(cst.Name, binary.left).value != target_expr.value
                    ):
                        return _FunctionAnalysisError(
                            target=qualname,
                            reason=f"{qualname}: unsupported reducer assignment form",
                        )
                    safe_operand, operand_reason = _is_side_effect_safe_expression(binary.right)
                    if not safe_operand:
                        return _FunctionAnalysisError(
                            target=qualname,
                            reason=f"{qualname}: reducer operand is unsafe ({operand_reason})",
                        )
                    operations.append(
                        _LoopOperation(
                            kind="REDUCE",
                            target=target_expr.value,
                            operator=op_token,
                            value_expr=cast(cst.BaseExpression, binary.right.deep_clone()),
                        )
                    )
                    continue
                return _FunctionAnalysisError(
                    target=qualname,
                    reason=f"{qualname}: assignment form is unsupported",
                )
            if type(only) is cst.AugAssign:
                aug = cast(cst.AugAssign, only)
                if type(aug.target) is not cst.Name:
                    return _FunctionAnalysisError(
                        target=qualname,
                        reason=f"{qualname}: reducer target must be a simple name",
                    )
                op_token = _operator_token(aug.operator)
                if not op_token:
                    return _FunctionAnalysisError(
                        target=qualname,
                        reason=f"{qualname}: reducer operator is not in the safe subset",
                    )
                safe_operand, operand_reason = _is_side_effect_safe_expression(aug.value)
                if not safe_operand:
                    return _FunctionAnalysisError(
                        target=qualname,
                        reason=f"{qualname}: reducer operand is unsafe ({operand_reason})",
                    )
                operations.append(
                    _LoopOperation(
                        kind="REDUCE",
                        target=cast(cst.Name, aug.target).value,
                        operator=op_token,
                        value_expr=cast(cst.BaseExpression, aug.value.deep_clone()),
                    )
                )
                continue
            return _FunctionAnalysisError(
                target=qualname,
                reason=f"{qualname}: statement `{type(only).__name__}` is unsupported",
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

    def _is_already_rewritten(
        self,
        non_doc_body: list[cst.BaseStatement],
        *,
        function_name: str,
    ) -> bool:
        if len(non_doc_body) != 1:
            return False
        stmt = non_doc_body[0]
        if type(stmt) is not cst.SimpleStatementLine:
            return False
        line = cast(cst.SimpleStatementLine, stmt)
        if len(line.body) != 1 or type(line.body[0]) is not cst.Return:
            return False
        ret = cast(cst.Return, line.body[0])
        if type(ret.value) is not cst.Call or type(ret.value.func) is not cst.Name:
            return False
        helper_name = cast(cst.Name, ret.value.func).value
        match = _LOOP_HELPER_PATTERN.match(helper_name)
        return bool(match and match.group("name") == function_name)


def plan_loop_generator_rewrite(
    *,
    request: LoopGeneratorRequest,
    project_root = None,
) -> RefactorPlan:
    check_deadline()
    path = Path(request.target_path)
    if project_root is not None and not path.is_absolute():
        path = project_root / path
    targets = {name.strip() for name in request.target_functions if name.strip()}
    if not targets:
        return RefactorPlan(
            outcome=RefactorPlanOutcome.ERROR,
            errors=["loop_generator mode requires non-empty target_functions"],
        )
    loop_lines: set[int] = set()
    for line in request.target_loop_lines:
        check_deadline()
        if line <= 0:
            return RefactorPlan(
                outcome=RefactorPlanOutcome.ERROR,
                errors=[f"target_loop_lines must be 1-based positive integers (got {line})"],
            )
        loop_lines.add(int(line))
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

    for root in sorted(targets):
        check_deadline()
        effective = root_to_effective.get(root, ())
        matched = any(target in transformer.matched_targets for target in effective)
        if matched:
            continue
        if root in roots_with_chase_issue and not effective:
            continue
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
