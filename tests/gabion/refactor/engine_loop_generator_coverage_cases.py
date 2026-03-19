from __future__ import annotations

from pathlib import Path
import textwrap

import libcst as cst
import pytest

from gabion.refactor.engine import RefactorEngine
from gabion.refactor.model import LoopGeneratorRequest, RefactorPlanOutcome
from gabion.refactor import loop_generator as lg
from tests.path_helpers import REPO_ROOT


def _parse_module(source: str) -> cst.Module:
    return cst.parse_module(textwrap.dedent(source).strip() + "\n")


def _first_function(module: cst.Module, name: str = "f") -> cst.FunctionDef:
    for stmt in module.body:
        if type(stmt) is cst.FunctionDef and stmt.name.value == name:
            return stmt
    raise AssertionError(f"function {name!r} not found")


def _first_for_stmt_from_indented(block: cst.IndentedBlock) -> cst.For:
    for stmt in block.body:
        if type(stmt) is cst.For:
            return stmt
    raise AssertionError("no for statement found")


def _first_for_stmt(module: cst.Module, name: str = "f") -> cst.For:
    fn = _first_function(module, name=name)
    if type(fn.body) is not cst.IndentedBlock:
        raise AssertionError("expected indented function body")
    return _first_for_stmt_from_indented(fn.body)


def _build_analysis_transformer(module: cst.Module) -> lg._LoopGeneratorTransformer:
    return lg._LoopGeneratorTransformer(module=module, targets=set(), target_loop_lines=set())


def _analyze_for_source(source: str, *, qualname: str = "f") -> lg._FunctionAnalysis:
    loop_block = textwrap.indent(textwrap.dedent(source).strip(), "    ")
    module = _parse_module(
        f"def f(xs, out, seen, mapping, acc, fn, obj):\n{loop_block}\n"
    )
    fn = _first_function(module)
    if type(fn.body) is not cst.IndentedBlock:
        raise AssertionError("expected indented function body")
    loop = _first_for_stmt_from_indented(fn.body)
    transformer = _build_analysis_transformer(module)
    return transformer._analyze_for_loop(
        loop,
        qualname=qualname,
        function_name=fn.name.value,
        loop_line=1,
        params=fn.params,
    )


# gabion:behavior primary=desired
@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("x", ""),
        ("fn(x)", "calls are not side-effect-safe"),
        ("[x for x in xs]", "list comprehensions are not side-effect-safe"),
        ("{x for x in xs}", "set comprehensions are not side-effect-safe"),
        ("{x: x for x in xs}", "dict comprehensions are not side-effect-safe"),
        ("(x for x in xs)", "generator comprehensions are not side-effect-safe"),
        ("(y := x)", "assignment expressions are not side-effect-safe"),
    ],
)
def test_side_effect_safety_expression_matrix(expr: str, expected: str) -> None:
    safe, reason = lg._is_side_effect_safe_expression(cst.parse_expression(expr))
    assert safe is (expected == "")
    assert reason == expected


# gabion:behavior primary=desired
def test_side_effect_visitor_mark_only_once() -> None:
    visitor = lg._SideEffectSafetyVisitor()
    visitor.visit_Call(cst.Call(func=cst.Name("fn"), args=[]))
    visitor.visit_ListComp(cst.parse_expression("[x for x in xs]"))
    assert visitor.reason == "calls are not side-effect-safe"
    visitor = lg._SideEffectSafetyVisitor()
    visitor.visit_Await(cst.Await(expression=cst.Name("x")))
    assert visitor.reason == "await expressions are not side-effect-safe"
    visitor = lg._SideEffectSafetyVisitor()
    visitor.visit_Yield(cst.Yield(value=cst.Name("x")))
    assert visitor.reason == "yield expressions are not side-effect-safe"
    visitor = lg._SideEffectSafetyVisitor()
    visitor.visit_From(cst.From(item=cst.Name("xs")))
    assert visitor.reason == "yield from expressions are not side-effect-safe"


# gabion:behavior primary=desired
def test_loop_hazard_visitor_marks_all_hazards() -> None:
    visitor = lg._LoopHazardVisitor()
    visitor.visit_Break(cst.Break())
    visitor.visit_Return(cst.Return())
    assert visitor.reason == "break is not supported in loop_generator mode"
    visitor = lg._LoopHazardVisitor()
    visitor.visit_For(
        cst.For(
            target=cst.Name("x"),
            iter=cst.Name("xs"),
            body=cst.IndentedBlock(body=[cst.SimpleStatementLine([cst.Pass()])]),
        )
    )
    assert visitor.reason == "nested loops are not supported"
    visitor = lg._LoopHazardVisitor()
    visitor.visit_While(
        cst.While(
            test=cst.Name("x"),
            body=cst.IndentedBlock(body=[cst.SimpleStatementLine([cst.Pass()])]),
        )
    )
    assert visitor.reason == "nested loops are not supported"
    visitor = lg._LoopHazardVisitor()
    visitor.visit_Raise(cst.Raise(exc=cst.Name("e")))
    assert visitor.reason == "raise is not supported inside targeted loops"
    visitor = lg._LoopHazardVisitor()
    visitor.visit_Try(
        cst.parse_statement("try:\n    pass\nfinally:\n    pass\n")
    )
    assert visitor.reason == "try is not supported inside targeted loops"
    visitor = lg._LoopHazardVisitor()
    visitor.visit_With(
        cst.With(
            items=[cst.WithItem(item=cst.Name("ctx"))],
            body=cst.IndentedBlock(body=[cst.SimpleStatementLine([cst.Pass()])]),
        )
    )
    assert visitor.reason == "with is not supported inside targeted loops"
    visitor = lg._LoopHazardVisitor()
    visitor.visit_Yield(cst.Yield(value=cst.Name("x")))
    assert visitor.reason == "yield is not supported inside targeted loops"
    visitor = lg._LoopHazardVisitor()
    visitor.visit_From(cst.From(item=cst.Name("xs")))
    assert visitor.reason == "yield from is not supported inside targeted loops"


# gabion:behavior primary=desired
def test_is_docstring_statement_variants() -> None:
    assert not lg._is_docstring_statement(cst.Pass())
    assert not lg._is_docstring_statement(cst.SimpleStatementLine(body=[]))
    assert not lg._is_docstring_statement(cst.SimpleStatementLine(body=[cst.Pass()]))
    assert not lg._is_docstring_statement(cst.SimpleStatementLine(body=[cst.Expr(cst.Name("x"))]))
    assert lg._is_docstring_statement(
        cst.SimpleStatementLine(body=[cst.Expr(cst.SimpleString('"doc"'))])
    )


# gabion:behavior primary=verboten facets=unsupported
def test_operator_token_supported_and_unsupported() -> None:
    assert lg._operator_token(cst.AddAssign()) == "+"
    assert lg._operator_token(cst.Multiply()) == "*"
    assert lg._operator_token(cst.MatrixMultiply()) == ""


# gabion:behavior primary=desired
def test_extract_subscript_key_matrix() -> None:
    two_index = cst.parse_expression("d[a, b]")
    assert type(two_index) is cst.Subscript
    assert lg._extract_subscript_key(two_index) is None
    slice_index = cst.parse_expression("d[a:b]")
    assert type(slice_index) is cst.Subscript
    assert lg._extract_subscript_key(slice_index) is None
    one_index = cst.parse_expression("d[a]")
    assert type(one_index) is cst.Subscript
    key = lg._extract_subscript_key(one_index)
    assert type(key) is cst.Name and key.value == "a"


# gabion:behavior primary=desired
def test_contains_loop_hazards_matrix() -> None:
    one_line_module = _parse_module("def f(xs):\n    for x in xs: pass")
    one_line_loop = _first_for_stmt(one_line_module)
    assert lg._contains_loop_hazards(one_line_loop) == "loop body must be a block"
    break_module = _parse_module(
        """
        def f(xs):
            for x in xs:
                break
        """
    )
    break_loop = _first_for_stmt(break_module)
    assert lg._contains_loop_hazards(break_loop) == "break is not supported in loop_generator mode"
    safe_module = _parse_module(
        """
        def f(xs, out):
            for x in xs:
                out.append(x)
        """
    )
    safe_loop = _first_for_stmt(safe_module)
    assert lg._contains_loop_hazards(safe_loop) == ""


# gabion:behavior primary=desired
def test_parameter_call_args_all_parameter_kinds() -> None:
    module = _parse_module("def f(a, /, b, *args, c, d=1, **kwargs):\n    return 1")
    fn = _first_function(module)
    args = lg._parameter_call_args(fn.params)
    rendered = [cst.Module([]).code_for_node(arg).strip() for arg in args]
    assert rendered[0] == "a"
    assert rendered[1] == "b"
    assert rendered[2] == "*args"
    assert rendered[3].replace(" ", "") == "c=c"
    assert rendered[4].replace(" ", "") == "d=d"
    assert rendered[5] == "**kwargs"


# gabion:behavior primary=desired
def test_is_simple_continue_guard_matrix() -> None:
    assert lg._is_simple_continue_guard(cst.parse_statement("if x:\n    continue\n"))
    assert not lg._is_simple_continue_guard(cst.parse_statement("if x:\n    continue\nelse:\n    continue\n"))
    assert not lg._is_simple_continue_guard(cst.parse_statement("if x: continue\n"))
    assert not lg._is_simple_continue_guard(cst.parse_statement("if x:\n    continue\n    continue\n"))
    assert not lg._is_simple_continue_guard(cst.parse_statement("if x:\n    if y:\n        continue\n"))
    assert not lg._is_simple_continue_guard(cst.parse_statement("if x:\n    continue\n    pass\n"))
    assert not lg._is_simple_continue_guard(cst.parse_statement("if x:\n    continue; pass\n"))


# gabion:behavior primary=desired
def test_join_guard_expressions_multiple_entries() -> None:
    joined = lg._join_guard_expressions((cst.Name("a"), cst.Name("b"), cst.Name("c")))
    text = cst.Module([]).code_for_node(joined).strip()
    assert text == "a or b or c"


# gabion:behavior primary=desired
def test_find_import_insert_index_paths() -> None:
    mod = _parse_module(
        '''
        """doc"""
        import os
        from typing import Iterator
        x = 1
        '''
    )
    assert lg._find_import_insert_index(list(mod.body)) == 3
    mod = _parse_module("def f():\n    pass\n")
    assert lg._find_import_insert_index(list(mod.body)) == 0
    mod = _parse_module('"""doc only"""\n')
    assert lg._find_import_insert_index(list(mod.body)) == 1


# gabion:behavior primary=desired
def test_has_import_from_variants() -> None:
    with_alias = _parse_module("from dataclasses import dataclass\n")
    assert lg._has_import_from(list(with_alias.body), module_name="dataclasses", symbol="dataclass")
    with_star = _parse_module("from dataclasses import *\n")
    assert not lg._has_import_from(
        list(with_star.body),
        module_name="dataclasses",
        symbol="dataclass",
    )
    wrong_module = _parse_module("from typing import Iterator\n")
    assert not lg._has_import_from(
        list(wrong_module.body),
        module_name="dataclasses",
        symbol="dataclass",
    )
    wrong_symbol = _parse_module("from dataclasses import field\n")
    assert not lg._has_import_from(
        list(wrong_symbol.body),
        module_name="dataclasses",
        symbol="dataclass",
    )
    custom = cst.Module(
        body=[
            cst.SimpleStatementLine(
                body=[
                    cst.ImportFrom(
                        module=cst.Name("dataclasses"),
                        names=[
                            cst.ImportAlias(name=cst.Attribute(value=cst.Name("pkg"), attr=cst.Name("node"))),
                            cst.ImportAlias(name=cst.Name("dataclass")),
                        ],
                    )
                ]
            )
        ]
    )
    assert lg._has_import_from(list(custom.body), module_name="dataclasses", symbol="dataclass")


# gabion:behavior primary=desired
def test_defined_top_level_name_matrix() -> None:
    assert lg._defined_top_level_name(cst.parse_statement("class C:\n    pass\n")) == "C"
    assert lg._defined_top_level_name(cst.parse_statement("def f():\n    pass\n")) == "f"
    assert lg._defined_top_level_name(cst.parse_statement("x = 1\n")) == "x"
    assert lg._defined_top_level_name(cst.parse_statement("x = 1; y = 2\n")) == ""
    assert lg._defined_top_level_name(cst.parse_statement("obj.x = 1\n")) == ""
    assert lg._defined_top_level_name(cst.parse_statement("if x:\n    pass\n")) == ""


# gabion:behavior primary=desired
def test_ensure_scaffolding_inserts_then_becomes_idempotent() -> None:
    base = _parse_module("x = 1\n")
    with_scaffold = lg._ensure_loop_generator_scaffolding(base)
    rendered = with_scaffold.code
    assert "from dataclasses import dataclass" in rendered
    assert "from typing import Iterator" in rendered
    assert "class _LoopListAppendOp" in rendered
    roundtrip = lg._ensure_loop_generator_scaffolding(with_scaffold)
    assert roundtrip.code == with_scaffold.code


# gabion:behavior primary=desired
def test_build_helper_function_without_and_with_guards() -> None:
    params = cst.parse_statement("def f(xs, out):\n    pass\n").params  # type: ignore[attr-defined]
    spec_no_filter = lg._LoopRewriteSpec(
        function_name="f",
        qualname="f",
        loop_line=1,
        loop_var="item",
        iter_expr=cst.Name("xs"),
        guard_exprs=(),
        operations=(
            lg._LoopOperation(kind="LIST_APPEND", target="out", value_expr=cst.Name("item")),
            lg._LoopOperation(kind="SET_ADD", target="seen", value_expr=cst.Name("item")),
            lg._LoopOperation(
                kind="DICT_SET",
                target="mapping",
                key_expr=cst.Name("item"),
                value_expr=cst.Name("item"),
            ),
            lg._LoopOperation(
                kind="REDUCE",
                target="acc",
                operator="+",
                value_expr=cst.Name("item"),
            ),
        ),
        helper_name="_iter_f_loop_1",
        params=params,
        call_args=(cst.Arg(value=cst.Name("xs")), cst.Arg(value=cst.Name("out"))),
    )
    helper_no_filter = lg._build_helper_function(spec_no_filter)
    code_no_filter = cst.Module([]).code_for_node(helper_no_filter)
    assert "_filtered_iter" not in code_no_filter
    assert "_LoopListAppendOp" in code_no_filter
    assert "_LoopSetAddOp" in code_no_filter
    assert "_LoopDictSetOp" in code_no_filter
    assert "_LoopReduceOp" in code_no_filter

    spec_with_filter = lg._LoopRewriteSpec(
        function_name="f",
        qualname="f",
        loop_line=1,
        loop_var="item",
        iter_expr=cst.Name("xs"),
        guard_exprs=(cst.Name("a"), cst.Name("b")),
        operations=(lg._LoopOperation(kind="LIST_APPEND", target="out", value_expr=cst.Name("item")),),
        helper_name="_iter_f_loop_1",
        params=params,
        call_args=(cst.Arg(value=cst.Name("xs")), cst.Arg(value=cst.Name("out"))),
    )
    helper_with_filter = lg._build_helper_function(spec_with_filter)
    code_with_filter = cst.Module([]).code_for_node(helper_with_filter)
    assert "_filtered_iter = filter(" in code_with_filter


# gabion:behavior primary=desired
def test_target_resolution_helper_chase_paths() -> None:
    one_line = _first_function(_parse_module("def one(): return 1\n"), name="one")
    assert lg._function_non_doc_body(one_line) == ()

    with_doc = _first_function(
        _parse_module(
            '''
            def d():
                """doc"""
                return 1
            '''
        ),
        name="d",
    )
    assert len(lg._function_non_doc_body(with_doc)) == 1

    not_call = _first_function(_parse_module("def f(xs):\n    return xs\n"))
    assert lg._trampoline_helper_name(not_call) == ""
    not_return = _first_function(_parse_module("def f(xs):\n    x = xs\n"))
    assert lg._trampoline_helper_name(not_return) == ""
    not_pattern = _first_function(_parse_module("def f(xs):\n    return helper(xs)\n"))
    assert lg._trampoline_helper_name(not_pattern) == ""
    wrong_name = _first_function(_parse_module("def f(xs):\n    return _iter_g_loop_1(xs)\n"))
    assert lg._trampoline_helper_name(wrong_name) == ""

    chain_module = _parse_module(
        """
        def f(xs):
            return _iter_f_loop_1(xs)
        def _iter_f_loop_1(xs):
            return _iter_f_loop_2(xs)
        def _iter_f_loop_2(xs):
            return xs
        """
    )
    by_qualname, _ = lg._build_function_index(chain_module)
    assert lg._trampoline_helper_name(_first_function(chain_module, name="f")) == "_iter_f_loop_1"

    terminal, issue = lg._follow_trampoline_chain(start_qualname="missing", by_qualname={})
    assert terminal == "missing"
    assert issue == ""

    cycle_module = _parse_module(
        """
        def f(xs):
            return _iter_f_loop_1(xs)
        def _iter_f_loop_1(xs):
            return _iter_f_loop_1(xs)
        """
    )
    cycle_index, _ = lg._build_function_index(cycle_module)
    terminal, issue = lg._follow_trampoline_chain(start_qualname="f", by_qualname=cycle_index)
    assert terminal
    assert "cycle" in issue

    class_module = _parse_module(
        """
        class C:
            def f(xs):
                return _iter_f_loop_1(xs)

        def _iter_f_loop_1(xs):
            return xs
        """
    )
    class_index, _ = lg._build_function_index(class_module)
    terminal, issue = lg._follow_trampoline_chain(start_qualname="C.f", by_qualname=class_index)
    assert terminal == "_iter_f_loop_1"
    assert issue == ""

    unresolved_module = _parse_module(
        """
        class C:
            def f(xs):
                return _iter_f_loop_1(xs)
        """
    )
    unresolved_index, _ = lg._build_function_index(unresolved_module)
    terminal, issue = lg._follow_trampoline_chain(
        start_qualname="C.f",
        by_qualname=unresolved_index,
    )
    assert terminal == "C._iter_f_loop_1"
    assert issue == ""

    original_depth = lg._MAX_HELPER_CHASE_DEPTH
    lg._MAX_HELPER_CHASE_DEPTH = 1
    try:
        terminal, issue = lg._follow_trampoline_chain(start_qualname="f", by_qualname=by_qualname)
    finally:
        lg._MAX_HELPER_CHASE_DEPTH = original_depth
    assert terminal
    assert "exceeded max depth" in issue

    resolution = lg._resolve_loop_generator_targets(module=cycle_module, requested_targets={"f"})
    assert resolution.chase_issues


# gabion:behavior primary=desired
def test_recursive_candidate_helpers_cover_statement_variants() -> None:
    module = _parse_module(
        """
        def f(xs, ctx):
            for x in xs:
                pass
            else:
                pass
            while xs:
                pass
            else:
                pass
            if xs:
                pass
            else:
                pass
            with ctx:
                pass
            try:
                pass
            except Exception:
                pass
            else:
                pass
            finally:
                pass
            match xs:
                case _:
                    pass
        """
    )
    fn = _first_function(module)
    assert type(fn.body) is cst.IndentedBlock
    transformer = _build_analysis_transformer(module)
    statements = list(fn.body.body)
    for_stmt = statements[0]
    while_stmt = statements[1]
    if_stmt = statements[2]
    with_stmt = statements[3]
    try_stmt = statements[4]
    match_stmt = statements[5]
    assert type(for_stmt) is cst.For
    assert type(while_stmt) is cst.While
    assert type(if_stmt) is cst.If
    assert type(with_stmt) is cst.With
    assert type(try_stmt) is cst.Try
    assert type(match_stmt) is cst.Match

    assert len(transformer._child_statement_blocks(for_stmt)) == 2
    assert len(transformer._child_statement_blocks(while_stmt)) == 2
    assert len(transformer._child_statement_blocks(if_stmt)) == 2
    assert len(transformer._child_statement_blocks(with_stmt)) == 1
    assert len(transformer._child_statement_blocks(try_stmt)) == 4
    assert len(transformer._child_statement_blocks(match_stmt)) == 1
    assert transformer._suite_statements(cst.parse_statement("if xs: pass\n").body) == ()

    try_only_module = _parse_module(
        """
        def g():
            try:
                pass
            except Exception:
                pass
        """
    )
    try_only_fn = _first_function(try_only_module, name="g")
    assert type(try_only_fn.body) is cst.IndentedBlock
    try_only_stmt = try_only_fn.body.body[0]
    assert type(try_only_stmt) is cst.Try
    try_only_transformer = _build_analysis_transformer(try_only_module)
    assert len(try_only_transformer._child_statement_blocks(try_only_stmt)) == 2


# gabion:behavior primary=desired
def test_transformer_stack_cleanup_direct_calls() -> None:
    module = _parse_module("x = 1\n")
    transformer = lg._LoopGeneratorTransformer(module=module, targets=set(), target_loop_lines=set())
    class_node = cst.parse_statement("class C:\n    pass\n")
    fn_node = cst.parse_statement("def f():\n    pass\n")
    assert type(class_node) is cst.ClassDef
    assert type(fn_node) is cst.FunctionDef
    assert transformer.leave_ClassDef(class_node, class_node) is class_node
    assert transformer.leave_FunctionDef(fn_node, fn_node) is fn_node


# gabion:behavior primary=desired
def test_rewrite_target_function_handles_one_line_definition() -> None:
    module = _parse_module("def f(): return 1\n")
    fn = _first_function(module)
    spec = lg._LoopRewriteSpec(
        function_name="f",
        qualname="f",
        loop_line=1,
        loop_var="item",
        iter_expr=cst.Name("xs"),
        guard_exprs=(),
        operations=(lg._LoopOperation(kind="LIST_APPEND", target="out", value_expr=cst.Name("item")),),
        helper_name="_iter_f_loop_1",
        params=fn.params,
        call_args=(),
    )
    transformer = _build_analysis_transformer(module)
    rewritten = transformer._rewrite_target_function(fn, spec)
    assert rewritten is fn


# gabion:behavior primary=desired
def test_rewrite_target_function_preserves_docstring() -> None:
    module = _parse_module(
        '''
        def f(xs):
            """doc"""
            for item in xs:
                pass
        '''
    )
    fn = _first_function(module)
    if type(fn.body) is not cst.IndentedBlock:
        raise AssertionError("expected indented function body")
    spec = lg._LoopRewriteSpec(
        function_name="f",
        qualname="f",
        loop_line=2,
        loop_var="item",
        iter_expr=cst.Name("xs"),
        guard_exprs=(),
        operations=(lg._LoopOperation(kind="LIST_APPEND", target="out", value_expr=cst.Name("item")),),
        helper_name="_iter_f_loop_2",
        params=fn.params,
        call_args=(cst.Arg(value=cst.Name("xs")),),
    )
    transformer = _build_analysis_transformer(module)
    rewritten = transformer._rewrite_target_function(fn, spec)
    rendered = cst.Module([]).code_for_node(rewritten)
    assert '"""doc"""' in rendered
    assert "return _iter_f_loop_2(xs)" in rendered


# gabion:behavior primary=allowed_unwanted facets=noop
def test_analyze_function_matrix_non_loop_and_noop_paths() -> None:
    module = _parse_module("def f(): return 1\n")
    fn = _first_function(module)
    transformer = _build_analysis_transformer(module)
    outcome = transformer._analyze_function(fn, "f")
    assert type(outcome) is lg._FunctionAnalysisError
    assert "function body must be a block" in outcome.reason

    module = _parse_module(
        '''
        def f():
            """doc only"""
        '''
    )
    fn = _first_function(module)
    transformer = _build_analysis_transformer(module)
    outcome = transformer._analyze_function(fn, "f")
    assert type(outcome) is lg._FunctionAnalysisError
    assert "function body is empty" in outcome.reason

    module = _parse_module(
        """
        def f():
            value = 1
        """
    )
    fn = _first_function(module)
    transformer = _build_analysis_transformer(module)
    outcome = transformer._analyze_function(fn, "f")
    assert type(outcome) is lg._FunctionAnalysisError
    assert "no loop found" in outcome.reason

    module = _parse_module(
        """
        def f(xs, out):
            return _iter_f_loop_2(xs, out)
        """
    )
    fn = _first_function(module)
    transformer = _build_analysis_transformer(module)
    outcome = transformer._analyze_function(fn, "f")
    assert type(outcome) is lg._FunctionAnalysisNoop

    helper_terminal_module = _parse_module(
        """
        def _iter_f_loop_2():
            x = 1
        """
    )
    helper_terminal_fn = _first_function(helper_terminal_module, name="_iter_f_loop_2")
    helper_transformer = _build_analysis_transformer(helper_terminal_module)
    outcome = helper_transformer._analyze_function(helper_terminal_fn, "_iter_f_loop_2")
    assert type(outcome) is lg._FunctionAnalysisNoop


# gabion:behavior primary=verboten facets=error
def test_analyze_for_loop_error_matrix() -> None:
    outcome = _analyze_for_source(
        """
        for item in xs:
            out.append(item)
        else:
            out.append(item)
        """
    )
    assert type(outcome) is lg._FunctionAnalysisError
    assert "for-else loops are not supported" in outcome.reason

    async_module = _parse_module(
        """
        async def f(xs, out, seen, mapping, acc, fn, obj):
            async for item in xs:
                out.append(item)
        """
    )
    async_loop = _first_for_stmt(async_module)
    async_fn = _first_function(async_module)
    transformer = _build_analysis_transformer(async_module)
    outcome = transformer._analyze_for_loop(
        async_loop,
        qualname="f",
        function_name="f",
        loop_line=1,
        params=async_fn.params,
    )
    assert type(outcome) is lg._FunctionAnalysisError
    assert "async-for loops are not supported" in outcome.reason

    outcome = _analyze_for_source(
        """
        for a, b in xs:
            out.append(a)
        """
    )
    assert type(outcome) is lg._FunctionAnalysisError
    assert "loop target must be a simple name" in outcome.reason

    outcome = _analyze_for_source(
        """
        for item in xs:
            break
        """
    )
    assert type(outcome) is lg._FunctionAnalysisError
    assert "break is not supported" in outcome.reason

    one_line_module = _parse_module(
        """
        def f(xs, out, seen, mapping, acc, fn, obj):
            for item in xs: out.append(item)
        """
    )
    one_line_fn = _first_function(one_line_module)
    if type(one_line_fn.body) is not cst.IndentedBlock:
        raise AssertionError("expected indented function body")
    one_line_loop = _first_for_stmt_from_indented(one_line_fn.body)
    transformer = _build_analysis_transformer(one_line_module)
    outcome = transformer._analyze_for_loop(
        one_line_loop,
        qualname="f",
        function_name="f",
        loop_line=1,
        params=one_line_fn.params,
    )
    assert type(outcome) is lg._FunctionAnalysisError
    assert "loop body must be a block" in outcome.reason

    matrix = [
        (
            """
            for item in xs:
                if item:
                    out.append(item)
            """,
            "only `if <predicate>: continue` guards are allowed",
        ),
        (
            """
            for item in xs:
                if fn(item):
                    continue
                out.append(item)
            """,
            "guard predicate is unsafe",
        ),
        (
            """
            for item in xs:
                class Inner:
                    pass
            """,
            "unsupported statement type ClassDef",
        ),
        (
            """
            for item in xs:
                out.append(item); seen.add(item)
            """,
            "compound simple statements are not supported",
        ),
        (
            """
            for item in xs:
                fn(item)
            """,
            "only list.append/set.add calls are supported",
        ),
        (
            """
            for item in xs:
                out.extend(item)
            """,
            "unsupported mutation method `extend`",
        ),
        (
            """
            for item in xs:
                out.append(item, item)
            """,
            "mutation calls must have exactly one argument",
        ),
        (
            """
            for item in xs:
                out.append(value=item)
            """,
            "mutation calls may not use keyword/star arguments",
        ),
        (
            """
            for item in xs:
                out.append(fn(item))
            """,
            "mutation operand is unsafe",
        ),
        (
            """
            for item in xs:
                out = seen = item
            """,
            "only single-target assignment is supported",
        ),
        (
            """
            for item in xs:
                mapping[fn(item)] = item
            """,
            "dict key expression is unsafe",
        ),
        (
            """
            for item in xs:
                mapping[item] = fn(item)
            """,
            "dict value expression is unsafe",
        ),
        (
            """
            for item in xs:
                acc = other + item
            """,
            "unsupported reducer assignment form",
        ),
        (
            """
            for item in xs:
                acc = acc + fn(item)
            """,
            "reducer operand is unsafe",
        ),
        (
            """
            for item in xs:
                acc = item
            """,
            "assignment form is unsupported",
        ),
        (
            """
            for item in xs:
                obj.acc += item
            """,
            "reducer target must be a simple name",
        ),
        (
            """
            for item in xs:
                acc @= item
            """,
            "reducer operator is not in the safe subset",
        ),
        (
            """
            for item in xs:
                acc += fn(item)
            """,
            "reducer operand is unsafe",
        ),
        (
            """
            for item in xs:
                pass
            """,
            "statement `Pass` is unsupported",
        ),
        (
            """
            for item in xs:
                if item:
                    continue
            """,
            "loop has no supported mutation operations",
        ),
        (
            """
            for item in xs:
                mapping[item, item] = item
            """,
            "dict assignment must use a single index key",
        ),
    ]
    for source, expected in matrix:
        result = _analyze_for_source(source)
        assert type(result) is lg._FunctionAnalysisError
        assert expected in result.reason

    ok = _analyze_for_source(
        """
        for item in xs:
            acc += item
        """
    )
    assert type(ok) is lg._FunctionAnalysisSuccess

    first_error_wins = _analyze_for_source(
        """
        for item in xs:
            for nested in xs:
                break
        """
    )
    assert type(first_error_wins) is lg._FunctionAnalysisError


# gabion:behavior primary=desired
def test_is_already_rewritten_matrix() -> None:
    module = _parse_module("def f(xs):\n    return _iter_f_loop_1(xs)\n")
    fn = _first_function(module)
    transformer = _build_analysis_transformer(module)
    assert type(fn.body) is cst.IndentedBlock
    non_doc = list(fn.body.body)
    assert transformer._is_already_rewritten(non_doc, function_name="f")
    assert not transformer._is_already_rewritten([], function_name="f")
    assert not transformer._is_already_rewritten([cst.parse_statement("if x:\n    pass\n")], function_name="f")
    assert not transformer._is_already_rewritten([cst.parse_statement("x = 1\n")], function_name="f")
    assert not transformer._is_already_rewritten(
        [cst.parse_statement("return 1\n")],
        function_name="f",
    )
    assert not transformer._is_already_rewritten(
        [cst.parse_statement("return call()\n")],
        function_name="f",
    )
    assert not transformer._is_already_rewritten(
        [cst.parse_statement("return _iter_g_loop_1(xs)\n")],
        function_name="f",
    )


# gabion:behavior primary=allowed_unwanted facets=error,noop
def test_plan_loop_generator_rewrite_error_and_noop_paths(tmp_path: Path) -> None:
    engine = RefactorEngine(project_root=tmp_path)

    empty_targets = engine.plan_loop_generator_rewrite(
        LoopGeneratorRequest(target_path=tmp_path / "sample.py", target_functions=["", "   "])
    )
    assert empty_targets.outcome == RefactorPlanOutcome.ERROR
    assert "requires non-empty target_functions" in empty_targets.errors[0]

    bad_line = engine.plan_loop_generator_rewrite(
        LoopGeneratorRequest(
            target_path=tmp_path / "sample.py",
            target_functions=["f"],
            target_loop_lines=[0],
        )
    )
    assert bad_line.outcome == RefactorPlanOutcome.ERROR
    assert "1-based positive integers" in bad_line.errors[0]

    missing = engine.plan_loop_generator_rewrite(
        LoopGeneratorRequest(
            target_path=tmp_path / "missing.py",
            target_functions=["f"],
        )
    )
    assert missing.outcome == RefactorPlanOutcome.ERROR
    assert "Failed to read" in missing.errors[0]

    bad_source = tmp_path / "bad.py"
    bad_source.write_text("def f(:\n", encoding="utf-8")
    parse_fail = engine.plan_loop_generator_rewrite(
        LoopGeneratorRequest(
            target_path=bad_source,
            target_functions=["f"],
        )
    )
    assert parse_fail.outcome == RefactorPlanOutcome.ERROR
    assert "LibCST parse failed" in parse_fail.errors[0]

    sample = tmp_path / "sample.py"
    sample.write_text(
        "def f(xs, out):\n    for x in xs:\n        out.append(x)\n",
        encoding="utf-8",
    )
    no_match = engine.plan_loop_generator_rewrite(
        LoopGeneratorRequest(
            target_path=sample,
            target_functions=["missing_target"],
        )
    )
    assert no_match.outcome == RefactorPlanOutcome.ERROR
    assert "target function was not found" in no_match.errors[0]
    assert no_match.rewrite_plans and no_match.rewrite_plans[0].status == "ABSTAINED"

    applied = engine.plan_loop_generator_rewrite(
        LoopGeneratorRequest(target_path=sample, target_functions=["f"])
    )
    assert applied.outcome == RefactorPlanOutcome.APPLIED
    sample.write_text(applied.edits[0].replacement, encoding="utf-8")
    no_changes = engine.plan_loop_generator_rewrite(
        LoopGeneratorRequest(target_path=sample, target_functions=["f"])
    )
    assert no_changes.outcome == RefactorPlanOutcome.NO_CHANGES
    assert any("helper chase from: f" in entry.summary for entry in no_changes.rewrite_plans)

    cycle_target = tmp_path / "cycle.py"
    cycle_target.write_text(
        textwrap.dedent(
            """
            def f(xs):
                return _iter_f_loop_1(xs)
            def _iter_f_loop_1(xs):
                return _iter_f_loop_1(xs)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    cycle_error = engine.plan_loop_generator_rewrite(
        LoopGeneratorRequest(target_path=cycle_target, target_functions=["f"])
    )
    assert cycle_error.outcome == RefactorPlanOutcome.ERROR
    assert any("helper chase cycle detected" in reason for reason in cycle_error.errors)
    assert not any("target function was not found" in reason for reason in cycle_error.errors)


# gabion:behavior primary=verboten facets=error
def test_plan_loop_generator_target_selection_and_async_errors(tmp_path: Path) -> None:
    engine = RefactorEngine(project_root=tmp_path)
    source = tmp_path / "source.py"
    source.write_text(
        textwrap.dedent(
            """
            class C:
                def f(xs, out):
                    for x in xs:
                        out.append(x)

            def g(xs, out):
                for x in xs:
                    out.append(x)

            async def af(xs, out):
                for x in xs:
                    out.append(x)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    by_name = engine.plan_loop_generator_rewrite(
        LoopGeneratorRequest(
            target_path=source,
            target_functions=["f"],
        )
    )
    assert by_name.outcome == RefactorPlanOutcome.APPLIED

    source.write_text(
        textwrap.dedent(
            """
            def g(xs, out):
                for x in xs:
                    out.append(x)
                for y in xs:
                    out.append(y)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    exact_line_match = engine.plan_loop_generator_rewrite(
        LoopGeneratorRequest(
            target_path=source,
            target_functions=["g"],
            target_loop_lines=[2],
        )
    )
    assert exact_line_match.outcome == RefactorPlanOutcome.APPLIED

    source.write_text(
        textwrap.dedent(
            """
            def g(xs, out):
                for x in xs:
                    out.append(x)
                for y in xs:
                    out.append(y)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    no_line_match = engine.plan_loop_generator_rewrite(
        LoopGeneratorRequest(
            target_path=source,
            target_functions=["g"],
            target_loop_lines=[999],
        )
    )
    assert no_line_match.outcome == RefactorPlanOutcome.ERROR
    assert "no loop matched requested target_loop_lines" in no_line_match.errors[0]

    source.write_text(
        textwrap.dedent(
            """
            def g(xs, out):
                for x in xs:
                    out.append(x)
                for y in xs:
                    out.append(y)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    multi_line_match = engine.plan_loop_generator_rewrite(
        LoopGeneratorRequest(
            target_path=source,
            target_functions=["g"],
            target_loop_lines=[2, 4],
        )
    )
    assert multi_line_match.outcome == RefactorPlanOutcome.ERROR
    assert "multiple loops matched target_loop_lines" in multi_line_match.errors[0]

    source.write_text(
        textwrap.dedent(
            """
            def g(xs, out):
                for x in xs:
                    out.append(x)
                for y in xs:
                    out.append(y)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    multi_top_level = engine.plan_loop_generator_rewrite(
        LoopGeneratorRequest(
            target_path=source,
            target_functions=["g"],
        )
    )
    assert multi_top_level.outcome == RefactorPlanOutcome.APPLIED

    source.write_text(
        textwrap.dedent(
            """
            def g(xs, out):
                while xs:
                    out.append(xs.pop())
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    while_error = engine.plan_loop_generator_rewrite(
        LoopGeneratorRequest(target_path=source, target_functions=["g"])
    )
    assert while_error.outcome == RefactorPlanOutcome.ERROR
    assert "only for-loops are supported" in while_error.errors[0]

    source.write_text(
        textwrap.dedent(
            """
            def g(xs, out):
                for x in xs:
                    out.append(x)
                out.append(1)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    extra_stmt_error = engine.plan_loop_generator_rewrite(
        LoopGeneratorRequest(target_path=source, target_functions=["g"])
    )
    assert extra_stmt_error.outcome == RefactorPlanOutcome.APPLIED

    source.write_text(
        textwrap.dedent(
            """
            async def af(xs, out):
                for x in xs:
                    out.append(x)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    async_error = engine.plan_loop_generator_rewrite(
        LoopGeneratorRequest(target_path=source, target_functions=["af"])
    )
    assert async_error.outcome == RefactorPlanOutcome.ERROR
    assert "async functions are not supported" in async_error.errors[0]

    source.write_text(
        textwrap.dedent(
            """
            def h(xs, out):
                while xs:
                    pass
                while xs:
                    pass
                for item in xs:
                    out.append(item)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    skip_ineligible = engine.plan_loop_generator_rewrite(
        LoopGeneratorRequest(target_path=source, target_functions=["h"])
    )
    assert skip_ineligible.outcome == RefactorPlanOutcome.APPLIED


# gabion:behavior primary=desired
def test_loop_generator_runs_against_its_own_module() -> None:
    # Running the refactor against this refactor engine source exercises
    # parse/selection/error control flow without mutating repository files.
    engine = RefactorEngine(project_root=REPO_ROOT)
    plan = engine.plan_loop_generator_rewrite(
        LoopGeneratorRequest(
            target_path=REPO_ROOT / "src/gabion/refactor/loop_generator.py",
            target_functions=["_string_literal"],
            rationale="coverage self-application",
        )
    )
    assert plan.outcome == RefactorPlanOutcome.ERROR
    assert plan.errors
