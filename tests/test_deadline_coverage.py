from __future__ import annotations

import ast
from pathlib import Path
import sys
import textwrap


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit as da

    return da


def _make_fn_info(
    da,
    *,
    name: str,
    qual: str | None = None,
    path: Path | None = None,
    params: list[str] | None = None,
    positional_params: tuple[str, ...] | None = None,
    kwonly_params: tuple[str, ...] = (),
    vararg: str | None = None,
    kwarg: str | None = None,
    calls=None,
):
    if qual is None:
        qual = name
    if positional_params is None:
        positional_params = tuple(params or ())
    if params is None:
        params = list(positional_params)
    if path is None:
        path = Path("mod.py")
    if calls is None:
        calls = []
    return da.FunctionInfo(
        name=name,
        qual=qual,
        path=path,
        params=list(params),
        annots={},
        calls=list(calls),
        unused_params=set(),
        defaults=set(),
        transparent=True,
        class_name=None,
        scope=(),
        lexical_scope=(),
        decision_params=set(),
        value_decision_params=set(),
        value_decision_reasons=set(),
        positional_params=tuple(positional_params),
        kwonly_params=tuple(kwonly_params),
        vararg=vararg,
        kwarg=kwarg,
        param_spans={},
    )


def test_deadline_helper_classification_and_unparse_error() -> None:
    da = _load()
    bad_call = ast.Call(func=ast.Name(id=None, ctx=ast.Load()), args=[], keywords=[])
    assert da._is_deadline_origin_call(bad_call) is False

    origin_call = ast.parse("Deadline()").body[0].value
    from_timeout = ast.parse("Deadline.from_timeout(1.0)").body[0].value
    assert da._is_deadline_origin_call(origin_call) is True
    assert da._is_deadline_origin_call(from_timeout) is True

    assert da._is_deadline_param("deadline", None) is True
    info_const = da._classify_deadline_expr(
        ast.Constant(value=5),
        alias_to_param={},
        origin_vars=set(),
    )
    assert info_const.kind == "const"
    info_origin = da._classify_deadline_expr(
        origin_call,
        alias_to_param={},
        origin_vars=set(),
    )
    assert info_origin.kind == "origin"


def test_deadline_local_info_aliasing() -> None:
    da = _load()
    source = """
    async def f(deadline):
        def inner():
            return 1
        async def inner_async():
            return 2
        _ = lambda x: x
        ctx.check_deadline(deadline)
        ctx.check_deadline()
        count = 0
        count += 1
        for _ in range(1):
            pass
        async for item in aiter():
            pass
        while False:
            pass
        deadline.check()
        check_deadline(deadline)
        check_deadline()
        deadline: Deadline
        origin = Deadline()
        origin_alias = origin
        origin = deadline
        alias = deadline
        deadline = None
        other = something()
        return alias
    """
    tree = ast.parse(textwrap.dedent(source))
    fn = tree.body[0]
    collector = da._DeadlineFunctionCollector(fn, {"deadline"})
    collector.visit(fn)
    info = da._collect_deadline_local_info(collector.assignments, {"deadline"})
    assert "origin_alias" in info.origin_vars
    assert info.alias_to_param.get("alias") == "deadline"


def test_deadline_function_facts_parse_error_and_scopes(tmp_path: Path) -> None:
    da = _load()
    valid = tmp_path / "mod.py"
    invalid = tmp_path / "bad.py"
    valid.write_text(
        textwrap.dedent(
            """
            class Box:
                def method(self, deadline: Deadline):
                    return deadline
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    invalid.write_text("def oops(:\n", encoding="utf-8")
    facts = da._collect_deadline_function_facts(
        [valid, invalid],
        project_root=tmp_path,
        ignore_params=set(),
    )
    assert "mod.Box.method" in facts
    call_nodes = da._collect_call_nodes_by_path([invalid])
    assert invalid not in call_nodes


def test_collect_call_nodes_handles_missing_span(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "dummy.py"
    call = ast.Call(func=ast.Name(id="f", ctx=ast.Load()), args=[], keywords=[])
    tree = ast.Module(body=[ast.Expr(value=call)], type_ignores=[])
    result = da._collect_call_nodes_by_path([path], trees={path: tree})
    assert result[path] == {}


def test_collect_call_edges_and_recursive_helpers() -> None:
    da = _load()
    call = da.CallArgs(
        callee="callee",
        pos_map={},
        kw_map={},
        const_pos={},
        const_kw={},
        non_const_pos=set(),
        non_const_kw=set(),
        star_pos=[],
        star_kw=[],
        is_test=True,
        span=None,
    )
    test_info = _make_fn_info(
        da,
        name="test_func",
        qual="tests.test_func",
        path=Path("tests/test_mod.py"),
    )
    live_info = _make_fn_info(
        da,
        name="live",
        qual="mod.live",
        path=Path("mod.py"),
        calls=[call],
    )
    by_name = {"test_func": [test_info], "live": [live_info]}
    by_qual = {test_info.qual: test_info, live_info.qual: live_info}
    edges = da._collect_call_edges(
        by_name=by_name,
        by_qual=by_qual,
        symbol_table=da.SymbolTable(external_filter=True),
        project_root=None,
        class_index={},
    )
    assert edges == {}

    recursive = da._collect_recursive_functions(
        {"a": {"a"}, "b": {"c"}, "c": {"b"}}
    )
    assert recursive == {"a", "b", "c"}


def test_deadline_arg_info_binding_and_fallback() -> None:
    da = _load()
    callee = _make_fn_info(
        da,
        name="callee",
        qual="mod.callee",
        params=["a", "b", "opt"],
        positional_params=("a", "b"),
        kwonly_params=("opt",),
        vararg="args",
        kwarg="kwargs",
    )
    call_node = ast.parse("callee(x, *star, opt=1, extra=2, **kw)").body[0].value
    mapping = da._bind_call_args(call_node, callee, strictness="low")
    assert "a" in mapping
    assert "b" in mapping
    assert "kwargs" in mapping

    extra_call = ast.parse("callee(x, y, z)").body[0].value
    mapping = da._bind_call_args(extra_call, callee, strictness="high")
    assert "args" in mapping

    call = da.CallArgs(
        callee="callee",
        pos_map={"0": "p0", "2": "p2"},
        kw_map={"opt": "popt", "extra": "pextra"},
        const_pos={"1": "None", "3": "7"},
        const_kw={"opt": "3", "other": "None"},
        non_const_pos={"4"},
        non_const_kw={"opt2"},
        star_pos=[(5, "star")],
        star_kw=["starkw"],
        is_test=False,
        span=(0, 0, 0, 1),
    )
    info = da._fallback_deadline_arg_info(call, callee, strictness="high")
    assert info["a"].kind == "param"
    assert info["b"].kind == "none"
    assert "kwargs" in info

    call_low = da.CallArgs(
        callee="callee",
        pos_map={"0": "p0"},
        kw_map={},
        const_pos={},
        const_kw={},
        non_const_pos=set(),
        non_const_kw=set(),
        star_pos=[(1, "star")],
        star_kw=["starkw"],
        is_test=False,
        span=None,
    )
    info_low = da._fallback_deadline_arg_info(call_low, callee, strictness="low")
    assert "opt" in info_low

    call_non_const = da.CallArgs(
        callee="callee",
        pos_map={},
        kw_map={},
        const_pos={},
        const_kw={},
        non_const_pos={"0"},
        non_const_kw={"opt"},
        star_pos=[],
        star_kw=[],
        is_test=False,
        span=None,
    )
    info_non_const = da._fallback_deadline_arg_info(call_non_const, callee, strictness="high")
    assert info_non_const["a"].kind == "unknown"
    assert info_non_const["opt"].kind == "unknown"

    arg_map = da._deadline_arg_info_map(
        call_low,
        callee,
        call_node=None,
        alias_to_param={"p0": "p0"},
        origin_vars=set(),
        strictness="low",
    )
    assert arg_map


def test_collect_deadline_obligations_full_matrix(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "mod.py"
    target.write_text(
        textwrap.dedent(
            """
            def root(deadline: Deadline):
                callee(deadline)
                callee(Deadline())
                return deadline

            def callee(deadline: Deadline):
                return deadline

            def callee_kw(deadline: Deadline, **kwargs):
                return deadline

            def callee_var(deadline: Deadline, *rest):
                return deadline

            def callee_origin_only(deadline: Deadline):
                return deadline

            def wrapper(d):
                callee(d)

            def extra_positional(deadline: Deadline):
                callee(deadline, deadline)

            def pass_kw_named(deadline: Deadline):
                callee(deadline=deadline)

            def pass_kwarg(deadline: Deadline):
                callee_kw(deadline, extra=deadline)

            def pass_vararg(deadline: Deadline):
                callee_var(deadline, deadline)

            def loop_unchecked(deadline: Deadline):
                for _ in range(1):
                    pass

            def loop_checked(deadline: Deadline):
                check_deadline(deadline)
                for _ in range(1):
                    pass

            def loop_ambient():
                check_deadline()
                for _ in range(1):
                    pass

            def loop_ambient_with_carrier(deadline: Deadline):
                check_deadline()
                for _ in range(1):
                    pass

            def loop_missing_carrier():
                for _ in range(1):
                    pass

            def default_deadline(deadline: Deadline = Deadline()):
                return deadline

            def pass_missing():
                callee()

            def pass_missing_unknown():
                args = (1,)
                callee(*args)

            def pass_none():
                callee(None)

            def pass_const():
                callee(5)

            def pass_origin_not_root():
                callee(Deadline())

            def pass_param_untrusted(deadline: Deadline):
                callee(deadline)

            def pass_unknown(deadline: Deadline):
                callee(deadline.foo)

            def local_origin_ok():
                d = Deadline()
                return d

            def local_origin_overwritten():
                d = Deadline()
                d = None
                return d

            def unresolved(deadline: Deadline):
                missing(deadline)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    test_target = tmp_path / "tests" / "test_mod.py"
    test_target.parent.mkdir(parents=True, exist_ok=True)
    test_target.write_text(
        textwrap.dedent(
            """
            def test_loop(deadline: Deadline):
                for _ in range(1):
                    pass
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    config = da.AuditConfig(
        project_root=tmp_path,
        exclude_dirs=set(),
        ignore_params=set(),
        external_filter=True,
        strictness="high",
        deadline_roots={"mod.root"},
    )
    extra_facts = {
        "missing.qual": da._DeadlineFunctionFacts(
            path=Path("missing.py"),
            qual="missing.qual",
            span=None,
            loop=False,
            check_params=set(),
            ambient_check=False,
            local_info=da._DeadlineLocalInfo(
                origin_vars=set(),
                origin_spans={},
                alias_to_param={},
            ),
        )
    }
    extra_call_infos = {
        "missing.caller": [
            (
                da.CallArgs(
                    callee="mod.callee",
                    pos_map={},
                    kw_map={},
                    const_pos={},
                    const_kw={},
                    non_const_pos=set(),
                    non_const_kw=set(),
                    star_pos=[],
                    star_kw=[],
                    is_test=False,
                    span=None,
                ),
                _make_fn_info(da, name="callee", qual="mod.callee", params=["deadline"]),
                {},
            )
        ],
        "mod.root": [
            (
                da.CallArgs(
                    callee="mod.callee",
                    pos_map={},
                    kw_map={},
                    const_pos={},
                    const_kw={},
                    non_const_pos=set(),
                    non_const_kw=set(),
                    star_pos=[],
                    star_kw=[],
                    is_test=False,
                    span=None,
                ),
                _make_fn_info(da, name="callee", qual="mod.callee", params=["deadline"]),
                {"deadline": da._DeadlineArgInfo(kind="origin")},
            ),
            (
                da.CallArgs(
                    callee="mod.callee_origin_only",
                    pos_map={},
                    kw_map={},
                    const_pos={},
                    const_kw={},
                    non_const_pos=set(),
                    non_const_kw=set(),
                    star_pos=[],
                    star_kw=[],
                    is_test=False,
                    span=None,
                ),
                _make_fn_info(
                    da,
                    name="callee_origin_only",
                    qual="mod.callee_origin_only",
                    params=["deadline"],
                ),
                {"deadline": da._DeadlineArgInfo(kind="origin")},
            ),
            (
                da.CallArgs(
                    callee="mod.callee",
                    pos_map={},
                    kw_map={},
                    const_pos={},
                    const_kw={},
                    non_const_pos=set(),
                    non_const_kw=set(),
                    star_pos=[],
                    star_kw=[],
                    is_test=False,
                    span=None,
                ),
                _make_fn_info(da, name="callee", qual="mod.callee", params=["deadline"]),
                {},
            ),
        ],
    }
    extra_deadline_params = {"tests.test_mod.test_loop": {"deadline"}}
    obligations = da._collect_deadline_obligations(
        [target, test_target],
        project_root=tmp_path,
        config=config,
        extra_facts_by_qual=extra_facts,
        extra_call_infos=extra_call_infos,
        extra_deadline_params=extra_deadline_params,
    )
    kinds = {entry.get("kind") for entry in obligations}
    assert "default_param" in kinds
    assert "missing_carrier" in kinds
    assert "unchecked_deadline" in kinds
    assert "missing_arg" in kinds
    assert "missing_arg_unknown" in kinds
    assert "none_arg" in kinds
    assert "const_arg" in kinds
    assert "origin_not_allowlisted" in kinds
    assert "untrusted_param" in kinds
    assert "unknown_arg" in kinds

    summary = da._summarize_deadline_obligations(obligations, max_entries=1)
    assert summary
    assert da._summarize_deadline_obligations([]) == []

    lint_lines = da._deadline_lint_lines(
        [
            {
                "site": {"path": "", "function": "f", "bundle": []},
                "status": "UNKNOWN",
                "span": ["x", "y", "z", "w"],
            },
            {
                "site": {"path": "mod.py", "function": "f", "bundle": []},
                "status": "VIOLATION",
                "kind": "missing",
                "detail": "oops",
                "span": [0, 0, 0, 1],
            },
        ]
    )
    assert any("GABION_DEADLINE" in line for line in lint_lines)

    report, violations = da._emit_report(
        {},
        0,
        deadline_obligations=obligations,
    )
    assert "Deadline propagation:" in report
    assert violations


def test_collect_deadline_obligations_strictness_low_star(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "mod.py"
    target.write_text(
        textwrap.dedent(
            """
            def callee(deadline: Deadline, *args, **kwargs):
                return deadline

            def caller(deadline: Deadline, *args, **kwargs):
                callee(*args, **kwargs)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    config = da.AuditConfig(
        project_root=tmp_path,
        exclude_dirs=set(),
        ignore_params=set(),
        external_filter=True,
        strictness="low",
        deadline_roots={"mod.caller"},
    )
    obligations = da._collect_deadline_obligations(
        [target],
        project_root=tmp_path,
        config=config,
    )
    assert obligations is not None
