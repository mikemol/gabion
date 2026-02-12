from __future__ import annotations

import ast
from pathlib import Path
import sys
import textwrap
import pytest


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
        function_span=(0, 0, 0, 1),
    )


def test_deadline_helper_classification_and_unparse_error() -> None:
    da = _load()
    bad_call = ast.Call(func=ast.Name(id=None, ctx=ast.Load()), args=[], keywords=[])
    assert da._is_deadline_origin_call(bad_call) is False

    origin_call = ast.parse("Deadline()").body[0].value
    from_timeout = ast.parse("Deadline.from_timeout(1)").body[0].value
    from_timeout_ms = ast.parse("Deadline.from_timeout_ms(1)").body[0].value
    from_timeout_ticks = ast.parse("Deadline.from_timeout_ticks(1, 1000000)").body[0].value
    assert da._is_deadline_origin_call(origin_call) is True
    assert da._is_deadline_origin_call(from_timeout) is True
    assert da._is_deadline_origin_call(from_timeout_ms) is True
    assert da._is_deadline_origin_call(from_timeout_ticks) is True
    prefixed = ast.parse("mod.Deadline.from_timeout(1)").body[0].value
    prefixed_ms = ast.parse("mod.Deadline.from_timeout_ms(1)").body[0].value
    prefixed_ticks = ast.parse("mod.Deadline.from_timeout_ticks(1, 1000000)").body[0].value
    assert da._is_deadline_origin_call(prefixed) is True
    assert da._is_deadline_origin_call(prefixed_ms) is True
    assert da._is_deadline_origin_call(prefixed_ticks) is True

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


def test_deadline_collector_handles_missing_span_and_orelse() -> None:
    da = _load()
    source = """
    def f():
        for _ in range(1):
            pass
        else:
            check_deadline()
    """
    tree = ast.parse(textwrap.dedent(source))
    fn = tree.body[0]
    collector = da._DeadlineFunctionCollector(fn, set())
    collector.visit(fn)
    assert collector.ambient_check is True
    loop_fact = da._DeadlineLoopFacts(span=(0, 0, 0, 1), kind="for")
    collector._loop_stack.append(loop_fact)
    call = ast.Call(func=ast.Name(id="noop", ctx=ast.Load()), args=[], keywords=[])
    collector._record_call_span(call)
    collector._loop_stack.pop()
    assert not loop_fact.call_spans


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
    parse_failures: list[dict[str, object]] = []
    facts = da._collect_deadline_function_facts(
        [valid, invalid],
        project_root=tmp_path,
        ignore_params=set(),
        parse_failure_witnesses=parse_failures,
    )
    assert "mod.Box.method" in facts
    assert any(entry["stage"] == "deadline_function_facts" for entry in parse_failures)
    call_nodes = da._collect_call_nodes_by_path(
        [invalid],
        parse_failure_witnesses=parse_failures,
    )
    assert invalid not in call_nodes
    assert any(entry["stage"] == "call_nodes" for entry in parse_failures)


def test_collect_call_nodes_handles_missing_span(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "dummy.py"
    call = ast.Call(func=ast.Name(id="f", ctx=ast.Load()), args=[], keywords=[])
    tree = ast.Module(body=[ast.Expr(value=call)], type_ignores=[])
    result = da._collect_call_nodes_by_path(
        [path],
        trees={path: tree},
        parse_failure_witnesses=[],
    )
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
        span=(0, 0, 0, 1),
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


def test_deadline_loop_forwarded_params_branches() -> None:
    da = _load()
    loop_fact = da._DeadlineLoopFacts(span=(0, 0, 0, 1), kind="for")
    call = da.CallArgs(
        callee="mod.callee",
        pos_map={"0": "deadline"},
        kw_map={},
        const_pos={},
        const_kw={},
        non_const_pos=set(),
        non_const_kw=set(),
        star_pos=[],
        star_kw=[],
        is_test=False,
        span=(0, 0, 0, 1),
    )
    callee = _make_fn_info(da, name="callee", qual="mod.callee", params=["deadline"])

    forwarded = da._deadline_loop_forwarded_params(
        qual="mod.caller",
        loop_fact=loop_fact,
        deadline_params={},
        call_infos={},
    )
    assert forwarded == set()

    deadline_params = {"mod.caller": {"deadline"}, "mod.callee": {"deadline"}}
    call_infos = {
        "mod.caller": [
            (
                call,
                callee,
                {"deadline": da._DeadlineArgInfo(kind="param", param="deadline")},
            )
        ]
    }
    forwarded = da._deadline_loop_forwarded_params(
        qual="mod.caller",
        loop_fact=loop_fact,
        deadline_params=deadline_params,
        call_infos=call_infos,
    )
    assert forwarded == set()

    loop_fact.call_spans.add(call.span)
    call_infos = {
        "mod.caller": [
            (
                call,
                callee,
                {},
            )
        ]
    }
    forwarded = da._deadline_loop_forwarded_params(
        qual="mod.caller",
        loop_fact=loop_fact,
        deadline_params=deadline_params,
        call_infos=call_infos,
    )
    assert forwarded == set()


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
        span=(0, 0, 0, 1),
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
        span=(0, 0, 0, 1),
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
                for _ in range(1):
                    check_deadline(deadline)
                    pass

            def loop_forward(deadline: Deadline):
                for _ in range(1):
                    callee(deadline)

            def loop_ambient():
                for _ in range(1):
                    check_deadline()
                    pass

            def loop_ambient_with_carrier(deadline: Deadline):
                for _ in range(1):
                    check_deadline()
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
            span=(0, 0, 0, 1),
            loop=False,
            check_params=set(),
            ambient_check=False,
            loop_sites=[],
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
                    span=(0, 0, 0, 1),
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
                    span=(1, 0, 1, 5),
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
                    span=(0, 0, 0, 1),
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
                    span=(0, 0, 0, 1),
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
        forest=da.Forest(),
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

    summary = da._summarize_deadline_obligations(
        obligations, max_entries=1, forest=da.Forest()
    )
    assert summary
    assert da._summarize_deadline_obligations([], forest=da.Forest()) == []

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
        report=da.ReportCarrier(
            forest=da.Forest(),
            deadline_obligations=obligations,
        ),
    )
    assert "Deadline propagation:" in report
    assert violations


def test_deadline_obligations_include_call_resolution_requirement(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "mod.py"
    target.write_text(
        textwrap.dedent(
            """
            def root(deadline: Deadline):
                return deadline
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
    forest = da.Forest()
    suite_id = forest.add_suite_site(
        "mod.py",
        "mod.root",
        "call",
        span=(0, 0, 0, 1),
    )
    forest.add_alt(
        "CallResolutionObligation",
        (suite_id,),
        evidence={"callee": "mod.missing", "phase": "unresolved"},
    )
    obligations = da._collect_deadline_obligations(
        [target],
        project_root=tmp_path,
        config=config,
        forest=forest,
    )
    hits = [entry for entry in obligations if entry.get("kind") == "call_resolution_required"]
    assert hits
    assert hits[0].get("status") == "OBLIGATION"
    assert "requires resolution" in str(hits[0].get("detail", ""))


def test_call_resolution_obligation_is_discharged_by_call_candidate(
    tmp_path: Path,
) -> None:
    da = _load()
    target = tmp_path / "mod.py"
    target.write_text(
        textwrap.dedent(
            """
            def root(deadline: Deadline):
                return deadline

            def callee(deadline: Deadline):
                return deadline
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
    forest = da.Forest()
    suite_id = forest.add_suite_site(
        "mod.py",
        "mod.root",
        "call",
        span=(0, 0, 0, 1),
    )
    forest.add_alt(
        "CallResolutionObligation",
        (suite_id,),
        evidence={"callee": "mod.callee", "phase": "unresolved"},
    )
    callee_id = forest.add_site("mod.py", "mod.callee")
    forest.add_alt(
        "CallCandidate",
        (suite_id, callee_id),
        evidence={"resolution": "resolved", "phase": "resolved", "callee": "mod.callee"},
    )

    obligations = da._collect_deadline_obligations(
        [target],
        project_root=tmp_path,
        config=config,
        forest=forest,
    )
    assert not any(
        entry.get("kind") == "call_resolution_required" for entry in obligations
    )


def test_call_edges_include_resolution_obligation_candidates() -> None:
    da = _load()
    caller = _make_fn_info(da, name="root", qual="mod.root", path=Path("mod.py"))
    callee = _make_fn_info(da, name="callee", qual="mod.callee", path=Path("mod.py"))
    by_name = {"callee": [callee]}
    forest = da.Forest()
    call_suite_id = forest.add_suite_site(
        "mod.py",
        "mod.root",
        "call",
        span=(0, 0, 0, 1),
    )
    forest.add_alt(
        "CallResolutionObligation",
        (call_suite_id,),
        evidence={
            "callee": "callee",
            "phase": "unresolved",
            "kind": "unresolved_internal_callee",
        },
    )
    edges = da._collect_call_edges_from_forest(
        forest,
        by_name=by_name,
    )
    caller_id = da.NodeId("SuiteSite", (caller.path.name, caller.qual, "function"))
    callee_id = da.NodeId("SuiteSite", (callee.path.name, callee.qual, "function"))
    assert edges[caller_id] == {callee_id}


# gabion:evidence E:deadline/call_resolution::dataflow_audit.py::gabion.analysis.dataflow_audit._materialize_call_candidates
def test_materialized_call_candidates_target_function_suites(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "mod.py"
    target.write_text(
        textwrap.dedent(
            """
            def root(deadline: Deadline):
                return callee(deadline)

            def callee(deadline: Deadline):
                return deadline
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
    forest = da.Forest()
    da._collect_deadline_obligations(
        [target],
        project_root=tmp_path,
        config=config,
        forest=forest,
    )
    call_candidate_targets = [
        forest.nodes[alt.inputs[1]]
        for alt in forest.alts
        if alt.kind == "CallCandidate"
        and len(alt.inputs) >= 2
        and alt.inputs[1] in forest.nodes
    ]
    assert any(
        node.kind == "SuiteSite" and node.meta.get("suite_kind") == "function"
        for node in call_candidate_targets
    )


def test_deadline_summary_handles_bad_span() -> None:
    da = _load()
    from gabion.exceptions import NeverThrown

    entries = [
        {
            "deadline_id": "deadline:mod.py:f:missing",
            "site": {"path": "mod.py", "function": "f", "bundle": []},
            "status": "VIOLATION",
            "kind": "missing",
            "detail": "oops",
            "span": ["x", "y", "z", "w"],
        }
    ]
    with pytest.raises(NeverThrown):
        da._summarize_deadline_obligations(
            entries, max_entries=1, forest=da.Forest()
        )


def test_deadline_summary_materializes_spec_facets() -> None:
    da = _load()
    forest = da.Forest()
    entries = [
        {
            "deadline_id": "deadline:mod.py:f:missing",
            "site": {"path": "mod.py", "function": "f", "bundle": []},
            "status": "VIOLATION",
            "kind": "missing",
            "detail": "oops",
            "span": [0, 1, 0, 2],
        }
    ]
    summary = da._summarize_deadline_obligations(entries, max_entries=1, forest=forest)
    assert summary
    spec_sites = [
        node
        for node in forest.nodes.values()
        if node.kind == "SuiteSite" and node.meta.get("suite_kind") == "spec"
    ]
    assert spec_sites
    assert any(alt.kind == "SpecFacet" for alt in forest.alts)


def test_suite_order_spec_materializes_spec_facets() -> None:
    da = _load()
    forest = da.Forest()
    forest.add_suite_site("mod.py", "mod.fn", "loop", span=(0, 0, 0, 1))
    da._materialize_suite_order_spec(forest=forest)
    spec_sites = [
        node
        for node in forest.nodes.values()
        if node.kind == "SuiteSite"
        and node.meta.get("suite_kind") == "spec"
        and node.meta.get("spec_name") == "suite_order"
    ]
    assert spec_sites
    spec_facets = [alt for alt in forest.alts if alt.kind == "SpecFacet"]
    assert spec_facets
    assert any("order_key" in alt.evidence for alt in spec_facets)


def test_suite_order_relation_skips_spec_sites() -> None:
    da = _load()
    forest = da.Forest()
    forest.add_spec_site(
        spec_hash="spec",
        spec_name="suite_order",
        spec_domain="suite_order",
        spec_version=1,
    )
    relation, suite_index = da._suite_order_relation(forest)
    assert relation == []
    assert suite_index == {}


def test_suite_order_relation_requires_path_and_qual() -> None:
    da = _load()
    from gabion.exceptions import NeverThrown

    forest = da.Forest()
    forest.add_node(
        "SuiteSite",
        ("missing",),
        {"suite_kind": "loop", "span": [0, 0, 0, 1]},
    )
    with pytest.raises(NeverThrown):
        da._suite_order_relation(forest)


def test_suite_order_relation_requires_span() -> None:
    da = _load()
    from gabion.exceptions import NeverThrown

    forest = da.Forest()
    forest.add_node(
        "SuiteSite",
        ("missing-span",),
        {"suite_kind": "loop", "path": "mod.py", "qual": "mod.fn"},
    )
    with pytest.raises(NeverThrown):
        da._suite_order_relation(forest)


def test_suite_order_relation_requires_int_span_fields() -> None:
    da = _load()
    from gabion.exceptions import NeverThrown

    forest = da.Forest()
    forest.add_node(
        "SuiteSite",
        ("bad-span",),
        {"suite_kind": "loop", "path": "mod.py", "qual": "mod.fn", "span": ["x", 0, 0, 1]},
    )
    with pytest.raises(NeverThrown):
        da._suite_order_relation(forest)


def test_suite_order_row_to_site_rejects_missing_fields() -> None:
    da = _load()
    suite_index = {}
    assert da._suite_order_row_to_site(
        {"suite_path": "", "suite_qual": "q", "suite_kind": "loop"},
        suite_index,
    ) is None


def test_suite_order_row_to_site_rejects_invalid_span() -> None:
    da = _load()
    suite_index = {}
    assert da._suite_order_row_to_site(
        {
            "suite_path": "mod.py",
            "suite_qual": "mod.fn",
            "suite_kind": "loop",
            "span_line": "x",
            "span_col": 0,
            "span_end_line": 0,
            "span_end_col": 1,
        },
        suite_index,
    ) is None


def test_spec_row_span_handles_invalid_and_valid() -> None:
    da = _load()
    from gabion.exceptions import NeverThrown

    with pytest.raises(NeverThrown):
        da._spec_row_span({"span_line": "x"})
    with pytest.raises(NeverThrown):
        da._spec_row_span(
            {"span_line": -1, "span_col": 0, "span_end_line": 0, "span_end_col": 1}
        )
    assert da._spec_row_span(
        {"span_line": 1, "span_col": 2, "span_end_line": 3, "span_end_col": 4}
    ) == (1, 2, 3, 4)


def test_spec_row_span_raises_on_none() -> None:
    da = _load()
    from gabion.exceptions import NeverThrown

    with pytest.raises(NeverThrown):
        da._spec_row_span({"span_line": None})


def test_materialize_projection_spec_rows_handles_empty_and_missing_site() -> None:
    da = _load()
    spec = da.DEADLINE_OBLIGATIONS_SUMMARY_SPEC
    da._materialize_projection_spec_rows(
        spec=spec,
        projected=[],
        forest=da.Forest(),
        row_to_site=lambda row: None,
    )
    forest = da.Forest()
    da._materialize_projection_spec_rows(
        spec=spec,
        projected=[{"site_path": "mod.py", "site_function": "f"}],
        forest=forest,
        row_to_site=lambda row: None,
    )
    assert not any(alt.kind == "SpecFacet" for alt in forest.alts)


def test_deadline_summary_row_to_site_handles_missing_path() -> None:
    da = _load()
    forest = da.Forest()
    entries = [
        {
            "deadline_id": "deadline::f:missing",
            "site": {"path": "", "function": "f", "bundle": []},
            "status": "VIOLATION",
            "kind": "missing",
            "detail": "oops",
            "span": [0, 0, 0, 1],
        }
    ]
    summary = da._summarize_deadline_obligations(entries, max_entries=1, forest=forest)
    assert summary

    entries = [
        {
            "deadline_id": "deadline:mod.py:f:missing",
            "site": {"path": "mod.py", "function": "f", "bundle": []},
            "status": "VIOLATION",
            "kind": "missing",
            "detail": "oops",
            "span": [0, 0, 0, 1],
        }
    ]
    summary = da._summarize_deadline_obligations(entries, max_entries=1, forest=forest)
    assert summary


def test_deadline_obligation_span_fallbacks_param_and_facts(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "mod.py"
    target.write_text(
        textwrap.dedent(
            """
            def root(deadline):
                pass

            def callee(deadline):
                pass

            def loop_missing_check(deadline):
                pass
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    config = da.AuditConfig(
        project_root=tmp_path,
        deadline_roots={"mod.root"},
    )
    extra_call_infos = {
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
                {},
            )
        ]
    }
    extra_facts = {
        "mod.loop_missing_check": da._DeadlineFunctionFacts(
            path=target,
            qual="mod.loop_missing_check",
            span=(5, 0, 5, 10),
            loop=True,
            check_params=set(),
            ambient_check=False,
            loop_sites=[da._DeadlineLoopFacts(span=None, kind="for")],
            local_info=da._DeadlineLocalInfo(
                origin_vars=set(),
                origin_spans={},
                alias_to_param={},
            ),
        )
    }
    obligations = da._collect_deadline_obligations(
        [target],
        project_root=tmp_path,
        config=config,
        forest=da.Forest(),
        extra_call_infos=extra_call_infos,
        extra_facts_by_qual=extra_facts,
    )
    _, by_qual = da._build_function_index(
        [target],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        transparent_decorators=None,
    )
    root_span = by_qual["mod.root"].param_spans["deadline"]
    assert any(
        entry.get("kind") == "missing_arg"
        and entry.get("span") == list(root_span)
        for entry in obligations
    )
    assert any(
        entry.get("kind") == "unchecked_deadline"
        and entry.get("span") == list(extra_facts["mod.loop_missing_check"].span)
        for entry in obligations
    )


def test_deadline_obligation_span_fallback_missing_raises(tmp_path: Path) -> None:
    da = _load()
    from gabion.exceptions import NeverThrown

    target = tmp_path / "mod.py"
    target.write_text(
        textwrap.dedent(
            """
            def root():
                pass

            def callee(deadline):
                pass
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    config = da.AuditConfig(
        project_root=tmp_path,
        deadline_roots={"mod.root"},
    )
    extra_call_infos = {
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
                {},
            )
        ]
    }
    extra_facts = {
        "mod.root": da._DeadlineFunctionFacts(
            path=target,
            qual="mod.root",
            span=None,
            loop=False,
            check_params=set(),
            ambient_check=False,
            loop_sites=[],
            local_info=da._DeadlineLocalInfo(
                origin_vars=set(),
                origin_spans={},
                alias_to_param={},
            ),
        )
    }
    with pytest.raises(NeverThrown):
        da._collect_deadline_obligations(
            [target],
            project_root=tmp_path,
            config=config,
            forest=da.Forest(),
            extra_call_infos=extra_call_infos,
            extra_facts_by_qual=extra_facts,
        )


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
        forest=da.Forest(),
    )
    assert obligations is not None


def test_deadline_obligations_emit_suite_sites(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "mod.py"
    target.write_text(
        textwrap.dedent(
            """
            def loop_missing_check(deadline: Deadline):
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
        deadline_roots={"mod.loop_missing_check"},
    )
    forest = da.Forest()
    obligations = da._collect_deadline_obligations(
        [target],
        project_root=tmp_path,
        config=config,
        forest=forest,
    )
    assert obligations
    assert any(node.kind == "SuiteSite" for node in forest.nodes.values())
    assert any(alt.kind == "DeadlineObligation" for alt in forest.alts)
    assert any(alt.kind == "SuiteSiteInFunction" for alt in forest.alts)


def test_deadline_recursion_missing_carrier(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "mod.py"
    target.write_text(
        textwrap.dedent(
            """
            def recur():
                return recur()
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
    obligations = da._collect_deadline_obligations(
        [target],
        project_root=tmp_path,
        config=config,
        forest=da.Forest(),
    )
    assert any(entry.get("kind") == "missing_carrier" for entry in obligations)


def test_deadline_recursion_unchecked(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "mod.py"
    target.write_text(
        textwrap.dedent(
            """
            def recur(deadline: Deadline):
                return recur(None)
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
    obligations = da._collect_deadline_obligations(
        [target],
        project_root=tmp_path,
        config=config,
        forest=da.Forest(),
    )
    assert any(entry.get("kind") == "unchecked_deadline" for entry in obligations)


def test_deadline_recursion_loop_ambient_no_carrier(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "mod.py"
    target.write_text(
        textwrap.dedent(
            """
            def recur():
                for _ in range(1):
                    check_deadline()
                return recur()
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
    obligations = da._collect_deadline_obligations(
        [target],
        project_root=tmp_path,
        config=config,
        forest=da.Forest(),
    )
    assert not any(entry.get("kind") == "missing_carrier" for entry in obligations)


def test_deadline_recursion_loop_ambient_with_carrier(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "mod.py"
    target.write_text(
        textwrap.dedent(
            """
            def recur(deadline: Deadline):
                for _ in range(1):
                    check_deadline()
                return recur(deadline)
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
    obligations = da._collect_deadline_obligations(
        [target],
        project_root=tmp_path,
        config=config,
        forest=da.Forest(),
    )
    assert not any(entry.get("kind") == "unchecked_deadline" for entry in obligations)


def test_deadline_recursion_skips_missing_facts(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "mod.py"
    target.write_text(
        textwrap.dedent(
            """
            def recur(deadline: Deadline):
                return recur(deadline)
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
    obligations = da._collect_deadline_obligations(
        [target],
        project_root=tmp_path,
        config=config,
        forest=da.Forest(),
        extra_facts_by_qual={"mod.recur": None},
    )
    assert obligations is not None


def test_deadline_exempt_prefix_is_skipped(tmp_path: Path) -> None:
    da = _load()
    dummy = tmp_path / "mod.py"
    dummy.write_text("def f():\n    return 1\n", encoding="utf-8")
    facts = {
        "gabion.analysis.timeout_context.fake": da._DeadlineFunctionFacts(
            path=Path("timeout_context.py"),
            qual="gabion.analysis.timeout_context.fake",
            span=(0, 0, 0, 1),
            loop=False,
            check_params=set(),
            ambient_check=False,
            loop_sites=[],
            local_info=da._DeadlineLocalInfo(
                origin_vars=set(),
                origin_spans={},
                alias_to_param={},
            ),
        )
    }
    config = da.AuditConfig(
        project_root=tmp_path,
        exclude_dirs=set(),
        ignore_params=set(),
        external_filter=True,
        strictness="high",
        deadline_roots={"mod.root"},
    )
    obligations = da._collect_deadline_obligations(
        [dummy],
        project_root=tmp_path,
        config=config,
        forest=da.Forest(),
        extra_facts_by_qual=facts,
    )
    assert obligations is not None


def test_deadline_loop_requires_check_in_body(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "mod.py"
    target.write_text(
        textwrap.dedent(
            """
            def loop_precheck(deadline: Deadline):
                check_deadline(deadline)
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
    obligations = da._collect_deadline_obligations(
        [target],
        project_root=tmp_path,
        config=config,
        forest=da.Forest(),
    )
    assert any(entry.get("kind") == "unchecked_deadline" for entry in obligations)
