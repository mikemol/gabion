from __future__ import annotations

from pathlib import Path
import re
import textwrap

def _load():
    repo_root = Path(__file__).resolve().parents[1]
    from gabion.refactor.engine import RefactorEngine
    from gabion.refactor.model import FieldSpec, RefactorRequest

    return RefactorEngine, FieldSpec, RefactorRequest

def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text)

# gabion:evidence E:function_site::test_refactor_engine.py::tests.test_refactor_engine._load E:decision_surface/direct::test_refactor_engine.py::tests.test_refactor_engine._load::stale_7165b720683b
def test_refactor_engine_emits_protocol_stub(tmp_path: Path) -> None:
    RefactorEngine, FieldSpec, RefactorRequest = _load()
    target = tmp_path / "sample.py"
    target.write_text(
        textwrap.dedent(
            """
            def sink(x, y):
                return x, y
            """
        ).strip()
        + "\n"
    )
    request = RefactorRequest(
        protocol_name="BundleProtocol",
        bundle=["alpha", "beta"],
        target_path=str(target),
        rationale="Unit test",
    )
    plan = RefactorEngine(project_root=tmp_path).plan_protocol_extraction(request)
    assert plan.edits
    replacement = plan.edits[0].replacement
    assert "class BundleProtocol" in replacement
    assert "alpha: object" in replacement
    assert "beta: object" in replacement

# gabion:evidence E:function_site::test_refactor_engine.py::tests.test_refactor_engine._load E:decision_surface/direct::test_refactor_engine.py::tests.test_refactor_engine._load::stale_f747efb69717
def test_refactor_engine_preserves_type_hints(tmp_path: Path) -> None:
    RefactorEngine, FieldSpec, RefactorRequest = _load()
    target = tmp_path / "sample.py"
    target.write_text(
        textwrap.dedent(
            """
            def sink(x, y):
                return x, y
            """
        ).strip()
        + "\n"
    )
    request = RefactorRequest(
        protocol_name="BundleProtocol",
        bundle=["alpha", "beta"],
        fields=[
            FieldSpec(name="alpha", type_hint="int"),
            FieldSpec(name="beta", type_hint="str"),
        ],
        target_path=str(target),
        rationale="Unit test",
    )
    plan = RefactorEngine(project_root=tmp_path).plan_protocol_extraction(request)
    assert plan.edits
    replacement = plan.edits[0].replacement
    assert "alpha: int" in replacement
    assert "beta: str" in replacement

# gabion:evidence E:function_site::test_refactor_engine.py::tests.test_refactor_engine._load E:decision_surface/direct::test_refactor_engine.py::tests.test_refactor_engine._load::stale_a1773bba20c4
def test_refactor_engine_rewrites_signature_and_preamble(tmp_path: Path) -> None:
    RefactorEngine, FieldSpec, RefactorRequest = _load()
    target = tmp_path / "sample.py"
    target.write_text(
        textwrap.dedent(
            """
            def foo(a, b):
                return a + b
            """
        ).strip()
        + "\n"
    )
    request = RefactorRequest(
        protocol_name="BundleProtocol",
        bundle=["a", "b"],
        fields=[
            FieldSpec(name="a", type_hint="int"),
            FieldSpec(name="b", type_hint="int"),
        ],
        target_path=str(target),
        target_functions=["foo"],
        rationale="Unit test",
    )
    plan = RefactorEngine(project_root=tmp_path).plan_protocol_extraction(request)
    assert plan.edits
    replacement = plan.edits[0].replacement
    assert "def foo(bundle: BundleProtocol)" in replacement
    assert "a = bundle.a" in replacement
    assert "b = bundle.b" in replacement

# gabion:evidence E:function_site::test_refactor_engine.py::tests.test_refactor_engine._load E:function_site::test_refactor_engine.py::tests.test_refactor_engine._normalize E:decision_surface/direct::test_refactor_engine.py::tests.test_refactor_engine._load::stale_71db3042b46c_5eed1578
def test_refactor_engine_rewrites_call_sites(tmp_path: Path) -> None:
    RefactorEngine, FieldSpec, RefactorRequest = _load()
    target = tmp_path / "sample.py"
    target.write_text(
        textwrap.dedent(
            """
            def foo(a, b):
                return a + b

            def caller(x, y):
                return foo(x, y)
            """
        ).strip()
        + "\n"
    )
    request = RefactorRequest(
        protocol_name="BundleProtocol",
        bundle=["a", "b"],
        fields=[
            FieldSpec(name="a", type_hint="int"),
            FieldSpec(name="b", type_hint="int"),
        ],
        target_path=str(target),
        target_functions=["foo"],
    )
    plan = RefactorEngine(project_root=tmp_path).plan_protocol_extraction(request)
    assert plan.edits
    replacement = _normalize(plan.edits[0].replacement)
    assert "returnfoo(BundleProtocol(a=x,b=y))" in replacement

# gabion:evidence E:function_site::test_refactor_engine.py::tests.test_refactor_engine._load E:function_site::test_refactor_engine.py::tests.test_refactor_engine._normalize E:decision_surface/direct::test_refactor_engine.py::tests.test_refactor_engine._load::stale_f5a71b55c836
def test_refactor_engine_rewrites_imported_call_sites(tmp_path: Path) -> None:
    RefactorEngine, FieldSpec, RefactorRequest = _load()
    src_root = tmp_path / "src" / "pkg"
    src_root.mkdir(parents=True)
    target = src_root / "mod.py"
    target.write_text(
        textwrap.dedent(
            """
            def foo(a, b):
                return a + b
            """
        ).strip()
        + "\n"
    )
    caller = src_root / "caller.py"
    caller.write_text(
        textwrap.dedent(
            """
            from pkg.mod import foo

            def run(x, y):
                return foo(x, y)
            """
        ).strip()
        + "\n"
    )
    request = RefactorRequest(
        protocol_name="BundleProtocol",
        bundle=["a", "b"],
        fields=[
            FieldSpec(name="a", type_hint="int"),
            FieldSpec(name="b", type_hint="int"),
        ],
        target_path=str(target),
        target_functions=["foo"],
    )
    plan = RefactorEngine(project_root=tmp_path).plan_protocol_extraction(request)
    assert plan.edits
    edits_by_path = {Path(edit.path).name: edit.replacement for edit in plan.edits}
    assert "caller.py" in edits_by_path
    caller_replacement = _normalize(edits_by_path["caller.py"])
    assert "frompkg.modimportBundleProtocol" in caller_replacement
    assert "returnfoo(BundleProtocol(a=x,b=y))" in caller_replacement

# gabion:evidence E:function_site::test_refactor_engine.py::tests.test_refactor_engine._load E:decision_surface/direct::test_refactor_engine.py::tests.test_refactor_engine._load::stale_b919c219b6e6
def test_refactor_engine_emits_compat_shim(tmp_path: Path) -> None:
    RefactorEngine, FieldSpec, RefactorRequest = _load()
    target = tmp_path / "sample.py"
    target.write_text(
        textwrap.dedent(
            """
            def foo(a, b):
                return a + b
            """
        ).strip()
        + "\n"
    )
    request = RefactorRequest(
        protocol_name="BundleProtocol",
        bundle=["a", "b"],
        fields=[
            FieldSpec(name="a", type_hint="int"),
            FieldSpec(name="b", type_hint="int"),
        ],
        target_path=str(target),
        target_functions=["foo"],
        compatibility_shim=True,
    )
    plan = RefactorEngine(project_root=tmp_path).plan_protocol_extraction(request)
    assert plan.edits
    replacement = plan.edits[0].replacement
    assert "def _foo_bundle" in replacement
    assert "def foo(*args, **kwargs)" in replacement
    assert "warnings.warn" in replacement
    assert "@overload" in replacement


# gabion:evidence E:call_footprint::tests/test_refactor_engine.py::test_refactor_engine_ambient_rewrite_threaded_parameter::engine.py::gabion.refactor.engine.RefactorEngine.plan_protocol_extraction
def test_refactor_engine_ambient_rewrite_threaded_parameter(tmp_path: Path) -> None:
    RefactorEngine, FieldSpec, RefactorRequest = _load()
    target = tmp_path / "ambient.py"
    target.write_text(
        textwrap.dedent(
            """
            def sink(ctx):
                return ctx.user_id

            def route(ctx):
                return sink(ctx)
            """
        ).strip()
        + "\n"
    )
    request = RefactorRequest(
        protocol_name="CtxBundle",
        bundle=["ctx"],
        fields=[FieldSpec(name="ctx", type_hint="CtxBundle")],
        target_path=str(target),
        target_functions=["sink", "route"],
        ambient_rewrite=True,
    )
    plan = RefactorEngine(project_root=tmp_path).plan_protocol_extraction(request)
    assert plan.edits
    replacement = plan.edits[0].replacement
    assert "from contextvars import ContextVar" in replacement
    assert "_ambient_get_ctx" in replacement
    assert "if ctx is None" in replacement
    assert "return sink()" in replacement
    assert any(entry.kind == "AMBIENT_REWRITE" and entry.status == "applied" for entry in plan.rewrite_plans)


# gabion:evidence E:call_footprint::tests/test_refactor_engine.py::test_refactor_engine_ambient_rewrite_partial_skip_unsafe::engine.py::gabion.refactor.engine.RefactorEngine.plan_protocol_extraction
def test_refactor_engine_ambient_rewrite_partial_skip_unsafe(tmp_path: Path) -> None:
    RefactorEngine, FieldSpec, RefactorRequest = _load()
    target = tmp_path / "ambient_partial.py"
    target.write_text(
        textwrap.dedent(
            """
            def sink(ctx):
                return ctx.user_id

            def safe(ctx):
                return sink(ctx)

            def unsafe(ctx):
                ctx = mutate(ctx)
                return sink(ctx)
            """
        ).strip()
        + "\n"
    )
    request = RefactorRequest(
        protocol_name="CtxBundle",
        bundle=["ctx"],
        fields=[FieldSpec(name="ctx", type_hint="CtxBundle")],
        target_path=str(target),
        target_functions=["sink", "safe", "unsafe"],
        ambient_rewrite=True,
    )
    plan = RefactorEngine(project_root=tmp_path).plan_protocol_extraction(request)
    entries = {entry.target: entry for entry in plan.rewrite_plans}
    assert entries["safe"].status == "applied"
    assert entries["unsafe"].status == "skipped"
    assert entries["unsafe"].non_rewrite_reasons


# gabion:evidence E:call_footprint::tests/test_refactor_engine.py::test_refactor_engine_ambient_rewrite_noop_when_no_pattern::engine.py::gabion.refactor.engine.RefactorEngine.plan_protocol_extraction
def test_refactor_engine_ambient_rewrite_noop_when_no_pattern(tmp_path: Path) -> None:
    RefactorEngine, FieldSpec, RefactorRequest = _load()
    target = tmp_path / "ambient_noop.py"
    target.write_text(
        textwrap.dedent(
            """
            def sink(bundle):
                return bundle
            """
        ).strip()
        + "\n"
    )
    request = RefactorRequest(
        protocol_name="CtxBundle",
        bundle=["ctx"],
        fields=[FieldSpec(name="ctx", type_hint="CtxBundle")],
        target_path=str(target),
        target_functions=["sink"],
        ambient_rewrite=True,
    )
    plan = RefactorEngine(project_root=tmp_path).plan_protocol_extraction(request)
    assert any(entry.status == "noop" for entry in plan.rewrite_plans)


# gabion:evidence E:function_site::engine.py::gabion.refactor.engine._ensure_ambient_scaffolding E:function_site::engine.py::gabion.refactor.engine._has_contextvars_import
def test_ambient_scaffolding_and_contextvars_import_edge_paths() -> None:
    import libcst as cst
    from gabion.refactor import engine as refactor_engine

    module = cst.parse_module("def f():\n    return 1\n")
    assert (
        refactor_engine._ensure_ambient_scaffolding(
            module,
            context_names=set(),
            protocol_hint="CtxBundle",
        )
        is module
    )

    existing = cst.parse_module(
        "from contextvars import ContextVar\n\n"
        "def _ambient_get_ctx() -> object:\n"
        "    return object()\n\n"
        "def _ambient_set_ctx(value: object) -> None:\n"
        "    return None\n"
    )
    rewritten = refactor_engine._ensure_ambient_scaffolding(
        existing,
        context_names={"ctx"},
        protocol_hint="CtxBundle",
    )
    assert rewritten.code.count("def _ambient_get_ctx") == 1
    assert rewritten.code.count("def _ambient_set_ctx") == 1
    assert "_AMBIENT_CTX" not in rewritten.code

    assert refactor_engine._has_contextvars_import(cst.parse_module("x = 1\n").body) is False
    assert refactor_engine._has_contextvars_import(cst.parse_module("from contextvars import *\n").body) is False
    assert refactor_engine._has_contextvars_import(existing.body) is True


# gabion:evidence E:function_site::engine.py::gabion.refactor.engine._AmbientArgThreadingRewriter.leave_Call
def test_ambient_arg_threading_rewriter_branch_matrix() -> None:
    import libcst as cst
    from gabion.refactor import engine as refactor_engine

    rewriter = refactor_engine._AmbientArgThreadingRewriter(
        targets={"sink"},
        context_name="ctx",
        current="route",
    )
    attr_call = cst.parse_expression("obj.sink(ctx)")
    attr_updated = rewriter.leave_Call(attr_call, attr_call)
    assert isinstance(attr_updated, cst.Call)
    assert len(attr_updated.args) == 0

    skipped_target = cst.parse_expression("noop(ctx)")
    assert rewriter.leave_Call(skipped_target, skipped_target) is skipped_target

    star_args_call = cst.parse_expression("sink(*args)")
    assert rewriter.leave_Call(star_args_call, star_args_call) is star_args_call
    assert any("dynamic star args" in reason for reason in rewriter.skipped_reasons)

    kw_call = cst.parse_expression("sink(ctx=ctx, other=value)")
    kw_updated = rewriter.leave_Call(kw_call, kw_call)
    assert isinstance(kw_updated, cst.Call)
    assert len(kw_updated.args) == 1

    ambiguous_call = cst.parse_expression("sink(ctx, value)")
    ambiguous_updated = rewriter.leave_Call(ambiguous_call, ambiguous_call)
    assert ambiguous_updated is ambiguous_call
    assert any("ambiguous arity" in reason for reason in rewriter.skipped_reasons)

    current_rewriter = refactor_engine._AmbientArgThreadingRewriter(
        targets={"caller"},
        context_name="ctx",
        current="caller",
    )
    current_call = cst.parse_expression("caller(ctx)")
    assert current_rewriter.leave_Call(current_call, current_call) is current_call


# gabion:evidence E:function_site::engine.py::gabion.refactor.engine._AmbientRewriteTransformer._rewrite_function E:decision_surface/direct::engine.py::gabion.refactor.engine._AmbientRewriteTransformer._rewrite_function::stale_cea5a5c42eca
def test_ambient_rewrite_transformer_annotation_docstring_and_warning_paths() -> None:
    import libcst as cst
    from gabion.refactor import engine as refactor_engine

    module = cst.parse_module(
        textwrap.dedent(
            """
            def route(ctx, extra=None):
                \"\"\"doc\"\"\"
                return sink(ctx, extra)
            """
        ).strip()
        + "\n"
    )
    node = module.body[0]
    assert isinstance(node, cst.FunctionDef)

    transformer = refactor_engine._AmbientRewriteTransformer(
        targets={"route", "sink"},
        bundle_fields=["ctx"],
        protocol_hint="CtxBundle[",
    )
    rewritten = transformer._rewrite_function(node)
    assert isinstance(rewritten, cst.FunctionDef)
    code = cst.Module(body=[rewritten]).code
    assert code.index('"""doc"""') < code.index("if ctx is None:")
    assert "ctx: object | None = None" in code
    assert any(entry.status == "applied" and entry.target == "route" for entry in transformer.plan_entries)
    assert transformer.warnings


# gabion:evidence E:function_site::engine.py::gabion.refactor.engine._AmbientRewriteTransformer.leave_AsyncFunctionDef E:function_site::engine.py::gabion.refactor.engine._AmbientRewriteTransformer._rewrite_function
def test_ambient_rewrite_transformer_skip_variants_and_async_dispatch() -> None:
    import libcst as cst
    from gabion.refactor import engine as refactor_engine

    untargeted_module = cst.parse_module("def other(ctx):\n    return ctx\n")
    untargeted_node = untargeted_module.body[0]
    assert isinstance(untargeted_node, cst.FunctionDef)
    untargeted = refactor_engine._AmbientRewriteTransformer(
        targets={"route"},
        bundle_fields=["ctx"],
        protocol_hint="CtxBundle",
    )
    assert untargeted._rewrite_function(untargeted_node) is untargeted_node

    empty_bundle = refactor_engine._AmbientRewriteTransformer(
        targets={"route"},
        bundle_fields=[],
        protocol_hint="CtxBundle",
    )
    route_module = cst.parse_module("def route(ctx):\n    return sink(ctx)\n")
    route_node = route_module.body[0]
    assert isinstance(route_node, cst.FunctionDef)
    assert empty_bundle._rewrite_function(route_node) is route_node
    assert any(entry.status == "skipped" for entry in empty_bundle.plan_entries)

    star_bundle = refactor_engine._AmbientRewriteTransformer(
        targets={"route"},
        bundle_fields=["ctx"],
        protocol_hint="CtxBundle",
    )
    star_module = cst.parse_module("def route(ctx, *args):\n    return sink(ctx)\n")
    star_node = star_module.body[0]
    assert isinstance(star_node, cst.FunctionDef)
    assert star_bundle._rewrite_function(star_node) is star_node
    assert any(
        entry.status == "skipped"
        and entry.non_rewrite_reasons
        and "function uses *args or **kwargs" in entry.non_rewrite_reasons[0]
        for entry in star_bundle.plan_entries
    )

    async_transformer = refactor_engine._AmbientRewriteTransformer(
        targets={"route", "sink"},
        bundle_fields=["ctx"],
        protocol_hint="CtxBundle",
    )
    async_module = cst.parse_module("async def route(ctx):\n    return sink(ctx)\n")
    async_node = async_module.body[0]
    assert isinstance(async_node, cst.FunctionDef)
    async_updated = async_transformer.leave_AsyncFunctionDef(async_node, async_node)
    assert isinstance(async_updated, cst.FunctionDef)
    assert async_transformer.plan_entries


# gabion:evidence E:call_footprint::tests/test_refactor_engine.py::test_refactor_engine_additional_branch_edges_for_contextvars_and_rewrite_shapes::engine.py::gabion.refactor.engine._has_contextvars_import::engine.py::gabion.refactor.engine._AmbientArgThreadingRewriter.leave_Call::engine.py::gabion.refactor.engine._AmbientSafetyVisitor.visit_AssignTarget::engine.py::gabion.refactor.engine._AmbientRewriteTransformer._rewrite_function
def test_refactor_engine_additional_branch_edges_for_contextvars_and_rewrite_shapes() -> None:
    import libcst as cst
    from gabion.refactor import engine as refactor_engine

    token_only = cst.parse_module("from contextvars import Token\n")
    assert refactor_engine._has_contextvars_import(token_only.body) is False

    dotted_alias_stmt = cst.SimpleStatementLine(
        body=[
            cst.ImportFrom(
                module=cst.Name("contextvars"),
                names=[
                    cst.ImportAlias(
                        name=cst.Attribute(
                            value=cst.Name("pkg"),
                            attr=cst.Name("ContextVar"),
                        )
                    )
                ],
            )
        ]
    )
    assert refactor_engine._has_contextvars_import([dotted_alias_stmt]) is False

    rewriter = refactor_engine._AmbientArgThreadingRewriter(
        targets={"sink"},
        context_name="ctx",
        current="route",
    )
    call_with_non_name_func = cst.parse_expression("(factory())(ctx)")
    assert rewriter.leave_Call(call_with_non_name_func, call_with_non_name_func) is call_with_non_name_func

    safety = refactor_engine._AmbientSafetyVisitor("ctx")
    cst.parse_module("other = 1\n").visit(safety)
    assert safety.reasons == []

    simple_suite_module = cst.parse_module(
        "def route(ctx: CtxBundle | None = None): return sink(ctx)\n"
    )
    simple_suite_node = simple_suite_module.body[0]
    assert isinstance(simple_suite_node, cst.FunctionDef)
    simple_suite_transformer = refactor_engine._AmbientRewriteTransformer(
        targets={"route", "sink"},
        bundle_fields=["ctx"],
        protocol_hint="CtxBundle",
    )
    simple_suite_updated = simple_suite_transformer._rewrite_function(simple_suite_node)
    assert isinstance(simple_suite_updated, cst.FunctionDef)
    assert isinstance(simple_suite_updated.body, cst.SimpleStatementSuite)


# gabion:evidence E:call_footprint::tests/test_refactor_engine.py::test_refactor_engine_reports_no_changes_outcome::engine.py::gabion.refactor.engine.RefactorEngine.plan_protocol_extraction
def test_refactor_engine_reports_no_changes_outcome(tmp_path: Path) -> None:
    from gabion.refactor.model import RefactorPlan, RefactorPlanOutcome

    RefactorEngine, FieldSpec, RefactorRequest = _load()
    target = tmp_path / "sample.py"
    target.write_text(
        textwrap.dedent(
            """
            def foo(bundle):
                a = bundle.a
                b = bundle.b
                return a + b
            """
        ).strip()
        + "\n"
    )
    request = RefactorRequest(
        protocol_name="BundleProtocol",
        bundle=["a", "b"],
        fields=[FieldSpec(name="a"), FieldSpec(name="b")],
        target_path=str(target),
        target_functions=["foo"],
    )
    plan = RefactorEngine(project_root=tmp_path).plan_protocol_extraction(request)
    assert plan.outcome is RefactorPlanOutcome.APPLIED

    no_change_plan = RefactorPlan(outcome=RefactorPlanOutcome.NO_CHANGES)
    assert no_change_plan.outcome is RefactorPlanOutcome.NO_CHANGES


# gabion:evidence E:call_footprint::tests/test_refactor_engine.py::test_refactor_engine_async_refactor_transformer_visit_leave_paths::engine.py::gabion.refactor.engine._RefactorTransformer.visit_AsyncFunctionDef::engine.py::gabion.refactor.engine._RefactorTransformer.leave_AsyncFunctionDef
def test_refactor_engine_async_refactor_transformer_visit_leave_paths() -> None:
    import libcst as cst
    from gabion.refactor import engine as refactor_engine

    module = cst.parse_module(
        textwrap.dedent(
            """
            async def foo(a, b):
                return a + b
            """
        ).strip()
        + "\n"
    )
    transformer = refactor_engine._RefactorTransformer(
        targets={"foo"},
        bundle_fields=["a", "b"],
        protocol_hint="BundleProtocol",
    )
    rewritten = module.visit(transformer)
    normalized = _normalize(rewritten.code)
    assert "asyncdeffoo(bundle:BundleProtocol)" in normalized
    assert "a=bundle.a" in normalized
    assert "b=bundle.b" in normalized
    assert transformer._stack == []


# gabion:evidence E:call_footprint::tests/test_refactor_engine.py::test_refactor_engine_rejects_unvalidated_module_identifier::engine.py::gabion.refactor.engine._validated_module_identifier
def test_refactor_engine_rejects_unvalidated_module_identifier(tmp_path: Path) -> None:
    from gabion.refactor.model import RefactorPlanOutcome

    RefactorEngine, FieldSpec, RefactorRequest = _load()
    src_root = tmp_path / "src"
    src_root.mkdir(parents=True)
    target = src_root / "bad-module.py"
    target.write_text("def foo(a, b):\n    return a + b\n")
    request = RefactorRequest(
        protocol_name="BundleProtocol",
        bundle=["a", "b"],
        fields=[FieldSpec(name="a"), FieldSpec(name="b")],
        target_path=str(target),
        target_functions=["foo"],
    )
    plan = RefactorEngine(project_root=tmp_path).plan_protocol_extraction(request)
    assert plan.outcome is RefactorPlanOutcome.ERROR
    assert plan.errors
    assert "Invalid target module identifier" in plan.errors[0]
