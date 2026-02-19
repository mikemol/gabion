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

# gabion:evidence E:function_site::test_refactor_engine.py::tests.test_refactor_engine._load
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

# gabion:evidence E:function_site::test_refactor_engine.py::tests.test_refactor_engine._load
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

# gabion:evidence E:function_site::test_refactor_engine.py::tests.test_refactor_engine._load
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

# gabion:evidence E:function_site::test_refactor_engine.py::tests.test_refactor_engine._load E:function_site::test_refactor_engine.py::tests.test_refactor_engine._normalize
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

# gabion:evidence E:function_site::test_refactor_engine.py::tests.test_refactor_engine._load E:function_site::test_refactor_engine.py::tests.test_refactor_engine._normalize
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

# gabion:evidence E:function_site::test_refactor_engine.py::tests.test_refactor_engine._load
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
