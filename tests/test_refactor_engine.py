from __future__ import annotations

from pathlib import Path
import re
import sys
import textwrap


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.refactor.engine import RefactorEngine
    from gabion.refactor.model import FieldSpec, RefactorRequest

    return RefactorEngine, FieldSpec, RefactorRequest


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text)


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
