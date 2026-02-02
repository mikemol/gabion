from __future__ import annotations

from pathlib import Path
import sys
import textwrap


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.refactor.engine import RefactorEngine
    from gabion.refactor.model import FieldSpec, RefactorRequest

    return RefactorEngine, FieldSpec, RefactorRequest


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
