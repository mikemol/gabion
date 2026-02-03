from __future__ import annotations

from pathlib import Path
import sys
import textwrap


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis.dataflow_audit import AuditConfig, analyze_paths, build_refactor_plan
    from gabion.refactor.engine import RefactorEngine
    from gabion.refactor.model import FieldSpec, RefactorRequest

    return AuditConfig, analyze_paths, build_refactor_plan, RefactorEngine, FieldSpec, RefactorRequest


def _apply_plan(plan) -> None:
    for edit in plan.edits:
        Path(edit.path).write_text(edit.replacement)


def test_refactor_idempotency(tmp_path: Path) -> None:
    (
        AuditConfig,
        analyze_paths,
        build_refactor_plan,
        RefactorEngine,
        FieldSpec,
        RefactorRequest,
    ) = _load()
    source = textwrap.dedent(
        """
        def sink(x=None, y=None):
            return x, y

        def g(a, b):
            sink(x=a, y=a)
            sink(x=b, y=b)

        def f(a, b):
            return g(a, b)
        """
    )
    file_path = tmp_path / "mod.py"
    file_path.write_text(source.strip() + "\n")
    config = AuditConfig(project_root=tmp_path)
    analysis = analyze_paths(
        [file_path],
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        config=config,
    )
    plan = build_refactor_plan(analysis.groups_by_path, [file_path], config=config)
    bundles = plan.get("bundles", [])
    assert bundles
    bundle = bundles[0].get("bundle", [])
    assert bundle

    request = RefactorRequest(
        protocol_name="BundleProtocol",
        bundle=list(bundle),
        fields=[FieldSpec(name=name, type_hint="int") for name in bundle],
        target_path=str(file_path),
        target_functions=["g"],
        rationale="Idempotency test",
    )
    refactor_plan = RefactorEngine(project_root=tmp_path).plan_protocol_extraction(request)
    assert refactor_plan.edits
    _apply_plan(refactor_plan)

    analysis_after = analyze_paths(
        [file_path],
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        config=config,
    )
    plan_after = build_refactor_plan(analysis_after.groups_by_path, [file_path], config=config)
    assert not plan_after.get("bundles", [])
