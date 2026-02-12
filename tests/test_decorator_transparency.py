from __future__ import annotations

from pathlib import Path
import sys
import textwrap
from gabion.analysis.aspf import Forest


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis.dataflow_audit import analyze_paths, AuditConfig

    return analyze_paths, AuditConfig


def _analyze(source: str, tmp_path: Path, *, transparent: set[str] | None = None):
    analyze_paths, AuditConfig = _load()
    file_path = tmp_path / "mod.py"
    file_path.write_text(source)
    config = AuditConfig(project_root=tmp_path, transparent_decorators=transparent)
    analysis = analyze_paths(
        forest=Forest(),
        paths=[file_path],
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        config=config,
    )
    return analysis.groups_by_path[file_path]


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report
def test_decorated_function_transparent_by_default(tmp_path: Path) -> None:
    source = textwrap.dedent(
        """
        def sink(x=None, y=None):
            return x, y

        def deco(fn):
            return fn

        @deco
        def g(a, b):
            sink(x=a, y=a)
            sink(x=b, y=b)

        def f(a, b):
            return g(a, b)
        """
    ).lstrip()
    groups = _analyze(source, tmp_path)
    assert "g" in groups
    assert "f" in groups
    assert {"a", "b"} in groups["g"]
    assert {"a", "b"} in groups["f"]


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report
def test_decorated_function_opaque_without_allowlist(tmp_path: Path) -> None:
    source = textwrap.dedent(
        """
        def sink(x=None, y=None):
            return x, y

        def deco(fn):
            return fn

        @deco
        def g(a, b):
            sink(x=a, y=a)
            sink(x=b, y=b)

        def f(a, b):
            return g(a, b)
        """
    ).lstrip()
    groups = _analyze(source, tmp_path, transparent={"other"})
    assert "g" in groups
    assert "f" in groups
    assert {"a", "b"} in groups["g"]
    assert {"a", "b"} not in groups["f"]
