from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit as da

    return da


def test_never_invariants_emit_forest_and_report(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "from gabion.invariants import never\n"
        "\n"
        "def f(a):\n"
        "    if a:\n"
        "        never('boom')\n"
    )
    config = da.AuditConfig(project_root=tmp_path)
    analysis = da.analyze_paths(
        [path],
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_never_invariants=True,
        include_bundle_forest=True,
        config=config,
    )
    assert analysis.never_invariants
    assert analysis.forest is not None
    assert any(
        alt.kind == "NeverInvariantSink" for alt in analysis.forest.alts
    )
    report, _ = da.render_report(
        analysis.groups_by_path,
        max_components=3,
        forest=analysis.forest,
        never_invariants=analysis.never_invariants,
    )
    assert "Never invariants" in report


def test_never_invariant_violation(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "from gabion.invariants import never\n"
        "\n"
        "def f(a):\n"
        "    if a == 1:\n"
        "        never('boom')\n"
        "\n"
        "def g():\n"
        "    f(1)\n"
    )
    config = da.AuditConfig(project_root=tmp_path)
    analysis = da.analyze_paths(
        [path],
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_deadness_witnesses=True,
        include_never_invariants=True,
        include_lint_lines=True,
        include_bundle_forest=True,
        config=config,
    )
    assert any(entry.get("status") == "VIOLATION" for entry in analysis.never_invariants)
    assert any("GABION_NEVER_INVARIANT" in line for line in analysis.lint_lines)


def test_never_invariant_proven_unreachable(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "from gabion.invariants import never\n"
        "\n"
        "def f(a):\n"
        "    if a == 0:\n"
        "        never('boom')\n"
        "\n"
        "def g():\n"
        "    f(1)\n"
    )
    config = da.AuditConfig(project_root=tmp_path)
    analysis = da.analyze_paths(
        [path],
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_deadness_witnesses=True,
        include_never_invariants=True,
        include_lint_lines=True,
        include_bundle_forest=True,
        config=config,
    )
    assert any(
        entry.get("status") == "PROVEN_UNREACHABLE" for entry in analysis.never_invariants
    )
    assert not any("GABION_NEVER_INVARIANT" in line for line in analysis.lint_lines)


def test_never_invariant_obligation(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "from gabion.invariants import never\n"
        "\n"
        "def f(a):\n"
        "    if a == 0:\n"
        "        never('boom')\n"
        "\n"
        "def g(x):\n"
        "    f(x)\n"
    )
    config = da.AuditConfig(project_root=tmp_path)
    analysis = da.analyze_paths(
        [path],
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_deadness_witnesses=True,
        include_never_invariants=True,
        include_lint_lines=True,
        include_bundle_forest=True,
        config=config,
    )
    assert any(
        entry.get("status") == "OBLIGATION" for entry in analysis.never_invariants
    )
    assert any("GABION_NEVER_INVARIANT" in line for line in analysis.lint_lines)
