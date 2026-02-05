from __future__ import annotations

import ast
from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit as da

    return da


def test_decision_surface_params_collects_names() -> None:
    da = _load()
    tree = ast.parse(
        "def f(a, b, cfg):\n"
        "    if a > 0:\n"
        "        return b\n"
        "    while cfg.flag:\n"
        "        break\n"
        "    assert b\n"
    )
    fn = tree.body[0]
    params = da._decision_surface_params(fn, ignore_params=None)
    assert params == {"a", "b", "cfg"}


def test_value_encoded_decision_params_collects_names() -> None:
    da = _load()
    tree = ast.parse(
        "def f(a, b, mask, value):\n"
        "    x = min(a, b)\n"
        "    y = value * (a > 0)\n"
        "    z = mask & 1\n"
        "    return x + y + z\n"
    )
    fn = tree.body[0]
    params, reasons = da._value_encoded_decision_params(fn, ignore_params=None)
    assert params == {"a", "b", "mask"}
    assert reasons == {"min/max", "boolean arithmetic", "bitmask"}


def test_analyze_decision_surfaces_repo(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "def f(a, b):\n"
        "    if b:\n"
        "        return a\n"
        "    return b\n"
    )
    surfaces, warnings, lint_lines = da.analyze_decision_surfaces_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        transparent_decorators=None,
    )
    assert surfaces == ["mod.py:mod.f decision surface params: b (boundary)"]
    assert warnings == []
    assert any("GABION_DECISION_SURFACE" in line for line in lint_lines)

    analysis = da.analyze_paths(
        [path],
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_decision_surfaces=True,
        config=da.AuditConfig(project_root=tmp_path),
    )
    assert analysis.decision_surfaces == surfaces
    assert analysis.decision_warnings == warnings

    (
        value_surfaces,
        value_warnings,
        value_rewrites,
        value_lint_lines,
    ) = da.analyze_value_encoded_decisions_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        transparent_decorators=None,
    )
    assert value_surfaces == []
    assert value_warnings == []
    assert value_rewrites == []
    assert value_lint_lines == []


def test_analyze_value_encoded_decisions_repo(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "def f(a, b, mask, value):\n"
        "    x = min(a, b)\n"
        "    y = value * (a > 0)\n"
        "    z = mask & 1\n"
        "    return x + y + z\n"
    )
    surfaces, warnings, rewrites, lint_lines = da.analyze_value_encoded_decisions_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        transparent_decorators=None,
    )
    assert surfaces == [
        "mod.py:mod.f value-encoded decision params: a, b, mask (bitmask, boolean arithmetic, min/max)"
    ]
    assert warnings == []
    assert rewrites == [
        "mod.py:mod.f consider rebranching value-encoded decision params: a, b, mask (bitmask, boolean arithmetic, min/max)"
    ]
    assert any("GABION_VALUE_DECISION_SURFACE" in line for line in lint_lines)


def test_emit_report_includes_value_rewrites(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text("def f(a):\n    return a\n")
    groups_by_path = {path: {"f": [set(["a", "b"])]}}
    report, _ = da._emit_report(
        groups_by_path,
        1,
        value_decision_rewrites=["mod.py:f consider rebranching value-encoded decision params: a (min/max)"],
    )
    assert "Value-encoded decision rebranch suggestions" in report


def test_decision_surface_internal_caller(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "def f(a, b):\n"
        "    if b:\n"
        "        return a\n"
        "    return b\n"
        "\n"
        "def g(x, y):\n"
        "    return f(x, y)\n"
    )
    surfaces, warnings, lint_lines = da.analyze_decision_surfaces_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        transparent_decorators=None,
    )
    assert surfaces == [
        "mod.py:mod.f decision surface params: b (internal callers (transitive): 1)"
    ]
    assert warnings == []
    assert lint_lines == []

    analysis = da.analyze_paths(
        [path],
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_decision_surfaces=True,
        config=da.AuditConfig(project_root=tmp_path),
    )
    assert analysis.context_suggestions == [
        "Consider contextvar for mod.py:mod.f decision surface params: b (internal callers (transitive): 1)"
    ]


def test_emit_report_includes_decision_surfaces(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text("def f(a):\n    return a\n")
    groups_by_path = {path: {"f": [set(["a", "b"])]}}
    report, _ = da._emit_report(
        groups_by_path,
        1,
        decision_surfaces=["mod.py:f decision surface params: a"],
    )
    assert "Decision surface candidates" in report


def test_decision_surface_tier_warning_internal(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "def internal(user_mode):\n"
        "    if user_mode:\n"
        "        return 1\n"
        "    return 0\n"
        "\n"
        "def api(user_mode):\n"
        "    return internal(user_mode)\n"
    )
    _, warnings, lint_lines = da.analyze_decision_surfaces_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        transparent_decorators=None,
        decision_tiers={"user_mode": 3},
    )
    assert any("tier-3 decision param 'user_mode'" in warning for warning in warnings)
    assert any("GABION_DECISION_TIER" in line for line in lint_lines)


def test_decision_surface_location_tier_suppresses_lint(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "def f(a):\n"
        "    if a:\n"
        "        return 1\n"
        "    return 0\n"
    )
    surfaces, warnings, lint_lines = da.analyze_decision_surfaces_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        transparent_decorators=None,
        decision_tiers={"mod.py:1:7": 1},
    )
    assert surfaces
    assert warnings == []
    assert not any("GABION_DECISION_SURFACE" in line for line in lint_lines)
