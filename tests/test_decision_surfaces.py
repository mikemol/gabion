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
    surfaces = da.analyze_decision_surfaces_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        transparent_decorators=None,
    )
    assert surfaces == ["mod.py:mod.f decision surface params: b (boundary)"]

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

    value_surfaces = da.analyze_value_encoded_decisions_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        transparent_decorators=None,
    )
    assert value_surfaces == []


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
    surfaces = da.analyze_value_encoded_decisions_repo(
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
    surfaces = da.analyze_decision_surfaces_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        transparent_decorators=None,
    )
    assert surfaces == ["mod.py:mod.f decision surface params: b (internal callers: 1)"]

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
        "Consider contextvar for mod.py:mod.f decision surface params: b (internal callers: 1)"
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
