from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis.dataflow_audit import (
        AuditConfig,
        analyze_constant_flow_repo,
        analyze_deadness_flow_repo,
        analyze_paths,
    )

    return AuditConfig, analyze_constant_flow_repo, analyze_deadness_flow_repo, analyze_paths


def test_constant_flow_smells_and_star_paths(tmp_path: Path) -> None:
    AuditConfig, _, _, analyze_paths = _load()
    code = (
        "def target(a, b, c):\n"
        "    return a\n"
        "\n"
        "def caller(x, *args, **kwargs):\n"
        "    target(1, b=2, c=3)\n"
        "    target(*args, **kwargs)\n"
    )
    path = tmp_path / "mod.py"
    path.write_text(code)
    config = AuditConfig(
        project_root=tmp_path,
        exclude_dirs=set(),
        ignore_params=set(),
        external_filter=False,
        strictness="low",
        transparent_decorators=None,
    )
    analysis = analyze_paths(
        [path],
        recursive=False,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=5,
        include_constant_smells=True,
        include_unused_arg_smells=False,
        config=config,
    )
    assert isinstance(analysis.constant_smells, list)


def test_constant_flow_detects_constant_kw_and_ignores_non_const(tmp_path: Path) -> None:
    _, analyze_constant_flow_repo, _, _ = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "def callee(a, b):\n"
        "    return a\n"
        "\n"
        "def caller(x):\n"
        "    return callee(a=1, b=x)\n"
    )
    smells = analyze_constant_flow_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
    )
    assert any("callee.a only observed constant 1" in smell for smell in smells)


def test_constant_flow_skips_test_paths(tmp_path: Path) -> None:
    _, analyze_constant_flow_repo, _, _ = _load()
    path = tmp_path / "tests" / "test_mod.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "def callee(a):\n"
        "    return a\n"
        "\n"
        "def caller():\n"
        "    return callee(1)\n"
    )
    smells = analyze_constant_flow_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
    )
    assert smells == []


def test_constant_flow_low_strictness_star_handling(tmp_path: Path) -> None:
    _, analyze_constant_flow_repo, _, _ = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "def callee(a, b):\n"
        "    return a\n"
        "\n"
        "def caller(*args, **kwargs):\n"
        "    return callee(*args, **kwargs)\n"
    )
    smells = analyze_constant_flow_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="low",
        external_filter=True,
    )
    assert smells == []


def test_constant_flow_ignores_extra_pos_args(tmp_path: Path) -> None:
    _, analyze_constant_flow_repo, _, _ = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "def callee(a):\n"
        "    return a\n"
        "\n"
        "def caller(x, y):\n"
        "    return callee(x, y)\n"
    )
    smells = analyze_constant_flow_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
    )
    assert smells == []


def test_constant_flow_tracks_non_const_kw(tmp_path: Path) -> None:
    _, analyze_constant_flow_repo, _, _ = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "def callee(a, b):\n"
        "    return b\n"
        "\n"
        "def caller(x):\n"
        "    return callee(a=1, b=x + 1)\n"
    )
    smells = analyze_constant_flow_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
    )
    assert any("callee.a only observed constant 1" in smell for smell in smells)


def test_deadness_witnesses_from_constant_flow(tmp_path: Path) -> None:
    _, _, analyze_deadness_flow_repo, _ = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "def callee(a):\n"
        "    return a\n"
        "\n"
        "def caller():\n"
        "    return callee(1)\n"
    )
    witnesses = analyze_deadness_flow_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
    )
    assert witnesses
    entry = witnesses[0]
    assert entry["path"].endswith("mod.py")
    assert entry["function"] == "callee"
    assert entry["bundle"] == ["a"]
    assert entry["environment"] == {"a": "1"}
    assert entry["result"] == "UNREACHABLE"
