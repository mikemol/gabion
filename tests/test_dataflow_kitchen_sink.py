from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis.dataflow_audit import (
        AuditConfig,
        analyze_paths,
        build_refactor_plan,
        build_synthesis_plan,
        render_dot,
        render_protocol_stubs,
        render_refactor_plan,
        render_report,
    )

    return (
        AuditConfig,
        analyze_paths,
        build_refactor_plan,
        build_synthesis_plan,
        render_dot,
        render_protocol_stubs,
        render_refactor_plan,
        render_report,
    )


def _write_kitchen_sink(root: Path) -> None:
    pkg = root / "pkg"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text(
        "from .models import Config, UserConfig\n"
        "__all__ = [\"Config\", \"UserConfig\"]\n"
    )
    (pkg / "models.py").write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "from typing import Optional\n"
        "from .utils import helper\n"
        "\n"
        "@dataclass\n"
        "class Config:\n"
        "    name: str\n"
        "    count: int = 0\n"
        "\n"
        "@dataclass\n"
        "class UserConfig:\n"
        "    cfg: Config\n"
        "    tag: Optional[str] = None\n"
        "\n"
        "class Service:\n"
        "    def __init__(self, cfg: Config):\n"
        "        self.cfg = cfg\n"
        "\n"
        "    def run(self, a, b, *args, **kwargs):\n"
        "        return helper(a, b, *args, **kwargs)\n"
        "\n"
        "    @classmethod\n"
        "    def build(cls, cfg: Config):\n"
        "        return cls(cfg)\n"
        "\n"
        "    @staticmethod\n"
        "    def ping(x):\n"
        "        return x\n"
        "\n"
        "    @property\n"
        "    def value(self):\n"
        "        return self.cfg.count\n"
    )
    (pkg / "utils.py").write_text(
        "from __future__ import annotations\n"
        "\n"
        "def deco(fn):\n"
        "    def wrapper(*args, **kwargs):\n"
        "        return fn(*args, **kwargs)\n"
        "    return wrapper\n"
        "\n"
        "@deco\n"
        "def helper(a, b, *args, **kwargs):\n"
        "    # dataflow-bundle: a, b\n"
        "    c = a\n"
        "    d, e = (b, b)\n"
        "    data = {\"k\": c}\n"
        "    data[\"k\"] = d\n"
        "    return c, d\n"
        "\n"
        "def uses_kwargs(**kwargs):\n"
        "    return helper(**kwargs)\n"
    )
    (pkg / "calls.py").write_text(
        "from __future__ import annotations\n"
        "from .models import Config, Service\n"
        "\n"
        "def run(cfg: Config):\n"
        "    svc = Service.build(cfg)\n"
        "    return svc.run(cfg.name, cfg.count)\n"
    )
    (pkg / "nested.py").write_text(
        "def outer(x):\n"
        "    def inner(y):\n"
        "        return y\n"
        "    return inner(x)\n"
    )
    (pkg / "exports.py").write_text(
        "from .models import Config as Cfg, UserConfig\n"
        "__all__ = [\"Cfg\", \"UserConfig\"]\n"
    )
    (pkg / "star.py").write_text("from .exports import *\n")
    (root / "main.py").write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "from pkg.models import Config, UserConfig\n"
        "from pkg.calls import run\n"
        "\n"
        "@dataclass\n"
        "class LocalCfg:\n"
        "    value: int\n"
        "\n"
        "def entry():\n"
        "    cfg = Config(name=\"demo\", count=1)\n"
        "    user = UserConfig(cfg=cfg)\n"
        "    return run(cfg), user\n"
    )


def test_kitchen_sink_analysis_outputs(tmp_path: Path) -> None:
    (
        AuditConfig,
        analyze_paths,
        build_refactor_plan,
        build_synthesis_plan,
        render_dot,
        render_protocol_stubs,
        render_refactor_plan,
        render_report,
    ) = _load()
    _write_kitchen_sink(tmp_path)
    config = AuditConfig(
        project_root=tmp_path,
        exclude_dirs=set(),
        ignore_params=set(),
        external_filter=False,
        strictness="high",
        transparent_decorators={"deco"},
    )
    analysis = analyze_paths(
        [tmp_path],
        recursive=True,
        type_audit=True,
        type_audit_report=True,
        type_audit_max=5,
        include_constant_smells=True,
        include_unused_arg_smells=True,
        include_bundle_forest=True,
        config=config,
    )
    report, violations = render_report(
        analysis.groups_by_path,
        5,
        forest=analysis.forest,
        type_suggestions=analysis.type_suggestions,
        type_ambiguities=analysis.type_ambiguities,
        constant_smells=analysis.constant_smells,
        unused_arg_smells=analysis.unused_arg_smells,
    )
    assert "Dataflow grammar" in report or "dataflow-grammar" in report
    assert isinstance(violations, list)

    dot = render_dot(analysis.forest)
    assert "digraph" in dot

    plan = build_synthesis_plan(
        analysis.groups_by_path,
        project_root=tmp_path,
        max_tier=2,
        min_bundle_size=1,
        allow_singletons=True,
        config=config,
    )
    stubs = render_protocol_stubs(plan, kind="dataclass")
    assert "class" in stubs

    refactor_plan = build_refactor_plan(
        analysis.groups_by_path,
        [tmp_path],
        config=config,
    )
    refactor_summary = render_refactor_plan(refactor_plan)
    assert "Refactoring plan" in refactor_summary
