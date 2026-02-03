from __future__ import annotations

from pathlib import Path
import sys
import textwrap


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit as da

    return da


def test_flow_analyses_edge_cases(tmp_path: Path) -> None:
    da = _load()
    module_path = tmp_path / "mod.py"
    module_path.write_text(
        textwrap.dedent(
            """
            def short(a):
                return a

            def short_unused(a, b):
                return a

            def typed_int(x: int):
                return x

            def typed_str(x: str):
                return x

            def typed_only(u):
                return typed_int(u)

            def const_only(a):
                return a

            def with_default(a, b=1):
                return a

            def star_callee(a, b, c):
                return a

            def bundle_caller(p, q):
                short(p)
                short(q)

            def caller(x, y, *args, **kwargs):
                short(1, x)
                short(1, 2)
                short(x, x + 1)
                short(a=1, z=2)
                short(a=x, z=y)
                short(z=x + 1)
                short(*args, **kwargs)
                short_unused(x, y)
                short_unused(x, b=2)
                typed_int(x)
                typed_str(x)
                const_only(1)
                with_default(x)
                star_callee(*args, **kwargs)
            """
        ).strip()
        + "\n"
    )
    test_path = tmp_path / "tests" / "test_mod.py"
    test_path.parent.mkdir(parents=True)
    test_path.write_text(
        "def test_dummy(a):\n"
        "    short(a)\n"
    )
    paths = [module_path, test_path]
    config = da.AuditConfig(
        project_root=tmp_path,
        exclude_dirs=set(),
        ignore_params=set(),
        external_filter=True,
        strictness="low",
        transparent_decorators=None,
    )
    analysis = da.analyze_paths(
        paths,
        recursive=True,
        type_audit=True,
        type_audit_report=True,
        type_audit_max=5,
        include_constant_smells=True,
        include_unused_arg_smells=True,
        config=config,
    )
    assert analysis.type_ambiguities
    assert analysis.constant_smells is not None
    assert analysis.unused_arg_smells is not None

    plan = da.build_synthesis_plan(
        analysis.groups_by_path,
        project_root=tmp_path,
        max_tier=2,
        min_bundle_size=2,
        allow_singletons=False,
        merge_overlap_threshold=0.5,
        config=config,
    )
    assert "protocols" in plan

    suggestions, ambiguities = da.analyze_type_flow_repo(
        paths,
        project_root=tmp_path,
        ignore_params=set(),
        strictness="low",
        external_filter=True,
        transparent_decorators=None,
    )
    assert suggestions
    assert ambiguities

    const_smells = da.analyze_constant_flow_repo(
        paths,
        project_root=tmp_path,
        ignore_params=set(),
        strictness="low",
        external_filter=True,
        transparent_decorators=None,
    )
    assert const_smells

    unused_smells = da.analyze_unused_arg_flow_repo(
        paths,
        project_root=tmp_path,
        ignore_params=set(),
        strictness="low",
        external_filter=True,
        transparent_decorators=None,
    )
    assert unused_smells


def test_analyze_paths_config_default(tmp_path: Path) -> None:
    da = _load()
    module_path = tmp_path / "mod.py"
    module_path.write_text("def f(a, b):\n    return a\n")
    analysis = da.analyze_paths(
        [module_path],
        recursive=False,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        config=None,
    )
    assert analysis.groups_by_path
