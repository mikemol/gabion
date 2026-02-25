from __future__ import annotations

from pathlib import Path
import textwrap

import libcst as cst

def _load():
    repo_root = Path(__file__).resolve().parents[1]
    from gabion.refactor.engine import (
        RefactorEngine,
        _find_import_insert_index,
        _module_expr_to_str,
        _module_name,
        _has_typing_import,
    )
    from gabion.refactor.model import FieldSpec, RefactorRequest

    return (
        RefactorEngine,
        FieldSpec,
        RefactorRequest,
        _find_import_insert_index,
        _module_expr_to_str,
        _module_name,
        _has_typing_import,
    )

# gabion:evidence E:function_site::test_refactor_engine_edges.py::tests.test_refactor_engine_edges._load E:decision_surface/direct::test_refactor_engine_edges.py::tests.test_refactor_engine_edges._load::stale_f8de02c174ce_2adf8470
def test_refactor_engine_handles_missing_file(tmp_path: Path) -> None:
    RefactorEngine, _, RefactorRequest, *_ = _load()
    request = RefactorRequest(
        protocol_name="BundleProtocol",
        bundle=["a"],
        target_path=str(tmp_path / "missing.py"),
    )
    plan = RefactorEngine(project_root=tmp_path).plan_protocol_extraction(request)
    assert plan.errors

# gabion:evidence E:function_site::test_refactor_engine_edges.py::tests.test_refactor_engine_edges._load E:decision_surface/direct::test_refactor_engine_edges.py::tests.test_refactor_engine_edges._load::stale_5a02293adca9
def test_refactor_engine_handles_parse_error(tmp_path: Path) -> None:
    RefactorEngine, _, RefactorRequest, *_ = _load()
    target = tmp_path / "bad.py"
    target.write_text("def broken(")
    request = RefactorRequest(
        protocol_name="BundleProtocol",
        bundle=["a"],
        target_path=str(target),
    )
    plan = RefactorEngine(project_root=tmp_path).plan_protocol_extraction(request)
    assert plan.errors

# gabion:evidence E:function_site::test_refactor_engine_edges.py::tests.test_refactor_engine_edges._load E:decision_surface/direct::test_refactor_engine_edges.py::tests.test_refactor_engine_edges._load::stale_771c75339ecb
def test_refactor_engine_requires_protocol_name(tmp_path: Path) -> None:
    RefactorEngine, _, RefactorRequest, *_ = _load()
    target = tmp_path / "sample.py"
    target.write_text("def f(a):\n    return a\n")
    request = RefactorRequest(
        protocol_name="",
        bundle=["a"],
        target_path=str(target),
    )
    plan = RefactorEngine(project_root=tmp_path).plan_protocol_extraction(request)
    assert plan.errors

# gabion:evidence E:function_site::test_refactor_engine_edges.py::tests.test_refactor_engine_edges._load E:decision_surface/direct::test_refactor_engine_edges.py::tests.test_refactor_engine_edges._load::stale_c1015cdb03b3
def test_refactor_engine_requires_bundle_or_fields(tmp_path: Path) -> None:
    RefactorEngine, _, RefactorRequest, *_ = _load()
    target = tmp_path / "sample.py"
    target.write_text("def f(a):\n    return a\n")
    request = RefactorRequest(
        protocol_name="BundleProtocol",
        bundle=[],
        target_path=str(target),
    )
    plan = RefactorEngine(project_root=tmp_path).plan_protocol_extraction(request)
    assert plan.errors

# gabion:evidence E:function_site::test_refactor_engine_edges.py::tests.test_refactor_engine_edges._load E:decision_surface/direct::test_refactor_engine_edges.py::tests.test_refactor_engine_edges._load::stale_8dfdcde11c7c
def test_refactor_engine_invalid_type_hint_warns(tmp_path: Path) -> None:
    RefactorEngine, FieldSpec, RefactorRequest, *_ = _load()
    target = tmp_path / "sample.py"
    target.write_text("def f(a):\n    return a\n")
    request = RefactorRequest(
        protocol_name="BundleProtocol",
        bundle=["a"],
        fields=[FieldSpec(name="a", type_hint="list[")],
        target_path=str(target),
    )
    plan = RefactorEngine(project_root=tmp_path).plan_protocol_extraction(request)
    assert any("Failed to parse type hint" in warning for warning in plan.warnings)
    assert plan.edits

# gabion:evidence E:decision_surface/direct::engine.py::gabion.refactor.engine._module_name::project_root E:decision_surface/direct::engine.py::gabion.refactor.engine._module_name::stale_5a6c46cd6873_5cdcf300
def test_module_name_strips_src_prefix(tmp_path: Path) -> None:
    _, _, _, _, _, _module_name, _ = _load()
    path = Path("src") / "pkg" / "mod.py"
    assert _module_name(path, None) == "pkg.mod"
    rooted = tmp_path / "pkg" / "mod.py"
    assert _module_name(rooted, tmp_path) == "pkg.mod"

# gabion:evidence E:decision_surface/direct::engine.py::gabion.refactor.engine._find_import_insert_index::body E:decision_surface/direct::engine.py::gabion.refactor.engine._module_expr_to_str::expr E:decision_surface/direct::engine.py::gabion.refactor.engine._is_docstring::stmt E:decision_surface/direct::engine.py::gabion.refactor.engine._is_import::stmt E:decision_surface/direct::engine.py::gabion.refactor.engine._find_import_insert_index::stale_2234a356d297_c5c694a6
def test_module_expr_to_str_and_import_index() -> None:
    _, _, _, _find_import_insert_index, _module_expr_to_str, _, _ = _load()
    module = cst.parse_module(
        textwrap.dedent(
            '''
            """Docstring."""
            import os

            def f():
                return 1
            '''
        ).strip()
        + "\n"
    )
    assert _find_import_insert_index(list(module.body)) == 2
    expr = cst.parse_expression("pkg.mod.util")
    assert _module_expr_to_str(expr) == "pkg.mod.util"
    assert _module_expr_to_str(None) is None

# gabion:evidence E:decision_surface/direct::engine.py::gabion.refactor.engine._module_expr_to_str::expr E:decision_surface/direct::engine.py::gabion.refactor.engine._module_expr_to_str::stale_6eb418997c4a
def test_has_typing_import_handles_attribute_module() -> None:
    _, _, _, _, _, _, _has_typing_import = _load()
    module = cst.parse_module("import foo.typing\n")
    assert _has_typing_import(list(module.body)) is False
