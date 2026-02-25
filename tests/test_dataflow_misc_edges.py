from __future__ import annotations

import ast
from pathlib import Path

def _load():
    repo_root = Path(__file__).resolve().parents[1]
    from gabion.analysis import dataflow_audit as da

    return da

# gabion:evidence E:function_site::test_dataflow_misc_edges.py::tests.test_dataflow_misc_edges._load
def test_symbol_table_resolve_star_external_filtered() -> None:
    da = _load()
    table = da.SymbolTable(external_filter=True)
    table.star_imports["pkg.mod"] = {"ext.star"}
    table.module_exports["ext.star"] = {"Foo"}
    table.module_export_map["ext.star"] = {"Foo": "ext.star.Foo"}
    assert table.resolve_star("pkg.mod", "Foo") is None

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_callee::class_name,name E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_callee::stale_7fcf0ff40be9
def test_normalize_callee_long_self_chain() -> None:
    da = _load()
    assert da._normalize_callee("self.a.b", "Service") == "self.a.b"

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::stale_78bc4b38f078
def test_iter_paths_ignored_file(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "skip.py"
    path.write_text("x = 1\n")
    config = da.AuditConfig(exclude_dirs={"skip.py"})
    paths = da._iter_paths([str(path)], config)
    assert paths == []

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._decorator_name::node E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._decorator_name::stale_b6d27785d676
def test_decorator_name_attribute_without_root() -> None:
    da = _load()
    node = ast.Attribute(value=ast.Constant(value=1), attr="decor")
    assert da._decorator_name(node) is None

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_transparent_decorators::value E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_transparent_decorators::stale_066524f398df
def test_normalize_transparent_decorators_variants() -> None:
    da = _load()
    assert da._normalize_transparent_decorators(None) is None
    assert da._normalize_transparent_decorators("a, b") == {"a", "b"}
    assert da._normalize_transparent_decorators(["a", "b, c"]) == {"a", "b", "c"}
    assert da._normalize_transparent_decorators([1, "a"]) == {"a"}
    assert da._normalize_transparent_decorators([]) is None

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::stale_da182920897e
def test_iter_paths_ignored_candidate_file(tmp_path: Path) -> None:
    da = _load()
    root = tmp_path / "root"
    root.mkdir()
    ignored = root / "ignored.py"
    ignored.write_text("x = 1\n")
    keep = root / "keep.py"
    keep.write_text("x = 2\n")
    config = da.AuditConfig(exclude_dirs={"ignored.py"})
    paths = da._iter_paths([str(root)], config)
    assert keep in paths
    assert ignored not in paths

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._string_list::node E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._string_list::stale_02a72fa14c85
def test_string_list_non_list_returns_none() -> None:
    da = _load()
    assert da._string_list(ast.Constant(value=1)) is None

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_module_exports::import_map,module_name E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._string_list::node E:decision_surface/value_encoded::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_module_exports::import_map E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_module_exports::stale_08ad1e4c65fd
def test_collect_module_exports_includes_annassign() -> None:
    da = _load()
    tree = ast.parse("x: int = 1\n")
    export_names, export_map = da._collect_module_exports(
        tree, module_name="mod", import_map={}
    )
    assert "x" in export_names
    assert export_map["x"] == "mod.x"

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_dataclass_call_bundles._resolve_fields::call E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_dataclass_call_bundles::dataclass_registry,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._module_name::project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_dataclass_call_bundles::stale_66c915cc58e0
def test_iter_dataclass_call_bundles_captures_assign_fields(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass\n"
        "class C:\n"
        "    a: int\n"
        "    b = 1\n"
        "\n"
        "def f():\n"
        "    return C(a=1, b=2)\n"
    )
    bundles = da._iter_dataclass_call_bundles(
        path,
        project_root=tmp_path,
        parse_failure_witnesses=[],
    )
    assert ("a", "b") in bundles
