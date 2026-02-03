from __future__ import annotations

import ast
from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit as da

    return da


def test_symbol_table_resolve_star_external_filtered() -> None:
    da = _load()
    table = da.SymbolTable(external_filter=True)
    table.star_imports["pkg.mod"] = {"ext.star"}
    table.module_exports["ext.star"] = {"Foo"}
    table.module_export_map["ext.star"] = {"Foo": "ext.star.Foo"}
    assert table.resolve_star("pkg.mod", "Foo") is None


def test_normalize_callee_long_self_chain() -> None:
    da = _load()
    assert da._normalize_callee("self.a.b", "Service") == "self.a.b"


def test_iter_paths_ignored_file(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "skip.py"
    path.write_text("x = 1\n")
    config = da.AuditConfig(exclude_dirs={"skip.py"})
    paths = da._iter_paths([str(path)], config)
    assert paths == []


def test_decorator_name_attribute_without_root() -> None:
    da = _load()
    node = ast.Attribute(value=ast.Constant(value=1), attr="decor")
    assert da._decorator_name(node) is None


def test_normalize_transparent_decorators_variants() -> None:
    da = _load()
    assert da._normalize_transparent_decorators(None) is None
    assert da._normalize_transparent_decorators("a, b") == {"a", "b"}
    assert da._normalize_transparent_decorators(["a", "b, c"]) == {"a", "b", "c"}
    assert da._normalize_transparent_decorators([1, "a"]) == {"a"}
    assert da._normalize_transparent_decorators([]) is None
