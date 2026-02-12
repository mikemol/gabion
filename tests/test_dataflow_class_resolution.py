from __future__ import annotations

from pathlib import Path
import sys
import ast


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit as da

    return da


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_class_candidates::base,class_index,module,symbol_table
def test_resolve_class_candidates_variants() -> None:
    da = _load()
    symbol_table = da.SymbolTable()
    symbol_table.internal_roots.add("pkg")
    symbol_table.imports[("pkg.mod", "ext")] = "pkg.ext"
    class_index = {
        "pkg.ext.Base": da.ClassInfo(qual="pkg.ext.Base", module="pkg.ext", bases=[], methods=set()),
        "pkg.mod.Base": da.ClassInfo(qual="pkg.mod.Base", module="pkg.mod", bases=[], methods=set()),
    }
    dotted = da._resolve_class_candidates(
        "ext.Base",
        module="pkg.mod",
        symbol_table=symbol_table,
        class_index=class_index,
    )
    assert "pkg.ext.Base" in dotted

    symbol_table.imports[("pkg.mod", "Base")] = "pkg.mod.Base"
    symbol_table.star_imports["pkg.mod"] = {"pkg.mod"}
    symbol_table.module_exports["pkg.mod"] = {"Base"}
    bare = da._resolve_class_candidates(
        "Base",
        module="pkg.mod",
        symbol_table=symbol_table,
        class_index=class_index,
    )
    assert "pkg.mod.Base" in bare


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_class_candidates::base,class_index,module,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_method_in_hierarchy::by_qual,class_qual,seen
def test_resolve_method_in_hierarchy() -> None:
    da = _load()
    base_info = da.FunctionInfo(
        name="run",
        qual="pkg.Base.run",
        path=Path("base.py"),
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        function_span=(0, 0, 0, 1),
    )
    class_index = {
        "pkg.Base": da.ClassInfo(qual="pkg.Base", module="pkg", bases=[], methods={"run"}),
        "pkg.Child": da.ClassInfo(qual="pkg.Child", module="pkg", bases=["Base"], methods=set()),
    }
    by_qual = {"pkg.Base.run": base_info}
    symbol_table = da.SymbolTable()
    symbol_table.imports[("pkg", "Base")] = "pkg.Base"
    symbol_table.internal_roots.add("pkg")
    resolved = da._resolve_method_in_hierarchy(
        "pkg.Child",
        "run",
        class_index=class_index,
        by_qual=by_qual,
        symbol_table=symbol_table,
        seen=set(),
    )
    assert resolved is base_info


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._local_class_name::base,class_bases E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_local_method_in_hierarchy::class_name,local_functions,seen E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._base_identifier::node
def test_local_class_bases_and_method_resolution() -> None:
    da = _load()
    tree = ast.parse(
        "class Base:\n"
        "    def run(self):\n"
        "        return 1\n"
        "\n"
        "class Child(Base):\n"
        "    def other(self):\n"
        "        return 2\n"
    )
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    class_bases = da._collect_local_class_bases(tree, parents)
    assert class_bases["Base"] == []
    assert class_bases["Child"] == ["Base"]
    resolved = da._resolve_local_method_in_hierarchy(
        "Child",
        "run",
        class_bases=class_bases,
        local_functions={"Base.run"},
        seen=set(),
    )
    assert resolved == "Base.run"
