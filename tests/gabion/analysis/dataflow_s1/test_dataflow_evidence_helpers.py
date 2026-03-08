from __future__ import annotations

import ast
from pathlib import Path

from gabion.analysis.aspf.aspf import Alt, Forest, NodeId
from gabion.analysis.dataflow.engine.dataflow_contracts import ClassInfo, FunctionInfo, SymbolTable
from gabion.analysis.dataflow.engine import dataflow_evidence_helpers as helpers


def _fn(*, qual: str, path: Path) -> FunctionInfo:
    return FunctionInfo(
        name=qual.split(".")[-1],
        qual=qual,
        path=path,
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        function_span=(0, 0, 0, 1),
    )


# gabion:evidence E:function_site::dataflow_evidence_helpers.py::gabion.analysis.dataflow_evidence_helpers._is_test_path E:function_site::dataflow_evidence_helpers.py::gabion.analysis.dataflow_evidence_helpers._module_name
# gabion:behavior primary=verboten facets=edge
def test_test_path_and_module_name_contract_edges(tmp_path: Path) -> None:
    assert helpers._is_test_path(tmp_path / "tests" / "x.py")
    assert helpers._is_test_path(tmp_path / "pkg" / "test_x.py")
    assert not helpers._is_test_path(tmp_path / "pkg" / "x.py")

    rooted = helpers._module_name(tmp_path / "src" / "pkg" / "mod.py", project_root=tmp_path)
    assert rooted == "pkg.mod"

    external = helpers._module_name(Path("/outside/repo/file.py"), project_root=tmp_path)
    assert external == "/.outside.repo.file"


# gabion:evidence E:function_site::dataflow_evidence_helpers.py::gabion.analysis.dataflow_evidence_helpers._enclosing_scopes E:function_site::dataflow_evidence_helpers.py::gabion.analysis.dataflow_evidence_helpers._enclosing_class_scopes
# gabion:behavior primary=desired
def test_enclosing_scope_helpers_cover_class_and_async_function() -> None:
    tree = ast.parse(
        "class Outer:\n"
        "    async def afn(self):\n"
        "        class Local:\n"
        "            pass\n"
        "        return Local\n"
    )
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    local_class = next(node for node in ast.walk(tree) if type(node) is ast.ClassDef and node.name == "Local")
    assert helpers._enclosing_scopes(local_class, parents) == ["Outer", "afn"]
    assert helpers._enclosing_class_scopes(local_class, parents) == ["Outer"]


# gabion:evidence E:function_site::dataflow_evidence_helpers.py::gabion.analysis.dataflow_evidence_helpers._string_list E:function_site::dataflow_evidence_helpers.py::gabion.analysis.dataflow_evidence_helpers._target_names
# gabion:behavior primary=desired
def test_string_and_target_helpers_cover_non_string_and_nested_targets() -> None:
    ok_list = ast.parse("['a', 'b']").body[0].value
    bad_tuple = ast.parse("('a', 1)").body[0].value
    assert helpers._string_list(ok_list) == ["a", "b"]
    assert helpers._string_list(bad_tuple) is None
    assert helpers._string_list(ast.parse("x + 1").body[0].value) is None

    tuple_target = ast.parse("a, [b, c] = value").body[0].targets[0]
    assert helpers._target_names(tuple_target) == {"a", "b", "c"}
    assert helpers._target_names(ast.parse("obj.attr = value").body[0].targets[0]) == set()


# gabion:evidence E:function_site::dataflow_evidence_helpers.py::gabion.analysis.dataflow_evidence_helpers._collect_module_exports
# gabion:behavior primary=desired
def test_collect_module_exports_handles_assign_annassign_and_augassign() -> None:
    tree = ast.parse(
        "from pkg import imported as imported_alias\n"
        "__all__ = ['a']\n"
        "__all__ += ['b']\n"
        "__all__: list[str] = ['c']\n"
        "public_name = 1\n"
        "_private_name = 2\n"
        "def exported_func():\n"
        "    return 1\n"
    )
    export_names, export_map = helpers._collect_module_exports(
        tree,
        module_name="pkg.mod",
        import_map={
            "imported_alias": "pkg.imported",
            "_hidden": "pkg.hidden",
        },
    )
    assert export_names == {"c"}
    assert export_map == {}


# gabion:evidence E:function_site::dataflow_evidence_helpers.py::gabion.analysis.dataflow_evidence_helpers._collect_module_exports
# gabion:behavior primary=desired
def test_collect_module_exports_without_all_uses_locals_and_imports() -> None:
    tree = ast.parse(
        "imported = 1\n"
        "_skip = 2\n"
        "class C:\n"
        "    pass\n"
        "def fn():\n"
        "    return 1\n"
    )
    export_names, export_map = helpers._collect_module_exports(
        tree,
        module_name="pkg.mod",
        import_map={"imported_alias": "pkg.alias", "_hidden_alias": "pkg.hidden"},
    )
    assert export_names == {"imported", "C", "fn", "imported_alias"}
    assert export_map == {
        "imported": "pkg.mod.imported",
        "C": "pkg.mod.C",
        "fn": "pkg.mod.fn",
        "imported_alias": "pkg.alias",
    }


# gabion:evidence E:function_site::dataflow_evidence_helpers.py::gabion.analysis.dataflow_evidence_helpers._base_identifier
# gabion:behavior primary=desired
def test_base_identifier_variants_and_unparse_failure() -> None:
    assert helpers._base_identifier(ast.parse("name").body[0].value) == "name"
    assert helpers._base_identifier(ast.parse("pkg.Type").body[0].value) == "pkg.Type"
    assert helpers._base_identifier(ast.parse("Type[int]").body[0].value) == "Type"
    assert helpers._base_identifier(ast.parse("factory()").body[0].value) == "factory"
    assert helpers._base_identifier(ast.parse("123").body[0].value) is None

    bad_attr = ast.Attribute(value=ast.Constant(value=1), attr=None, ctx=ast.Load())
    assert helpers._base_identifier(bad_attr) is None


# gabion:evidence E:function_site::dataflow_evidence_helpers.py::gabion.analysis.dataflow_evidence_helpers._build_symbol_table
# gabion:behavior primary=desired
def test_build_symbol_table_handles_parse_failures_and_exports(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    pkg = src / "pkg"
    pkg.mkdir()
    valid = pkg / "mod.py"
    valid.write_text(
        "from pkg.other import Thing as Alias\n"
        "__all__ = ['Alias']\n"
        "public_name = 1\n"
    )
    invalid = pkg / "broken.py"
    invalid.write_text("def broken(:\n")

    parse_failure_witnesses: list[dict[str, object]] = []
    symbol_table = helpers._build_symbol_table(
        [valid, invalid],
        tmp_path,
        external_filter=True,
        parse_failure_witnesses=parse_failure_witnesses,
    )
    assert "pkg" in symbol_table.internal_roots
    assert symbol_table.module_exports["pkg.mod"] == {"Alias"}
    assert symbol_table.module_export_map["pkg.mod"] == {"Alias": "pkg.other.Thing"}
    assert any(witness.get("stage") == "symbol_table" for witness in parse_failure_witnesses)


# gabion:evidence E:function_site::dataflow_evidence_helpers.py::gabion.analysis.dataflow_evidence_helpers._collect_class_index
# gabion:behavior primary=desired
def test_collect_class_index_collects_nested_classes_and_skips_parse_failures(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    pkg = src / "pkg"
    pkg.mkdir()
    valid = pkg / "classes.py"
    valid.write_text(
        "class Base:\n"
        "    def run(self):\n"
        "        return 1\n"
        "class Outer:\n"
        "    class Inner(Base):\n"
        "        def run(self):\n"
        "            return 2\n"
    )
    invalid = pkg / "bad.py"
    invalid.write_text("class Missing(\n")

    parse_failure_witnesses: list[dict[str, object]] = []
    class_index = helpers._collect_class_index(
        [valid, invalid],
        tmp_path,
        parse_failure_witnesses=parse_failure_witnesses,
    )
    assert "pkg.classes.Base" in class_index
    assert "pkg.classes.Outer" in class_index
    assert "pkg.classes.Outer.Inner" in class_index
    assert class_index["pkg.classes.Outer.Inner"].bases == ["Base"]
    assert class_index["pkg.classes.Outer.Inner"].methods == {"run"}
    assert any(witness.get("stage") == "class_index" for witness in parse_failure_witnesses)


# gabion:evidence E:function_site::dataflow_evidence_helpers.py::gabion.analysis.dataflow_evidence_helpers._callee_key E:function_site::dataflow_evidence_helpers.py::gabion.analysis.dataflow_evidence_helpers._resolve_class_candidates
# gabion:behavior primary=verboten facets=empty
def test_callee_key_and_resolve_class_candidates_cover_empty_and_star_paths() -> None:
    assert helpers._callee_key("") == ""
    assert helpers._callee_key("a.b.C") == "C"

    table = SymbolTable()
    table.imports[("pkg.mod", "Alias")] = "pkg.base.Alias"
    table.star_imports["pkg.mod"] = {"pkg.star"}
    table.module_exports["pkg.star"] = {"FromStar"}
    table.module_export_map["pkg.star"] = {"FromStar": "pkg.star.FromStar"}
    table.internal_roots.add("pkg")

    class_index = {
        "pkg.base.Alias.Inner": ClassInfo(
            qual="pkg.base.Alias.Inner",
            module="pkg.base",
            bases=[],
            methods=set(),
        ),
        "pkg.mod.Alias": ClassInfo(qual="pkg.mod.Alias", module="pkg.mod", bases=[], methods=set()),
        "pkg.star.FromStar": ClassInfo(
            qual="pkg.star.FromStar",
            module="pkg.star",
            bases=[],
            methods=set(),
        ),
    }

    assert helpers._resolve_class_candidates(
        "",
        module="pkg.mod",
        symbol_table=table,
        class_index=class_index,
    ) == []
    assert helpers._resolve_class_candidates(
        "Alias.Inner",
        module="pkg.mod",
        symbol_table=table,
        class_index=class_index,
    ) == ["pkg.base.Alias.Inner"]
    assert helpers._resolve_class_candidates(
        "FromStar",
        module="pkg.mod",
        symbol_table=table,
        class_index=class_index,
    ) == ["pkg.star.FromStar"]
    assert helpers._resolve_class_candidates(
        "Alias",
        module="pkg.mod",
        symbol_table=None,
        class_index=class_index,
    ) == ["pkg.mod.Alias"]


# gabion:evidence E:function_site::dataflow_evidence_helpers.py::gabion.analysis.dataflow_evidence_helpers._resolve_method_in_hierarchy_outcome E:function_site::dataflow_evidence_helpers.py::gabion.analysis.dataflow_evidence_helpers._resolve_method_in_hierarchy
# gabion:behavior primary=verboten facets=missing
def test_resolve_method_hierarchy_outcome_found_missing_and_seen_cycle() -> None:
    caller_path = Path("pkg/mod.py")
    found = _fn(qual="pkg.Base.run", path=caller_path)
    class_index = {
        "pkg.Base": ClassInfo(qual="pkg.Base", module="pkg", bases=[], methods={"run"}),
        "pkg.Child": ClassInfo(qual="pkg.Child", module="pkg", bases=["Base"], methods=set()),
        "pkg.Loop": ClassInfo(qual="pkg.Loop", module="pkg", bases=["Loop"], methods=set()),
    }
    by_qual = {"pkg.Base.run": found}
    symbol_table = SymbolTable()
    symbol_table.imports[("pkg", "Base")] = "pkg.Base"
    symbol_table.imports[("pkg", "Loop")] = "pkg.Loop"
    symbol_table.internal_roots.add("pkg")

    seen_resolution = helpers._resolve_method_in_hierarchy_outcome(
        "pkg.Loop",
        "run",
        class_index=class_index,
        by_qual=by_qual,
        symbol_table=symbol_table,
        seen={"pkg.Loop"},
    )
    assert seen_resolution.kind == "not_found"

    missing_resolution = helpers._resolve_method_in_hierarchy_outcome(
        "pkg.Child",
        "missing",
        class_index=class_index,
        by_qual=by_qual,
        symbol_table=symbol_table,
        seen=set(),
    )
    assert missing_resolution.kind == "not_found"

    found_resolution = helpers._resolve_method_in_hierarchy(
        "pkg.Child",
        "run",
        class_index=class_index,
        by_qual=by_qual,
        symbol_table=symbol_table,
        seen=set(),
    )
    assert found_resolution.kind == "found"
    assert found_resolution.resolved is found


# gabion:evidence E:function_site::dataflow_evidence_helpers.py::gabion.analysis.dataflow_evidence_helpers._alt_input E:function_site::dataflow_evidence_helpers.py::gabion.analysis.dataflow_evidence_helpers._paramset_key
# gabion:behavior primary=desired
def test_alt_input_and_paramset_key_cover_fallbacks() -> None:
    left = NodeId("Left", ("a",))
    right = NodeId("Right", ("b",))
    alt = Alt(kind="Edge", inputs=(left, right))
    assert helpers._alt_input(alt, "Right") == right
    assert helpers._alt_input(alt, "Missing") is None

    forest = Forest()
    paramset = forest.add_paramset(["x", "y"])
    assert helpers._paramset_key(forest, paramset) == ("x", "y")
    assert helpers._paramset_key(forest, NodeId("ParamSet", ("z",))) == ("z",)
