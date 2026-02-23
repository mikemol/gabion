from __future__ import annotations

import ast
from pathlib import Path

def _load():
    repo_root = Path(__file__).resolve().parents[1]
    from gabion.analysis import dataflow_audit as da

    return da

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._const_repr::node
def test_const_repr_and_type_from_const_repr() -> None:
    da = _load()
    const_node = ast.parse("x = 1").body[0].value
    assert da._const_repr(const_node) == "1"
    unary_node = ast.parse("x = -1").body[0].value
    unary_repr = da._const_repr(unary_node)
    assert isinstance(unary_repr, str)
    assert unary_repr.replace("(", "").replace(")", "") == "-1"
    attr_node = ast.parse("MOD.CONST").body[0].value
    assert da._const_repr(attr_node) == "MOD.CONST"
    lower_attr = ast.parse("mod.const").body[0].value
    assert da._const_repr(lower_attr) is None
    class _BadRepr:
        def __repr__(self) -> str:
            raise ValueError("bad repr")

    weird_unary = ast.UnaryOp(op=ast.UAdd(), operand=ast.Constant(value=_BadRepr()))
    assert da._const_repr(weird_unary) is None
    weird_attr = ast.Attribute(
        value=ast.Constant(value=_BadRepr()),
        attr="CONST",
        ctx=ast.Load(),
    )
    assert da._const_repr(weird_attr) is None

    assert da._type_from_const_repr("None") == "None"
    assert da._type_from_const_repr("True") == "bool"
    assert da._type_from_const_repr("1") == "int"
    assert da._type_from_const_repr("1.5") == "float"
    assert da._type_from_const_repr("1+2j") == "complex"
    assert da._type_from_const_repr("'hi'") == "str"
    assert da._type_from_const_repr("b'hi'") == "bytes"
    assert da._type_from_const_repr("[1, 2]") == "list"
    assert da._type_from_const_repr("(1, 2)") == "tuple"
    assert da._type_from_const_repr("{1, 2}") == "set"
    assert da._type_from_const_repr("{'a': 1}") == "dict"
    assert da._type_from_const_repr("...") is None
    assert da._type_from_const_repr("not a literal") is None

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._expand_type_hint::hint E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._split_top_level::sep
def test_split_and_combine_type_hints() -> None:
    da = _load()
    assert da._split_top_level("A,B[C,D],E", ",") == ["A", "B[C,D]", "E"]
    assert da._expand_type_hint("") == set()
    assert da._expand_type_hint("Optional[int]") == {"int", "None"}
    assert da._expand_type_hint("Union[int, str]") == {"int", "str"}
    assert da._expand_type_hint("int|None") == {"int", "None"}

    assert da._combine_type_hints({"int"}) == ("int", False)
    assert da._combine_type_hints({"Optional[int]"}) == ("Optional[int]", False)
    union_hint, conflicted = da._combine_type_hints({"int", "str"})
    assert union_hint == "Union[int, str]"
    assert conflicted is True
    any_hint, any_conflicted = da._combine_type_hints({"None"})
    assert any_hint == "Any"
    assert any_conflicted is True
    optional_union, conflict = da._combine_type_hints({"Optional[int]", "Optional[str]"})
    assert optional_union == "Optional[Union[int, str]]"
    assert conflict is True

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._string_list::node E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._module_name::project_root
def test_module_name_and_strings() -> None:
    da = _load()
    project_root = Path("repo")
    path = project_root / "src" / "pkg" / "mod.py"
    assert da._module_name(path, project_root) == "pkg.mod"
    assert da._module_name(Path("other/mod.py"), project_root) == "other.mod"

    tree = ast.parse("__all__ = ['Foo', 'bar']")
    assert da._string_list(tree.body[0].value) == ["Foo", "bar"]
    tree = ast.parse("__all__ = ['Foo', 1]")
    assert da._string_list(tree.body[0].value) is None

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_module_exports::import_map,module_name E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._base_identifier::node E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._string_list::node E:decision_surface/value_encoded::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_module_exports::import_map
def test_base_identifier_and_exports() -> None:
    da = _load()
    name_node = ast.parse("x").body[0].value
    assert da._base_identifier(name_node) == "x"
    attr_node = ast.parse("mod.attr").body[0].value
    assert da._base_identifier(attr_node) == "mod.attr"
    sub_node = ast.parse("items[0]").body[0].value
    assert da._base_identifier(sub_node) == "items"
    call_node = ast.parse("factory().build").body[0].value
    assert da._base_identifier(call_node) == "factory().build"
    call_only = ast.parse("factory()").body[0].value
    assert da._base_identifier(call_only) == "factory"
    const_node = ast.parse("1").body[0].value
    assert da._base_identifier(const_node) is None

    module = ast.parse(
        "__all__ = ['Foo', 'bar']\n"
        "def Foo():\n"
        "    return 1\n"
        "def _hidden():\n"
        "    return 2\n"
        "value = 3\n"
    )
    export_names, export_map = da._collect_module_exports(
        module,
        module_name="demo",
        import_map={"bar": "other.bar"},
    )
    assert export_names == {"Foo", "bar"}
    assert export_map["Foo"] == "demo.Foo"
    assert export_map["bar"] == "other.bar"

    module = ast.parse(
        "__all__: list[str] = ['A']\n"
        "__all__ += ['B']\n"
        "A = 1\n"
        "B = 2\n"
    )
    export_names, export_map = da._collect_module_exports(
        module,
        module_name="demo",
        import_map={},
    )
    assert export_names == {"A", "B"}
    assert export_map["A"] == "demo.A"

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._is_test_path::path
def test_is_test_path() -> None:
    da = _load()
    assert da._is_test_path(Path("tests/test_sample.py")) is True
    assert da._is_test_path(Path("src/sample_test.py")) is False
    assert da._is_test_path(Path("src/test_module.py")) is True

# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit._call_context
def test_symbol_table_resolution_and_call_context() -> None:
    da = _load()
    table = da.SymbolTable(external_filter=True)
    table.imports[("pkg.mod", "Ext")] = "external.lib.Ext"
    table.imports[("pkg.mod", "Local")] = "pkg.local.Local"
    table.internal_roots.add("pkg")
    assert table.resolve("pkg.mod", "Local") == "pkg.local.Local"
    assert table.resolve("pkg.mod", "Ext") is None
    assert table.resolve("pkg.mod", "Name") == "pkg.mod.Name"

    table.external_filter = False
    assert table.resolve("pkg.mod", "Ext") == "external.lib.Ext"

    table.external_filter = True
    table.star_imports["pkg.mod"] = {"pkg.star", "ext.star"}
    table.module_exports["pkg.star"] = {"Foo"}
    table.module_export_map["pkg.star"] = {"Foo": "pkg.star.Foo"}
    table.module_exports["ext.star"] = {"Foo"}
    table.module_export_map["ext.star"] = {"Foo": "ext.star.Foo"}
    table.internal_roots.add("pkg")
    assert table.resolve_star("pkg.mod", "Foo") == "pkg.star.Foo"
    assert table.resolve_star("pkg.mod", "Missing") is None

    table.internal_roots.clear()
    assert table.resolve_star("pkg.mod", "Foo") is None

    table.external_filter = False
    assert table.resolve_star("pkg.mod", "Foo") == "ext.star.Foo"

    table.external_filter = True
    table.internal_roots.add("pkg")
    table.module_export_map["pkg.star"] = {}
    assert table.resolve_star("pkg.mod", "Foo") == "pkg.star.Foo"
    table.star_imports["pkg.mod"] = {"ext.star"}
    table.module_exports["ext.star"] = {"Foo"}
    table.module_export_map["ext.star"] = {}
    table.internal_roots = {"pkg"}
    assert table.resolve_star("pkg.mod", "Foo") is None
    table.star_imports["pkg.mod"] = {""}
    table.module_exports[""] = {"Bar"}
    assert table.resolve_star("pkg.mod", "Bar") == "Bar"

    tree = ast.parse("result = func(x)\n")
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    name_node = next(node for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id == "func")
    call_node, direct = da._call_context(name_node, parents)
    assert isinstance(call_node, ast.Call)
    assert direct is False

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._decorator_matches::allowlist,name E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_callee::class_name,name E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._decorators_transparent::fn,transparent_decorators E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._decorator_name::node
def test_callee_normalization_and_decorators(tmp_path: Path) -> None:
    da = _load()
    assert da._normalize_callee("self.run", "Service") == "Service.run"
    assert da._normalize_callee("cls.build", "Factory") == "Factory.build"
    assert da._normalize_callee("other", "Service") == "other"

    bad_call = ast.Call(func=ast.Name(id=None, ctx=ast.Load()), args=[], keywords=[])
    assert da._callee_name(bad_call) == "<call>"

    assert da._decorator_name(ast.Name(id="deco", ctx=ast.Load())) == "deco"
    attr = ast.Attribute(value=ast.Name(id="pkg", ctx=ast.Load()), attr="wrap")
    assert da._decorator_name(attr) == "pkg.wrap"
    call = ast.Call(func=attr, args=[], keywords=[])
    assert da._decorator_name(call) == "pkg.wrap"
    assert da._decorator_name(ast.Constant(value=1)) is None
    assert da._decorator_matches("pkg.decor", {"decor"}) is True
    assert da._decorator_matches("decor", {"decor"}) is True
    assert da._decorator_matches("other", {"decor"}) is False

    tree = ast.parse(
        "@deco\n"
        "def f():\n"
        "    return 1\n"
        "\n"
        "@other\n"
        "def g():\n"
        "    return 2\n"
    )
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    assert da._decorators_transparent(funcs[0], {"deco"}) is True
    assert da._decorators_transparent(funcs[1], {"deco"}) is False
    assert da._decorators_transparent(funcs[0], None) is True
    tree = ast.parse("@(lambda x: x)\ndef h():\n    return 3\n")
    func = tree.body[0]
    assert da._decorators_transparent(func, {"lambda"}) is False

    root = tmp_path / "root"
    root.mkdir()
    (root / "keep.py").write_text("x = 1\n")
    skip = root / "skip"
    skip.mkdir()
    (skip / "ignored.py").write_text("x = 2\n")
    config = da.AuditConfig(exclude_dirs={"skip"})
    paths = da._iter_paths([str(root)], config)
    assert root / "keep.py" in paths
    assert skip / "ignored.py" not in paths

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_annotations::fn,ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_names::fn,ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._node_span::node E:decision_surface/value_encoded::dataflow_audit.py::gabion.analysis.dataflow_audit._node_span::node
def test_param_helpers_and_scopes() -> None:
    da = _load()
    tree = ast.parse(
        "class C:\n"
        "    def method(self, x: int, *args, **kw):\n"
        "        def inner(y):\n"
        "            return y\n"
        "        return inner(x)\n"
    )
    class_node = tree.body[0]
    fn = class_node.body[0]
    inner = fn.body[0]

    assert da._param_names(fn) == ["x", "args", "kw"]
    assert da._param_names(fn, {"x"}) == ["args", "kw"]
    annots = da._param_annotations(fn)
    assert annots["x"] == "int"
    assert annots["args"] is None
    broken_tree = ast.parse("def broken(bad):\n    pass\n")
    broken_fn = broken_tree.body[0]
    broken_fn.args.args[0].annotation = ast.Name(id=None, ctx=ast.Load())
    annots = da._param_annotations(broken_fn)
    assert annots["bad"] is None

    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    assert da._enclosing_class(fn, parents) == "C"
    assert da._enclosing_class(inner, parents) == "C"
    assert da._enclosing_scopes(inner, parents) == ["C", "method"]
    assert da._enclosing_class_scopes(inner, parents) == ["C"]
    assert da._enclosing_function_scopes(inner, parents) == ["method"]
    assert da._function_key(["C", "method"], "inner") == "C.method.inner"

    arg = ast.arg(arg="z")
    arg.lineno = 1
    arg.col_offset = 0
    arg.end_lineno = 1
    arg.end_col_offset = 0
    assert da._node_span(arg) == (0, 0, 0, 1)
    assert da._node_span(ast.AST()) is None

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_spans::fn,ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._node_span::node E:decision_surface/value_encoded::dataflow_audit.py::gabion.analysis.dataflow_audit._node_span::node
def test_param_spans_ignore_var_kw() -> None:
    da = _load()
    tree = ast.parse("def f(a, *args, **kw):\n    return a\n")
    fn = tree.body[0]
    spans = da._param_spans(fn, {"a", "args", "kw"})
    assert spans == {}

# gabion:evidence E:function_site::test_dataflow_helpers.py::tests.test_dataflow_helpers._load
def test_audit_config_ignored_paths() -> None:
    da = _load()
    config = da.AuditConfig(exclude_dirs={"build", "dist"})
    assert config.is_ignored_path(Path("repo/build/main.py")) is True
    assert config.is_ignored_path(Path("repo/src/main.py")) is False

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._local_class_name::base,class_bases E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_local_method_in_hierarchy::class_name,local_functions,seen
def test_local_class_helpers() -> None:
    da = _load()
    class_bases = {
        "Base": [],
        "Child": ["Base"],
        "Loop": ["Loop"],
        "Other": ["external.Missing"],
    }
    assert da._local_class_name("pkg.Base", class_bases) == "Base"
    assert da._local_class_name("Missing", class_bases) is None
    assert da._local_class_name("pkg.Missing", class_bases) is None
    assert (
        da._resolve_local_method_in_hierarchy(
            "Child",
            "run",
            class_bases=class_bases,
            local_functions={"Base.run"},
            seen=set(),
        )
        == "Base.run"
    )
    assert (
        da._resolve_local_method_in_hierarchy(
            "Loop",
            "run",
            class_bases=class_bases,
            local_functions=set(),
            seen={"Loop"},
        )
        is None
    )
    assert (
        da._resolve_local_method_in_hierarchy(
            "Other",
            "run",
            class_bases=class_bases,
            local_functions=set(),
            seen=set(),
        )
        is None
    )

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._base_identifier::node
def test_collect_local_class_bases_skips_unknown() -> None:
    da = _load()
    tree = ast.parse(
        "class Base:\n"
        "    pass\n"
        "class Child(Base):\n"
        "    pass\n"
        "class Weird((lambda x: x)):\n"
        "    pass\n"
    )
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    bases = da._collect_local_class_bases(tree, parents)
    assert "Child" in bases
    assert bases["Child"] == ["Base"]
    assert "Weird" in bases
    assert bases["Weird"] == []

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_spans::fn,ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._node_span::node E:decision_surface/value_encoded::dataflow_audit.py::gabion.analysis.dataflow_audit._node_span::node
def test_param_spans_include_var_kw() -> None:
    da = _load()
    tree = ast.parse("def f(a, *args, **kw):\n    return a\n")
    fn = tree.body[0]
    spans = da._param_spans(fn)
    assert "a" in spans
    assert "args" in spans
    assert "kw" in spans

# gabion:evidence E:function_site::test_dataflow_helpers.py::tests.test_dataflow_helpers._load
def test_resolve_star_external_filtered() -> None:
    da = _load()
    table = da.SymbolTable(external_filter=True)
    table.star_imports["pkg.mod"] = {"ext.star"}
    table.module_exports["ext.star"] = {"Foo"}
    table.module_export_map["ext.star"] = {}
    table.internal_roots.add("pkg")
    assert table.resolve_star("pkg.mod", "Foo") is None

# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit._return_aliases E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_defaults::fn,ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_names::fn,ignore_params
def test_param_defaults_and_return_aliases() -> None:
    da = _load()
    tree = ast.parse(
        "class C:\n"
        "    def method(self, a, b=1, *, c=2, d=None):\n"
        "        def inner():\n"
        "            return 1\n"
        "        async def inner_async():\n"
        "            return 2\n"
        "        _ = (lambda z: z)\n"
        "        return a\n"
    )
    fn = tree.body[0].body[0]
    defaults = da._param_defaults(fn)
    assert defaults == {"b", "c", "d"}
    defaults_filtered = da._param_defaults(fn, {"b"})
    assert defaults_filtered == {"c", "d"}
    alias = da._return_aliases(fn)
    assert alias == ["a"]

# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit._return_aliases E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_names::fn,ignore_params
def test_return_aliases_tuple_and_conflict() -> None:
    da = _load()
    tree = ast.parse(
        "def tuple_alias(a, b):\n"
        "    return (a, b)\n"
        "\n"
        "def conflict(a, b):\n"
        "    if a:\n"
        "        return a\n"
        "    return b\n"
    )
    tuple_fn = tree.body[0]
    conflict_fn = tree.body[1]
    assert da._return_aliases(tuple_fn) == ["a", "b"]
    assert da._return_aliases(conflict_fn) is None

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_names::fn,ignore_params
def test_collect_return_aliases_conflict() -> None:
    da = _load()
    tree = ast.parse(
        "def foo(a):\n"
        "    return a\n"
        "\n"
        "def foo(b):\n"
        "    return b\n"
        "\n"
        "def foo(c):\n"
        "    return c\n"
    )
    funcs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    aliases = da._collect_return_aliases(funcs, parents, ignore_params=None)
    assert "foo" not in aliases

# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit._return_aliases E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_names::fn,ignore_params
def test_return_aliases_bare_return() -> None:
    da = _load()
    tree = ast.parse(
        "def f(a):\n"
        "    return\n"
    )
    fn = tree.body[0]
    assert isinstance(fn, ast.FunctionDef)
    assert da._return_aliases(fn, ignore_params=None) is None
