from __future__ import annotations

import ast
from pathlib import Path

def _load():
    repo_root = Path(__file__).resolve().parents[1]
    from gabion.analysis import dataflow_audit as da

    return da

def _fn(da, *, name: str, qual: str, path: Path, class_name: str | None = None):
    return da.FunctionInfo(
        name=name,
        qual=qual,
        path=path,
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        class_name=class_name,
        scope=(),
        lexical_scope=(),
        function_span=(0, 0, 0, 1),
    )

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_class_candidates::base,class_index,module,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_callee::by_qual,callee_key,caller,class_index,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_method_in_hierarchy::by_qual,class_qual,seen E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._callee_key::name E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._module_name::project_root
def test_resolve_callee_globals_only() -> None:
    da = _load()
    path = Path("pkg/mod.py")
    caller = _fn(da, name="caller", qual="pkg.mod.caller", path=path)
    callee = _fn(da, name="target", qual="pkg.mod.target", path=path)
    by_name = {"target": [callee]}
    by_qual = {callee.qual: callee}
    resolved = da._resolve_callee("target", caller, by_name, by_qual)
    assert resolved is callee

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_class_candidates::base,class_index,module,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_callee::by_qual,callee_key,caller,class_index,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_method_in_hierarchy::by_qual,class_qual,seen E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._callee_key::name E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._module_name::project_root
def test_resolve_callee_ambiguous_then_global() -> None:
    da = _load()
    path = Path("pkg/mod.py")
    caller = da.FunctionInfo(
        name="caller",
        qual="pkg.mod.caller",
        path=path,
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        class_name=None,
        scope=(),
        lexical_scope=(),
        function_span=(0, 0, 0, 1),
    )
    scoped_one = da.FunctionInfo(
        name="target",
        qual="pkg.mod.outer.caller.target",
        path=path,
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        class_name=None,
        scope=("outer", "caller"),
        lexical_scope=("caller",),
        function_span=(0, 0, 0, 1),
    )
    scoped_two = da.FunctionInfo(
        name="target",
        qual="pkg.mod.outer.caller.target2",
        path=path,
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        class_name=None,
        scope=("outer", "caller"),
        lexical_scope=("caller",),
        function_span=(0, 0, 0, 1),
    )
    global_candidate = da.FunctionInfo(
        name="target",
        qual="pkg.mod.target",
        path=path,
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        class_name=None,
        scope=(),
        lexical_scope=(),
        function_span=(0, 0, 0, 1),
    )
    by_name = {"target": [scoped_one, scoped_two, global_candidate]}
    by_qual = {
        scoped_one.qual: scoped_one,
        scoped_two.qual: scoped_two,
        global_candidate.qual: global_candidate,
    }
    resolved = da._resolve_callee("target", caller, by_name, by_qual)
    assert resolved is global_candidate

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_class_candidates::base,class_index,module,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_callee::by_qual,callee_key,caller,class_index,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_method_in_hierarchy::by_qual,class_qual,seen E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._callee_key::name E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._module_name::project_root
def test_resolve_callee_import_and_star() -> None:
    da = _load()
    path = Path("pkg/mod.py")
    caller = _fn(da, name="caller", qual="pkg.mod.caller", path=path)
    callee = _fn(da, name="Imported", qual="pkg.other.fn", path=Path("pkg/other.py"))
    by_name = {"Imported": [callee]}
    by_qual = {callee.qual: callee}
    table = da.SymbolTable()
    table.internal_roots.add("pkg")
    table.imports[("pkg.mod", "Imported")] = "pkg.other.fn"
    resolved = da._resolve_callee(
        "Imported",
        caller,
        by_name,
        by_qual,
        symbol_table=table,
        project_root=None,
    )
    assert resolved is callee

    table.star_imports["pkg.mod"] = {"pkg.star"}
    table.module_exports["pkg.star"] = {"StarFunc"}
    by_qual["pkg.star.StarFunc"] = _fn(
        da,
        name="StarFunc",
        qual="pkg.star.StarFunc",
        path=Path("pkg/star.py"),
    )
    resolved_star = da._resolve_callee(
        "StarFunc",
        caller,
        by_name,
        by_qual,
        symbol_table=table,
        project_root=None,
    )
    assert resolved_star is by_qual["pkg.star.StarFunc"]

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_class_candidates::base,class_index,module,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_callee::by_qual,callee_key,caller,class_index,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_method_in_hierarchy::by_qual,class_qual,seen E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._callee_key::name E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._module_name::project_root
def test_resolve_callee_import_filtered_out() -> None:
    da = _load()
    path = Path("pkg/mod.py")
    caller = _fn(da, name="caller", qual="pkg.mod.caller", path=path)
    callee = _fn(
        da,
        name="Ext",
        qual="ext.lib.fn",
        path=Path("ext/lib.py"),
        class_name="ExtClass",
    )
    by_name = {"Ext": [callee]}
    by_qual = {callee.qual: callee}
    table = da.SymbolTable(external_filter=True)
    table.imports[("pkg.mod", "Ext")] = "ext.lib.fn"
    resolved = da._resolve_callee(
        "Ext",
        caller,
        by_name,
        by_qual,
        symbol_table=table,
        project_root=None,
    )
    assert resolved is None

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_class_candidates::base,class_index,module,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_callee::by_qual,callee_key,caller,class_index,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_method_in_hierarchy::by_qual,class_qual,seen E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._callee_key::name E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._module_name::project_root
def test_resolve_callee_self_and_base_import() -> None:
    da = _load()
    path = Path("pkg/mod.py")
    caller = _fn(da, name="caller", qual="pkg.mod.Service.caller", path=path, class_name="Service")
    self_method = _fn(da, name="run", qual="pkg.mod.Service.run", path=path, class_name="Service")
    by_name = {"run": [self_method]}
    by_qual = {self_method.qual: self_method}
    resolved = da._resolve_callee(
        "self.run",
        caller,
        by_name,
        by_qual,
        symbol_table=da.SymbolTable(),
        project_root=None,
    )
    assert resolved is self_method

    table = da.SymbolTable()
    table.internal_roots.add("pkg")
    table.imports[("pkg.mod", "Alias")] = "pkg.other.Service"
    imported_method = _fn(
        da,
        name="method",
        qual="pkg.other.Service.method",
        path=Path("pkg/other.py"),
    )
    by_qual[imported_method.qual] = imported_method
    resolved_imported = da._resolve_callee(
        "Alias.method",
        caller,
        by_name,
        by_qual,
        symbol_table=table,
        project_root=None,
    )
    assert resolved_imported is imported_method

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_class_candidates::base,class_index,module,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_callee::by_qual,callee_key,caller,class_index,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_method_in_hierarchy::by_qual,class_qual,seen E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._callee_key::name E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._module_name::project_root
def test_resolve_callee_class_hierarchy() -> None:
    da = _load()
    path = Path("pkg/mod.py")
    caller = _fn(da, name="caller", qual="pkg.mod.caller", path=path)
    base_method = _fn(
        da,
        name="run",
        qual="pkg.Base.run",
        path=Path("pkg/base.py"),
    )
    by_name = {"run": [base_method]}
    by_qual = {base_method.qual: base_method}
    class_index = {
        "pkg.Base": da.ClassInfo(qual="pkg.Base", module="pkg", bases=[], methods={"run"}),
        "pkg.Service": da.ClassInfo(qual="pkg.Service", module="pkg", bases=["Base"], methods=set()),
    }
    table = da.SymbolTable()
    table.internal_roots.add("pkg")
    table.imports[("pkg.mod", "Service")] = "pkg.Service"
    resolved = da._resolve_callee(
        "Service.run",
        caller,
        by_name,
        by_qual,
        symbol_table=table,
        project_root=None,
        class_index=class_index,
    )
    assert resolved is base_method

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_class_candidates::base,class_index,module,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_callee::by_qual,callee_key,caller,class_index,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_method_in_hierarchy::by_qual,class_qual,seen E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._callee_key::name E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._module_name::project_root
def test_resolve_callee_self_class_candidates() -> None:
    da = _load()
    path = Path("pkg/mod.py")
    caller = _fn(
        da,
        name="caller",
        qual="pkg.mod.Service.caller",
        path=path,
        class_name="Service",
    )
    method = _fn(
        da,
        name="run",
        qual="pkg.mod.Service.run",
        path=path,
        class_name="Service",
    )
    by_name = {"run": [method]}
    by_qual = {method.qual: method}
    class_index = {
        "pkg.mod.Service": da.ClassInfo(
            qual="pkg.mod.Service",
            module="pkg.mod",
            bases=[],
            methods={"run"},
        )
    }
    resolved = da._resolve_callee(
        "self.run",
        caller,
        by_name,
        by_qual,
        symbol_table=None,
        project_root=None,
        class_index=class_index,
    )
    assert resolved is method


def _index_for_source(tmp_path: Path, source: str):
    da = _load()
    mod = tmp_path / "pkg" / "mod.py"
    mod.parent.mkdir(parents=True, exist_ok=True)
    mod.write_text(source)
    by_name, by_qual = da._build_function_index(
        [mod],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        parse_failure_witnesses=[],
    )
    caller = by_qual["pkg.mod.caller"]
    return da, by_name, by_qual, caller


def test_resolve_callee_direct_lambda_call(tmp_path: Path) -> None:
    da, by_name, by_qual, caller = _index_for_source(
        tmp_path,
        "def caller():\n    return (lambda value: value)(1)\n",
    )
    call = caller.calls[0]
    resolved = da._resolve_callee(call.callee, caller, by_name, by_qual)
    assert resolved is not None
    assert resolved.name.startswith("<lambda:")
    assert resolved.qual == call.callee


def test_resolve_callee_bound_lambda_call(tmp_path: Path) -> None:
    da, by_name, by_qual, caller = _index_for_source(
        tmp_path,
        "def caller():\n    fn = lambda value: value\n    return fn(1)\n",
    )
    call = caller.calls[0]
    outcome = da._resolve_callee_outcome(call.callee, caller, by_name, by_qual)
    assert outcome.status == "resolved"
    assert len(outcome.candidates) == 1
    assert outcome.candidates[0].name.startswith("<lambda:")


def test_resolve_callee_bound_lambda_ambiguous_aliasing(tmp_path: Path) -> None:
    da, by_name, by_qual, caller = _index_for_source(
        tmp_path,
        (
            "def caller(flag):\n"
            "    if flag:\n"
            "        fn = lambda value: value\n"
            "    else:\n"
            "        fn = lambda value: value + 1\n"
            "    return fn(1)\n"
        ),
    )
    call = caller.calls[0]
    outcome = da._resolve_callee_outcome(call.callee, caller, by_name, by_qual)
    assert outcome.status == "ambiguous"
    assert outcome.phase == "local_lambda_binding"
    assert len(outcome.candidates) == 2


def test_collect_lambda_function_infos_ignores_missing_span_nodes() -> None:
    da = _load()
    tree = ast.Module(
        body=[
            ast.Expr(
                value=ast.Lambda(
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[ast.arg(arg="x")],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[],
                    ),
                    body=ast.Name(id="x", ctx=ast.Load()),
                )
            )
        ],
        type_ignores=[],
    )
    parent_map: dict[ast.AST, ast.AST] = {}
    da.ParentAnnotator().visit(tree)
    infos = da._collect_lambda_function_infos(
        tree,
        path=Path("pkg/mod.py"),
        module="pkg.mod",
        parent_map=parent_map,
        ignore_params=None,
    )
    assert infos == []


def test_collect_lambda_function_infos_applies_ignore_params(tmp_path: Path) -> None:
    da = _load()
    mod = tmp_path / "pkg" / "mod.py"
    mod.parent.mkdir(parents=True, exist_ok=True)
    tree = ast.parse("def caller():\n    fn = lambda skip, keep: keep\n    return fn(skip=1, keep=2)\n")
    parents = da.ParentAnnotator()
    parents.visit(tree)
    infos = da._collect_lambda_function_infos(
        tree,
        path=mod,
        module="pkg.mod",
        parent_map=parents.parents,
        ignore_params={"skip"},
    )
    assert len(infos) == 1
    assert infos[0].params == ["keep"]


def test_collect_lambda_bindings_ignores_unmapped_lambda_spans() -> None:
    da = _load()
    tree = ast.parse("def caller():\n    fn = lambda x: x\n")
    parents = da.ParentAnnotator()
    parents.visit(tree)
    bindings = da._collect_lambda_bindings_by_caller(
        tree,
        module="pkg.mod",
        parent_map=parents.parents,
        lambda_infos=[],
    )
    assert bindings == {}


def test_direct_lambda_mapping_skips_missing_and_unmapped_spans() -> None:
    da = _load()
    locationless_tree = ast.Module(
        body=[
            ast.Expr(
                value=ast.Call(
                    func=ast.Lambda(
                        args=ast.arguments(
                            posonlyargs=[],
                            args=[ast.arg(arg="x")],
                            kwonlyargs=[],
                            kw_defaults=[],
                            defaults=[],
                        ),
                        body=ast.Name(id="x", ctx=ast.Load()),
                    ),
                    args=[ast.Constant(value=1)],
                    keywords=[],
                )
            )
        ],
        type_ignores=[],
    )
    assert da._direct_lambda_callee_by_call_span(locationless_tree, lambda_infos=[]) == {}

    parsed_tree = ast.parse("def caller():\n    return (lambda x: x)(1)\n")
    assert da._direct_lambda_callee_by_call_span(parsed_tree, lambda_infos=[]) == {}


def test_resolve_callee_single_missing_local_binding_falls_back_to_global() -> None:
    da = _load()
    path = Path("pkg/mod.py")
    caller = _fn(da, name="caller", qual="pkg.mod.caller", path=path)
    global_candidate = _fn(da, name="fn", qual="pkg.mod.fn", path=path)
    resolved = da._resolve_callee(
        "fn",
        caller,
        {"fn": [global_candidate]},
        {global_candidate.qual: global_candidate},
        local_lambda_bindings={"fn": ("pkg.mod.<lambda:missing>",)},
    )
    assert resolved is global_candidate


def test_resolve_callee_multi_bindings_with_one_present_returns_bound() -> None:
    da = _load()
    path = Path("pkg/mod.py")
    caller = _fn(da, name="caller", qual="pkg.mod.caller", path=path)
    bound = _fn(da, name="<lambda:bound>", qual="pkg.mod.<lambda:bound>", path=path)
    resolved = da._resolve_callee(
        "fn",
        caller,
        {},
        {bound.qual: bound},
        local_lambda_bindings={"fn": ("pkg.mod.<lambda:missing>", bound.qual)},
    )
    assert resolved is bound


def test_resolve_callee_multi_bindings_without_sink_returns_none() -> None:
    da = _load()
    path = Path("pkg/mod.py")
    caller = _fn(da, name="caller", qual="pkg.mod.caller", path=path)
    bound_a = _fn(da, name="<lambda:a>", qual="pkg.mod.<lambda:a>", path=path)
    bound_b = _fn(da, name="<lambda:b>", qual="pkg.mod.<lambda:b>", path=path)
    resolved = da._resolve_callee(
        "fn",
        caller,
        {},
        {bound_a.qual: bound_a, bound_b.qual: bound_b},
        local_lambda_bindings={"fn": (bound_a.qual, bound_b.qual)},
    )
    assert resolved is None


def test_resolve_callee_multi_missing_bindings_fall_back() -> None:
    da = _load()
    path = Path("pkg/mod.py")
    caller = _fn(da, name="caller", qual="pkg.mod.caller", path=path)
    global_candidate = _fn(da, name="fn", qual="pkg.mod.fn", path=path)
    resolved = da._resolve_callee(
        "fn",
        caller,
        {"fn": [global_candidate]},
        {global_candidate.qual: global_candidate},
        local_lambda_bindings={"fn": ("pkg.mod.<lambda:missing1>", "pkg.mod.<lambda:missing2>")},
    )
    assert resolved is global_candidate


def test_resolve_callee_outcome_adds_internal_candidates_from_bindings() -> None:
    da = _load()
    path = Path("pkg/mod.py")
    caller = _fn(da, name="caller", qual="pkg.mod.caller", path=path)
    bound = _fn(da, name="<lambda:bound>", qual="pkg.mod.<lambda:bound>", path=path)

    def _always_none(*_: object, **__: object):
        return None

    outcome = da._resolve_callee_outcome(
        "fn",
        caller,
        {},
        {bound.qual: bound},
        local_lambda_bindings={"fn": ("pkg.mod.<lambda:missing>", bound.qual)},
        resolve_callee_fn=_always_none,
    )
    assert outcome.status == "unresolved_internal"
    assert tuple(outcome.candidates) == (bound,)
