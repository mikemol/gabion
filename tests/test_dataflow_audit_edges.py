from __future__ import annotations

import ast
from pathlib import Path
import sys
import textwrap

from gabion.analysis.timeout_context import Deadline, deadline_scope


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit as da

    return da


def _deadline_obligations(tmp_path: Path, source: str, roots: set[str]) -> list[dict]:
    da = _load()
    target = tmp_path / "mod.py"
    target.write_text(textwrap.dedent(source), encoding="utf-8")
    config = da.AuditConfig(
        project_root=tmp_path,
        exclude_dirs=set(),
        ignore_params=set(),
        external_filter=True,
        strictness="high",
        deadline_roots=set(roots),
    )
    result = da.analyze_paths(
        [target],
        forest=da.Forest(),
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_deadness_witnesses=False,
        include_coherence_witnesses=False,
        include_rewrite_plans=False,
        include_exception_obligations=False,
        include_handledness_witnesses=False,
        include_never_invariants=False,
        include_decision_surfaces=False,
        include_value_decision_surfaces=False,
        include_invariant_propositions=False,
        include_lint_lines=False,
        include_ambiguities=False,
        include_bundle_forest=False,
        include_deadline_obligations=True,
        config=config,
    )
    return result.deadline_obligations


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_module_exports::import_map,module_name E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._string_list::node E:decision_surface/value_encoded::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_module_exports::import_map
def test_collect_module_exports_augassign_only() -> None:
    da = _load()
    module = ast.parse("__all__ += ['A']\nA = 1\n")
    export_names, export_map = da._collect_module_exports(
        module,
        module_name="demo",
        import_map={},
    )
    assert export_names == {"A"}
    assert export_map["A"] == "demo.A"


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._base_identifier::node
def test_base_identifier_invalid_attribute() -> None:
    da = _load()
    bad_attr = ast.Attribute(
        value=ast.Constant(value=1),
        attr=None,
        ctx=ast.Load(),
    )
    assert da._base_identifier(bad_attr) is None


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_names::fn,ignore_params
def test_collect_return_aliases_empty_params() -> None:
    da = _load()
    tree = ast.parse(
        "def f():\n"
        "    return 1\n"
    )
    fn = tree.body[0]
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    aliases = da._collect_return_aliases([fn], parents, ignore_params=None)
    assert aliases == {}


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_names::fn,ignore_params
def test_collect_return_aliases_all_params_ignored() -> None:
    da = _load()
    tree = ast.parse(
        "def f(a):\n"
        "    return a\n"
    )
    fn = tree.body[0]
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    aliases = da._collect_return_aliases([fn], parents, ignore_params={"a"})
    assert aliases == {}


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._analyze_file_internal::config,recursive
def test_analyze_file_recursive_false(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "mod.py"
    target.write_text(
        "def callee(x):\n"
        "    return x\n"
        "\n"
        "def caller(a, b):\n"
        "    callee(a)\n"
        "    callee(b)\n"
    )
    groups, spans = da.analyze_file(target, recursive=False)
    assert groups
    assert spans


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._analyze_file_internal::config,recursive
def test_resolve_local_callee_ambiguity_and_scope(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "mod.py"
    target.write_text(
        textwrap.dedent(
            """
            def outer():
                def inner(x):
                    return x
                def inner(x):
                    return x
                def caller(y):
                    return inner(y)
                return caller

            def helper(x):
                return x

            def global_caller(z):
                return helper(z)

            def outer2():
                def nested(x):
                    return x
                return nested

            def top():
                nested(1)
            """
        ).strip()
        + "\n"
    )
    config = da.AuditConfig(project_root=tmp_path)
    groups, _ = da.analyze_file(target, recursive=True, config=config)
    assert groups


# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_call_ambiguities
def test_ambiguity_witnesses_emit(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "mod.py"
    target.write_text(
        textwrap.dedent(
            """
            def outer():
                def inner(x):
                    return x
                def inner(x):
                    return x
                def caller(y):
                    return inner(y)
                return caller
            """
        ).strip()
        + "\n"
    )
    config = da.AuditConfig(project_root=tmp_path)
    analysis = da.analyze_paths(
        [tmp_path],
        forest=da.Forest(),
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=10,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_deadness_witnesses=False,
        include_coherence_witnesses=False,
        include_rewrite_plans=False,
        include_exception_obligations=False,
        include_handledness_witnesses=False,
        include_never_invariants=False,
        include_decision_surfaces=False,
        include_value_decision_surfaces=False,
        include_invariant_propositions=False,
        include_lint_lines=True,
        include_ambiguities=True,
        include_bundle_forest=True,
        config=config,
    )
    assert analysis.ambiguity_witnesses
    assert any(
        entry.get("kind") == "local_resolution_ambiguous"
        for entry in analysis.ambiguity_witnesses
        if isinstance(entry, dict)
    )
    assert any("GABION_AMBIGUITY" in line for line in analysis.lint_lines)
    assert analysis.forest is not None
    assert not any(node.kind == "AmbiguitySet" for node in analysis.forest.nodes.values())
    assert any(
        node.kind == "SuiteSite" and node.meta.get("suite_kind") == "call"
        for node in analysis.forest.nodes.values()
    )
    assert any(alt.kind == "CallCandidate" for alt in analysis.forest.alts)
    summary = da._summarize_call_ambiguities(analysis.ambiguity_witnesses)
    assert summary
    assert summary[0].startswith("generated_by_spec_id:")


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._analyze_file_internal::config,recursive
def test_analyze_file_local_callee_globals(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "mod.py"
    target.write_text(
        textwrap.dedent(
            """
            def helper(x):
                return x

            def outer():
                def inner(y):
                    return helper(y)
                return inner

            def caller(z):
                return helper(z)
            """
        ).strip()
        + "\n"
    )
    groups, _ = da.analyze_file(target, recursive=True, config=da.AuditConfig(project_root=tmp_path))
    assert groups


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_class_candidates::base,class_index,module,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_callee::by_qual,callee_key,caller,class_index,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_method_in_hierarchy::by_qual,class_qual,seen E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_annotations::fn,ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_defaults::fn,ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_names::fn,ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_spans::fn,ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._decorators_transparent::fn,transparent_decorators E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._build_function_index::ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_module_exports::import_map,module_name E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._callee_key::name E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._base_identifier::node E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._is_test_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._module_name::project_root E:decision_surface/value_encoded::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_module_exports::import_map
def test_resolve_callee_imports_and_self(tmp_path: Path) -> None:
    da = _load()
    pkg_root = tmp_path / "pkg"
    pkg_root.mkdir()
    mod_a = pkg_root / "a.py"
    mod_b = pkg_root / "b.py"
    mod_a.write_text(
        textwrap.dedent(
            """
            from pkg.b import ext

            class C:
                def method(self, x):
                    return x

                def call(self, y):
                    return self.method(y)

            def caller(z):
                return ext(z)
            """
        ).strip()
        + "\n"
    )
    mod_b.write_text(
        "def ext(v):\n"
        "    return v\n"
    )
    paths = [mod_a, mod_b]
    by_name, by_qual = da._build_function_index(
        paths,
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        transparent_decorators=None,
    )
    symbol_table = da._build_symbol_table(paths, tmp_path, external_filter=True)
    class_index = da._collect_class_index(paths, tmp_path)
    caller_info = by_name["caller"][0]
    resolved = da._resolve_callee(
        "ext",
        caller_info,
        by_name,
        by_qual,
        symbol_table,
        tmp_path,
        class_index,
    )
    assert resolved is not None
    method_info = by_name["call"][0]
    resolved_method = da._resolve_callee(
        "self.method",
        method_info,
        by_name,
        by_qual,
        symbol_table,
        tmp_path,
        class_index,
    )
    assert resolved_method is not None


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_class_candidates::base,class_index,module,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_callee::by_qual,callee_key,caller,class_index,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_method_in_hierarchy::by_qual,class_qual,seen E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_annotations::fn,ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_defaults::fn,ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_names::fn,ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_spans::fn,ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._decorators_transparent::fn,transparent_decorators E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._build_function_index::ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_module_exports::import_map,module_name E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._callee_key::name E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._base_identifier::node E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._is_test_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._module_name::project_root E:decision_surface/value_encoded::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_module_exports::import_map
def test_resolve_callee_external_filtered(tmp_path: Path) -> None:
    da = _load()
    mod_a = tmp_path / "a.py"
    mod_a.write_text(
        "import external.mod as ext\n"
        "def caller(x):\n"
        "    return ext.run(x)\n"
    )
    paths = [mod_a]
    by_name, by_qual = da._build_function_index(
        paths,
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        transparent_decorators=None,
    )
    symbol_table = da._build_symbol_table(paths, tmp_path, external_filter=True)
    class_index = da._collect_class_index(paths, tmp_path)
    caller_info = by_name["caller"][0]
    resolved = da._resolve_callee(
        "ext.run",
        caller_info,
        by_name,
        by_qual,
        symbol_table,
        tmp_path,
        class_index,
    )
    assert resolved is None


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_class_candidates::base,class_index,module,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_callee::by_qual,callee_key,caller,class_index,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_method_in_hierarchy::by_qual,class_qual,seen E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._callee_key::name E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._module_name::project_root
def test_resolve_callee_global_and_imported(tmp_path: Path) -> None:
    da = _load()
    mod_path = tmp_path / "mod.py"
    caller = da.FunctionInfo(
        name="caller",
        qual="mod.caller",
        path=mod_path,
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        function_span=(0, 0, 0, 1),
    )
    helper = da.FunctionInfo(
        name="helper",
        qual="mod.helper",
        path=mod_path,
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        function_span=(0, 0, 0, 1),
    )
    by_name = {"helper": [helper], "caller": [caller]}
    by_qual = {helper.qual: helper, caller.qual: caller}
    resolved = da._resolve_callee(
        "helper",
        caller,
        by_name,
        by_qual,
        None,
        tmp_path,
        None,
    )
    assert resolved == helper

    imported = da.FunctionInfo(
        name="target",
        qual="other.target",
        path=tmp_path / "other.py",
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        function_span=(0, 0, 0, 1),
    )
    by_qual[imported.qual] = imported
    by_name.setdefault("target", []).append(imported)
    table = da.SymbolTable(external_filter=True)
    table.imports[("mod", "alias")] = imported.qual
    table.internal_roots.add("other")
    resolved = da._resolve_callee(
        "alias",
        caller,
        by_name,
        by_qual,
        table,
        tmp_path,
        None,
    )
    assert resolved == imported

    by_qual["mod.service.run"] = imported
    resolved = da._resolve_callee(
        "service.run",
        caller,
        by_name,
        by_qual,
        table,
        tmp_path,
        None,
    )
    assert resolved == imported

    resolved = da._resolve_callee(
        "mod.helper",
        caller,
        by_name,
        by_qual,
        None,
        tmp_path,
        None,
    )
    assert resolved == helper


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_class_candidates::base,class_index,module,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_method_in_hierarchy::by_qual,class_qual,seen
def test_resolve_method_in_hierarchy_edges() -> None:
    da = _load()
    class_index = {
        "pkg.Base": da.ClassInfo(
            qual="pkg.Base", module="pkg", bases=["Missing"], methods={"run"}
        )
    }
    by_qual = {}
    assert (
        da._resolve_method_in_hierarchy(
            "pkg.Base",
            "run",
            class_index=class_index,
            by_qual=by_qual,
            symbol_table=None,
            seen={"pkg.Base"},
        )
        is None
    )
    assert (
        da._resolve_method_in_hierarchy(
            "pkg.Missing",
            "run",
            class_index=class_index,
            by_qual=by_qual,
            symbol_table=None,
            seen=set(),
        )
        is None
    )
    assert (
        da._resolve_method_in_hierarchy(
            "pkg.Base",
            "missing",
            class_index=class_index,
            by_qual=by_qual,
            symbol_table=None,
            seen=set(),
        )
        is None
    )


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._apply_baseline::baseline_allowlist E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._build_function_index::ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._callee_key::name E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._infer_root::groups_by_path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._load_baseline::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._module_name::project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_callee::ambiguity_sink,by_qual,callee_key,caller,class_index,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_class_candidates::base,class_index,module,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_method_in_hierarchy::by_qual,class_qual,seen E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind
def test_render_helpers_and_baseline(tmp_path: Path) -> None:
    da = _load()
    assert da._infer_root({}) == Path(".")
    assert da._callee_key("") == ""
    assert (
        da._resolve_callee(
            "",
            da.FunctionInfo(
                name="caller",
                qual="caller",
                path=tmp_path / "mod.py",
                params=[],
                annots={},
                calls=[],
                unused_params=set(),
                function_span=(0, 0, 0, 1),
            ),
            {},
            {},
            None,
            tmp_path,
            None,
        )
        is None
    )
    assert da._resolve_class_candidates(
        "",
        module="",
        symbol_table=None,
        class_index={},
    ) == []

    base_dir = tmp_path / "baseline_dir"
    base_dir.mkdir()
    assert da._load_baseline(base_dir) == set()
    assert da._apply_baseline(["a"], set()) == (["a"], [])

    report = da.render_synthesis_section(
        {
            "protocols": [
                {
                    "name": "Bundle",
                    "tier": 2,
                    "fields": [{"name": "a", "type_hint": "Optional[int]"}],
                }
            ],
            "warnings": ["warn"],
            "errors": ["err"],
        }
    )
    assert "Synthesis plan" in report

    stubs = da.render_protocol_stubs(
        {
            "protocols": [
                {
                    "name": "Bundle",
                    "tier": 3,
                    "bundle": ["a", "b"],
                    "rationale": "test",
                    "fields": [
                        {"name": "a", "type_hint": "Optional[int]"},
                        {"name": "b", "type_hint": "Union[int, str]"},
                    ],
                },
                {
                    "name": "Empty",
                    "tier": 2,
                    "bundle": [],
                    "fields": [],
                },
            ],
            "warnings": [],
            "errors": [],
        },
        kind="unknown",
    )
    assert "TODO_Name_Me" in stubs

    plan = da.build_refactor_plan({}, [], config=da.AuditConfig())
    refactor_text = da.render_refactor_plan(plan)
    assert "No refactoring plan" in refactor_text

    plan = {
        "bundles": [{"bundle": ["a", "b"], "order": ["x"], "cycles": [["x"]]}],
        "warnings": ["warn"],
    }
    refactor_text = da.render_refactor_plan(plan)
    assert "Cycles:" in refactor_text
    assert "Warnings:" in refactor_text


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._render_component_callsite_evidence::bundle_counts E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::bundle_sites_by_path,coherence_witnesses,constant_smells,context_suggestions,deadness_witnesses,decision_surfaces,decision_warnings,exception_obligations,fingerprint_matches,fingerprint_provenance,fingerprint_synth,fingerprint_warnings,forest,groups_by_path,handledness_witnesses,invariant_propositions,max_components,never_invariants,rewrite_plans,type_ambiguities,type_callsite_evidence,type_suggestions,unused_arg_smells,value_decision_rewrites,value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._render_mermaid_component::component,declared_global,nodes E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_dataclass_call_bundles::dataclass_registry,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_never_invariants::entries,include_proven_unreachable,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_coherence_witnesses::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_deadness_witnesses::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_exception_obligations::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_handledness_witnesses::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_rewrite_plans::entries,max_entries E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_fingerprint_provenance::entries,max_examples E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._bundle_projection_from_forest::file_paths E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._populate_bundle_forest::groups_by_path
def test_emit_report_tier2_violation(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "a.py"
    target.write_text(
        "def caller(a, b):\n"
        "    return a\n"
        "\n"
        "def caller2(a, b):\n"
        "    return b\n"
    )
    groups_by_path = {
        target: {
            "caller": [{"a", "b"}],
            "caller2": [{"a", "b"}],
        }
    }
    forest = da.Forest()
    da._populate_bundle_forest(
        forest,
        groups_by_path=groups_by_path,
        file_paths=[target],
        project_root=tmp_path,
    )
    report, violations = da._emit_report(
        groups_by_path, max_components=10, forest=forest
    )
    assert "tier-2" in report
    assert violations


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._infer_root::groups_by_path
def test_infer_root_with_groups(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "a.py"
    groups_by_path = {target: {}}
    assert da._infer_root(groups_by_path) == target


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_callee::by_qual,callee_key,caller,class_index,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._analyze_file_internal::config,recursive E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._build_function_index::ignore_params
def test_build_refactor_plan_skips_unresolved_and_opaque(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "mod.py"
    target.write_text(
        textwrap.dedent(
            """
            def callee(x):
                return x

            def opaque(fn):
                return fn

            @opaque
            def hidden(a, b):
                return a

            def caller(a, b):
                callee(a)
                callee(b)
                unknown(a)
                hidden(a, b)
            """
        ).strip()
        + "\n"
    )
    config = da.AuditConfig(
        project_root=tmp_path,
        exclude_dirs=set(),
        ignore_params=set(),
        external_filter=True,
        strictness="high",
        transparent_decorators={"transparent"},
    )
    groups, _ = da.analyze_file(target, recursive=True, config=config)
    groups_by_path = {target: groups}
    plan = da.build_refactor_plan(groups_by_path, [target], config=config)
    assert "bundles" in plan


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_callee::by_qual,callee_key,caller,class_index,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._build_function_index::ignore_params
def test_build_refactor_plan_skips_none_and_nontransparent(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "mod.py"
    target.write_text(
        textwrap.dedent(
            """
            def opaque(fn):
                return fn

            @opaque
            def hidden(a, b):
                return a

            def caller(a, b):
                hidden(a, b)
                unknown(a)
            """
        ).strip()
        + "\n"
    )
    config = da.AuditConfig(
        project_root=tmp_path,
        exclude_dirs=set(),
        ignore_params=set(),
        external_filter=True,
        strictness="high",
        transparent_decorators=None,
    )
    groups_by_path = {target: {"caller": [set(["a", "b"])]}}
    plan = da.build_refactor_plan(groups_by_path, [target], config=config)
    assert "bundles" in plan


# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit._render_type_mermaid
def test_render_type_mermaid_edges() -> None:
    da = _load()
    graph = da._render_type_mermaid(
        ["bad entry"],
        ["missing format", "file:fn.param downstream types conflict: []"],
    )
    assert "flowchart LR" in graph


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_callee::by_qual,callee_key,caller,class_index,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_annotations::fn,ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_defaults::fn,ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_names::fn,ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_spans::fn,ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._decorators_transparent::fn,transparent_decorators E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._build_function_index::ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_module_exports::import_map,module_name E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._base_identifier::node E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._is_test_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._module_name::project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_knob_param_names::strictness E:decision_surface/value_encoded::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_module_exports::import_map
def test_compute_knob_param_names_non_const_kw(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "mod.py"
    target.write_text(
        "def callee(a, b):\n"
        "    return b\n"
        "\n"
        "def caller(x):\n"
        "    return callee(a=1, b=x + 1)\n"
    )
    paths = [target]
    by_name, by_qual = da._build_function_index(
        paths,
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        transparent_decorators=None,
    )
    symbol_table = da._build_symbol_table(paths, tmp_path, external_filter=True)
    class_index = da._collect_class_index(paths, tmp_path)
    knob_names = da._compute_knob_param_names(
        by_name=by_name,
        by_qual=by_qual,
        symbol_table=symbol_table,
        project_root=tmp_path,
        class_index=class_index,
        strictness="high",
    )
    assert "a" in knob_names or "b" in knob_names


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_dataclass_call_bundles._resolve_fields::call E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_dataclass_call_bundles::dataclass_registry,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_module_exports::import_map,module_name E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._module_name::project_root E:decision_surface/value_encoded::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_module_exports::import_map
def test_iter_dataclass_call_bundles(tmp_path: Path) -> None:
    da = _load()
    root = tmp_path / "src"
    root.mkdir()
    mod_a = root / "a.py"
    mod_b = root / "b.py"
    mod_a.write_text(
        textwrap.dedent(
            """
            from dataclasses import dataclass

            @dataclass
            class Bundle:
                a: int
                b: int
            """
        ).strip()
        + "\n"
    )
    mod_b.write_text(
        textwrap.dedent(
            """
            from a import Bundle

            def build():
                Bundle(1, 2, 3)
                Bundle(a=1, b=2)
                mod.Bundle(1, 2)
            """
        ).strip()
        + "\n"
    )
    paths = [mod_a, mod_b]
    symbol_table = da._build_symbol_table(paths, root, external_filter=True)
    registry = da._collect_dataclass_registry(paths, project_root=root)
    bundles = da._iter_dataclass_call_bundles(
        mod_b,
        project_root=root,
        symbol_table=symbol_table,
        dataclass_registry=registry,
    )
    assert tuple(sorted(("a", "b"))) in bundles
    bundles = da._iter_dataclass_call_bundles(
        mod_b,
        project_root=root,
        symbol_table=None,
        dataclass_registry={"b.Bundle": ["a", "b"]},
    )
    assert tuple(sorted(("a", "b"))) in bundles
    bundles = da._iter_dataclass_call_bundles(
        mod_b,
        project_root=root,
        symbol_table=None,
        dataclass_registry={"Bundle": ["a", "b"]},
    )
    assert tuple(sorted(("a", "b"))) in bundles
    table = da.SymbolTable(external_filter=True)
    table.star_imports["b"] = {"pkg"}
    table.module_exports["pkg"] = {"mod"}
    table.module_export_map["pkg"] = {"mod": "pkg.mod"}
    table.internal_roots.add("pkg")
    bundles = da._iter_dataclass_call_bundles(
        mod_b,
        project_root=root,
        symbol_table=table,
        dataclass_registry={"pkg.mod.Bundle": ["a", "b"]},
    )
    assert tuple(sorted(("a", "b"))) in bundles


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_callee::by_qual,callee_key,caller,class_index,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._infer_root::groups_by_path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._build_function_index::ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._merge_counts_by_knobs::knob_names E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::merge.py::gabion.synthesis.merge.merge_bundles::min_overlap E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_knob_param_names::strictness
def test_build_synthesis_plan_skips_empty_members(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "mod.py"
    target.write_text("def f(a):\n    return a\n")
    groups_by_path = {target: {"f": [set()]}}
    plan = da.build_synthesis_plan(
        groups_by_path,
        project_root=tmp_path,
        max_tier=2,
        min_bundle_size=1,
        allow_singletons=True,
        config=da.AuditConfig(project_root=tmp_path),
    )
    assert "protocols" in plan


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._analyze_file_internal::config,recursive
def test_analyze_file_local_resolution_ambiguous(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        textwrap.dedent(
            """
            def outer(x):
                class C:
                    def foo(self, y):
                        return y

                class D:
                    def foo(self, y):
                        return y

                def caller(z):
                    return foo(z)

                return caller(x)
            """
        ).strip()
        + "\n"
    )
    groups, _ = da.analyze_file(
        path,
        recursive=True,
        config=da.AuditConfig(
            project_root=tmp_path,
            exclude_dirs=set(),
            ignore_params=set(),
            external_filter=True,
            strictness="high",
        ),
    )
    assert isinstance(groups, dict)


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._analyze_file_internal::config,recursive
def test_analyze_file_local_resolution_globals_only(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        textwrap.dedent(
            """
            def foo(a):
                return a

            def outer(x):
                def inner(y):
                    return foo(y)
                return inner(x)
            """
        ).strip()
        + "\n"
    )
    groups, _ = da.analyze_file(
        path,
        recursive=True,
        config=da.AuditConfig(
            project_root=tmp_path,
            exclude_dirs=set(),
            ignore_params=set(),
            external_filter=True,
            strictness="high",
        ),
    )
    assert isinstance(groups, dict)


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_callee::by_qual,callee_key,caller,class_index,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._build_function_index::ignore_params
def test_build_refactor_plan_skips_nontransparent_dependency(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "mod.py"
    target.write_text(
        textwrap.dedent(
            """
            def opaque(fn):
                return fn

            @opaque
            def hidden(a, b):
                return a

            def caller(a, b):
                hidden(a, b)
            """
        ).strip()
        + "\n"
    )
    groups_by_path = {target: {"caller": [set(["a", "b"])]}}
    config = da.AuditConfig(
        project_root=tmp_path,
        exclude_dirs=set(),
        ignore_params=set(),
        external_filter=True,
        strictness="high",
        transparent_decorators={"transparent"},
    )
    plan = da.build_refactor_plan(groups_by_path, [target], config=config)
    assert "bundles" in plan


# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit._populate_bundle_forest
def test_populate_bundle_forest_skips_test_sites(tmp_path: Path) -> None:
    da = _load()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_sample.py"
    test_file.write_text("def helper(x):\n    return x\n", encoding="utf-8")
    forest = da.Forest()
    da._populate_bundle_forest(
        forest,
        groups_by_path={test_file: {}},
        file_paths=[test_file],
        project_root=tmp_path,
        include_all_sites=True,
        ignore_params=set(),
        strictness="high",
        transparent_decorators=set(),
    )
    assert all(node.kind != "FunctionSite" for node in forest.nodes.values())


# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit._populate_bundle_forest
def test_populate_bundle_forest_empty_groups(tmp_path: Path) -> None:
    da = _load()
    forest = da.Forest()
    da._populate_bundle_forest(
        forest,
        groups_by_path={},
        file_paths=[],
        project_root=tmp_path,
        include_all_sites=True,
        ignore_params=set(),
        strictness="high",
        transparent_decorators=set(),
    )
    assert forest.nodes == {}


# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_warnings
def test_compute_fingerprint_warnings_missing_annotations(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "mod.py"
    groups_by_path = {target: {"caller": [set(["a", "b"])]}}
    annotations_by_path = {target: {"caller": {"a": "int"}}}
    warnings = da._compute_fingerprint_warnings(
        groups_by_path,
        annotations_by_path,
        registry=da.PrimeRegistry(),
        index={object(): set()},
    )
    assert warnings
    assert "missing type annotations" in warnings[0]


# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths
def test_analyze_paths_deadline_includes_forest_spec(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "mod.py"
    target.write_text("def callee(x):\n    return x\n", encoding="utf-8")
    with deadline_scope(Deadline.from_timeout_ms(10_000)):
        result = da.analyze_paths(
            [target],
            forest=da.Forest(),
            recursive=True,
            type_audit=False,
            type_audit_report=False,
            type_audit_max=0,
            include_constant_smells=False,
            include_unused_arg_smells=False,
            include_bundle_forest=True,
            config=da.AuditConfig(
                project_root=tmp_path,
                exclude_dirs=set(),
                ignore_params=set(),
                external_filter=True,
                strictness="high",
            ),
        )
    assert result.forest is not None


def test_deadline_missing_carrier_for_loop(tmp_path: Path) -> None:
    obligations = _deadline_obligations(
        tmp_path,
        """
        def loop():
            for _ in range(1):
                pass
        """,
        roots={"mod.root"},
    )
    assert any(entry.get("kind") == "missing_carrier" for entry in obligations)


def test_deadline_none_arg_violation(tmp_path: Path) -> None:
    obligations = _deadline_obligations(
        tmp_path,
        """
        def callee(deadline: Deadline):
            return 1

        def root():
            callee(None)
        """,
        roots={"mod.root"},
    )
    assert any(entry.get("kind") == "none_arg" for entry in obligations)


def test_deadline_origin_not_allowlisted(tmp_path: Path) -> None:
    obligations = _deadline_obligations(
        tmp_path,
        """
        def callee(deadline: Deadline):
            return 1

        def helper():
            deadline = Deadline.from_timeout_ms(1_000)
            callee(deadline)
        """,
        roots={"mod.root"},
    )
    assert any(entry.get("kind") == "origin_not_allowlisted" for entry in obligations)
