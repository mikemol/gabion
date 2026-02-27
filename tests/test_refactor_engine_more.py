from __future__ import annotations

from pathlib import Path
import os
import textwrap

import libcst as cst

from gabion.refactor.engine import (
    RefactorEngine,
    _CallSiteTransformer,
    _RefactorTransformer,
    _collect_import_context,
    _ensure_compat_imports,
    _has_typing_import,
    _has_typing_overload_import,
    _has_typing_protocol_import,
    _has_warnings_import,
    _module_expr_to_str,
    _module_name,
    _rewrite_call_sites,
    _rewrite_call_sites_in_project,
)
from gabion.refactor.model import (
    CompatibilityShimConfig,
    FieldSpec,
    RefactorRequest,
)


# gabion:evidence E:call_footprint::tests/test_refactor_engine_more.py::test_plan_protocol_extraction_relative_path_and_fields::engine.py::gabion.refactor.engine.RefactorEngine::model.py::gabion.refactor.model.FieldSpec::model.py::gabion.refactor.model.RefactorRequest
def test_plan_protocol_extraction_relative_path_and_fields(tmp_path: Path) -> None:
    target = tmp_path / "mod.py"
    target.write_text("def f(a, b):\n    return a\n")
    request = RefactorRequest(
        protocol_name="Bundle",
        bundle=[],
        fields=[FieldSpec(name="a"), FieldSpec(name=""), FieldSpec(name="a"), FieldSpec(name="b")],
        target_path="mod.py",
    )
    plan = RefactorEngine(project_root=tmp_path).plan_protocol_extraction(request)
    assert plan.errors == []
    assert plan.edits


# gabion:evidence E:call_footprint::tests/test_refactor_engine_more.py::test_plan_protocol_extraction_typing_import_variants::engine.py::gabion.refactor.engine.RefactorEngine::model.py::gabion.refactor.model.RefactorRequest
def test_plan_protocol_extraction_typing_import_variants(tmp_path: Path) -> None:
    engine = RefactorEngine(project_root=tmp_path)

    mod_typing = tmp_path / "typing_mod.py"
    mod_typing.write_text("import typing\n\ndef f(a):\n    return a\n")
    plan_typing = engine.plan_protocol_extraction(
        RefactorRequest(
            protocol_name="Proto",
            bundle=["a"],
            target_path=str(mod_typing),
        )
    )
    assert plan_typing.edits
    assert "typing.Protocol" in plan_typing.edits[0].replacement

    mod_protocol = tmp_path / "protocol_mod.py"
    mod_protocol.write_text("from typing import Protocol\n\ndef f(a):\n    return a\n")
    plan_protocol = engine.plan_protocol_extraction(
        RefactorRequest(
            protocol_name="Proto",
            bundle=["a"],
            target_path=str(mod_protocol),
        )
    )
    assert plan_protocol.edits
    assert "class Proto(Protocol)" in plan_protocol.edits[0].replacement


# gabion:evidence E:decision_surface/direct::engine.py::gabion.refactor.engine._module_name::project_root E:decision_surface/direct::engine.py::gabion.refactor.engine._module_name::stale_7287d9dfaf67
def test_module_name_handles_value_error(tmp_path: Path) -> None:
    assert _module_name(Path("mod.py"), tmp_path / "other") == "mod"


# gabion:evidence E:decision_surface/direct::engine.py::gabion.refactor.engine._module_expr_to_str::expr E:decision_surface/direct::engine.py::gabion.refactor.engine._module_expr_to_str::stale_9cdf361906ed
def test_typing_import_helpers_negative_branches() -> None:
    module = cst.parse_module("from typing_extensions import Protocol\n")
    assert _has_typing_protocol_import(list(module.body)) is False
    module = cst.parse_module("from other import overload\n")
    assert _has_typing_overload_import(list(module.body)) is False
    expr = cst.Attribute(cst.Name("pkg"), cst.Name("mod"))
    assert _module_expr_to_str(expr) == "pkg.mod"


# gabion:evidence E:decision_surface/direct::engine.py::gabion.refactor.engine._find_import_insert_index::body E:decision_surface/direct::engine.py::gabion.refactor.engine._collect_import_context::protocol_name,target_module E:decision_surface/direct::engine.py::gabion.refactor.engine._rewrite_call_sites::target_module,targets E:decision_surface/direct::engine.py::gabion.refactor.engine._collect_import_context::stale_89918537e417
def test_rewrite_call_sites_uses_protocol_alias(tmp_path: Path) -> None:
    source = (
        "from pkg.mod import target, Bundle as PB\n"
        "\n"
        "def f(a, b):\n"
        "    target(a, b)\n"
    )
    module = cst.parse_module(source)
    warnings, updated = _rewrite_call_sites(
        module,
        file_path=tmp_path / "consumer.py",
        target_path=tmp_path / "target.py",
        target_module="pkg.mod",
        protocol_name="Bundle",
        bundle_fields=["a", "b"],
        targets={"target"},
    )
    assert warnings == []
    assert updated is not None
    assert "PB(" in updated.code


# gabion:evidence E:decision_surface/direct::engine.py::gabion.refactor.engine._module_expr_to_str::expr E:decision_surface/direct::engine.py::gabion.refactor.engine._collect_import_context::protocol_name,target_module E:decision_surface/direct::engine.py::gabion.refactor.engine._collect_import_context::stale_70b6cf24add5
def test_collect_import_context_skips_nonmatching() -> None:
    module = cst.Module(
        body=[
            cst.SimpleStatementLine(
                [cst.Import(names=[cst.ImportAlias(name=cst.Name("other"))])]
            ),
            cst.SimpleStatementLine(
                [
                    cst.ImportFrom(
                        module=cst.Name("pkg"),
                        names=[cst.ImportStar()],
                    )
                ]
            ),
            cst.SimpleStatementLine(
                [
                    cst.ImportFrom(
                        module=cst.Attribute(cst.Name("pkg"), cst.Name("mod")),
                        names=[
                            cst.ImportAlias(
                                name=cst.Attribute(cst.Name("pkg"), cst.Name("Thing"))
                            )
                        ],
                    )
                ]
            ),
        ]
    )
    module_aliases, imported_targets, protocol_alias = _collect_import_context(
        module, target_module="pkg.mod", protocol_name="Protocol"
    )
    assert module_aliases == {}
    assert imported_targets == {}
    assert protocol_alias is None


# gabion:evidence E:decision_surface/direct::engine.py::gabion.refactor.engine._module_expr_to_str::expr E:decision_surface/direct::engine.py::gabion.refactor.engine._collect_import_context::protocol_name,target_module E:decision_surface/direct::engine.py::gabion.refactor.engine._collect_import_context::stale_addb8d790246
def test_collect_import_context_skips_import_star_matching_module() -> None:
    module = cst.Module(
        body=[
            cst.SimpleStatementLine(
                [
                    cst.ImportFrom(
                        module=cst.Name("pkg"),
                        names=[cst.ImportStar()],
                    )
                ]
            )
        ]
    )
    module_aliases, imported_targets, protocol_alias = _collect_import_context(
        module, target_module="pkg", protocol_name="Protocol"
    )
    assert module_aliases == {}
    assert imported_targets == {}
    assert protocol_alias is None


# gabion:evidence E:decision_surface/direct::engine.py::gabion.refactor.engine._find_import_insert_index::body E:decision_surface/direct::engine.py::gabion.refactor.engine._collect_import_context::protocol_name,target_module E:decision_surface/direct::engine.py::gabion.refactor.engine._rewrite_call_sites::target_module,targets E:decision_surface/direct::engine.py::gabion.refactor.engine._collect_import_context::stale_f003341d3700
def test_rewrite_call_sites_empty_targets(tmp_path: Path) -> None:
    module = cst.parse_module("def f(a):\n    return a\n")
    warnings, updated = _rewrite_call_sites(
        module,
        file_path=tmp_path / "a.py",
        target_path=tmp_path / "a.py",
        target_module="pkg.mod",
        protocol_name="Bundle",
        bundle_fields=["a"],
        targets=set(),
    )
    assert warnings == []
    assert updated is None


# gabion:evidence E:decision_surface/direct::engine.py::gabion.refactor.engine._collect_import_context::protocol_name,target_module E:decision_surface/direct::engine.py::gabion.refactor.engine._find_import_insert_index::body E:decision_surface/direct::engine.py::gabion.refactor.engine._rewrite_call_sites::target_module,targets E:decision_surface/direct::engine.py::gabion.refactor.engine._collect_import_context::stale_64532b829a42
def test_rewrite_call_sites_module_alias_and_method_target(tmp_path: Path) -> None:
    source = (
        "import pkg.mod as pm\n"
        "class C:\n"
        "    def m(self, a, b):\n"
        "        return a\n"
        "    def caller(self, a, b):\n"
        "        return self.m(a, b)\n"
        "\n"
        "def f(a, b):\n"
        "    return pm.target(a, b)\n"
    )
    module = cst.parse_module(source)
    warnings, updated = _rewrite_call_sites(
        module,
        file_path=tmp_path / "consumer.py",
        target_path=tmp_path / "target.py",
        target_module="pkg.mod",
        protocol_name="Bundle",
        bundle_fields=["a", "b"],
        targets={"target", "C.m"},
    )
    assert updated is not None
    assert "pm.Bundle" in updated.code
    assert warnings == []


# gabion:evidence E:decision_surface/direct::engine.py::gabion.refactor.engine._rewrite_call_sites::target_module,targets E:decision_surface/direct::engine.py::gabion.refactor.engine._rewrite_call_sites_in_project::target_path E:decision_surface/direct::engine.py::gabion.refactor.engine._rewrite_call_sites::stale_1922c3368ac0
def test_rewrite_call_sites_in_project_read_errors(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    target = root / "target.py"
    target.write_text("def target(a, b):\n    return a + b\n")
    unreadable = root / "unreadable.py"
    unreadable.write_text("def nope():\n    return 1\n")
    os.chmod(unreadable, 0)
    consumer = root / "consumer.py"
    consumer.write_text("def f(a):\n    return a\n")
    edits, warnings = _rewrite_call_sites_in_project(
        project_root=tmp_path,
        target_path=target,
        target_module="target",
        protocol_name="Bundle",
        bundle_fields=["a", "b"],
        targets={"target"},
    )
    os.chmod(unreadable, 0o644)
    # Running as root can bypass chmod-based unreadable fixtures.
    assert edits == []


# gabion:evidence E:call_footprint::tests/test_refactor_engine_more.py::test_refactor_transformer_helpers::engine.py::gabion.refactor.engine._RefactorTransformer
def test_refactor_transformer_helpers() -> None:
    transformer = _RefactorTransformer(
        targets={"f"},
        bundle_fields=["a"],
        protocol_hint="Bundle[",
    )
    params = cst.parse_module("def f(self, a, /, b, *, c):\n    pass\n").body[0].params
    names = transformer._ordered_param_names(params)
    assert names == ["self", "a", "b", "c"]
    self_param = transformer._find_self_param(params, "self")
    assert self_param is not None
    assert transformer._find_self_param(params, "cls") is None
    assert transformer._choose_bundle_name(["bundle", "bundle_1"]) == "bundle_2"
    built = transformer._build_parameters(self_param, "bundle")
    assert built.params[0].name.value == "self"
    assert transformer.warnings
    shim_params = transformer._build_shim_parameters(self_param)
    assert shim_params.params[0].name.value == "self"

    suite = cst.IndentedBlock(
        body=[cst.SimpleStatementLine([cst.Expr(cst.SimpleString('"""Doc"""'))])]
    )
    injected = transformer._inject_preamble(suite, "bundle", ["a"])
    assert isinstance(injected, cst.IndentedBlock)

    passthrough = transformer._inject_preamble(suite, "bundle", [])
    assert passthrough is suite
    passthrough = transformer._inject_preamble(cst.SimpleStatementLine([]), "bundle", ["a"])
    assert passthrough is not None


# gabion:evidence E:call_footprint::tests/test_refactor_engine_more.py::test_refactor_transformer_async_and_no_params::engine.py::gabion.refactor.engine.RefactorEngine::model.py::gabion.refactor.model.RefactorRequest
def test_refactor_transformer_async_and_no_params(tmp_path: Path) -> None:
    target = tmp_path / "mod.py"
    target.write_text(
        textwrap.dedent(
            """
            async def f():
                return 1

            class C:
                async def m(self, a):
                    return a
            """
        ).strip()
        + "\n"
    )
    plan = RefactorEngine(project_root=tmp_path).plan_protocol_extraction(
        RefactorRequest(
            protocol_name="Bundle",
            bundle=["a"],
            target_path=str(target),
            target_functions=["f", "C.m"],
        )
    )
    assert plan.edits


# gabion:evidence E:call_footprint::tests/test_refactor_engine_more.py::test_call_site_transformer_helpers::engine.py::gabion.refactor.engine._CallSiteTransformer
def test_call_site_transformer_helpers() -> None:
    transformer = _CallSiteTransformer(
        file_is_target=True,
        target_simple={"target"},
        target_methods={"C": {"m"}},
        module_aliases=set(),
        imported_targets=set(),
        bundle_fields=["a", "b"],
        constructor_expr=cst.Attribute(cst.Name("mod"), cst.Name("Bundle")),
    )
    transformer._class_stack.append("C")
    call = cst.parse_expression("self.m(a, b)")
    assert transformer._is_target_call(call.func) is True
    transformer._class_stack.pop()

    transformer = _CallSiteTransformer(
        file_is_target=False,
        target_simple={"target"},
        target_methods={},
        module_aliases={"mod"},
        imported_targets=set(),
        bundle_fields=["a", "b"],
        constructor_expr=cst.Attribute(cst.Name("mod"), cst.Name("Bundle")),
    )
    call = cst.parse_expression("mod.target(a, b)")
    assert transformer._is_target_call(call.func) is True

    wrapped_call = cst.parse_expression("target(mod.Bundle(a=a, b=b))")
    assert transformer._already_wrapped(wrapped_call) is True

    name_constructor = _CallSiteTransformer(
        file_is_target=True,
        target_simple={"target"},
        target_methods={},
        module_aliases=set(),
        imported_targets=set(),
        bundle_fields=["a", "b"],
        constructor_expr=cst.Name("Bundle"),
    )
    mismatch_call = cst.parse_expression("target(mod.Bundle(a=a, b=b))")
    assert name_constructor._already_wrapped(mismatch_call) is False


# gabion:evidence E:call_footprint::tests/test_refactor_engine_more.py::test_engine_helper_negative_branches::engine.py::gabion.refactor.engine._collect_import_context::engine.py::gabion.refactor.engine._ensure_compat_imports::engine.py::gabion.refactor.engine._has_typing_import::engine.py::gabion.refactor.engine._has_typing_overload_import::engine.py::gabion.refactor.engine._has_typing_protocol_import::engine.py::gabion.refactor.engine._has_warnings_import::engine.py::gabion.refactor.engine._module_expr_to_str
def test_engine_helper_negative_branches() -> None:
    module = cst.parse_module("import os\nimport pkg.typing\n")
    assert _has_typing_import(list(module.body)) is False

    module = cst.parse_module("from typing import List\n")
    assert _has_typing_protocol_import(list(module.body)) is False
    assert _has_typing_overload_import(list(module.body)) is False
    module = cst.Module(
        body=[
            cst.SimpleStatementLine(
                [
                    cst.ImportFrom(
                        module=cst.Name("typing"),
                        names=[
                            cst.ImportAlias(
                                name=cst.Attribute(cst.Name("typing"), cst.Name("Protocol"))
                            )
                        ],
                    )
                ]
            )
        ]
    )
    assert _has_typing_protocol_import(list(module.body)) is False
    assert _has_typing_overload_import(list(module.body)) is False
    module = cst.parse_module("from typing import *\n")
    assert _has_typing_protocol_import(list(module.body)) is False
    assert _has_typing_overload_import(list(module.body)) is False

    module = cst.parse_module("import warnings\nfrom typing import overload\n")
    updated = _ensure_compat_imports(module, CompatibilityShimConfig(enabled=True))
    # Existing imports should not be duplicated.
    assert updated.code.count("import warnings") == 1
    assert updated.code.count("from typing import overload") == 1
    module = cst.parse_module("import pkg.warnings\n")
    assert _has_warnings_import(list(module.body)) is False

    expr = cst.parse_expression("pkg().mod")
    assert _module_expr_to_str(expr) == "mod"

    context_module = cst.parse_module("value = 1\n")
    aliases, imported, proto = _collect_import_context(
        context_module,
        target_module="pkg.mod",
        protocol_name="Protocol",
    )
    assert aliases == {}
    assert imported == {}
    assert proto is None
    context_module = cst.Module(
        body=[
            cst.SimpleStatementLine(
                [
                    cst.ImportFrom(
                        module=cst.Attribute(cst.Name("pkg"), cst.Name("mod")),
                        names=cst.ImportStar(),
                    )
                ]
            )
        ]
    )
    aliases, imported, proto = _collect_import_context(
        context_module,
        target_module="pkg.mod",
        protocol_name="Protocol",
    )
    assert aliases == {}
    assert imported == {}
    assert proto is None


# gabion:evidence E:call_footprint::tests/test_refactor_engine_more.py::test_refactor_and_callsite_transformer_stack_and_param_edges::engine.py::gabion.refactor.engine._CallSiteTransformer::engine.py::gabion.refactor.engine._RefactorTransformer
def test_refactor_and_callsite_transformer_stack_and_param_edges() -> None:
    transformer = _RefactorTransformer(
        targets={"f"},
        bundle_fields=["a"],
        protocol_hint="",
    )
    # Empty protocol hint exercises the no-annotation path.
    params = transformer._build_parameters(None, "bundle")
    assert params.params[0].annotation is None

    # Empty body exercises the branch where no first statement is inspected.
    empty_suite = cst.IndentedBlock(body=[])
    injected = transformer._inject_preamble(empty_suite, "bundle", ["a"])
    assert isinstance(injected, cst.IndentedBlock)

    # Non-simple-statement first node takes the fast path.
    non_simple = cst.IndentedBlock(
        body=[
            cst.If(
                test=cst.Name("cond"),
                body=cst.IndentedBlock(
                    body=[cst.SimpleStatementLine([cst.Pass()])]
                ),
            )
        ]
    )
    injected = transformer._inject_preamble(non_simple, "bundle", ["a"])
    assert isinstance(injected, cst.IndentedBlock)

    # leave_* guards with empty stacks.
    class_node = cst.ClassDef(name=cst.Name("C"), body=cst.IndentedBlock(body=[]))
    fn_node = cst.FunctionDef(
        name=cst.Name("f"),
        params=cst.Parameters(params=[]),
        body=cst.IndentedBlock(body=[cst.SimpleStatementLine([cst.Pass()])]),
    )
    assert transformer.leave_ClassDef(class_node, class_node) is class_node
    assert transformer.leave_FunctionDef(fn_node, fn_node) is fn_node

    call_transformer = _CallSiteTransformer(
        file_is_target=True,
        target_simple=set(),
        target_methods={"C": {"m"}},
        module_aliases=set(),
        imported_targets=set(),
        bundle_fields=["a"],
        constructor_expr=cst.Name("Bundle"),
    )
    call_transformer._class_stack.append("C")
    # attr in methods but receiver not in allowed receiver set.
    call = cst.parse_expression("obj.m(a)")
    assert call_transformer._is_target_call(call.func) is False
    miss_call = cst.parse_expression("self.n(a)")
    assert call_transformer._is_target_call(miss_call.func) is False
    other_call = cst.parse_expression("(factory[0])(a)")
    assert isinstance(other_call, cst.Call)
    assert call_transformer._is_target_call(other_call.func) is False
    call_transformer._class_stack.pop()
    # leave_ClassDef guard with empty stack.
    class_node = cst.ClassDef(name=cst.Name("C"), body=cst.IndentedBlock(body=[]))
    assert call_transformer.leave_ClassDef(class_node, class_node) is class_node


# gabion:evidence E:call_footprint::tests/test_refactor_engine_more.py::test_compat_shim_config_controls_imports_and_nodes::engine.py::gabion.refactor.engine.RefactorEngine
def test_compat_shim_config_controls_imports_and_nodes(tmp_path: Path) -> None:
    target = tmp_path / "mod.py"
    target.write_text("def target(a, b):\n    return a + b\n")

    plan = RefactorEngine(project_root=tmp_path).plan_protocol_extraction(
        RefactorRequest(
            protocol_name="Bundle",
            bundle=["a", "b"],
            target_path=str(target),
            target_functions=["target"],
            compatibility_shim=CompatibilityShimConfig(
                enabled=True,
                emit_deprecation_warning=False,
                emit_overload_stubs=True,
            ),
        )
    )

    assert plan.errors == []
    assert plan.edits
    updated = plan.edits[0].replacement
    assert "from typing import overload" in updated
    assert "import warnings" not in updated
    assert "@overload" in updated


# gabion:evidence E:decision_surface/direct::engine.py::gabion.refactor.engine._rewrite_call_sites::target_module,targets E:decision_surface/direct::engine.py::gabion.refactor.engine._rewrite_call_sites::stale_7fcf22f649f0
def test_compat_shim_legacy_wrapper_and_callsite_interop(tmp_path: Path) -> None:
    target = tmp_path / "target.py"
    target.write_text(
        textwrap.dedent(
            """
            def target(a, b):
                return a + b

            def use_legacy(a, b):
                return target(a, b)
            """
        ).strip()
        + "\n"
    )

    plan = RefactorEngine(project_root=tmp_path).plan_protocol_extraction(
        RefactorRequest(
            protocol_name="Bundle",
            bundle=["a", "b"],
            target_path=str(target),
            target_functions=["target"],
            compatibility_shim=CompatibilityShimConfig(
                enabled=True,
                emit_deprecation_warning=False,
                emit_overload_stubs=False,
            ),
        )
    )

    assert plan.errors == []
    transformed = plan.edits[0].replacement
    assert "def target(*args, **kwargs):" in transformed
    assert "if args and isinstance(args[0], Bundle):" in transformed
    assert "bundle = Bundle(*args, **kwargs)" in transformed
    assert "def _target_bundle(bundle: Bundle):" in transformed
    assert "target(Bundle(a = a, b = b))" in transformed
    assert "@overload" not in transformed
