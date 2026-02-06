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
    _has_typing_overload_import,
    _has_typing_protocol_import,
    _module_expr_to_str,
    _module_name,
    _rewrite_call_sites,
    _rewrite_call_sites_in_project,
)
from gabion.refactor.model import FieldSpec, RefactorRequest


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


def test_module_name_handles_value_error(tmp_path: Path) -> None:
    assert _module_name(Path("mod.py"), tmp_path / "other") == "mod"


def test_typing_import_helpers_negative_branches() -> None:
    module = cst.parse_module("from typing_extensions import Protocol\n")
    assert _has_typing_protocol_import(list(module.body)) is False
    module = cst.parse_module("from other import overload\n")
    assert _has_typing_overload_import(list(module.body)) is False
    expr = cst.Attribute(cst.Name("pkg"), cst.Name("mod"))
    assert _module_expr_to_str(expr) == "pkg.mod"


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


# gabion:evidence E:bundle/alias_invariance
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
    assert warnings
    assert edits == []


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
