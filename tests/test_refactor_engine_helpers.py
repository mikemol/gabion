from __future__ import annotations

from pathlib import Path

import libcst as cst

from gabion.refactor import RefactorCompatibilityShimConfig, engine as refactor_engine


# gabion:evidence E:decision_surface/direct::engine.py::gabion.refactor.engine._find_import_insert_index::body E:decision_surface/direct::engine.py::gabion.refactor.engine._module_expr_to_str::expr E:decision_surface/direct::engine.py::gabion.refactor.engine._is_docstring::stmt E:decision_surface/direct::engine.py::gabion.refactor.engine._is_import::stmt
def test_import_helpers_and_insert_index() -> None:
    module = cst.parse_module('"""Doc"""\nimport typing\nfrom typing import Protocol\n')
    body = list(module.body)
    assert refactor_engine._is_docstring(body[0]) is True
    assert refactor_engine._is_import(body[1]) is True
    assert refactor_engine._find_import_insert_index(body) == 3
    assert refactor_engine._has_typing_import(body) is True
    assert refactor_engine._has_typing_protocol_import(body) is True
    assert refactor_engine._has_typing_overload_import(body) is False
    assert refactor_engine._has_warnings_import(body) is False

    module = refactor_engine._ensure_compat_imports(
        module, RefactorCompatibilityShimConfig(enabled=True)
    )
    new_body = list(module.body)
    assert refactor_engine._has_typing_overload_import(new_body) is True
    assert refactor_engine._has_warnings_import(new_body) is True


# gabion:evidence E:decision_surface/direct::engine.py::gabion.refactor.engine._module_expr_to_str::expr
def test_module_expr_to_str() -> None:
    assert refactor_engine._module_expr_to_str(cst.Name("typing")) == "typing"
    expr = cst.Attribute(cst.Name("a"), cst.Name("b"))
    assert refactor_engine._module_expr_to_str(expr) == "a.b"
    assert refactor_engine._module_expr_to_str(None) is None


# gabion:evidence E:decision_surface/direct::engine.py::gabion.refactor.engine._module_expr_to_str::expr E:decision_surface/direct::engine.py::gabion.refactor.engine._collect_import_context::protocol_name,target_module
def test_collect_import_context() -> None:
    module = cst.parse_module(
        "import pkg.mod as pm\n"
        "from pkg.mod import Target as Alias, Protocol as Proto\n"
        "from other import Thing\n"
    )
    module_aliases, imported_targets, protocol_alias = refactor_engine._collect_import_context(
        module, target_module="pkg.mod", protocol_name="Protocol"
    )
    assert module_aliases == {"pm": "pkg.mod"}
    assert imported_targets == {"Alias": "Target", "Proto": "Protocol"}
    assert protocol_alias == "Proto"


# gabion:evidence E:decision_surface/direct::engine.py::gabion.refactor.engine._find_import_insert_index::body E:decision_surface/direct::engine.py::gabion.refactor.engine._collect_import_context::protocol_name,target_module E:decision_surface/direct::engine.py::gabion.refactor.engine._rewrite_call_sites::target_module,targets
def test_rewrite_call_sites_target_and_imports(tmp_path: Path) -> None:
    source = (
        "from pkg.mod import target\n"
        "def f(a, b):\n"
        "    target(a, b)\n"
        "    target(a)\n"
        "    target(a, b, c)\n"
        "    target(a, c=1)\n"
        "    target(a, b=b)\n"
        "    target(Bundle(a=a, b=b))\n"
        "    target(*args)\n"
    )
    module = cst.parse_module(source)
    warnings, updated = refactor_engine._rewrite_call_sites(
        module,
        file_path=tmp_path / "consumer.py",
        target_path=tmp_path / "consumer.py",
        target_module="pkg.mod",
        protocol_name="Bundle",
        bundle_fields=["a", "b"],
        targets={"target"},
    )
    assert updated is not None
    assert any("extra positional" in w for w in warnings)
    assert any("unknown keyword" in w for w in warnings)
    assert any("missing bundle fields" in w for w in warnings)
    assert any("star args" in w for w in warnings)

    source = "from pkg.mod import target\n\ndef f(a, b):\n    target(a, b)\n"
    module = cst.parse_module(source)
    warnings, updated = refactor_engine._rewrite_call_sites(
        module,
        file_path=tmp_path / "other.py",
        target_path=tmp_path / "target.py",
        target_module="pkg.mod",
        protocol_name="Bundle",
        bundle_fields=["a", "b"],
        targets={"target"},
    )
    assert updated is not None
    assert "from pkg.mod import Bundle" in updated.code


# gabion:evidence E:decision_surface/direct::engine.py::gabion.refactor.engine._rewrite_call_sites::target_module,targets E:decision_surface/direct::engine.py::gabion.refactor.engine._rewrite_call_sites_in_project::target_path
def test_rewrite_call_sites_in_project(tmp_path: Path) -> None:
    target = tmp_path / "src" / "target.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("def target(a, b):\n    return a + b\n")
    bad = tmp_path / "src" / "bad.py"
    bad.write_text("def oops(\n")
    consumer = tmp_path / "src" / "consumer.py"
    consumer.write_text("from target import target\n\ndef f(a, b):\n    target(a, b)\n")
    edits, warnings = refactor_engine._rewrite_call_sites_in_project(
        project_root=tmp_path,
        target_path=target,
        target_module="target",
        protocol_name="Bundle",
        bundle_fields=["a", "b"],
        targets={"target"},
    )
    assert warnings
    assert edits
