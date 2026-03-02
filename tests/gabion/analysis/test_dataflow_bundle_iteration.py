from __future__ import annotations

import ast
from pathlib import Path

import pytest

from gabion.analysis import dataflow_bundle_iteration as bundle_iteration
from gabion.exceptions import NeverThrown


def test_module_identifier_rejects_non_string_and_blank() -> None:
    with pytest.raises(ValueError):
        bundle_iteration._module_identifier(123)
    with pytest.raises(ValueError):
        bundle_iteration._module_identifier("   ")


def test_apply_constructor_plan_rejects_unknown_operation_kind(tmp_path: Path) -> None:
    call = ast.parse("C(1)").body[0].value
    assert isinstance(call, ast.Call)
    projection = bundle_iteration._apply_constructor_plan(
        path=tmp_path / "mod.py",
        call=call,
        fields=("a",),
        plan=bundle_iteration._ConstructorPlan(
            operations=(bundle_iteration._ConstructorOperation(kind="unknown"),),
            witness_effects=(),
            terminal_status="apply",
        ),
    )
    assert isinstance(projection, bundle_iteration._ConstructorProjectionRejected)
    assert projection.reason == "unknown_operation"


def test_iter_dataclass_call_bundle_effects_rejects_unexpected_projection_type(
    tmp_path: Path,
) -> None:
    source = tmp_path / "mod.py"
    source.write_text(
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class C:\n"
        "    a: int\n"
        "    b: int\n"
        "def f():\n"
        "    return C(1, 2)\n",
        encoding="utf-8",
    )

    original_apply = bundle_iteration._apply_constructor_plan
    bundle_iteration._apply_constructor_plan = lambda **_kwargs: object()
    try:
        with pytest.raises(NeverThrown):
            bundle_iteration.iter_dataclass_call_bundle_effects(
                source,
                project_root=tmp_path,
                symbol_table=None,
                dataclass_registry=None,
                parse_failure_witnesses=[],
            )
    finally:
        bundle_iteration._apply_constructor_plan = original_apply
