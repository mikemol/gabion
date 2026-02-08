from __future__ import annotations

from pathlib import Path

from gabion.analysis.aspf import Forest
from gabion.analysis.timeout_context import (
    build_timeout_context_from_stack,
    pack_call_stack,
)


# gabion:evidence E:function_site::timeout_context.py::gabion.analysis.timeout_context.pack_call_stack
def test_pack_call_stack_orders_and_indexes() -> None:
    sites = [
        {"path": "b.py", "qual": "mod.b"},
        {"path": "a.py", "qual": "mod.a"},
        {"path": "b.py", "qual": "mod.b"},
    ]
    packed = pack_call_stack(sites)
    assert packed.site_table == [
        {"kind": "FunctionSite", "key": ["a.py", "mod.a"]},
        {"kind": "FunctionSite", "key": ["b.py", "mod.b"]},
    ]
    assert packed.stack == [1, 0, 1]


# gabion:evidence E:function_site::timeout_context.py::gabion.analysis.timeout_context.build_timeout_context_from_stack
def test_build_timeout_context_from_stack_uses_forest() -> None:
    forest = Forest()
    path_name = Path(__file__).name

    def outer() -> tuple[object, str]:
        def inner() -> object:
            return build_timeout_context_from_stack(
                forest=forest,
                project_root=Path(__file__).resolve().parents[1],
            )

        qual = f"{__name__}.{inner.__qualname__.replace('.<locals>.', '.')}"
        forest.add_site(path_name, qual)
        return inner(), qual

    context, qual = outer()
    sites = context.call_stack.site_table
    assert {"kind": "FunctionSite", "key": [path_name, qual]} in sites
