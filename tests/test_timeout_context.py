from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from gabion.analysis.aspf import Forest
from gabion.analysis.timeout_context import (
    Deadline,
    TimeoutContext,
    TimeoutExceeded,
    build_timeout_context_from_stack,
    build_site_index,
    pack_call_stack,
    _frame_site_key,
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


# gabion:evidence E:function_site::timeout_context.py::gabion.analysis.timeout_context.TimeoutContext.as_payload
def test_timeout_context_payload_includes_metadata() -> None:
    packed = pack_call_stack([{"path": "a.py", "qual": "mod.fn"}])
    context = TimeoutContext(
        call_stack=packed,
        forest_spec_id="spec",
        forest_signature={"version": 1},
    )
    payload = context.as_payload()
    assert payload["forest_spec_id"] == "spec"
    assert payload["forest_signature"] == {"version": 1}


# gabion:evidence E:function_site::timeout_context.py::gabion.analysis.timeout_context.TimeoutExceeded.__init__
def test_timeout_exceeded_carries_context() -> None:
    packed = pack_call_stack([{"path": "a.py", "qual": "mod.fn"}])
    context = TimeoutContext(call_stack=packed)
    exc = TimeoutExceeded(context)
    assert "timed out" in str(exc)
    assert exc.context is context


# gabion:evidence E:function_site::timeout_context.py::gabion.analysis.timeout_context.Deadline.from_timeout
# gabion:evidence E:function_site::timeout_context.py::gabion.analysis.timeout_context.Deadline.check
def test_deadline_expired_raises() -> None:
    deadline = Deadline.from_timeout(0.0)
    assert deadline.expired() is True
    with pytest.raises(TimeoutExceeded):
        deadline.check(lambda: TimeoutContext(call_stack=pack_call_stack([])))


# gabion:evidence E:function_site::timeout_context.py::gabion.analysis.timeout_context.build_site_index
def test_build_site_index_filters_nodes() -> None:
    forest = Forest()
    forest.add_node("Param", ("x",), {"name": "x"})
    forest.add_site("", "mod.missing")
    forest.add_site("ok.py", "mod.ok")
    index = build_site_index(forest)
    assert list(index.keys()) == [("ok.py", "mod.ok")]


# gabion:evidence E:function_site::timeout_context.py::gabion.analysis.timeout_context.pack_call_stack
def test_pack_call_stack_skips_invalid_entries_and_keeps_span() -> None:
    packed = pack_call_stack(
        [
            {"path": "", "qual": "missing"},
            {"path": "a.py", "qual": "mod.fn", "span": [1, 2, 3, 4]},
            {"kind": "FunctionSite", "key": ["b.py", "mod.other"]},
        ]
    )
    assert {"kind": "FunctionSite", "key": ["a.py", "mod.fn", 1, 2, 3, 4]} in packed.site_table
    assert {"kind": "FunctionSite", "key": ["b.py", "mod.other"]} in packed.site_table


# gabion:evidence E:function_site::timeout_context.py::gabion.analysis.timeout_context._frame_site_key
def test_frame_site_key_outside_root_returns_none() -> None:
    frame = inspect.currentframe()
    assert frame is not None
    project_root = Path(__file__).resolve().parents[1] / "missing_root"
    assert _frame_site_key(frame, project_root=project_root) is None


# gabion:evidence E:function_site::timeout_context.py::gabion.analysis.timeout_context.build_timeout_context_from_stack
def test_build_timeout_context_frame_fallback() -> None:
    frame = inspect.currentframe()
    assert frame is not None
    context = build_timeout_context_from_stack(
        forest=None,
        project_root=Path(__file__).resolve().parents[1],
        allow_frame_fallback=True,
        frames=[frame],
    )
    assert context.call_stack.site_table


# gabion:evidence E:function_site::timeout_context.py::gabion.analysis.timeout_context.build_timeout_context_from_stack
def test_build_timeout_context_skips_unmatched_frames() -> None:
    frame = inspect.currentframe()
    assert frame is not None
    context = build_timeout_context_from_stack(
        forest=None,
        project_root=Path(__file__).resolve().parents[1] / "missing_root",
        allow_frame_fallback=False,
        frames=[frame],
    )
    assert context.call_stack.site_table == []
