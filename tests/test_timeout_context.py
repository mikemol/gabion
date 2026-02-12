from __future__ import annotations

import inspect
from contextvars import Context
from pathlib import Path

import pytest

from gabion.analysis.aspf import Forest
from gabion.analysis.timeout_context import (
    Deadline,
    TimeoutContext,
    TimeoutExceeded,
    build_timeout_context_from_stack,
    build_site_index,
    check_deadline,
    deadline_profile_scope,
    deadline_scope,
    get_deadline,
    pack_call_stack,
    render_deadline_profile_markdown,
    set_deadline,
    _frame_site_key,
)
from gabion.exceptions import NeverThrown


# gabion:evidence E:function_site::timeout_context.py::gabion.analysis.timeout_context.pack_call_stack
def test_pack_call_stack_orders_and_indexes() -> None:
    sites = [
        {"path": "b.py", "qual": "mod.b"},
        {"path": "a.py", "qual": "mod.a"},
        {"path": "b.py", "qual": "mod.b"},
    ]
    packed = pack_call_stack(sites)
    assert packed.site_table == [
        {
            "kind": "FunctionSite",
            "key": [{"kind": "FileSite", "key": ["a.py"]}, "mod.a"],
        },
        {
            "kind": "FunctionSite",
            "key": [{"kind": "FileSite", "key": ["b.py"]}, "mod.b"],
        },
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
    assert {
        "kind": "FunctionSite",
        "key": [{"kind": "FileSite", "key": [path_name]}, qual],
    } in sites


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


def test_timeout_context_payload_includes_deadline_profile() -> None:
    packed = pack_call_stack([{"path": "a.py", "qual": "mod.fn"}])
    context = TimeoutContext(
        call_stack=packed,
        deadline_profile={"checks_total": 3},
    )
    payload = context.as_payload()
    assert payload["deadline_profile"] == {"checks_total": 3}


# gabion:evidence E:function_site::timeout_context.py::gabion.analysis.timeout_context.TimeoutExceeded.__init__
def test_timeout_exceeded_carries_context() -> None:
    packed = pack_call_stack([{"path": "a.py", "qual": "mod.fn"}])
    context = TimeoutContext(call_stack=packed)
    exc = TimeoutExceeded(context)
    assert "timed out" in str(exc)
    assert exc.context is context


# gabion:evidence E:function_site::timeout_context.py::gabion.analysis.timeout_context.Deadline.from_timeout_ms
# gabion:evidence E:function_site::timeout_context.py::gabion.analysis.timeout_context.Deadline.check
def test_deadline_expired_raises() -> None:
    deadline = Deadline.from_timeout_ms(0)
    assert deadline.expired() is True
    with pytest.raises(TimeoutExceeded):
        deadline.check(lambda: TimeoutContext(call_stack=pack_call_stack([])))


# gabion:evidence E:function_site::timeout_context.py::gabion.analysis.timeout_context.Deadline.from_timeout
# gabion:evidence E:function_site::timeout_context.py::gabion.analysis.timeout_context.Deadline.from_timeout_ticks
def test_deadline_from_timeout_variants() -> None:
    with pytest.raises(NeverThrown):
        Deadline.from_timeout_ticks(-5, 0)
    with pytest.raises(NeverThrown):
        Deadline.from_timeout(-1)
    with pytest.raises(NeverThrown):
        Deadline.from_timeout("nope")


# gabion:evidence E:function_site::timeout_context.py::gabion.analysis.timeout_context.build_site_index
def test_build_site_index_filters_nodes() -> None:
    forest = Forest()
    forest.add_node("Param", ("x",), {"name": "x"})
    forest.add_site("", "mod.missing")
    forest.add_site("ok.py", "mod.ok")
    index = build_site_index(forest)
    assert list(index.keys()) == [("ok.py", "mod.ok")]


# gabion:evidence E:function_site::timeout_context.py::gabion.analysis.timeout_context.pack_call_stack
def test_pack_call_stack_keeps_span() -> None:
    packed = pack_call_stack(
        [
            {"path": "a.py", "qual": "mod.fn", "span": [1, 2, 3, 4]},
            {"kind": "FunctionSite", "key": ["b.py", "mod.other"]},
        ]
    )
    assert {
        "kind": "FunctionSite",
        "key": [{"kind": "FileSite", "key": ["a.py"]}, "mod.fn", 1, 2, 3, 4],
    } in packed.site_table
    assert {
        "kind": "FunctionSite",
        "key": [{"kind": "FileSite", "key": ["b.py"]}, "mod.other"],
    } in packed.site_table


def test_pack_call_stack_rejects_invalid_entries() -> None:
    with pytest.raises(NeverThrown):
        pack_call_stack([{"path": "", "qual": "missing"}])


def test_pack_call_stack_accepts_list_key_part() -> None:
    packed = pack_call_stack(
        [{"kind": "FunctionSite", "key": [["file"], "mod.fn"]}]
    )
    assert packed.site_table == [
        {"kind": "FunctionSite", "key": [["file"], "mod.fn"]}
    ]


# gabion:evidence E:function_site::timeout_context.py::gabion.analysis.timeout_context._frame_site_key
def test_frame_site_key_outside_root_raises() -> None:
    frame = inspect.currentframe()
    assert frame is not None
    project_root = Path(__file__).resolve().parents[1] / "missing_root"
    with pytest.raises(NeverThrown):
        _frame_site_key(frame, project_root=project_root)


# gabion:evidence E:function_site::timeout_context.py::gabion.analysis.timeout_context._frame_site_key
def test_frame_site_key_without_module_name() -> None:
    frame = type(
        "DummyFrame",
        (),
        {
            "f_globals": {"__name__": ""},
            "f_code": type(
                "DummyCode",
                (),
                {
                    "co_qualname": "inner",
                    "co_name": "inner",
                    "co_filename": "mod.py",
                },
            )(),
        },
    )()
    assert _frame_site_key(frame, project_root=None) == ("mod.py", "inner")


def test_set_deadline_rejects_none() -> None:
    with pytest.raises(NeverThrown):
        set_deadline(None)  # type: ignore[arg-type]


def test_deadline_scope_rejects_none() -> None:
    with pytest.raises(NeverThrown):
        with deadline_scope(None):  # type: ignore[arg-type]
            pass


def test_get_deadline_requires_carrier() -> None:
    ctx = Context()
    with pytest.raises(NeverThrown):
        ctx.run(get_deadline)


# gabion:evidence E:function_site::timeout_context.py::gabion.analysis.timeout_context.build_timeout_context_from_stack
def test_build_timeout_context_frame_fallback() -> None:
    frame = inspect.currentframe()
    assert frame is not None
    context = build_timeout_context_from_stack(
        forest=Forest(),
        project_root=Path(__file__).resolve().parents[1],
        allow_frame_fallback=True,
        frames=[frame],
    )
    assert context.call_stack.site_table


def test_deadline_profile_scope_collects_heat() -> None:
    root = Path(__file__).resolve().parents[1]
    with deadline_profile_scope(project_root=root, enabled=True):
        with deadline_scope(Deadline.from_timeout_ms(100)):
            check_deadline()
            check_deadline()
            frame = inspect.currentframe()
            assert frame is not None
            context = build_timeout_context_from_stack(
                forest=Forest(),
                project_root=root,
                allow_frame_fallback=True,
                frames=[frame],
            )
    profile = context.deadline_profile
    assert isinstance(profile, dict)
    assert int(profile.get("checks_total", 0) or 0) >= 2
    assert isinstance(profile.get("sites", []), list)


def test_render_deadline_profile_markdown() -> None:
    profile = {
        "checks_total": 2,
        "total_elapsed_ns": 100,
        "unattributed_elapsed_ns": 10,
        "sites": [
            {
                "path": "a.py",
                "qual": "mod.fn",
                "check_count": 2,
                "elapsed_between_checks_ns": 90,
                "max_gap_ns": 70,
            }
        ],
        "edges": [
            {
                "from_path": "a.py",
                "from_qual": "mod.fn",
                "to_path": "a.py",
                "to_qual": "mod.fn",
                "transition_count": 1,
                "elapsed_ns": 80,
                "max_gap_ns": 80,
            }
        ],
    }
    rendered = render_deadline_profile_markdown(profile)
    assert "Deadline Profile Heat" in rendered
    assert "Site Heat" in rendered
    assert "Transition Heat" in rendered


# gabion:evidence E:function_site::timeout_context.py::gabion.analysis.timeout_context.build_timeout_context_from_stack
def test_build_timeout_context_skips_unmatched_frames() -> None:
    frame = inspect.currentframe()
    assert frame is not None
    context = build_timeout_context_from_stack(
        forest=Forest(),
        project_root=Path(__file__).resolve().parents[1] / "missing_root",
        allow_frame_fallback=False,
        frames=[frame],
    )
    assert context.call_stack.site_table == []
