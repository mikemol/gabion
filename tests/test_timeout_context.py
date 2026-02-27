from __future__ import annotations

import inspect
from contextlib import contextmanager
from contextvars import Context
from pathlib import Path

import pytest

import gabion.analysis.timeout_context as timeout_context
from gabion.analysis.aspf import Forest
from gabion.analysis.timeout_context import (
    Deadline,
    GasMeter,
    TimeoutContext,
    TimeoutExceeded,
    _deadline_profile_snapshot,
    _freeze_value,
    _profile_site_key,
    _record_deadline_check,
    _site_key_payload,
    _site_part_from_payload,
    _site_part_to_payload,
    _timeout_progress_snapshot,
    build_timeout_context_from_stack,
    build_site_index,
    check_deadline,
    consume_deadline_ticks,
    deadline_clock_scope,
    deadline_profile_scope,
    deadline_scope,
    forest_scope,
    get_deadline_clock,
    get_deadline,
    get_forest,
    pack_call_stack,
    record_deadline_io,
    render_deadline_profile_markdown,
    reset_forest,
    set_deadline,
    set_deadline_clock,
    set_forest,
    _frame_site_key,
)
from gabion.deadline_clock import DeadlineClockExhausted
from gabion.exceptions import NeverThrown


@contextmanager
def _deadline_test_scope(
    *,
    deadline: Deadline,
    clock: object | None = None,
):
    with forest_scope(Forest()):
        with deadline_scope(deadline):
            if clock is None:
                yield
            else:
                with deadline_clock_scope(clock):
                    yield


# gabion:evidence E:function_site::timeout_context.py::gabion.analysis.timeout_context.pack_call_stack E:decision_surface/direct::timeout_context.py::gabion.analysis.timeout_context.pack_call_stack::stale_5f83d4767c39
def test_pack_call_stack_orders_and_indexes() -> None:
    sites = [
        {"path": "b.py", "qual": "mod.b"},
        {"path": "a.py", "qual": "mod.a"},
        {"path": "b.py", "qual": "mod.b"},
    ]
    packed = pack_call_stack(sites)
    assert packed.as_payload()["site_table"] == [
        {
            "kind": "FunctionSite",
            "key": [{"kind": "FileSite", "key": ["b.py"]}, "mod.b"],
        },
        {
            "kind": "FunctionSite",
            "key": [{"kind": "FileSite", "key": ["a.py"]}, "mod.a"],
        },
    ]
    assert packed.stack == (0, 1, 0)


# gabion:evidence E:function_site::timeout_context.py::gabion.analysis.timeout_context.build_timeout_context_from_stack E:decision_surface/direct::timeout_context.py::gabion.analysis.timeout_context.build_timeout_context_from_stack::stale_dbe32a0500ec
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
    sites = context.call_stack.as_payload()["site_table"]
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


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_timeout_context_payload_includes_deadline_profile::timeout_context.py::gabion.analysis.timeout_context.pack_call_stack
def test_timeout_context_payload_includes_deadline_profile() -> None:
    packed = pack_call_stack([{"path": "a.py", "qual": "mod.fn"}])
    context = TimeoutContext(
        call_stack=packed,
        deadline_profile={"checks_total": 3},
    )
    payload = context.as_payload()
    assert payload["deadline_profile"] == {"checks_total": 3}


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_timeout_context_payload_includes_progress::timeout_context.py::gabion.analysis.timeout_context.pack_call_stack
def test_timeout_context_payload_includes_progress() -> None:
    packed = pack_call_stack([{"path": "a.py", "qual": "mod.fn"}])
    context = TimeoutContext(
        call_stack=packed,
        progress={"classification": "timed_out_progress_resume"},
    )
    payload = context.as_payload()
    assert payload["progress"] == {"classification": "timed_out_progress_resume"}


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


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_deadline_check_noop_when_not_expired::timeout_context.py::gabion.analysis.timeout_context.Deadline.from_timeout_ms::timeout_context.py::gabion.analysis.timeout_context.pack_call_stack
def test_deadline_check_noop_when_not_expired() -> None:
    deadline = Deadline.from_timeout_ms(1_000)
    assert deadline.expired() is False
    deadline.check(lambda: TimeoutContext(call_stack=pack_call_stack([])))
    assert deadline.expired() is False


# gabion:evidence E:function_site::timeout_context.py::gabion.analysis.timeout_context.Deadline.from_timeout
# gabion:evidence E:function_site::timeout_context.py::gabion.analysis.timeout_context.Deadline.from_timeout_ticks
def test_deadline_from_timeout_variants() -> None:
    with pytest.raises(NeverThrown):
        Deadline.from_timeout_ticks(-5, 0)
    with pytest.raises(NeverThrown):
        Deadline.from_timeout_ticks(1, 0)
    with pytest.raises(NeverThrown):
        Deadline.from_timeout(-1)
    with pytest.raises(NeverThrown):
        Deadline.from_timeout("nope")
    assert isinstance(Deadline.from_timeout(0.001), Deadline)


# gabion:evidence E:function_site::timeout_context.py::gabion.analysis.timeout_context.build_site_index
def test_build_site_index_filters_nodes() -> None:
    forest = Forest()
    forest.add_node("Param", ("x",), {"name": "x"})
    forest.add_site("", "mod.missing")
    forest.add_site("ok.py", "mod.ok")
    index = build_site_index(forest)
    assert list(index.keys()) == [("ok.py", "mod.ok")]


# gabion:evidence E:function_site::timeout_context.py::gabion.analysis.timeout_context.pack_call_stack E:decision_surface/direct::timeout_context.py::gabion.analysis.timeout_context.pack_call_stack::stale_1a892a545319_b666ebce
def test_pack_call_stack_keeps_span() -> None:
    packed = pack_call_stack(
        [
            {"path": "a.py", "qual": "mod.fn", "span": [1, 2, 3, 4]},
            {"kind": "FunctionSite", "key": ["b.py", "mod.other"]},
        ]
    )
    site_table = packed.as_payload()["site_table"]
    assert {
        "kind": "FunctionSite",
        "key": [{"kind": "FileSite", "key": ["a.py"]}, "mod.fn", 1, 2, 3, 4],
    } in site_table
    assert {
        "kind": "FunctionSite",
        "key": [{"kind": "FileSite", "key": ["b.py"]}, "mod.other"],
    } in site_table


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_pack_call_stack_rejects_invalid_entries::timeout_context.py::gabion.analysis.timeout_context.pack_call_stack
def test_pack_call_stack_rejects_invalid_entries() -> None:
    with pytest.raises(NeverThrown):
        pack_call_stack([{"path": "", "qual": "missing"}])


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_pack_call_stack_accepts_list_key_part::timeout_context.py::gabion.analysis.timeout_context.pack_call_stack
def test_pack_call_stack_accepts_list_key_part() -> None:
    packed = pack_call_stack(
        [{"kind": "FunctionSite", "key": [["file"], "mod.fn"]}]
    )
    assert packed.as_payload()["site_table"] == [
        {"kind": "FunctionSite", "key": [["file"], "mod.fn"]}
    ]


def test_function_site_identity_payload_round_trip() -> None:
    packed = pack_call_stack(
        [
            {
                "kind": "FunctionSite",
                "key": [{"kind": "FileSite", "key": ["a.py"]}, "mod.fn", 1, 2, 3, 4],
            }
        ]
    )
    payload = packed.as_payload()
    restored = pack_call_stack(payload["site_table"])
    assert restored.as_payload()["site_table"] == payload["site_table"]


def test_function_site_identity_decode_rejects_partial_identity() -> None:
    with pytest.raises(NeverThrown):
        pack_call_stack([{"path": "a.py", "qual": "", "span": [1, 2, 3]}])


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_pack_call_stack_uses_first_seen_site_order::timeout_context.py::gabion.analysis.timeout_context.pack_call_stack
def test_pack_call_stack_uses_first_seen_site_order() -> None:
    packed = pack_call_stack(
        [
            {"path": "z.py", "qual": "mod.z"},
            {"path": "a.py", "qual": "mod.a"},
            {"path": "z.py", "qual": "mod.z"},
        ]
    )
    assert packed.as_payload()["site_table"] == [
        {
            "kind": "FunctionSite",
            "key": [{"kind": "FileSite", "key": ["z.py"]}, "mod.z"],
        },
        {
            "kind": "FunctionSite",
            "key": [{"kind": "FileSite", "key": ["a.py"]}, "mod.a"],
        },
    ]
    assert packed.stack == (0, 1, 0)


# gabion:evidence E:function_site::timeout_context.py::gabion.analysis.timeout_context._frame_site_key E:decision_surface/direct::timeout_context.py::gabion.analysis.timeout_context._frame_site_key::stale_973f9f2d7f85_f67ec520
def test_frame_site_key_outside_root_raises() -> None:
    frame = inspect.currentframe()
    assert frame is not None
    project_root = Path(__file__).resolve().parents[1] / "missing_root"
    with pytest.raises(NeverThrown):
        _frame_site_key(frame, project_root=project_root)


# gabion:evidence E:function_site::timeout_context.py::gabion.analysis.timeout_context._frame_site_key E:decision_surface/direct::timeout_context.py::gabion.analysis.timeout_context._frame_site_key::stale_88f48c1fe168
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


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_set_deadline_rejects_none::timeout_context.py::gabion.analysis.timeout_context.set_deadline
def test_set_deadline_rejects_none() -> None:
    with pytest.raises(NeverThrown):
        set_deadline(None)  # type: ignore[arg-type]


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_set_deadline_clock_rejects_none::timeout_context.py::gabion.analysis.timeout_context.set_deadline_clock
def test_set_deadline_clock_rejects_none() -> None:
    with pytest.raises(NeverThrown):
        set_deadline_clock(None)  # type: ignore[arg-type]


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_deadline_scope_rejects_none::timeout_context.py::gabion.analysis.timeout_context.deadline_scope
def test_deadline_scope_rejects_none() -> None:
    with pytest.raises(NeverThrown):
        with deadline_scope(None):  # type: ignore[arg-type]
            pass


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_get_deadline_requires_carrier::exceptions.py::gabion.exceptions.NeverThrown::timeout_context.py::gabion.analysis.timeout_context.get_deadline
def test_get_deadline_requires_carrier() -> None:
    ctx = Context()
    with pytest.raises(NeverThrown):
        ctx.run(get_deadline)


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_get_deadline_clock_requires_carrier::exceptions.py::gabion.exceptions.NeverThrown::timeout_context.py::gabion.analysis.timeout_context.get_deadline_clock
def test_get_deadline_clock_requires_carrier() -> None:
    ctx = Context()
    with pytest.raises(NeverThrown):
        ctx.run(get_deadline_clock)


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_check_deadline_requires_clock_scope::timeout_context.py::gabion.analysis.timeout_context.Deadline.from_timeout_ms::timeout_context.py::gabion.analysis.timeout_context.check_deadline::timeout_context.py::gabion.analysis.timeout_context.deadline_scope::timeout_context.py::gabion.analysis.timeout_context.forest_scope
def test_check_deadline_requires_clock_scope() -> None:
    ctx = Context()

    def _run() -> None:
        with forest_scope(Forest()):
            with deadline_scope(Deadline.from_timeout_ms(100)):
                check_deadline()

    with pytest.raises(NeverThrown):
        ctx.run(_run)


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_get_forest_requires_valid_carrier::timeout_context.py::gabion.analysis.timeout_context.get_forest::timeout_context.py::gabion.analysis.timeout_context.reset_forest::timeout_context.py::gabion.analysis.timeout_context.set_forest
def test_get_forest_requires_valid_carrier() -> None:
    ctx = Context()
    with pytest.raises(NeverThrown):
        ctx.run(get_forest)
    token = set_forest("bad")  # type: ignore[arg-type]
    try:
        with pytest.raises(NeverThrown):
            get_forest()
    finally:
        reset_forest(token)


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_get_forest_returns_active_carrier::timeout_context.py::gabion.analysis.timeout_context.get_forest::timeout_context.py::gabion.analysis.timeout_context.reset_forest::timeout_context.py::gabion.analysis.timeout_context.set_forest
def test_get_forest_returns_active_carrier() -> None:
    forest = Forest()
    token = set_forest(forest)
    try:
        assert get_forest() is forest
    finally:
        reset_forest(token)


# gabion:evidence E:function_site::timeout_context.py::gabion.analysis.timeout_context.build_timeout_context_from_stack E:decision_surface/direct::timeout_context.py::gabion.analysis.timeout_context.build_timeout_context_from_stack::stale_253a380d6acc_563ed243
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


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_build_timeout_context_progress_classification_no_progress::timeout_context.py::gabion.analysis.timeout_context.build_timeout_context_from_stack
def test_build_timeout_context_progress_classification_no_progress() -> None:
    frame = inspect.currentframe()
    assert frame is not None
    context = build_timeout_context_from_stack(
        forest=Forest(),
        project_root=None,
        frames=[frame],
        deadline_profile={"checks_total": 0, "sites": []},
    )
    progress = context.as_payload().get("progress")
    assert isinstance(progress, dict)
    assert progress.get("classification") == "timed_out_no_progress"
    assert progress.get("retry_recommended") is False


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_build_timeout_context_progress_classification_with_progress::timeout_context.py::gabion.analysis.timeout_context.build_timeout_context_from_stack
def test_build_timeout_context_progress_classification_with_progress() -> None:
    frame = inspect.currentframe()
    assert frame is not None
    forest = Forest()
    forest.add_node("Param", ("x",), {"name": "x"})
    context = build_timeout_context_from_stack(
        forest=forest,
        project_root=None,
        frames=[frame],
        deadline_profile={"checks_total": 0, "sites": []},
    )
    progress = context.as_payload().get("progress")
    assert isinstance(progress, dict)
    assert progress.get("classification") == "timed_out_progress_resume"
    assert progress.get("retry_recommended") is True


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_deadline_profile_scope_collects_heat::timeout_context.py::gabion.analysis.timeout_context.Deadline.from_timeout_ms::timeout_context.py::gabion.analysis.timeout_context.build_timeout_context_from_stack::timeout_context.py::gabion.analysis.timeout_context.check_deadline::timeout_context.py::gabion.analysis.timeout_context.deadline_profile_scope::timeout_context.py::gabion.analysis.timeout_context.deadline_scope::timeout_context.py::gabion.analysis.timeout_context.record_deadline_io
def test_deadline_profile_scope_collects_heat() -> None:
    root = Path(__file__).resolve().parents[1]
    with deadline_profile_scope(project_root=root, enabled=True):
        with deadline_scope(Deadline.from_timeout_ms(100)):
            check_deadline()
            check_deadline()
            record_deadline_io(
                name="analysis_resume.checkpoint_write",
                elapsed_ns=5_000_000,
                bytes_count=4096,
            )
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
    assert int(profile.get("checks_total", 0) or 0) == 2
    assert isinstance(profile.get("sites", []), list)
    io_rows = profile.get("io")
    assert isinstance(io_rows, list)
    assert any(
        isinstance(entry, dict)
        and entry.get("name") == "analysis_resume.checkpoint_write"
        for entry in io_rows
    )


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_check_deadline_uses_gas_meter_ticks::test_timeout_context.py::tests.test_timeout_context._deadline_test_scope::timeout_context.py::gabion.analysis.timeout_context.Deadline.from_timeout_ms::timeout_context.py::gabion.analysis.timeout_context.check_deadline
def test_check_deadline_uses_gas_meter_ticks() -> None:
    with _deadline_test_scope(
        deadline=Deadline.from_timeout_ms(1_000),
        clock=GasMeter(limit=2),
    ):
        check_deadline()
        with pytest.raises(TimeoutExceeded):
            check_deadline()


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_deadline_profile_uses_logical_ticks_when_clock_injected::timeout_context.py::gabion.analysis.timeout_context.Deadline.from_timeout_ms::timeout_context.py::gabion.analysis.timeout_context._deadline_profile_snapshot::timeout_context.py::gabion.analysis.timeout_context.check_deadline::timeout_context.py::gabion.analysis.timeout_context.deadline_clock_scope::timeout_context.py::gabion.analysis.timeout_context.deadline_profile_scope::timeout_context.py::gabion.analysis.timeout_context.deadline_scope::timeout_context.py::gabion.analysis.timeout_context.forest_scope
def test_deadline_profile_uses_logical_ticks_when_clock_injected() -> None:
    with forest_scope(Forest()):
        with deadline_clock_scope(GasMeter(limit=16)):
            with deadline_profile_scope(enabled=True):
                with deadline_scope(Deadline.from_timeout_ms(1_000)):
                    check_deadline()
                    check_deadline()
                    snapshot = _deadline_profile_snapshot()
    assert isinstance(snapshot, dict)
    assert int(snapshot.get("checks_total", 0) or 0) == 2
    assert int(snapshot.get("total_elapsed_ns", 0) or 0) == 2
    assert int(snapshot.get("ticks_consumed", 0) or 0) == 2
    assert isinstance(snapshot.get("wall_total_elapsed_ns"), int)
    assert "ticks_per_ns" in snapshot


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_timeout_progress_reports_tick_budget::timeout_context.py::gabion.analysis.timeout_context.Deadline.from_timeout_ms::timeout_context.py::gabion.analysis.timeout_context.build_timeout_context_from_stack::timeout_context.py::gabion.analysis.timeout_context.check_deadline::timeout_context.py::gabion.analysis.timeout_context.deadline_clock_scope::timeout_context.py::gabion.analysis.timeout_context.deadline_profile_scope::timeout_context.py::gabion.analysis.timeout_context.deadline_scope::timeout_context.py::gabion.analysis.timeout_context.forest_scope
def test_timeout_progress_reports_tick_budget() -> None:
    forest = Forest()
    with forest_scope(forest):
        with deadline_profile_scope(enabled=True):
            with deadline_scope(Deadline.from_timeout_ms(1_000)):
                with deadline_clock_scope(GasMeter(limit=5)):
                    check_deadline()
                    context = build_timeout_context_from_stack(
                        forest=forest,
                        project_root=None,
                        allow_frame_fallback=True,
                        frames=[],
                    )
    progress = context.progress
    assert isinstance(progress, dict)
    assert int(progress.get("ticks_consumed", 0) or 0) == 1
    assert progress.get("tick_limit") == 5
    assert int(progress.get("ticks_remaining", 0) or 0) == 4
    assert "ticks_per_ns" in progress


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_consume_deadline_ticks_propagates_exhaustion_without_forest::timeout_context.py::gabion.analysis.timeout_context.consume_deadline_ticks::timeout_context.py::gabion.analysis.timeout_context.deadline_clock_scope
def test_consume_deadline_ticks_propagates_exhaustion_without_forest() -> None:
    def _run() -> None:
        with deadline_clock_scope(GasMeter(limit=1)):
            with pytest.raises(DeadlineClockExhausted):
                consume_deadline_ticks()

    Context().run(_run)


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_consume_deadline_ticks_requires_clock_scope::exceptions.py::gabion.exceptions.NeverThrown::timeout_context.py::gabion.analysis.timeout_context.consume_deadline_ticks
def test_consume_deadline_ticks_requires_clock_scope() -> None:
    with pytest.raises(NeverThrown):
        Context().run(consume_deadline_ticks)


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_check_deadline_ignores_wall_deadline_when_gas_clock_present::test_timeout_context.py::tests.test_timeout_context._deadline_test_scope::timeout_context.py::gabion.analysis.timeout_context.check_deadline
def test_check_deadline_ignores_wall_deadline_when_gas_clock_present() -> None:
    clock = GasMeter(limit=3)
    with _deadline_test_scope(
        deadline=Deadline(deadline_ns=0),
        clock=clock,
    ):
        check_deadline()
        check_deadline()
    assert clock.current == 2


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_render_deadline_profile_markdown::timeout_context.py::gabion.analysis.timeout_context.render_deadline_profile_markdown
def test_render_deadline_profile_markdown() -> None:
    profile = {
        "checks_total": 2,
        "total_elapsed_ns": 100,
        "wall_total_elapsed_ns": 120,
        "ticks_consumed": 7,
        "ticks_per_ns": 0.25,
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
        "io": [
            {
                "name": "report_markdown.write",
                "event_count": 2,
                "elapsed_ns": 500,
                "max_event_ns": 400,
                "bytes_total": 1024,
            }
        ],
    }
    rendered = render_deadline_profile_markdown(profile)
    assert "Deadline Profile Heat" in rendered
    assert "ticks_consumed" in rendered
    assert "ticks_per_ns" in rendered


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_render_deadline_profile_markdown_skips_invalid_rows_and_truncates::timeout_context.py::gabion.analysis.timeout_context.render_deadline_profile_markdown
def test_render_deadline_profile_markdown_skips_invalid_rows_and_truncates() -> None:
    rendered = render_deadline_profile_markdown(
        {
            "checks_total": 1,
            "total_elapsed_ns": 1,
            "unattributed_elapsed_ns": 0,
            "sites": [{"path": "a.py", "qual": "q", "check_count": 1}, "bad"],
            "edges": [{"from_path": "a.py", "from_qual": "q"}, 1],
            "io": [{"name": "x"}, 2],
        },
        max_rows=1,
    )
    assert "## Site Heat" in rendered
    assert "| ... | ... | ... | ... | ... |" in rendered


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_render_deadline_profile_markdown_skips_non_mapping_rows::timeout_context.py::gabion.analysis.timeout_context.render_deadline_profile_markdown
def test_render_deadline_profile_markdown_skips_non_mapping_rows() -> None:
    rendered = render_deadline_profile_markdown(
        {
            "checks_total": 1,
            "total_elapsed_ns": 1,
            "unattributed_elapsed_ns": 0,
            "sites": ["bad"],
            "edges": ["bad"],
            "io": ["bad"],
        }
    )
    assert "Deadline Profile Heat" in rendered


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_render_deadline_profile_markdown_ignores_non_list_sections::timeout_context.py::gabion.analysis.timeout_context.render_deadline_profile_markdown
def test_render_deadline_profile_markdown_ignores_non_list_sections() -> None:
    rendered = render_deadline_profile_markdown(
        {
            "checks_total": 1,
            "total_elapsed_ns": 1,
            "unattributed_elapsed_ns": 0,
            "sites": "bad",
            "edges": "bad",
            "io": "bad",
        }
    )
    assert "## Site Heat" in rendered


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_site_part_payload_roundtrip_helpers::timeout_context.py::gabion.analysis.timeout_context._site_key_payload::timeout_context.py::gabion.analysis.timeout_context._site_part_from_payload::timeout_context.py::gabion.analysis.timeout_context._site_part_to_payload
def test_site_part_payload_roundtrip_helpers() -> None:
    payload = _site_key_payload(path="mod.py", qual="pkg.mod.fn", span=[1, 2, 3, 4])
    restored = _site_part_from_payload(payload["key"][0])
    assert _site_part_to_payload(restored) == payload["key"][0]


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_site_part_helpers_reject_invalid_payloads::timeout_context.py::gabion.analysis.timeout_context._freeze_value::timeout_context.py::gabion.analysis.timeout_context._site_part_from_payload::timeout_context.py::gabion.analysis.timeout_context._site_part_to_payload
def test_site_part_helpers_reject_invalid_payloads() -> None:
    with pytest.raises(NeverThrown):
        _site_part_from_payload({"kind": "FileSite", "key": []})
    with pytest.raises(NeverThrown):
        _site_part_from_payload(object())
    with pytest.raises(NeverThrown):
        _site_part_to_payload(object())
    with pytest.raises(NeverThrown):
        _freeze_value(object())


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_site_part_from_payload_rejects_non_string_filesite_key::timeout_context.py::gabion.analysis.timeout_context._site_part_from_payload
def test_site_part_from_payload_rejects_non_string_filesite_key() -> None:
    with pytest.raises(NeverThrown):
        _site_part_from_payload({"kind": "FileSite", "key": [123]})


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_deadline_profile_private_helpers_cover_fallback_paths::timeout_context.py::gabion.analysis.timeout_context._profile_site_key::timeout_context.py::gabion.analysis.timeout_context._record_deadline_check::timeout_context.py::gabion.analysis.timeout_context.deadline_profile_scope
def test_deadline_profile_private_helpers_cover_fallback_paths() -> None:
    frame = inspect.currentframe()
    assert frame is not None
    assert _profile_site_key(frame, project_root=None)[1]
    assert Context().run(_deadline_profile_snapshot) is None
    with deadline_profile_scope(project_root=None, enabled=True):
        _record_deadline_check(project_root=None, frame_getter=lambda: None)


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_record_deadline_check_caches_site_keys_per_code_object::timeout_context.py::gabion.analysis.timeout_context._record_deadline_check
def test_record_deadline_check_caches_site_keys_per_code_object() -> None:
    calls = 0
    original = timeout_context._profile_site_key

    def counting_profile_site_key(frame: object, *, project_root: Path | None) -> tuple[str, str]:
        nonlocal calls
        calls += 1
        return original(frame, project_root=project_root)

    def _site_repeat() -> None:
        _record_deadline_check(
            project_root=None,
            frame_getter=inspect.currentframe,
            profile_site_key_fn=counting_profile_site_key,
        )

    with _deadline_test_scope(
        deadline=Deadline.from_timeout_ms(1_000),
        clock=GasMeter(limit=8),
    ):
        with deadline_profile_scope(enabled=True):
            _site_repeat()
            _site_repeat()

    assert calls == 1


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_record_deadline_check_resolves_project_root_once_and_caches_it::timeout_context.py::gabion.analysis.timeout_context._deadline_profile_snapshot::timeout_context.py::gabion.analysis.timeout_context._record_deadline_check::timeout_context.py::gabion.analysis.timeout_context.deadline_profile_scope
def test_record_deadline_check_resolves_project_root_once_and_caches_it(
    tmp_path: Path,
) -> None:
    def _site() -> None:
        _record_deadline_check(
            project_root=tmp_path,
            frame_getter=inspect.currentframe,
        )

    with _deadline_test_scope(
        deadline=Deadline.from_timeout_ms(1_000),
        clock=GasMeter(limit=16),
    ):
        with deadline_profile_scope(project_root=None, enabled=True):
            _site()
            _site()
            snapshot = _deadline_profile_snapshot()

    assert isinstance(snapshot, dict)
    assert int(snapshot.get("checks_total", 0) or 0) == 2


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_record_deadline_check_reuses_existing_site_id_without_reallocating::timeout_context.py::gabion.analysis.timeout_context._deadline_profile_snapshot::timeout_context.py::gabion.analysis.timeout_context._record_deadline_check::timeout_context.py::gabion.analysis.timeout_context.deadline_profile_scope
def test_record_deadline_check_reuses_existing_site_id_without_reallocating() -> None:
    def _constant_site_key(_frame: object, *, project_root: Path | None) -> tuple[str, str]:
        _ = project_root
        return ("tests/test_timeout_context.py", "constant.qual")

    def _site_a() -> None:
        _record_deadline_check(
            project_root=None,
            frame_getter=inspect.currentframe,
            profile_site_key_fn=_constant_site_key,
        )

    def _site_b() -> None:
        _record_deadline_check(
            project_root=None,
            frame_getter=inspect.currentframe,
            profile_site_key_fn=_constant_site_key,
        )

    with _deadline_test_scope(
        deadline=Deadline.from_timeout_ms(1_000),
        clock=GasMeter(limit=16),
    ):
        with deadline_profile_scope(enabled=True):
            _site_a()
            state = timeout_context._deadline_profile_var.get()
            assert state is not None
            state.frame_site_cache.clear()
            _site_b()
            snapshot = _deadline_profile_snapshot()

    assert isinstance(snapshot, dict)
    sites = snapshot.get("sites", [])
    assert isinstance(sites, list)
    assert len(sites) == 1


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_deadline_profile_sampling_preserves_total_checks_with_pending_tail::timeout_context.py::gabion.analysis.timeout_context.Deadline.from_timeout_ms::timeout_context.py::gabion.analysis.timeout_context._deadline_profile_snapshot::timeout_context.py::gabion.analysis.timeout_context.check_deadline::timeout_context.py::gabion.analysis.timeout_context.deadline_profile_scope
def test_deadline_profile_sampling_preserves_total_checks_with_pending_tail() -> None:
    with _deadline_test_scope(
        deadline=Deadline.from_timeout_ms(1_000),
        clock=GasMeter(limit=32),
    ):
        with deadline_profile_scope(enabled=True, sample_interval=4):
            for _ in range(7):
                check_deadline()
            snapshot = _deadline_profile_snapshot()

    assert isinstance(snapshot, dict)
    assert snapshot["checks_total"] == 7
    assert snapshot["sample_interval"] == 4
    assert snapshot["sampled_checks_total"] == 4
    assert snapshot["sample_pending_checks"] == 3
    sites = snapshot.get("sites", [])
    assert isinstance(sites, list)
    assert sites
    first_site = sites[0]
    assert isinstance(first_site, dict)
    assert int(first_site.get("check_count", 0) or 0) == 4


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_profile_site_key_falls_back_to_external_path_when_root_misses::timeout_context.py::gabion.analysis.timeout_context._profile_site_key
def test_profile_site_key_falls_back_to_external_path_when_root_misses() -> None:
    frame = inspect.currentframe()
    assert frame is not None
    path, qual = _profile_site_key(
        frame,
        project_root=Path(__file__).resolve().parents[1] / "missing_root",
    )
    assert path == "<external>"
    assert qual


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_record_deadline_check_keeps_edge_max_gap_when_delta_not_greater::test_timeout_context.py::tests.test_timeout_context._deadline_test_scope::test_timeout_context.py::tests.test_timeout_context.test_record_deadline_check_keeps_edge_max_gap_when_delta_not_greater._site_a::test_timeout_context.py::tests.test_timeout_context.test_record_deadline_check_keeps_edge_max_gap_when_delta_not_greater._site_b::timeout_context.py::gabion.analysis.timeout_context.Deadline.from_timeout_ms::timeout_context.py::gabion.analysis.timeout_context._deadline_profile_snapshot::timeout_context.py::gabion.analysis.timeout_context.check_deadline::timeout_context.py::gabion.analysis.timeout_context.deadline_profile_scope
def test_record_deadline_check_keeps_edge_max_gap_when_delta_not_greater() -> None:
    def _site_a() -> None:
        check_deadline()

    def _site_b() -> None:
        check_deadline()

    with _deadline_test_scope(
        deadline=Deadline.from_timeout_ms(1_000),
        clock=GasMeter(limit=16),
    ):
        with deadline_profile_scope(enabled=True):
            _site_a()
            _site_b()
            _site_a()
            _site_b()
            snapshot = _deadline_profile_snapshot()

    assert isinstance(snapshot, dict)
    edges = snapshot.get("edges", [])
    assert isinstance(edges, list)
    a_to_b = [
        row
        for row in edges
        if isinstance(row, dict)
        and str(row.get("from_qual", "")).endswith("._site_a")
        and str(row.get("to_qual", "")).endswith("._site_b")
    ]
    assert a_to_b
    assert int(a_to_b[0].get("transition_count", 0) or 0) == 2


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_record_deadline_io_does_not_lower_max_event_ns::timeout_context.py::gabion.analysis.timeout_context._deadline_profile_snapshot::timeout_context.py::gabion.analysis.timeout_context.deadline_profile_scope::timeout_context.py::gabion.analysis.timeout_context.record_deadline_io
def test_record_deadline_io_does_not_lower_max_event_ns() -> None:
    with deadline_profile_scope(enabled=True):
        record_deadline_io(name="io.write", elapsed_ns=5)
        record_deadline_io(name="io.write", elapsed_ns=2)
        snapshot = _deadline_profile_snapshot()
    assert isinstance(snapshot, dict)
    io_rows = snapshot.get("io", [])
    assert isinstance(io_rows, list)
    row = next(
        entry
        for entry in io_rows
        if isinstance(entry, dict) and entry.get("name") == "io.write"
    )
    assert row["max_event_ns"] == 5
    assert row["event_count"] == 2


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_deadline_profile_disabled_scope_noops_profile_recording::timeout_context.py::gabion.analysis.timeout_context._deadline_profile_snapshot::timeout_context.py::gabion.analysis.timeout_context._record_deadline_check::timeout_context.py::gabion.analysis.timeout_context.deadline_profile_scope::timeout_context.py::gabion.analysis.timeout_context.record_deadline_io
def test_deadline_profile_disabled_scope_noops_profile_recording() -> None:
    with deadline_profile_scope(enabled=False):
        _record_deadline_check(project_root=None)
        record_deadline_io(name="io.disabled", elapsed_ns=5)
        assert _deadline_profile_snapshot() is None


# gabion:evidence E:function_site::timeout_context.py::gabion.analysis.timeout_context.build_timeout_context_from_stack E:decision_surface/direct::timeout_context.py::gabion.analysis.timeout_context.build_timeout_context_from_stack::stale_6140a2b7922b
def test_build_timeout_context_skips_unmatched_frames() -> None:
    frame = inspect.currentframe()
    assert frame is not None
    context = build_timeout_context_from_stack(
        forest=Forest(),
        project_root=Path(__file__).resolve().parents[1] / "missing_root",
        allow_frame_fallback=False,
        frames=[frame],
    )
    assert context.call_stack.site_table == ()


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_check_deadline_accepts_explicit_deadline_argument::test_timeout_context.py::tests.test_timeout_context._deadline_test_scope::timeout_context.py::gabion.analysis.timeout_context.Deadline.from_timeout_ms::timeout_context.py::gabion.analysis.timeout_context.check_deadline
def test_check_deadline_accepts_explicit_deadline_argument() -> None:
    clock = GasMeter(limit=5)
    with _deadline_test_scope(
        deadline=Deadline.from_timeout_ms(100),
        clock=clock,
    ):
        check_deadline(deadline=Deadline.from_timeout_ms(100))
    assert clock.current == 1


class _DummyClock:
    def __init__(self) -> None:
        self._mark = 0

    def consume(self, ticks: int = 1) -> None:
        self._mark += int(ticks)

    def get_mark(self) -> int:
        return self._mark


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_deadline_profile_snapshot_omits_gas_fields_for_non_gasmeter_clock::test_timeout_context.py::tests.test_timeout_context._deadline_test_scope::timeout_context.py::gabion.analysis.timeout_context.Deadline.from_timeout_ms::timeout_context.py::gabion.analysis.timeout_context._deadline_profile_snapshot::timeout_context.py::gabion.analysis.timeout_context.check_deadline::timeout_context.py::gabion.analysis.timeout_context.deadline_profile_scope
def test_deadline_profile_snapshot_omits_gas_fields_for_non_gasmeter_clock() -> None:
    with _deadline_test_scope(
        deadline=Deadline.from_timeout_ms(100),
        clock=_DummyClock(),
    ):
        with deadline_profile_scope(enabled=True):
            check_deadline()
            snapshot = _deadline_profile_snapshot()
    assert isinstance(snapshot, dict)
    assert snapshot["ticks_consumed"] is None
    assert snapshot["ticks_per_ns"] is None


# gabion:evidence E:call_footprint::tests/test_timeout_context.py::test_timeout_progress_snapshot_handles_non_list_sites_and_non_gasmeter_clock::timeout_context.py::gabion.analysis.timeout_context.Deadline.from_timeout_ms::timeout_context.py::gabion.analysis.timeout_context._timeout_progress_snapshot::timeout_context.py::gabion.analysis.timeout_context.deadline_clock_scope::timeout_context.py::gabion.analysis.timeout_context.deadline_scope::timeout_context.py::gabion.analysis.timeout_context.forest_scope
def test_timeout_progress_snapshot_handles_non_list_sites_and_non_gasmeter_clock() -> None:
    forest = Forest()
    with forest_scope(forest):
        with deadline_scope(Deadline.from_timeout_ms(100)):
            with deadline_clock_scope(_DummyClock()):
                progress = _timeout_progress_snapshot(
                    forest=forest,
                    deadline_profile={"checks_total": 1, "sites": "invalid"},
                )
    assert progress["site_count"] == 0
    assert progress["tick_limit"] is None
    assert progress["ticks_remaining"] is None
