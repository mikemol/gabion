from __future__ import annotations

from dataclasses import dataclass

from gabion.analysis.indexed_scan.deadline_fallback import fallback_deadline_arg_info


@dataclass(frozen=True)
class _FakeInfo:
    kind: str
    param: str | None = None
    const: str | None = None


@dataclass(frozen=True)
class _FakeCall:
    pos_map: dict[str, str]
    const_pos: dict[str, str]
    non_const_pos: tuple[str, ...]
    kw_map: dict[str, str]
    const_kw: dict[str, str]
    non_const_kw: tuple[str, ...]
    star_pos: tuple[tuple[str, str], ...]
    star_kw: tuple[str, ...]


@dataclass(frozen=True)
class _FakeCallee:
    positional_params: tuple[str, ...]
    params: tuple[str, ...]
    kwonly_params: tuple[str, ...]
    vararg: str | None
    kwarg: str | None


def _mk_info(*, kind: str, param: str | None = None, const: str | None = None) -> _FakeInfo:
    return _FakeInfo(kind=kind, param=param, const=const)


# gabion:evidence E:function_site::indexed_scan/deadline_fallback.py::gabion.analysis.indexed_scan.deadline_fallback.fallback_deadline_arg_info
def test_fallback_deadline_arg_info_high_strictness() -> None:
    call = _FakeCall(
        pos_map={"0": "deadline_arg"},
        const_pos={"1": "None"},
        non_const_pos=(),
        kw_map={"kctx": "ctx_arg"},
        const_kw={},
        non_const_kw=(),
        star_pos=(),
        star_kw=(),
    )
    callee = _FakeCallee(
        positional_params=("deadline", "kctx"),
        params=("deadline", "kctx"),
        kwonly_params=(),
        vararg=None,
        kwarg=None,
    )
    mapping = fallback_deadline_arg_info(
        call,
        callee,
        strictness="high",
        deadline_arg_info_factory=_mk_info,
    )
    assert mapping["deadline"] == _FakeInfo(kind="param", param="deadline_arg")
    assert mapping["kctx"] == _FakeInfo(kind="param", param="ctx_arg")


# gabion:evidence E:function_site::indexed_scan/deadline_fallback.py::gabion.analysis.indexed_scan.deadline_fallback.fallback_deadline_arg_info::strictness
def test_fallback_deadline_arg_info_low_strictness_applies_star_sources() -> None:
    call = _FakeCall(
        pos_map={},
        const_pos={},
        non_const_pos=(),
        kw_map={},
        const_kw={},
        non_const_kw=(),
        star_pos=(("starred", "star_param"),),
        star_kw=("star_kw_param",),
    )
    callee = _FakeCallee(
        positional_params=("deadline", "kctx"),
        params=("deadline", "kctx"),
        kwonly_params=(),
        vararg=None,
        kwarg=None,
    )
    mapping = fallback_deadline_arg_info(
        call,
        callee,
        strictness="low",
        deadline_arg_info_factory=_mk_info,
    )
    assert mapping["deadline"] == _FakeInfo(kind="param", param="star_param")
    assert mapping["kctx"] == _FakeInfo(kind="param", param="star_param")


# gabion:evidence E:function_site::indexed_scan/deadline_fallback.py::gabion.analysis.indexed_scan.deadline_fallback.fallback_deadline_arg_info::varargs
def test_fallback_deadline_arg_info_covers_vararg_kwarg_and_unknown_paths() -> None:
    call = _FakeCall(
        pos_map={"5": "caller_pos"},
        const_pos={"6": "1"},
        non_const_pos=("7",),
        kw_map={"unexpected": "caller_kw"},
        const_kw={"unexpected_const": "None"},
        non_const_kw=("unexpected_unknown",),
        star_pos=(),
        star_kw=("star_kw_param",),
    )
    callee = _FakeCallee(
        positional_params=("deadline",),
        params=("deadline",),
        kwonly_params=("kctx",),
        vararg="rest",
        kwarg="kwargs",
    )
    mapping = fallback_deadline_arg_info(
        call,
        callee,
        strictness="low",
        deadline_arg_info_factory=_mk_info,
    )
    assert mapping["rest"] == _FakeInfo(kind="param", param="caller_pos")
    assert mapping["kwargs"] == _FakeInfo(kind="param", param="caller_kw")
    assert mapping["deadline"] == _FakeInfo(kind="param", param="star_kw_param")
    assert mapping["kctx"] == _FakeInfo(kind="param", param="star_kw_param")
