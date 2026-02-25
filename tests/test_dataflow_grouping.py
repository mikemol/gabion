from __future__ import annotations

from pathlib import Path

def _load():
    repo_root = Path(__file__).resolve().parents[1]
    from gabion.analysis import dataflow_audit as da

    return da

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._union_groups::groups
def test_group_by_signature_and_union_groups() -> None:
    da = _load()
    use_map = {
        "a": da.ParamUse(direct_forward={("g", "arg[0]")}, non_forward=False, current_aliases={"a"}),
        "b": da.ParamUse(direct_forward={("g", "arg[0]")}, non_forward=False, current_aliases={"b"}),
        "c": da.ParamUse(direct_forward={("g", "arg[1]")}, non_forward=False, current_aliases={"c"}),
        "d": da.ParamUse(direct_forward=set(), non_forward=True, current_aliases={"d"}),
    }
    groups = da._group_by_signature(use_map)
    assert any(group == {"a", "b"} for group in groups)

    merged = da._union_groups([set(["a", "b"]), set(["b", "c"]), set(["d"])])
    assert any(group == {"a", "b", "c"} for group in merged)
    assert any(group == {"d"} for group in merged)

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._propagate_groups::callee_groups,opaque_callees,strictness E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._propagate_groups::stale_f3771317a034_c44c06a3
def test_propagate_groups_low_strictness_star() -> None:
    da = _load()
    call = da.CallArgs(
        callee="callee",
        pos_map={"0": "x"},
        kw_map={},
        const_pos={},
        const_kw={},
        non_const_pos=set(),
        non_const_kw=set(),
        star_pos=[(1, "args")],
        star_kw=["kwargs"],
        is_test=False,
    )
    callee_groups = {"callee": [set(["a", "b"])]}
    callee_param_orders = {"callee": ["a", "b", "c"]}
    groups = da._propagate_groups(
        [call],
        callee_groups,
        callee_param_orders,
        "low",
        opaque_callees=None,
    )
    assert any(group == {"x", "args"} for group in groups)

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._propagate_groups::callee_groups,opaque_callees,strictness E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._propagate_groups::stale_4618bfaa2c64
def test_propagate_groups_skips_opaque() -> None:
    da = _load()
    call = da.CallArgs(
        callee="opaque",
        pos_map={"0": "x"},
        kw_map={},
        const_pos={},
        const_kw={},
        non_const_pos=set(),
        non_const_kw=set(),
        star_pos=[],
        star_kw=[],
        is_test=False,
    )
    groups = da._propagate_groups(
        [call],
        {"opaque": [set(["a", "b"])]},
        {"opaque": ["a", "b"]},
        "high",
        opaque_callees={"opaque"},
    )
    assert groups == []

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._analyze_file_internal::config,recursive E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._analyze_file_internal::stale_96b5b22cdd76
def test_analyze_file_ambiguous_local_callee(tmp_path: Path) -> None:
    da = _load()
    code = (
        "def outer():\n"
        "    def inner(a, b):\n"
        "        return a + b\n"
        "    def inner(a, b):\n"
        "        return a * b\n"
        "    def caller(x, y):\n"
        "        return inner(x, y)\n"
        "    return caller(1, 2)\n"
    )
    path = tmp_path / "mod.py"
    path.write_text(code)
    groups, spans = da.analyze_file(path, recursive=True, config=da.AuditConfig())
    assert "outer.caller" in groups
    assert spans
