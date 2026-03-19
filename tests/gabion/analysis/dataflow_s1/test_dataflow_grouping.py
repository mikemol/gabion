from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from gabion.analysis.dataflow.engine import dataflow_analysis_index as index_owner
from gabion.analysis.dataflow.engine import dataflow_ingested_analysis_support as ingest_support
from gabion.analysis.dataflow.engine.dataflow_contracts import AuditConfig, CallArgs, ParamUse

da = SimpleNamespace(
    _group_by_signature=ingest_support._group_by_signature,
    _union_groups=ingest_support._union_groups,
    _propagate_groups=ingest_support._propagate_groups,
    analyze_file=index_owner.analyze_file,
)


# gabion:evidence E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._union_groups::groups
# gabion:behavior primary=desired
def test_group_by_signature_and_union_groups() -> None:
    use_map = {
        "a": ParamUse(direct_forward={("g", "arg[0]")}, non_forward=False, current_aliases={"a"}),
        "b": ParamUse(direct_forward={("g", "arg[0]")}, non_forward=False, current_aliases={"b"}),
        "c": ParamUse(direct_forward={("g", "arg[1]")}, non_forward=False, current_aliases={"c"}),
        "d": ParamUse(direct_forward=set(), non_forward=True, current_aliases={"d"}),
    }
    groups = da._group_by_signature(use_map)
    assert any(group == {"a", "b"} for group in groups)

    merged = da._union_groups([set(["a", "b"]), set(["b", "c"]), set(["d"])])
    assert any(group == {"a", "b", "c"} for group in merged)
    assert any(group == {"d"} for group in merged)

# gabion:evidence E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._propagate_groups::callee_groups,opaque_callees,strictness E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._propagate_groups::stale_f3771317a034_c44c06a3
# gabion:behavior primary=desired
def test_propagate_groups_low_strictness_star() -> None:
    call = CallArgs(
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

# gabion:evidence E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._propagate_groups::callee_groups,opaque_callees,strictness E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._propagate_groups::stale_4618bfaa2c64
# gabion:behavior primary=desired
def test_propagate_groups_skips_opaque() -> None:
    call = CallArgs(
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

# gabion:evidence E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._analyze_file_internal::config,recursive E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._analyze_file_internal::stale_96b5b22cdd76
# gabion:behavior primary=desired
def test_analyze_file_ambiguous_local_callee(tmp_path: Path) -> None:
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
    groups, spans = da.analyze_file(
        path,
        recursive=True,
        config=AuditConfig(project_root=tmp_path),
    )
    assert "outer.caller" in groups
    assert spans
