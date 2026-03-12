from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gabion.exceptions import NeverThrown


def _load():
    from gabion.analysis.aspf.aspf import Forest
    from gabion.analysis.dataflow.engine.dataflow_contracts import (
        CallArgs,
        FunctionInfo,
        ReportCarrier,
    )
    from gabion.analysis.dataflow.engine.dataflow_projection_materialization import (
        CallAmbiguity,
        _collect_call_ambiguities,
        _dedupe_call_ambiguities,
        _emit_call_ambiguities,
        _lint_lines_from_call_ambiguities,
        _materialize_ambiguity_suite_agg_spec,
        _materialize_ambiguity_virtual_set_spec,
        _summarize_call_ambiguities,
    )
    from gabion.analysis.dataflow.io.dataflow_reporting import render_report

    return SimpleNamespace(
        CallArgs=CallArgs,
        CallAmbiguity=CallAmbiguity,
        Forest=Forest,
        FunctionInfo=FunctionInfo,
        ReportCarrier=ReportCarrier,
        _collect_call_ambiguities=_collect_call_ambiguities,
        _dedupe_call_ambiguities=_dedupe_call_ambiguities,
        _emit_call_ambiguities=_emit_call_ambiguities,
        _lint_lines_from_call_ambiguities=_lint_lines_from_call_ambiguities,
        _materialize_ambiguity_suite_agg_spec=_materialize_ambiguity_suite_agg_spec,
        _materialize_ambiguity_virtual_set_spec=_materialize_ambiguity_virtual_set_spec,
        _summarize_call_ambiguities=_summarize_call_ambiguities,
        render_report=render_report,
    )


def _make_function(path: Path, qual: str) -> da.FunctionInfo:
    return da.FunctionInfo(
        name=qual.split(".")[-1],
        qual=qual,
        path=path,
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        function_span=(0, 0, 0, 1),
    )


da = _load()


# gabion:evidence E:function_site::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._collect_call_ambiguities E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._collect_call_ambiguities::stale_584c89f239e5_d202bfed
# gabion:behavior primary=desired
def test_collect_call_ambiguities_skips_test_calls(tmp_path: Path) -> None:
    source = tmp_path / "mod.py"
    source.write_text(
        "\n".join(
            [
                "def helper(x):",
                "    return x",
                "",
                "def test_call():",
                "    helper(1)",
            ]
        )
        + "\n"
    )
    ambiguities = da._collect_call_ambiguities(
        [source],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=False,
        parse_failure_witnesses=[],
    )
    assert ambiguities == []


# gabion:evidence E:function_site::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._collect_call_ambiguities E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._collect_call_ambiguities::stale_e9001418057f
# gabion:behavior primary=desired
def test_collect_call_ambiguities_skips_test_calls_in_tests_dir(
    tmp_path: Path,
) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    source = tests_dir / "test_mod.py"
    source.write_text(
        "\n".join(
            [
                "def helper(x):",
                "    return x",
                "",
                "def test_call():",
                "    helper(1)",
            ]
        )
        + "\n"
    )
    ambiguities = da._collect_call_ambiguities(
        [source],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=False,
        parse_failure_witnesses=[],
    )
    assert ambiguities == []


# gabion:evidence E:function_site::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._emit_call_ambiguities
# gabion:behavior primary=desired
def test_dedupe_emit_and_lint_call_ambiguities(tmp_path: Path) -> None:
    caller = _make_function(tmp_path / "mod.py", "mod.caller")
    candidate = _make_function(tmp_path / "mod.py", "mod.target")
    call = da.CallArgs(
        callee="target",
        pos_map={},
        kw_map={},
        const_pos={},
        const_kw={},
        non_const_pos=set(),
        non_const_kw=set(),
        star_pos=[],
        star_kw=[],
        is_test=False,
        span=(0, 0, 0, 1),
    )
    entry = da.CallAmbiguity(
        kind="local_resolution_ambiguous",
        caller=caller,
        call=call,
        callee_key="target",
        candidates=(candidate,),
        phase="resolve_local_callee",
    )
    deduped = da._dedupe_call_ambiguities([entry, entry])
    assert len(deduped) == 1

    emitted = da._emit_call_ambiguities(
        deduped,
        project_root=tmp_path,
        forest=da.Forest(),
    )
    assert emitted[0]["candidate_count"] == 1

    lint_lines = da._lint_lines_from_call_ambiguities(
        [
            "bad",
            {"kind": "x", "site": "bad"},
            {"kind": "x", "site": {"path": "", "span": [1, 2, 3, 4]}},
            {
                "kind": "x",
                "site": {"path": "mod.py", "span": ["x", "y", 0, 0]},
                "candidate_count": "bad",
            },
        ]
    )
    assert any("GABION_AMBIGUITY" in line for line in lint_lines)

    summary = da._summarize_call_ambiguities(
        [
            emitted[0],
            dict(emitted[0]),
        ],
        max_entries=1,
    )
    assert any("Counts by witness kind" in line for line in summary)
    assert any("... " in line for line in summary)


# gabion:evidence E:function_site::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._summarize_call_ambiguities
# gabion:behavior primary=desired
def test_summarize_call_ambiguities_uses_execution_ops(monkeypatch) -> None:
    from gabion.analysis.indexed_scan.calls import call_ambiguity_summary
    from gabion.analysis.dataflow.engine import (
        dataflow_projection_materialization as projection_materialization,
    )

    seen: dict[str, object] = {}
    monkeypatch.setattr(
        call_ambiguity_summary,
        "_ambiguity_summary_execution_ops",
        lambda: ("typed-ambiguity-op",),
    )

    def _fake_apply_execution_ops(ops, rows):
        seen["ops"] = ops
        seen["rows"] = rows
        return rows

    monkeypatch.setattr(
        call_ambiguity_summary,
        "apply_execution_ops",
        _fake_apply_execution_ops,
    )

    summary = projection_materialization._summarize_call_ambiguities(
        [
            {
                "kind": "local_resolution_ambiguous",
                "site": {
                    "path": "mod.py",
                    "function": "mod.helper",
                    "span": [3, 2, 3, 8],
                },
                "candidate_count": 2,
            }
        ]
    )

    assert seen["ops"] == ("typed-ambiguity-op",)
    assert seen["rows"] == [
        {
            "kind": "local_resolution_ambiguous",
            "site_path": "mod.py",
            "site_function": "mod.helper",
            "span_line": 3,
            "span_col": 2,
            "span_end_line": 3,
            "span_end_col": 8,
            "candidate_count": 2,
        }
    ]
    assert any("Counts by witness kind" in line for line in summary)
    assert any("Top ambiguous sites:" in line for line in summary)


# gabion:evidence E:call_footprint::tests/test_ambiguity_helpers.py::test_emit_call_ambiguities_uses_call_suite::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._emit_call_ambiguities::test_ambiguity_helpers.py::tests.test_ambiguity_helpers._make_function
# gabion:behavior primary=desired
def test_emit_call_ambiguities_uses_call_suite(tmp_path: Path) -> None:
    caller = _make_function(tmp_path / "mod.py", "mod.caller")
    candidate = _make_function(tmp_path / "mod.py", "mod.target")
    call = da.CallArgs(
        callee="target",
        pos_map={},
        kw_map={},
        const_pos={},
        const_kw={},
        non_const_pos=set(),
        non_const_kw=set(),
        star_pos=[],
        star_kw=[],
        is_test=False,
        span=(0, 0, 0, 1),
    )
    entry = da.CallAmbiguity(
        kind="local_resolution_ambiguous",
        caller=caller,
        call=call,
        callee_key="target",
        candidates=(candidate,),
        phase="resolve_local_callee",
    )
    forest = da.Forest()
    emitted = da._emit_call_ambiguities(
        [entry],
        project_root=tmp_path,
        forest=forest,
    )
    assert emitted
    assert any(
        node.kind == "SuiteSite" and node.meta.get("suite_kind") == "call"
        for node in forest.nodes.values()
    )
    assert any(
        alt.kind == "CallCandidate"
        and forest.nodes.get(alt.inputs[0], None) is not None
        and forest.nodes[alt.inputs[0]].kind == "SuiteSite"
        for alt in forest.alts
    )
    assert any(
        alt.kind == "CallCandidate"
        and len(alt.inputs) >= 2
        and forest.nodes.get(alt.inputs[1], None) is not None
        and forest.nodes[alt.inputs[1]].kind == "SuiteSite"
        and forest.nodes[alt.inputs[1]].meta.get("suite_kind") == "function"
        for alt in forest.alts
    )


# gabion:evidence E:call_footprint::tests/test_ambiguity_helpers.py::test_emit_call_ambiguities_requires_candidate_function_span::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._emit_call_ambiguities::test_ambiguity_helpers.py::tests.test_ambiguity_helpers._make_function
# gabion:behavior primary=desired
def test_emit_call_ambiguities_requires_candidate_function_span(tmp_path: Path) -> None:
    caller = _make_function(tmp_path / "mod.py", "mod.caller")
    candidate = da.FunctionInfo(
        name="target",
        qual="mod.target",
        path=tmp_path / "mod.py",
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        function_span=None,
    )
    call = da.CallArgs(
        callee="target",
        pos_map={},
        kw_map={},
        const_pos={},
        const_kw={},
        non_const_pos=set(),
        non_const_kw=set(),
        star_pos=[],
        star_kw=[],
        is_test=False,
        span=(0, 0, 0, 1),
    )
    entry = da.CallAmbiguity(
        kind="local_resolution_ambiguous",
        caller=caller,
        call=call,
        callee_key="target",
        candidates=(candidate,),
        phase="resolve_local_callee",
    )
    with pytest.raises(NeverThrown):
        da._emit_call_ambiguities(
            [entry],
            project_root=tmp_path,
            forest=da.Forest(),
        )


# gabion:evidence E:call_footprint::tests/test_ambiguity_helpers.py::test_ambiguity_suite_agg_materializes_spec_facets::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._emit_call_ambiguities::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._materialize_ambiguity_suite_agg_spec::test_ambiguity_helpers.py::tests.test_ambiguity_helpers._make_function
# gabion:behavior primary=desired
def test_ambiguity_suite_agg_materializes_spec_facets(tmp_path: Path) -> None:
    caller = _make_function(tmp_path / "mod.py", "mod.caller")
    candidate = _make_function(tmp_path / "mod.py", "mod.target")
    call = da.CallArgs(
        callee="target",
        pos_map={},
        kw_map={},
        const_pos={},
        const_kw={},
        non_const_pos=set(),
        non_const_kw=set(),
        star_pos=[],
        star_kw=[],
        is_test=False,
        span=(1, 0, 1, 1),
    )
    entry = da.CallAmbiguity(
        kind="local_resolution_ambiguous",
        caller=caller,
        call=call,
        callee_key="target",
        candidates=(candidate,),
        phase="resolve_local_callee",
    )
    forest = da.Forest()
    da._emit_call_ambiguities(
        [entry],
        project_root=tmp_path,
        forest=forest,
    )
    da._materialize_ambiguity_suite_agg_spec(forest=forest)
    assert any(
        alt.kind == "SpecFacet"
        and alt.evidence.get("spec_name") == "ambiguity_suite_agg"
        and forest.nodes.get(alt.inputs[1], None) is not None
        and forest.nodes[alt.inputs[1]].kind == "SuiteSite"
        and forest.nodes[alt.inputs[1]].meta.get("suite_kind") == "call"
        for alt in forest.alts
    )


# gabion:evidence E:call_footprint::tests/test_ambiguity_helpers.py::test_ambiguity_virtual_set_spec_requires_multiple_candidates::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._emit_call_ambiguities::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._materialize_ambiguity_virtual_set_spec::test_ambiguity_helpers.py::tests.test_ambiguity_helpers._make_function
# gabion:behavior primary=desired
def test_ambiguity_virtual_set_spec_requires_multiple_candidates(tmp_path: Path) -> None:
    caller = _make_function(tmp_path / "mod.py", "mod.caller")
    candidate = _make_function(tmp_path / "mod.py", "mod.target")
    call = da.CallArgs(
        callee="target",
        pos_map={},
        kw_map={},
        const_pos={},
        const_kw={},
        non_const_pos=set(),
        non_const_kw=set(),
        star_pos=[],
        star_kw=[],
        is_test=False,
        span=(1, 0, 1, 1),
    )
    entry = da.CallAmbiguity(
        kind="local_resolution_ambiguous",
        caller=caller,
        call=call,
        callee_key="target",
        candidates=(candidate,),
        phase="resolve_local_callee",
    )
    forest = da.Forest()
    da._emit_call_ambiguities(
        [entry],
        project_root=tmp_path,
        forest=forest,
    )
    da._materialize_ambiguity_virtual_set_spec(forest=forest)
    assert not any(
        alt.kind == "SpecFacet"
        and alt.evidence.get("spec_name") == "ambiguity_virtual_set"
        for alt in forest.alts
    )


# gabion:evidence E:call_footprint::tests/test_ambiguity_helpers.py::test_ambiguity_virtual_set_spec_materializes_suite_facets::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._emit_call_ambiguities::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._materialize_ambiguity_virtual_set_spec::test_ambiguity_helpers.py::tests.test_ambiguity_helpers._make_function
# gabion:behavior primary=desired
def test_ambiguity_virtual_set_spec_materializes_suite_facets(tmp_path: Path) -> None:
    caller = _make_function(tmp_path / "mod.py", "mod.caller")
    candidate_a = _make_function(tmp_path / "mod.py", "mod.target_a")
    candidate_b = _make_function(tmp_path / "mod.py", "mod.target_b")
    call = da.CallArgs(
        callee="target",
        pos_map={},
        kw_map={},
        const_pos={},
        const_kw={},
        non_const_pos=set(),
        non_const_kw=set(),
        star_pos=[],
        star_kw=[],
        is_test=False,
        span=(1, 0, 1, 1),
    )
    entry = da.CallAmbiguity(
        kind="local_resolution_ambiguous",
        caller=caller,
        call=call,
        callee_key="target",
        candidates=(candidate_a, candidate_b),
        phase="resolve_local_callee",
    )
    forest = da.Forest()
    da._emit_call_ambiguities(
        [entry],
        project_root=tmp_path,
        forest=forest,
    )
    da._materialize_ambiguity_virtual_set_spec(forest=forest)
    assert any(
        alt.kind == "SpecFacet"
        and alt.evidence.get("spec_name") == "ambiguity_virtual_set"
        and forest.nodes.get(alt.inputs[1], None) is not None
        and forest.nodes[alt.inputs[1]].kind == "SuiteSite"
        and forest.nodes[alt.inputs[1]].meta.get("suite_kind") == "call"
        for alt in forest.alts
    )


# gabion:evidence E:function_site::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._summarize_call_ambiguities
# gabion:behavior primary=verboten facets=empty,invalid
def test_summarize_call_ambiguities_handles_empty_and_invalid_entries() -> None:
    assert da._summarize_call_ambiguities([]) == []
    summary = da._summarize_call_ambiguities(
        [
            {
                "kind": "local_resolution_ambiguous",
                "site": {"path": "mod.py", "function": "f", "span": ["x", "y", 0, 0]},
                "candidate_count": "bad",
            }
        ],
        max_entries=1,
    )
    assert any("Counts by witness kind" in line for line in summary)


# gabion:evidence E:function_site::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan.render_report
# gabion:behavior primary=desired
def test_render_report_includes_ambiguities() -> None:
    report, _ = da.render_report(
        {},
        0,
        report=da.ReportCarrier(
            forest=da.Forest(),
            ambiguity_witnesses=[
                {
                    "kind": "local_resolution_ambiguous",
                    "site": {"path": "mod.py", "function": "f", "span": [1, 2, 3, 4]},
                    "candidate_count": 2,
                }
            ],
        ),
    )
    assert "Ambiguities:" in report
