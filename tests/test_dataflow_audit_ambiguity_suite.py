from __future__ import annotations

from pathlib import Path

import pytest

from gabion.analysis.aspf import Forest
from gabion.analysis.dataflow_audit import (
    CallAmbiguity,
    FunctionInfo,
    _ambiguity_suite_relation,
    _ambiguity_suite_row_to_site,
    _ambiguity_suite_row_to_suite,
    _ambiguity_virtual_count_gt_1,
    _emit_call_ambiguities,
)
from gabion.analysis.timeout_context import Deadline, deadline_scope
from gabion.exceptions import NeverThrown


def _dummy_info(path: str = "mod.py", qual: str = "mod.fn") -> FunctionInfo:
    return FunctionInfo(
        name=qual.split(".")[-1],
        qual=qual,
        path=Path(path),
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        function_span=(0, 0, 0, 1),
    )


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_ambiguity_suite.py::test_ambiguity_suite_relation_skips_missing_function_meta::dataflow_audit.py::gabion.analysis.dataflow_audit._ambiguity_suite_relation
def test_ambiguity_suite_relation_skips_missing_function_meta() -> None:
    forest = Forest()
    forest.add_node("FunctionSite", ("", ""), {})
    relation, function_index = _ambiguity_suite_relation(forest)
    assert relation == []
    assert function_index == {}


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_ambiguity_suite.py::test_ambiguity_suite_relation_skips_empty_alt_inputs::dataflow_audit.py::gabion.analysis.dataflow_audit._ambiguity_suite_relation
def test_ambiguity_suite_relation_skips_empty_alt_inputs() -> None:
    forest = Forest()
    forest.add_alt("CallCandidate", ())
    relation, _ = _ambiguity_suite_relation(forest)
    assert relation == []


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_ambiguity_suite.py::test_ambiguity_suite_relation_skips_non_suite_node::dataflow_audit.py::gabion.analysis.dataflow_audit._ambiguity_suite_relation
def test_ambiguity_suite_relation_skips_non_suite_node() -> None:
    forest = Forest()
    func_id = forest.add_site("a.py", "mod.fn")
    forest.add_alt("CallCandidate", (func_id, func_id))
    relation, _ = _ambiguity_suite_relation(forest)
    assert relation == []


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_ambiguity_suite.py::test_ambiguity_suite_relation_skips_non_call_suite::dataflow_audit.py::gabion.analysis.dataflow_audit._ambiguity_suite_relation
def test_ambiguity_suite_relation_skips_non_call_suite() -> None:
    forest = Forest()
    suite_id = forest.add_suite_site("a.py", "mod.fn", "function", span=(0, 0, 0, 1))
    candidate_id = forest.add_site("a.py", "mod.target")
    forest.add_alt("CallCandidate", (suite_id, candidate_id))
    relation, _ = _ambiguity_suite_relation(forest)
    assert relation == []


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_ambiguity_suite.py::test_ambiguity_suite_relation_requires_path_qual::dataflow_audit.py::gabion.analysis.dataflow_audit._ambiguity_suite_relation
def test_ambiguity_suite_relation_requires_path_qual() -> None:
    forest = Forest()
    suite_id = forest.add_node(
        "SuiteSite",
        ("", "", "call"),
        {"suite_kind": "call"},
    )
    candidate_id = forest.add_site("a.py", "mod.target")
    forest.add_alt("CallCandidate", (suite_id, candidate_id))
    with pytest.raises(NeverThrown):
        _ambiguity_suite_relation(forest)


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_ambiguity_suite.py::test_ambiguity_suite_relation_requires_span::dataflow_audit.py::gabion.analysis.dataflow_audit._ambiguity_suite_relation
def test_ambiguity_suite_relation_requires_span() -> None:
    forest = Forest()
    suite_id = forest.add_suite_site("a.py", "mod.fn", "call")
    candidate_id = forest.add_site("a.py", "mod.target")
    forest.add_alt("CallCandidate", (suite_id, candidate_id))
    with pytest.raises(NeverThrown):
        _ambiguity_suite_relation(forest)


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_ambiguity_suite.py::test_ambiguity_suite_relation_requires_int_span::dataflow_audit.py::gabion.analysis.dataflow_audit._ambiguity_suite_relation
def test_ambiguity_suite_relation_requires_int_span() -> None:
    forest = Forest()
    suite_id = forest.add_node(
        "SuiteSite",
        ("a.py", "mod.fn", "call"),
        {
            "path": "a.py",
            "qual": "mod.fn",
            "suite_kind": "call",
            "span": ["a", "b", "c", "d"],
        },
    )
    candidate_id = forest.add_site("a.py", "mod.target")
    forest.add_alt("CallCandidate", (suite_id, candidate_id))
    with pytest.raises(NeverThrown):
        _ambiguity_suite_relation(forest)


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_ambiguity_suite.py::test_ambiguity_suite_row_to_site_requires_path_qual::dataflow_audit.py::gabion.analysis.dataflow_audit._ambiguity_suite_row_to_site
def test_ambiguity_suite_row_to_site_requires_path_qual() -> None:
    assert _ambiguity_suite_row_to_site({}, {}) is None


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_ambiguity_suite.py::test_ambiguity_suite_row_to_suite_requires_identity::dataflow_audit.py::gabion.analysis.dataflow_audit._ambiguity_suite_row_to_suite
def test_ambiguity_suite_row_to_suite_requires_identity() -> None:
    forest = Forest()
    with pytest.raises(NeverThrown):
        _ambiguity_suite_row_to_suite({}, forest)


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_ambiguity_suite.py::test_ambiguity_virtual_count_gt_1_handles_invalid_count::dataflow_audit.py::gabion.analysis.dataflow_audit._ambiguity_virtual_count_gt_1
def test_ambiguity_virtual_count_gt_1_handles_invalid_count() -> None:
    assert _ambiguity_virtual_count_gt_1({"count": "bad"}, {}) is False


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_ambiguity_suite.py::test_emit_call_ambiguities_requires_span_when_forest::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_call_ambiguities::test_dataflow_audit_ambiguity_suite.py::tests.test_dataflow_audit_ambiguity_suite._dummy_info::timeout_context.py::gabion.analysis.timeout_context.Deadline.from_timeout_ms::timeout_context.py::gabion.analysis.timeout_context.deadline_scope
def test_emit_call_ambiguities_requires_span_when_forest() -> None:
    forest = Forest()
    entry = CallAmbiguity(
        kind="local_resolution_ambiguous",
        caller=_dummy_info(),
        call=None,
        callee_key="callee",
        candidates=tuple(),
        phase="local_resolution",
    )
    with deadline_scope(Deadline.from_timeout_ms(100)):
        with pytest.raises(NeverThrown):
            _emit_call_ambiguities([entry], project_root=None, forest=forest)
