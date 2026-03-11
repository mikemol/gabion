from __future__ import annotations

import ast
from pathlib import Path

from gabion.analysis.aspf import aspf_lattice_algebra as lattice_algebra
from gabion.tooling.policy_substrate import dataflow_fibration as dataflow_fibration_adapter
from gabion.tooling.policy_substrate.aspf_union_view import ASPFUnionView, CSTParseFailureEvent, build_aspf_union_view
from gabion.tooling.policy_substrate.dataflow_fibration import (
    BranchWitnessRequest,
    branch_required_symbols,
    build_fiber_bundle_for_qualname,
    compute_lattice_witness,
    eta_data_to_exec,
    eta_exec_to_data,
    iter_lattice_witnesses,
    join,
    meet,
)
from gabion.tooling.policy_substrate.overlap_eval import evaluate_condition_overlaps
from gabion.tooling.policy_substrate.projection_lens import LensEvent
from gabion.tooling.policy_substrate.rule_runtime import cst_failure_seeds, decorate_site, new_run_context
from gabion.tooling.policy_substrate.taint_intervals import build_taint_intervals
from gabion.tooling.runtime.policy_scan_batch import build_policy_scan_batch_from_sources


# gabion:behavior primary=desired

def test_policy_substrate_builds_union_view_with_deterministic_paths(tmp_path: Path) -> None:
    batch = build_policy_scan_batch_from_sources(
        root=tmp_path,
        source_by_rel_path={
            "src/gabion/b.py": "def beta():\n    return 2\n",
            "src/gabion/a.py": "def alpha():\n    return 1\n",
        },
    )
    union_view = build_aspf_union_view(batch=batch)
    rel_paths = [module.rel_path for module in union_view.modules]
    assert rel_paths == ["src/gabion/a.py", "src/gabion/b.py"]


# gabion:behavior primary=desired

def test_policy_substrate_builds_lifo_taint_intervals() -> None:
    events = (
        LensEvent(
            ordinal=1,
            site_id="s",
            path="p",
            qualname="q",
            line=1,
            column=1,
            node_kind="node",
            surface="pyast",
            fiber_id="f",
            event_kind="intro1",
            event_phase="taint_intro",
            input_slot="slot",
            taint_class="t",
            action="taint_intro",
        ),
        LensEvent(
            ordinal=2,
            site_id="s",
            path="p",
            qualname="q",
            line=2,
            column=1,
            node_kind="node",
            surface="pyast",
            fiber_id="f",
            event_kind="intro2",
            event_phase="taint_intro",
            input_slot="slot",
            taint_class="t",
            action="taint_intro",
        ),
        LensEvent(
            ordinal=3,
            site_id="s",
            path="p",
            qualname="q",
            line=3,
            column=1,
            node_kind="node",
            surface="pyast",
            fiber_id="f",
            event_kind="erase2",
            event_phase="taint_erase",
            input_slot="slot",
            taint_class="t",
            action="taint_erase",
        ),
        LensEvent(
            ordinal=4,
            site_id="s",
            path="p",
            qualname="q",
            line=4,
            column=1,
            node_kind="node",
            surface="pyast",
            fiber_id="f",
            event_kind="erase1",
            event_phase="taint_erase",
            input_slot="slot",
            taint_class="t",
            action="taint_erase",
        ),
    )
    intervals = tuple(build_taint_intervals(events=events))
    assert len(intervals) == 2
    spans = {(item.start_ordinal, item.end_ordinal) for item in intervals}
    assert spans == {(1, 4), (2, 3)}


# gabion:behavior primary=desired

def test_policy_substrate_condition_overlap_is_subpath_of_interval() -> None:
    events = (
        LensEvent(
            ordinal=1,
            site_id="s",
            path="p",
            qualname="q",
            line=1,
            column=1,
            node_kind="node",
            surface="pyast",
            fiber_id="f",
            event_kind="intro",
            event_phase="taint_intro",
            input_slot="slot",
            taint_class="t",
            action="taint_intro",
        ),
        LensEvent(
            ordinal=2,
            site_id="s",
            path="p",
            qualname="q",
            line=2,
            column=1,
            node_kind="node",
            surface="pyast",
            fiber_id="f",
            event_kind="condition",
            event_phase="condition",
            input_slot="slot",
            taint_class="t",
            action="condition",
        ),
        LensEvent(
            ordinal=3,
            site_id="s",
            path="p",
            qualname="q",
            line=3,
            column=1,
            node_kind="node",
            surface="pyast",
            fiber_id="f",
            event_kind="erase",
            event_phase="taint_erase",
            input_slot="slot",
            taint_class="t",
            action="taint_erase",
        ),
    )
    intervals = tuple(build_taint_intervals(events=events))
    overlaps = tuple(
        evaluate_condition_overlaps(
        intervals=intervals,
        condition_events=(events[1],),
        )
    )
    assert len(overlaps) == 1
    assert overlaps[0].start_ordinal == 2


# gabion:behavior primary=desired

def test_policy_substrate_decorates_site_with_overlap_provenance() -> None:
    context = new_run_context(rule_name="test_rule")
    decoration = decorate_site(
        run_context=context,
        rule_name="test_rule",
        rel_path="src/gabion/example.py",
        qualname="example.fn",
        line=10,
        column=4,
        node_kind="if",
        input_slot="branch:if",
        taint_class="branch_control",
        intro_kind="syntax:intro",
        condition_kind="syntax:condition",
        erase_kind="syntax:erase",
        rationale="test rationale",
    )
    assert decoration.flow_identity
    assert decoration.fiber_id
    assert decoration.taint_interval_id
    assert decoration.condition_overlap_id
    assert len(decoration.fiber_trace) == 3


# gabion:behavior primary=desired

def test_policy_substrate_cst_failure_seed_projection() -> None:
    union_view = ASPFUnionView(
        root=Path("."),
        modules=(),
        cst_failures=(
            CSTParseFailureEvent(
                path=Path("src/gabion/x.py"),
                rel_path="src/gabion/x.py",
                line=2,
                column=7,
                message="bad cst",
            ),
        ),
    )
    seeds = tuple(cst_failure_seeds(union_view=union_view))
    assert len(seeds) == 1
    assert seeds[0].kind == "cst_parse_failure"
    assert seeds[0].line == 2


# gabion:behavior primary=desired
def test_policy_substrate_dataflow_bundle_frontier_recombines_required_symbols() -> None:
    source = (
        "def fn(a, b):\n"
        "    x = a + 1\n"
        "    y = b + 2\n"
        "    if x > y:\n"
        "        return x\n"
        "    return y\n"
    )
    module_tree = ast.parse(source)
    bundle = build_fiber_bundle_for_qualname(
        rel_path="src/gabion/example.py",
        module_tree=module_tree,
        qualname="fn",
    )
    branch_node = next(node for node in ast.walk(module_tree) if isinstance(node, ast.If))
    required_symbols = branch_required_symbols(branch_node)
    witness = compute_lattice_witness(
        rel_path="src/gabion/example.py",
        qualname="fn",
        bundle=bundle,
        branch_line=branch_node.lineno,
        branch_column=branch_node.col_offset + 1,
        branch_node_kind="branch:if",
        required_symbols=required_symbols,
    )
    assert set(witness.required_symbols) == {"x", "y"}
    assert list(witness.unresolved_symbols) == []
    assert list(witness.data_upstream_site_ids)
    assert witness.data_anchor_line == 3
    assert witness.bundle_event_count >= 1
    assert witness.execution_event_count >= 1
    assert witness.exec_frontier_ordinal >= 0
    assert list(witness.exec_upstream_site_ids)
    assert witness.exec_frontier_line == 2
    assert witness.complete


# gabion:behavior primary=desired
def test_policy_substrate_lattice_join_meet_are_commutative_idempotent() -> None:
    left = ("a", "b", "a")
    right = ("b", "c")
    j1 = join(left_ids=left, right_ids=right)
    j2 = join(left_ids=right, right_ids=left)
    m1 = meet(left_ids=left, right_ids=right)
    m2 = meet(left_ids=right, right_ids=left)
    assert tuple(j1.result_ids) == ("a", "b", "c")
    assert tuple(j1.result_ids) == tuple(j2.result_ids)
    assert tuple(m1.result_ids) == ("b",)
    assert tuple(m1.result_ids) == tuple(m2.result_ids)
    self_join = join(left_ids=left, right_ids=left)
    self_meet = meet(left_ids=left, right_ids=left)
    assert tuple(self_join.result_ids) == ("a", "b")
    assert tuple(self_meet.result_ids) == ("a", "b")


# gabion:behavior primary=desired
def test_policy_substrate_naturality_witnesses_are_complete_for_simple_function() -> None:
    source = (
        "def fn(a):\n"
        "    x = a + 1\n"
        "    if x:\n"
        "        return x\n"
        "    return 0\n"
    )
    module_tree = ast.parse(source)
    bundle = build_fiber_bundle_for_qualname(
        rel_path="src/gabion/example.py",
        module_tree=module_tree,
        qualname="fn",
    )
    data_events = tuple(bundle.data.events)
    exec_events = tuple(bundle.exec.events)
    forward = eta_data_to_exec(data_events=data_events, exec_events=exec_events)
    reverse = eta_exec_to_data(data_events=data_events, exec_events=exec_events)
    assert forward.complete
    assert reverse.complete


# gabion:behavior primary=desired
def test_lazy_pull_no_work_before_consumption() -> None:
    source = (
        "def fn(a):\n"
        "    x = a + 1\n"
        "    if x:\n"
        "        return x\n"
        "    return 0\n"
    )
    module_tree = ast.parse(source)
    branch_node = next(node for node in ast.walk(module_tree) if isinstance(node, ast.If))
    request = BranchWitnessRequest(
        branch_line=branch_node.lineno,
        branch_column=branch_node.col_offset + 1,
        branch_node_kind="branch:if",
        required_symbols=tuple(branch_required_symbols(branch_node)),
    )
    call_count = {"count": 0}
    real_builder = lattice_algebra.build_fiber_bundle_for_qualname

    def _wrapped_builder(*, rel_path: str, module_tree: ast.AST, qualname: str):
        call_count["count"] += 1
        return real_builder(
            rel_path=rel_path,
            module_tree=module_tree,
            qualname=qualname,
        )

    original_builder = lattice_algebra.build_fiber_bundle_for_qualname
    lattice_algebra.build_fiber_bundle_for_qualname = _wrapped_builder  # type: ignore[assignment]
    try:
        witness_iter = iter_lattice_witnesses(
            rel_path="src/gabion/example.py",
            qualname="fn",
            module_tree=module_tree,
            requests=(request,),
        )
        assert call_count["count"] == 0
        next(witness_iter)
        assert call_count["count"] == 1
    finally:
        lattice_algebra.build_fiber_bundle_for_qualname = original_builder  # type: ignore[assignment]


# gabion:behavior primary=desired
def test_policy_substrate_iter_lattice_witnesses_is_lazy_until_consumed() -> None:
    source = (
        "def fn(a):\n"
        "    x = a + 1\n"
        "    if x:\n"
        "        return x\n"
        "    return 0\n"
    )
    module_tree = ast.parse(source)
    branch_node = next(node for node in ast.walk(module_tree) if isinstance(node, ast.If))
    request = BranchWitnessRequest(
        branch_line=branch_node.lineno,
        branch_column=branch_node.col_offset + 1,
        branch_node_kind="branch:if",
        required_symbols=tuple(branch_required_symbols(branch_node)),
    )
    call_count = {"count": 0}
    real_builder = lattice_algebra.build_fiber_bundle_for_qualname

    def _wrapped_builder(*, rel_path: str, module_tree: ast.AST, qualname: str):
        call_count["count"] += 1
        return real_builder(
            rel_path=rel_path,
            module_tree=module_tree,
            qualname=qualname,
        )

    original_builder = lattice_algebra.build_fiber_bundle_for_qualname
    lattice_algebra.build_fiber_bundle_for_qualname = _wrapped_builder  # type: ignore[assignment]
    try:
        witness_iter = iter_lattice_witnesses(
            rel_path="src/gabion/example.py",
            qualname="fn",
            module_tree=module_tree,
            requests=(request,),
        )
        assert call_count["count"] == 0
        first = next(witness_iter)
        assert first.branch_line == branch_node.lineno
        assert call_count["count"] == 1
    finally:
        lattice_algebra.build_fiber_bundle_for_qualname = original_builder  # type: ignore[assignment]


# gabion:behavior primary=desired
def test_policy_substrate_unresolved_symbols_do_not_apply_projection_transforms() -> None:
    source = (
        "def fn(a):\n"
        "    if missing_symbol:\n"
        "        return a\n"
        "    return a\n"
    )
    module_tree = ast.parse(source)
    bundle = build_fiber_bundle_for_qualname(
        rel_path="src/gabion/example.py",
        module_tree=module_tree,
        qualname="fn",
    )
    branch_node = next(node for node in ast.walk(module_tree) if isinstance(node, ast.If))
    witness = compute_lattice_witness(
        rel_path="src/gabion/example.py",
        qualname="fn",
        bundle=bundle,
        branch_line=branch_node.lineno,
        branch_column=branch_node.col_offset + 1,
        branch_node_kind="branch:if",
        required_symbols=branch_required_symbols(branch_node),
    )
    assert list(witness.unresolved_symbols) == ["missing_symbol"]
    assert list(witness.obligations) == []
    assert list(witness.erasures) == []
    assert list(witness.boundary_crossings) == []
    assert witness.violation is None
    assert witness.complete is False


# gabion:behavior primary=desired
def test_policy_substrate_compute_lattice_witness_reuses_artifact_cache() -> None:
    source = (
        "def fn(a):\n"
        "    x = a + 1\n"
        "    if x:\n"
        "        return x\n"
        "    return 0\n"
    )
    module_tree = ast.parse(source)
    bundle = build_fiber_bundle_for_qualname(
        rel_path="src/gabion/example.py",
        module_tree=module_tree,
        qualname="fn",
    )
    branch_node = next(node for node in ast.walk(module_tree) if isinstance(node, ast.If))
    first = compute_lattice_witness(
        rel_path="src/gabion/example.py",
        qualname="fn",
        bundle=bundle,
        branch_line=branch_node.lineno,
        branch_column=branch_node.col_offset + 1,
        branch_node_kind="branch:if",
        required_symbols=branch_required_symbols(branch_node),
    )
    assert first.complete

    def _raise_frontier(**_: object):
        raise AssertionError("frontier should not be recomputed when cache is warm")

    original_frontier = lattice_algebra.frontier
    lattice_algebra.frontier = _raise_frontier  # type: ignore[assignment]
    try:
        second = compute_lattice_witness(
            rel_path="src/gabion/example.py",
            qualname="fn",
            bundle=bundle,
            branch_line=branch_node.lineno,
            branch_column=branch_node.col_offset + 1,
            branch_node_kind="branch:if",
            required_symbols=branch_required_symbols(branch_node),
        )
        assert second.branch_site_id == first.branch_site_id
    finally:
        lattice_algebra.frontier = original_frontier  # type: ignore[assignment]


# gabion:behavior primary=desired
def test_policy_substrate_adapter_removes_legacy_recombination_exports() -> None:
    assert "compute_recombination_frontier" not in dataflow_fibration_adapter.__all__
    assert "empty_recombination_frontier" not in dataflow_fibration_adapter.__all__
    assert "RecombinationFrontier" not in dataflow_fibration_adapter.__all__
    assert not hasattr(dataflow_fibration_adapter, "compute_recombination_frontier")
    assert not hasattr(dataflow_fibration_adapter, "empty_recombination_frontier")
    assert not hasattr(dataflow_fibration_adapter, "RecombinationFrontier")
