from __future__ import annotations

import ast
from pathlib import Path

from gabion.tooling.policy_substrate.aspf_union_view import ASPFUnionView, CSTParseFailureEvent, build_aspf_union_view
from gabion.tooling.policy_substrate.dataflow_fibration import (
    branch_required_symbols,
    build_dataflow_fiber_bundle_for_qualname,
    compute_recombination_frontier,
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
    bundle = build_dataflow_fiber_bundle_for_qualname(
        rel_path="src/gabion/example.py",
        module_tree=module_tree,
        qualname="fn",
    )
    branch_node = next(node for node in ast.walk(module_tree) if isinstance(node, ast.If))
    required_symbols = branch_required_symbols(branch_node)
    frontier = compute_recombination_frontier(
        rel_path="src/gabion/example.py",
        qualname="fn",
        bundle=bundle,
        branch_line=branch_node.lineno,
        branch_column=branch_node.col_offset + 1,
        branch_node_kind="branch:if",
        required_symbols=required_symbols,
    )
    assert set(frontier.required_symbols) == {"x", "y"}
    assert frontier.unresolved_symbols == ()
    assert frontier.upstream_site_ids
    assert frontier.anchor_line == 3
    assert frontier.bundle_event_count >= 1
    assert frontier.execution_event_count >= 1
    assert frontier.execution_frontier_ordinal >= 0
    assert frontier.execution_upstream_site_ids
    assert frontier.execution_frontier_line == 2
