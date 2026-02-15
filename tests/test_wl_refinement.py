from __future__ import annotations

import json

import pytest

from gabion.analysis.aspf import Forest, NodeId
from gabion.analysis.dataflow_audit import AuditConfig, analyze_paths
from gabion.analysis.projection_spec import ProjectionSpec
from gabion.analysis.projection_registry import WL_REFINEMENT_SPEC
from gabion.analysis.wl_refinement import (
    _bool_param,
    _int_param,
    _seed_struct,
    _string_list_param,
    emit_wl_refinement_facets,
)
from gabion.exceptions import NeverThrown
from gabion.invariants import proof_mode_scope


def _build_suite_forest(*, child_kinds: tuple[str, ...]) -> Forest:
    forest = Forest()
    parent = forest.add_suite_site("mod.py", "pkg.mod.fn", "function")
    span_line_by_kind = {
        "if_body": 2,
        "while_body": 3,
        "for_body": 4,
    }
    for suite_kind in child_kinds:
        line = span_line_by_kind.get(suite_kind, 10)
        forest.add_suite_site(
            "mod.py",
            "pkg.mod.fn",
            suite_kind,
            span=(line, 0, line, 4),
            parent=parent,
        )
    return forest


def _wl_facet_payload(forest: Forest) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    payload = forest.to_json()
    nodes = [
        node
        for node in payload["nodes"]
        if isinstance(node, dict) and node.get("kind") == "WLLabel"
    ]
    facets = []
    for alt in payload["alts"]:
        if not isinstance(alt, dict) or alt.get("kind") != "SpecFacet":
            continue
        inputs = alt.get("inputs")
        if not isinstance(inputs, list) or len(inputs) != 3:
            continue
        label_input = inputs[2]
        if not isinstance(label_input, dict) or label_input.get("kind") != "WLLabel":
            continue
        facets.append(alt)
    return nodes, facets


def test_wl_refinement_is_deterministic_across_insertion_order() -> None:
    left = _build_suite_forest(child_kinds=("while_body", "if_body"))
    right = _build_suite_forest(child_kinds=("if_body", "while_body"))

    emit_wl_refinement_facets(forest=left, spec=WL_REFINEMENT_SPEC)
    emit_wl_refinement_facets(forest=right, spec=WL_REFINEMENT_SPEC)

    left_nodes, left_facets = _wl_facet_payload(left)
    right_nodes, right_facets = _wl_facet_payload(right)
    assert left_nodes == right_nodes
    assert left_facets == right_facets
    assert left_nodes
    first_key = left_nodes[0]["key"][0]
    assert isinstance(first_key, str)
    assert json.loads(first_key)[0] == "wl"


def test_wl_refinement_proof_mode_emits_sink_and_raises() -> None:
    forest = _build_suite_forest(child_kinds=("while_body", "if_body"))
    with proof_mode_scope(True):
        with pytest.raises(NeverThrown):
            emit_wl_refinement_facets(forest=forest, spec=WL_REFINEMENT_SPEC)

    assert any(alt.kind == "NeverInvariantSink" for alt in forest.alts)


def test_analyze_paths_emits_structured_suite_contains(tmp_path) -> None:
    source = (
        "def fn(x):\n"
        "    if x:\n"
        "        y = 1\n"
        "    else:\n"
        "        y = 2\n"
        "    for i in range(1):\n"
        "        y += i\n"
        "    else:\n"
        "        y += 0\n"
        "    while False:\n"
        "        y += 1\n"
        "    else:\n"
        "        y += 2\n"
        "    try:\n"
        "        y += 3\n"
        "    except Exception:\n"
        "        y += 4\n"
        "    else:\n"
        "        y += 5\n"
        "    finally:\n"
        "        y += 6\n"
        "    return y\n"
    )
    path = tmp_path / "mod.py"
    path.write_text(source, encoding="utf-8")

    forest = Forest()
    analyze_paths(
        [path],
        forest=forest,
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_bundle_forest=True,
        config=AuditConfig(project_root=tmp_path),
    )

    suite_kinds = {
        str(node.meta.get("suite_kind", ""))
        for node in forest.nodes.values()
        if node.kind == "SuiteSite" and node.meta.get("path") == "mod.py"
    }
    assert {
        "function_body",
        "if_body",
        "if_else",
        "for_body",
        "for_else",
        "while_body",
        "while_else",
        "try_body",
        "except_body",
        "try_else",
        "try_finally",
    }.issubset(suite_kinds)
    assert any(alt.kind == "SuiteContains" for alt in forest.alts)


def test_analyze_paths_emits_wl_facets_when_enabled(tmp_path) -> None:
    path = tmp_path / "mod.py"
    path.write_text(
        "def fn(x):\n"
        "    if x:\n"
        "        return 1\n"
        "    return 0\n",
        encoding="utf-8",
    )
    forest = Forest()
    analyze_paths(
        [path],
        forest=forest,
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_bundle_forest=True,
        include_wl_refinement=True,
        config=AuditConfig(project_root=tmp_path),
    )

    assert any(node.kind == "WLLabel" for node in forest.nodes.values())
    assert any(
        alt.kind == "SpecFacet"
        and len(alt.inputs) == 3
        and alt.inputs[2].kind == "WLLabel"
        for alt in forest.alts
    )


def test_emit_wl_refinement_facets_respects_emit_all_and_directed_edges() -> None:
    forest = _build_suite_forest(child_kinds=("if_body", "while_body"))
    spec = ProjectionSpec(
        spec_version=1,
        name="wl_refinement_directed",
        domain="wl_refinement",
        params={
            **WL_REFINEMENT_SPEC.params,
            "direction": "directed",
            "emit_steps": "all",
            "steps": 3,
            "require_injective_on_scope": True,
        },
    )
    emit_wl_refinement_facets(forest=forest, spec=spec)
    wl_nodes = [node for node in forest.nodes if node.kind == "WLLabel"]
    assert wl_nodes
    assert any(
        alt.kind == "SpecFacet"
        and len(alt.inputs) == 3
        and alt.inputs[2].kind == "WLLabel"
        for alt in forest.alts
    )


def test_emit_wl_refinement_facets_no_targets_is_noop() -> None:
    forest = Forest()
    spec = ProjectionSpec(spec_version=1, name="wl_none", domain="wl_refinement", params={})
    emit_wl_refinement_facets(forest=forest, spec=spec)
    assert not forest.alts


def test_wl_refinement_private_param_helpers_and_seed_struct() -> None:
    params = {"flag_true": "yes", "flag_false": "off", "steps": "bad", "fields": []}
    assert _bool_param(params, "flag_true", False) is True
    assert _bool_param(params, "flag_false", True) is False
    assert _int_param(params, "steps", 9) == 9
    assert _string_list_param(params, "fields", default=("suite_kind",)) == ("suite_kind",)

    forest = Forest()
    site = forest.add_suite_site(
        "mod.py",
        "pkg.mod.fn",
        "function",
        span=(1, 0, 1, 2),
    )
    forest.nodes[site].meta.update(
        {"complex": object(), "tags": ["a"], "attrs": {"x": 1}}
    )
    seed = _seed_struct(
        node_id=site,
        forest=forest,
        seed_fields=("degree", "complex", "tags", "attrs", "suite_kind"),
        degree=2,
    )
    assert seed["degree"] == 2
    assert isinstance(seed["complex"], str)
    assert seed["tags"] == ["a"]
    assert seed["attrs"] == {"x": 1}
    assert _seed_struct(
        node_id=NodeId(kind="SuiteSite", key=("missing.py", "mod.fn", "body")),
        forest=forest,
        seed_fields=("suite_kind",),
        degree=0,
    ) == {}
    assert _bool_param({"flag": 1}, "flag", False) is True
    assert _bool_param({"flag": "maybe"}, "flag", False) is False


def test_emit_wl_refinement_covers_duplicate_neighbor_counts_and_skip_non_targets() -> None:
    forest = Forest()
    root = forest.add_suite_site("mod.py", "pkg.mod.fn", "function")
    child = forest.add_suite_site("mod.py", "pkg.mod.fn", "if_body", parent=root)
    # Duplicate neighbor edge exercises multiset count accumulation branch.
    forest.add_suite_contains(root, child)
    # Non-target child exercises adjacency skip branch.
    param = forest.add_param("x")
    forest.add_alt("SuiteContains", (root, param))
    emit_wl_refinement_facets(forest=forest, spec=WL_REFINEMENT_SPEC)
    assert any(node.kind == "WLLabel" for node in forest.nodes.values())


def test_emit_wl_refinement_stabilize_early_branch() -> None:
    forest = _build_suite_forest(child_kinds=("if_body",))
    emit_wl_refinement_facets(
        forest=forest,
        spec=WL_REFINEMENT_SPEC,
        canon_fn=lambda _value: "stable",
    )
    nodes, facets = _wl_facet_payload(forest)
    assert len(nodes) == 1
    assert len(facets) == 2
    assert {facet.get("evidence", {}).get("wl_step") for facet in facets} == {0}
