from __future__ import annotations

import pytest

from gabion.analysis import dataflow_audit as da
from gabion.exceptions import NeverThrown


def _suite_fixture() -> tuple[tuple[str, tuple[int, int, int, int]], ...]:
    return (
        ("loop", (2, 0, 3, 1)),
        ("call", (4, 0, 4, 8)),
        ("function", (1, 0, 8, 0)),
    )


def _canonical_suite_order_facets(forest: da.Forest) -> list[tuple[tuple[object, ...], tuple[tuple[str, object], ...]]]:
    canonical_rows: list[tuple[tuple[object, ...], tuple[tuple[str, object], ...]]] = []
    for alt in forest.alts:
        if alt.kind != "SpecFacet" or alt.evidence.get("spec_name") != "suite_order":
            continue
        evidence = tuple(sorted((str(key), value) for key, value in alt.evidence.items()))
        canonical_rows.append((alt.inputs, evidence))
    return canonical_rows


# gabion:evidence E:call_footprint::tests/test_suite_order_projection_spec.py::test_suite_order_spec_materializes_spec_facets::dataflow_audit.py::gabion.analysis.dataflow_audit._materialize_suite_order_spec
def test_suite_order_spec_materializes_spec_facets() -> None:
    forest = da.Forest()
    forest.add_suite_site("mod.py", "mod.fn", "loop", span=(2, 0, 3, 1))

    da._materialize_suite_order_spec(forest=forest)

    facets = [
        alt
        for alt in forest.alts
        if alt.kind == "SpecFacet"
        and alt.evidence.get("spec_name") == "suite_order"
    ]
    assert facets


# gabion:evidence E:call_footprint::tests/test_suite_order_projection_spec.py::test_suite_order_projection_roundtrip_quotient_and_reinternment::dataflow_audit.py::gabion.analysis.dataflow_audit._materialize_suite_order_spec::dataflow_audit.py::gabion.analysis.dataflow_audit._suite_order_relation
def test_suite_order_projection_roundtrip_quotient_and_reinternment() -> None:
    forest = da.Forest()
    for suite_kind, span in _suite_fixture():
        forest.add_suite_site("mod.py", "mod.fn", suite_kind, span=span)

    da._materialize_suite_order_spec(forest=forest)
    first = _canonical_suite_order_facets(forest)

    da._materialize_suite_order_spec(forest=forest)
    second = _canonical_suite_order_facets(forest)

    assert first
    assert len(first) == len(second)
    assert first == second[: len(first)]


# gabion:evidence E:call_footprint::tests/test_suite_order_projection_spec.py::test_suite_order_projection_gauge_fixing_is_deterministic::dataflow_audit.py::gabion.analysis.dataflow_audit._materialize_suite_order_spec
def test_suite_order_projection_gauge_fixing_is_deterministic() -> None:
    left = da.Forest()
    right = da.Forest()

    for suite_kind, span in reversed(_suite_fixture()):
        left.add_suite_site("mod.py", "mod.fn", suite_kind, span=span)
    for suite_kind, span in _suite_fixture():
        right.add_suite_site("mod.py", "mod.fn", suite_kind, span=span)

    right.add_spec_site(
        spec_hash="preexisting",
        spec_name="suite_order",
        spec_domain="suite_order",
        spec_version=1,
    )

    da._materialize_suite_order_spec(forest=left)
    da._materialize_suite_order_spec(forest=right)

    assert _canonical_suite_order_facets(left) == _canonical_suite_order_facets(right)


# gabion:evidence E:call_footprint::tests/test_suite_order_projection_spec.py::test_suite_order_noncanonical_representatives_collapse_to_same_relation::dataflow_audit.py::gabion.analysis.dataflow_audit._suite_order_relation
def test_suite_order_noncanonical_representatives_collapse_to_same_relation() -> None:
    canonical = da.Forest()
    noncanonical = da.Forest()

    canonical.add_suite_site("mod.py", "mod.fn", "loop", span=(2, 0, 3, 1))
    noncanonical.add_spec_site(
        spec_hash="spec",
        spec_name="suite_order",
        spec_domain="suite_order",
        spec_version=1,
    )
    noncanonical.add_suite_site("mod.py", "mod.fn", "loop", span=(2, 0, 3, 1))

    canonical_relation, _ = da._suite_order_relation(canonical)
    noncanonical_relation, _ = da._suite_order_relation(noncanonical)

    assert canonical_relation == noncanonical_relation


# gabion:evidence E:call_footprint::tests/test_suite_order_projection_spec.py::test_suite_order_relation_requires_path_qual_and_span::dataflow_audit.py::gabion.analysis.dataflow_audit._suite_order_relation
def test_suite_order_relation_requires_path_qual_and_span() -> None:
    missing_path = da.Forest()
    missing_path.add_node(
        "SuiteSite",
        ("missing",),
        {"suite_kind": "loop", "span": [0, 0, 0, 1]},
    )
    with pytest.raises(NeverThrown):
        da._suite_order_relation(missing_path)

    missing_span = da.Forest()
    missing_span.add_node(
        "SuiteSite",
        ("missing-span",),
        {"suite_kind": "loop", "path": "mod.py", "qual": "mod.fn"},
    )
    with pytest.raises(NeverThrown):
        da._suite_order_relation(missing_span)

    bad_span = da.Forest()
    bad_span.add_node(
        "SuiteSite",
        ("bad-span",),
        {"suite_kind": "loop", "path": "mod.py", "qual": "mod.fn", "span": ["x", 0, 0, 1]},
    )
    with pytest.raises(NeverThrown):
        da._suite_order_relation(bad_span)


# gabion:evidence E:call_footprint::tests/test_suite_order_projection_spec.py::test_suite_order_row_to_site_rejects_noncanonical_rows::dataflow_audit.py::gabion.analysis.dataflow_audit._suite_order_row_to_site
def test_suite_order_row_to_site_rejects_noncanonical_rows() -> None:
    suite_index: dict[tuple[object, ...], da.NodeId] = {}
    assert da._suite_order_row_to_site(
        {"suite_path": "", "suite_qual": "q", "suite_kind": "loop"},
        suite_index,
    ) is None

    assert da._suite_order_row_to_site(
        {
            "suite_path": "mod.py",
            "suite_qual": "mod.fn",
            "suite_kind": "loop",
            "span_line": "x",
            "span_col": 0,
            "span_end_line": 0,
            "span_end_col": 1,
        },
        suite_index,
    ) is None
