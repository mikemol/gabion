from __future__ import annotations

from contextvars import Context

import pytest

from gabion.analysis.aspf import Forest
from gabion.analysis.timeout_context import (
    Deadline,
    TimeoutExceeded,
    deadline_clock_scope,
    deadline_scope,
    forest_scope,
)
from gabion.deadline_clock import GasMeter
from gabion.exceptions import NeverThrown


# gabion:evidence E:call_footprint::tests/test_aspf.py::test_paramset_packed_reuse::aspf.py::gabion.analysis.aspf.Forest
def test_paramset_packed_reuse() -> None:
    forest = Forest()
    site_a = forest.add_site("a.py", "mod.fn_a")
    site_b = forest.add_site("b.py", "mod.fn_b")
    paramset = forest.add_paramset(["alpha", "beta"])
    forest.add_alt("DecisionSurface", (site_a, paramset))
    forest.add_alt("ValueDecisionSurface", (site_b, paramset))

    paramsets = [node for node in forest.nodes.values() if node.kind == "ParamSet"]
    assert len(paramsets) == 1

    alts = [alt for alt in forest.alts if alt.kind in {"DecisionSurface", "ValueDecisionSurface"}]
    assert len(alts) == 2


# gabion:evidence E:call_footprint::tests/test_aspf.py::test_add_site_records_span::aspf.py::gabion.analysis.aspf.Forest
def test_add_site_records_span() -> None:
    forest = Forest()
    site = forest.add_site("mod.py", "mod.fn", span=(1, 2, 3, 4))
    node = forest.nodes[site]
    assert node.meta["span"] == [1, 2, 3, 4]


# gabion:evidence E:call_footprint::tests/test_aspf.py::test_add_site_records_file_site::aspf.py::gabion.analysis.aspf.Forest
def test_add_site_records_file_site() -> None:
    forest = Forest()
    site = forest.add_site("mod.py", "mod.fn")
    file_nodes = [node for node in forest.nodes.values() if node.kind == "FileSite"]
    assert len(file_nodes) == 1
    assert file_nodes[0].meta["path"] == "mod.py"
    assert any(
        alt.kind == "FunctionSiteInFile" and alt.inputs[0] == site for alt in forest.alts
    )


# gabion:evidence E:call_footprint::tests/test_aspf.py::test_add_suite_site_records_file_site::aspf.py::gabion.analysis.aspf.Forest
def test_add_suite_site_records_file_site() -> None:
    forest = Forest()
    suite = forest.add_suite_site("mod.py", "mod.fn", "loop", span=(0, 1, 2, 3))
    file_nodes = [node for node in forest.nodes.values() if node.kind == "FileSite"]
    assert len(file_nodes) == 1
    assert any(
        alt.kind == "SuiteSiteInFile" and alt.inputs[0] == suite for alt in forest.alts
    )


def test_add_suite_site_parent_emits_suite_contains() -> None:
    forest = Forest()
    parent = forest.add_suite_site("mod.py", "mod.fn", "function")
    child = forest.add_suite_site(
        "mod.py",
        "mod.fn",
        "if_body",
        span=(2, 4, 3, 8),
        parent=parent,
    )

    assert any(
        alt.kind == "SuiteContains" and alt.inputs == (parent, child)
        for alt in forest.alts
    )


def test_add_alt_consumes_logical_ticks() -> None:
    forest = Forest()
    left = forest.add_site("mod.py", "mod.left")
    right = forest.add_site("mod.py", "mod.right")
    with forest_scope(forest):
        with deadline_scope(Deadline.from_timeout_ms(1_000)):
            meter = GasMeter(limit=1)
            with deadline_clock_scope(meter):
                with pytest.raises(TimeoutExceeded):
                    forest.add_alt("Edge", (left, right))
    assert meter.current == 1


def test_to_json_omits_meta_for_nodes_without_metadata() -> None:
    forest = Forest()
    forest.add_node("Sentinel", ("id",))
    payload = forest.to_json()
    sentinel = next(
        node for node in payload["nodes"] if node["kind"] == "Sentinel"
    )
    assert "meta" not in sentinel


def test_add_spec_site_records_optional_spec_fields() -> None:
    forest = Forest()
    node_id = forest.add_spec_site(
        spec_hash="abc123",
        spec_name="demo",
        spec_domain="analysis",
        spec_version=2,
    )
    meta = forest.nodes[node_id].meta
    assert meta["spec_domain"] == "analysis"
    assert meta["spec_version"] == 2


def test_add_spec_site_omits_optional_spec_fields_when_absent() -> None:
    forest = Forest()
    node_id = forest.add_spec_site(spec_hash="abc123", spec_name="demo")
    meta = forest.nodes[node_id].meta
    assert "spec_domain" not in meta
    assert "spec_version" not in meta


def test_add_alt_requires_deadline_clock_scope() -> None:
    def _run() -> None:
        forest = Forest()
        left = forest.add_site("mod.py", "mod.left")
        right = forest.add_site("mod.py", "mod.right")
        with forest_scope(forest):
            with deadline_scope(Deadline.from_timeout_ms(1_000)):
                forest.add_alt("Edge", (left, right))

    with pytest.raises(NeverThrown):
        Context().run(_run)


def test_node_intern_uses_fingerprint_identity_with_legacy_keys() -> None:
    forest = Forest()

    first = forest.add_node("Sentinel", (1, True, "1"))
    second = forest.add_node("Sentinel", (1, True, "1"))

    assert first == second
    assert len(forest.nodes) == 1
    assert forest.has_node("Sentinel", (1, True, "1"))


