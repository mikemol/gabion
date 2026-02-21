from __future__ import annotations

from contextvars import Context

import pytest

from gabion.analysis.aspf import Forest, NodeId
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


# gabion:evidence E:call_footprint::tests/test_aspf.py::test_add_suite_site_parent_emits_suite_contains::aspf.py::gabion.analysis.aspf.Forest
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


# gabion:evidence E:call_footprint::tests/test_aspf.py::test_add_alt_consumes_logical_ticks::timeout_context.py::gabion.analysis.timeout_context.Deadline.from_timeout_ms::timeout_context.py::gabion.analysis.timeout_context.deadline_clock_scope::timeout_context.py::gabion.analysis.timeout_context.deadline_scope::timeout_context.py::gabion.analysis.timeout_context.forest_scope
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


# gabion:evidence E:call_footprint::tests/test_aspf.py::test_to_json_omits_meta_for_nodes_without_metadata::aspf.py::gabion.analysis.aspf.Forest
def test_to_json_omits_meta_for_nodes_without_metadata() -> None:
    forest = Forest()
    forest.add_node("Sentinel", ("id",))
    payload = forest.to_json()
    sentinel = next(
        node for node in payload["nodes"] if node["kind"] == "Sentinel"
    )
    assert "meta" not in sentinel


# gabion:evidence E:call_footprint::tests/test_aspf.py::test_add_spec_site_records_optional_spec_fields::aspf.py::gabion.analysis.aspf.Forest
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


# gabion:evidence E:call_footprint::tests/test_aspf.py::test_add_spec_site_omits_optional_spec_fields_when_absent::aspf.py::gabion.analysis.aspf.Forest
def test_add_spec_site_omits_optional_spec_fields_when_absent() -> None:
    forest = Forest()
    node_id = forest.add_spec_site(spec_hash="abc123", spec_name="demo")
    meta = forest.nodes[node_id].meta
    assert "spec_domain" not in meta
    assert "spec_version" not in meta


# gabion:evidence E:call_footprint::tests/test_aspf.py::test_add_alt_requires_deadline_clock_scope::timeout_context.py::gabion.analysis.timeout_context.Deadline.from_timeout_ms::timeout_context.py::gabion.analysis.timeout_context.deadline_scope::timeout_context.py::gabion.analysis.timeout_context.forest_scope
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


# gabion:evidence E:call_footprint::tests/test_aspf.py::test_node_intern_uses_fingerprint_identity_with_legacy_keys::aspf.py::gabion.analysis.aspf.Forest
def test_node_intern_uses_fingerprint_identity_with_legacy_keys() -> None:
    forest = Forest()

    first = forest.add_node("Sentinel", (1, True, "1"))
    second = forest.add_node("Sentinel", (1, True, "1"))

    assert first == second
    assert len(forest.nodes) == 1
    assert forest.has_node("Sentinel", (1, True, "1"))


# gabion:evidence E:call_footprint::tests/test_aspf.py::test_node_fingerprint_covers_float_none_and_repr_fallback::aspf.py::gabion.analysis.aspf.NodeId
def test_node_fingerprint_covers_float_none_and_repr_fallback() -> None:
    class Dummy:
        def __repr__(self) -> str:
            return "dummy-token"

    node_id = NodeId(kind="Sentinel", key=(1.5, None, Dummy()))

    assert node_id.fingerprint() == (
        "Sentinel",
        ("float:1.5", "none:null", "repr:dummy-token"),
    )
# gabion:evidence E:call_footprint::tests/test_aspf.py::test_add_alt_interns_structural_duplicates::timeout_context.py::gabion.analysis.timeout_context.Deadline.from_timeout_ms::timeout_context.py::gabion.analysis.timeout_context.deadline_clock_scope::timeout_context.py::gabion.analysis.timeout_context.deadline_scope::timeout_context.py::gabion.analysis.timeout_context.forest_scope
def test_add_alt_interns_structural_duplicates() -> None:
    forest = Forest()
    left = forest.add_site("mod.py", "mod.left")
    right = forest.add_site("mod.py", "mod.right")
    with forest_scope(forest):
        with deadline_scope(Deadline.from_timeout_ms(1_000)):
            with deadline_clock_scope(GasMeter(limit=32)):
                first = forest.add_alt("Edge", (left, right), evidence={"b": 2, "a": 1})
                second = forest.add_alt(" Edge ", (left, right), evidence={"a": 1, "b": 2})
    assert first is second
    assert len([alt for alt in forest.alts if alt.kind == "Edge"]) == 1


# gabion:evidence E:call_footprint::tests/test_aspf.py::test_canonicalize_evidence_rejects_non_mapping_payloads::aspf.py::gabion.analysis.aspf._canonicalize_evidence
def test_canonicalize_evidence_rejects_non_mapping_payloads() -> None:
    from gabion.analysis import aspf

    assert aspf._canonicalize_evidence(["not", "a", "mapping"]) == {}
