from __future__ import annotations

from gabion.analysis.call_cluster_shared import (
    cluster_identity_from_key,
    render_cluster_heading,
    render_string_codeblock,
    sorted_unique_strings,
)
from gabion.analysis import evidence_keys
from gabion.analysis.report_doc import ReportDoc


# gabion:evidence E:call_footprint::tests/test_call_cluster_shared.py::test_cluster_identity_from_key_and_sorted_unique_strings::call_cluster_shared.py::gabion.analysis.call_cluster_shared.cluster_identity_from_key::call_cluster_shared.py::gabion.analysis.call_cluster_shared.sorted_unique_strings::evidence_keys.py::gabion.analysis.evidence_keys.make_call_cluster_key
def test_cluster_identity_from_key_and_sorted_unique_strings() -> None:
    metadata = cluster_identity_from_key(
        evidence_keys.make_call_cluster_key(
            targets=[{"path": "mod.py", "qual": "pkg.fn"}]
        )
    )
    assert metadata.identity
    assert metadata.display.startswith("E:call_cluster")
    assert sorted_unique_strings(["b", "a", "a"], source="test") == ("a", "b")


# gabion:evidence E:call_footprint::tests/test_call_cluster_shared.py::test_render_cluster_heading_and_non_empty_codeblock::call_cluster_shared.py::gabion.analysis.call_cluster_shared.render_cluster_heading::call_cluster_shared.py::gabion.analysis.call_cluster_shared.render_string_codeblock
def test_render_cluster_heading_and_non_empty_codeblock() -> None:
    doc = ReportDoc("out_call_cluster_shared")
    render_cluster_heading(doc, display="Cluster", count=2)
    render_string_codeblock(doc, ["t2", "t1"])
    rendered = doc.emit()
    assert "Cluster: Cluster (count: 2)" in rendered
    assert "t1" in rendered and "t2" in rendered


# gabion:evidence E:call_footprint::tests/test_call_cluster_shared.py::test_render_string_codeblock_skips_empty_values::call_cluster_shared.py::gabion.analysis.call_cluster_shared.render_string_codeblock
def test_render_string_codeblock_skips_empty_values() -> None:
    doc = ReportDoc("out_call_cluster_shared_empty")
    render_string_codeblock(doc, [])
    assert "```" not in doc.emit()
