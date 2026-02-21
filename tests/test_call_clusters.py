from __future__ import annotations

from pathlib import Path

from gabion.analysis import call_clusters
from gabion.analysis.projection_spec import ProjectionOp, ProjectionSpec


# gabion:evidence E:function_site::call_clusters.py::gabion.analysis.call_clusters.build_call_clusters_payload
# gabion:evidence E:function_site::call_clusters.py::gabion.analysis.call_clusters.render_markdown
def test_call_clusters_payload_and_render(
    tmp_path: Path,
    write_test_evidence_payload,
    test_evidence_path: Path,
) -> None:
    src_dir = tmp_path / "src" / "pkg"
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").write_text("")
    (src_dir / "core.py").write_text("def helper(x):\n    return x\n")

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_core.py").write_text(
        "from pkg.core import helper\n\ndef test_helper():\n    assert helper(1) == 1\n"
    )

    evidence_path = test_evidence_path
    entries = [
        {
            "test_id": "tests/test_core.py::test_helper",
            "file": "tests/test_core.py",
            "line": 1,
            "evidence": [],
            "status": "unmapped",
        }
    ]
    write_test_evidence_payload(evidence_path, entries=entries)

    payload = call_clusters.build_call_clusters_payload(
        [tests_dir, tmp_path / "src"],
        root=tmp_path,
        evidence_path=evidence_path,
    )
    assert payload["summary"]["clusters"] == 1
    assert payload["summary"]["tests"] == 1
    assert payload["clusters"]
    cluster = payload["clusters"][0]
    assert cluster["key"]["k"] == "call_cluster"
    assert cluster["count"] == 1
    markdown = call_clusters.render_markdown(payload)
    assert "generated_by_spec_id" in markdown


# gabion:evidence E:function_site::call_clusters.py::gabion.analysis.call_clusters.build_call_clusters_payload
def test_call_clusters_payload_handles_empty_targets(
    tmp_path: Path,
    write_test_evidence_payload,
    test_evidence_path: Path,
) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_empty.py").write_text("def test_empty():\n    assert True\n")

    evidence_path = test_evidence_path
    entries = [
        {
            "test_id": "tests/test_empty.py::test_empty",
            "file": "tests/test_empty.py",
            "line": 1,
            "evidence": [],
            "status": "unmapped",
        }
    ]
    write_test_evidence_payload(evidence_path, entries=entries)

    payload = call_clusters.build_call_clusters_payload(
        [tests_dir],
        root=tmp_path,
        evidence_path=evidence_path,
    )
    assert payload["summary"]["clusters"] == 0
    assert payload["summary"]["tests"] == 0


# gabion:evidence E:function_site::call_clusters.py::gabion.analysis.call_clusters.build_call_clusters_payload
def test_call_clusters_payload_projection_skips_unknown_identity(
    tmp_path: Path,
    write_test_evidence_payload,
    test_evidence_path: Path,
) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "mod.py").write_text("def helper(x):\n    return x\n")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_mod.py").write_text(
        "from mod import helper\n\ndef test_helper():\n    helper(1)\n"
    )

    evidence_path = test_evidence_path
    entries = [
        {
            "test_id": "tests/test_mod.py::test_helper",
            "file": "tests/test_mod.py",
            "line": 1,
            "evidence": [],
            "status": "unmapped",
        }
    ]
    write_test_evidence_payload(evidence_path, entries=entries)

    spec = ProjectionSpec(
        spec_version=1,
        name="call_cluster_summary_test",
        domain="call_clusters",
        pipeline=(ProjectionOp("project", {"fields": ["count"]}),),
    )

    payload = call_clusters.build_call_clusters_payload(
        [tests_dir, src_dir],
        root=tmp_path,
        evidence_path=evidence_path,
        summary_spec=spec,
    )
    assert payload["summary"]["clusters"] == 0
    assert payload["summary"]["tests"] == 0


# gabion:evidence E:function_site::call_clusters.py::gabion.analysis.call_clusters.render_markdown
def test_call_clusters_render_empty() -> None:
    payload = {"summary": {"clusters": 0, "tests": 0}, "clusters": []}
    markdown = call_clusters.render_markdown(payload)
    assert "No call clusters found." in markdown


# gabion:evidence E:function_site::call_clusters.py::gabion.analysis.call_clusters.render_markdown
def test_call_clusters_render_skips_non_mapping() -> None:
    payload = {
        "summary": {"clusters": 1, "tests": 1},
        "clusters": ["bad", {"display": "E:call_cluster", "count": 2, "tests": ["t1"]}],
    }
    markdown = call_clusters.render_markdown(payload)
    assert "Cluster: E:call_cluster" in markdown


# gabion:evidence E:function_site::call_clusters.py::gabion.analysis.call_clusters.write_call_clusters
def test_call_clusters_write_creates_file(tmp_path: Path) -> None:
    payload = {"summary": {"clusters": 0, "tests": 0}, "clusters": []}
    output_path = tmp_path / "nested" / "call_clusters.json"
    call_clusters.write_call_clusters(payload, output_path=output_path)
    assert output_path.read_text(encoding="utf-8").strip().startswith("{")


# gabion:evidence E:function_site::call_clusters.py::gabion.analysis.call_clusters.render_markdown
def test_call_clusters_render_uses_payload_spec_metadata() -> None:
    payload = {
        "summary": {"clusters": 0, "tests": 0},
        "clusters": [],
        "generated_by_spec_id": "custom-spec-id",
        "generated_by_spec": {"name": "custom", "spec_version": 99},
    }
    markdown = call_clusters.render_markdown(payload)
    assert "generated_by_spec_id: custom-spec-id" in markdown
    assert 'generated_by_spec: {"name":"custom","spec_version":99}' in markdown


# gabion:evidence E:call_footprint::tests/test_call_clusters.py::test_call_clusters_payload_merges_repeated_cluster_identity::call_clusters.py::gabion.analysis.call_clusters.build_call_clusters_payload
def test_call_clusters_payload_merges_repeated_cluster_identity(
    tmp_path: Path,
    write_test_evidence_payload,
    test_evidence_path: Path,
) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "mod.py").write_text("def helper(x):\n    return x\n")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_mod.py").write_text(
        "from mod import helper\n\n"
        "def test_one():\n    helper(1)\n\n"
        "def test_two():\n    helper(2)\n"
    )
    evidence_path = test_evidence_path
    entries = [
        {
            "test_id": "tests/test_mod.py::test_one",
            "file": "tests/test_mod.py",
            "line": 1,
            "evidence": [],
            "status": "unmapped",
        },
        {
            "test_id": "tests/test_mod.py::test_two",
            "file": "tests/test_mod.py",
            "line": 1,
            "evidence": [],
            "status": "unmapped",
        },
    ]
    write_test_evidence_payload(evidence_path, entries=entries)
    payload = call_clusters.build_call_clusters_payload(
        [tests_dir, src_dir],
        root=tmp_path,
        evidence_path=evidence_path,
    )
    assert payload["summary"]["clusters"] == 1
    assert payload["summary"]["tests"] == 2


# gabion:evidence E:call_footprint::tests/test_call_clusters.py::test_call_clusters_render_handles_empty_tests_list::call_clusters.py::gabion.analysis.call_clusters.render_markdown
def test_call_clusters_render_handles_empty_tests_list() -> None:
    markdown = call_clusters.render_markdown(
        {
            "summary": {"clusters": 1, "tests": 0},
            "clusters": [{"display": "Cluster", "count": 0, "tests": []}],
        }
    )
    assert "Cluster: Cluster (count: 0)" in markdown
