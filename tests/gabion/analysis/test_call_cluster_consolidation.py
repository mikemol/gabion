from __future__ import annotations

from pathlib import Path

from gabion.analysis import call_cluster_consolidation, evidence_keys


def _call_footprint_display(*, test_id: str, file: str, targets: list[tuple[str, str]]) -> str:
    key = evidence_keys.make_call_footprint_key(
        path=file,
        qual=test_id.split("::", 1)[1],
        targets=[{"path": path, "qual": qual} for path, qual in targets],
    )
    return evidence_keys.render_display(evidence_keys.normalize_key(key))


# gabion:evidence E:function_site::call_cluster_consolidation.py::gabion.analysis.call_cluster_consolidation.build_call_cluster_consolidation_payload
# gabion:evidence E:function_site::call_cluster_consolidation.py::gabion.analysis.call_cluster_consolidation.render_markdown E:decision_surface/direct::call_cluster_consolidation.py::gabion.analysis.call_cluster_consolidation.render_markdown::stale_53de2d5d4377_a503856f
def test_call_cluster_consolidation_payload_and_render(
    tmp_path: Path,
    write_test_evidence_payload,
    test_evidence_path: Path,
) -> None:
    evidence_path = test_evidence_path
    shared_targets = [("sample.py", "pkg.fn")]
    display_one = _call_footprint_display(
        test_id="tests/test_sample.py::test_one",
        file="tests/test_sample.py",
        targets=shared_targets,
    )
    display_two = _call_footprint_display(
        test_id="tests/test_sample.py::test_two",
        file="tests/test_sample.py",
        targets=shared_targets,
    )
    display_other = _call_footprint_display(
        test_id="tests/test_sample.py::test_three",
        file="tests/test_sample.py",
        targets=[("other.py", "pkg.other")],
    )
    cluster_key = evidence_keys.make_call_cluster_key(targets=[{"path": "sample.py", "qual": "pkg.fn"}])
    cluster_display = evidence_keys.render_display(evidence_keys.normalize_key(cluster_key))
    entries = [
        {
            "test_id": "tests/test_sample.py::test_one",
            "file": "tests/test_sample.py",
            "line": 10,
            "evidence": [display_one],
            "status": "mapped",
        },
        {
            "test_id": "tests/test_sample.py::test_two",
            "file": "tests/test_sample.py",
            "line": 20,
            "evidence": [display_two],
            "status": "mapped",
        },
        {
            "test_id": "tests/test_sample.py::test_three",
            "file": "tests/test_sample.py",
            "line": 30,
            "evidence": [display_other],
            "status": "mapped",
        },
        {
            "test_id": "tests/test_sample.py::test_four",
            "file": "tests/test_sample.py",
            "line": 40,
            "evidence": [display_one, cluster_display],
            "status": "mapped",
        },
    ]
    write_test_evidence_payload(evidence_path, entries=entries)

    payload = call_cluster_consolidation.build_call_cluster_consolidation_payload(
        evidence_path=evidence_path,
        min_cluster_size=2,
    )
    summary = payload.get("summary", {})
    assert summary.get("clusters") == 1
    assert summary.get("tests") == 2
    assert payload.get("clusters") and payload.get("plan")

    markdown = call_cluster_consolidation.render_markdown(payload)
    assert "Consolidation plan" in markdown
    assert "E:call_cluster" in markdown


# gabion:evidence E:function_site::call_cluster_consolidation.py::gabion.analysis.call_cluster_consolidation.render_markdown E:decision_surface/direct::call_cluster_consolidation.py::gabion.analysis.call_cluster_consolidation.render_markdown::stale_d3f9e19479c9
def test_call_cluster_consolidation_render_empty() -> None:
    payload = {"summary": {}, "clusters": [], "plan": []}
    markdown = call_cluster_consolidation.render_markdown(payload)
    assert "No consolidation candidates" in markdown


# gabion:evidence E:function_site::call_cluster_consolidation.py::gabion.analysis.call_cluster_consolidation.build_call_cluster_consolidation_payload E:decision_surface/direct::call_cluster_consolidation.py::gabion.analysis.call_cluster_consolidation.build_call_cluster_consolidation_payload::stale_f457d4364b38
def test_call_cluster_consolidation_skips_unparseable_and_empty_targets(
    tmp_path: Path,
    write_test_evidence_payload,
    test_evidence_path: Path,
) -> None:
    evidence_path = test_evidence_path
    display_empty = _call_footprint_display(
        test_id="tests/test_sample.py::test_empty",
        file="tests/test_sample.py",
        targets=[],
    )
    entries = [
        {
            "test_id": "tests/test_sample.py::test_invalid",
            "file": "tests/test_sample.py",
            "line": 10,
            "evidence": ["not-a-key"],
            "status": "mapped",
        },
        {
            "test_id": "tests/test_sample.py::test_empty",
            "file": "tests/test_sample.py",
            "line": 20,
            "evidence": [display_empty],
            "status": "mapped",
        },
    ]
    write_test_evidence_payload(evidence_path, entries=entries)

    payload = call_cluster_consolidation.build_call_cluster_consolidation_payload(
        evidence_path=evidence_path,
        min_cluster_size=2,
    )
    assert payload["summary"]["clusters"] == 0
    assert payload["summary"]["tests"] == 0
    assert payload["plan"] == []


# gabion:evidence E:function_site::call_cluster_consolidation.py::gabion.analysis.call_cluster_consolidation.build_call_cluster_consolidation_payload E:decision_surface/direct::call_cluster_consolidation.py::gabion.analysis.call_cluster_consolidation.build_call_cluster_consolidation_payload::stale_0ac9f5d3d039_0e8f5b26
def test_call_cluster_consolidation_skips_multiple_target_sets(
    tmp_path: Path,
    write_test_evidence_payload,
    test_evidence_path: Path,
) -> None:
    evidence_path = test_evidence_path
    display_one = _call_footprint_display(
        test_id="tests/test_sample.py::test_multi",
        file="tests/test_sample.py",
        targets=[("sample.py", "pkg.fn")],
    )
    display_two = _call_footprint_display(
        test_id="tests/test_sample.py::test_multi",
        file="tests/test_sample.py",
        targets=[("other.py", "pkg.other")],
    )
    entries = [
        {
            "test_id": "tests/test_sample.py::test_multi",
            "file": "tests/test_sample.py",
            "line": 10,
            "evidence": [display_one, display_two],
            "status": "mapped",
        }
    ]
    write_test_evidence_payload(evidence_path, entries=entries)

    payload = call_cluster_consolidation.build_call_cluster_consolidation_payload(
        evidence_path=evidence_path,
        min_cluster_size=2,
    )
    assert payload["summary"]["clusters"] == 0
    assert payload["summary"]["tests"] == 0


# gabion:evidence E:call_footprint::tests/test_call_cluster_consolidation.py::test_call_cluster_consolidation_accepts_call_cluster_tokens_in_evidence::call_cluster_consolidation.py::gabion.analysis.call_cluster_consolidation.build_call_cluster_consolidation_payload::evidence_keys.py::gabion.analysis.evidence_keys.make_call_cluster_key::evidence_keys.py::gabion.analysis.evidence_keys.normalize_key::evidence_keys.py::gabion.analysis.evidence_keys.render_display
def test_call_cluster_consolidation_accepts_call_cluster_tokens_in_evidence(
    tmp_path: Path,
    write_test_evidence_payload,
    test_evidence_path: Path,
) -> None:
    evidence_path = test_evidence_path
    cluster_key = evidence_keys.make_call_cluster_key(
        targets=[{"path": "sample.py", "qual": "pkg.fn"}]
    )
    cluster_display = evidence_keys.render_display(evidence_keys.normalize_key(cluster_key))
    entries = [
        {
            "test_id": "tests/test_sample.py::test_cluster_only",
            "file": "tests/test_sample.py",
            "line": 10,
            "evidence": [cluster_display],
            "status": "mapped",
        }
    ]
    write_test_evidence_payload(evidence_path, entries=entries)

    payload = call_cluster_consolidation.build_call_cluster_consolidation_payload(
        evidence_path=evidence_path,
        min_cluster_size=2,
    )
    assert payload["summary"]["clusters"] == 0
    assert payload["summary"]["tests"] == 0


# gabion:evidence E:call_footprint::tests/test_call_cluster_consolidation.py::test_call_cluster_consolidation_ignores_non_cluster_kinds_with_targets::call_cluster_consolidation.py::gabion.analysis.call_cluster_consolidation.build_call_cluster_consolidation_payload::evidence_keys.py::gabion.analysis.evidence_keys.render_display
def test_call_cluster_consolidation_ignores_non_cluster_kinds_with_targets(
    tmp_path: Path,
    write_test_evidence_payload,
    test_evidence_path: Path,
) -> None:
    evidence_path = test_evidence_path
    unknown_display = evidence_keys.render_display(
        {
            "k": "custom_kind",
            "targets": [{"path": "sample.py", "qual": "pkg.fn"}],
        }
    )
    entries = [
        {
            "test_id": "tests/test_sample.py::test_custom",
            "file": "tests/test_sample.py",
            "line": 10,
            "evidence": [unknown_display],
            "status": "mapped",
        }
    ]
    write_test_evidence_payload(evidence_path, entries=entries)

    payload = call_cluster_consolidation.build_call_cluster_consolidation_payload(
        evidence_path=evidence_path,
        min_cluster_size=2,
    )
    assert payload["summary"]["clusters"] == 0
    assert payload["summary"]["tests"] == 0


# gabion:evidence E:call_footprint::tests/test_call_cluster_consolidation.py::test_call_cluster_consolidation_takes_call_cluster_branch_via_payload::call_cluster_consolidation.py::gabion.analysis.call_cluster_consolidation.build_call_cluster_consolidation_payload::evidence_keys.py::gabion.analysis.evidence_keys.make_call_cluster_key::evidence_keys.py::gabion.analysis.evidence_keys.render_display
def test_call_cluster_consolidation_takes_call_cluster_branch_via_payload(
    write_test_evidence_payload,
    test_evidence_path: Path,
) -> None:
    key = evidence_keys.make_call_cluster_key(
        targets=[{"path": "sample.py", "qual": "pkg.fn"}]
    )
    write_test_evidence_payload(
        test_evidence_path,
        entries=[
            {
                "test_id": "tests/test_sample.py::test_call_cluster",
                "file": "tests/test_sample.py",
                "line": 10,
                "evidence": [evidence_keys.render_display(key)],
                "status": "mapped",
            }
        ],
    )
    payload = call_cluster_consolidation.build_call_cluster_consolidation_payload(
        evidence_path=test_evidence_path,
        min_cluster_size=2,
    )
    assert payload["summary"]["clusters"] == 0
    assert payload["summary"]["tests"] == 0


# gabion:evidence E:function_site::call_cluster_consolidation.py::gabion.analysis.call_cluster_consolidation.render_markdown E:decision_surface/direct::call_cluster_consolidation.py::gabion.analysis.call_cluster_consolidation.render_markdown::stale_b48fa52912da
def test_call_cluster_consolidation_render_handles_invalid_entries() -> None:
    payload = {
        "summary": {"clusters": 1},
        "clusters": ["bad", {"identity": "cluster-1", "display": "Cluster", "count": 1}],
        "plan": [
            "bad",
            {
                "cluster_identity": "cluster-1",
                "cluster_display": "Cluster",
                "cluster_count": 1,
                "test_id": "tests/test_sample.py::test_one",
                "file": "tests/test_sample.py",
                "line": 12,
                "replace": "E:call_footprint::sample.py::pkg.fn",
                "with": "not-a-mapping",
            },
        ],
    }
    markdown = call_cluster_consolidation.render_markdown(payload)
    assert "Cluster: Cluster" in markdown
    assert "replace [E:call_footprint::sample.py::pkg.fn]" in markdown


# gabion:evidence E:function_site::call_cluster_consolidation.py::gabion.analysis.call_cluster_consolidation._targets_signature
def test_call_cluster_consolidation_targets_signature_filters_invalid() -> None:
    assert call_cluster_consolidation._targets_signature(None) == ()
    targets = [
        "skip",
        {"path": "", "qual": "q"},
        {"path": "p", "qual": ""},
        {"path": "p", "qual": "q"},
    ]
    assert call_cluster_consolidation._targets_signature(targets) == (("p", "q"),)


# gabion:evidence E:function_site::call_cluster_consolidation.py::gabion.analysis.call_cluster_consolidation.write_call_cluster_consolidation
def test_call_cluster_consolidation_write_creates_file(tmp_path: Path) -> None:
    payload = {"summary": {}, "clusters": [], "plan": []}
    output_path = tmp_path / "nested" / "call_cluster_consolidation.json"
    call_cluster_consolidation.write_call_cluster_consolidation(
        payload, output_path=output_path
    )
    assert output_path.exists()


# gabion:evidence E:call_footprint::tests/test_call_cluster_consolidation.py::test_call_cluster_consolidation_render_accepts_non_list_clusters::call_cluster_consolidation.py::gabion.analysis.call_cluster_consolidation.render_markdown
def test_call_cluster_consolidation_render_accepts_non_list_clusters() -> None:
    markdown = call_cluster_consolidation.render_markdown(
        {
            "summary": {"clusters": 1},
            "clusters": {"unexpected": True},
            "plan": [
                {
                    "cluster_identity": "cluster-1",
                    "cluster_display": "Cluster",
                    "cluster_count": 1,
                    "test_id": "tests/test_sample.py::test_one",
                    "file": "tests/test_sample.py",
                    "line": 12,
                    "replace": [],
                    "with": {"display": "Cluster"},
                }
            ],
        }
    )
    assert "Cluster: Cluster (count: 1)" in markdown


# gabion:evidence E:call_footprint::tests/test_call_cluster_consolidation.py::test_call_cluster_consolidation_render_skips_empty_cluster_identity::call_cluster_consolidation.py::gabion.analysis.call_cluster_consolidation.render_markdown
def test_call_cluster_consolidation_render_skips_empty_cluster_identity() -> None:
    markdown = call_cluster_consolidation.render_markdown(
        {
            "summary": {"clusters": 1},
            "clusters": [{"identity": "", "display": "ignored", "count": 2}],
            "plan": [
                {
                    "cluster_identity": "cluster-x",
                    "cluster_display": "fallback",
                    "cluster_count": 2,
                    "test_id": "tests/test_sample.py::test_one",
                    "file": "tests/test_sample.py",
                    "line": 12,
                    "replace": [],
                    "with": {"display": "fallback"},
                }
            ],
        }
    )
    assert "Cluster: fallback (count: 2)" in markdown
