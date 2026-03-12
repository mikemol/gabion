from __future__ import annotations

from pathlib import Path

from gabion.analysis.call_cluster import call_clusters


def _payload(
    *,
    clusters: int = 0,
    tests: int = 0,
    entries: tuple[call_clusters.CallClusterEntry, ...] = (),
    generated_by_spec_id: str = "call_cluster_summary",
    generated_by_spec: dict[str, object] | None = None,
) -> call_clusters.CallClustersPayload:
    spec_payload = generated_by_spec or {
        "name": "call_cluster_summary",
        "spec_version": 1,
    }
    return call_clusters.CallClustersPayload(
        version=call_clusters.CALL_CLUSTER_VERSION,
        summary=call_clusters.CallClustersSummary(clusters=clusters, tests=tests),
        clusters=entries,
        generated_by_spec_id=generated_by_spec_id,
        generated_by_spec={
            str(key): value
            for key, value in spec_payload.items()
        },
    )


def _emitted_payload(
    payload: call_clusters.CallClustersPayload,
) -> dict[str, object]:
    return call_clusters.render_json_payload(payload)


# gabion:evidence E:function_site::call_clusters.py::gabion.analysis.call_clusters.build_call_clusters_payload
# gabion:evidence E:function_site::call_clusters.py::gabion.analysis.call_clusters.render_markdown E:decision_surface/direct::call_clusters.py::gabion.analysis.call_clusters.render_markdown::stale_f2f22f585967_a8050d05
# gabion:behavior primary=desired
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
    assert payload.summary.clusters == 1
    assert payload.summary.tests == 1
    assert payload.clusters
    cluster = payload.clusters[0]
    assert cluster.key["k"] == "call_cluster"
    assert cluster.count == 1
    markdown = call_clusters.render_markdown(payload)
    assert "generated_by_spec_id" in markdown


# gabion:evidence E:function_site::call_clusters.py::gabion.analysis.call_clusters.build_call_clusters_payload E:decision_surface/direct::call_clusters.py::gabion.analysis.call_clusters.build_call_clusters_payload::stale_9c9d25e9aa91_268453e6
# gabion:behavior primary=verboten facets=empty
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
    assert payload.summary.clusters == 0
    assert payload.summary.tests == 0


# gabion:evidence E:function_site::call_clusters.py::gabion.analysis.call_clusters.render_markdown E:decision_surface/direct::call_clusters.py::gabion.analysis.call_clusters.render_markdown::stale_2aeab116568e
# gabion:behavior primary=verboten facets=empty
def test_call_clusters_render_empty() -> None:
    payload = _payload()
    markdown = call_clusters.render_markdown(payload)
    assert "No call clusters found." in markdown


# gabion:evidence E:function_site::server_core/command_orchestrator.py::gabion.server_core.command_orchestrator._emit_analysis_reports
# gabion:behavior primary=desired
def test_call_clusters_emitted_payload_shape() -> None:
    payload = _payload()
    assert _emitted_payload(payload) == {
        "version": call_clusters.CALL_CLUSTER_VERSION,
        "summary": {"clusters": 0, "tests": 0},
        "clusters": [],
        "generated_by_spec_id": "call_cluster_summary",
        "generated_by_spec": {
            "name": "call_cluster_summary",
            "spec_version": 1,
        },
    }


# gabion:evidence E:function_site::call_clusters.py::gabion.analysis.call_clusters.render_markdown E:decision_surface/direct::call_clusters.py::gabion.analysis.call_clusters.render_markdown::stale_18bb6454e9b7
# gabion:behavior primary=desired
def test_call_clusters_render_uses_payload_spec_metadata() -> None:
    payload = _payload(
        generated_by_spec_id="custom-spec-id",
        generated_by_spec={"name": "custom", "spec_version": 99},
    )
    markdown = call_clusters.render_markdown(payload)
    assert "generated_by_spec_id: custom-spec-id" in markdown
    assert 'generated_by_spec: {"name":"custom","spec_version":99}' in markdown


# gabion:evidence E:function_site::call_clusters.py::gabion.analysis.call_clusters.render_json_payload
# gabion:behavior primary=desired
def test_call_clusters_emitted_payload_preserves_identity() -> None:
    payload = _payload(
        clusters=1,
        tests=2,
        entries=(
            call_clusters.CallClusterEntry(
                identity="call-cluster-1",
                key={"k": "call_cluster", "targets": ["pkg.mod:helper"]},
                display="pkg.mod:helper",
                tests=("tests/test_mod.py::test_one", "tests/test_mod.py::test_two"),
                count=2,
            ),
        ),
    )
    wire_payload = call_clusters.render_json_payload(payload)
    assert wire_payload["clusters"] == [
        {
            "identity": "call-cluster-1",
            "key": {"k": "call_cluster", "targets": ["pkg.mod:helper"]},
            "display": "pkg.mod:helper",
            "tests": [
                "tests/test_mod.py::test_one",
                "tests/test_mod.py::test_two",
            ],
            "count": 2,
        }
    ]


# gabion:evidence E:call_footprint::tests/test_call_clusters.py::test_call_clusters_payload_merges_repeated_cluster_identity::call_clusters.py::gabion.analysis.call_clusters.build_call_clusters_payload
# gabion:behavior primary=desired
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
    assert payload.summary.clusters == 1
    assert payload.summary.tests == 2


# gabion:evidence E:function_site::call_clusters.py::gabion.analysis.call_clusters.build_call_clusters_payload
# gabion:behavior primary=desired
def test_call_clusters_payload_uses_execution_ops_for_default_summary_spec(
    tmp_path: Path,
    write_test_evidence_payload,
    test_evidence_path: Path,
    monkeypatch,
) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "mod.py").write_text("def helper(x):\n    return x\n")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_mod.py").write_text(
        "from mod import helper\n\ndef test_helper():\n    helper(1)\n"
    )
    write_test_evidence_payload(
        test_evidence_path,
        entries=[
            {
                "test_id": "tests/test_mod.py::test_helper",
                "file": "tests/test_mod.py",
                "line": 1,
                "evidence": [],
                "status": "unmapped",
            }
        ],
    )

    seen: dict[str, object] = {}
    monkeypatch.setattr(
        call_clusters,
        "_call_cluster_summary_execution_ops",
        lambda: ("typed-summary-op",),
    )

    def _fake_apply_execution_ops(ops, rows):
        seen["ops"] = ops
        seen["rows"] = rows
        return rows

    monkeypatch.setattr(
        call_clusters,
        "apply_execution_ops",
        _fake_apply_execution_ops,
    )

    payload = call_clusters.build_call_clusters_payload(
        [tests_dir, src_dir],
        root=tmp_path,
        evidence_path=test_evidence_path,
    )

    assert seen["ops"] == ("typed-summary-op",)
    rows = seen["rows"]
    assert isinstance(rows, list)
    assert len(rows) == 1
    assert rows[0]["count"] == 1
    assert payload.summary.clusters == 1


# gabion:evidence E:call_footprint::tests/test_call_clusters.py::test_call_clusters_render_handles_empty_tests_list::call_clusters.py::gabion.analysis.call_clusters.render_markdown
# gabion:behavior primary=verboten facets=empty
def test_call_clusters_render_handles_empty_tests_list() -> None:
    markdown = call_clusters.render_markdown(
        _payload(
            clusters=1,
            entries=(
                call_clusters.CallClusterEntry(
                    identity="cluster-1",
                    key={"k": "call_cluster"},
                    display="Cluster",
                    tests=(),
                    count=0,
                ),
            ),
        )
    )
    assert "Cluster: Cluster (count: 0)" in markdown
