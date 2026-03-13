from __future__ import annotations

import json
from pathlib import Path

from tests.path_helpers import REPO_ROOT

from gabion.analysis.aspf.aspf_lattice_algebra import ReplayableStream
from gabion.tooling.policy_substrate import invariant_graph
from gabion.tooling.policy_substrate.policy_queue_identity import PolicyQueueIdentitySpace
from gabion.tooling.runtime import invariant_graph as invariant_graph_runtime


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _disable_phase5_enricher(monkeypatch) -> None:
    monkeypatch.setattr(invariant_graph, "iter_phase5_queues", lambda: ())
    monkeypatch.setattr(invariant_graph, "iter_phase5_subqueues", lambda: ())
    monkeypatch.setattr(invariant_graph, "iter_phase5_touchpoints", lambda: ())
    monkeypatch.setattr(invariant_graph, "iter_prf_queues", lambda: ())
    monkeypatch.setattr(invariant_graph, "iter_prf_subqueues", lambda: ())


def _sample_repo(tmp_path: Path) -> Path:
    _write(tmp_path / "src" / "gabion" / "__init__.py", "")
    _write(
        tmp_path / "src" / "gabion" / "sample.py",
        "\n".join(
            [
                "from gabion.invariants import never, todo_decorator",
                "",
                "@todo_decorator(",
                "    reason='decorate sample',",
                "    owner='policy',",
                "    expiry='2099-01-01',",
                "    reasoning={",
                "        'summary': 'sample decorated marker',",
                "        'control': 'sample.decorated',",
                "        'blocking_dependencies': ['MISSING-OBJECT'],",
                "    },",
                "    links=[{'kind': 'object_id', 'value': 'OBJ-TODO'}],",
                ")",
                "def decorated() -> None:",
                "    never(",
                "        'callsite marker',",
                "        owner='policy',",
                "        expiry='2099-01-01',",
                "        reasoning={",
                "            'summary': 'sample callsite marker',",
                "            'control': 'sample.callsite',",
                "            'blocking_dependencies': ['ANOTHER-MISSING-OBJECT'],",
                "        },",
                "        links=[{'kind': 'doc_id', 'value': 'sample_doc'}],",
                "    )",
                "    return None",
            ]
        ),
    )
    return tmp_path


def _stream_from_items(items):
    return ReplayableStream(factory=lambda items=tuple(items): iter(items))


def test_build_invariant_graph_scans_decorated_symbols_and_marker_callsites(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _disable_phase5_enricher(monkeypatch)
    root = _sample_repo(tmp_path)

    graph = invariant_graph.build_invariant_graph(root)
    payload = graph.as_payload()

    node_kind_counts = payload["counts"]["node_kind_counts"]
    assert node_kind_counts["decorated_symbol"] == 1
    assert node_kind_counts["marker_callsite"] == 1
    traced = invariant_graph.trace_nodes(graph, "OBJ-TODO")
    assert len(traced) == 2
    assert any(node.node_kind == "decorated_symbol" for node in traced)
    assert any(
        item.code == "unresolved_blocking_dependency"
        and item.raw_dependency == "MISSING-OBJECT"
        for item in graph.diagnostics
    )
    assert any(
        item.code == "unresolved_blocking_dependency"
        and item.raw_dependency == "ANOTHER-MISSING-OBJECT"
        for item in graph.diagnostics
    )


def test_invariant_graph_write_and_load_round_trip(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _disable_phase5_enricher(monkeypatch)
    root = _sample_repo(tmp_path)
    artifact = tmp_path / "artifacts/out/invariant_graph.json"

    graph = invariant_graph.build_invariant_graph(root)
    invariant_graph.write_invariant_graph(artifact, graph)
    reloaded = invariant_graph.load_invariant_graph(artifact)

    assert artifact.exists()
    assert len(reloaded.nodes) == len(graph.nodes)
    assert len(reloaded.edges) == len(graph.edges)
    assert len(reloaded.diagnostics) == len(graph.diagnostics)


def test_build_psf_phase5_projection_matches_current_live_repo_state() -> None:
    graph = invariant_graph.build_invariant_graph(REPO_ROOT)
    projection = invariant_graph.build_psf_phase5_projection(graph)
    workstreams = invariant_graph.build_invariant_workstreams(graph)

    assert not hasattr(invariant_graph, "_PHASE5_SURVIVING_TOUCHSITE_BOUNDARY_NAMES")
    assert projection["queue_id"] == "PSF-007"
    assert projection["remaining_touchsite_count"] == 73
    assert projection["collapsible_touchsite_count"] == 47
    assert projection["surviving_touchsite_count"] == 26
    assert len(projection["subqueues"]) == 5
    assert len(projection["touchpoints"]) == 6
    workstreams_payload = workstreams.as_payload()
    projected_ids = [
        str(item.get("object_id", ""))
        for item in workstreams_payload.get("workstreams", [])
        if isinstance(item, dict)
    ]
    assert projected_ids == ["PRF", "PSF-007"]
    prf = next(
        item
        for item in workstreams_payload["workstreams"]
        if isinstance(item, dict) and item.get("object_id") == "PRF"
    )
    assert prf["status"] == "landed"
    assert prf["touchsite_count"] == 0
    psf = next(
        item
        for item in workstreams_payload["workstreams"]
        if isinstance(item, dict) and item.get("object_id") == "PSF-007"
    )
    assert psf["touchsite_count"] == 73
    assert psf["collapsible_touchsite_count"] == 47
    assert psf["surviving_touchsite_count"] == 26
    assert psf["health_summary"]["covered_touchsite_count"] == 3
    assert psf["health_summary"]["uncovered_touchsite_count"] == 70
    assert psf["health_summary"]["governed_touchsite_count"] == 0
    assert psf["health_summary"]["diagnosed_touchsite_count"] == 0
    assert psf["health_summary"]["ready_touchsite_count"] == 3
    assert psf["health_summary"]["coverage_gap_touchsite_count"] == 70
    assert psf["health_summary"]["policy_blocked_touchsite_count"] == 0
    assert psf["health_summary"]["diagnostic_blocked_touchsite_count"] == 0
    assert psf["health_summary"]["ready_touchpoint_cut_count"] == 1
    assert psf["health_summary"]["coverage_gap_touchpoint_cut_count"] == 5
    assert psf["health_summary"]["ready_subqueue_cut_count"] == 0
    assert psf["health_summary"]["coverage_gap_subqueue_cut_count"] == 5
    assert psf["next_actions"]["recommended_cut"]["object_id"] == "PSF-007-TP-005"
    assert psf["next_actions"]["recommended_cut"]["cut_kind"] == "touchpoint_cut"
    assert psf["next_actions"]["recommended_cut"]["touchsite_count"] == 1
    assert psf["next_actions"]["recommended_ready_cut"]["object_id"] == "PSF-007-TP-005"
    assert psf["next_actions"]["recommended_ready_cut"]["readiness_class"] == (
        "ready_structural"
    )
    assert psf["next_actions"]["recommended_coverage_gap_cut"]["object_id"] == (
        "PSF-007-TP-001"
    )
    assert psf["next_actions"]["recommended_coverage_gap_cut"]["readiness_class"] == (
        "coverage_gap"
    )
    assert psf["next_actions"]["dominant_blocker_class"] == "coverage_gap"
    assert psf["next_actions"]["recommended_remediation_family"] == "structural_cut"
    assert psf["next_actions"]["recommended_policy_blocked_cut"] is None
    assert psf["next_actions"]["recommended_diagnostic_blocked_cut"] is None
    assert psf["next_actions"]["remediation_lanes"][0]["remediation_family"] == (
        "structural_cut"
    )
    assert psf["next_actions"]["remediation_lanes"][0]["best_cut"]["object_id"] == (
        "PSF-007-TP-005"
    )
    assert psf["next_actions"]["remediation_lanes"][1]["remediation_family"] == (
        "coverage_gap"
    )
    assert psf["next_actions"]["remediation_lanes"][1]["best_cut"]["object_id"] == (
        "PSF-007-TP-001"
    )
    assert psf["next_actions"]["ranked_touchpoint_cuts"][0]["object_id"] == "PSF-007-TP-005"
    assert psf["next_actions"]["ranked_touchpoint_cuts"][0]["readiness_class"] == (
        "ready_structural"
    )
    assert psf["next_actions"]["ranked_subqueue_cuts"][0]["object_id"] == "PSF-007-SQ-001"
    assert psf["next_actions"]["ranked_subqueue_cuts"][0]["readiness_class"] == (
        "coverage_gap"
    )


def test_workstream_projection_surfaces_policy_and_diagnostic_remediation_families() -> None:
    space = PolicyQueueIdentitySpace()
    workstream_id = space.workstream_id("WS-SYNTH")
    subqueue_policy = space.subqueue_id("WS-SYNTH-SQ-POLICY")
    subqueue_diag = space.subqueue_id("WS-SYNTH-SQ-DIAG")
    touchpoint_policy = space.touchpoint_id("WS-SYNTH-TP-POLICY")
    touchpoint_diag = space.touchpoint_id("WS-SYNTH-TP-DIAG")
    policy_touchsite = invariant_graph.InvariantTouchsiteProjection(
        object_id=space.touchsite_id("WS-SYNTH-TS-POLICY"),
        touchpoint_id=touchpoint_policy,
        subqueue_id=subqueue_policy,
        title="policy touchsite",
        status="in_progress",
        rel_path="src/gabion/sample.py",
        qualname="policy_touchsite",
        boundary_name="policy_touchsite",
        line=10,
        column=1,
        node_kind="marker_callsite",
        site_identity=space.site_ref_id("site.policy"),
        structural_identity=space.structural_ref_id("struct.policy"),
        seam_class="surviving_carrier_seam",
        touchpoint_marker_identity="tp.policy",
        touchpoint_structural_identity=space.structural_ref_id("tp.struct.policy"),
        subqueue_marker_identity="sq.policy",
        subqueue_structural_identity=space.structural_ref_id("sq.struct.policy"),
        policy_signal_count=1,
        coverage_count=1,
        diagnostic_count=0,
        object_ids=(),
    )
    diagnostic_touchsite = invariant_graph.InvariantTouchsiteProjection(
        object_id=space.touchsite_id("WS-SYNTH-TS-DIAG"),
        touchpoint_id=touchpoint_diag,
        subqueue_id=subqueue_diag,
        title="diagnostic touchsite",
        status="in_progress",
        rel_path="src/gabion/sample.py",
        qualname="diagnostic_touchsite",
        boundary_name="diagnostic_touchsite",
        line=20,
        column=1,
        node_kind="marker_callsite",
        site_identity=space.site_ref_id("site.diagnostic"),
        structural_identity=space.structural_ref_id("struct.diagnostic"),
        seam_class="surviving_carrier_seam",
        touchpoint_marker_identity="tp.diagnostic",
        touchpoint_structural_identity=space.structural_ref_id("tp.struct.diagnostic"),
        subqueue_marker_identity="sq.diagnostic",
        subqueue_structural_identity=space.structural_ref_id("sq.struct.diagnostic"),
        policy_signal_count=0,
        coverage_count=1,
        diagnostic_count=1,
        object_ids=(),
    )
    projection = invariant_graph.InvariantWorkstreamProjection(
        object_id=workstream_id,
        title="synthetic workstream",
        status="in_progress",
        site_identity=space.site_ref_id("ws.site"),
        structural_identity=space.structural_ref_id("ws.struct"),
        marker_identity="ws.marker",
        reasoning_summary="synthetic",
        reasoning_control="synthetic.control",
        blocking_dependencies=(),
        object_ids=(),
        touchsite_count=2,
        collapsible_touchsite_count=0,
        surviving_touchsite_count=2,
        policy_signal_count=1,
        coverage_count=2,
        diagnostic_count=1,
        subqueues=_stream_from_items(
            (
                invariant_graph.InvariantSubqueueProjection(
                    object_id=subqueue_policy,
                    title="policy subqueue",
                    status="in_progress",
                    site_identity=space.site_ref_id("sq.site.policy"),
                    structural_identity=space.structural_ref_id("sq.struct.policy.root"),
                    marker_identity="sq.policy.marker",
                    reasoning_summary="policy",
                    reasoning_control="policy",
                    blocking_dependencies=(),
                    object_ids=(),
                    touchpoint_ids=(touchpoint_policy,),
                    touchsite_count=1,
                    collapsible_touchsite_count=0,
                    surviving_touchsite_count=1,
                    policy_signal_count=1,
                    coverage_count=1,
                    diagnostic_count=0,
                ),
                invariant_graph.InvariantSubqueueProjection(
                    object_id=subqueue_diag,
                    title="diagnostic subqueue",
                    status="in_progress",
                    site_identity=space.site_ref_id("sq.site.diagnostic"),
                    structural_identity=space.structural_ref_id("sq.struct.diagnostic.root"),
                    marker_identity="sq.diagnostic.marker",
                    reasoning_summary="diagnostic",
                    reasoning_control="diagnostic",
                    blocking_dependencies=(),
                    object_ids=(),
                    touchpoint_ids=(touchpoint_diag,),
                    touchsite_count=1,
                    collapsible_touchsite_count=0,
                    surviving_touchsite_count=1,
                    policy_signal_count=0,
                    coverage_count=1,
                    diagnostic_count=1,
                ),
            )
        ),
        touchpoints=_stream_from_items(
            (
                invariant_graph.InvariantTouchpointProjection(
                    object_id=touchpoint_policy,
                    subqueue_id=subqueue_policy,
                    title="policy touchpoint",
                    status="in_progress",
                    rel_path="src/gabion/sample.py",
                    site_identity=space.site_ref_id("tp.site.policy"),
                    structural_identity=space.structural_ref_id("tp.struct.policy.root"),
                    marker_identity="tp.policy.marker",
                    reasoning_summary="policy",
                    reasoning_control="policy",
                    blocking_dependencies=(),
                    object_ids=(),
                    touchsite_count=1,
                    collapsible_touchsite_count=0,
                    surviving_touchsite_count=1,
                    policy_signal_count=1,
                    coverage_count=1,
                    diagnostic_count=0,
                    touchsites=_stream_from_items((policy_touchsite,)),
                ),
                invariant_graph.InvariantTouchpointProjection(
                    object_id=touchpoint_diag,
                    subqueue_id=subqueue_diag,
                    title="diagnostic touchpoint",
                    status="in_progress",
                    rel_path="src/gabion/sample.py",
                    site_identity=space.site_ref_id("tp.site.diagnostic"),
                    structural_identity=space.structural_ref_id("tp.struct.diagnostic.root"),
                    marker_identity="tp.diagnostic.marker",
                    reasoning_summary="diagnostic",
                    reasoning_control="diagnostic",
                    blocking_dependencies=(),
                    object_ids=(),
                    touchsite_count=1,
                    collapsible_touchsite_count=0,
                    surviving_touchsite_count=1,
                    policy_signal_count=0,
                    coverage_count=1,
                    diagnostic_count=1,
                    touchsites=_stream_from_items((diagnostic_touchsite,)),
                ),
            )
        ),
    )

    health_summary = projection.health_summary()
    assert health_summary.ready_touchsite_count == 0
    assert health_summary.coverage_gap_touchsite_count == 0
    assert health_summary.policy_blocked_touchsite_count == 1
    assert health_summary.diagnostic_blocked_touchsite_count == 1
    assert projection.dominant_blocker_class() == "diagnostic_blocked"
    assert projection.recommended_remediation_family() == "diagnostic_blocked"
    assert projection.recommended_policy_blocked_cut() is not None
    assert projection.recommended_policy_blocked_cut().object_id.wire() == (
        "WS-SYNTH-TP-POLICY"
    )
    assert projection.recommended_diagnostic_blocked_cut() is not None
    assert projection.recommended_diagnostic_blocked_cut().object_id.wire() == (
        "WS-SYNTH-TP-DIAG"
    )
    payload = projection.as_payload()
    assert payload["next_actions"]["dominant_blocker_class"] == "diagnostic_blocked"
    assert payload["next_actions"]["recommended_remediation_family"] == (
        "diagnostic_blocked"
    )
    assert payload["next_actions"]["recommended_policy_blocked_cut"]["object_id"] == (
        "WS-SYNTH-TP-POLICY"
    )
    assert payload["next_actions"]["recommended_diagnostic_blocked_cut"]["object_id"] == (
        "WS-SYNTH-TP-DIAG"
    )
    assert payload["next_actions"]["remediation_lanes"][0]["remediation_family"] == (
        "diagnostic_blocked"
    )
    assert payload["next_actions"]["remediation_lanes"][0]["best_cut"]["object_id"] == (
        "WS-SYNTH-TP-DIAG"
    )
    assert payload["next_actions"]["remediation_lanes"][1]["remediation_family"] == (
        "policy_blocked"
    )
    assert payload["next_actions"]["remediation_lanes"][1]["best_cut"]["object_id"] == (
        "WS-SYNTH-TP-POLICY"
    )


def test_runtime_invariant_graph_cli_build_summary_trace_and_blockers(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _disable_phase5_enricher(monkeypatch)
    root = _sample_repo(tmp_path)
    artifact = tmp_path / "artifacts/out/invariant_graph.json"

    assert (
        invariant_graph_runtime.main(
            ["--root", str(root), "--artifact", str(artifact), "build"]
        )
        == 0
    )
    assert artifact.exists()

    assert (
        invariant_graph_runtime.main(
            ["--root", str(root), "--artifact", str(artifact), "summary"]
        )
        == 0
    )
    summary_output = capsys.readouterr().out
    assert "nodes:" in summary_output
    assert "edges:" in summary_output

    assert (
        invariant_graph_runtime.main(
            ["--root", str(root), "--artifact", str(artifact), "trace", "--id", "OBJ-TODO"]
        )
        == 0
    )
    trace_output = capsys.readouterr().out
    assert "marker_id:" in trace_output
    assert "ownership_chain:" in trace_output

    assert (
        invariant_graph_runtime.main(
            [
                "--artifact",
                str(artifact),
                "blockers",
                "--object-id",
                "PSF-007",
            ]
        )
        == 1
    )
    blocker_output = capsys.readouterr().out
    assert "no blocker chains" in blocker_output


def test_runtime_invariant_graph_cli_blast_radius_flags_impacted_tests(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _disable_phase5_enricher(monkeypatch)
    root = _sample_repo(tmp_path)
    (root / "out").mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / "audit_reports").mkdir(parents=True, exist_ok=True)
    (root / "out" / "test_evidence.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tests": [
                    {
                        "test_id": "tests/test_sample.py::test_decorated",
                        "file": "tests/test_sample.py",
                        "line": 10,
                        "status": "pass",
                        "evidence": [
                            {
                                "key": {
                                    "site": {
                                        "path": "src/gabion/sample.py",
                                        "qual": "decorated",
                                    }
                                }
                            }
                        ],
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    impact_artifact = root / "artifacts" / "audit_reports" / "impact_selection.json"
    impact_artifact.write_text(
        json.dumps(
            {
                "selection": {
                    "impacted_tests": ["tests/test_sample.py::test_decorated"],
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    artifact = tmp_path / "artifacts/out/invariant_graph.json"

    assert (
        invariant_graph_runtime.main(
            ["--root", str(root), "--artifact", str(artifact), "build"]
        )
        == 0
    )
    assert (
        invariant_graph_runtime.main(
            [
                "--root",
                str(root),
                "--artifact",
                str(artifact),
                "blast-radius",
                "--id",
                "OBJ-TODO",
                "--impact-artifact",
                str(impact_artifact),
            ]
        )
        == 0
    )
    blast_radius_output = capsys.readouterr().out
    assert "tests/test_sample.py::test_decorated [impacted]" in blast_radius_output


def test_build_invariant_graph_joins_policy_signals_and_test_coverage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _disable_phase5_enricher(monkeypatch)
    root = _sample_repo(tmp_path)
    (root / "artifacts" / "out").mkdir(parents=True, exist_ok=True)
    (root / "out").mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / "out" / "ambiguity_contract_policy_check.json").write_text(
        json.dumps(
            {
                "ast": {"violations": []},
                "grade": {
                    "violations": [
                        {
                            "rule_id": "GMP-001",
                            "message": "sample grade issue",
                            "path": "src/gabion/sample.py",
                            "line": 14,
                            "qualname": "decorated",
                            "details": {},
                        }
                    ]
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "out" / "test_evidence.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tests": [
                    {
                        "test_id": "tests/test_sample.py::test_decorated",
                        "file": "tests/test_sample.py",
                        "line": 10,
                        "status": "pass",
                        "evidence": [
                            {
                                "key": {
                                    "site": {
                                        "path": "src/gabion/sample.py",
                                        "qual": "decorated",
                                    }
                                }
                            }
                        ],
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    graph = invariant_graph.build_invariant_graph(root)
    node_kind_counts = graph.as_payload()["counts"]["node_kind_counts"]
    assert node_kind_counts["policy_signal"] == 1
    assert node_kind_counts["test_case"] == 1
    edges = {(edge.edge_kind, edge.source_id, edge.target_id) for edge in graph.edges}
    matched_nodes = invariant_graph.trace_nodes(graph, "OBJ-TODO")
    assert any(node.node_kind == "decorated_symbol" for node in matched_nodes)
    decorated_node = next(node for node in matched_nodes if node.node_kind == "decorated_symbol")
    assert any(
        edge_kind == "covered_by" and source_id == decorated_node.node_id
        for edge_kind, source_id, _target_id in edges
    )
    assert any(
        edge_kind == "blocks"
        and target_id == decorated_node.node_id
        for edge_kind, _source_id, target_id in edges
    )


def test_build_invariant_graph_fails_closed_on_declared_workstream_dependency(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _disable_phase5_enricher(monkeypatch)
    root = tmp_path
    _write(root / "src" / "gabion" / "__init__.py", "")
    _write(
        root / "src" / "gabion" / "broken.py",
        "\n".join(
            [
                "from gabion.invariants import todo_decorator",
                "",
                "@todo_decorator(",
                "    reason='broken dependency',",
                "    owner='policy',",
                "    expiry='2099-01-01',",
                "    reasoning={",
                "        'summary': 'declared workstream dependency should fail closed',",
                "        'control': 'broken.dependency',",
                "        'blocking_dependencies': ['PRF-999'],",
                "    },",
                ")",
                "def broken() -> None:",
                "    return None",
            ]
        ),
    )

    import pytest

    with pytest.raises(ValueError, match="declared workstream blocking dependency"):
        invariant_graph.build_invariant_graph(root)


def test_runtime_invariant_graph_cli_blockers_reports_psf007_chains(
    tmp_path: Path,
    capsys,
) -> None:
    artifact = tmp_path / "artifacts/out/invariant_graph.json"

    assert (
        invariant_graph_runtime.main(
            ["--root", str(REPO_ROOT), "--artifact", str(artifact), "build"]
        )
        == 0
    )
    assert (
        invariant_graph_runtime.main(
            [
                "--root",
                str(REPO_ROOT),
                "--artifact",
                str(artifact),
                "blockers",
                "--object-id",
                "PSF-007",
            ]
        )
        == 0
    )
    blocker_output = capsys.readouterr().out
    assert "collapsible_helper_seam:" in blocker_output
    assert "surviving_carrier_seam:" in blocker_output

    assert (
        invariant_graph_runtime.main(
            [
                "--root",
                str(REPO_ROOT),
                "--artifact",
                str(artifact),
                "workstream",
                "--object-id",
                "PSF-007",
            ]
        )
        == 0
    )
    workstream_output = capsys.readouterr().out
    assert "object_id: PSF-007" in workstream_output
    assert "touchsites: 73" in workstream_output
    assert (
        "health_summary: covered=3 :: uncovered=70 :: governed=0 :: diagnosed=0"
        in workstream_output
    )
    assert (
        "touchsite_blockers: ready=3 :: coverage_gap=70 :: policy=0 :: diagnostic=0"
        in workstream_output
    )
    assert (
        "health_cuts: touchpoints(ready=1, coverage_gap=5, policy=0, diagnostic=0) :: subqueues(ready=0, coverage_gap=5, policy=0, diagnostic=0)"
        in workstream_output
    )
    assert "dominant_blocker_class: coverage_gap" in workstream_output
    assert "recommended_remediation_family: structural_cut" in workstream_output
    assert "recommended_cut: touchpoint_cut :: PSF-007-TP-005 :: touchsites=1 :: surviving=1" in workstream_output
    assert (
        "recommended_ready_cut: touchpoint_cut :: PSF-007-TP-005 :: touchsites=1 :: uncovered=0"
        in workstream_output
    )
    assert (
        "recommended_coverage_gap_cut: touchpoint_cut :: PSF-007-TP-001 :: touchsites=4 :: uncovered=4"
        in workstream_output
    )
    assert "recommended_policy_blocked_cut: <none>" in workstream_output
    assert "recommended_diagnostic_blocked_cut: <none>" in workstream_output
    assert "remediation_lanes:" in workstream_output
    assert (
        "- structural_cut :: blocker=ready_structural :: touchsites=3 :: touchpoints=1 :: subqueues=0 :: best=touchpoint_cut::PSF-007-TP-005"
        in workstream_output
    )
    assert (
        "- coverage_gap :: blocker=coverage_gap :: touchsites=70 :: touchpoints=5 :: subqueues=5 :: best=touchpoint_cut::PSF-007-TP-001"
        in workstream_output
    )
    assert "ranked_touchpoint_cuts:" in workstream_output
    assert (
        "- PSF-007-TP-005 :: readiness=ready_structural :: touchsites=1 :: collapsible=0 :: surviving=1 :: uncovered=0"
        in workstream_output
    )
    assert "ranked_subqueue_cuts:" in workstream_output
    assert (
        "- PSF-007-SQ-001 :: readiness=coverage_gap :: touchsites=4 :: collapsible=0 :: surviving=4 :: uncovered=4"
        in workstream_output
    )

    assert (
        invariant_graph_runtime.main(
            [
                "--root",
                str(REPO_ROOT),
                "--artifact",
                str(artifact),
                "blast-radius",
                "--id",
                "PSF-007",
            ]
        )
        == 0
    )
    blast_radius_output = capsys.readouterr().out
    assert "covering_tests:" in blast_radius_output
