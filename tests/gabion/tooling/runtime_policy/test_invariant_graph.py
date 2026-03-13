from __future__ import annotations

import json
from pathlib import Path

from tests.path_helpers import REPO_ROOT

from gabion.tooling.policy_substrate import invariant_graph
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
    projected_ids = [
        str(item.get("object_id", ""))
        for item in workstreams.get("workstreams", [])
        if isinstance(item, dict)
    ]
    assert projected_ids == ["PRF", "PSF-007"]
    prf = next(
        item
        for item in workstreams["workstreams"]
        if isinstance(item, dict) and item.get("object_id") == "PRF"
    )
    assert prf["status"] == "landed"
    assert prf["touchsite_count"] == 0
    psf = next(
        item
        for item in workstreams["workstreams"]
        if isinstance(item, dict) and item.get("object_id") == "PSF-007"
    )
    assert psf["touchsite_count"] == 73
    assert psf["collapsible_touchsite_count"] == 47
    assert psf["surviving_touchsite_count"] == 26


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
