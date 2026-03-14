from __future__ import annotations

import json
from pathlib import Path

from gabion.tooling.policy_substrate import invariant_graph
from gabion.tooling.runtime import invariant_graph as invariant_graph_runtime


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _disable_phase5_enricher(monkeypatch) -> None:
    if hasattr(invariant_graph, "phase5_workstream_registry"):
        monkeypatch.setattr(invariant_graph, "phase5_workstream_registry", lambda: None)
        monkeypatch.setattr(invariant_graph, "prf_workstream_registry", lambda: None)
        monkeypatch.setattr(
            invariant_graph,
            "connectivity_synergy_workstream_registries",
            lambda: (),
        )
        return
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
                "    links=[",
                "        {'kind': 'doc_id', 'value': 'sample_doc'},",
                "        {'kind': 'object_id', 'value': 'OBJ-TODO'},",
                "    ],",
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
    _write(
        tmp_path / "docs" / "sample.md",
        "\n".join(
            [
                "---",
                "doc_id: sample_doc",
                "doc_targets:",
                "  - gabion.sample.decorated",
                "---",
                "",
                "# Sample Doc",
            ]
        ),
    )
    return tmp_path


def test_runtime_invariant_graph_cli_perf_heat_maps_profile_artifacts(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _disable_phase5_enricher(monkeypatch)
    root = _sample_repo(tmp_path)
    (root / "artifacts" / "audit_reports").mkdir(parents=True, exist_ok=True)
    artifact = tmp_path / "artifacts/out/invariant_graph.json"
    workstreams_artifact = tmp_path / "artifacts/out/invariant_workstreams.json"
    ledger_artifact = tmp_path / "artifacts/out/invariant_ledger_projections.json"
    perf_artifact = root / "artifacts" / "audit_reports" / "perf_profile.json"

    assert (
        invariant_graph_runtime.main(
            [
                "--root",
                str(root),
                "--artifact",
                str(artifact),
                "--workstreams-artifact",
                str(workstreams_artifact),
                "--ledger-artifact",
                str(ledger_artifact),
                "build",
            ]
        )
        == 0
    )
    graph = invariant_graph.load_invariant_graph(artifact)
    decorated_node = next(
        node
        for node in graph.nodes
        if node.rel_path == "src/gabion/sample.py" and node.qualname == "decorated"
    )
    perf_artifact.write_text(
        json.dumps(
            {
                "format_version": 1,
                "profiles": [
                    {
                        "profiler": "py-spy",
                        "metric_kind": "wall_samples",
                        "unit": "samples",
                        "samples": [
                            {
                                "artifact_node": {
                                    "wire": "::".join(
                                        (
                                            decorated_node.site_identity,
                                            decorated_node.structural_identity,
                                        )
                                    ),
                                    "site_identity": decorated_node.site_identity,
                                    "structural_identity": (
                                        decorated_node.structural_identity
                                    ),
                                    "rel_path": decorated_node.rel_path,
                                    "qualname": decorated_node.qualname,
                                    "line": decorated_node.line,
                                    "column": decorated_node.column,
                                },
                                "inclusive_value": 31,
                            },
                            {
                                "rel_path": "src/gabion/missing.py",
                                "qualname": "missing",
                                "line": 1,
                                "inclusive_value": 99,
                            },
                        ],
                    },
                    {
                        "profiler": "cProfile",
                        "metric_kind": "cpu_time",
                        "unit": "seconds",
                        "samples": [
                            {
                                "artifact_node": {
                                    "wire": "::".join(
                                        (
                                            decorated_node.site_identity,
                                            decorated_node.structural_identity,
                                        )
                                    ),
                                    "site_identity": decorated_node.site_identity,
                                    "structural_identity": (
                                        decorated_node.structural_identity
                                    ),
                                    "rel_path": decorated_node.rel_path,
                                    "qualname": decorated_node.qualname,
                                    "line": decorated_node.line,
                                    "column": decorated_node.column,
                                },
                                "inclusive_value": 0.42,
                            }
                        ],
                    },
                    {
                        "profiler": "pyinstrument",
                        "metric_kind": "wall_time",
                        "unit": "seconds",
                        "samples": [
                            {
                                "artifact_node": {
                                    "wire": "::".join(
                                        (
                                            decorated_node.site_identity,
                                            decorated_node.structural_identity,
                                        )
                                    ),
                                    "site_identity": decorated_node.site_identity,
                                    "structural_identity": (
                                        decorated_node.structural_identity
                                    ),
                                    "rel_path": decorated_node.rel_path,
                                    "qualname": decorated_node.qualname,
                                    "line": decorated_node.line,
                                    "column": decorated_node.column,
                                },
                                "inclusive_value": 0.99,
                            }
                        ],
                    },
                    {
                        "profiler": "memray",
                        "metric_kind": "allocated_bytes",
                        "unit": "bytes",
                        "samples": [
                            {
                                "rel_path": "src/gabion/sample.py",
                                "qualname": "decorated",
                                "line": decorated_node.line,
                                "inclusive_value": 2048,
                            }
                        ],
                    },
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        invariant_graph_runtime.main(
            [
                "--root",
                str(root),
                "--artifact",
                str(artifact),
                "perf-heat",
                "--id",
                "OBJ-TODO",
                "--perf-artifact",
                str(perf_artifact),
            ]
        )
        == 0
    )

    perf_heat_output = capsys.readouterr().out
    assert "perf_query_overlay:" in perf_heat_output
    assert "doc_ids: sample_doc" in perf_heat_output
    assert "doc_paths: docs/sample.md" in perf_heat_output
    assert "target_symbols: gabion.sample.decorated" in perf_heat_output
    assert "source: dsl_overlay" in perf_heat_output
    assert "matched_profile_observations: 4" in perf_heat_output
    assert "wall_samples:samples total=31" in perf_heat_output
    assert "cpu_time:seconds total=0.42" in perf_heat_output
    assert "wall_time:seconds total=0.99" in perf_heat_output
    assert "allocated_bytes:bytes total=2048" in perf_heat_output
    assert "py-spy :: 31" in perf_heat_output
    assert "cProfile :: 0.42" in perf_heat_output
    assert "pyinstrument :: 0.99" in perf_heat_output
    assert "memray :: 2048" in perf_heat_output
    assert f"src/gabion/sample.py:{decorated_node.line}::decorated" in perf_heat_output
    assert "perf_infimum_buckets:" in perf_heat_output
    assert "- <none>" in perf_heat_output


def test_perf_dsl_overlay_resolves_doc_targets_to_invariant_candidates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _disable_phase5_enricher(monkeypatch)
    root = _sample_repo(tmp_path)
    graph = invariant_graph.build_invariant_graph(root)
    traced = invariant_graph.trace_nodes(graph, "OBJ-TODO")
    descendant_ids = tuple(
        invariant_graph_runtime._sorted(
            list(
                {
                    descendant_id
                    for node in traced
                    for descendant_id in invariant_graph_runtime._descendant_ids(
                        graph,
                        node.node_id,
                    )
                }
            )
        )
    )

    overlay = invariant_graph_runtime._resolve_perf_dsl_overlay(
        root=root,
        graph=graph,
        scope_node_ids=descendant_ids,
    )

    assert overlay.doc_ids == ("sample_doc",)
    assert overlay.doc_paths == ("docs/sample.md",)
    assert overlay.target_symbols == ("gabion.sample.decorated",)
    candidate_nodes = {graph.node_by_id()[node_id] for node_id in overlay.candidate_node_ids}
    assert {node.qualname for node in candidate_nodes} == {"decorated"}


def test_perf_dsl_overlay_reuses_shared_doc_selector_for_inferred_targets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _disable_phase5_enricher(monkeypatch)
    root = _sample_repo(tmp_path)
    _write(
        root / "docs" / "sample.md",
        "\n".join(
            [
                "---",
                "doc_id: sample_doc",
                "---",
                "",
                "Touches `gabion.sample.decorated`.",
            ]
        ),
    )
    graph = invariant_graph.build_invariant_graph(root)
    traced = invariant_graph.trace_nodes(graph, "OBJ-TODO")
    descendant_ids = tuple(
        invariant_graph_runtime._sorted(
            list(
                {
                    descendant_id
                    for node in traced
                    for descendant_id in invariant_graph_runtime._descendant_ids(
                        graph,
                        node.node_id,
                    )
                }
            )
        )
    )

    overlay = invariant_graph_runtime._resolve_perf_dsl_overlay(
        root=root,
        graph=graph,
        scope_node_ids=descendant_ids,
    )

    assert overlay.doc_ids == ("sample_doc",)
    assert overlay.doc_paths == ("docs/sample.md",)
    assert overlay.target_symbols == ("gabion.sample.decorated",)
    candidate_nodes = {graph.node_by_id()[node_id] for node_id in overlay.candidate_node_ids}
    assert {node.qualname for node in candidate_nodes} == {"decorated"}


def test_perf_dsl_overlay_infers_script_symbols_from_shared_selector(tmp_path: Path) -> None:
    _write(
        tmp_path / "scripts" / "policy" / "policy_check.py",
        "def main():\n    return None\n",
    )
    _write(
        tmp_path / "docs" / "perf.md",
        "\n".join(
            [
                "---",
                "doc_id: perf_doc",
                "---",
                "",
                "Touches `scripts.policy.policy_check.main`.",
            ]
        ),
    )
    graph = invariant_graph.InvariantGraph(
        root=str(tmp_path),
        workstream_root_ids=(),
        nodes=(
            invariant_graph.InvariantGraphNode(
                node_id="script-main",
                node_kind="touchsite",
                title="policy_check main",
                marker_name="todo",
                marker_kind="touchsite",
                marker_id="marker:script-main",
                site_identity="site:script-main",
                structural_identity="struct:script-main",
                object_ids=("CSA-IVL-TS-016",),
                doc_ids=("perf_doc",),
                policy_ids=(),
                invariant_ids=(),
                reasoning_summary="script main",
                reasoning_control="script.main",
                blocking_dependencies=(),
                rel_path="scripts/policy/policy_check.py",
                qualname="main",
                line=1,
                column=1,
                ast_node_kind="FunctionDef",
                seam_class="boundary",
                source_marker_node_id="marker-node:script-main",
                status_hint="open",
            ),
        ),
        edges=(),
        diagnostics=(),
    )

    overlay = invariant_graph_runtime._resolve_perf_dsl_overlay(
        root=tmp_path,
        graph=graph,
        scope_node_ids=("script-main",),
    )

    assert overlay.doc_ids == ("perf_doc",)
    assert overlay.doc_paths == ("docs/perf.md",)
    assert overlay.target_symbols == ("scripts.policy.policy_check.main",)
    assert overlay.candidate_node_ids == ("script-main",)


def test_perf_infimum_buckets_use_meet_over_containment_topology() -> None:
    graph = invariant_graph.InvariantGraph(
        root=".",
        workstream_root_ids=("ROOT",),
        nodes=(
            invariant_graph.InvariantGraphNode(
                node_id="root",
                node_kind="synthetic_work_item",
                title="Root Workstream",
                marker_name="todo",
                marker_kind="todo",
                marker_id="root",
                site_identity="site.root",
                structural_identity="struct.root",
                object_ids=("ROOT",),
                doc_ids=(),
                policy_ids=(),
                invariant_ids=(),
                reasoning_summary="root",
                reasoning_control="root",
                blocking_dependencies=(),
                rel_path="",
                qualname="",
                line=0,
                column=0,
                ast_node_kind="",
                seam_class="",
                source_marker_node_id="",
                status_hint="",
            ),
            invariant_graph.InvariantGraphNode(
                node_id="touchpoint",
                node_kind="synthetic_work_item",
                title="Shared Touchpoint",
                marker_name="todo",
                marker_kind="todo",
                marker_id="touchpoint",
                site_identity="site.touchpoint",
                structural_identity="struct.touchpoint",
                object_ids=("ROOT-TP",),
                doc_ids=(),
                policy_ids=(),
                invariant_ids=(),
                reasoning_summary="touchpoint",
                reasoning_control="touchpoint",
                blocking_dependencies=(),
                rel_path="",
                qualname="",
                line=0,
                column=0,
                ast_node_kind="",
                seam_class="",
                source_marker_node_id="",
                status_hint="",
            ),
            invariant_graph.InvariantGraphNode(
                node_id="leaf-a",
                node_kind="synthetic_touchsite",
                title="Leaf A",
                marker_name="todo",
                marker_kind="todo",
                marker_id="leaf-a",
                site_identity="site.leaf_a",
                structural_identity="struct.leaf_a",
                object_ids=("ROOT-TP-TS-A",),
                doc_ids=(),
                policy_ids=(),
                invariant_ids=(),
                reasoning_summary="leaf-a",
                reasoning_control="leaf-a",
                blocking_dependencies=(),
                rel_path="src/gabion/a.py",
                qualname="a",
                line=10,
                column=1,
                ast_node_kind="function_def",
                seam_class="surviving_carrier_seam",
                source_marker_node_id="",
                status_hint="",
            ),
            invariant_graph.InvariantGraphNode(
                node_id="leaf-b",
                node_kind="synthetic_touchsite",
                title="Leaf B",
                marker_name="todo",
                marker_kind="todo",
                marker_id="leaf-b",
                site_identity="site.leaf_b",
                structural_identity="struct.leaf_b",
                object_ids=("ROOT-TP-TS-B",),
                doc_ids=(),
                policy_ids=(),
                invariant_ids=(),
                reasoning_summary="leaf-b",
                reasoning_control="leaf-b",
                blocking_dependencies=(),
                rel_path="src/gabion/b.py",
                qualname="b",
                line=20,
                column=1,
                ast_node_kind="function_def",
                seam_class="surviving_carrier_seam",
                source_marker_node_id="",
                status_hint="",
            ),
        ),
        edges=(
            invariant_graph.InvariantGraphEdge(
                edge_id="root-touchpoint",
                edge_kind="contains",
                source_id="root",
                target_id="touchpoint",
            ),
            invariant_graph.InvariantGraphEdge(
                edge_id="touchpoint-a",
                edge_kind="contains",
                source_id="touchpoint",
                target_id="leaf-a",
            ),
            invariant_graph.InvariantGraphEdge(
                edge_id="touchpoint-b",
                edge_kind="contains",
                source_id="touchpoint",
                target_id="leaf-b",
            ),
        ),
        diagnostics=(),
    )
    matched = (
        invariant_graph_runtime._MatchedProfileObservation(
            node_id="leaf-a",
            profiler="cProfile",
            metric_kind="cpu_time",
            unit="seconds",
            rel_path="src/gabion/a.py",
            qualname="a",
            line=10,
            title="Leaf A",
            inclusive_value=3.0,
        ),
        invariant_graph_runtime._MatchedProfileObservation(
            node_id="leaf-b",
            profiler="cProfile",
            metric_kind="cpu_time",
            unit="seconds",
            rel_path="src/gabion/b.py",
            qualname="b",
            line=20,
            title="Leaf B",
            inclusive_value=5.0,
        ),
    )

    buckets = invariant_graph_runtime._perf_infimum_buckets(
        graph=graph,
        descendant_ids=("root", "touchpoint", "leaf-a", "leaf-b"),
        matched=matched,
    )

    assert buckets
    assert buckets[0].node_id == "touchpoint"
    assert buckets[0].is_global_infimum is True
    assert buckets[0].is_virtual_intersection is True
    assert buckets[0].matched_leaf_node_count == 2
    assert buckets[0].total_inclusive_value == 8.0
