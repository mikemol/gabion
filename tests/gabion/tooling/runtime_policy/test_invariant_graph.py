from __future__ import annotations

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

    assert projection["queue_id"] == "PSF-007"
    assert projection["remaining_touchsite_count"] == 73
    assert projection["collapsible_touchsite_count"] == 47
    assert projection["surviving_touchsite_count"] == 26
    assert len(projection["subqueues"]) == 5
    assert len(projection["touchpoints"]) == 6


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
