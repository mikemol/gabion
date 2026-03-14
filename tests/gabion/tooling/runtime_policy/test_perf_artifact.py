from __future__ import annotations

import cProfile
import importlib.util
import json
from pathlib import Path

from gabion.tooling.policy_substrate import invariant_graph
from gabion.tooling.runtime.perf_artifact import build_cprofile_perf_artifact_payload


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"failed to load module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_cprofile_perf_artifact_payload_profiles_repo_function(
    tmp_path: Path,
) -> None:
    sample_path = tmp_path / "src" / "gabion" / "sample.py"
    _write(
        sample_path,
        "\n".join(
            [
                "def decorated() -> int:",
                "    total = 0",
                "    for value in range(500):",
                "        total += value",
                "    return total",
            ]
        )
        + "\n",
    )
    module = _load_module("gabion_sample_perf_basic", sample_path)
    profile = cProfile.Profile()
    profile.runcall(module.decorated)

    payload = build_cprofile_perf_artifact_payload(
        profile=profile,
        root=tmp_path,
        command=("demo",),
        requested_checks=("workflows",),
        returncode=0,
    )

    assert payload["profiles"][0]["profiler"] == "cProfile"
    sample = next(
        item
        for item in payload["profiles"][0]["samples"]
        if item["rel_path"] == "src/gabion/sample.py" and item["qualname"] == "decorated"
    )
    assert sample["inclusive_value"] > 0
    assert "artifact_node" not in sample


def test_build_cprofile_perf_artifact_payload_enriches_structural_identity_from_graph(
    tmp_path: Path,
) -> None:
    sample_path = tmp_path / "src" / "gabion" / "sample.py"
    _write(tmp_path / "src" / "gabion" / "__init__.py", "")
    _write(
        sample_path,
        "\n".join(
            [
                "from gabion.invariants import todo_decorator",
                "",
                "@todo_decorator(",
                "    reason='profile sample',",
                "    owner='gabion.tooling.runtime',",
                "    expiry='2099-01-01',",
                "    reasoning={",
                "        'summary': 'profile sample',",
                "        'control': 'sample.profile',",
                "        'blocking_dependencies': ['OBJ-TODO'],",
                "    },",
                "    links=[{'kind': 'object_id', 'value': 'OBJ-TODO'}],",
                ")",
                "def decorated() -> int:",
                "    total = 0",
                "    for value in range(500):",
                "        total += value",
                "    return total",
            ]
        )
        + "\n",
    )
    module = _load_module("gabion_sample_perf_graph", sample_path)
    profile = cProfile.Profile()
    profile.runcall(module.decorated)

    graph = invariant_graph.build_invariant_graph(tmp_path)
    graph_artifact = tmp_path / "artifacts/out/invariant_graph.json"
    invariant_graph.write_invariant_graph(graph_artifact, graph)
    expected_structural_identity = next(
        node.structural_identity
        for node in graph.nodes
        if node.rel_path == "src/gabion/sample.py" and node.qualname == "decorated"
    )

    payload = build_cprofile_perf_artifact_payload(
        profile=profile,
        root=tmp_path,
        command=("demo",),
        requested_checks=("workflows",),
        returncode=0,
        graph_artifact=graph_artifact,
    )

    sample = next(
        item
        for item in payload["profiles"][0]["samples"]
        if item["artifact_node"]["rel_path"] == "src/gabion/sample.py"
        and item["artifact_node"]["qualname"] == "decorated"
    )
    assert sample["artifact_node"]["site_identity"] == next(
        node.site_identity
        for node in graph.nodes
        if node.rel_path == "src/gabion/sample.py" and node.qualname == "decorated"
    )
    assert sample["artifact_node"]["structural_identity"] == expected_structural_identity
    assert sample["artifact_node"]["wire"] == "::".join(
        (
            sample["artifact_node"]["site_identity"],
            sample["artifact_node"]["structural_identity"],
        )
    )
    assert json.loads(json.dumps(payload))["observation_count"] >= 1
