from __future__ import annotations

import json
from pathlib import Path

from scripts.policy import hotspot_neighborhood_queue


def _violation(path: str, render: str) -> dict[str, object]:
    return {"path": path, "render": render}


def _payload() -> dict[str, object]:
    return {
        "format_version": 1,
        "generated_at_utc": "2026-03-09T00:00:00Z",
        "inventory_hash": "inv",
        "rule_set_hash": "rules",
        "policy_results_hash": "policy",
        "changed_scope_hash": "scope",
        "counts": {
            "branchless": 0,
            "fiber_filter_processor_contract": 0,
            "fiber_loop_structure_contract": 0,
            "defensive_fallback": 0,
            "fiber_scalar_sentinel_contract": 0,
        },
        "violations": {
            "branchless": [
                _violation("src/gabion/server_core/a.py", "a"),
                _violation("src/gabion/server_core/a.py", "a"),
                _violation("src/gabion/server_core/a.py", "a"),
                _violation("src/gabion/server_core/a.py", "a"),
                _violation("src/gabion/server_core/a.py", "a"),
                _violation("src/gabion/server_core/a.py", "a"),
                _violation("src/gabion/server_core/a.py", "a"),
                _violation("src/gabion/server_core/a.py", "a"),
                _violation("src/gabion/server_core/b.py", "b"),
                _violation("src/gabion/server_core/b.py", "b"),
                _violation("src/gabion/server_core/b.py", "b"),
                _violation("src/gabion/server_core/b.py", "b"),
                _violation("src/gabion/server_core/b.py", "b"),
                _violation("src/gabion/tooling/runtime/run_dataflow_stage.py", "r"),
                _violation("src/gabion/tooling/runtime/run_dataflow_stage.py", "r"),
                _violation("src/gabion/tooling/runtime/run_dataflow_stage.py", "r"),
                _violation("src/gabion/tooling/runtime/run_dataflow_stage.py", "r"),
                _violation("src/gabion/tooling/runtime/run_dataflow_stage.py", "r"),
                _violation("src/gabion/tooling/runtime/run_dataflow_stage.py", "r"),
                _violation("src/gabion/tooling/runtime/run_dataflow_stage.py", "r"),
                _violation("src/gabion/tooling/runtime/run_dataflow_stage.py", "r"),
            ],
            "fiber_filter_processor_contract": [
                _violation("src/gabion/server_core/a.py", "a"),
                _violation("src/gabion/server_core/a.py", "a"),
                _violation("src/gabion/server_core/a.py", "a"),
                _violation("src/gabion/server_core/a.py", "a"),
                _violation("src/gabion/server_core/b.py", "b"),
                _violation("src/gabion/server_core/b.py", "b"),
                _violation("src/gabion/tooling/runtime/run_dataflow_stage.py", "r"),
                _violation("src/gabion/tooling/runtime/run_dataflow_stage.py", "r"),
                _violation("src/gabion/tooling/runtime/run_dataflow_stage.py", "r"),
                _violation("src/gabion/tooling/runtime/run_dataflow_stage.py", "r"),
            ],
            "fiber_loop_structure_contract": [
                _violation("src/gabion/server_core/a.py", "a"),
                _violation("src/gabion/server_core/a.py", "a"),
                _violation("src/gabion/server_core/b.py", "b"),
                _violation("src/gabion/tooling/runtime/run_dataflow_stage.py", "r"),
                _violation("src/gabion/tooling/runtime/run_dataflow_stage.py", "r"),
            ],
            "defensive_fallback": [
                _violation("src/gabion/server_core/a.py", "a"),
                _violation("src/gabion/tooling/runtime/run_dataflow_stage.py", "r"),
            ],
            "fiber_scalar_sentinel_contract": [
                _violation("src/gabion/server_core/a.py", "a"),
                _violation("src/gabion/server_core/b.py", "b"),
                _violation("src/gabion/tooling/runtime/run_dataflow_stage.py", "r"),
            ],
        },
    }


# gabion:evidence E:function_site::test_hotspot_neighborhood_queue.py::tests.gabion.tooling.policy.test_hotspot_neighborhood_queue.test_analyze_builds_deterministic_ranked_neighborhoods
# gabion:behavior primary=desired
def test_analyze_builds_deterministic_ranked_neighborhoods() -> None:
    payload = _payload()
    config = hotspot_neighborhood_queue.QueueConfig(
        min_seed_families=5,
        min_seed_total=5,
        ring2_similarity_threshold=0.99,
        ring2_min_total=5,
        ring2_limit=4,
        ring2_weight=0.35,
    )
    queue = hotspot_neighborhood_queue.analyze(payload=payload, config=config)
    neighborhoods = queue["neighborhoods"]
    assert isinstance(neighborhoods, list)
    assert neighborhoods
    first = neighborhoods[0]
    assert first["ring_1_scope"] == "src/gabion/server_core"
    assert first["seed_path"] == "src/gabion/server_core/a.py"
    assert first["ring_1"]["file_count"] == 2
    assert first["ring_1"]["total"] == 25
    assert float(first["score"]["ring_1_equal_family_score"]) > 0.0
    assert float(first["score"]["ring_2_equal_family_score"]) > 0.0
    ring2_paths = [item["path"] for item in first["ring_2"]]
    assert "src/gabion/tooling/runtime/run_dataflow_stage.py" in ring2_paths


# gabion:evidence E:function_site::test_hotspot_neighborhood_queue.py::tests.gabion.tooling.policy.test_hotspot_neighborhood_queue.test_analyze_uses_single_representative_seed_per_ring1_scope
# gabion:behavior primary=desired
def test_analyze_uses_single_representative_seed_per_ring1_scope() -> None:
    payload = _payload()
    queue = hotspot_neighborhood_queue.analyze(
        payload=payload,
        config=hotspot_neighborhood_queue.QueueConfig(
            min_seed_families=5,
            min_seed_total=5,
            ring2_similarity_threshold=0.99,
            ring2_min_total=5,
            ring2_limit=4,
            ring2_weight=0.35,
        ),
    )
    scopes = [item["ring_1_scope"] for item in queue["neighborhoods"]]
    assert scopes.count("src/gabion/server_core") == 1


# gabion:evidence E:function_site::test_hotspot_neighborhood_queue.py::tests.gabion.tooling.policy.test_hotspot_neighborhood_queue.test_run_writes_json_and_markdown_outputs
# gabion:behavior primary=desired
def test_run_writes_json_and_markdown_outputs(tmp_path: Path) -> None:
    suite = tmp_path / "artifacts/out/policy_suite_results.json"
    out = tmp_path / "artifacts/out/hotspot_neighborhood_queue.json"
    md = tmp_path / "artifacts/out/hotspot_neighborhood_queue.md"
    suite.parent.mkdir(parents=True, exist_ok=True)
    suite.write_text(json.dumps(_payload(), indent=2) + "\n", encoding="utf-8")

    rc = hotspot_neighborhood_queue.run(
        policy_suite_path=suite,
        out_path=out,
        markdown_out=md,
        config=hotspot_neighborhood_queue.QueueConfig(
            min_seed_families=5,
            min_seed_total=5,
            ring2_similarity_threshold=0.99,
            ring2_min_total=5,
            ring2_limit=4,
            ring2_weight=0.35,
        ),
    )

    assert rc == 0
    assert out.exists()
    assert md.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["counts"]["neighborhood_count"] >= 1
    markdown = md.read_text(encoding="utf-8")
    assert "# Hotspot Neighborhood Queue" in markdown
    assert "## Large-Zone Backlog" not in markdown


# gabion:evidence E:function_site::test_hotspot_neighborhood_queue.py::tests.gabion.tooling.policy.test_hotspot_neighborhood_queue.test_analyze_moves_large_scope_to_backlog
# gabion:behavior primary=desired
def test_analyze_moves_large_scope_to_backlog() -> None:
    payload = _payload()
    for index in range(22):
        path = f"src/gabion/analysis/dataflow/engine/f{index:02d}.py"
        for family in hotspot_neighborhood_queue.ACTIVE_FAMILIES:
            payload["violations"][family].append(_violation(path, "x"))
    queue = hotspot_neighborhood_queue.analyze(
        payload=payload,
        config=hotspot_neighborhood_queue.QueueConfig(
            min_seed_families=5,
            min_seed_total=5,
            ring2_similarity_threshold=0.99,
            ring2_min_total=5,
            ring2_limit=4,
            ring2_weight=0.35,
            ring1_backlog_file_threshold=20,
        ),
    )
    scopes = [item["ring_1_scope"] for item in queue["neighborhoods"]]
    assert "src/gabion/analysis/dataflow/engine" not in scopes
    backlog_scopes = [item["ring_1_scope"] for item in queue["large_zone_backlog"]]
    assert "src/gabion/analysis/dataflow/engine" in backlog_scopes
