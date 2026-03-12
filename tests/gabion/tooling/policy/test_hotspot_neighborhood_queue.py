from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from scripts.policy import hotspot_neighborhood_queue


def _violation(path: str, render: str) -> dict[str, object]:
    return {"path": path, "render": render}


def _payload() -> dict[str, object]:
    return {
        "format_version": 1,
        "generated_at_utc": "2026-03-09T00:00:00Z",
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
    assert queue["config"]["scoring"] == "balanced_5_family_logsum"
    assert float(first["score"]["ring_1_equal_family_score"]) > 0.0
    assert float(first["score"]["ring_2_equal_family_score"]) > 0.0
    assert math.isclose(
        float(first["score"]["ring_1_balanced_component"]),
        math.log1p(float(first["score"]["ring_1_equal_family_score"])),
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    assert math.isclose(
        float(first["score"]["ring_2_balanced_component"]),
        math.log1p(
            float(first["score"]["ring_2_weight"])
            * float(first["score"]["ring_2_equal_family_score"])
        ),
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    assert math.isclose(
        float(first["score"]["balanced"]),
        float(first["score"]["ring_1_balanced_component"])
        + float(first["score"]["ring_2_balanced_component"]),
        rel_tol=0.0,
        abs_tol=1e-6,
    )
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
    source_artifact = tmp_path / "artifacts/out/hotspot_source_artifact.json"
    out = tmp_path / "artifacts/out/hotspot_neighborhood_queue.json"
    md = tmp_path / "artifacts/out/hotspot_neighborhood_queue.md"
    source_artifact.parent.mkdir(parents=True, exist_ok=True)
    source_artifact.write_text(json.dumps(_payload(), indent=2) + "\n", encoding="utf-8")

    rc = hotspot_neighborhood_queue.run(
        source_artifact_path=source_artifact,
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
    assert payload["source"]["source_generated_at_utc"] == "2026-03-09T00:00:00Z"
    assert "inventory_hash" not in payload["source"]
    assert "rule_set_hash" not in payload["source"]
    assert "policy_results_hash" not in payload["source"]
    assert "changed_scope_hash" not in payload["source"]
    assert payload["counts"]["source_counts"]["branchless"] == 0
    assert payload["counts"]["neighborhood_count"] >= 1
    markdown = md.read_text(encoding="utf-8")
    assert "# Hotspot Neighborhood Queue" in markdown
    assert "## Large-Zone Backlog" not in markdown


def test_main_requires_explicit_source_artifact() -> None:
    with pytest.raises(SystemExit) as excinfo:
        hotspot_neighborhood_queue.main([])
    assert excinfo.value.code == 2


def test_run_from_payload_writes_json_and_markdown_outputs(tmp_path: Path) -> None:
    out = tmp_path / "artifacts/out/hotspot_neighborhood_queue.json"
    md = tmp_path / "artifacts/out/hotspot_neighborhood_queue.md"

    rc = hotspot_neighborhood_queue.run_from_payload(
        payload=_payload(),
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


def test_run_reads_projection_fiber_summary_from_policy_results_payload(
    tmp_path: Path,
) -> None:
    source_artifact = tmp_path / "artifacts/out/hotspot_source_artifact.json"
    out = tmp_path / "artifacts/out/hotspot_neighborhood_queue.json"
    md = tmp_path / "artifacts/out/hotspot_neighborhood_queue.md"
    source_artifact.parent.mkdir(parents=True, exist_ok=True)
    payload = _payload()
    payload["projection_fiber_semantics"] = {
        "decision": {"rule_id": "projection_fiber.convergence.ok"},
        "report": {
            "semantic_rows": [
                {
                    "structural_identity": "row-1",
                    "obligation_state": "discharged",
                    "payload": {
                        "path": "src/gabion/example.py",
                        "qualname": "example.frontier",
                        "structural_path": "example.frontier::branch[0]",
                        "complete": True,
                    },
                }
            ],
            "compiled_projection_semantic_bundles": [
                {
                    "spec_name": "projection_fiber_frontier",
                    "bindings": [
                        {
                            "quotient_face": "projection_fiber.frontier",
                            "source_structural_identity": "row-1",
                        }
                    ],
                }
            ],
        },
    }
    source_artifact.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    rc = hotspot_neighborhood_queue.run(
        source_artifact_path=source_artifact,
        out_path=out,
        markdown_out=md,
    )

    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    source = payload["source"]
    assert source["projection_fiber_decision"]["rule_id"] == "projection_fiber.convergence.ok"
    assert source["projection_fiber_semantic_previews"][0]["path"] == "src/gabion/example.py"


# gabion:evidence E:function_site::test_hotspot_neighborhood_queue.py::tests.gabion.tooling.policy.test_hotspot_neighborhood_queue.test_analyze_carries_projection_fiber_semantic_fields
# gabion:behavior primary=desired
def test_analyze_carries_projection_fiber_semantic_fields() -> None:
    payload = _payload()
    payload["projection_fiber_semantics"] = {
        "decision": {"rule_id": "projection_fiber.convergence.ok"},
        "report": {
            "semantic_rows": [
                {
                    "structural_identity": "row-1",
                    "obligation_state": "discharged",
                    "payload": {
                        "path": "src/gabion/example.py",
                        "qualname": "example.frontier",
                        "structural_path": "example.frontier::branch[0]",
                        "complete": True,
                    },
                }
            ],
            "compiled_projection_semantic_bundles": [
                {
                    "spec_name": "projection_fiber_frontier",
                    "bindings": [
                        {
                            "quotient_face": "projection_fiber.frontier",
                            "source_structural_identity": "row-1",
                        }
                    ],
                }
            ],
        },
    }

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

    source = queue["source"]
    assert source["source_generated_at_utc"] == "2026-03-09T00:00:00Z"
    assert source["projection_fiber_decision"]["rule_id"] == (
        "projection_fiber.convergence.ok"
    )
    assert source["projection_fiber_semantic_bundle_count"] == 1
    assert source["projection_fiber_semantic_preview_count"] == 1
    assert source["projection_fiber_semantic_previews"][0]["path"] == (
        "src/gabion/example.py"
    )
    assert queue["counts"]["source_counts"]["branchless"] == 0


# gabion:evidence E:function_site::test_hotspot_neighborhood_queue.py::tests.gabion.tooling.policy.test_hotspot_neighborhood_queue.test_markdown_summary_includes_projection_fiber_semantic_fields
# gabion:behavior primary=desired
def test_markdown_summary_includes_projection_fiber_semantic_fields() -> None:
    payload = _payload()
    payload["projection_fiber_semantics"] = {
        "decision": {"rule_id": "projection_fiber.convergence.ok"},
        "report": {
            "semantic_rows": [
                {
                    "structural_identity": "row-1",
                    "obligation_state": "discharged",
                    "payload": {
                        "path": "src/gabion/example.py",
                        "qualname": "example.frontier",
                        "structural_path": "example.frontier::branch[0]",
                        "complete": True,
                    },
                }
            ],
            "compiled_projection_semantic_bundles": [
                {
                    "spec_name": "projection_fiber_frontier",
                    "bindings": [
                        {
                            "quotient_face": "projection_fiber.frontier",
                            "source_structural_identity": "row-1",
                        }
                    ],
                }
            ],
        },
    }
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

    markdown = hotspot_neighborhood_queue._markdown_summary(queue)
    assert "projection_fiber_decision: projection_fiber.convergence.ok" in markdown
    assert "projection_fiber_semantic_bundles: 1" in markdown
    assert "## Projection Fiber Semantic Previews" in markdown
    assert "src/gabion/example.py" in markdown


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
