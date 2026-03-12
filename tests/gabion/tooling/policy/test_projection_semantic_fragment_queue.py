from __future__ import annotations

import json
from pathlib import Path

from scripts.policy import projection_semantic_fragment_queue


def _policy_check_payload() -> dict[str, object]:
    return {
        "rule_id": "policy_check",
        "status": "pass",
        "violations": [],
        "projection_fiber_semantics": {
            "decision": {
                "rule_id": "projection_fiber.convergence.ok",
                "outcome": "pass",
                "severity": "info",
            },
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
        },
    }


# gabion:evidence E:function_site::test_projection_semantic_fragment_queue.py::tests.gabion.tooling.policy.test_projection_semantic_fragment_queue.test_analyze_emits_landed_and_active_queue_rows
# gabion:behavior primary=desired
def test_analyze_emits_landed_and_active_queue_rows() -> None:
    queue = projection_semantic_fragment_queue.analyze(
        payload=_policy_check_payload(),
        source_artifact="artifacts/out/policy_check_result.json",
    ).as_payload()

    current_state = queue["current_state"]
    assert current_state["decision"]["rule_id"] == "projection_fiber.convergence.ok"
    assert current_state["semantic_row_count"] == 1
    assert current_state["compiled_projection_semantic_bundle_count"] == 1
    assert current_state["semantic_preview_count"] == 1
    assert current_state["compiled_projection_semantic_spec_names"] == [
        "projection_fiber_frontier"
    ]
    items = {item["queue_id"]: item for item in queue["items"]}
    assert items["PSF-001"]["status"] == "landed"
    assert items["PSF-002"]["status"] == "landed"
    assert items["PSF-003"]["status"] == "landed"
    assert items["PSF-004"]["status"] == "in_progress"
    assert items["PSF-005"]["status"] == "queued"
    assert queue["next_queue_ids"] == ["PSF-004", "PSF-005", "PSF-006", "PSF-007"]


# gabion:evidence E:function_site::test_projection_semantic_fragment_queue.py::tests.gabion.tooling.policy.test_projection_semantic_fragment_queue.test_markdown_summary_lists_queue_and_semantic_preview_context
# gabion:behavior primary=desired
def test_markdown_summary_lists_queue_and_semantic_preview_context() -> None:
    queue = projection_semantic_fragment_queue.analyze(
        payload=_policy_check_payload(),
        source_artifact="artifacts/out/policy_check_result.json",
    )

    markdown = projection_semantic_fragment_queue._markdown_summary(queue)
    assert "# Projection Semantic Fragment Queue" in markdown
    assert "decision_rule: `projection_fiber.convergence.ok`" in markdown
    assert "compiled_specs: `projection_fiber_frontier`" in markdown
    assert "| PSF-004 | Phase 4 | in_progress | Friendly-surface convergence via typed ProjectionSpec lowering |" in markdown
    assert "## Semantic Previews" in markdown
    assert "src/gabion/example.py" in markdown


# gabion:evidence E:function_site::test_projection_semantic_fragment_queue.py::tests.gabion.tooling.policy.test_projection_semantic_fragment_queue.test_analyze_accepts_policy_check_result_payloads_directly
# gabion:behavior primary=desired
def test_analyze_accepts_policy_check_result_payloads_directly() -> None:
    queue = projection_semantic_fragment_queue.analyze(
        payload=_policy_check_payload(),
        source_artifact="artifacts/out/policy_check_result.json",
    ).as_payload()

    current_state = queue["current_state"]
    assert current_state["decision"]["rule_id"] == "projection_fiber.convergence.ok"
    assert current_state["semantic_row_count"] == 1
    assert current_state["compiled_projection_semantic_bundle_count"] == 1
    assert current_state["semantic_previews"][0]["path"] == "src/gabion/example.py"


# gabion:evidence E:function_site::test_projection_semantic_fragment_queue.py::tests.gabion.tooling.policy.test_projection_semantic_fragment_queue.test_run_writes_json_and_markdown_outputs
# gabion:behavior primary=desired
def test_run_writes_json_and_markdown_outputs(tmp_path: Path) -> None:
    policy_check = tmp_path / "artifacts/out/policy_check_result.json"
    out = tmp_path / "artifacts/out/projection_semantic_fragment_queue.json"
    markdown_out = tmp_path / "artifacts/out/projection_semantic_fragment_queue.md"
    policy_check.parent.mkdir(parents=True, exist_ok=True)
    policy_check.write_text(json.dumps(_policy_check_payload(), indent=2) + "\n", encoding="utf-8")

    rc = projection_semantic_fragment_queue.run(
        source_artifact_path=policy_check,
        out_path=out,
        markdown_out=markdown_out,
    )

    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["current_state"]["semantic_row_count"] == 1
    assert payload["items"][3]["queue_id"] == "PSF-004"
    assert markdown_out.exists()
