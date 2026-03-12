from __future__ import annotations

import json
from pathlib import Path

from scripts.policy import policy_check


# gabion:evidence E:function_site::tests/gabion/tooling/runtime_policy/test_policy_check_output.py::tests.gabion.tooling.runtime_policy.test_policy_check_output.test_policy_check_output_carries_projection_fiber_semantics_on_pass::policy_check.py::scripts.policy.policy_check.main
# gabion:behavior primary=desired
def test_policy_check_output_carries_projection_fiber_semantics_on_pass(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    output = tmp_path / "policy_check_result.json"
    monkeypatch.setattr(policy_check, "check_policy_dsl", lambda: None)
    monkeypatch.setattr(
        policy_check,
        "collect_aspf_lattice_convergence_result",
        lambda: policy_check.ProjectionFiberLatticeConvergenceResult(
            decision_rule_id="projection_fiber.convergence.ok",
            decision_outcome="pass",
            decision_severity="info",
            decision_message="projection fiber clean",
            report_payload={
                "semantic_rows": [
                    {
                        "structural_identity": "row-1",
                        "surface": "projection_fiber",
                    }
                ],
                "compiled_projection_semantic_bundles": [
                    {
                        "spec_name": "projection_fiber_frontier",
                        "bindings": [{"quotient_face": "projection_fiber.frontier"}],
                        "compiled_shacl_plans": [],
                        "compiled_sparql_plans": [],
                    }
                ],
            },
            error_messages=(),
        ),
    )

    result = policy_check.main(
        [
            "--policy-dsl",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["rule_id"] == "policy_check"
    assert payload["status"] == "pass"
    semantics = payload["projection_fiber_semantics"]
    assert semantics["decision"]["rule_id"] == "projection_fiber.convergence.ok"
    assert semantics["report"]["compiled_projection_semantic_bundles"][0]["spec_name"] == (
        "projection_fiber_frontier"
    )
    queue_payload = json.loads(
        (tmp_path / "projection_semantic_fragment_queue.json").read_text(
            encoding="utf-8"
        )
    )
    assert queue_payload["source_artifact"] == str(output)
    assert queue_payload["current_state"]["decision"]["rule_id"] == (
        "projection_fiber.convergence.ok"
    )
    assert (tmp_path / "projection_semantic_fragment_queue.md").exists()


# gabion:evidence E:function_site::tests/gabion/tooling/runtime_policy/test_policy_check_output.py::tests.gabion.tooling.runtime_policy.test_policy_check_output.test_policy_check_output_carries_projection_fiber_semantics_on_block::policy_check.py::scripts.policy.policy_check.main
# gabion:behavior primary=desired
def test_policy_check_output_carries_projection_fiber_semantics_on_block(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    output = tmp_path / "policy_check_result.json"
    monkeypatch.setattr(policy_check, "check_policy_dsl", lambda: None)
    monkeypatch.setattr(
        policy_check,
        "collect_aspf_lattice_convergence_result",
        lambda: policy_check.ProjectionFiberLatticeConvergenceResult(
            decision_rule_id="projection_fiber.convergence.blocking",
            decision_outcome="block",
            decision_severity="error",
            decision_message="projection fiber blocked",
            report_payload={
                "semantic_rows": [],
                "compiled_projection_semantic_bundles": [],
            },
            error_messages=("frontier witness incomplete",),
        ),
    )

    result = policy_check.main(
        [
            "--policy-dsl",
            "--output",
            str(output),
        ]
    )

    assert result == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["rule_id"] == "policy_check"
    assert payload["status"] == "fail"
    assert payload["violations"][0]["message"] == "projection fiber blocked"
    semantics = payload["projection_fiber_semantics"]
    assert semantics["decision"]["outcome"] == "block"
    assert semantics["error_messages"] == ["frontier witness incomplete"]
