from __future__ import annotations

import json
from pathlib import Path

from gabion.commands.boundary_order import CanonicalBoundaryDict
from gabion.plan import ExecutionPlan, write_execution_plan_artifact


# gabion:evidence E:call_footprint::tests/test_plan.py::test_write_execution_plan_artifact_serializes_boundary_carriers::plan.py::gabion.plan.write_execution_plan_artifact
def test_write_execution_plan_artifact_serializes_boundary_carriers(
    tmp_path: Path,
) -> None:
    class _FallbackValue:
        def __str__(self) -> str:
            return "fallback-value"

    aux_operation = CanonicalBoundaryDict(source="tests.plan.aux")
    aux_operation["domain"] = "obsolescence"
    aux_operation["action"] = "delta"

    inputs = CanonicalBoundaryDict(source="tests.plan.inputs")
    inputs["aux_operation"] = aux_operation
    inputs["checkpoint_path"] = Path("artifacts/out/resume.json")
    inputs["tuple_value"] = ("alpha", 3)
    inputs["set_value"] = {"beta", "alpha"}
    inputs["fallback_value"] = _FallbackValue()

    plan = ExecutionPlan(
        requested_operations=["gabion.check"],
        inputs=inputs,
    )
    artifact_path = write_execution_plan_artifact(plan, root=tmp_path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload["requested_operations"] == ["gabion.check"]
    assert payload["inputs"]["aux_operation"] == {
        "domain": "obsolescence",
        "action": "delta",
    }
    assert payload["inputs"]["checkpoint_path"] == "artifacts/out/resume.json"
    assert payload["inputs"]["tuple_value"] == ["alpha", 3]
    assert payload["inputs"]["set_value"] == ["alpha", "beta"]
    assert payload["inputs"]["fallback_value"] == "fallback-value"
