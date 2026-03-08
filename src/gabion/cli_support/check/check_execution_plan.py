from __future__ import annotations

from pathlib import Path
from typing import Callable

from gabion.commands.check_contract import CheckArtifactFlags, CheckPolicyFlags
from gabion.plan import (
    ExecutionPlan, ExecutionPlanObligations, ExecutionPlanPolicyMetadata)
from gabion.json_types import JSONObject

DerivedArtifactsFn = Callable[..., list[str]]
ExecutionPlanRequestCtor = Callable[..., object]


def check_derived_artifacts(
    *,
    report: Path,
    decision_snapshot: Path | None,
    artifact_flags: CheckArtifactFlags,
    emit_test_obsolescence_state: bool,
    emit_test_obsolescence_delta: bool,
    emit_test_annotation_drift_delta: bool,
    emit_ambiguity_delta: bool,
    emit_ambiguity_state: bool,
    aspf_trace_json: Path | None,
    aspf_opportunities_json: Path | None,
    aspf_state_json: Path | None,
    aspf_delta_jsonl: Path | None,
    aspf_equivalence_enabled: bool,
) -> list[str]:
    derived = [str(report), "artifacts/out/execution_plan.json"]
    if decision_snapshot is not None:
        derived.append(str(decision_snapshot))
    if artifact_flags.emit_test_obsolescence:
        derived.append("artifacts/out/test_obsolescence_report.json")
    if emit_test_obsolescence_state:
        derived.append("artifacts/out/test_obsolescence_state.json")
    if emit_test_obsolescence_delta:
        derived.append("artifacts/out/test_obsolescence_delta.json")
    if artifact_flags.emit_test_evidence_suggestions:
        derived.append("artifacts/out/test_evidence_suggestions.json")
    if artifact_flags.emit_call_clusters:
        derived.append("artifacts/out/call_clusters.json")
    if artifact_flags.emit_call_cluster_consolidation:
        derived.append("artifacts/out/call_cluster_consolidation.json")
    if artifact_flags.emit_test_annotation_drift:
        derived.append("artifacts/out/test_annotation_drift.json")
    if artifact_flags.emit_semantic_coverage_map:
        derived.append("artifacts/out/semantic_coverage_map.json")
    if emit_test_annotation_drift_delta:
        derived.append("artifacts/out/test_annotation_drift_delta.json")
    if emit_ambiguity_delta:
        derived.append("artifacts/out/ambiguity_delta.json")
    if emit_ambiguity_state:
        derived.append("artifacts/out/ambiguity_state.json")
    aspf_enabled = (
        aspf_trace_json is not None
        or aspf_opportunities_json is not None
        or aspf_state_json is not None
        or aspf_equivalence_enabled
    )
    if aspf_enabled:
        derived.append(
            str(aspf_trace_json)
            if aspf_trace_json is not None
            else "artifacts/out/aspf_trace.json"
        )
        derived.append("artifacts/out/aspf_equivalence.json")
        derived.append(
            str(aspf_opportunities_json)
            if aspf_opportunities_json is not None
            else "artifacts/out/aspf_opportunities.json"
        )
        derived.append(
            str(aspf_state_json)
            if aspf_state_json is not None
            else "artifacts/out/aspf_state.json"
        )
        derived.append(
            str(aspf_delta_jsonl)
            if aspf_delta_jsonl is not None
            else "artifacts/out/aspf_delta.jsonl"
        )
    return derived


def build_check_execution_plan_request(
    *,
    payload: JSONObject,
    report: Path,
    decision_snapshot: Path | None,
    baseline: Path | None,
    baseline_write: bool,
    policy: CheckPolicyFlags,
    profile: str,
    artifact_flags: CheckArtifactFlags,
    emit_test_obsolescence_state: bool,
    emit_test_obsolescence_delta: bool,
    emit_test_annotation_drift_delta: bool,
    emit_ambiguity_delta: bool,
    emit_ambiguity_state: bool,
    aspf_trace_json: Path | None = None,
    aspf_opportunities_json: Path | None = None,
    aspf_state_json: Path | None = None,
    aspf_delta_jsonl: Path | None = None,
    aspf_equivalence_enabled: bool = False,
    check_derived_artifacts_fn: DerivedArtifactsFn,
    execution_plan_request_ctor: ExecutionPlanRequestCtor,
    dataflow_command: str,
    check_command: str,
) -> object:
    operations = [dataflow_command, check_command]
    obligations = ExecutionPlanObligations(
        preconditions=[
            "input paths resolve under root",
            "analysis timeout budget is configured",
        ],
        postconditions=[
            "exit_code reflects policy gates",
            "execution plan artifact is emitted",
        ],
    )
    baseline_mode = "read"
    if baseline is None:
        baseline_mode = "none"
    elif baseline_write:
        baseline_mode = "write"
    policy_metadata = ExecutionPlanPolicyMetadata(
        deadline={
            "analysis_timeout_ticks": int(payload.get("analysis_timeout_ticks") or 0),
            "analysis_timeout_tick_ns": int(payload.get("analysis_timeout_tick_ns") or 0),
        },
        baseline_mode=baseline_mode,
        docflow_mode="disabled",
    )
    plan = ExecutionPlan(
        requested_operations=operations,
        inputs=dict(payload),
        derived_artifacts=check_derived_artifacts_fn(
            report=report,
            decision_snapshot=decision_snapshot,
            artifact_flags=artifact_flags,
            emit_test_obsolescence_state=emit_test_obsolescence_state,
            emit_test_obsolescence_delta=emit_test_obsolescence_delta,
            emit_test_annotation_drift_delta=emit_test_annotation_drift_delta,
            emit_ambiguity_delta=emit_ambiguity_delta,
            emit_ambiguity_state=emit_ambiguity_state,
            aspf_trace_json=aspf_trace_json,
            aspf_opportunities_json=aspf_opportunities_json,
            aspf_state_json=aspf_state_json,
            aspf_delta_jsonl=aspf_delta_jsonl,
            aspf_equivalence_enabled=aspf_equivalence_enabled,
        ),
        obligations=obligations,
        policy_metadata=policy_metadata,
    )
    plan_payload = plan.as_json_dict()
    plan_payload["policy_metadata"] = dict(plan_payload["policy_metadata"])
    plan_payload["policy_metadata"]["check_profile"] = profile
    plan_payload["policy_metadata"]["fail_on_violations"] = bool(
        policy.fail_on_violations
    )
    plan_payload["policy_metadata"]["fail_on_type_ambiguities"] = bool(
        policy.fail_on_type_ambiguities
    )
    plan_payload["policy_metadata"]["lint"] = bool(policy.lint)
    return execution_plan_request_ctor(
        requested_operations=list(plan_payload["requested_operations"]),
        inputs=dict(plan_payload["inputs"]),
        derived_artifacts=list(plan_payload["derived_artifacts"]),
        obligations=dict(plan_payload["obligations"]),
        policy_metadata=dict(plan_payload["policy_metadata"]),
    )
