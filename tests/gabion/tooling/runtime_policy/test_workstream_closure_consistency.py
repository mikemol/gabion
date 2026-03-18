from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from scripts.policy import policy_check
from tests.gabion.tooling.runtime_policy.invariant_graph_test_support import (
    write_minimal_invariant_repo,
)

from gabion.invariants import landed_todo_decorator, todo_decorator
from gabion.tooling.policy_substrate.workstream_registry import (
    RegisteredRootDefinition,
    RegisteredSubqueueDefinition,
    RegisteredTouchpointDefinition,
    WorkstreamRegistry,
    registry_marker_metadata,
    validate_workstream_closure_consistency,
)


def _decorated_symbol(
    *,
    landed: bool,
    reason: str,
    summary: str,
    blocking_dependencies: tuple[str, ...] = (),
) -> Callable[[], None]:
    decorator = landed_todo_decorator if landed else todo_decorator

    @decorator(
        reason=reason,
        owner="tests.gabion.tooling.runtime_policy",
        expiry="2099-01-01",
        reasoning={
            "summary": summary,
            "control": "tests.runtime_policy.workstream_closure_consistency",
            "blocking_dependencies": list(blocking_dependencies),
        },
    )
    def _symbol() -> None:
        return None

    return _symbol


def _root_definition(
    *,
    root_id: str,
    symbol: Callable[[], None],
    subqueue_ids: tuple[str, ...],
    status_hint: str,
) -> RegisteredRootDefinition:
    metadata = registry_marker_metadata(
        symbol,
        surface="test_workstream_root",
        structural_path=f"test.root::{root_id}",
    )
    return RegisteredRootDefinition(
        root_id=root_id,
        title=root_id,
        rel_path=metadata.rel_path,
        qualname=metadata.qualname,
        line=metadata.line,
        site_identity=metadata.site_identity,
        structural_identity=metadata.structural_identity,
        marker_identity=metadata.marker_identity,
        marker_payload=metadata.marker_payload,
        subqueue_ids=subqueue_ids,
        status_hint=status_hint,
    )


def _subqueue_definition(
    *,
    root_id: str,
    subqueue_id: str,
    symbol: Callable[[], None],
    touchpoint_ids: tuple[str, ...],
    status_hint: str,
) -> RegisteredSubqueueDefinition:
    metadata = registry_marker_metadata(
        symbol,
        surface="test_workstream_subqueue",
        structural_path=f"test.subqueue::{subqueue_id}",
    )
    return RegisteredSubqueueDefinition(
        root_id=root_id,
        subqueue_id=subqueue_id,
        title=subqueue_id,
        rel_path=metadata.rel_path,
        qualname=metadata.qualname,
        line=metadata.line,
        site_identity=metadata.site_identity,
        structural_identity=metadata.structural_identity,
        marker_identity=metadata.marker_identity,
        marker_payload=metadata.marker_payload,
        touchpoint_ids=touchpoint_ids,
        status_hint=status_hint,
    )


def _touchpoint_definition(
    *,
    root_id: str,
    subqueue_id: str,
    touchpoint_id: str,
    symbol: Callable[[], None],
    status_hint: str,
) -> RegisteredTouchpointDefinition:
    metadata = registry_marker_metadata(
        symbol,
        surface="test_workstream_touchpoint",
        structural_path=f"test.touchpoint::{touchpoint_id}",
    )
    return RegisteredTouchpointDefinition(
        root_id=root_id,
        subqueue_id=subqueue_id,
        touchpoint_id=touchpoint_id,
        title=touchpoint_id,
        rel_path=metadata.rel_path,
        qualname=metadata.qualname,
        line=metadata.line,
        site_identity=metadata.site_identity,
        structural_identity=metadata.structural_identity,
        marker_identity=metadata.marker_identity,
        marker_payload=metadata.marker_payload,
        status_hint=status_hint,
    )


def _registry(
    *,
    root_landed: bool,
    subqueue_landed: bool,
    touchpoint_landed: bool,
    root_symbol: Callable[[], None] | None = None,
    subqueue_symbol: Callable[[], None] | None = None,
    touchpoint_symbol: Callable[[], None] | None = None,
) -> WorkstreamRegistry:
    root_id = "TEST-ROOT"
    subqueue_id = "TEST-SQ-001"
    touchpoint_id = "TEST-TP-001"
    actual_root_symbol = root_symbol or _decorated_symbol(
        landed=root_landed,
        reason="recorded landed root",
        summary="recorded landed root",
    )
    actual_subqueue_symbol = subqueue_symbol or _decorated_symbol(
        landed=subqueue_landed,
        reason="recorded landed subqueue",
        summary="recorded landed subqueue",
    )
    actual_touchpoint_symbol = touchpoint_symbol or _decorated_symbol(
        landed=touchpoint_landed,
        reason="recorded landed touchpoint",
        summary="recorded landed touchpoint",
    )
    return WorkstreamRegistry(
        root=_root_definition(
            root_id=root_id,
            symbol=actual_root_symbol,
            subqueue_ids=(subqueue_id,),
            status_hint="landed" if root_landed else "in_progress",
        ),
        subqueues=(
            _subqueue_definition(
                root_id=root_id,
                subqueue_id=subqueue_id,
                symbol=actual_subqueue_symbol,
                touchpoint_ids=(touchpoint_id,),
                status_hint="landed" if subqueue_landed else "in_progress",
            ),
        ),
        touchpoints=(
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id=subqueue_id,
                touchpoint_id=touchpoint_id,
                symbol=actual_touchpoint_symbol,
                status_hint="landed" if touchpoint_landed else "in_progress",
            ),
        ),
    )


# gabion:behavior primary=desired
def test_validate_workstream_closure_consistency_accepts_well_formed_landed_registry() -> None:
    violations = validate_workstream_closure_consistency(
        (_registry(root_landed=True, subqueue_landed=True, touchpoint_landed=True),)
    )
    assert violations == ()


# gabion:behavior primary=desired
def test_validate_workstream_closure_consistency_flags_landed_active_lifecycle() -> None:
    registry = _registry(
        root_landed=True,
        subqueue_landed=True,
        touchpoint_landed=True,
        root_symbol=_decorated_symbol(
            landed=False,
            reason="recorded landed root",
            summary="recorded landed root",
            blocking_dependencies=("dep-a",),
        ),
    )
    violations = validate_workstream_closure_consistency((registry,))
    assert {item.code for item in violations} >= {
        "landed_requires_landed_lifecycle",
        "landed_forbids_blocking_dependencies",
    }


# gabion:behavior primary=desired
def test_validate_workstream_closure_consistency_flags_landed_active_language() -> None:
    registry = _registry(
        root_landed=True,
        subqueue_landed=True,
        touchpoint_landed=True,
        root_symbol=_decorated_symbol(
            landed=True,
            reason="remains active until closure arrives",
            summary="still drift across parallel surfaces",
        ),
    )
    violations = validate_workstream_closure_consistency((registry,))
    assert {item.code for item in violations} >= {"landed_requires_closed_language"}


# gabion:behavior primary=desired
def test_validate_workstream_closure_consistency_flags_landed_parent_with_open_descendant() -> None:
    registry = _registry(
        root_landed=True,
        subqueue_landed=False,
        touchpoint_landed=False,
    )
    violations = validate_workstream_closure_consistency((registry,))
    assert {item.code for item in violations} >= {
        "landed_parent_has_nonlanded_descendant",
    }


# gabion:behavior primary=desired
def test_policy_check_workflows_emits_workstream_closure_drift(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    repo_root = write_minimal_invariant_repo(tmp_path)
    output = tmp_path / "policy_check_result.json"
    registry = _registry(
        root_landed=True,
        subqueue_landed=True,
        touchpoint_landed=True,
        root_symbol=_decorated_symbol(
            landed=False,
            reason="recorded landed root",
            summary="recorded landed root",
            blocking_dependencies=("dep-a",),
        ),
    )
    monkeypatch.setattr(policy_check, "check_workflows", lambda: None)
    monkeypatch.setattr(policy_check, "check_aspf_taint_crosswalk_ack", lambda: None)
    monkeypatch.setattr(policy_check, "check_policy_dsl", lambda: None)
    monkeypatch.setattr(
        policy_check,
        "collect_aspf_lattice_convergence_result",
        lambda: policy_check.ProjectionFiberLatticeConvergenceResult(
            decision_rule_id="projection_fiber.convergence.ok",
            decision_outcome="pass",
            decision_severity="info",
            decision_message="projection fiber clean",
            report_payload={},
            error_messages=(),
        ),
    )
    monkeypatch.setattr(
        policy_check,
        "_write_invariant_graph_artifact",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        policy_check,
        "_write_projection_semantic_fragment_queue_artifacts",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        policy_check,
        "_write_git_state_artifact",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        policy_check,
        "_write_cross_origin_witness_contract_artifact",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        policy_check,
        "_write_kernel_vm_alignment_artifact",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        policy_check,
        "_write_identity_grammar_completion_artifact",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        policy_check,
        "_write_ingress_merge_parity_artifact",
        lambda **_kwargs: None,
    )

    result = policy_check.main(
        [
            "--workflows",
            "--output",
            str(output),
        ],
        repo_root=repo_root,
        invariant_declared_registries=(registry,),
    )

    assert result == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "fail"
    assert payload["violations"][0]["kind"] == "workstream_closure_drift"
