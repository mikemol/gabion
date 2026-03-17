from __future__ import annotations

from gabion.invariants import landed_todo_decorator, todo_decorator
from gabion.tooling.policy_substrate.workstream_registry import (
    RegisteredRootDefinition,
    RegisteredSubqueueDefinition,
    RegisteredTouchpointDefinition,
    WorkstreamRegistry,
    registry_marker_metadata,
)


@todo_decorator(
    reason="UTR remains active until the full repo pytest suite is green and repo-drain readiness is restored.",
    reasoning={
        "summary": "The synthetic unit-test readiness root tracks current full-suite red-state indicators from junit evidence until repo-drain readiness converges.",
        "control": "unit_test_readiness.root",
        "blocking_dependencies": (
            "UTR-SQ-001",
            "UTR-SQ-002",
            "UTR-SQ-003",
            "UTR-SQ-004",
        ),
    },
    owner="gabion.tooling.runtime_policy",
    expiry="UTR closure",
    links=[{"kind": "object_id", "value": "UTR"}],
)
def _utr_root() -> None:
    return None


@todo_decorator(
    reason="UTR-SQ-001 remains active until helper normalization and strictification regressions stop reaching downstream never() sites.",
    reasoning={
        "summary": "Helper-level runtime-shape regressions across evidence, type, config, and semantic payload surfaces still feed the red suite and must converge before repo-drain readiness closes.",
        "control": "unit_test_readiness.helper_normalization",
        "blocking_dependencies": ("UTR-TP-001", "UTR-TP-002"),
    },
    owner="gabion.analysis",
    expiry="UTR closure",
    links=[
        {"kind": "object_id", "value": "UTR"},
        {"kind": "object_id", "value": "UTR-SQ-001"},
    ],
)
def _utr_sq_helper_normalization() -> None:
    return None


@todo_decorator(
    reason="UTR-SQ-002 remains active until dataflow runtime and legacy monolith execution paths converge on the current deterministic ingress contract.",
    reasoning={
        "summary": "dataflow_raw_runtime and adjacent legacy monolith paths still carry red-suite regressions across normalization, post-phase, and reuse surfaces.",
        "control": "unit_test_readiness.dataflow_runtime",
        "blocking_dependencies": ("UTR-TP-003", "UTR-TP-004"),
    },
    owner="gabion.analysis.dataflow",
    expiry="UTR closure",
    links=[
        {"kind": "object_id", "value": "UTR"},
        {"kind": "object_id", "value": "UTR-SQ-002"},
    ],
)
def _utr_sq_dataflow_runtime() -> None:
    return None


@todo_decorator(
    reason="UTR-SQ-003 remains active until policy and tooling contract regressions stop failing the full repo suite.",
    reasoning={
        "summary": "Policy-check helper APIs, generated-artifact-manifest tooling, and deadline/runtime-policy contract paths still expose red-suite regressions.",
        "control": "unit_test_readiness.policy_tooling",
        "blocking_dependencies": ("UTR-TP-005", "UTR-TP-006", "UTR-TP-007"),
    },
    owner="gabion.tooling",
    expiry="UTR closure",
    links=[
        {"kind": "object_id", "value": "UTR"},
        {"kind": "object_id", "value": "UTR-SQ-003"},
    ],
)
def _utr_sq_policy_tooling() -> None:
    return None


@todo_decorator(
    reason="UTR-SQ-004 remains active until server/runtime envelope regressions stop failing progress and handledness contract tests.",
    reasoning={
        "summary": "Server progress-transition and handledness/LSP envelope regressions still block repo-drain readiness.",
        "control": "unit_test_readiness.server_runtime_envelope",
        "blocking_dependencies": ("UTR-TP-008",),
    },
    owner="gabion.server_core",
    expiry="UTR closure",
    links=[
        {"kind": "object_id", "value": "UTR"},
        {"kind": "object_id", "value": "UTR-SQ-004"},
    ],
)
def _utr_sq_server_runtime_envelope() -> None:
    return None


@landed_todo_decorator(
    reason="UTR-TP-001 is recorded as landed metadata after helper normalization regressions in evidence, type, config, and adjacent helper surfaces were remediated.",
    reasoning={
        "summary": "Helper-level normalization and strictification surfaces now discharge lawful ignored variants structurally, and the cleared helper-regression cluster is recorded as closed touchpoint state.",
        "control": "unit_test_readiness.helper_normalization.touchpoint",
        "blocking_dependencies": (),
    },
    owner="gabion.analysis",
    expiry="UTR closure",
    links=[
        {"kind": "object_id", "value": "UTR"},
        {"kind": "object_id", "value": "UTR-SQ-001"},
        {"kind": "object_id", "value": "UTR-TP-001"},
    ],
)
def _utr_tp_helper_normalization() -> None:
    return None


@todo_decorator(
    reason="UTR-TP-002 remains queued until semantic payload, report, rewrite, and never-invariant helper regressions are remediated.",
    reasoning={
        "summary": "Semantic payload and rewrite/report helper surfaces still assume broader raw object shapes downstream than the current strict contract permits.",
        "control": "unit_test_readiness.semantic_payload.touchpoint",
    },
    owner="gabion.analysis",
    expiry="UTR closure",
    links=[
        {"kind": "object_id", "value": "UTR"},
        {"kind": "object_id", "value": "UTR-SQ-001"},
        {"kind": "object_id", "value": "UTR-TP-002"},
    ],
)
def _utr_tp_semantic_payload_helpers() -> None:
    return None


@todo_decorator(
    reason="UTR-TP-003 remains queued until dataflow_raw_runtime and legacy monolith execution paths stop failing under the current ingress contract.",
    reasoning={
        "summary": "Shared raw runtime normalization and legacy monolith execution surfaces still dominate the red suite under dataflow_s1 and adjacent monolith tests.",
        "control": "unit_test_readiness.dataflow_raw_runtime.touchpoint",
    },
    owner="gabion.analysis.dataflow",
    expiry="UTR closure",
    links=[
        {"kind": "object_id", "value": "UTR"},
        {"kind": "object_id", "value": "UTR-SQ-002"},
        {"kind": "object_id", "value": "UTR-TP-003"},
    ],
)
def _utr_tp_dataflow_raw_runtime() -> None:
    return None


@todo_decorator(
    reason="UTR-TP-004 remains queued until dataflow post-phase, WL, reuse, and exception-obligation collateral regressions converge.",
    reasoning={
        "summary": "Dataflow_s2 and adjacent post-phase/reuse/exception-obligation surfaces still assume broader downstream helper shapes than the current strict contract allows.",
        "control": "unit_test_readiness.dataflow_collateral.touchpoint",
    },
    owner="gabion.analysis.dataflow",
    expiry="UTR closure",
    links=[
        {"kind": "object_id", "value": "UTR"},
        {"kind": "object_id", "value": "UTR-SQ-002"},
        {"kind": "object_id", "value": "UTR-TP-004"},
    ],
)
def _utr_tp_dataflow_collateral() -> None:
    return None


@todo_decorator(
    reason="UTR-TP-005 remains queued until scripts.policy.policy_check contract helpers are restored to the tested surface.",
    reasoning={
        "summary": "policy_check helper API and policy-governance contract tests still fail under the current tool surface.",
        "control": "unit_test_readiness.policy_check.touchpoint",
    },
    owner="scripts.policy",
    expiry="UTR closure",
    links=[
        {"kind": "object_id", "value": "UTR"},
        {"kind": "object_id", "value": "UTR-SQ-003"},
        {"kind": "object_id", "value": "UTR-TP-005"},
    ],
)
def _utr_tp_policy_check_contracts() -> None:
    return None


@todo_decorator(
    reason="UTR-TP-006 remains queued until the generated-artifact-manifest renderer and drift checks are green under the full repo suite.",
    reasoning={
        "summary": "Generated artifact manifest rendering, drift detection, and real-catalog coverage tests still fail under the current implementation.",
        "control": "unit_test_readiness.generated_artifact_manifest.touchpoint",
    },
    owner="gabion.tooling.policy_substrate",
    expiry="UTR closure",
    links=[
        {"kind": "object_id", "value": "UTR"},
        {"kind": "object_id", "value": "UTR-SQ-003"},
        {"kind": "object_id", "value": "UTR-TP-006"},
    ],
)
def _utr_tp_generated_artifact_manifest() -> None:
    return None


@todo_decorator(
    reason="UTR-TP-007 remains queued until deadline/script/runtime-policy path regressions converge on one stable contract surface.",
    reasoning={
        "summary": "Deadline script-policy and runtime-policy artifact paths still expose missing or drifted contract surfaces in the red suite.",
        "control": "unit_test_readiness.deadline_script_policy.touchpoint",
    },
    owner="gabion.tooling.runtime_policy",
    expiry="UTR closure",
    links=[
        {"kind": "object_id", "value": "UTR"},
        {"kind": "object_id", "value": "UTR-SQ-003"},
        {"kind": "object_id", "value": "UTR-TP-007"},
    ],
)
def _utr_tp_deadline_script_policy() -> None:
    return None


@todo_decorator(
    reason="UTR-TP-008 remains queued until server progress-transition and handledness envelope regressions are green.",
    reasoning={
        "summary": "Server-core progress transition and server handledness/LSP contract tests still fail under the current runtime envelope.",
        "control": "unit_test_readiness.server_runtime_envelope.touchpoint",
    },
    owner="gabion.server_core",
    expiry="UTR closure",
    links=[
        {"kind": "object_id", "value": "UTR"},
        {"kind": "object_id", "value": "UTR-SQ-004"},
        {"kind": "object_id", "value": "UTR-TP-008"},
    ],
)
def _utr_tp_server_runtime_envelope() -> None:
    return None


def _root_definition(
    *,
    root_id: str,
    title: str,
    subqueue_ids: tuple[str, ...],
    symbol,
    status_hint: str,
) -> RegisteredRootDefinition:
    metadata = registry_marker_metadata(
        symbol,
        surface="unit_test_readiness_root",
        structural_path=f"unit_test_readiness.root::{root_id}",
    )
    return RegisteredRootDefinition(
        root_id=root_id,
        title=title,
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
    title: str,
    touchpoint_ids: tuple[str, ...],
    symbol,
    status_hint: str,
) -> RegisteredSubqueueDefinition:
    metadata = registry_marker_metadata(
        symbol,
        surface="unit_test_readiness_subqueue",
        structural_path=f"unit_test_readiness.subqueue::{subqueue_id}",
    )
    return RegisteredSubqueueDefinition(
        root_id=root_id,
        subqueue_id=subqueue_id,
        title=title,
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
    title: str,
    symbol,
    status_hint: str,
    test_path_prefixes: tuple[str, ...],
) -> RegisteredTouchpointDefinition:
    metadata = registry_marker_metadata(
        symbol,
        surface="unit_test_readiness_touchpoint",
        structural_path=f"unit_test_readiness.touchpoint::{touchpoint_id}",
    )
    return RegisteredTouchpointDefinition(
        root_id=root_id,
        touchpoint_id=touchpoint_id,
        subqueue_id=subqueue_id,
        title=title,
        rel_path=metadata.rel_path,
        qualname=metadata.qualname,
        line=metadata.line,
        site_identity=metadata.site_identity,
        structural_identity=metadata.structural_identity,
        marker_identity=metadata.marker_identity,
        marker_payload=metadata.marker_payload,
        status_hint=status_hint,
        test_path_prefixes=test_path_prefixes,
    )


def unit_test_readiness_workstream_registry() -> WorkstreamRegistry:
    root_id = "UTR"
    return WorkstreamRegistry(
        root=_root_definition(
            root_id=root_id,
            title="Unit-Test Readiness / Repo-Drain Indicators",
            subqueue_ids=(
                "UTR-SQ-001",
                "UTR-SQ-002",
                "UTR-SQ-003",
                "UTR-SQ-004",
            ),
            symbol=_utr_root,
            status_hint="in_progress",
        ),
        subqueues=(
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="UTR-SQ-001",
                title="Helper normalization and strictification regressions",
                touchpoint_ids=("UTR-TP-001", "UTR-TP-002"),
                symbol=_utr_sq_helper_normalization,
                status_hint="in_progress",
            ),
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="UTR-SQ-002",
                title="Dataflow runtime and legacy monolith regressions",
                touchpoint_ids=("UTR-TP-003", "UTR-TP-004"),
                symbol=_utr_sq_dataflow_runtime,
                status_hint="in_progress",
            ),
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="UTR-SQ-003",
                title="Policy and tooling contract regressions",
                touchpoint_ids=("UTR-TP-005", "UTR-TP-006", "UTR-TP-007"),
                symbol=_utr_sq_policy_tooling,
                status_hint="in_progress",
            ),
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="UTR-SQ-004",
                title="Server/runtime envelope regressions",
                touchpoint_ids=("UTR-TP-008",),
                symbol=_utr_sq_server_runtime_envelope,
                status_hint="in_progress",
            ),
        ),
        touchpoints=(
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="UTR-SQ-001",
                touchpoint_id="UTR-TP-001",
                title="Helper normalization and never() fall-through regressions",
                symbol=_utr_tp_helper_normalization,
                status_hint="landed",
                test_path_prefixes=(
                    "tests/gabion/analysis/evidence/",
                    "tests/gabion/analysis/type/",
                    "tests/gabion/config/",
                    "tests/gabion/analysis/indexed_scan/",
                    "tests/gabion/analysis/structure/",
                    "tests/gabion/analysis/forest/",
                    "tests/gabion/analysis/call_cluster/",
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="UTR-SQ-001",
                touchpoint_id="UTR-TP-002",
                title="Semantic payload, report, and rewrite helper regressions",
                symbol=_utr_tp_semantic_payload_helpers,
                status_hint="queued",
                test_path_prefixes=(
                    "tests/gabion/analysis/misc_s2/",
                    "tests/gabion/analysis/misc_s3/test_never_invariants.py",
                    "tests/gabion/analysis/misc_s3/test_rewrite_plan_verification.py",
                    "tests/gabion/analysis/misc_s3/test_derivation_cache.py",
                    "tests/gabion/analysis/misc_s2/test_semantic_coverage_map.py",
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="UTR-SQ-002",
                touchpoint_id="UTR-TP-003",
                title="dataflow_raw_runtime and legacy monolith regressions",
                symbol=_utr_tp_dataflow_raw_runtime,
                status_hint="queued",
                test_path_prefixes=(
                    "tests/gabion/analysis/dataflow_s1/",
                    "tests/gabion/analysis/misc_s1/test_legacy_dataflow_monolith_run.py",
                    "tests/gabion/analysis/dataflow_s2/test_dataflow_main.py",
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="UTR-SQ-002",
                touchpoint_id="UTR-TP-004",
                title="Dataflow post-phase, WL, reuse, and exception-obligation collateral regressions",
                symbol=_utr_tp_dataflow_collateral,
                status_hint="queued",
                test_path_prefixes=(
                    "tests/gabion/analysis/dataflow_s2/",
                    "tests/gabion/analysis/misc_s1/test_wl_refinement.py",
                    "tests/gabion/analysis/dataflow_s1/test_dataflow_dataclass_bundles.py",
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="UTR-SQ-003",
                touchpoint_id="UTR-TP-005",
                title="policy_check helper and governance contract regressions",
                symbol=_utr_tp_policy_check_contracts,
                status_hint="queued",
                test_path_prefixes=(
                    "tests/gabion/tooling/ci/test_ci_governance_scripts.py",
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="UTR-SQ-003",
                touchpoint_id="UTR-TP-006",
                title="Generated artifact manifest renderer and drift regressions",
                symbol=_utr_tp_generated_artifact_manifest,
                status_hint="queued",
                test_path_prefixes=(
                    "tests/gabion/tooling/policy/test_render_generated_artifact_manifest.py",
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="UTR-SQ-003",
                touchpoint_id="UTR-TP-007",
                title="Deadline, script-policy, and runtime-policy path regressions",
                symbol=_utr_tp_deadline_script_policy,
                status_hint="queued",
                test_path_prefixes=(
                    "tests/gabion/tooling/script_policy/",
                    "tests/gabion/tooling/runtime_policy/test_perf_artifact.py",
                    "tests/gabion/tooling/runtime_policy/test_deprecated_nonerasability_policy_check.py",
                    "tests/gabion/analysis/misc_s3/test_private_symbol_import_guard.py",
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="UTR-SQ-004",
                touchpoint_id="UTR-TP-008",
                title="Server progress-transition and handledness/LSP envelope regressions",
                symbol=_utr_tp_server_runtime_envelope,
                status_hint="queued",
                test_path_prefixes=(
                    "tests/gabion/server_core/test_command_orchestrator.py",
                    "tests/gabion/server/test_server.py",
                    "tests/gabion/runtime/test_runtime_kernel_contracts.py",
                ),
            ),
        ),
        tags=("unit_test_readiness",),
    )


__all__ = ["unit_test_readiness_workstream_registry"]
