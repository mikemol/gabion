from __future__ import annotations

from gabion.invariants import todo_decorator
from gabion.tooling.policy_substrate.workstream_registry import (
    RegisteredRootDefinition,
    RegisteredSubqueueDefinition,
    RegisteredTouchpointDefinition,
    WorkstreamRegistry,
    declared_touchsite_definition,
    registry_marker_metadata,
)


@todo_decorator(
    reason="SCC remains active until coercion, bridge-contract, compatibility-boundary, and review-note metadata surfaces converge onto canonical contracts.",
    reasoning={
        "summary": "Runtime-shape coercion, server-core bridge contracts, staged indexed dataflow facade retirement, and review-note consistency still drift across parallel local surfaces.",
        "control": "surface_contract_convergence.root",
        "blocking_dependencies": (
            "SCC-SQ-001",
            "SCC-SQ-002",
            "SCC-SQ-003",
            "SCC-SQ-004",
        ),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="SCC closure",
    links=[{"kind": "object_id", "value": "SCC"}],
)
def _scc_root() -> None:
    return None


@todo_decorator(
    reason="SCC-SQ-001 remains active until runtime-shape coercion logic converges on one shared contract substrate.",
    reasoning={
        "summary": "Optional mapping/string/int/float coercion still exists in duplicated singledispatch families with diverging bool and fallback semantics.",
        "control": "surface_contract_convergence.coercion_substrate",
        "blocking_dependencies": (
            "SCC-TP-001",
            "SCC-TP-002",
        ),
    },
    owner="gabion.runtime",
    expiry="SCC closure",
    links=[
        {"kind": "object_id", "value": "SCC"},
        {"kind": "object_id", "value": "SCC-SQ-001"},
    ],
)
def _scc_sq_coercion_substrate() -> None:
    return None


@todo_decorator(
    reason="SCC-SQ-002 remains active until server-core primitive bridges materialize from one registry-backed contract layer.",
    reasoning={
        "summary": "First-layer and downstream server-core bridge wrappers still mirror command_orchestrator_primitives through brittle hand-wired dataclass/staticmethod assignments.",
        "control": "surface_contract_convergence.server_core_bridge_contracts",
        "blocking_dependencies": (
            "SCC-TP-003",
            "SCC-TP-004",
        ),
    },
    owner="gabion.server_core",
    expiry="SCC closure",
    links=[
        {"kind": "object_id", "value": "SCC"},
        {"kind": "object_id", "value": "SCC-SQ-002"},
    ],
)
def _scc_sq_server_core_bridge_contracts() -> None:
    return None


@todo_decorator(
    reason="SCC-SQ-003 remains active until the indexed dataflow compatibility facade is reduced to explicit boundary adapters with retirement telemetry.",
    reasoning={
        "summary": "The staged indexed facade still operates as a broad import hub instead of an explicit alias inventory tied to retirement/debt artifacts.",
        "control": "surface_contract_convergence.indexed_dataflow_boundary",
        "blocking_dependencies": (
            "SCC-TP-005",
            "SCC-TP-006",
        ),
    },
    owner="gabion.analysis.dataflow",
    expiry="SCC closure",
    links=[
        {"kind": "doc_id", "value": "dataflow_runtime_debt_ledger"},
        {"kind": "doc_id", "value": "dataflow_runtime_retirement_ledger"},
        {"kind": "object_id", "value": "SCC"},
        {"kind": "object_id", "value": "SCC-SQ-003"},
    ],
)
def _scc_sq_indexed_dataflow_boundary() -> None:
    return None


@todo_decorator(
    reason="SCC-SQ-004 remains active until review-note metadata drift is mechanically checked and normalized on core governance docs.",
    reasoning={
        "summary": "doc_review_notes prose can lag current document and section revisions because the audit only checks pin maps, not the explanatory note text.",
        "control": "surface_contract_convergence.review_note_consistency",
        "blocking_dependencies": ("SCC-TP-007",),
    },
    owner="gabion_governance",
    expiry="SCC closure",
    links=[
        {"kind": "object_id", "value": "SCC"},
        {"kind": "object_id", "value": "SCC-SQ-004"},
    ],
)
def _scc_sq_review_note_consistency() -> None:
    return None


@todo_decorator(
    reason="SCC-TP-001 tracks the first coercion cutover onto a shared contract module for the runtime-shape and progress-contract entry surfaces.",
    reasoning={
        "summary": "runtime_shape_dispatch and commands.progress_contract should stop carrying parallel coercion singledispatch families and instead consume one named contract substrate.",
        "control": "surface_contract_convergence.coercion_pair_touchpoint",
        "blocking_dependencies": ("SCC-SQ-001",),
    },
    owner="gabion.runtime",
    expiry="SCC closure",
    links=[
        {"kind": "object_id", "value": "SCC"},
        {"kind": "object_id", "value": "SCC-SQ-001"},
        {"kind": "object_id", "value": "SCC-TP-001"},
    ],
)
def _scc_tp_coercion_pair() -> None:
    return None


@todo_decorator(
    reason="SCC-TP-002 tracks the staged migration of remaining duplicate coercion helper families onto the same shared coercion contract.",
    reasoning={
        "summary": "CLI, server-core progress, governance, ASPF, and test-obsolescence still each carry local mapping/int/float coercion helpers that should reuse the shared runtime contract.",
        "control": "surface_contract_convergence.coercion_followons_touchpoint",
        "blocking_dependencies": ("SCC-SQ-001",),
    },
    owner="gabion.runtime",
    expiry="SCC closure",
    links=[
        {"kind": "object_id", "value": "SCC"},
        {"kind": "object_id", "value": "SCC-SQ-001"},
        {"kind": "object_id", "value": "SCC-TP-002"},
    ],
)
def _scc_tp_coercion_followons() -> None:
    return None


@todo_decorator(
    reason="SCC-TP-003 tracks replacement of the first-layer server-core bridge wrappers with a registry-backed primitive contract layer.",
    reasoning={
        "summary": "analysis_primitives, timeout_runtime, progress_contracts, and report_projection_runtime should materialize from one declarative bridge contract instead of hand-maintained symbol lists.",
        "control": "surface_contract_convergence.bridge_layer_one_touchpoint",
        "blocking_dependencies": ("SCC-SQ-002",),
    },
    owner="gabion.server_core",
    expiry="SCC closure",
    links=[
        {"kind": "object_id", "value": "SCC"},
        {"kind": "object_id", "value": "SCC-SQ-002"},
        {"kind": "object_id", "value": "SCC-TP-003"},
    ],
)
def _scc_tp_bridge_layer_one() -> None:
    return None


@todo_decorator(
    reason="SCC-TP-004 tracks migration of downstream mirror bridge layers and policy surfaces onto the same primitive contract registry.",
    reasoning={
        "summary": "output/progress/timeout primitives, ingress deps, and barrel-growth policy checks should consume the same registry-backed bridge contract as the first-layer wrappers.",
        "control": "surface_contract_convergence.bridge_layer_two_touchpoint",
        "blocking_dependencies": ("SCC-SQ-002",),
    },
    owner="gabion.server_core",
    expiry="SCC closure",
    links=[
        {"kind": "object_id", "value": "SCC"},
        {"kind": "object_id", "value": "SCC-SQ-002"},
        {"kind": "object_id", "value": "SCC-TP-004"},
    ],
)
def _scc_tp_bridge_layer_two() -> None:
    return None


@todo_decorator(
    reason="SCC-TP-005 tracks conversion of the indexed dataflow compatibility facade into an explicit alias-inventory boundary with retirement telemetry.",
    reasoning={
        "summary": "The staged dataflow_indexed_file_scan compatibility module should declare its alias boundary surface explicitly instead of remaining a broad import hub.",
        "control": "surface_contract_convergence.indexed_alias_inventory_touchpoint",
        "blocking_dependencies": ("SCC-SQ-003",),
    },
    owner="gabion.analysis.dataflow",
    expiry="SCC closure",
    links=[
        {"kind": "doc_id", "value": "dataflow_runtime_debt_ledger"},
        {"kind": "doc_id", "value": "dataflow_runtime_retirement_ledger"},
        {"kind": "object_id", "value": "SCC"},
        {"kind": "object_id", "value": "SCC-SQ-003"},
        {"kind": "object_id", "value": "SCC-TP-005"},
    ],
)
def _scc_tp_indexed_alias_inventory() -> None:
    return None


@todo_decorator(
    reason="SCC-TP-006 tracks the follow-on deflation of the indexed facade's remaining hot import fan-in while keeping the compatibility path stable.",
    reasoning={
        "summary": "After explicit alias inventory exists, remaining owner-domain adapters and debt rows should converge until the facade is boundary-only.",
        "control": "surface_contract_convergence.indexed_adapter_deflation_touchpoint",
        "blocking_dependencies": ("SCC-SQ-003",),
    },
    owner="gabion.analysis.dataflow",
    expiry="SCC closure",
    links=[
        {"kind": "doc_id", "value": "dataflow_runtime_debt_ledger"},
        {"kind": "doc_id", "value": "dataflow_runtime_retirement_ledger"},
        {"kind": "object_id", "value": "SCC"},
        {"kind": "object_id", "value": "SCC-SQ-003"},
        {"kind": "object_id", "value": "SCC-TP-006"},
    ],
)
def _scc_tp_indexed_adapter_deflation() -> None:
    return None


@todo_decorator(
    reason="SCC-TP-007 tracks the review-note consistency lint and normalization of core governance document notes.",
    reasoning={
        "summary": "Core governance docs should fail when doc_review_notes prose does not mention current document and section revision context alongside the pinned review metadata.",
        "control": "surface_contract_convergence.review_note_lint_touchpoint",
        "blocking_dependencies": ("SCC-SQ-004",),
    },
    owner="gabion_governance",
    expiry="SCC closure",
    links=[
        {"kind": "object_id", "value": "SCC"},
        {"kind": "object_id", "value": "SCC-SQ-004"},
        {"kind": "object_id", "value": "SCC-TP-007"},
    ],
)
def _scc_tp_review_note_lint() -> None:
    return None


def _root_definition(
    *,
    root_id: str,
    title: str,
    subqueue_ids: tuple[str, ...],
    symbol,
    status_hint: str = "",
) -> RegisteredRootDefinition:
    metadata = registry_marker_metadata(
        symbol,
        surface="surface_contract_convergence_root",
        structural_path=f"surface_contract_convergence.root::{root_id}",
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
    status_hint: str = "",
) -> RegisteredSubqueueDefinition:
    metadata = registry_marker_metadata(
        symbol,
        surface="surface_contract_convergence_subqueue",
        structural_path=f"surface_contract_convergence.subqueue::{subqueue_id}",
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
    declared_touchsites: tuple = (),
    status_hint: str = "",
) -> RegisteredTouchpointDefinition:
    metadata = registry_marker_metadata(
        symbol,
        surface="surface_contract_convergence_touchpoint",
        structural_path=f"surface_contract_convergence.touchpoint::{touchpoint_id}",
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
        declared_touchsites=declared_touchsites,
        status_hint=status_hint,
    )


def _module_touchsite(
    *,
    touchsite_id: str,
    rel_path: str,
    qualname: str,
    seam_class: str = "surviving_carrier_seam",
) -> object:
    return declared_touchsite_definition(
        touchsite_id=touchsite_id,
        rel_path=rel_path,
        qualname=qualname,
        boundary_name=qualname,
        line=1,
        node_kind="module",
        surface="surface_contract_convergence_touchsite",
        structural_path=f"surface_contract_convergence.touchsite::{touchsite_id}",
        seam_class=seam_class,
    )


def _document_touchsite(
    *,
    touchsite_id: str,
    rel_path: str,
    qualname: str,
) -> object:
    return declared_touchsite_definition(
        touchsite_id=touchsite_id,
        rel_path=rel_path,
        qualname=qualname,
        boundary_name=qualname,
        line=1,
        node_kind="document",
        surface="surface_contract_convergence_touchsite",
        structural_path=f"surface_contract_convergence.touchsite::{touchsite_id}",
        seam_class="document_boundary",
    )


def surface_contract_convergence_workstream_registry() -> WorkstreamRegistry:
    root_id = "SCC"
    return WorkstreamRegistry(
        root=_root_definition(
            root_id=root_id,
            title="Surface contract convergence and boundary retirement cleanup",
            symbol=_scc_root,
            subqueue_ids=(
                "SCC-SQ-001",
                "SCC-SQ-002",
                "SCC-SQ-003",
                "SCC-SQ-004",
            ),
            status_hint="in_progress",
        ),
        subqueues=(
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="SCC-SQ-001",
                title="Runtime-shape coercion substrate convergence",
                symbol=_scc_sq_coercion_substrate,
                touchpoint_ids=("SCC-TP-001", "SCC-TP-002"),
                status_hint="in_progress",
            ),
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="SCC-SQ-002",
                title="Server-core primitive bridge contract convergence",
                symbol=_scc_sq_server_core_bridge_contracts,
                touchpoint_ids=("SCC-TP-003", "SCC-TP-004"),
                status_hint="in_progress",
            ),
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="SCC-SQ-003",
                title="Indexed dataflow compatibility-boundary retirement",
                symbol=_scc_sq_indexed_dataflow_boundary,
                touchpoint_ids=("SCC-TP-005", "SCC-TP-006"),
                status_hint="in_progress",
            ),
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="SCC-SQ-004",
                title="Core governance review-note metadata consistency",
                symbol=_scc_sq_review_note_consistency,
                touchpoint_ids=("SCC-TP-007",),
                status_hint="in_progress",
            ),
        ),
        touchpoints=(
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="SCC-SQ-001",
                touchpoint_id="SCC-TP-001",
                title="Canonical coercion contract for runtime_shape_dispatch and progress_contract",
                symbol=_scc_tp_coercion_pair,
                status_hint="landed",
                declared_touchsites=(
                    _module_touchsite(
                        touchsite_id="SCC-TS-001-A",
                        rel_path="src/gabion/runtime/coercion_contract.py",
                        qualname="coercion_contract",
                    ),
                    _module_touchsite(
                        touchsite_id="SCC-TS-001-B",
                        rel_path="src/gabion/runtime_shape_dispatch.py",
                        qualname="runtime_shape_dispatch",
                    ),
                    _module_touchsite(
                        touchsite_id="SCC-TS-001-C",
                        rel_path="src/gabion/commands/progress_contract.py",
                        qualname="progress_contract",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="SCC-SQ-001",
                touchpoint_id="SCC-TP-002",
                title="Follow-on migration of duplicate coercion helper families",
                symbol=_scc_tp_coercion_followons,
                status_hint="landed",
                declared_touchsites=(
                    _module_touchsite(
                        touchsite_id="SCC-TS-002-A",
                        rel_path="src/gabion/server_core/command_orchestrator_progress.py",
                        qualname="command_orchestrator_progress",
                    ),
                    _module_touchsite(
                        touchsite_id="SCC-TS-002-B",
                        rel_path="src/gabion/cli.py",
                        qualname="cli",
                    ),
                    _module_touchsite(
                        touchsite_id="SCC-TS-002-C",
                        rel_path="src/gabion/tooling/governance/normative_symdiff.py",
                        qualname="normative_symdiff",
                    ),
                    _module_touchsite(
                        touchsite_id="SCC-TS-002-D",
                        rel_path="src/gabion/analysis/foundation/aspf_execution_fibration_impl.py",
                        qualname="aspf_execution_fibration_impl",
                    ),
                    _module_touchsite(
                        touchsite_id="SCC-TS-002-E",
                        rel_path="src/gabion/analysis/surfaces/test_obsolescence_delta.py",
                        qualname="test_obsolescence_delta",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="SCC-SQ-002",
                touchpoint_id="SCC-TP-003",
                title="Registry-backed first-layer server-core primitive contracts",
                symbol=_scc_tp_bridge_layer_one,
                status_hint="landed",
                declared_touchsites=(
                    _module_touchsite(
                        touchsite_id="SCC-TS-003-A",
                        rel_path="src/gabion/server_core/primitive_contract_registry.py",
                        qualname="primitive_contract_registry",
                    ),
                    _module_touchsite(
                        touchsite_id="SCC-TS-003-B",
                        rel_path="src/gabion/server_core/analysis_primitives.py",
                        qualname="analysis_primitives",
                    ),
                    _module_touchsite(
                        touchsite_id="SCC-TS-003-C",
                        rel_path="src/gabion/server_core/timeout_runtime.py",
                        qualname="timeout_runtime",
                    ),
                    _module_touchsite(
                        touchsite_id="SCC-TS-003-D",
                        rel_path="src/gabion/server_core/progress_contracts.py",
                        qualname="progress_contracts",
                    ),
                    _module_touchsite(
                        touchsite_id="SCC-TS-003-E",
                        rel_path="src/gabion/server_core/report_projection_runtime.py",
                        qualname="report_projection_runtime",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="SCC-SQ-002",
                touchpoint_id="SCC-TP-004",
                title="Registry-backed downstream server-core mirror contracts",
                symbol=_scc_tp_bridge_layer_two,
                status_hint="landed",
                declared_touchsites=(
                    _module_touchsite(
                        touchsite_id="SCC-TS-004-A",
                        rel_path="src/gabion/server_core/output_primitives.py",
                        qualname="output_primitives",
                    ),
                    _module_touchsite(
                        touchsite_id="SCC-TS-004-B",
                        rel_path="src/gabion/server_core/progress_primitives.py",
                        qualname="progress_primitives",
                    ),
                    _module_touchsite(
                        touchsite_id="SCC-TS-004-C",
                        rel_path="src/gabion/server_core/timeout_primitives.py",
                        qualname="timeout_primitives",
                    ),
                    _module_touchsite(
                        touchsite_id="SCC-TS-004-D",
                        rel_path="src/gabion/server_core/ingress_contracts.py",
                        qualname="ingress_contracts",
                    ),
                    _module_touchsite(
                        touchsite_id="SCC-TS-004-E",
                        rel_path="src/gabion/tooling/policy_rules/orchestrator_primitive_barrel_rule.py",
                        qualname="orchestrator_primitive_barrel_rule",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="SCC-SQ-003",
                touchpoint_id="SCC-TP-005",
                title="Explicit alias-inventory boundary for indexed dataflow facade",
                symbol=_scc_tp_indexed_alias_inventory,
                status_hint="queued",
                declared_touchsites=(
                    _module_touchsite(
                        touchsite_id="SCC-TS-005-A",
                        rel_path="src/gabion/analysis/dataflow/engine/dataflow_indexed_file_scan.py",
                        qualname="dataflow_indexed_file_scan",
                    ),
                    _document_touchsite(
                        touchsite_id="SCC-TS-005-B",
                        rel_path="docs/audits/dataflow_runtime_debt_ledger.md",
                        qualname="dataflow_runtime_debt_ledger",
                    ),
                    _document_touchsite(
                        touchsite_id="SCC-TS-005-C",
                        rel_path="docs/audits/dataflow_runtime_retirement_ledger.md",
                        qualname="dataflow_runtime_retirement_ledger",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="SCC-SQ-003",
                touchpoint_id="SCC-TP-006",
                title="Owner-domain adapter deflation for indexed dataflow facade",
                symbol=_scc_tp_indexed_adapter_deflation,
                status_hint="queued",
                declared_touchsites=(
                    _module_touchsite(
                        touchsite_id="SCC-TS-006-A",
                        rel_path="src/gabion/analysis/dataflow/engine/dataflow_indexed_file_scan.py",
                        qualname="dataflow_indexed_file_scan",
                    ),
                    _document_touchsite(
                        touchsite_id="SCC-TS-006-B",
                        rel_path="docs/audits/dataflow_runtime_debt_ledger.md",
                        qualname="dataflow_runtime_debt_ledger",
                    ),
                    _document_touchsite(
                        touchsite_id="SCC-TS-006-C",
                        rel_path="docs/audits/dataflow_runtime_retirement_ledger.md",
                        qualname="dataflow_runtime_retirement_ledger",
                    ),
                    _document_touchsite(
                        touchsite_id="SCC-TS-006-D",
                        rel_path="docs/audits/dataflow_legacy_monolith_test_replacement_matrix.md",
                        qualname="dataflow_legacy_monolith_test_replacement_matrix",
                    ),
                    _document_touchsite(
                        touchsite_id="SCC-TS-006-E",
                        rel_path="docs/compatibility_layer_debt_register.md",
                        qualname="compatibility_layer_debt_register",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="SCC-SQ-004",
                touchpoint_id="SCC-TP-007",
                title="Core-governance review-note consistency lint and normalization",
                symbol=_scc_tp_review_note_lint,
                status_hint="queued",
                declared_touchsites=(
                    _module_touchsite(
                        touchsite_id="SCC-TS-007-A",
                        rel_path="src/gabion_governance/governance_audit_impl.py",
                        qualname="governance_audit_impl",
                    ),
                    _document_touchsite(
                        touchsite_id="SCC-TS-007-B",
                        rel_path="AGENTS.md",
                        qualname="AGENTS.md#agent_obligations",
                    ),
                    _document_touchsite(
                        touchsite_id="SCC-TS-007-C",
                        rel_path="README.md",
                        qualname="README.md#repo_contract",
                    ),
                    _document_touchsite(
                        touchsite_id="SCC-TS-007-D",
                        rel_path="CONTRIBUTING.md",
                        qualname="CONTRIBUTING.md#contributing_contract",
                    ),
                    _document_touchsite(
                        touchsite_id="SCC-TS-007-E",
                        rel_path="POLICY_SEED.md",
                        qualname="POLICY_SEED.md#policy_seed",
                    ),
                    _document_touchsite(
                        touchsite_id="SCC-TS-007-F",
                        rel_path="glossary.md",
                        qualname="glossary.md#contract",
                    ),
                    _document_touchsite(
                        touchsite_id="SCC-TS-007-G",
                        rel_path="docs/normative_clause_index.md",
                        qualname="docs/normative_clause_index.md#normative_clause_index",
                    ),
                ),
            ),
        ),
        tags=("contract_convergence",),
    )


__all__ = ["surface_contract_convergence_workstream_registry"]
