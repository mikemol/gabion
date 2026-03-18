from __future__ import annotations

from gabion.invariants import todo_decorator
from gabion.tooling.policy_substrate.workstream_registry import (
    RegisteredRootDefinition,
    RegisteredSubqueueDefinition,
    RegisteredTouchpointDefinition,
    RegisteredTouchsiteDefinition,
    WorkstreamRegistry,
    declared_touchsite_definition,
    registry_marker_metadata,
)


@todo_decorator(
    reason="PSN remains active until pure import-forwarder modules are retired from product code and cross-module private imports are replaced with explicit public owner interfaces.",
    reasoning={
        "summary": "The public-surface normalization root tracks the repo-wide drain of import-forwarder modules and private cross-module imports after WRD closed the command-family migration.",
        "control": "public_surface_normalization.root",
        "blocking_dependencies": (
            "PSN-SQ-001",
            "PSN-SQ-002",
            "PSN-SQ-003",
            "PSN-SQ-004",
        ),
    },
    owner="gabion.tooling.runtime",
    expiry="PSN closure",
    links=[{"kind": "object_id", "value": "PSN"}],
)
def _psn_root() -> None:
    return None


@todo_decorator(
    reason="PSN-SQ-001 remains active until namespace-family gabion commands call real owner modules directly and script-side pure forwarders are drained.",
    reasoning={
        "summary": "The runtime command surface still carries forwarder-style CLI shims and script-side import adapters that should collapse into direct owner-module command implementations.",
        "control": "public_surface_normalization.command_runtime_forwarders",
        "blocking_dependencies": ("PSN-TP-001", "PSN-TP-002"),
    },
    owner="gabion.tooling.runtime",
    expiry="PSN closure",
    links=[
        {"kind": "object_id", "value": "PSN"},
        {"kind": "object_id", "value": "PSN-SQ-001"},
        {"kind": "object_id", "value": "WRD-SQ-001"},
        {"kind": "object_id", "value": "WRD-SQ-002"},
    ],
)
def _psn_sq_command_runtime_forwarders() -> None:
    return None


@todo_decorator(
    reason="PSN-SQ-002 remains active until governance/package re-export veneers stop serving as public import surfaces and command ownership sits on real owner modules.",
    reasoning={
        "summary": "Governance-facing package adapters and command veneer modules still re-export implementation symbols instead of exposing explicit owner-module public interfaces.",
        "control": "public_surface_normalization.governance_package_reexports",
        "blocking_dependencies": ("PSN-TP-003",),
    },
    owner="gabion_governance",
    expiry="PSN closure",
    links=[
        {"kind": "object_id", "value": "PSN"},
        {"kind": "object_id", "value": "PSN-SQ-002"},
        {"kind": "object_id", "value": "WRD-SQ-003"},
    ],
)
def _psn_sq_governance_package_reexports() -> None:
    return None


@todo_decorator(
    reason="PSN-SQ-003 remains active until product-code cross-module imports stop reaching underscore-prefixed foreign symbols and the needed helpers are publicized on owner modules.",
    reasoning={
        "summary": "Product-code dependency clusters still import underscore-prefixed helpers across module boundaries, which leaves the public interface contract implicit and unstable.",
        "control": "public_surface_normalization.private_import_publicization",
        "blocking_dependencies": ("PSN-TP-004", "PSN-TP-005", "PSN-TP-006"),
    },
    owner="gabion.tooling.runtime_policy",
    expiry="PSN closure",
    links=[
        {"kind": "object_id", "value": "PSN"},
        {"kind": "object_id", "value": "PSN-SQ-003"},
        {"kind": "object_id", "value": "WRD"},
    ],
)
def _psn_sq_private_import_publicization() -> None:
    return None


@todo_decorator(
    reason="PSN-SQ-004 remains active until product-code guards block new private imports and pure forwarder modules while the remaining operator references are cut over.",
    reasoning={
        "summary": "The repo still needs durable enforcement for no-private-import and no-forwarder-module policy so public-surface drift does not re-enter after the initial drain.",
        "control": "public_surface_normalization.policy_guards",
        "blocking_dependencies": ("PSN-TP-007", "PSN-TP-008"),
    },
    owner="gabion.tooling.runtime_policy",
    expiry="PSN closure",
    links=[
        {"kind": "object_id", "value": "PSN"},
        {"kind": "object_id", "value": "PSN-SQ-004"},
        {"kind": "object_id", "value": "WRD-SQ-005"},
    ],
)
def _psn_sq_policy_guards() -> None:
    return None


@todo_decorator(
    reason="PSN-TP-001 remains queued until namespace-family gabion command publication no longer routes through pure runtime forwarder modules.",
    reasoning={
        "summary": "The CLI runner map still points at runtime shim modules whose only role is forwarding into script-owned or owner-owned command implementations.",
        "control": "public_surface_normalization.runtime_forwarders.touchpoint",
    },
    owner="gabion.tooling.runtime",
    expiry="PSN closure",
    links=[
        {"kind": "object_id", "value": "PSN"},
        {"kind": "object_id", "value": "PSN-SQ-001"},
        {"kind": "object_id", "value": "PSN-TP-001"},
        {"kind": "object_id", "value": "WRD-TP-002"},
        {"kind": "object_id", "value": "WRD-TP-004"},
    ],
)
def _psn_tp_runtime_forwarders() -> None:
    return None


@todo_decorator(
    reason="PSN-TP-002 remains queued until script-side pure import forwarders and compatibility veneers are removed or reduced to policy-compliant hard-fail stubs.",
    reasoning={
        "summary": "Some script/operator surfaces still exist only to forward imports into runtime modules, which leaves dead-end compatibility veneers in product code.",
        "control": "public_surface_normalization.script_forwarders.touchpoint",
    },
    owner="gabion.tooling.runtime",
    expiry="PSN closure",
    links=[
        {"kind": "object_id", "value": "PSN"},
        {"kind": "object_id", "value": "PSN-SQ-001"},
        {"kind": "object_id", "value": "PSN-TP-002"},
        {"kind": "object_id", "value": "WRD-TP-002"},
        {"kind": "object_id", "value": "WRD-TP-003"},
    ],
)
def _psn_tp_script_forwarders() -> None:
    return None


@todo_decorator(
    reason="PSN-TP-003 remains queued until governance/package adapter modules stop re-exporting implementation helpers as the de facto public surface.",
    reasoning={
        "summary": "Governance entrypoint and package adapter seams still publish re-export veneers instead of stable owner-module public interfaces.",
        "control": "public_surface_normalization.governance_reexports.touchpoint",
    },
    owner="gabion_governance",
    expiry="PSN closure",
    links=[
        {"kind": "object_id", "value": "PSN"},
        {"kind": "object_id", "value": "PSN-SQ-002"},
        {"kind": "object_id", "value": "PSN-TP-003"},
        {"kind": "object_id", "value": "WRD-TP-006"},
    ],
)
def _psn_tp_governance_reexports() -> None:
    return None


@todo_decorator(
    reason="PSN-TP-004 remains queued until command/CLI ingress and runtime helper imports stop reaching underscore-prefixed foreign symbols across product-code boundaries.",
    reasoning={
        "summary": "CLI-facing and runtime dependency surfaces still import private helpers from foreign modules instead of consuming explicit public owner exports.",
        "control": "public_surface_normalization.cli_runtime_private_imports.touchpoint",
    },
    owner="gabion.tooling.runtime",
    expiry="PSN closure",
    links=[
        {"kind": "object_id", "value": "PSN"},
        {"kind": "object_id", "value": "PSN-SQ-003"},
        {"kind": "object_id", "value": "PSN-TP-004"},
        {"kind": "object_id", "value": "WRD-TP-001"},
    ],
)
def _psn_tp_cli_runtime_private_imports() -> None:
    return None


@todo_decorator(
    reason="PSN-TP-005 remains queued until planning-substrate and governance helper clusters expose public owner interfaces instead of underscore reach-ins.",
    reasoning={
        "summary": "Planning-substrate and governance surfaces still rely on private cross-module helper imports, which keeps registry/runtime ownership implicit.",
        "control": "public_surface_normalization.planning_governance_private_imports.touchpoint",
    },
    owner="gabion.tooling.runtime_policy",
    expiry="PSN closure",
    links=[
        {"kind": "object_id", "value": "PSN"},
        {"kind": "object_id", "value": "PSN-SQ-003"},
        {"kind": "object_id", "value": "PSN-TP-005"},
        {"kind": "object_id", "value": "WRD-TP-006"},
    ],
)
def _psn_tp_planning_governance_private_imports() -> None:
    return None


@todo_decorator(
    reason="PSN-TP-006 remains queued until dataflow/reporting helper clusters stop importing underscore-prefixed foreign symbols and instead depend on explicit public owner contracts.",
    reasoning={
        "summary": "Dataflow and reporting helpers still carry private cross-module imports that need publicization or local ownership collapse.",
        "control": "public_surface_normalization.dataflow_private_imports.touchpoint",
    },
    owner="gabion.analysis.dataflow",
    expiry="PSN closure",
    links=[
        {"kind": "object_id", "value": "PSN"},
        {"kind": "object_id", "value": "PSN-SQ-003"},
        {"kind": "object_id", "value": "PSN-TP-006"},
        {"kind": "object_id", "value": "SCC-TP-005"},
    ],
)
def _psn_tp_dataflow_private_imports() -> None:
    return None


@todo_decorator(
    reason="PSN-TP-007 remains queued until product-code policy enforcement blocks new underscore-prefixed foreign imports.",
    reasoning={
        "summary": "The repo has a private-symbol import guard, but PSN is not closed until product-code enforcement explicitly carries the no-private-import contract.",
        "control": "public_surface_normalization.private_import_guard.touchpoint",
    },
    owner="gabion.tooling.runtime_policy",
    expiry="PSN closure",
    links=[
        {"kind": "object_id", "value": "PSN"},
        {"kind": "object_id", "value": "PSN-SQ-004"},
        {"kind": "object_id", "value": "PSN-TP-007"},
    ],
)
def _psn_tp_private_import_guard() -> None:
    return None


@todo_decorator(
    reason="PSN-TP-008 remains queued until the repo carries an explicit no-pure-forwarder-module guard and the remaining operator references to removed forwarders are cut over.",
    reasoning={
        "summary": "Public-surface normalization needs a durable forwarder guard plus final operator-surface cutover so pure import adapters do not reappear after the drain.",
        "control": "public_surface_normalization.forwarder_guard.touchpoint",
    },
    owner="gabion.tooling.runtime_policy",
    expiry="PSN closure",
    links=[
        {"kind": "object_id", "value": "PSN"},
        {"kind": "object_id", "value": "PSN-SQ-004"},
        {"kind": "object_id", "value": "PSN-TP-008"},
        {"kind": "object_id", "value": "WRD-TP-008"},
    ],
)
def _psn_tp_forwarder_guard() -> None:
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
        surface="public_surface_normalization_root",
        structural_path=f"public_surface_normalization.root::{root_id}",
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
        surface="public_surface_normalization_subqueue",
        structural_path=f"public_surface_normalization.subqueue::{subqueue_id}",
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
    declared_touchsites: tuple[RegisteredTouchsiteDefinition, ...],
) -> RegisteredTouchpointDefinition:
    metadata = registry_marker_metadata(
        symbol,
        surface="public_surface_normalization_touchpoint",
        structural_path=f"public_surface_normalization.touchpoint::{touchpoint_id}",
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
        declared_touchsites=declared_touchsites,
    )


def _touchsite(
    *,
    touchpoint_id: str,
    slug: str,
    rel_path: str,
    qualname: str,
    boundary_name: str,
    line: int,
) -> RegisteredTouchsiteDefinition:
    return declared_touchsite_definition(
        touchsite_id=f"{touchpoint_id.lower()}::{slug}",
        rel_path=rel_path,
        qualname=qualname,
        boundary_name=boundary_name,
        line=line,
        surface="public_surface_normalization_touchsite",
        structural_path=f"public_surface_normalization.touchsite::{touchpoint_id}::{slug}",
        object_ids=(touchpoint_id,),
    )


def public_surface_normalization_workstream_registry() -> WorkstreamRegistry:
    root_id = "PSN"
    return WorkstreamRegistry(
        root=_root_definition(
            root_id=root_id,
            title="Public Surface Normalization / Explicit Public Interfaces",
            subqueue_ids=(
                "PSN-SQ-001",
                "PSN-SQ-002",
                "PSN-SQ-003",
                "PSN-SQ-004",
            ),
            symbol=_psn_root,
            status_hint="in_progress",
        ),
        subqueues=(
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="PSN-SQ-001",
                title="Command and runtime forwarder drain",
                touchpoint_ids=("PSN-TP-001", "PSN-TP-002"),
                symbol=_psn_sq_command_runtime_forwarders,
                status_hint="in_progress",
            ),
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="PSN-SQ-002",
                title="Governance and package re-export drain",
                touchpoint_ids=("PSN-TP-003",),
                symbol=_psn_sq_governance_package_reexports,
                status_hint="in_progress",
            ),
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="PSN-SQ-003",
                title="Private-import publicization in product code",
                touchpoint_ids=("PSN-TP-004", "PSN-TP-005", "PSN-TP-006"),
                symbol=_psn_sq_private_import_publicization,
                status_hint="in_progress",
            ),
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="PSN-SQ-004",
                title="Policy guards, docs, and closure enforcement",
                touchpoint_ids=("PSN-TP-007", "PSN-TP-008"),
                symbol=_psn_sq_policy_guards,
                status_hint="in_progress",
            ),
        ),
        touchpoints=(
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="PSN-SQ-001",
                touchpoint_id="PSN-TP-001",
                title="Drain runtime namespace command forwarders",
                symbol=_psn_tp_runtime_forwarders,
                status_hint="queued",
                declared_touchsites=(
                    _touchsite(
                        touchpoint_id="PSN-TP-001",
                        slug="tooling-runner-map",
                        rel_path="src/gabion/cli.py",
                        qualname="_TOOLING_ARGV_RUNNERS",
                        boundary_name="tooling argv runner map",
                        line=1627,
                    ),
                    _touchsite(
                        touchpoint_id="PSN-TP-001",
                        slug="policy-check-cli",
                        rel_path="src/gabion/tooling/runtime/policy_check_cli.py",
                        qualname="main",
                        boundary_name="policy check runtime CLI forwarder",
                        line=13,
                    ),
                    _touchsite(
                        touchpoint_id="PSN-TP-001",
                        slug="docflow-packetize-cli",
                        rel_path="src/gabion/tooling/runtime/docflow_packetize_cli.py",
                        qualname="main",
                        boundary_name="docflow packetize runtime CLI forwarder",
                        line=13,
                    ),
                    _touchsite(
                        touchpoint_id="PSN-TP-001",
                        slug="docflow-packet-enforce-cli",
                        rel_path="src/gabion/tooling/runtime/docflow_packet_enforce_cli.py",
                        qualname="main",
                        boundary_name="docflow packet enforce runtime CLI forwarder",
                        line=13,
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="PSN-SQ-001",
                touchpoint_id="PSN-TP-002",
                title="Remove script-side pure forwarders and compatibility veneers",
                symbol=_psn_tp_script_forwarders,
                status_hint="queued",
                declared_touchsites=(
                    _touchsite(
                        touchpoint_id="PSN-TP-002",
                        slug="ci-watch-script",
                        rel_path="scripts/ci/ci_watch.py",
                        qualname="main",
                        boundary_name="ci watch script forwarder",
                        line=1,
                    ),
                    _touchsite(
                        touchpoint_id="PSN-TP-002",
                        slug="ci-watch-runtime",
                        rel_path="src/gabion/tooling/runtime/ci_watch.py",
                        qualname="main",
                        boundary_name="ci watch runtime owner",
                        line=559,
                    ),
                    _touchsite(
                        touchpoint_id="PSN-TP-002",
                        slug="aspf-handoff-stub",
                        rel_path="scripts/misc/aspf_handoff.py",
                        qualname="main",
                        boundary_name="aspf handoff legacy script stub",
                        line=12,
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="PSN-SQ-002",
                touchpoint_id="PSN-TP-003",
                title="Drain governance/package re-export veneers",
                symbol=_psn_tp_governance_reexports,
                status_hint="queued",
                declared_touchsites=(
                    _touchsite(
                        touchpoint_id="PSN-TP-003",
                        slug="governance-audit-adapter",
                        rel_path="src/gabion/tooling/governance/governance_audit.py",
                        qualname="BOUNDARY_ADAPTER_METADATA",
                        boundary_name="governance audit adapter veneer",
                        line=17,
                    ),
                    _touchsite(
                        touchpoint_id="PSN-TP-003",
                        slug="governance-entrypoint",
                        rel_path="src/gabion_governance/governance_entrypoint.py",
                        qualname="main",
                        boundary_name="governance entrypoint owner",
                        line=39,
                    ),
                    _touchsite(
                        touchpoint_id="PSN-TP-003",
                        slug="governance-docflow-command",
                        rel_path="src/gabion_governance/docflow_command.py",
                        qualname="run_docflow_cli",
                        boundary_name="governance docflow re-export veneer",
                        line=3,
                    ),
                    _touchsite(
                        touchpoint_id="PSN-TP-003",
                        slug="governance-status-consistency-command",
                        rel_path="src/gabion_governance/status_consistency_command.py",
                        qualname="run_status_consistency_cli",
                        boundary_name="governance status-consistency re-export veneer",
                        line=5,
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="PSN-SQ-003",
                touchpoint_id="PSN-TP-004",
                title="Publicize CLI and runtime helper imports",
                symbol=_psn_tp_cli_runtime_private_imports,
                status_hint="queued",
                declared_touchsites=(
                    _touchsite(
                        touchpoint_id="PSN-TP-004",
                        slug="cli-runner-map",
                        rel_path="src/gabion/cli.py",
                        qualname="_TOOLING_ARGV_RUNNERS",
                        boundary_name="CLI runner dependency surface",
                        line=1627,
                    ),
                    _touchsite(
                        touchpoint_id="PSN-TP-004",
                        slug="governance-entrypoint-parser",
                        rel_path="src/gabion_governance/governance_entrypoint.py",
                        qualname="parse_single_command_args",
                        boundary_name="governance parser helper surface",
                        line=11,
                    ),
                    _touchsite(
                        touchpoint_id="PSN-TP-004",
                        slug="policy-check-runtime-owner",
                        rel_path="scripts/policy/policy_check.py",
                        qualname="main",
                        boundary_name="policy check owner command surface",
                        line=3165,
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="PSN-SQ-003",
                touchpoint_id="PSN-TP-005",
                title="Publicize planning-substrate and governance helper imports",
                symbol=_psn_tp_planning_governance_private_imports,
                status_hint="queued",
                declared_touchsites=(
                    _touchsite(
                        touchpoint_id="PSN-TP-005",
                        slug="wrd-registry",
                        rel_path="src/gabion/tooling/policy_substrate/wrapper_retirement_drain_registry.py",
                        qualname="wrapper_retirement_drain_workstream_registry",
                        boundary_name="WRD registry history anchor",
                        line=318,
                    ),
                    _touchsite(
                        touchpoint_id="PSN-TP-005",
                        slug="governance-audit-adapter",
                        rel_path="src/gabion/tooling/governance/governance_audit.py",
                        qualname="BOUNDARY_ADAPTER_METADATA",
                        boundary_name="governance package adapter surface",
                        line=17,
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="PSN-SQ-003",
                touchpoint_id="PSN-TP-006",
                title="Publicize dataflow and reporting helper imports",
                symbol=_psn_tp_dataflow_private_imports,
                status_hint="queued",
                declared_touchsites=(
                    _touchsite(
                        touchpoint_id="PSN-TP-006",
                        slug="ci-watch-runtime",
                        rel_path="src/gabion/tooling/runtime/ci_watch.py",
                        qualname="main",
                        boundary_name="runtime helper cluster anchor",
                        line=559,
                    ),
                    _touchsite(
                        touchpoint_id="PSN-TP-006",
                        slug="run-dataflow-stage",
                        rel_path="src/gabion/tooling/runtime/run_dataflow_stage.py",
                        qualname="main",
                        boundary_name="dataflow runtime helper anchor",
                        line=1079,
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="PSN-SQ-004",
                touchpoint_id="PSN-TP-007",
                title="Add product-code no-private-import guard",
                symbol=_psn_tp_private_import_guard,
                status_hint="queued",
                declared_touchsites=(
                    _touchsite(
                        touchpoint_id="PSN-TP-007",
                        slug="private-import-guard",
                        rel_path="scripts/policy/private_symbol_import_guard.py",
                        qualname="main",
                        boundary_name="private symbol import guard",
                        line=232,
                    ),
                    _touchsite(
                        touchpoint_id="PSN-TP-007",
                        slug="policy-check",
                        rel_path="scripts/policy/policy_check.py",
                        qualname="main",
                        boundary_name="policy check enforcement surface",
                        line=3165,
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="PSN-SQ-004",
                touchpoint_id="PSN-TP-008",
                title="Add no-forwarder-module guard and closure enforcement",
                symbol=_psn_tp_forwarder_guard,
                status_hint="queued",
                declared_touchsites=(
                    _touchsite(
                        touchpoint_id="PSN-TP-008",
                        slug="policy-check",
                        rel_path="scripts/policy/policy_check.py",
                        qualname="main",
                        boundary_name="forwarder guard policy surface",
                        line=3165,
                    ),
                    _touchsite(
                        touchpoint_id="PSN-TP-008",
                        slug="user-workflows",
                        rel_path="docs/user_workflows.md",
                        qualname="user_workflows",
                        boundary_name="operator workflow guidance",
                        line=1,
                    ),
                ),
            ),
        ),
        tags=("public_surface_normalization",),
    )

