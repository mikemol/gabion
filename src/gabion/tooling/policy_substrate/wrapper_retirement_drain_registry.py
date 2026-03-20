from __future__ import annotations

from gabion.invariants import landed_todo_decorator
from gabion.tooling.policy_substrate.workstream_registry import (
    RegisteredRootDefinition,
    RegisteredSubqueueDefinition,
    RegisteredTouchpointDefinition,
    WorkstreamRegistry,
    declared_touchsite_definition,
    registry_marker_metadata,
)


@landed_todo_decorator(
    reason="WRD is recorded as landed metadata after wrapper-style operational entrypoints converged on canonical namespace-family gabion subcommands and the legacy wrappers were reduced to hard-fail stubs.",
    reasoning={
        "summary": "The repo-wide wrapper retirement drain is completed on canonical namespace-family gabion command surfaces, and the closed migration is recorded as landed root state.",
        "control": "wrapper_retirement_drain.root",
        "blocking_dependencies": (),
    },
    owner="gabion.tooling.runtime_policy",
    expiry="WRD closure",
    links=[{"kind": "object_id", "value": "WRD"}],
)
def _wrd_root() -> None:
    return None


@landed_todo_decorator(
    reason="WRD-SQ-001 is recorded as landed metadata after the dataflow and CI wrapper surfaces converged on namespace-family gabion command ownership across CLI publication, workflows, docs, and stubs.",
    reasoning={
        "summary": "run-dataflow-stage, ci local-repro/watch, and ASPF handoff now use the namespace-family command surface end to end, and the completed tranche is recorded as closed subqueue state.",
        "control": "wrapper_retirement_drain.dataflow_ci",
        "blocking_dependencies": (),
    },
    owner="gabion.tooling.runtime",
    expiry="WRD closure",
    links=[
        {"kind": "object_id", "value": "WRD"},
        {"kind": "object_id", "value": "WRD-SQ-001"},
    ],
)
def _wrd_sq_dataflow_ci() -> None:
    return None


@landed_todo_decorator(
    reason="WRD-SQ-002 is recorded as landed metadata after policy and docflow wrappers converged on namespace-family gabion policy commands across runtime, workflows, docs, and stubs.",
    reasoning={
        "summary": "Policy check, the docflow packet loop, and the policy scanner now use the canonical namespace-family policy commands throughout the active operator surface, and the completed tranche is recorded as closed subqueue state.",
        "control": "wrapper_retirement_drain.policy_docflow",
        "blocking_dependencies": (),
    },
    owner="scripts.policy",
    expiry="WRD closure",
    links=[
        {"kind": "object_id", "value": "WRD"},
        {"kind": "object_id", "value": "WRD-SQ-002"},
    ],
)
def _wrd_sq_policy_docflow() -> None:
    return None


@landed_todo_decorator(
    reason="WRD-SQ-003 is recorded as landed metadata after governance and SPPF wrapper surfaces converged on namespace-family gabion commands across runtime, workflows, docs, and stubs.",
    reasoning={
        "summary": "Governance controller audit, telemetry emit, and SPPF sync/status now publish and document only the namespace-family command surface, and the completed tranche is recorded as closed subqueue state.",
        "control": "wrapper_retirement_drain.governance_sppf",
        "blocking_dependencies": (),
    },
    owner="gabion_governance",
    expiry="WRD closure",
    links=[
        {"kind": "object_id", "value": "WRD"},
        {"kind": "object_id", "value": "WRD-SQ-003"},
    ],
)
def _wrd_sq_governance_sppf() -> None:
    return None


@landed_todo_decorator(
    reason="WRD-SQ-004 is recorded as landed metadata after release and repo-utility wrappers converged on canonical gabion release and gabion repo command ownership.",
    reasoning={
        "summary": "Release helpers and repo-local operational utilities now publish only canonical release and repo command families, and the completed tranche is recorded as closed subqueue state.",
        "control": "wrapper_retirement_drain.release_repo_utilities",
        "blocking_dependencies": (),
    },
    owner="gabion.tooling.runtime",
    expiry="WRD closure",
    links=[
        {"kind": "object_id", "value": "WRD"},
        {"kind": "object_id", "value": "WRD-SQ-004"},
    ],
)
def _wrd_sq_release_repo_utilities() -> None:
    return None


@landed_todo_decorator(
    reason="WRD-SQ-005 is recorded as landed metadata after operator docs, workflows, generated manifests, governance loop docs, Make targets, and guardrails converged on canonical namespace-family gabion command guidance.",
    reasoning={
        "summary": "Operator docs, workflows, generated manifests, governance loop docs, Make targets, and the wrapper guard surface now point at canonical namespace-family commands, and the completed tranche is recorded as closed subqueue state.",
        "control": "wrapper_retirement_drain.docs_workflows_guard",
        "blocking_dependencies": (),
    },
    owner="gabion.tooling.runtime_policy",
    expiry="WRD closure",
    links=[
        {"kind": "object_id", "value": "WRD"},
        {"kind": "object_id", "value": "WRD-SQ-005"},
    ],
)
def _wrd_sq_docs_workflows_guard() -> None:
    return None


@landed_todo_decorator(
    reason="WRD-TP-001 is recorded as landed metadata after run-dataflow-stage converged on a hidden hard-fail stub and active surfaces closed on gabion check delta-bundle.",
    reasoning={
        "summary": "The dataflow-stage retirement is completed: the wrapper is a hidden hard-fail stub, CI uses gabion check delta-bundle, and the closed cutover is recorded as landed touchpoint state.",
        "control": "wrapper_retirement_drain.run_dataflow_stage.touchpoint",
        "blocking_dependencies": (),
    },
    owner="gabion.tooling.runtime",
    expiry="WRD closure",
    links=[
        {"kind": "object_id", "value": "WRD"},
        {"kind": "object_id", "value": "WRD-SQ-001"},
        {"kind": "object_id", "value": "WRD-TP-001"},
    ],
)
def _wrd_tp_run_dataflow_stage() -> None:
    return None


@landed_todo_decorator(
    reason="WRD-TP-002 is recorded as landed metadata after ci local-repro and ci watch converged on the namespace-family gabion ci command surface and the wrapper surface closed onto that replacement.",
    reasoning={
        "summary": "CI local-repro and watch now publish only the gabion ci namespace surface across CLI help, tests, workflows, and wrapper stubs, and the closed cutover is recorded as landed touchpoint state.",
        "control": "wrapper_retirement_drain.ci_wrappers.touchpoint",
        "blocking_dependencies": (),
    },
    owner="gabion.tooling.runtime",
    expiry="WRD closure",
    links=[
        {"kind": "object_id", "value": "WRD"},
        {"kind": "object_id", "value": "WRD-SQ-001"},
        {"kind": "object_id", "value": "WRD-TP-002"},
    ],
)
def _wrd_tp_ci_wrappers() -> None:
    return None


@landed_todo_decorator(
    reason="WRD-TP-003 is recorded as landed metadata after ASPF handoff converged on the namespace-family gabion aspf handoff surface across CLI publication, docs, and workflows.",
    reasoning={
        "summary": "ASPF handoff now publishes and documents only gabion aspf handoff, and the closed cutover is recorded as landed touchpoint state.",
        "control": "wrapper_retirement_drain.aspf_handoff.touchpoint",
        "blocking_dependencies": (),
    },
    owner="gabion.tooling.runtime",
    expiry="WRD closure",
    links=[
        {"kind": "object_id", "value": "WRD"},
        {"kind": "object_id", "value": "WRD-SQ-001"},
        {"kind": "object_id", "value": "WRD-TP-003"},
    ],
)
def _wrd_tp_aspf_handoff() -> None:
    return None


@landed_todo_decorator(
    reason="WRD-TP-004 is recorded as landed metadata after policy check and the docflow packet loop converged on namespace-family gabion policy commands and the retired wrappers closed onto hard-fail replacement guidance.",
    reasoning={
        "summary": "Policy check and the docflow packet loop now publish only gabion policy family commands across runtime, docs, workflows, and wrapper stubs, and the closed cutover is recorded as landed touchpoint state.",
        "control": "wrapper_retirement_drain.policy_docflow_wrappers.touchpoint",
        "blocking_dependencies": (),
    },
    owner="scripts.policy",
    expiry="WRD closure",
    links=[
        {"kind": "object_id", "value": "WRD"},
        {"kind": "object_id", "value": "WRD-SQ-002"},
        {"kind": "object_id", "value": "WRD-TP-004"},
    ],
)
def _wrd_tp_policy_docflow_wrappers() -> None:
    return None


@landed_todo_decorator(
    reason="WRD-TP-005 is recorded as landed metadata after policy scanner converged on the namespace-family gabion policy scanner command across runtime, docs, workflows, and stubs.",
    reasoning={
        "summary": "Policy scanner now publishes only the gabion policy scanner surface across CLI publication, runtime invocations, docs, workflows, and stubs, and the closed cutover is recorded as landed touchpoint state.",
        "control": "wrapper_retirement_drain.policy_scanner_wrappers.touchpoint",
        "blocking_dependencies": (),
    },
    owner="scripts.policy",
    expiry="WRD closure",
    links=[
        {"kind": "object_id", "value": "WRD"},
        {"kind": "object_id", "value": "WRD-SQ-002"},
        {"kind": "object_id", "value": "WRD-TP-005"},
    ],
)
def _wrd_tp_policy_scanner_wrappers() -> None:
    return None


@landed_todo_decorator(
    reason="WRD-TP-006 is recorded as landed metadata after governance and SPPF wrappers converged on namespace-family gabion commands across runtime, docs, workflows, and stubs.",
    reasoning={
        "summary": "Governance controller-audit, telemetry-emit, and SPPF sync/status now publish only the namespace-family command surface across runtime, docs, workflows, and stubs, and the closed cutover is recorded as landed touchpoint state.",
        "control": "wrapper_retirement_drain.governance_sppf_wrappers.touchpoint",
        "blocking_dependencies": (),
    },
    owner="gabion_governance",
    expiry="WRD closure",
    links=[
        {"kind": "object_id", "value": "WRD"},
        {"kind": "object_id", "value": "WRD-SQ-003"},
        {"kind": "object_id", "value": "WRD-TP-006"},
    ],
)
def _wrd_tp_governance_sppf_wrappers() -> None:
    return None


@landed_todo_decorator(
    reason="WRD-TP-007 is recorded as landed metadata after release helpers and repo-utility wrappers converged on namespace-family gabion release and gabion repo command ownership.",
    reasoning={
        "summary": "Release helpers and repo-local operational utilities now publish only canonical release and repo command families, and the closed cutover is recorded as landed touchpoint state.",
        "control": "wrapper_retirement_drain.release_repo_wrappers.touchpoint",
        "blocking_dependencies": (),
    },
    owner="gabion.tooling.runtime",
    expiry="WRD closure",
    links=[
        {"kind": "object_id", "value": "WRD"},
        {"kind": "object_id", "value": "WRD-SQ-004"},
        {"kind": "object_id", "value": "WRD-TP-007"},
    ],
)
def _wrd_tp_release_repo_wrappers() -> None:
    return None


@landed_todo_decorator(
    reason="WRD-TP-008 is recorded as landed metadata after operator docs, workflows, Make targets, generated manifest docs, governance loop docs, and the wrapper guard converged on namespace-family gabion command guidance.",
    reasoning={
        "summary": "Operator docs, workflows, Make targets, generated manifests, governance loop docs, and the wrapper guard now all point to canonical namespace-family gabion command usage, and the closed cutover is recorded as landed touchpoint state.",
        "control": "wrapper_retirement_drain.docs_workflows_guard.touchpoint",
        "blocking_dependencies": (),
    },
    owner="gabion.tooling.runtime_policy",
    expiry="WRD closure",
    links=[
        {"kind": "object_id", "value": "WRD"},
        {"kind": "object_id", "value": "WRD-SQ-005"},
        {"kind": "object_id", "value": "WRD-TP-008"},
    ],
)
def _wrd_tp_docs_workflows_guard() -> None:
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
        surface="wrapper_retirement_drain_root",
        structural_path=f"wrapper_retirement_drain.root::{root_id}",
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
        surface="wrapper_retirement_drain_subqueue",
        structural_path=f"wrapper_retirement_drain.subqueue::{subqueue_id}",
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
    declared_touchsites: tuple[object, ...],
) -> RegisteredTouchpointDefinition:
    metadata = registry_marker_metadata(
        symbol,
        surface="wrapper_retirement_drain_touchpoint",
        structural_path=f"wrapper_retirement_drain.touchpoint::{touchpoint_id}",
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


def _touchsite(
    *,
    touchsite_id: str,
    rel_path: str,
    qualname: str,
    line: int,
    surface: str,
    structural_path: str,
    node_kind: str = "function_def",
) -> object:
    return declared_touchsite_definition(
        touchsite_id=touchsite_id,
        rel_path=rel_path,
        qualname=qualname,
        boundary_name=qualname,
        line=line,
        node_kind=node_kind,
        surface=surface,
        structural_path=structural_path,
        seam_class="surviving_carrier_seam",
    )


def wrapper_retirement_drain_workstream_registry() -> WorkstreamRegistry:
    root_id = "WRD"
    return WorkstreamRegistry(
        root=_root_definition(
            root_id=root_id,
            title="Wrapper Retirement Drain / Canonical Command Surface",
            subqueue_ids=(
                "WRD-SQ-001",
                "WRD-SQ-002",
                "WRD-SQ-003",
                "WRD-SQ-004",
                "WRD-SQ-005",
            ),
            symbol=_wrd_root,
            status_hint="landed",
        ),
        subqueues=(
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="WRD-SQ-001",
                title="Dataflow and CI wrapper drain",
                touchpoint_ids=("WRD-TP-001", "WRD-TP-002", "WRD-TP-003"),
                symbol=_wrd_sq_dataflow_ci,
                status_hint="landed",
            ),
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="WRD-SQ-002",
                title="Policy and docflow wrapper drain",
                touchpoint_ids=("WRD-TP-004", "WRD-TP-005"),
                symbol=_wrd_sq_policy_docflow,
                status_hint="landed",
            ),
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="WRD-SQ-003",
                title="Governance and SPPF wrapper drain",
                touchpoint_ids=("WRD-TP-006",),
                symbol=_wrd_sq_governance_sppf,
                status_hint="landed",
            ),
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="WRD-SQ-004",
                title="Release and repo-utility wrapper drain",
                touchpoint_ids=("WRD-TP-007",),
                symbol=_wrd_sq_release_repo_utilities,
                status_hint="landed",
            ),
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="WRD-SQ-005",
                title="Documentation, workflow, and guard cutover",
                touchpoint_ids=("WRD-TP-008",),
                symbol=_wrd_sq_docs_workflows_guard,
                status_hint="landed",
            ),
        ),
        touchpoints=(
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="WRD-SQ-001",
                touchpoint_id="WRD-TP-001",
                title="Retire run-dataflow-stage wrapper entrypoint",
                symbol=_wrd_tp_run_dataflow_stage,
                status_hint="landed",
                declared_touchsites=(
                    _touchsite(
                        touchsite_id="WRD-TP-001-TS-001",
                        rel_path="src/gabion/tooling/runtime/run_dataflow_stage.py",
                        qualname="main",
                        line=1079,
                        surface="wrapper_runtime_entrypoint",
                        structural_path="wrapper_runtime_entrypoint::run_dataflow_stage.main",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-001-TS-002",
                        rel_path="src/gabion/cli_support/tooling_commands.py",
                        qualname="register_tooling_passthrough_commands.<locals>.run_dataflow_stage",
                        line=148,
                        surface="gabion_cli_entrypoint",
                        structural_path="gabion_cli_entrypoint::run_dataflow_stage",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-001-TS-003",
                        rel_path="docs/user_workflows.md",
                        qualname="user_workflows",
                        line=56,
                        surface="operator_workflow_doc",
                        structural_path="operator_workflow_doc::user_workflows",
                        node_kind="markdown_section",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-001-TS-004",
                        rel_path=".github/workflows/ci.yml",
                        qualname="ci_workflow_dataflow_grammar_invocation",
                        line=51,
                        surface="workflow_wrapper_invocation",
                        structural_path="workflow_wrapper_invocation::ci.run_dataflow_stage",
                        node_kind="assign",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="WRD-SQ-001",
                touchpoint_id="WRD-TP-002",
                title="Migrate CI local repro and watch wrappers",
                symbol=_wrd_tp_ci_wrappers,
                status_hint="landed",
                declared_touchsites=(
                    _touchsite(
                        touchsite_id="WRD-TP-002-TS-001",
                        rel_path="src/gabion/tooling/runtime/ci_local_repro.py",
                        qualname="main",
                        line=835,
                        surface="wrapper_runtime_entrypoint",
                        structural_path="wrapper_runtime_entrypoint::ci_local_repro.main",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-002-TS-002",
                        rel_path="scripts/ci_local_repro.sh",
                        qualname="ci_local_repro_wrapper",
                        line=30,
                        surface="wrapper_shell_entrypoint",
                        structural_path="wrapper_shell_entrypoint::ci_local_repro",
                        node_kind="assign",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-002-TS-003",
                        rel_path="src/gabion/tooling/runtime/ci_watch.py",
                        qualname="main",
                        line=559,
                        surface="wrapper_runtime_entrypoint",
                        structural_path="wrapper_runtime_entrypoint::ci_watch.main",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-002-TS-004",
                        rel_path="src/gabion/cli_support/tooling_commands.py",
                        qualname="register_ci_watch_command.<locals>.ci_watch",
                        line=272,
                        surface="gabion_cli_entrypoint",
                        structural_path="gabion_cli_entrypoint::ci_watch",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-002-TS-005",
                        rel_path="CONTRIBUTING.md",
                        qualname="contributing_contract",
                        line=604,
                        surface="operator_workflow_doc",
                        structural_path="operator_workflow_doc::contributing_contract.ci_local_repro",
                        node_kind="markdown_section",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="WRD-SQ-001",
                touchpoint_id="WRD-TP-003",
                title="Migrate ASPF handoff helper entrypoints",
                symbol=_wrd_tp_aspf_handoff,
                status_hint="landed",
                declared_touchsites=(
                    _touchsite(
                        touchsite_id="WRD-TP-003-TS-001",
                        rel_path="src/gabion/tooling/runtime/aspf_handoff.py",
                        qualname="prepare_step",
                        line=114,
                        surface="gabion_runtime_helper",
                        structural_path="gabion_runtime_helper::aspf_handoff.prepare_step",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-003-TS-002",
                        rel_path="docs/user_workflows.md",
                        qualname="user_workflows",
                        line=182,
                        surface="operator_workflow_doc",
                        structural_path="operator_workflow_doc::user_workflows.aspf_handoff",
                        node_kind="markdown_section",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-003-TS-003",
                        rel_path=".github/workflows/ci.yml",
                        qualname="ci_workflow_aspf_handoff_step",
                        line=293,
                        surface="workflow_wrapper_invocation",
                        structural_path="workflow_wrapper_invocation::ci.aspf_handoff",
                        node_kind="assign",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="WRD-SQ-002",
                touchpoint_id="WRD-TP-004",
                title="Migrate policy-check and packet-loop wrappers",
                symbol=_wrd_tp_policy_docflow_wrappers,
                status_hint="landed",
                declared_touchsites=(
                    _touchsite(
                        touchsite_id="WRD-TP-004-TS-001",
                        rel_path="scripts/policy/policy_check.py",
                        qualname="main",
                        line=3194,
                        surface="wrapper_runtime_entrypoint",
                        structural_path="wrapper_runtime_entrypoint::policy_check.main",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-004-TS-002",
                        rel_path="scripts/policy/docflow_packetize.py",
                        qualname="main",
                        line=327,
                        surface="wrapper_runtime_entrypoint",
                        structural_path="wrapper_runtime_entrypoint::docflow_packetize.main",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-004-TS-003",
                        rel_path="scripts/policy/docflow_packet_enforce.py",
                        qualname="main",
                        line=394,
                        surface="wrapper_runtime_entrypoint",
                        structural_path="wrapper_runtime_entrypoint::docflow_packet_enforce.main",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-004-TS-004",
                        rel_path=".github/workflows/ci.yml",
                        qualname="ci_workflow_policy_docflow_steps",
                        line=154,
                        surface="workflow_wrapper_invocation",
                        structural_path="workflow_wrapper_invocation::ci.policy_docflow",
                        node_kind="assign",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="WRD-SQ-002",
                touchpoint_id="WRD-TP-005",
                title="Migrate policy scanner-suite style wrappers",
                symbol=_wrd_tp_policy_scanner_wrappers,
                status_hint="landed",
                declared_touchsites=(
                    _touchsite(
                        touchsite_id="WRD-TP-005-TS-001",
                        rel_path="scripts/policy/policy_scanner_suite.py",
                        qualname="main",
                        line=121,
                        surface="wrapper_runtime_entrypoint",
                        structural_path="wrapper_runtime_entrypoint::policy_scanner_suite.main",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-005-TS-002",
                        rel_path="src/gabion/tooling/runtime/checks_runtime.py",
                        qualname="main",
                        line=642,
                        surface="gabion_runtime_helper",
                        structural_path="gabion_runtime_helper::checks_runtime.main",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-005-TS-003",
                        rel_path="CONTRIBUTING.md",
                        qualname="contributing_contract",
                        line=773,
                        surface="operator_workflow_doc",
                        structural_path="operator_workflow_doc::contributing_contract.policy_check",
                        node_kind="markdown_section",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="WRD-SQ-003",
                touchpoint_id="WRD-TP-006",
                title="Migrate governance and SPPF wrappers",
                symbol=_wrd_tp_governance_sppf_wrappers,
                status_hint="landed",
                declared_touchsites=(
                    _touchsite(
                        touchsite_id="WRD-TP-006-TS-001",
                        rel_path="scripts/governance/governance_controller_audit.py",
                        qualname="main",
                        line=303,
                        surface="wrapper_runtime_entrypoint",
                        structural_path="wrapper_runtime_entrypoint::governance_controller_audit.main",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-006-TS-002",
                        rel_path="scripts/governance/governance_telemetry_emit.py",
                        qualname="main",
                        line=322,
                        surface="wrapper_runtime_entrypoint",
                        structural_path="wrapper_runtime_entrypoint::governance_telemetry_emit.main",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-006-TS-003",
                        rel_path="scripts/sppf/sppf_sync.py",
                        qualname="main",
                        line=615,
                        surface="wrapper_runtime_entrypoint",
                        structural_path="wrapper_runtime_entrypoint::sppf_sync.main",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-006-TS-004",
                        rel_path="scripts/sppf/sppf_status_audit.py",
                        qualname="main",
                        line=197,
                        surface="wrapper_runtime_entrypoint",
                        structural_path="wrapper_runtime_entrypoint::sppf_status_audit.main",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="WRD-SQ-004",
                touchpoint_id="WRD-TP-007",
                title="Migrate release and repo-utility wrappers",
                symbol=_wrd_tp_release_repo_wrappers,
                status_hint="landed",
                declared_touchsites=(
                    _touchsite(
                        touchsite_id="WRD-TP-007-TS-001",
                        rel_path="scripts/release/release_tag.py",
                        qualname="main",
                        line=36,
                        surface="wrapper_runtime_entrypoint",
                        structural_path="wrapper_runtime_entrypoint::release_tag.main",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-007-TS-002",
                        rel_path="scripts/release/release_read_project_version.py",
                        qualname="main",
                        line=16,
                        surface="wrapper_runtime_entrypoint",
                        structural_path="wrapper_runtime_entrypoint::release_read_project_version.main",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-007-TS-003",
                        rel_path="scripts/release/release_set_test_version.py",
                        qualname="main",
                        line=46,
                        surface="wrapper_runtime_entrypoint",
                        structural_path="wrapper_runtime_entrypoint::release_set_test_version.main",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-007-TS-004",
                        rel_path="scripts/release/release_verify_test_tag.py",
                        qualname="main",
                        line=16,
                        surface="wrapper_runtime_entrypoint",
                        structural_path="wrapper_runtime_entrypoint::release_verify_test_tag.main",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-007-TS-005",
                        rel_path="scripts/release/release_verify_pypi_tag.py",
                        qualname="main",
                        line=35,
                        surface="wrapper_runtime_entrypoint",
                        structural_path="wrapper_runtime_entrypoint::release_verify_pypi_tag.main",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-007-TS-006",
                        rel_path="scripts/misc/extract_test_evidence.py",
                        qualname="main",
                        line=101,
                        surface="wrapper_runtime_entrypoint",
                        structural_path="wrapper_runtime_entrypoint::extract_test_evidence.main",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-007-TS-007",
                        rel_path="scripts/misc/extract_test_behavior.py",
                        qualname="main",
                        line=30,
                        surface="wrapper_runtime_entrypoint",
                        structural_path="wrapper_runtime_entrypoint::extract_test_behavior.main",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-007-TS-008",
                        rel_path="scripts/misc/refresh_baselines.py",
                        qualname="main",
                        line=479,
                        surface="wrapper_runtime_entrypoint",
                        structural_path="wrapper_runtime_entrypoint::refresh_baselines.main",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-007-TS-009",
                        rel_path="scripts/audit_snapshot.sh",
                        qualname="audit_snapshot_wrapper",
                        line=4,
                        surface="wrapper_shell_entrypoint",
                        structural_path="wrapper_shell_entrypoint::audit_snapshot",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-007-TS-010",
                        rel_path="scripts/latest_snapshot.sh",
                        qualname="latest_snapshot_wrapper",
                        line=1,
                        surface="wrapper_shell_entrypoint",
                        structural_path="wrapper_shell_entrypoint::latest_snapshot",
                        node_kind="assign",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="WRD-SQ-005",
                touchpoint_id="WRD-TP-008",
                title="Cut over operator docs, workflows, and no-new-wrapper guard",
                symbol=_wrd_tp_docs_workflows_guard,
                status_hint="landed",
                declared_touchsites=(
                    _touchsite(
                        touchsite_id="WRD-TP-008-TS-001",
                        rel_path="docs/user_workflows.md",
                        qualname="user_workflows",
                        line=56,
                        surface="operator_workflow_doc",
                        structural_path="operator_workflow_doc::user_workflows",
                        node_kind="markdown_section",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-008-TS-002",
                        rel_path="README.md",
                        qualname="repo_contract",
                        line=241,
                        surface="operator_workflow_doc",
                        structural_path="operator_workflow_doc::repo_contract.wrapper_guidance",
                        node_kind="markdown_section",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-008-TS-003",
                        rel_path="CONTRIBUTING.md",
                        qualname="contributing_contract",
                        line=604,
                        surface="operator_workflow_doc",
                        structural_path="operator_workflow_doc::contributing_contract.wrapper_guidance",
                        node_kind="markdown_section",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-008-TS-004",
                        rel_path=".github/workflows/ci.yml",
                        qualname="ci_workflow_wrapper_invocations",
                        line=154,
                        surface="workflow_wrapper_invocation",
                        structural_path="workflow_wrapper_invocation::ci.wrapper_calls",
                        node_kind="assign",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-008-TS-005",
                        rel_path="docs/generated_artifact_manifest.md",
                        qualname="generated_artifact_manifest",
                        line=76,
                        surface="operator_workflow_doc",
                        structural_path="operator_workflow_doc::generated_artifact_manifest.md",
                        node_kind="markdown_section",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-008-TS-006",
                        rel_path="docs/generated_artifact_manifest.yaml",
                        qualname="generated_artifact_manifest",
                        line=1,
                        surface="operator_workflow_doc",
                        structural_path="operator_workflow_doc::generated_artifact_manifest.yaml",
                        node_kind="assign",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-008-TS-007",
                        rel_path="docs/governance_control_loops.yaml",
                        qualname="governance_control_loops",
                        line=1,
                        surface="operator_workflow_doc",
                        structural_path="operator_workflow_doc::governance_control_loops.yaml",
                        node_kind="assign",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-008-TS-008",
                        rel_path="docs/governance_loop_matrix.md",
                        qualname="governance_loop_matrix",
                        line=87,
                        surface="operator_workflow_doc",
                        structural_path="operator_workflow_doc::governance_loop_matrix.md",
                        node_kind="markdown_section",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-008-TS-009",
                        rel_path=".github/workflows/pr-dataflow-grammar.yml",
                        qualname="pr_dataflow_grammar_workflow_wrapper_invocations",
                        line=104,
                        surface="workflow_wrapper_invocation",
                        structural_path="workflow_wrapper_invocation::pr_dataflow_grammar.wrapper_calls",
                        node_kind="assign",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-008-TS-010",
                        rel_path=".github/workflows/release-tag.yml",
                        qualname="release_tag_workflow_wrapper_invocations",
                        line=1,
                        surface="workflow_wrapper_invocation",
                        structural_path="workflow_wrapper_invocation::release_tag.wrapper_calls",
                        node_kind="assign",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-008-TS-011",
                        rel_path=".github/workflows/auto-test-tag.yml",
                        qualname="auto_test_tag_workflow_wrapper_invocations",
                        line=1,
                        surface="workflow_wrapper_invocation",
                        structural_path="workflow_wrapper_invocation::auto_test_tag.wrapper_calls",
                        node_kind="assign",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-008-TS-012",
                        rel_path=".github/workflows/release-testpypi.yml",
                        qualname="release_testpypi_workflow_wrapper_invocations",
                        line=1,
                        surface="workflow_wrapper_invocation",
                        structural_path="workflow_wrapper_invocation::release_testpypi.wrapper_calls",
                        node_kind="assign",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-008-TS-013",
                        rel_path=".github/workflows/release-pypi.yml",
                        qualname="release_pypi_workflow_wrapper_invocations",
                        line=1,
                        surface="workflow_wrapper_invocation",
                        structural_path="workflow_wrapper_invocation::release_pypi.wrapper_calls",
                        node_kind="assign",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-008-TS-014",
                        rel_path="Makefile",
                        qualname="make_targets_wrapper_guidance",
                        line=1,
                        surface="workflow_wrapper_invocation",
                        structural_path="workflow_wrapper_invocation::make.wrapper_calls",
                        node_kind="assign",
                    ),
                    _touchsite(
                        touchsite_id="WRD-TP-008-TS-015",
                        rel_path="scripts/policy/policy_check.py",
                        qualname="main",
                        line=2077,
                        surface="workflow_wrapper_invocation",
                        structural_path="workflow_wrapper_invocation::policy_check.wrapper_guard",
                    ),
                ),
            ),
        ),
        tags=("wrapper_retirement",),
    )
