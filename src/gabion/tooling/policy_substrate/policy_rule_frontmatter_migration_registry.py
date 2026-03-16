from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Callable

from gabion.analysis.aspf.aspf_lattice_algebra import canonical_structural_identity
from gabion.analysis.foundation.marker_protocol import MarkerPayload, marker_identity
from gabion.invariants import invariant_decorations, landed_todo_decorator
from gabion.tooling.policy_substrate.site_identity import canonical_site_identity
from gabion.tooling.policy_substrate.workstream_registry import (
    RegisteredRootDefinition,
    RegisteredSubqueueDefinition,
    RegisteredTouchpointDefinition,
    WorkstreamRegistry,
    declared_touchsite_definition,
    declared_touchsite_definition_from_symbol,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class PolicyRuleFrontmatterMigrationQueueDefinition:
    queue_id: str
    title: str
    rel_path: str
    qualname: str
    line: int
    site_identity: str
    structural_identity: str
    marker_identity: str
    marker_payload: MarkerPayload
    status_hint: str
    subqueue_ids: tuple[str, ...]


@dataclass(frozen=True)
class PolicyRuleFrontmatterMigrationSubqueueDefinition:
    subqueue_id: str
    title: str
    rel_path: str
    qualname: str
    line: int
    site_identity: str
    structural_identity: str
    marker_identity: str
    marker_payload: MarkerPayload
    status_hint: str
    touchpoint_ids: tuple[str, ...]


@dataclass(frozen=True)
class PolicyRuleFrontmatterMigrationTouchpointDefinition:
    touchpoint_id: str
    subqueue_id: str
    title: str
    rel_path: str
    qualname: str
    line: int
    site_identity: str
    structural_identity: str
    marker_identity: str
    marker_payload: MarkerPayload
    status_hint: str


def _registry_site_metadata(symbol: Callable[..., object]) -> tuple[str, str, int]:
    source_path = Path(inspect.getsourcefile(symbol) or __file__).resolve()
    rel_path = str(source_path.relative_to(_REPO_ROOT))
    start_line = int(inspect.getsourcelines(symbol)[1])
    return (rel_path, str(symbol.__qualname__), start_line)


def _todo_metadata(
    symbol: Callable[..., object],
    *,
    surface: str,
    structural_path: str,
) -> tuple[MarkerPayload, str, str, str, str, str, int]:
    decorations = invariant_decorations(symbol)
    if len(decorations) != 1:
        raise ValueError("PRF registry symbols must carry exactly one invariant decoration.")
    payload = decorations[0]
    if payload.marker_kind.value != "todo":
        raise ValueError("PRF registry symbols must use todo_decorator.")
    rel_path, qualname, start_line = _registry_site_metadata(symbol)
    site_id = canonical_site_identity(
        rel_path=rel_path,
        qualname=qualname,
        line=start_line,
        column=1,
        node_kind="function_def",
        surface=surface,
    )
    structural_id = canonical_structural_identity(
        rel_path=rel_path,
        qualname=qualname,
        structural_path=structural_path,
        node_kind="function_def",
        surface=surface,
    )
    return (
        payload,
        marker_identity(payload),
        site_id,
        structural_id,
        rel_path,
        qualname,
        start_line,
    )


@landed_todo_decorator(
    reason="PRF migration queue remains machine-projected as landed metadata after the cheat-sheet mechanization follow-on converged.",
    reasoning={
        "summary": "PRF landed the markdown-frontmatter migration, governance-loop registry mechanization, policy-rule playbook rendering, clause-deck generation, and cheat-sheet follow-on rendering.",
        "control": "prf.queue.policy_rule_frontmatter_migration",
        "blocking_dependencies": (),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="graph workstream generalization superseded",
    links=[
        {"kind": "object_id", "value": "PRF"},
        {"kind": "doc_id", "value": "policy_rule_frontmatter_migration_ledger"},
        {"kind": "doc_id", "value": "enforceable_rules_cheat_sheet"},
    ],
)
def _prf_queue() -> None:
    return None


@landed_todo_decorator(
    reason="PRF-001 remains recorded as landed metadata for the frontmatter migration queue.",
    reasoning={
        "summary": "Completed landed rejection of non-object rules entries during policy document compilation.",
        "control": "prf.item.reject_non_object_rules",
        "blocking_dependencies": (),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="graph workstream generalization superseded",
    links=[
        {"kind": "object_id", "value": "PRF"},
        {"kind": "object_id", "value": "PRF-001"},
        {"kind": "doc_id", "value": "policy_rule_frontmatter_migration_ledger"},
    ],
)
def _prf_001() -> None:
    return None


@landed_todo_decorator(
    reason="PRF-002 remains recorded as landed metadata for the frontmatter migration queue.",
    reasoning={
        "summary": "Completed landed strict failure behavior for malformed YAML frontmatter.",
        "control": "prf.item.strict_frontmatter_parse_failure",
        "blocking_dependencies": (),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="graph workstream generalization superseded",
    links=[
        {"kind": "object_id", "value": "PRF"},
        {"kind": "object_id", "value": "PRF-002"},
        {"kind": "doc_id", "value": "policy_rule_frontmatter_migration_ledger"},
    ],
)
def _prf_002() -> None:
    return None


@landed_todo_decorator(
    reason="PRF-003 remains recorded as landed metadata for the frontmatter migration queue.",
    reasoning={
        "summary": "Completed landed rejection of blank playbook_anchor values.",
        "control": "prf.item.reject_blank_playbook_anchor",
        "blocking_dependencies": (),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="graph workstream generalization superseded",
    links=[
        {"kind": "object_id", "value": "PRF"},
        {"kind": "object_id", "value": "PRF-003"},
        {"kind": "doc_id", "value": "policy_rule_frontmatter_migration_ledger"},
    ],
)
def _prf_003() -> None:
    return None


@landed_todo_decorator(
    reason="PRF-004 remains recorded as landed metadata for the frontmatter migration queue.",
    reasoning={
        "summary": "Completed landed emission of per-violation grade guidance from the markdown playbook body rather than duplicated runtime strings.",
        "control": "prf.item.markdown_authoritative_grade_guidance",
        "blocking_dependencies": (),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="graph workstream generalization superseded",
    links=[
        {"kind": "object_id", "value": "PRF"},
        {"kind": "object_id", "value": "PRF-004"},
        {"kind": "doc_id", "value": "policy_rule_frontmatter_migration_ledger"},
    ],
)
def _prf_004() -> None:
    return None


@landed_todo_decorator(
    reason="PRF-005 remains recorded as landed metadata once the cheat-sheet Rule Matrix moved to structured catalog ownership.",
    reasoning={
        "summary": "Completed landed bootstrap of the first mechanically-owned cheat-sheet surface with a structured rule catalog plus renderer.",
        "control": "prf.item.enforceable_rules_cheat_sheet_catalog",
        "blocking_dependencies": (),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="cheat-sheet matrix generation converged",
    links=[
        {"kind": "object_id", "value": "PRF"},
        {"kind": "object_id", "value": "PRF-005"},
        {"kind": "doc_id", "value": "policy_rule_frontmatter_migration_ledger"},
        {"kind": "doc_id", "value": "enforceable_rules_cheat_sheet"},
    ],
)
def _prf_005() -> None:
    return None


@landed_todo_decorator(
    reason="PRF-006 remains recorded as landed metadata once governance loop docs moved to a shared normalized registry plus renderer path.",
    reasoning={
        "summary": "Completed landed migration of governance loop semantics out of duplicated prose and into a shared structured registry that renders both the loop registry and the gate matrix.",
        "control": "prf.item.governance_loop_registry_renderer",
        "blocking_dependencies": (),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="governance loop registry generation converged",
    links=[
        {"kind": "object_id", "value": "PRF"},
        {"kind": "object_id", "value": "PRF-006"},
        {"kind": "doc_id", "value": "policy_rule_frontmatter_migration_ledger"},
        {"kind": "doc_id", "value": "governance_control_loops"},
        {"kind": "doc_id", "value": "governance_loop_matrix"},
    ],
)
def _prf_006() -> None:
    return None


@landed_todo_decorator(
    reason="PRF-007 remains recorded as landed metadata once policy-rule playbook docs moved from duplicated prose to frontmatter-backed generated sections.",
    reasoning={
        "summary": "Completed landed rendering of the ambiguity-contract and grade-monotonicity playbooks from their canonical markdown-frontmatter rules.",
        "control": "prf.item.policy_rule_playbook_renderer",
        "blocking_dependencies": (),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="policy rule playbook generation converged",
    links=[
        {"kind": "object_id", "value": "PRF"},
        {"kind": "object_id", "value": "PRF-007"},
        {"kind": "doc_id", "value": "policy_rule_frontmatter_migration_ledger"},
        {"kind": "doc_id", "value": "ambiguity_contract_policy_rules"},
        {"kind": "doc_id", "value": "grade_monotonicity_policy_rules"},
    ],
)
def _prf_007() -> None:
    return None


@landed_todo_decorator(
    reason="PRF-008 remains recorded as landed metadata once AGENTS and CONTRIBUTING moved their clause-backed obligation bullets to a shared audience-specific catalog plus renderer.",
    reasoning={
        "summary": "Completed landed generation of the repetitive clause-backed obligation bullets in AGENTS and CONTRIBUTING from a small audience-specific catalog.",
        "control": "prf.item.clause_backed_obligation_decks",
        "blocking_dependencies": (),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="clause-backed obligation deck generation converged",
    links=[
        {"kind": "object_id", "value": "PRF"},
        {"kind": "object_id", "value": "PRF-008"},
        {"kind": "doc_id", "value": "policy_rule_frontmatter_migration_ledger"},
        {"kind": "doc_id", "value": "agents"},
        {"kind": "doc_id", "value": "contributing"},
    ],
)
def _prf_008() -> None:
    return None


@landed_todo_decorator(
    reason="PRF-009 remains recorded as landed metadata once the remaining cheat-sheet guardrail and validation sections moved to generated catalog ownership.",
    reasoning={
        "summary": "Completed landed rendering of cheat-sheet guardrails and validation bundles from the stabilized governance catalogs beyond the original Rule Matrix slice.",
        "control": "prf.item.cheat_sheet_guardrails_renderer",
        "blocking_dependencies": (),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="broader cheat-sheet generation converged",
    links=[
        {"kind": "object_id", "value": "PRF"},
        {"kind": "object_id", "value": "PRF-009"},
        {"kind": "doc_id", "value": "policy_rule_frontmatter_migration_ledger"},
        {"kind": "doc_id", "value": "enforceable_rules_cheat_sheet"},
    ],
)
def _prf_009() -> None:
    return None


@landed_todo_decorator(
    reason="PRF-TP-006 remains recorded as landed metadata once governance loop docs moved to a shared structured registry.",
    reasoning={
        "summary": "Completed landed mechanization of the shared governance loop registry that renders both governance loop docs from one catalog.",
        "control": "prf.touchpoint.governance_loop_registry_renderer",
        "blocking_dependencies": (),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="governance loop registry generation converged",
    links=[
        {"kind": "object_id", "value": "PRF"},
        {"kind": "object_id", "value": "PRF-006"},
        {"kind": "object_id", "value": "PRF-TP-006"},
        {"kind": "doc_id", "value": "policy_rule_frontmatter_migration_ledger"},
        {"kind": "doc_id", "value": "governance_control_loops"},
        {"kind": "doc_id", "value": "governance_loop_matrix"},
    ],
)
def _prf_tp_006() -> None:
    return None


@landed_todo_decorator(
    reason="PRF-TP-007 remains recorded as landed metadata once policy-rule playbook docs moved to generated frontmatter-backed sections.",
    reasoning={
        "summary": "Completed landed rendering of the ambiguity-contract and grade-monotonicity docs from their frontmatter-backed canonical rules metadata.",
        "control": "prf.touchpoint.policy_rule_playbook_renderer",
        "blocking_dependencies": (),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="policy rule playbook generation converged",
    links=[
        {"kind": "object_id", "value": "PRF"},
        {"kind": "object_id", "value": "PRF-007"},
        {"kind": "object_id", "value": "PRF-TP-007"},
        {"kind": "doc_id", "value": "policy_rule_frontmatter_migration_ledger"},
        {"kind": "doc_id", "value": "ambiguity_contract_policy_rules"},
        {"kind": "doc_id", "value": "grade_monotonicity_policy_rules"},
    ],
)
def _prf_tp_007() -> None:
    return None


@landed_todo_decorator(
    reason="PRF-TP-008 remains recorded as landed metadata once clause-backed obligation decks moved to the shared audience-scoped catalog plus renderer.",
    reasoning={
        "summary": "Completed landed migration of the repetitive clause bullet decks in AGENTS and CONTRIBUTING to an audience-scoped clause selection catalog and renderer.",
        "control": "prf.touchpoint.clause_backed_obligation_decks",
        "blocking_dependencies": (),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="clause-backed obligation deck generation converged",
    links=[
        {"kind": "object_id", "value": "PRF"},
        {"kind": "object_id", "value": "PRF-008"},
        {"kind": "object_id", "value": "PRF-TP-008"},
        {"kind": "doc_id", "value": "policy_rule_frontmatter_migration_ledger"},
        {"kind": "doc_id", "value": "agents"},
        {"kind": "doc_id", "value": "contributing"},
    ],
)
def _prf_tp_008() -> None:
    return None


@landed_todo_decorator(
    reason="PRF-TP-009 remains recorded as landed metadata once cheat-sheet guardrails and validation bundles moved to generated catalog ownership.",
    reasoning={
        "summary": "Completed landed rendering of the remaining repetitive cheat-sheet sections from the stabilized loop and rule catalogs rather than leaving them hand-authored.",
        "control": "prf.touchpoint.cheat_sheet_guardrails_renderer",
        "blocking_dependencies": (),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="broader cheat-sheet generation converged",
    links=[
        {"kind": "object_id", "value": "PRF"},
        {"kind": "object_id", "value": "PRF-009"},
        {"kind": "object_id", "value": "PRF-TP-009"},
        {"kind": "doc_id", "value": "policy_rule_frontmatter_migration_ledger"},
        {"kind": "doc_id", "value": "enforceable_rules_cheat_sheet"},
    ],
)
def _prf_tp_009() -> None:
    return None


def iter_prf_queues() -> tuple[PolicyRuleFrontmatterMigrationQueueDefinition, ...]:
    payload, marker_id, site_id, structural_id, rel_path, qualname, line = _todo_metadata(
        _prf_queue,
        surface="policy_rule_frontmatter_migration_queue",
        structural_path="prf.queue::PRF",
    )
    return (
        PolicyRuleFrontmatterMigrationQueueDefinition(
            queue_id="PRF",
            title="Policy rule markdown-frontmatter migration",
            rel_path=rel_path,
            qualname=qualname,
            line=line,
            site_identity=site_id,
            structural_identity=structural_id,
            marker_identity=marker_id,
            marker_payload=payload,
            status_hint="landed",
            subqueue_ids=(
                "PRF-001",
                "PRF-002",
                "PRF-003",
                "PRF-004",
                "PRF-005",
                "PRF-006",
                "PRF-007",
                "PRF-008",
                "PRF-009",
            ),
        ),
    )


def iter_prf_subqueues() -> tuple[PolicyRuleFrontmatterMigrationSubqueueDefinition, ...]:
    definitions: list[PolicyRuleFrontmatterMigrationSubqueueDefinition] = []
    for subqueue_id, title, symbol in (
        ("PRF-001", "Reject non-object `rules:` entries", _prf_001),
        ("PRF-002", "Fail strictly on malformed YAML frontmatter", _prf_002),
        ("PRF-003", "Reject blank `playbook_anchor` values", _prf_003),
        ("PRF-004", "Emit `GMP-*` guidance from markdown playbooks", _prf_004),
        (
            "PRF-005",
            "Bootstrap the enforceable-rules cheat-sheet catalog + renderer",
            _prf_005,
        ),
        (
            "PRF-006",
            "Normalize governance loop registry data and render governance loop docs",
            _prf_006,
        ),
        (
            "PRF-007",
            "Render ambiguity-contract and grade-monotonicity playbooks from markdown frontmatter",
            _prf_007,
        ),
        (
            "PRF-008",
            "Autodenormalize clause-backed obligation decks in AGENTS and CONTRIBUTING",
            _prf_008,
        ),
        (
            "PRF-009",
            "Extend cheat-sheet generation to guardrails and validation bundles",
            _prf_009,
        ),
    ):
        (
            payload,
            marker_id,
            site_id,
            structural_id,
            rel_path,
            qualname,
            line,
        ) = _todo_metadata(
            symbol,
            surface="policy_rule_frontmatter_migration_subqueue",
            structural_path=f"prf.subqueue::{subqueue_id}",
        )
        definitions.append(
            PolicyRuleFrontmatterMigrationSubqueueDefinition(
                subqueue_id=subqueue_id,
                title=title,
                rel_path=rel_path,
                qualname=qualname,
                line=line,
                site_identity=site_id,
                structural_identity=structural_id,
                marker_identity=marker_id,
                marker_payload=payload,
                status_hint={
                    "PRF-005": "landed",
                    "PRF-006": "landed",
                    "PRF-007": "landed",
                    "PRF-008": "landed",
                    "PRF-009": "landed",
                }.get(subqueue_id, "landed"),
                touchpoint_ids={
                    "PRF-006": ("PRF-TP-006",),
                    "PRF-007": ("PRF-TP-007",),
                    "PRF-008": ("PRF-TP-008",),
                    "PRF-009": ("PRF-TP-009",),
                }.get(subqueue_id, ()),
            )
        )
    return tuple(definitions)


def iter_prf_touchpoints() -> tuple[PolicyRuleFrontmatterMigrationTouchpointDefinition, ...]:
    definitions: list[PolicyRuleFrontmatterMigrationTouchpointDefinition] = []
    for touchpoint_id, subqueue_id, title, symbol in (
        (
            "PRF-TP-006",
            "PRF-006",
            "Render governance loop registry and matrix from a shared structured catalog",
            _prf_tp_006,
        ),
        (
            "PRF-TP-007",
            "PRF-007",
            "Render policy-rule playbook sections from markdown frontmatter",
            _prf_tp_007,
        ),
        (
            "PRF-TP-008",
            "PRF-008",
            "Generate clause-backed obligation decks for agent/contributor docs",
            _prf_tp_008,
        ),
        (
            "PRF-TP-009",
            "PRF-009",
            "Generate cheat-sheet guardrails and validation bundles from governance catalogs",
            _prf_tp_009,
        ),
    ):
        (
            payload,
            marker_id,
            site_id,
            structural_id,
            rel_path,
            qualname,
            line,
        ) = _todo_metadata(
            symbol,
            surface="policy_rule_frontmatter_migration_touchpoint",
            structural_path=f"prf.touchpoint::{touchpoint_id}",
        )
        definitions.append(
            PolicyRuleFrontmatterMigrationTouchpointDefinition(
                touchpoint_id=touchpoint_id,
                subqueue_id=subqueue_id,
                title=title,
                rel_path=rel_path,
                qualname=qualname,
                line=line,
                site_identity=site_id,
                structural_identity=structural_id,
                marker_identity=marker_id,
                marker_payload=payload,
                status_hint="landed",
            )
        )
    return tuple(definitions)


def prf_workstream_registry() -> WorkstreamRegistry:
    queue_definition = iter_prf_queues()[0]
    subqueue_definitions = iter_prf_subqueues()
    touchpoint_definitions = iter_prf_touchpoints()
    return WorkstreamRegistry(
        root=RegisteredRootDefinition(
            root_id=queue_definition.queue_id,
            title=queue_definition.title,
            rel_path=queue_definition.rel_path,
            qualname=queue_definition.qualname,
            line=queue_definition.line,
            site_identity=queue_definition.site_identity,
            structural_identity=queue_definition.structural_identity,
            marker_identity=queue_definition.marker_identity,
            marker_payload=queue_definition.marker_payload,
            subqueue_ids=queue_definition.subqueue_ids,
            status_hint=queue_definition.status_hint,
        ),
        subqueues=tuple(
            RegisteredSubqueueDefinition(
                root_id=queue_definition.queue_id,
                subqueue_id=item.subqueue_id,
                title=item.title,
                rel_path=item.rel_path,
                qualname=item.qualname,
                line=item.line,
                site_identity=item.site_identity,
                structural_identity=item.structural_identity,
                marker_identity=item.marker_identity,
                marker_payload=item.marker_payload,
                touchpoint_ids=item.touchpoint_ids,
                status_hint=item.status_hint,
            )
            for item in subqueue_definitions
        ),
        touchpoints=tuple(
            RegisteredTouchpointDefinition(
                root_id=queue_definition.queue_id,
                touchpoint_id=item.touchpoint_id,
                subqueue_id=item.subqueue_id,
                title=item.title,
                rel_path=item.rel_path,
                qualname=item.qualname,
                line=item.line,
                site_identity=item.site_identity,
                structural_identity=item.structural_identity,
                marker_identity=item.marker_identity,
                marker_payload=item.marker_payload,
                status_hint=item.status_hint,
                declared_touchsites={
                    "PRF-TP-006": (
                        declared_touchsite_definition(
                            touchsite_id="PRF-TS-006-A",
                            rel_path="docs/governance_control_loops.md",
                            qualname="governance_control_loops#registry",
                            boundary_name="governance_control_loops#registry",
                            line=1,
                            node_kind="document",
                            surface="policy_rule_frontmatter_migration_touchsite",
                            structural_path="prf.touchsite::PRF-TS-006-A",
                        ),
                        declared_touchsite_definition(
                            touchsite_id="PRF-TS-006-B",
                            rel_path="docs/governance_loop_matrix.md",
                            qualname="governance_loop_matrix#generated_matrix",
                            boundary_name="governance_loop_matrix#generated_matrix",
                            line=1,
                            node_kind="document",
                            surface="policy_rule_frontmatter_migration_touchsite",
                            structural_path="prf.touchsite::PRF-TS-006-B",
                        ),
                        declared_touchsite_definition(
                            touchsite_id="PRF-TS-006-C",
                            rel_path="docs/governance_rules.yaml",
                            qualname="governance_rules.gates",
                            boundary_name="governance_rules.gates",
                            line=1,
                            node_kind="document",
                            surface="policy_rule_frontmatter_migration_touchsite",
                            structural_path="prf.touchsite::PRF-TS-006-C",
                        ),
                    ),
                    "PRF-TP-007": (
                        declared_touchsite_definition(
                            touchsite_id="PRF-TS-007-A",
                            rel_path="docs/policy_rules/ambiguity_contract.md",
                            qualname="ambiguity_contract_policy_rules",
                            boundary_name="ambiguity_contract_policy_rules",
                            line=1,
                            node_kind="document",
                            surface="policy_rule_frontmatter_migration_touchsite",
                            structural_path="prf.touchsite::PRF-TS-007-A",
                        ),
                        declared_touchsite_definition(
                            touchsite_id="PRF-TS-007-B",
                            rel_path="docs/policy_rules/grade_monotonicity.md",
                            qualname="grade_monotonicity_policy_rules",
                            boundary_name="grade_monotonicity_policy_rules",
                            line=1,
                            node_kind="document",
                            surface="policy_rule_frontmatter_migration_touchsite",
                            structural_path="prf.touchsite::PRF-TS-007-B",
                        ),
                        declared_touchsite_definition(
                            touchsite_id="PRF-TS-007-C",
                            rel_path="src/gabion/tooling/policy_substrate/policy_rule_playbook_docs.py",
                            qualname="render_policy_rule_playbook_docs",
                            boundary_name="render_policy_rule_playbook_docs",
                            line=1,
                            node_kind="module",
                            surface="policy_rule_frontmatter_migration_touchsite",
                            structural_path="prf.touchsite::PRF-TS-007-C",
                        ),
                        declared_touchsite_definition(
                            touchsite_id="PRF-TS-007-D",
                            rel_path="scripts/policy/render_policy_rule_playbooks.py",
                            qualname="render_policy_rule_playbooks",
                            boundary_name="render_policy_rule_playbooks",
                            line=1,
                            node_kind="module",
                            surface="policy_rule_frontmatter_migration_touchsite",
                            structural_path="prf.touchsite::PRF-TS-007-D",
                        ),
                    ),
                    "PRF-TP-008": (
                        declared_touchsite_definition(
                            touchsite_id="PRF-TS-008-A",
                            rel_path="AGENTS.md",
                            qualname="AGENTS.md#agent_obligations",
                            boundary_name="AGENTS.md#agent_obligations",
                            line=1,
                            node_kind="document",
                            surface="policy_rule_frontmatter_migration_touchsite",
                            structural_path="prf.touchsite::PRF-TS-008-A",
                        ),
                        declared_touchsite_definition(
                            touchsite_id="PRF-TS-008-B",
                            rel_path="CONTRIBUTING.md",
                            qualname="CONTRIBUTING.md#contributing_contract",
                            boundary_name="CONTRIBUTING.md#contributing_contract",
                            line=1,
                            node_kind="document",
                            surface="policy_rule_frontmatter_migration_touchsite",
                            structural_path="prf.touchsite::PRF-TS-008-B",
                        ),
                        declared_touchsite_definition(
                            touchsite_id="PRF-TS-008-C",
                            rel_path="docs/normative_clause_index.md",
                            qualname="normative_clause_index",
                            boundary_name="normative_clause_index",
                            line=1,
                            node_kind="document",
                            surface="policy_rule_frontmatter_migration_touchsite",
                            structural_path="prf.touchsite::PRF-TS-008-C",
                        ),
                        declared_touchsite_definition(
                            touchsite_id="PRF-TS-008-D",
                            rel_path="docs/clause_obligation_decks.yaml",
                            qualname="clause_obligation_decks",
                            boundary_name="clause_obligation_decks",
                            line=1,
                            node_kind="document",
                            surface="policy_rule_frontmatter_migration_touchsite",
                            structural_path="prf.touchsite::PRF-TS-008-D",
                        ),
                        declared_touchsite_definition(
                            touchsite_id="PRF-TS-008-E",
                            rel_path="src/gabion/tooling/policy_substrate/clause_obligation_decks.py",
                            qualname="render_clause_obligation_decks",
                            boundary_name="render_clause_obligation_decks",
                            line=1,
                            node_kind="module",
                            surface="policy_rule_frontmatter_migration_touchsite",
                            structural_path="prf.touchsite::PRF-TS-008-E",
                        ),
                        declared_touchsite_definition(
                            touchsite_id="PRF-TS-008-F",
                            rel_path="scripts/policy/render_clause_obligation_decks.py",
                            qualname="render_clause_obligation_decks",
                            boundary_name="render_clause_obligation_decks",
                            line=1,
                            node_kind="module",
                            surface="policy_rule_frontmatter_migration_touchsite",
                            structural_path="prf.touchsite::PRF-TS-008-F",
                        ),
                    ),
                    "PRF-TP-009": (
                        declared_touchsite_definition(
                            touchsite_id="PRF-TS-009-A",
                            rel_path="docs/enforceable_rules_cheat_sheet.md",
                            qualname="enforceable_rules_cheat_sheet",
                            boundary_name="enforceable_rules_cheat_sheet",
                            line=1,
                            node_kind="document",
                            surface="policy_rule_frontmatter_migration_touchsite",
                            structural_path="prf.touchsite::PRF-TS-009-A",
                        ),
                        declared_touchsite_definition(
                            touchsite_id="PRF-TS-009-B",
                            rel_path="docs/enforceable_rules_catalog.yaml",
                            qualname="enforceable_rules_catalog",
                            boundary_name="enforceable_rules_catalog",
                            line=1,
                            node_kind="document",
                            surface="policy_rule_frontmatter_migration_touchsite",
                            structural_path="prf.touchsite::PRF-TS-009-B",
                        ),
                        declared_touchsite_definition(
                            touchsite_id="PRF-TS-009-C",
                            rel_path="docs/governance_control_loops.yaml",
                            qualname="governance_control_loops",
                            boundary_name="governance_control_loops",
                            line=1,
                            node_kind="document",
                            surface="policy_rule_frontmatter_migration_touchsite",
                            structural_path="prf.touchsite::PRF-TS-009-C",
                        ),
                        declared_touchsite_definition(
                            touchsite_id="PRF-TS-009-D",
                            rel_path="src/gabion/tooling/policy_substrate/enforceable_rules_cheat_sheet.py",
                            qualname="render_enforceable_rules_cheat_sheet",
                            boundary_name="render_enforceable_rules_cheat_sheet",
                            line=1,
                            node_kind="module",
                            surface="policy_rule_frontmatter_migration_touchsite",
                            structural_path="prf.touchsite::PRF-TS-009-D",
                        ),
                        declared_touchsite_definition(
                            touchsite_id="PRF-TS-009-E",
                            rel_path="scripts/policy/render_enforceable_rules_cheat_sheet.py",
                            qualname="render_enforceable_rules_cheat_sheet",
                            boundary_name="render_enforceable_rules_cheat_sheet",
                            line=1,
                            node_kind="module",
                            surface="policy_rule_frontmatter_migration_touchsite",
                            structural_path="prf.touchsite::PRF-TS-009-E",
                        ),
                    ),
                }[item.touchpoint_id],
            )
            for item in touchpoint_definitions
        ),
        tags=("registry_convergence",),
    )


__all__ = [
    "PolicyRuleFrontmatterMigrationQueueDefinition",
    "PolicyRuleFrontmatterMigrationSubqueueDefinition",
    "PolicyRuleFrontmatterMigrationTouchpointDefinition",
    "iter_prf_queues",
    "iter_prf_subqueues",
    "iter_prf_touchpoints",
    "prf_workstream_registry",
]
