from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Callable

from gabion.analysis.aspf.aspf_lattice_algebra import canonical_structural_identity
from gabion.analysis.foundation.marker_protocol import MarkerPayload, marker_identity
from gabion.invariants import invariant_decorations, todo_decorator
from gabion.tooling.policy_substrate.site_identity import canonical_site_identity
from gabion.tooling.policy_substrate.workstream_registry import (
    RegisteredRootDefinition,
    RegisteredSubqueueDefinition,
    WorkstreamRegistry,
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


@todo_decorator(
    reason="PRF migration queue remains machine-projected as a landed multi-workstream proof surface.",
    reasoning={
        "summary": "PRF is fully landed, but the workstream identities remain dogfooded in tooling metadata so the invariant graph can project more than one queue family.",
        "control": "prf.queue.policy_rule_frontmatter_migration",
        "blocking_dependencies": ("PRF-001", "PRF-002", "PRF-003", "PRF-004"),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="graph workstream generalization superseded",
    links=[
        {"kind": "object_id", "value": "PRF"},
        {"kind": "doc_id", "value": "policy_rule_frontmatter_migration_ledger"},
    ],
)
def _prf_queue() -> None:
    return None


@todo_decorator(
    reason="PRF-001 remains recorded as landed metadata for the frontmatter migration queue.",
    reasoning={
        "summary": "Reject non-object rules entries during policy document compilation.",
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


@todo_decorator(
    reason="PRF-002 remains recorded as landed metadata for the frontmatter migration queue.",
    reasoning={
        "summary": "Treat malformed YAML frontmatter as a strict compiler failure.",
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


@todo_decorator(
    reason="PRF-003 remains recorded as landed metadata for the frontmatter migration queue.",
    reasoning={
        "summary": "Reject blank playbook_anchor values.",
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


@todo_decorator(
    reason="PRF-004 remains recorded as landed metadata for the frontmatter migration queue.",
    reasoning={
        "summary": "Per-violation grade guidance is emitted from the markdown playbook body rather than duplicated runtime strings.",
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
            subqueue_ids=("PRF-001", "PRF-002", "PRF-003", "PRF-004"),
        ),
    )


def iter_prf_subqueues() -> tuple[PolicyRuleFrontmatterMigrationSubqueueDefinition, ...]:
    definitions: list[PolicyRuleFrontmatterMigrationSubqueueDefinition] = []
    for subqueue_id, title, symbol in (
        ("PRF-001", "Reject non-object `rules:` entries", _prf_001),
        ("PRF-002", "Fail strictly on malformed YAML frontmatter", _prf_002),
        ("PRF-003", "Reject blank `playbook_anchor` values", _prf_003),
        ("PRF-004", "Emit `GMP-*` guidance from markdown playbooks", _prf_004),
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
                status_hint="landed",
            )
        )
    return tuple(definitions)


def prf_workstream_registry() -> WorkstreamRegistry:
    queue_definition = iter_prf_queues()[0]
    subqueue_definitions = iter_prf_subqueues()
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
                touchpoint_ids=(),
                status_hint=item.status_hint,
            )
            for item in subqueue_definitions
        ),
        tags=("registry_convergence",),
    )


__all__ = [
    "PolicyRuleFrontmatterMigrationQueueDefinition",
    "PolicyRuleFrontmatterMigrationSubqueueDefinition",
    "iter_prf_queues",
    "iter_prf_subqueues",
    "prf_workstream_registry",
]
