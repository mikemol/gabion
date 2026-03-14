from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
import json
from pathlib import Path
import re
import subprocess
from typing import cast
from xml.etree import ElementTree

from gabion.analysis.core.type_fingerprints import PrimeRegistry
from gabion.frontmatter import FrontmatterParseError, parse_strict_yaml_frontmatter
from gabion.json_types import JSONValue
from gabion.policy_dsl.compile import compile_document
from gabion.policy_dsl.registry import _build_registry_for_root
from gabion.server_core.command_orchestrator_primitives import (
    _normalize_dataflow_response,
)
from gabion.tooling.docflow.compliance_identity import (
    stable_docflow_compliance_row_id,
)
from gabion.tooling.policy_substrate.identity_zone import (
    IdentityAtom,
    IdentityDecomposition,
    IdentityDecompositionRelation,
    IdentityLocalInterner,
)
from gabion_governance import governance_audit_impl


class StructuredArtifactKind(StrEnum):
    TEST_EVIDENCE = "test_evidence"
    JUNIT_FAILURES = "junit_failures"
    DOCFLOW_COMPLIANCE = "docflow_compliance"
    DOCFLOW_PACKET_ENFORCEMENT = "docflow_packet_enforcement"
    CONTROLLER_DRIFT = "controller_drift"
    LOCAL_REPRO_CLOSURE_LEDGER = "local_repro_closure_ledger"
    LOCAL_CI_REPRO_CONTRACT = "local_ci_repro_contract"
    KERNEL_VM_ALIGNMENT = "kernel_vm_alignment"
    IDENTITY_GRAMMAR_COMPLETION = "identity_grammar_completion"
    GIT_STATE = "git_state"
    CROSS_ORIGIN_WITNESS_CONTRACT = "cross_origin_witness_contract"
    INGRESS_MERGE_PARITY = "ingress_merge_parity"


class StructuredArtifactIdentityNamespace(StrEnum):
    ARTIFACT = "structured_artifact.artifact"
    UNIT = "structured_artifact.unit"
    DECOMPOSITION = "structured_artifact.decomposition"


class StructuredArtifactDecompositionKind(StrEnum):
    CANONICAL = "canonical"
    ARTIFACT_KIND = "artifact_kind"
    SOURCE_PATH = "source_path"
    SOURCE_PATH_SEGMENT = "source_path_segment"
    ITEM_KIND = "item_kind"
    ITEM_KEY = "item_key"
    ITEM_KEY_SEGMENT = "item_key_segment"


class StructuredArtifactDecompositionRelationKind(StrEnum):
    CANONICAL_OF = "canonical_of"
    ALTERNATE_OF = "alternate_of"
    EQUIVALENT_UNDER = "equivalent_under"
    DERIVED_FROM = "derived_from"


_PrimeBackedIdentity = IdentityAtom[StructuredArtifactIdentityNamespace]


StructuredArtifactDecompositionIdentity = IdentityDecomposition[
    StructuredArtifactIdentityNamespace,
    StructuredArtifactDecompositionKind,
]


StructuredArtifactDecompositionRelation = IdentityDecompositionRelation[
    StructuredArtifactIdentityNamespace,
    StructuredArtifactDecompositionKind,
    StructuredArtifactDecompositionRelationKind,
]


@dataclass(frozen=True, order=True)
class StructuredArtifactIdentity:
    canonical: _PrimeBackedIdentity
    artifact_kind: StructuredArtifactKind = field(compare=False)
    source_path: str = field(compare=False)
    item_kind: str = field(compare=False, default="")
    item_key: str = field(compare=False, default="")
    label: str = field(compare=False, default="")
    _decomposition_loader: Callable[
        [],
        tuple[
            tuple[StructuredArtifactDecompositionIdentity, ...],
            tuple[StructuredArtifactDecompositionRelation, ...],
        ],
    ] | None = field(default=None, compare=False, repr=False)
    _decompositions: tuple[StructuredArtifactDecompositionIdentity, ...] = field(
        default=(),
        compare=False,
        repr=False,
    )
    _relations: tuple[StructuredArtifactDecompositionRelation, ...] = field(
        default=(),
        compare=False,
        repr=False,
    )

    def wire(self) -> str:
        return self.canonical.token

    def __str__(self) -> str:
        return self.label or self.canonical.token

    def _ensure_decomposition_bundle(
        self,
    ) -> tuple[
        tuple[StructuredArtifactDecompositionIdentity, ...],
        tuple[StructuredArtifactDecompositionRelation, ...],
    ]:
        if self._decomposition_loader is None:
            return (self._decompositions, self._relations)
        if self._decompositions or self._relations:
            return (self._decompositions, self._relations)
        decompositions, relations = self._decomposition_loader()
        object.__setattr__(self, "_decompositions", decompositions)
        object.__setattr__(self, "_relations", relations)
        object.__setattr__(self, "_decomposition_loader", None)
        return (decompositions, relations)

    @property
    def decompositions(self) -> tuple[StructuredArtifactDecompositionIdentity, ...]:
        decompositions, _ = self._ensure_decomposition_bundle()
        return decompositions

    @property
    def relations(self) -> tuple[StructuredArtifactDecompositionRelation, ...]:
        _, relations = self._ensure_decomposition_bundle()
        return relations

    def to_payload(self) -> dict[str, object]:
        return {
            "wire": self.wire(),
            "artifact_kind": self.artifact_kind.value,
            "source_path": self.source_path,
            "item_kind": self.item_kind,
            "item_key": self.item_key,
            "label": self.label or self.wire(),
            "decompositions": [item.to_payload() for item in self.decompositions],
            "relations": [item.to_payload() for item in self.relations],
        }


@dataclass
class StructuredArtifactIdentitySpace:
    registry: PrimeRegistry = field(default_factory=PrimeRegistry)
    _interner: IdentityLocalInterner[StructuredArtifactIdentityNamespace] = field(
        init=False,
        repr=False,
    )
    _decomposition_cache: dict[
        _PrimeBackedIdentity,
        tuple[
            tuple[StructuredArtifactDecompositionIdentity, ...],
            tuple[StructuredArtifactDecompositionRelation, ...],
        ],
    ] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._interner = IdentityLocalInterner(registry=self.registry)

    @staticmethod
    def _structural_segments(value: str) -> tuple[str, ...]:
        return IdentityLocalInterner.structural_segments(value)

    def _identity(
        self,
        *,
        namespace: StructuredArtifactIdentityNamespace,
        token: str,
    ) -> _PrimeBackedIdentity:
        return self._interner.identity(namespace=namespace, token=token)

    def _decomposition_identity(
        self,
        *,
        origin: _PrimeBackedIdentity,
        decomposition_kind: StructuredArtifactDecompositionKind,
        label: str,
        part_index: int = -1,
    ) -> StructuredArtifactDecompositionIdentity:
        return self._interner.decomposition_identity(
            origin=origin,
            decomposition_namespace=StructuredArtifactIdentityNamespace.DECOMPOSITION,
            decomposition_kind=decomposition_kind,
            label=label,
            part_index=part_index,
            token_builder=lambda item_origin, item_kind, item_label, item_part_index: (
                f"{item_origin.token}|{item_kind.value}|{item_part_index}|{item_label}"
            ),
        )

    def _decomposition_bundle(
        self,
        *,
        origin: _PrimeBackedIdentity,
        artifact_kind: StructuredArtifactKind,
        source_path: str,
        item_kind: str,
        item_key: str,
    ) -> tuple[
        tuple[StructuredArtifactDecompositionIdentity, ...],
        tuple[StructuredArtifactDecompositionRelation, ...],
    ]:
        cached = self._decomposition_cache.get(origin)
        if cached is not None:
            return cached

        canonical = self._decomposition_identity(
            origin=origin,
            decomposition_kind=StructuredArtifactDecompositionKind.CANONICAL,
            label=origin.token,
        )
        artifact_kind_view = self._decomposition_identity(
            origin=origin,
            decomposition_kind=StructuredArtifactDecompositionKind.ARTIFACT_KIND,
            label=artifact_kind.value,
        )
        source_path_view = self._decomposition_identity(
            origin=origin,
            decomposition_kind=StructuredArtifactDecompositionKind.SOURCE_PATH,
            label=source_path,
        )
        source_path_segments = tuple(
            self._decomposition_identity(
                origin=origin,
                decomposition_kind=StructuredArtifactDecompositionKind.SOURCE_PATH_SEGMENT,
                label=segment,
                part_index=index,
            )
            for index, segment in enumerate(self._structural_segments(source_path))
        )
        item_kind_view = (
            self._decomposition_identity(
                origin=origin,
                decomposition_kind=StructuredArtifactDecompositionKind.ITEM_KIND,
                label=item_kind,
            )
            if item_kind
            else None
        )
        item_key_view = (
            self._decomposition_identity(
                origin=origin,
                decomposition_kind=StructuredArtifactDecompositionKind.ITEM_KEY,
                label=item_key,
            )
            if item_key
            else None
        )
        item_key_segments = tuple(
            self._decomposition_identity(
                origin=origin,
                decomposition_kind=StructuredArtifactDecompositionKind.ITEM_KEY_SEGMENT,
                label=segment,
                part_index=index,
            )
            for index, segment in enumerate(self._structural_segments(item_key))
        )
        decompositions = tuple(
            item
            for item in (
                canonical,
                artifact_kind_view,
                source_path_view,
                item_kind_view,
                item_key_view,
                *source_path_segments,
                *item_key_segments,
            )
            if item is not None
        )
        relations: list[StructuredArtifactDecompositionRelation] = [
            StructuredArtifactDecompositionRelation(
                relation_kind=StructuredArtifactDecompositionRelationKind.CANONICAL_OF,
                source=canonical,
                target=artifact_kind_view,
                rationale="artifact_kind_view",
            ),
            StructuredArtifactDecompositionRelation(
                relation_kind=StructuredArtifactDecompositionRelationKind.ALTERNATE_OF,
                source=artifact_kind_view,
                target=canonical,
                rationale="artifact_kind_view",
            ),
            StructuredArtifactDecompositionRelation(
                relation_kind=StructuredArtifactDecompositionRelationKind.EQUIVALENT_UNDER,
                source=artifact_kind_view,
                target=canonical,
                rationale="artifact_kind",
            ),
            StructuredArtifactDecompositionRelation(
                relation_kind=StructuredArtifactDecompositionRelationKind.CANONICAL_OF,
                source=canonical,
                target=source_path_view,
                rationale="source_path_view",
            ),
            StructuredArtifactDecompositionRelation(
                relation_kind=StructuredArtifactDecompositionRelationKind.ALTERNATE_OF,
                source=source_path_view,
                target=canonical,
                rationale="source_path_view",
            ),
            StructuredArtifactDecompositionRelation(
                relation_kind=StructuredArtifactDecompositionRelationKind.EQUIVALENT_UNDER,
                source=source_path_view,
                target=canonical,
                rationale="source_path",
            ),
        ]
        if item_kind_view is not None:
            relations.extend(
                (
                    StructuredArtifactDecompositionRelation(
                        relation_kind=StructuredArtifactDecompositionRelationKind.CANONICAL_OF,
                        source=canonical,
                        target=item_kind_view,
                        rationale="item_kind_view",
                    ),
                    StructuredArtifactDecompositionRelation(
                        relation_kind=StructuredArtifactDecompositionRelationKind.ALTERNATE_OF,
                        source=item_kind_view,
                        target=canonical,
                        rationale="item_kind_view",
                    ),
                    StructuredArtifactDecompositionRelation(
                        relation_kind=StructuredArtifactDecompositionRelationKind.EQUIVALENT_UNDER,
                        source=item_kind_view,
                        target=canonical,
                        rationale="item_kind",
                    ),
                )
            )
        if item_key_view is not None:
            relations.extend(
                (
                    StructuredArtifactDecompositionRelation(
                        relation_kind=StructuredArtifactDecompositionRelationKind.CANONICAL_OF,
                        source=canonical,
                        target=item_key_view,
                        rationale="item_key_view",
                    ),
                    StructuredArtifactDecompositionRelation(
                        relation_kind=StructuredArtifactDecompositionRelationKind.ALTERNATE_OF,
                        source=item_key_view,
                        target=canonical,
                        rationale="item_key_view",
                    ),
                    StructuredArtifactDecompositionRelation(
                        relation_kind=StructuredArtifactDecompositionRelationKind.EQUIVALENT_UNDER,
                        source=item_key_view,
                        target=canonical,
                        rationale="item_key",
                    ),
                )
            )
        relations.extend(
            StructuredArtifactDecompositionRelation(
                relation_kind=StructuredArtifactDecompositionRelationKind.DERIVED_FROM,
                source=item,
                target=source_path_view,
                rationale="source_path_segment",
            )
            for item in source_path_segments
        )
        if item_key_view is not None:
            relations.extend(
                StructuredArtifactDecompositionRelation(
                    relation_kind=StructuredArtifactDecompositionRelationKind.DERIVED_FROM,
                    source=item,
                    target=item_key_view,
                    rationale="item_key_segment",
                )
                for item in item_key_segments
            )
        bundle = (decompositions, tuple(relations))
        self._decomposition_cache[origin] = bundle
        return bundle

    def artifact_id(
        self,
        *,
        artifact_kind: StructuredArtifactKind,
        source_path: str,
        label: str = "",
    ) -> StructuredArtifactIdentity:
        canonical = self._identity(
            namespace=StructuredArtifactIdentityNamespace.ARTIFACT,
            token=f"{artifact_kind.value}:{source_path}",
        )
        decompositions, relations = self._decomposition_bundle(
            origin=canonical,
            artifact_kind=artifact_kind,
            source_path=source_path,
            item_kind="",
            item_key="",
        )
        return StructuredArtifactIdentity(
            canonical=canonical,
            artifact_kind=artifact_kind,
            source_path=source_path,
            label=label,
            _decompositions=decompositions,
            _relations=relations,
        )

    def item_id(
        self,
        *,
        artifact_kind: StructuredArtifactKind,
        source_path: str,
        item_kind: str,
        item_key: str,
        label: str = "",
    ) -> StructuredArtifactIdentity:
        canonical = self._identity(
            namespace=StructuredArtifactIdentityNamespace.UNIT,
            token=f"{artifact_kind.value}:{source_path}:{item_kind}:{item_key}",
        )
        return StructuredArtifactIdentity(
            canonical=canonical,
            artifact_kind=artifact_kind,
            source_path=source_path,
            item_kind=item_kind,
            item_key=item_key,
            label=label,
            _decomposition_loader=lambda: self._decomposition_bundle(
                origin=canonical,
                artifact_kind=artifact_kind,
                source_path=source_path,
                item_kind=item_kind,
                item_key=item_key,
            ),
        )


@dataclass(frozen=True)
class StructuredArtifactSource:
    rel_path: str
    schema_version: int = 0
    producer: str = ""

    def __str__(self) -> str:
        return self.rel_path


@dataclass(frozen=True)
class TestEvidenceSite:
    path: str
    qualname: str = ""
    span: tuple[int, int, int, int] = ()

    def __str__(self) -> str:
        return self.qualname or self.path


@dataclass(frozen=True)
class TestEvidenceCase:
    identity: StructuredArtifactIdentity
    test_id: str
    rel_path: str
    line: int
    status: str
    evidence_sites: tuple[TestEvidenceSite, ...]

    def __str__(self) -> str:
        return self.test_id


@dataclass(frozen=True)
class TestEvidenceArtifact:
    identity: StructuredArtifactIdentity
    source: StructuredArtifactSource
    cases: tuple[TestEvidenceCase, ...]

    def __str__(self) -> str:
        return f"{self.source.rel_path} ({len(self.cases)} tests)"


@dataclass(frozen=True)
class JUnitFailureCase:
    identity: StructuredArtifactIdentity
    test_id: str
    raw_name: str
    classname: str
    rel_path: str
    line: int
    failure_kind: str
    title: str
    message: str
    traceback_text: str

    def __str__(self) -> str:
        return self.title or self.test_id


@dataclass(frozen=True)
class JUnitFailureArtifact:
    identity: StructuredArtifactIdentity
    source: StructuredArtifactSource
    failures: tuple[JUnitFailureCase, ...]

    def __str__(self) -> str:
        return f"{self.source.rel_path} ({len(self.failures)} failures)"


@dataclass(frozen=True)
class DocflowComplianceRow:
    identity: StructuredArtifactIdentity
    row_id: str
    status: str
    invariant: str
    invariant_kind: str
    rel_path: str
    source_row_kind: str
    detail: str
    evidence_id: str

    def __str__(self) -> str:
        return self.row_id or self.invariant or self.identity.wire()


@dataclass(frozen=True)
class DocflowObligationEntry:
    identity: StructuredArtifactIdentity
    obligation_id: str
    triggered: bool
    status: str
    enforcement: str
    description: str

    def __str__(self) -> str:
        return self.obligation_id or self.identity.wire()


@dataclass(frozen=True)
class DocflowCommit:
    identity: StructuredArtifactIdentity
    sha: str
    subject: str

    def __str__(self) -> str:
        return self.sha[:12] if self.sha else self.identity.wire()


@dataclass(frozen=True)
class DocflowIssueReference:
    identity: StructuredArtifactIdentity
    issue_id: str
    commit_count: int

    def __str__(self) -> str:
        return f"GH-{self.issue_id}" if self.issue_id else self.identity.wire()


@dataclass(frozen=True)
class DocflowIssueLifecycle:
    identity: StructuredArtifactIdentity
    issue_id: str
    state: str
    labels: tuple[str, ...]
    url: str

    def __str__(self) -> str:
        if self.issue_id:
            return f"GH-{self.issue_id}"
        return self.identity.wire()


@dataclass(frozen=True)
class DocflowComplianceArtifact:
    identity: StructuredArtifactIdentity
    source: StructuredArtifactSource
    compliant_count: int
    contradiction_count: int
    excess_count: int
    proposed_count: int
    rev_range: str
    changed_paths: tuple[str, ...]
    sppf_relevant_paths_changed: bool
    gh_reference_validated: bool
    baseline_write_emitted: bool
    delta_guard_checked: bool
    doc_status_changed: bool
    checklist_influence_consistent: bool
    unmet_fail_count: int
    unmet_warn_count: int
    commits: tuple[DocflowCommit, ...]
    issue_references: tuple[DocflowIssueReference, ...]
    issue_lifecycle_fetch_status: str
    issue_lifecycle_errors: tuple[str, ...]
    issue_lifecycles: tuple[DocflowIssueLifecycle, ...]
    rows: tuple[DocflowComplianceRow, ...]
    obligations: tuple[DocflowObligationEntry, ...]

    def __str__(self) -> str:
        return (
            f"{self.source.rel_path} contradicts={self.contradiction_count} "
            f"excess={self.excess_count} unmet_fail={self.unmet_fail_count}"
        )


@dataclass(frozen=True)
class DocflowPacketRow:
    identity: StructuredArtifactIdentity
    row_id: str
    packet_path: str
    status: str

    def __str__(self) -> str:
        return self.row_id


@dataclass(frozen=True)
class DocflowPacket:
    identity: StructuredArtifactIdentity
    packet_path: str
    classification: str
    status: str
    row_ids: tuple[str, ...]
    rows: tuple[DocflowPacketRow, ...]

    def __str__(self) -> str:
        return self.packet_path or self.classification or self.identity.wire()


@dataclass(frozen=True)
class DocflowPacketEnforcementArtifact:
    identity: StructuredArtifactIdentity
    source: StructuredArtifactSource
    active_packets: int
    active_rows: int
    ready: int
    blocked: int
    drifted: int
    new_row_count: int
    drifted_row_count: int
    changed_paths: tuple[str, ...]
    out_of_scope_touches: tuple[str, ...]
    unresolved_touched_packets: tuple[str, ...]
    packets: tuple[DocflowPacket, ...]

    def __str__(self) -> str:
        return (
            f"{self.source.rel_path} blocked={self.blocked} drifted={self.drifted} "
            f"packets={len(self.packets)}"
        )


@dataclass(frozen=True)
class ControllerDriftFinding:
    identity: StructuredArtifactIdentity
    sensor: str
    severity: str
    anchor: str
    detail: str
    doc_paths: tuple[str, ...]

    def __str__(self) -> str:
        return self.detail or self.sensor or self.anchor or self.identity.wire()


@dataclass(frozen=True)
class ControllerDriftArtifact:
    identity: StructuredArtifactIdentity
    source: StructuredArtifactSource
    anchors_scanned: int
    commands_scanned: int
    normative_docs: tuple[str, ...]
    policy: str
    highest_severity: str
    total_findings: int
    findings: tuple[ControllerDriftFinding, ...]

    def __str__(self) -> str:
        return (
            f"{self.source.rel_path} findings={self.total_findings} "
            f"highest_severity={self.highest_severity or 'none'}"
        )


@dataclass(frozen=True)
class LocalReproClosureEntry:
    identity: StructuredArtifactIdentity
    cu_id: str
    summary: str
    validation_statuses: tuple[str, ...]

    def __str__(self) -> str:
        return self.cu_id or self.summary or self.identity.wire()


@dataclass(frozen=True)
class LocalReproClosureLedgerArtifact:
    identity: StructuredArtifactIdentity
    source: StructuredArtifactSource
    workstream: str
    generated_by: str
    entries: tuple[LocalReproClosureEntry, ...]

    def __str__(self) -> str:
        return f"{self.source.rel_path} entries={len(self.entries)}"


@dataclass(frozen=True)
class LocalCiReproCapability:
    identity: StructuredArtifactIdentity
    capability_id: str
    summary: str
    status: str
    source_alternative_token_groups: tuple[tuple[str, ...], ...]
    command_alternative_token_groups: tuple[tuple[str, ...], ...]
    matched_source_alternative_index: int | None
    matched_command_alternative_index: int | None

    def __str__(self) -> str:
        return self.capability_id or self.identity.wire()


@dataclass(frozen=True)
class LocalCiReproSurface:
    identity: StructuredArtifactIdentity
    surface_id: str
    surface_kind: str
    title: str
    summary: str
    source_ref: str
    mode: str
    status: str
    required_capabilities: tuple[LocalCiReproCapability, ...]
    missing_capability_ids: tuple[str, ...]
    required_token_groups: tuple[tuple[str, ...], ...]
    missing_token_groups: tuple[tuple[str, ...], ...]
    commands: tuple[str, ...]
    artifacts: tuple[str, ...]

    def __str__(self) -> str:
        return self.title or self.surface_id or self.identity.wire()


@dataclass(frozen=True)
class LocalCiReproRelation:
    identity: StructuredArtifactIdentity
    relation_id: str
    relation_kind: str
    source_surface_id: str
    target_surface_id: str
    source_missing_capability_ids: tuple[str, ...]
    target_missing_capability_ids: tuple[str, ...]
    status: str
    summary: str

    def __str__(self) -> str:
        return self.relation_id or self.identity.wire()


@dataclass(frozen=True)
class LocalCiReproContractArtifact:
    identity: StructuredArtifactIdentity
    source: StructuredArtifactSource
    summary: str
    surfaces: tuple[LocalCiReproSurface, ...]
    relations: tuple[LocalCiReproRelation, ...]

    def __str__(self) -> str:
        return (
            f"{self.source.rel_path} surfaces={len(self.surfaces)} "
            f"relations={len(self.relations)}"
        )


@dataclass(frozen=True)
class KernelVmEvidenceRef:
    rel_path: str
    evidence_kind: str
    symbol: str
    present: bool

    def __str__(self) -> str:
        return f"{self.rel_path}::{self.symbol}"


@dataclass(frozen=True)
class KernelVmCapability:
    identity: StructuredArtifactIdentity
    capability_id: str
    requirement_kind: str
    status: str
    match_mode: str
    description: str
    residue_kind: str
    severity: str
    score: int
    expected_refs: tuple[KernelVmEvidenceRef, ...]
    matched_refs: tuple[KernelVmEvidenceRef, ...]
    missing_refs: tuple[KernelVmEvidenceRef, ...]

    def __str__(self) -> str:
        return self.capability_id or self.identity.wire()


@dataclass(frozen=True)
class KernelVmBinding:
    identity: StructuredArtifactIdentity
    binding_id: str
    fragment_id: str
    title: str
    status: str
    summary: str
    kernel_terms: tuple[str, ...]
    runtime_surface_symbols: tuple[str, ...]
    realizer_symbols: tuple[str, ...]
    runtime_object_symbols: tuple[str, ...]
    missing_capability_ids: tuple[str, ...]
    residue_ids: tuple[str, ...]
    evidence_paths: tuple[str, ...]
    capabilities: tuple[KernelVmCapability, ...]

    def __str__(self) -> str:
        return self.title or self.binding_id or self.identity.wire()


@dataclass(frozen=True)
class KernelVmResidue:
    identity: StructuredArtifactIdentity
    residue_id: str
    binding_id: str
    fragment_id: str
    residue_kind: str
    severity: str
    score: int
    title: str
    message: str
    missing_capability_ids: tuple[str, ...]
    kernel_terms: tuple[str, ...]
    runtime_surface_symbols: tuple[str, ...]
    realizer_symbols: tuple[str, ...]
    runtime_object_symbols: tuple[str, ...]
    evidence_paths: tuple[str, ...]

    def __str__(self) -> str:
        return self.title or self.residue_id or self.identity.wire()


@dataclass(frozen=True)
class KernelVmAlignmentArtifact:
    identity: StructuredArtifactIdentity
    source: StructuredArtifactSource
    fragment_id: str
    binding_count: int
    pass_count: int
    partial_count: int
    fail_count: int
    residue_count: int
    bindings: tuple[KernelVmBinding, ...]
    residues: tuple[KernelVmResidue, ...]

    def __str__(self) -> str:
        return (
            f"{self.source.rel_path} bindings={self.binding_count} "
            f"residues={self.residue_count}"
        )


@dataclass(frozen=True)
class IdentityGrammarCompletionSurface:
    identity: StructuredArtifactIdentity
    surface_id: str
    title: str
    status: str
    summary: str
    residue_ids: tuple[str, ...]
    evidence_paths: tuple[str, ...]

    def __str__(self) -> str:
        return self.title or self.surface_id or self.identity.wire()


@dataclass(frozen=True)
class IdentityGrammarCompletionResidue:
    identity: StructuredArtifactIdentity
    residue_id: str
    surface_id: str
    residue_kind: str
    severity: str
    score: int
    title: str
    message: str
    evidence_paths: tuple[str, ...]

    def __str__(self) -> str:
        return self.title or self.residue_id or self.identity.wire()


@dataclass(frozen=True)
class IdentityGrammarCompletionArtifact:
    identity: StructuredArtifactIdentity
    source: StructuredArtifactSource
    surface_count: int
    pass_count: int
    fail_count: int
    residue_count: int
    highest_severity: str
    surfaces: tuple[IdentityGrammarCompletionSurface, ...]
    residues: tuple[IdentityGrammarCompletionResidue, ...]

    def __str__(self) -> str:
        return (
            f"{self.source.rel_path} surfaces={self.surface_count} "
            f"residues={self.residue_count}"
        )


@dataclass(frozen=True)
class GitStateLineSpan:
    start_line: int
    line_count: int

    def __str__(self) -> str:
        return f"{self.start_line}+{self.line_count}"


@dataclass(frozen=True)
class GitStateEntry:
    identity: StructuredArtifactIdentity
    state_class: str
    change_code: str
    rel_path: str
    previous_path: str
    current_line_spans: tuple[GitStateLineSpan, ...]

    def __str__(self) -> str:
        return f"{self.state_class}:{self.rel_path}"


@dataclass(frozen=True)
class GitStateArtifact:
    identity: StructuredArtifactIdentity
    source: StructuredArtifactSource
    head_sha: str
    branch: str
    upstream: str
    is_detached: bool
    entries: tuple[GitStateEntry, ...]

    def __str__(self) -> str:
        return (
            f"{self.source.rel_path} committed={sum(1 for item in self.entries if item.state_class == 'committed')} "
            f"staged={sum(1 for item in self.entries if item.state_class == 'staged')} "
            f"unstaged={sum(1 for item in self.entries if item.state_class == 'unstaged')} "
            f"untracked={sum(1 for item in self.entries if item.state_class == 'untracked')}"
        )


@dataclass(frozen=True)
class CrossOriginWitnessFieldCheck:
    field_name: str
    matches: bool
    left_value: str
    right_value: str

    def __str__(self) -> str:
        return self.field_name


@dataclass(frozen=True)
class CrossOriginWitnessRow:
    identity: StructuredArtifactIdentity
    row_key: str
    row_kind: str
    left_origin_kind: str
    left_origin_key: str
    right_origin_kind: str
    right_origin_key: str
    remap_key: str
    summary: str

    def __str__(self) -> str:
        return self.row_key or self.remap_key or self.identity.wire()


@dataclass(frozen=True)
class CrossOriginWitnessContractCase:
    identity: StructuredArtifactIdentity
    case_key: str
    case_kind: str
    title: str
    status: str
    summary: str
    left_label: str
    right_label: str
    evidence_paths: tuple[str, ...]
    row_keys: tuple[str, ...]
    field_checks: tuple[CrossOriginWitnessFieldCheck, ...]

    @property
    def mismatch_count(self) -> int:
        return sum(1 for item in self.field_checks if not item.matches)

    def __str__(self) -> str:
        return self.title


@dataclass(frozen=True)
class CrossOriginWitnessContractArtifact:
    identity: StructuredArtifactIdentity
    source: StructuredArtifactSource
    cases: tuple[CrossOriginWitnessContractCase, ...]
    witness_rows: tuple[CrossOriginWitnessRow, ...]

    def __str__(self) -> str:
        return (
            f"{self.source.rel_path} cases={len(self.cases)} "
            f"witness_rows={len(self.witness_rows)}"
        )


@dataclass(frozen=True)
class IngressMergeParityFieldCheck:
    field_name: str
    matches: bool
    left_value: str
    right_value: str

    def __str__(self) -> str:
        return self.field_name

    def to_payload(self) -> dict[str, object]:
        return {
            "field_name": self.field_name,
            "matches": self.matches,
            "left_value": self.left_value,
            "right_value": self.right_value,
        }


@dataclass(frozen=True)
class IngressMergeParityCase:
    identity: StructuredArtifactIdentity
    case_key: str
    case_kind: str
    title: str
    status: str
    summary: str
    left_label: str
    right_label: str
    evidence_paths: tuple[str, ...]
    field_checks: tuple[IngressMergeParityFieldCheck, ...]

    @property
    def mismatch_count(self) -> int:
        return sum(1 for item in self.field_checks if not item.matches)

    def __str__(self) -> str:
        return self.title

    def to_payload(self) -> dict[str, object]:
        return {
            "case_key": self.case_key,
            "case_kind": self.case_kind,
            "title": self.title,
            "status": self.status,
            "summary": self.summary,
            "left_label": self.left_label,
            "right_label": self.right_label,
            "mismatch_count": self.mismatch_count,
            "evidence_paths": list(self.evidence_paths),
            "field_checks": [item.to_payload() for item in self.field_checks],
            "identity": self.identity.to_payload(),
        }


@dataclass(frozen=True)
class IngressMergeParityArtifact:
    identity: StructuredArtifactIdentity
    source: StructuredArtifactSource
    cases: tuple[IngressMergeParityCase, ...]

    def __str__(self) -> str:
        return f"{self.source.rel_path} cases={len(self.cases)}"

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.source.schema_version,
            "artifact_kind": StructuredArtifactKind.INGRESS_MERGE_PARITY.value,
            "producer": self.source.producer,
            "summary": {
                "case_count": len(self.cases),
                "passing_case_count": sum(1 for item in self.cases if item.status == "pass"),
                "failing_case_count": sum(1 for item in self.cases if item.status != "pass"),
            },
            "identity": self.identity.to_payload(),
            "cases": [item.to_payload() for item in self.cases],
        }


_BACKTICK_LITERAL_RE = re.compile(r"`([^`\n]+)`")
_INGRESS_MERGE_PARITY_SCHEMA_VERSION = 1
_INGRESS_MERGE_PARITY_ARTIFACT = Path("artifacts/out/ingress_merge_parity.json")
_INGRESS_MERGE_PARITY_PRODUCER = (
    "gabion.tooling.policy_substrate.structured_artifact_ingress."
    "build_ingress_merge_parity_artifact"
)


def _normalize_rel_path(root: Path, raw_path: object) -> str:
    text = str(raw_path or "").strip()
    if not text:
        return ""
    path = Path(text)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            return path.as_posix()
    return path.as_posix()


def _load_json_mapping_artifact(path: Path) -> Mapping[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, Mapping) else None


def _markdown_paths_in_detail(detail: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            literal.strip()
            for literal in _BACKTICK_LITERAL_RE.findall(detail)
            if literal.strip().endswith(".md")
        )
    )


def _render_boundary_value(value: object) -> str:
    try:
        rendered = json.dumps(
            value,
            sort_keys=False,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    except TypeError:
        rendered = str(value)
    return rendered if len(rendered) <= 240 else rendered[:237] + "..."


def _field_check(
    *,
    field_name: str,
    left_value: object,
    right_value: object,
) -> IngressMergeParityFieldCheck:
    return IngressMergeParityFieldCheck(
        field_name=field_name,
        matches=left_value == right_value,
        left_value=_render_boundary_value(left_value),
        right_value=_render_boundary_value(right_value),
    )


def _normalized_ingest_primitives(payload: Mapping[str, object]) -> dict[str, object]:
    envelope = _normalize_dataflow_response(cast(Mapping[str, object], payload))
    normalized = envelope.payload
    return {
        "exit_code": normalized["exit_code"],
        "timeout": normalized["timeout"],
        "analysis_state": normalized["analysis_state"],
        "errors": normalized["errors"],
        "lint_entries": normalized["lint_entries"],
        "decision_surfaces": normalized.get("decision_surfaces", []),
        "bundle_sites_by_path": normalized.get("bundle_sites_by_path", {}),
    }


def _first_lint_field(entries: object, field_name: str) -> object:
    if not isinstance(entries, list) or not entries:
        return None
    first = entries[0]
    if not isinstance(first, Mapping):
        return None
    return first.get(field_name)


def _policy_source_documents(root: Path) -> tuple[Path, ...]:
    markdown_rule_root = root / "docs" / "policy_rules"
    markdown_docs = (
        tuple(sorted(markdown_rule_root.glob("*.md"), key=lambda item: item.name))
        if markdown_rule_root.exists()
        else ()
    )
    return (
        root / "docs" / "policy_rules.yaml",
        *markdown_docs,
        root / "docs" / "aspf_opportunity_rules.yaml",
        root / "docs" / "projection_fiber_rules.yaml",
    )


def _artifact_rel_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _adapter_parity_case(
    *,
    root: Path,
    rel_path: str,
    identities: StructuredArtifactIdentitySpace,
) -> IngressMergeParityCase | None:
    fixture_root = root / "tests" / "fixtures" / "ingest_adapter"
    python_raw = _load_json_mapping_artifact(fixture_root / "python_raw.json")
    synthetic_raw = _load_json_mapping_artifact(fixture_root / "synthetic_raw.json")
    python_expected = _load_json_mapping_artifact(fixture_root / "python_expected.json")
    synthetic_expected = _load_json_mapping_artifact(fixture_root / "synthetic_expected.json")
    if (
        python_raw is None
        or synthetic_raw is None
        or python_expected is None
        or synthetic_expected is None
    ):
        return None
    python_normalized = _normalized_ingest_primitives(python_raw)
    synthetic_normalized = _normalized_ingest_primitives(synthetic_raw)
    field_checks = (
        _field_check(
            field_name="python_expected_decision_surfaces",
            left_value=python_normalized.get("decision_surfaces"),
            right_value=python_expected.get("decision_surfaces"),
        ),
        _field_check(
            field_name="synthetic_expected_decision_surfaces",
            left_value=synthetic_normalized.get("decision_surfaces"),
            right_value=synthetic_expected.get("decision_surfaces"),
        ),
        _field_check(
            field_name="exit_code",
            left_value=python_normalized.get("exit_code"),
            right_value=synthetic_normalized.get("exit_code"),
        ),
        _field_check(
            field_name="timeout",
            left_value=python_normalized.get("timeout"),
            right_value=synthetic_normalized.get("timeout"),
        ),
        _field_check(
            field_name="analysis_state",
            left_value=python_normalized.get("analysis_state"),
            right_value=synthetic_normalized.get("analysis_state"),
        ),
        _field_check(
            field_name="decision_surfaces",
            left_value=python_normalized.get("decision_surfaces"),
            right_value=synthetic_normalized.get("decision_surfaces"),
        ),
        _field_check(
            field_name="lint_entries[0].code",
            left_value=_first_lint_field(
                python_normalized.get("lint_entries"),
                "code",
            ),
            right_value=_first_lint_field(
                synthetic_normalized.get("lint_entries"),
                "code",
            ),
        ),
        _field_check(
            field_name="lint_entries[0].message",
            left_value=_first_lint_field(
                python_normalized.get("lint_entries"),
                "message",
            ),
            right_value=_first_lint_field(
                synthetic_normalized.get("lint_entries"),
                "message",
            ),
        ),
    )
    case_key = "adapter_decision_surface_parity"
    case_identity = identities.item_id(
        artifact_kind=StructuredArtifactKind.INGRESS_MERGE_PARITY,
        source_path=rel_path,
        item_kind="case",
        item_key=case_key,
        label="adapter decision surface parity",
    )
    mismatch_count = sum(1 for item in field_checks if not item.matches)
    return IngressMergeParityCase(
        identity=case_identity,
        case_key=case_key,
        case_kind="adapter_normalization_parity",
        title="adapter decision surface parity",
        status="pass" if mismatch_count == 0 else "fail",
        summary=(
            "python and synthetic ingest adapters normalized to the same overlapping "
            f"decision-surface boundary with mismatches={mismatch_count}"
        ),
        left_label="python_ingest",
        right_label="synthetic_ingest",
        evidence_paths=tuple(
            _artifact_rel_path(root, fixture_root / name)
            for name in (
                "python_raw.json",
                "synthetic_raw.json",
                "python_expected.json",
                "synthetic_expected.json",
            )
        ),
        field_checks=field_checks,
    )


def _frontmatter_adapter_projection_case(
    *,
    root: Path,
    rel_path: str,
    identities: StructuredArtifactIdentitySpace,
) -> IngressMergeParityCase:
    valid_text = "\n".join(
        [
            "---",
            "doc_id: sample_policy_rules",
            "doc_revision: 1",
            "---",
            "## sample",
        ]
    )
    invalid_text = "\n".join(
        [
            "---",
            "doc_id: [sample_policy_rules",
            "---",
            "## sample",
        ]
    )
    missing_text = "## sample"
    strict_valid_payload: dict[str, JSONValue] = {}
    strict_valid_body = ""
    strict_valid_status = "ok"
    try:
        strict_valid_payload, strict_valid_body = parse_strict_yaml_frontmatter(
            valid_text,
            require_parser=True,
        )
    except FrontmatterParseError as exc:
        strict_valid_status = str(exc)
    governance_valid_payload, governance_valid_body, governance_valid_mode, _ = (
        governance_audit_impl._parse_frontmatter_with_mode(valid_text)
    )
    strict_invalid_error = "ok"
    try:
        parse_strict_yaml_frontmatter(
            invalid_text,
            require_parser=True,
        )
    except FrontmatterParseError as exc:
        strict_invalid_error = str(exc)
    _, _, governance_invalid_mode, governance_invalid_detail = (
        governance_audit_impl._parse_frontmatter_with_mode(invalid_text)
    )
    _, _, governance_missing_mode, _ = governance_audit_impl._parse_frontmatter_with_mode(
        missing_text
    )
    field_checks = (
        _field_check(
            field_name="strict_valid_status",
            left_value=strict_valid_status,
            right_value="ok",
        ),
        _field_check(
            field_name="valid_payload",
            left_value=strict_valid_payload,
            right_value=governance_valid_payload,
        ),
        _field_check(
            field_name="valid_body",
            left_value=strict_valid_body,
            right_value=governance_valid_body,
        ),
        _field_check(
            field_name="governance_valid_mode",
            left_value=governance_valid_mode,
            right_value="yaml",
        ),
        _field_check(
            field_name="strict_invalid_error",
            left_value=strict_invalid_error,
            right_value="invalid YAML frontmatter",
        ),
        _field_check(
            field_name="governance_invalid_mode",
            left_value=governance_invalid_mode,
            right_value="yaml_parse_failed",
        ),
        _field_check(
            field_name="governance_invalid_detail_present",
            left_value=bool(governance_invalid_detail),
            right_value=True,
        ),
        _field_check(
            field_name="governance_missing_mode",
            left_value=governance_missing_mode,
            right_value="absent",
        ),
    )
    case_key = "frontmatter_adapter_projection_parity"
    case_identity = identities.item_id(
        artifact_kind=StructuredArtifactKind.INGRESS_MERGE_PARITY,
        source_path=rel_path,
        item_kind="case",
        item_key=case_key,
        label="frontmatter adapter projection parity",
    )
    mismatch_count = sum(1 for item in field_checks if not item.matches)
    return IngressMergeParityCase(
        identity=case_identity,
        case_key=case_key,
        case_kind="frontmatter_adapter_projection_parity",
        title="frontmatter adapter projection parity",
        status="pass" if mismatch_count == 0 else "fail",
        summary=(
            "strict and governance frontmatter projections stayed aligned across "
            f"valid/invalid/absent markdown carriers with mismatches={mismatch_count}"
        ),
        left_label="strict_frontmatter_boundary",
        right_label="governance_frontmatter_boundary",
        evidence_paths=tuple(
            _artifact_rel_path(root, path)
            for path in (
                root / "src/gabion/frontmatter.py",
                root / "src/gabion/frontmatter_ingress.py",
                root / "src/gabion_governance/governance_audit_impl.py",
            )
            if path.exists()
        ),
        field_checks=field_checks,
    )


def _policy_source_uniqueness_case(
    *,
    root: Path,
    rel_path: str,
    identities: StructuredArtifactIdentitySpace,
) -> IngressMergeParityCase:
    source_paths = tuple(
        path for path in _policy_source_documents(root) if path.exists()
    )
    issues: list[object] = []
    rule_sources: dict[str, set[str]] = {}
    transform_sources: dict[str, set[str]] = {}
    for path in source_paths:
        program, source_issues = compile_document(path)
        issues.extend(source_issues)
        if program is None:
            continue
        source_id = _artifact_rel_path(root, path)
        for rule in program.rules:
            rule_sources.setdefault(rule.rule_id, set()).add(source_id)
        for transform in program.transforms:
            transform_sources.setdefault(transform.transform_id, set()).add(source_id)
    duplicate_rule_ids = tuple(
        sorted(rule_id for rule_id, sources in rule_sources.items() if len(sources) > 1)
    )
    duplicate_transform_ids = tuple(
        sorted(
            transform_id
            for transform_id, sources in transform_sources.items()
            if len(sources) > 1
        )
    )
    issue_rows = tuple(
        sorted(
            (
                "{code}:{rule_id}:{message}".format(
                    code=str(getattr(issue, "code", "")).strip(),
                    rule_id=str(getattr(issue, "rule_id", "")).strip(),
                    message=str(getattr(issue, "message", "")).strip(),
                )
            )
            for issue in issues
        )
    )
    field_checks = (
        _field_check(
            field_name="compile_issue_count",
            left_value=len(issue_rows),
            right_value=0,
        ),
        _field_check(
            field_name="duplicate_rule_id_source_count",
            left_value=len(duplicate_rule_ids),
            right_value=0,
        ),
        _field_check(
            field_name="duplicate_transform_id_source_count",
            left_value=len(duplicate_transform_ids),
            right_value=0,
        ),
    )
    case_key = "policy_source_uniqueness"
    case_identity = identities.item_id(
        artifact_kind=StructuredArtifactKind.INGRESS_MERGE_PARITY,
        source_path=rel_path,
        item_kind="case",
        item_key=case_key,
        label="policy source uniqueness",
    )
    mismatch_count = sum(1 for item in field_checks if not item.matches)
    return IngressMergeParityCase(
        identity=case_identity,
        case_key=case_key,
        case_kind="policy_source_uniqueness",
        title="policy source uniqueness",
        status="pass" if mismatch_count == 0 else "fail",
        summary=(
            "yaml/markdown policy ingress produced compile_issues={issues} "
            "duplicate_rules={rules} duplicate_transforms={transforms}".format(
                issues=len(issue_rows),
                rules=len(duplicate_rule_ids),
                transforms=len(duplicate_transform_ids),
            )
        ),
        left_label="observed",
        right_label="expected",
        evidence_paths=tuple(_artifact_rel_path(root, path) for path in source_paths),
        field_checks=field_checks,
    )


def _policy_registry_determinism_case(
    *,
    root: Path,
    rel_path: str,
    identities: StructuredArtifactIdentitySpace,
) -> IngressMergeParityCase:
    build_error = ""
    first_rule_ids: tuple[str, ...] = ()
    second_rule_ids: tuple[str, ...] = ()
    first_transform_ids: tuple[str, ...] = ()
    second_transform_ids: tuple[str, ...] = ()
    try:
        first = _build_registry_for_root(root).program
        second = _build_registry_for_root(root).program
        first_rule_ids = tuple(rule.rule_id for rule in first.rules)
        second_rule_ids = tuple(rule.rule_id for rule in second.rules)
        first_transform_ids = tuple(
            transform.transform_id for transform in first.transforms
        )
        second_transform_ids = tuple(
            transform.transform_id for transform in second.transforms
        )
    except ValueError as exc:
        build_error = str(exc)
    field_checks = (
        _field_check(
            field_name="registry_build_status",
            left_value=build_error or "ok",
            right_value="ok",
        ),
        _field_check(
            field_name="rule_order_stable",
            left_value=first_rule_ids,
            right_value=second_rule_ids,
        ),
        _field_check(
            field_name="transform_order_stable",
            left_value=first_transform_ids,
            right_value=second_transform_ids,
        ),
    )
    case_key = "policy_registry_determinism"
    case_identity = identities.item_id(
        artifact_kind=StructuredArtifactKind.INGRESS_MERGE_PARITY,
        source_path=rel_path,
        item_kind="case",
        item_key=case_key,
        label="policy registry determinism",
    )
    mismatch_count = sum(1 for item in field_checks if not item.matches)
    return IngressMergeParityCase(
        identity=case_identity,
        case_key=case_key,
        case_kind="policy_registry_determinism",
        title="policy registry determinism",
        status="pass" if mismatch_count == 0 else "fail",
        summary=(
            "registry rebuild determinism rules={rule_count} transforms={transform_count} "
            "mismatches={mismatches}".format(
                rule_count=len(first_rule_ids),
                transform_count=len(first_transform_ids),
                mismatches=mismatch_count,
            )
        ),
        left_label="first_build",
        right_label="second_build",
        evidence_paths=tuple(
            _artifact_rel_path(root, path)
            for path in _policy_source_documents(root)
            if path.exists()
        ),
        field_checks=field_checks,
    )


def build_ingress_merge_parity_artifact(
    *,
    root: Path,
    rel_path: str = _INGRESS_MERGE_PARITY_ARTIFACT.as_posix(),
    identities: StructuredArtifactIdentitySpace | None = None,
) -> IngressMergeParityArtifact:
    identity_space = identities or StructuredArtifactIdentitySpace()
    source = StructuredArtifactSource(
        rel_path=rel_path,
        schema_version=_INGRESS_MERGE_PARITY_SCHEMA_VERSION,
        producer=_INGRESS_MERGE_PARITY_PRODUCER,
    )
    identity = identity_space.artifact_id(
        artifact_kind=StructuredArtifactKind.INGRESS_MERGE_PARITY,
        source_path=rel_path,
        label=rel_path,
    )
    cases = tuple(
        sorted(
            (
                item
                for item in (
                    _adapter_parity_case(
                        root=root,
                        rel_path=rel_path,
                        identities=identity_space,
                    ),
                    _frontmatter_adapter_projection_case(
                        root=root,
                        rel_path=rel_path,
                        identities=identity_space,
                    ),
                    _policy_source_uniqueness_case(
                        root=root,
                        rel_path=rel_path,
                        identities=identity_space,
                    ),
                    _policy_registry_determinism_case(
                        root=root,
                        rel_path=rel_path,
                        identities=identity_space,
                    ),
                )
                if item is not None
            ),
            key=lambda item: item.case_key,
        )
    )
    return IngressMergeParityArtifact(
        identity=identity,
        source=source,
        cases=cases,
    )


def write_ingress_merge_parity_artifact(
    *,
    root: Path,
    rel_path: str,
    artifact: IngressMergeParityArtifact,
) -> Path:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact.to_payload(), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return path


def load_ingress_merge_parity_artifact(
    *,
    root: Path,
    rel_path: str,
    identities: StructuredArtifactIdentitySpace,
) -> IngressMergeParityArtifact | None:
    payload = _load_json_mapping_artifact(root / rel_path)
    if payload is None:
        return None
    if payload.get("artifact_kind") != StructuredArtifactKind.INGRESS_MERGE_PARITY.value:
        return None
    source = StructuredArtifactSource(
        rel_path=rel_path,
        schema_version=int(payload.get("schema_version", 0) or 0),
        producer=str(payload.get("producer", "")).strip(),
    )
    identity = identities.artifact_id(
        artifact_kind=StructuredArtifactKind.INGRESS_MERGE_PARITY,
        source_path=rel_path,
        label=rel_path,
    )
    cases: list[IngressMergeParityCase] = []
    raw_cases = payload.get("cases", [])
    if isinstance(raw_cases, list):
        for raw_case in raw_cases:
            if not isinstance(raw_case, Mapping):
                continue
            case_key = str(raw_case.get("case_key", "")).strip()
            if not case_key:
                continue
            field_checks: list[IngressMergeParityFieldCheck] = []
            raw_field_checks = raw_case.get("field_checks", [])
            if isinstance(raw_field_checks, list):
                for raw_field_check in raw_field_checks:
                    if not isinstance(raw_field_check, Mapping):
                        continue
                    field_name = str(raw_field_check.get("field_name", "")).strip()
                    if not field_name:
                        continue
                    field_checks.append(
                        IngressMergeParityFieldCheck(
                            field_name=field_name,
                            matches=bool(raw_field_check.get("matches", False)),
                            left_value=str(raw_field_check.get("left_value", "")),
                            right_value=str(raw_field_check.get("right_value", "")),
                        )
                    )
            cases.append(
                IngressMergeParityCase(
                    identity=identities.item_id(
                        artifact_kind=StructuredArtifactKind.INGRESS_MERGE_PARITY,
                        source_path=rel_path,
                        item_kind="case",
                        item_key=case_key,
                        label=str(raw_case.get("title", "")).strip() or case_key,
                    ),
                    case_key=case_key,
                    case_kind=str(raw_case.get("case_kind", "")).strip(),
                    title=str(raw_case.get("title", "")).strip() or case_key,
                    status=str(raw_case.get("status", "")).strip(),
                    summary=str(raw_case.get("summary", "")).strip(),
                    left_label=str(raw_case.get("left_label", "")).strip(),
                    right_label=str(raw_case.get("right_label", "")).strip(),
                    evidence_paths=tuple(
                        str(value).strip()
                        for value in raw_case.get("evidence_paths", [])
                        if str(value).strip()
                    ),
                    field_checks=tuple(field_checks),
                )
            )
    return IngressMergeParityArtifact(
        identity=identity,
        source=source,
        cases=tuple(cases),
    )


def load_test_evidence_artifact(
    *,
    root: Path,
    rel_path: str,
    identities: StructuredArtifactIdentitySpace,
) -> TestEvidenceArtifact | None:
    payload = _load_json_mapping_artifact(root / rel_path)
    if payload is None:
        return None
    tests = payload.get("tests", [])
    if not isinstance(tests, list):
        return None
    source = StructuredArtifactSource(rel_path=rel_path, schema_version=2)
    identity = identities.artifact_id(
        artifact_kind=StructuredArtifactKind.TEST_EVIDENCE,
        source_path=rel_path,
        label=rel_path,
    )
    cases: list[TestEvidenceCase] = []
    for index, raw_test in enumerate(tests, start=1):
        if not isinstance(raw_test, Mapping):
            continue
        test_id = str(raw_test.get("test_id", "")).strip()
        if not test_id:
            continue
        evidence = raw_test.get("evidence", [])
        if not isinstance(evidence, list):
            continue
        evidence_sites: list[TestEvidenceSite] = []
        for raw_item in evidence:
            if not isinstance(raw_item, Mapping):
                continue
            key = raw_item.get("key")
            if not isinstance(key, Mapping):
                continue
            site = key.get("site")
            if not isinstance(site, Mapping):
                continue
            span_values = site.get("span")
            span = (
                tuple(int(value) for value in span_values)
                if isinstance(span_values, list)
                and len(span_values) == 4
                and all(isinstance(value, int) for value in span_values)
                else ()
            )
            evidence_sites.append(
                TestEvidenceSite(
                    path=_normalize_rel_path(root, site.get("path")),
                    qualname=str(site.get("qual", "")).strip(),
                    span=cast(tuple[int, int, int, int], span),
                )
            )
        case_identity = identities.item_id(
            artifact_kind=StructuredArtifactKind.TEST_EVIDENCE,
            source_path=rel_path,
            item_kind="test_case",
            item_key=test_id,
            label=test_id,
        )
        cases.append(
            TestEvidenceCase(
                identity=case_identity,
                test_id=test_id,
                rel_path=_normalize_rel_path(root, raw_test.get("file")),
                line=int(raw_test.get("line", 0) or 0),
                status=str(raw_test.get("status", "")).strip(),
                evidence_sites=tuple(evidence_sites),
            )
        )
    return TestEvidenceArtifact(identity=identity, source=source, cases=tuple(cases))


def load_junit_failure_artifact(
    *,
    root: Path,
    rel_path: str,
    identities: StructuredArtifactIdentitySpace,
) -> JUnitFailureArtifact | None:
    path = root / rel_path
    if not path.exists():
        return None
    try:
        tree = ElementTree.parse(path)
    except ElementTree.ParseError:
        return None
    source = StructuredArtifactSource(rel_path=rel_path)
    identity = identities.artifact_id(
        artifact_kind=StructuredArtifactKind.JUNIT_FAILURES,
        source_path=rel_path,
        label=rel_path,
    )
    failures: list[JUnitFailureCase] = []
    for index, testcase in enumerate(tree.iterfind(".//testcase"), start=1):
        failure_like = next(
            (child for child in testcase if child.tag in {"failure", "error"}),
            None,
        )
        if failure_like is None:
            continue
        raw_name = str(testcase.attrib.get("name", "")).strip()
        if not raw_name:
            continue
        rel_test_path = _normalize_rel_path(root, testcase.attrib.get("file"))
        line = int(testcase.attrib.get("line", 0) or 0)
        classname = str(testcase.attrib.get("classname", "")).strip()
        test_id = raw_name
        if rel_test_path:
            class_suffix = classname.split(".")[-1].strip() if classname else ""
            if class_suffix and class_suffix != raw_name:
                test_id = f"{rel_test_path}::{class_suffix}::{raw_name}"
            else:
                test_id = f"{rel_test_path}::{raw_name}"
        message = str(failure_like.attrib.get("message", "")).strip()
        traceback_text = (failure_like.text or "").strip()
        title = message or raw_name
        failure_identity = identities.item_id(
            artifact_kind=StructuredArtifactKind.JUNIT_FAILURES,
            source_path=rel_path,
            item_kind="failure",
            item_key=f"{test_id}:{index}",
            label=title,
        )
        failures.append(
            JUnitFailureCase(
                identity=failure_identity,
                test_id=test_id,
                raw_name=raw_name,
                classname=classname,
                rel_path=rel_test_path,
                line=line,
                failure_kind=failure_like.tag,
                title=title,
                message=message,
                traceback_text=traceback_text,
            )
        )
    return JUnitFailureArtifact(
        identity=identity,
        source=source,
        failures=tuple(failures),
    )


def load_docflow_compliance_artifact(
    *,
    root: Path,
    rel_path: str,
    identities: StructuredArtifactIdentitySpace,
) -> DocflowComplianceArtifact | None:
    payload = _load_json_mapping_artifact(root / rel_path)
    if payload is None:
        return None
    summary = payload.get("summary", {})
    summary_mapping = summary if isinstance(summary, Mapping) else {}
    obligations_payload = payload.get("obligations", {})
    obligations_mapping = (
        obligations_payload if isinstance(obligations_payload, Mapping) else {}
    )
    obligation_summary = obligations_mapping.get("summary", {})
    obligation_summary_mapping = (
        obligation_summary if isinstance(obligation_summary, Mapping) else {}
    )
    obligation_context = obligations_mapping.get("context", {})
    obligation_context_mapping = (
        obligation_context if isinstance(obligation_context, Mapping) else {}
    )
    source = StructuredArtifactSource(rel_path=rel_path)
    identity = identities.artifact_id(
        artifact_kind=StructuredArtifactKind.DOCFLOW_COMPLIANCE,
        source_path=rel_path,
        label=rel_path,
    )
    rev_range = str(obligation_context_mapping.get("rev_range", "")).strip()
    commits: list[DocflowCommit] = []
    raw_commits = obligation_context_mapping.get("commits", [])
    if isinstance(raw_commits, list):
        for raw_commit in raw_commits:
            if not isinstance(raw_commit, Mapping):
                continue
            sha = str(raw_commit.get("sha", "")).strip()
            if not sha:
                continue
            commit_identity = identities.item_id(
                artifact_kind=StructuredArtifactKind.DOCFLOW_COMPLIANCE,
                source_path=rel_path,
                item_kind="commit",
                item_key=sha,
                label=sha[:12],
            )
            commits.append(
                DocflowCommit(
                    identity=commit_identity,
                    sha=sha,
                    subject=str(raw_commit.get("subject", "")).strip(),
                )
            )
    issue_ids = tuple(
        sorted(
            [
                str(value).strip()
                for value in obligation_context_mapping.get("issue_ids", [])
                if str(value).strip()
            ]
        )
    )
    checklist_impact_by_issue_id: dict[str, int] = {}
    raw_checklist_impact = obligation_context_mapping.get("checklist_impact", [])
    if isinstance(raw_checklist_impact, list):
        for raw_entry in raw_checklist_impact:
            if isinstance(raw_entry, Mapping):
                issue_id = str(raw_entry.get("issue_id", "")).strip()
                if not issue_id:
                    continue
                checklist_impact_by_issue_id[issue_id] = int(
                    raw_entry.get("commit_count", 0) or 0
                )
                continue
            if (
                isinstance(raw_entry, Sequence)
                and not isinstance(raw_entry, str)
                and len(raw_entry) == 2
            ):
                issue_id = str(raw_entry[0]).strip()
                if not issue_id:
                    continue
                checklist_impact_by_issue_id[issue_id] = int(raw_entry[1] or 0)
    issue_references: list[DocflowIssueReference] = []
    for issue_id in sorted(set(issue_ids) | set(checklist_impact_by_issue_id.keys())):
        issue_identity = identities.item_id(
            artifact_kind=StructuredArtifactKind.DOCFLOW_COMPLIANCE,
            source_path=rel_path,
            item_kind="issue_reference",
            item_key=issue_id,
            label=f"GH-{issue_id}",
        )
        issue_references.append(
            DocflowIssueReference(
                identity=issue_identity,
                issue_id=issue_id,
                commit_count=checklist_impact_by_issue_id.get(issue_id, 0),
            )
        )
    issue_lifecycles: list[DocflowIssueLifecycle] = []
    raw_issue_lifecycles = obligation_context_mapping.get("issue_lifecycles", [])
    if isinstance(raw_issue_lifecycles, list):
        for raw_lifecycle in raw_issue_lifecycles:
            if not isinstance(raw_lifecycle, Mapping):
                continue
            issue_id = str(raw_lifecycle.get("issue_id", "")).strip()
            if not issue_id:
                continue
            lifecycle_identity = identities.item_id(
                artifact_kind=StructuredArtifactKind.DOCFLOW_COMPLIANCE,
                source_path=rel_path,
                item_kind="issue_lifecycle",
                item_key=issue_id,
                label=f"GH-{issue_id}",
            )
            labels = raw_lifecycle.get("labels", [])
            labels_values = (
                tuple(
                    sorted(
                        str(value).strip()
                        for value in labels
                        if str(value).strip()
                    )
                )
                if isinstance(labels, list)
                else ()
            )
            issue_lifecycles.append(
                DocflowIssueLifecycle(
                    identity=lifecycle_identity,
                    issue_id=issue_id,
                    state=str(raw_lifecycle.get("state", "")).strip(),
                    labels=labels_values,
                    url=str(raw_lifecycle.get("url", "")).strip(),
                )
            )
    rows: list[DocflowComplianceRow] = []
    raw_rows = payload.get("rows", [])
    if isinstance(raw_rows, list):
        for raw_row in raw_rows:
            if not isinstance(raw_row, Mapping):
                continue
            row_id = stable_docflow_compliance_row_id(raw_row)
            row_identity = identities.item_id(
                artifact_kind=StructuredArtifactKind.DOCFLOW_COMPLIANCE,
                source_path=rel_path,
                item_kind="row",
                item_key=row_id,
                label=row_id,
            )
            rows.append(
                DocflowComplianceRow(
                    identity=row_identity,
                    row_id=row_id,
                    status=str(raw_row.get("status", "")).strip(),
                    invariant=str(raw_row.get("invariant", "")).strip(),
                    invariant_kind=str(raw_row.get("invariant_kind", "")).strip(),
                    rel_path=_normalize_rel_path(root, raw_row.get("path")),
                    source_row_kind=str(raw_row.get("source_row_kind", "")).strip(),
                    detail=str(raw_row.get("detail", "")).strip(),
                    evidence_id=str(raw_row.get("evidence_id", "")).strip(),
                )
            )
    obligations: list[DocflowObligationEntry] = []
    raw_entries = obligations_mapping.get("entries", [])
    if isinstance(raw_entries, list):
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, Mapping):
                continue
            obligation_id = str(raw_entry.get("obligation_id", "")).strip()
            if not obligation_id:
                continue
            obligation_identity = identities.item_id(
                artifact_kind=StructuredArtifactKind.DOCFLOW_COMPLIANCE,
                source_path=rel_path,
                item_kind="obligation",
                item_key=obligation_id,
                label=obligation_id,
            )
            obligations.append(
                DocflowObligationEntry(
                    identity=obligation_identity,
                    obligation_id=obligation_id,
                    triggered=raw_entry.get("triggered") is True,
                    status=str(raw_entry.get("status", "")).strip(),
                    enforcement=str(raw_entry.get("enforcement", "")).strip(),
                    description=str(raw_entry.get("description", "")).strip(),
                )
            )
    return DocflowComplianceArtifact(
        identity=identity,
        source=source,
        compliant_count=int(summary_mapping.get("compliant", 0) or 0),
        contradiction_count=int(summary_mapping.get("contradicts", 0) or 0),
        excess_count=int(summary_mapping.get("excess", 0) or 0),
        proposed_count=int(summary_mapping.get("proposed", 0) or 0),
        rev_range=rev_range,
        changed_paths=tuple(
            _normalize_rel_path(root, value)
            for value in obligation_context_mapping.get("changed_paths", [])
        ),
        sppf_relevant_paths_changed=(
            obligation_context_mapping.get("sppf_relevant_paths_changed") is True
        ),
        gh_reference_validated=(
            obligation_context_mapping.get("gh_reference_validated") is True
        ),
        baseline_write_emitted=(
            obligation_context_mapping.get("baseline_write_emitted") is True
        ),
        delta_guard_checked=(
            obligation_context_mapping.get("delta_guard_checked") is True
        ),
        doc_status_changed=(
            obligation_context_mapping.get("doc_status_changed") is True
        ),
        checklist_influence_consistent=(
            obligation_context_mapping.get("checklist_influence_consistent") is True
        ),
        unmet_fail_count=int(obligation_summary_mapping.get("unmet_fail", 0) or 0),
        unmet_warn_count=int(obligation_summary_mapping.get("unmet_warn", 0) or 0),
        commits=tuple(commits),
        issue_references=tuple(issue_references),
        issue_lifecycle_fetch_status=str(
            obligation_context_mapping.get("issue_lifecycle_fetch_status", "")
        ).strip(),
        issue_lifecycle_errors=tuple(
            str(value).strip()
            for value in obligation_context_mapping.get("issue_lifecycle_errors", [])
            if str(value).strip()
        ),
        issue_lifecycles=tuple(issue_lifecycles),
        rows=tuple(rows),
        obligations=tuple(obligations),
    )


def load_docflow_packet_enforcement_artifact(
    *,
    root: Path,
    rel_path: str,
    identities: StructuredArtifactIdentitySpace,
) -> DocflowPacketEnforcementArtifact | None:
    payload = _load_json_mapping_artifact(root / rel_path)
    if payload is None:
        return None
    summary = payload.get("summary", {})
    summary_mapping = summary if isinstance(summary, Mapping) else {}
    new_rows = payload.get("new_rows", [])
    drifted_rows = payload.get("drifted_rows", [])
    new_row_ids = {
        str(item.get("row_id", "")).strip()
        for item in new_rows
        if isinstance(item, Mapping) and str(item.get("row_id", "")).strip()
    }
    drifted_row_ids = {
        str(item).strip()
        for item in drifted_rows
        if str(item).strip()
    }
    source = StructuredArtifactSource(rel_path=rel_path)
    identity = identities.artifact_id(
        artifact_kind=StructuredArtifactKind.DOCFLOW_PACKET_ENFORCEMENT,
        source_path=rel_path,
        label=rel_path,
    )
    packets: list[DocflowPacket] = []
    packet_status_items = payload.get("packet_status", [])
    if isinstance(packet_status_items, list):
        for raw_packet in packet_status_items:
            if not isinstance(raw_packet, Mapping):
                continue
            packet_path = _normalize_rel_path(root, raw_packet.get("path"))
            classification = str(raw_packet.get("classification", "")).strip()
            packet_status = str(raw_packet.get("status", "")).strip()
            row_ids = tuple(
                sorted(
                    str(value).strip()
                    for value in raw_packet.get("row_ids", [])
                    if str(value).strip()
                )
            )
            packet_identity = identities.item_id(
                artifact_kind=StructuredArtifactKind.DOCFLOW_PACKET_ENFORCEMENT,
                source_path=rel_path,
                item_kind="packet",
                item_key=f"{packet_path}:{classification}",
                label=packet_path or classification or "docflow packet",
            )
            rows: list[DocflowPacketRow] = []
            for row_id in row_ids:
                row_status = (
                    "new"
                    if row_id in new_row_ids
                    else "drifted"
                    if row_id in drifted_row_ids
                    else packet_status
                )
                row_identity = identities.item_id(
                    artifact_kind=StructuredArtifactKind.DOCFLOW_PACKET_ENFORCEMENT,
                    source_path=rel_path,
                    item_kind="row",
                    item_key=f"{packet_path}:{row_id}",
                    label=row_id,
                )
                rows.append(
                    DocflowPacketRow(
                        identity=row_identity,
                        row_id=row_id,
                        packet_path=packet_path,
                        status=row_status,
                    )
                )
            packets.append(
                DocflowPacket(
                    identity=packet_identity,
                    packet_path=packet_path,
                    classification=classification,
                    status=packet_status,
                    row_ids=row_ids,
                    rows=tuple(rows),
                )
            )
    return DocflowPacketEnforcementArtifact(
        identity=identity,
        source=source,
        active_packets=int(summary_mapping.get("active_packets", 0) or 0),
        active_rows=int(summary_mapping.get("active_rows", 0) or 0),
        ready=int(summary_mapping.get("ready", 0) or 0),
        blocked=int(summary_mapping.get("blocked", 0) or 0),
        drifted=int(summary_mapping.get("drifted", 0) or 0),
        new_row_count=int(summary_mapping.get("new_rows", 0) or 0),
        drifted_row_count=int(summary_mapping.get("drifted_rows", 0) or 0),
        changed_paths=tuple(
            _normalize_rel_path(root, value) for value in payload.get("changed_paths", [])
        ),
        out_of_scope_touches=tuple(
            _normalize_rel_path(root, value)
            for value in payload.get("out_of_scope_touches", [])
        ),
        unresolved_touched_packets=tuple(
            _normalize_rel_path(root, value)
            for value in payload.get("unresolved_touched_packets", [])
        ),
        packets=tuple(packets),
    )


def load_controller_drift_artifact(
    *,
    root: Path,
    rel_path: str,
    identities: StructuredArtifactIdentitySpace,
) -> ControllerDriftArtifact | None:
    payload = _load_json_mapping_artifact(root / rel_path)
    if payload is None:
        return None
    summary = payload.get("summary", {})
    summary_mapping = summary if isinstance(summary, Mapping) else {}
    source = StructuredArtifactSource(
        rel_path=rel_path,
        producer=str(payload.get("policy", "")).strip(),
    )
    identity = identities.artifact_id(
        artifact_kind=StructuredArtifactKind.CONTROLLER_DRIFT,
        source_path=rel_path,
        label=rel_path,
    )
    findings: list[ControllerDriftFinding] = []
    raw_findings = payload.get("findings", [])
    if isinstance(raw_findings, list):
        for index, raw_finding in enumerate(raw_findings, start=1):
            if not isinstance(raw_finding, Mapping):
                continue
            sensor = str(raw_finding.get("sensor", "")).strip()
            severity = str(raw_finding.get("severity", "")).strip()
            anchor = str(raw_finding.get("anchor", "")).strip()
            detail = str(raw_finding.get("detail", "")).strip()
            finding_identity = identities.item_id(
                artifact_kind=StructuredArtifactKind.CONTROLLER_DRIFT,
                source_path=rel_path,
                item_kind="finding",
                item_key=anchor or f"{sensor}:{index}",
                label=detail or sensor or anchor or f"finding-{index}",
            )
            findings.append(
                ControllerDriftFinding(
                    identity=finding_identity,
                    sensor=sensor,
                    severity=severity,
                    anchor=anchor,
                    detail=detail,
                    doc_paths=tuple(
                        _normalize_rel_path(root, value)
                        for value in _markdown_paths_in_detail(detail)
                    ),
                )
            )
    return ControllerDriftArtifact(
        identity=identity,
        source=source,
        anchors_scanned=int(payload.get("anchors_scanned", 0) or 0),
        commands_scanned=int(payload.get("commands_scanned", 0) or 0),
        normative_docs=tuple(
            _normalize_rel_path(root, value) for value in payload.get("normative_docs", [])
        ),
        policy=str(payload.get("policy", "")).strip(),
        highest_severity=str(summary_mapping.get("highest_severity", "")).strip(),
        total_findings=int(summary_mapping.get("total_findings", 0) or 0),
        findings=tuple(findings),
    )


def load_git_state_artifact(
    *,
    root: Path,
    rel_path: str,
    identities: StructuredArtifactIdentitySpace,
    prefer_live_repo_state: bool = False,
) -> GitStateArtifact | None:
    payload: Mapping[str, object] | None = None
    if prefer_live_repo_state:
        payload = _build_live_git_state_payload(root)
    if payload is None:
        payload = _load_json_mapping_artifact(root / rel_path)
    return _git_state_artifact_from_payload(
        payload=payload,
        rel_path=rel_path,
        identities=identities,
    )


def _build_live_git_state_payload(root: Path) -> Mapping[str, object] | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return None
    if completed.stdout.strip() != "true":
        return None
    from gabion.tooling.runtime.git_state_artifact import (
        build_git_state_artifact_payload,
    )

    return cast(Mapping[str, object], build_git_state_artifact_payload(root=root))


def _git_state_artifact_from_payload(
    *,
    payload: Mapping[str, object] | None,
    rel_path: str,
    identities: StructuredArtifactIdentitySpace,
) -> GitStateArtifact | None:
    if payload is None:
        return None
    if payload.get("artifact_kind") != StructuredArtifactKind.GIT_STATE.value:
        return None
    source = StructuredArtifactSource(
        rel_path=rel_path,
        schema_version=int(payload.get("schema_version", 0) or 0),
    )
    identity = identities.artifact_id(
        artifact_kind=StructuredArtifactKind.GIT_STATE,
        source_path=rel_path,
        label=rel_path,
    )
    entries: list[GitStateEntry] = []
    raw_entries = payload.get("entries", [])
    if isinstance(raw_entries, list):
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, Mapping):
                continue
            state_class = str(raw_entry.get("state_class", "")).strip()
            rel_entry_path = str(raw_entry.get("path", "")).strip()
            if not state_class or not rel_entry_path:
                continue
            change_code = str(raw_entry.get("change_code", "")).strip()
            previous_path = str(raw_entry.get("previous_path", "")).strip()
            current_line_spans: list[GitStateLineSpan] = []
            raw_line_spans = raw_entry.get("current_line_spans", [])
            if isinstance(raw_line_spans, list):
                for raw_line_span in raw_line_spans:
                    if not isinstance(raw_line_span, Mapping):
                        continue
                    start_line = int(raw_line_span.get("start_line", 0) or 0)
                    line_count = int(raw_line_span.get("line_count", 0) or 0)
                    if start_line <= 0 or line_count <= 0:
                        continue
                    current_line_spans.append(
                        GitStateLineSpan(
                            start_line=start_line,
                            line_count=line_count,
                        )
                    )
            entries.append(
                GitStateEntry(
                    identity=identities.item_id(
                        artifact_kind=StructuredArtifactKind.GIT_STATE,
                        source_path=rel_path,
                        item_kind=state_class,
                        item_key=rel_entry_path,
                        label=f"{state_class}:{rel_entry_path}",
                    ),
                    state_class=state_class,
                    change_code=change_code,
                    rel_path=rel_entry_path,
                    previous_path=previous_path,
                    current_line_spans=tuple(current_line_spans),
                )
            )
    return GitStateArtifact(
        identity=identity,
        source=source,
        head_sha=str(payload.get("head_sha", "")).strip(),
        branch=str(payload.get("branch", "")).strip(),
        upstream=str(payload.get("upstream", "")).strip(),
        is_detached=bool(payload.get("is_detached", False)),
        entries=tuple(entries),
    )


def load_cross_origin_witness_contract_artifact(
    *,
    root: Path,
    rel_path: str,
    identities: StructuredArtifactIdentitySpace,
) -> CrossOriginWitnessContractArtifact | None:
    payload = _load_json_mapping_artifact(root / rel_path)
    if payload is None:
        return None
    if payload.get("artifact_kind") != StructuredArtifactKind.CROSS_ORIGIN_WITNESS_CONTRACT.value:
        return None
    source = StructuredArtifactSource(
        rel_path=rel_path,
        schema_version=int(payload.get("schema_version", 0) or 0),
        producer=str(payload.get("producer", "")).strip(),
    )
    identity = identities.artifact_id(
        artifact_kind=StructuredArtifactKind.CROSS_ORIGIN_WITNESS_CONTRACT,
        source_path=rel_path,
        label=rel_path,
    )
    rows: list[CrossOriginWitnessRow] = []
    raw_rows = payload.get("witness_rows", [])
    if isinstance(raw_rows, list):
        for index, raw_row in enumerate(raw_rows, start=1):
            if not isinstance(raw_row, Mapping):
                continue
            row_key = str(raw_row.get("row_key", "")).strip()
            row_kind = str(raw_row.get("row_kind", "")).strip()
            remap_key = str(raw_row.get("remap_key", "")).strip()
            if not row_key or not row_kind:
                continue
            rows.append(
                CrossOriginWitnessRow(
                    identity=identities.item_id(
                        artifact_kind=StructuredArtifactKind.CROSS_ORIGIN_WITNESS_CONTRACT,
                        source_path=rel_path,
                        item_kind=row_kind,
                        item_key=row_key or f"row-{index}",
                        label=row_key or remap_key or f"row-{index}",
                    ),
                    row_key=row_key,
                    row_kind=row_kind,
                    left_origin_kind=str(raw_row.get("left_origin_kind", "")).strip(),
                    left_origin_key=str(raw_row.get("left_origin_key", "")).strip(),
                    right_origin_kind=str(raw_row.get("right_origin_kind", "")).strip(),
                    right_origin_key=str(raw_row.get("right_origin_key", "")).strip(),
                    remap_key=remap_key,
                    summary=str(raw_row.get("summary", "")).strip(),
                )
            )
    cases: list[CrossOriginWitnessContractCase] = []
    raw_cases = payload.get("cases", [])
    if isinstance(raw_cases, list):
        for index, raw_case in enumerate(raw_cases, start=1):
            if not isinstance(raw_case, Mapping):
                continue
            case_key = str(raw_case.get("case_key", "")).strip()
            title = str(raw_case.get("title", "")).strip()
            if not case_key:
                continue
            raw_field_checks = raw_case.get("field_checks", [])
            field_checks: list[CrossOriginWitnessFieldCheck] = []
            if isinstance(raw_field_checks, list):
                for raw_check in raw_field_checks:
                    if not isinstance(raw_check, Mapping):
                        continue
                    field_checks.append(
                        CrossOriginWitnessFieldCheck(
                            field_name=str(raw_check.get("field_name", "")).strip(),
                            matches=bool(raw_check.get("matches", False)),
                            left_value=str(raw_check.get("left_value", "")).strip(),
                            right_value=str(raw_check.get("right_value", "")).strip(),
                        )
                    )
            cases.append(
                CrossOriginWitnessContractCase(
                    identity=identities.item_id(
                        artifact_kind=StructuredArtifactKind.CROSS_ORIGIN_WITNESS_CONTRACT,
                        source_path=rel_path,
                        item_kind="case",
                        item_key=case_key or f"case-{index}",
                        label=title or case_key,
                    ),
                    case_key=case_key,
                    case_kind=str(raw_case.get("case_kind", "")).strip(),
                    title=title or case_key,
                    status=str(raw_case.get("status", "")).strip(),
                    summary=str(raw_case.get("summary", "")).strip(),
                    left_label=str(raw_case.get("left_label", "")).strip(),
                    right_label=str(raw_case.get("right_label", "")).strip(),
                    evidence_paths=tuple(
                        _normalize_rel_path(root, value)
                        for value in raw_case.get("evidence_paths", [])
                    ),
                    row_keys=tuple(
                        str(value).strip()
                        for value in raw_case.get("row_keys", [])
                        if str(value).strip()
                    ),
                    field_checks=tuple(field_checks),
                )
            )
    return CrossOriginWitnessContractArtifact(
        identity=identity,
        source=source,
        cases=tuple(cases),
        witness_rows=tuple(rows),
    )


def load_local_repro_closure_ledger_artifact(
    *,
    root: Path,
    rel_path: str,
    identities: StructuredArtifactIdentitySpace,
) -> LocalReproClosureLedgerArtifact | None:
    payload = _load_json_mapping_artifact(root / rel_path)
    if payload is None:
        return None
    source = StructuredArtifactSource(
        rel_path=rel_path,
        schema_version=int(payload.get("schema_version", 0) or 0),
        producer=str(payload.get("generated_by", "")).strip(),
    )
    identity = identities.artifact_id(
        artifact_kind=StructuredArtifactKind.LOCAL_REPRO_CLOSURE_LEDGER,
        source_path=rel_path,
        label=rel_path,
    )
    entries: list[LocalReproClosureEntry] = []
    raw_entries = payload.get("entries", [])
    if isinstance(raw_entries, list):
        for index, raw_entry in enumerate(raw_entries, start=1):
            if not isinstance(raw_entry, Mapping):
                continue
            cu_id = str(raw_entry.get("cu_id", "")).strip()
            summary = str(raw_entry.get("summary", "")).strip()
            validation = raw_entry.get("validation", {})
            validation_mapping = validation if isinstance(validation, Mapping) else {}
            entry_identity = identities.item_id(
                artifact_kind=StructuredArtifactKind.LOCAL_REPRO_CLOSURE_LEDGER,
                source_path=rel_path,
                item_kind="entry",
                item_key=cu_id or f"entry-{index}",
                label=cu_id or summary or f"entry-{index}",
            )
            entries.append(
                LocalReproClosureEntry(
                    identity=entry_identity,
                    cu_id=cu_id,
                    summary=summary,
                    validation_statuses=tuple(
                        str(value).strip()
                        for value in validation_mapping.values()
                        if str(value).strip()
                    ),
                )
            )
    return LocalReproClosureLedgerArtifact(
        identity=identity,
        source=source,
        workstream=str(payload.get("workstream", "")).strip(),
        generated_by=str(payload.get("generated_by", "")).strip(),
        entries=tuple(entries),
    )


def load_local_ci_repro_contract_artifact(
    *,
    root: Path,
    rel_path: str,
    identities: StructuredArtifactIdentitySpace,
) -> LocalCiReproContractArtifact | None:
    payload = _load_json_mapping_artifact(root / rel_path)
    if payload is None:
        return None
    source = StructuredArtifactSource(
        rel_path=rel_path,
        schema_version=int(payload.get("schema_version", 0) or 0),
        producer=str(payload.get("generated_by", "")).strip(),
    )
    identity = identities.artifact_id(
        artifact_kind=StructuredArtifactKind.LOCAL_CI_REPRO_CONTRACT,
        source_path=rel_path,
        label=rel_path,
    )
    surfaces: list[LocalCiReproSurface] = []
    raw_surfaces = payload.get("surfaces", [])
    if isinstance(raw_surfaces, list):
        for index, raw_surface in enumerate(raw_surfaces, start=1):
            if not isinstance(raw_surface, Mapping):
                continue
            surface_id = str(raw_surface.get("surface_id", "")).strip()
            surface_identity = identities.item_id(
                artifact_kind=StructuredArtifactKind.LOCAL_CI_REPRO_CONTRACT,
                source_path=rel_path,
                item_kind="surface",
                item_key=surface_id or f"surface-{index}",
                label=str(raw_surface.get("title", "")).strip()
                or surface_id
                or f"surface-{index}",
            )
            capabilities: list[LocalCiReproCapability] = []
            raw_capabilities = raw_surface.get("required_capabilities", [])
            if isinstance(raw_capabilities, list):
                for capability_index, raw_capability in enumerate(
                    raw_capabilities,
                    start=1,
                ):
                    if not isinstance(raw_capability, Mapping):
                        continue
                    capability_id = str(
                        raw_capability.get("capability_id", "")
                    ).strip()
                    capability_identity = identities.item_id(
                        artifact_kind=StructuredArtifactKind.LOCAL_CI_REPRO_CONTRACT,
                        source_path=rel_path,
                        item_kind="capability",
                        item_key=(
                            f"{surface_id}:{capability_id}"
                            if surface_id or capability_id
                            else f"capability-{index}-{capability_index}"
                        ),
                        label=capability_id
                        or str(raw_capability.get("summary", "")).strip()
                        or f"capability-{index}-{capability_index}",
                    )
                    source_groups: list[tuple[str, ...]] = []
                    raw_source_groups = raw_capability.get(
                        "source_alternative_token_groups",
                        [],
                    )
                    if isinstance(raw_source_groups, list):
                        for raw_group in raw_source_groups:
                            if not isinstance(raw_group, list):
                                continue
                            group = tuple(
                                str(item).strip()
                                for item in raw_group
                                if str(item).strip()
                            )
                            if group:
                                source_groups.append(group)
                    command_groups: list[tuple[str, ...]] = []
                    raw_command_groups = raw_capability.get(
                        "command_alternative_token_groups",
                        [],
                    )
                    if isinstance(raw_command_groups, list):
                        for raw_group in raw_command_groups:
                            if not isinstance(raw_group, list):
                                continue
                            group = tuple(
                                str(item).strip()
                                for item in raw_group
                                if str(item).strip()
                            )
                            if group:
                                command_groups.append(group)
                    matched_source = raw_capability.get(
                        "matched_source_alternative_index"
                    )
                    matched_command = raw_capability.get(
                        "matched_command_alternative_index"
                    )
                    capabilities.append(
                        LocalCiReproCapability(
                            identity=capability_identity,
                            capability_id=capability_id,
                            summary=str(
                                raw_capability.get("summary", "")
                            ).strip(),
                            status=str(raw_capability.get("status", "")).strip(),
                            source_alternative_token_groups=tuple(source_groups),
                            command_alternative_token_groups=tuple(command_groups),
                            matched_source_alternative_index=(
                                int(matched_source)
                                if isinstance(matched_source, int)
                                else None
                            ),
                            matched_command_alternative_index=(
                                int(matched_command)
                                if isinstance(matched_command, int)
                                else None
                            ),
                        )
                    )
            required_groups: list[tuple[str, ...]] = []
            raw_required_groups = raw_surface.get("required_token_groups", [])
            if isinstance(raw_required_groups, list):
                for raw_group in raw_required_groups:
                    if not isinstance(raw_group, list):
                        continue
                    group = tuple(
                        str(item).strip()
                        for item in raw_group
                        if str(item).strip()
                    )
                    if group:
                        required_groups.append(group)
            missing_groups: list[tuple[str, ...]] = []
            raw_missing_groups = raw_surface.get("missing_token_groups", [])
            if isinstance(raw_missing_groups, list):
                for raw_group in raw_missing_groups:
                    if not isinstance(raw_group, list):
                        continue
                    group = tuple(
                        str(item).strip()
                        for item in raw_group
                        if str(item).strip()
                    )
                    if group:
                        missing_groups.append(group)
            missing_capability_ids = tuple(
                str(item).strip()
                for item in raw_surface.get("missing_capability_ids", [])
                if str(item).strip()
            )
            surfaces.append(
                LocalCiReproSurface(
                    identity=surface_identity,
                    surface_id=surface_id,
                    surface_kind=str(raw_surface.get("surface_kind", "")).strip(),
                    title=str(raw_surface.get("title", "")).strip(),
                    summary=str(raw_surface.get("summary", "")).strip(),
                    source_ref=_normalize_rel_path(
                        root,
                        raw_surface.get("source_ref", ""),
                    ),
                    mode=str(raw_surface.get("mode", "")).strip(),
                    status=str(raw_surface.get("status", "")).strip(),
                    required_capabilities=tuple(capabilities),
                    missing_capability_ids=missing_capability_ids,
                    required_token_groups=tuple(required_groups),
                    missing_token_groups=tuple(missing_groups),
                    commands=tuple(
                        str(item).strip()
                        for item in raw_surface.get("commands", [])
                        if str(item).strip()
                    ),
                    artifacts=tuple(
                        _normalize_rel_path(root, item)
                        for item in raw_surface.get("artifacts", [])
                        if str(item).strip()
                    ),
                )
            )
    relations: list[LocalCiReproRelation] = []
    raw_relations = payload.get("relations", [])
    if isinstance(raw_relations, list):
        for index, raw_relation in enumerate(raw_relations, start=1):
            if not isinstance(raw_relation, Mapping):
                continue
            relation_id = str(raw_relation.get("relation_id", "")).strip()
            relation_identity = identities.item_id(
                artifact_kind=StructuredArtifactKind.LOCAL_CI_REPRO_CONTRACT,
                source_path=rel_path,
                item_kind="relation",
                item_key=relation_id or f"relation-{index}",
                label=relation_id or f"relation-{index}",
            )
            relations.append(
                LocalCiReproRelation(
                    identity=relation_identity,
                    relation_id=relation_id,
                    relation_kind=str(raw_relation.get("relation_kind", "")).strip(),
                    source_surface_id=str(
                        raw_relation.get("source_surface_id", "")
                    ).strip(),
                    target_surface_id=str(
                        raw_relation.get("target_surface_id", "")
                    ).strip(),
                    source_missing_capability_ids=tuple(
                        str(item).strip()
                        for item in raw_relation.get(
                            "source_missing_capability_ids",
                            [],
                        )
                        if str(item).strip()
                    ),
                    target_missing_capability_ids=tuple(
                        str(item).strip()
                        for item in raw_relation.get(
                            "target_missing_capability_ids",
                            [],
                        )
                        if str(item).strip()
                    ),
                    status=str(raw_relation.get("status", "")).strip(),
                    summary=str(raw_relation.get("summary", "")).strip(),
                )
            )
    return LocalCiReproContractArtifact(
        identity=identity,
        source=source,
        summary=str(payload.get("summary", "")).strip(),
        surfaces=tuple(surfaces),
        relations=tuple(relations),
    )


def load_kernel_vm_alignment_artifact(
    *,
    root: Path,
    rel_path: str,
    identities: StructuredArtifactIdentitySpace,
) -> KernelVmAlignmentArtifact | None:
    payload = _load_json_mapping_artifact(root / rel_path)
    if payload is None:
        return None
    if payload.get("artifact_kind") != StructuredArtifactKind.KERNEL_VM_ALIGNMENT.value:
        return None
    source = StructuredArtifactSource(
        rel_path=rel_path,
        schema_version=int(payload.get("schema_version", 0) or 0),
        producer=str(payload.get("generated_by", "")).strip(),
    )
    identity = identities.artifact_id(
        artifact_kind=StructuredArtifactKind.KERNEL_VM_ALIGNMENT,
        source_path=rel_path,
        label=rel_path,
    )

    def _evidence_refs(raw_items: object) -> tuple[KernelVmEvidenceRef, ...]:
        items: list[KernelVmEvidenceRef] = []
        if isinstance(raw_items, list):
            for raw_item in raw_items:
                if not isinstance(raw_item, Mapping):
                    continue
                ref_path = _normalize_rel_path(root, raw_item.get("rel_path", ""))
                symbol = str(raw_item.get("symbol", "")).strip()
                if not ref_path or not symbol:
                    continue
                items.append(
                    KernelVmEvidenceRef(
                        rel_path=ref_path,
                        evidence_kind=str(raw_item.get("evidence_kind", "")).strip(),
                        symbol=symbol,
                        present=bool(raw_item.get("present", False)),
                    )
                )
        return tuple(items)

    bindings: list[KernelVmBinding] = []
    raw_bindings = payload.get("bindings", [])
    if isinstance(raw_bindings, list):
        for binding_index, raw_binding in enumerate(raw_bindings, start=1):
            if not isinstance(raw_binding, Mapping):
                continue
            binding_id = str(raw_binding.get("binding_id", "")).strip()
            if not binding_id:
                binding_id = f"binding-{binding_index}"
            binding_identity = identities.item_id(
                artifact_kind=StructuredArtifactKind.KERNEL_VM_ALIGNMENT,
                source_path=rel_path,
                item_kind="binding",
                item_key=binding_id,
                label=str(raw_binding.get("title", "")).strip()
                or binding_id,
            )
            capabilities: list[KernelVmCapability] = []
            raw_capabilities = raw_binding.get("capabilities", [])
            if isinstance(raw_capabilities, list):
                for capability_index, raw_capability in enumerate(
                    raw_capabilities,
                    start=1,
                ):
                    if not isinstance(raw_capability, Mapping):
                        continue
                    capability_id = str(
                        raw_capability.get("capability_id", "")
                    ).strip()
                    capability_identity = identities.item_id(
                        artifact_kind=StructuredArtifactKind.KERNEL_VM_ALIGNMENT,
                        source_path=rel_path,
                        item_kind="capability",
                        item_key=(
                            f"{binding_id}:{capability_id}"
                            if capability_id
                            else f"{binding_id}:capability-{capability_index}"
                        ),
                        label=capability_id or f"capability-{capability_index}",
                    )
                    capabilities.append(
                        KernelVmCapability(
                            identity=capability_identity,
                            capability_id=capability_id,
                            requirement_kind=str(
                                raw_capability.get("requirement_kind", "")
                            ).strip(),
                            status=str(raw_capability.get("status", "")).strip(),
                            match_mode=str(
                                raw_capability.get("match_mode", "")
                            ).strip(),
                            description=str(
                                raw_capability.get("description", "")
                            ).strip(),
                            residue_kind=str(
                                raw_capability.get("residue_kind", "")
                            ).strip(),
                            severity=str(
                                raw_capability.get("severity", "")
                            ).strip(),
                            score=int(raw_capability.get("score", 0) or 0),
                            expected_refs=_evidence_refs(
                                raw_capability.get("expected_refs", [])
                            ),
                            matched_refs=_evidence_refs(
                                raw_capability.get("matched_refs", [])
                            ),
                            missing_refs=_evidence_refs(
                                raw_capability.get("missing_refs", [])
                            ),
                        )
                    )
            bindings.append(
                KernelVmBinding(
                    identity=binding_identity,
                    binding_id=binding_id,
                    fragment_id=str(raw_binding.get("fragment_id", "")).strip(),
                    title=str(raw_binding.get("title", "")).strip(),
                    status=str(raw_binding.get("status", "")).strip(),
                    summary=str(raw_binding.get("summary", "")).strip(),
                    kernel_terms=tuple(
                        str(item).strip()
                        for item in raw_binding.get("kernel_terms", [])
                        if str(item).strip()
                    ),
                    runtime_surface_symbols=tuple(
                        str(item).strip()
                        for item in raw_binding.get("runtime_surface_symbols", [])
                        if str(item).strip()
                    ),
                    realizer_symbols=tuple(
                        str(item).strip()
                        for item in raw_binding.get("realizer_symbols", [])
                        if str(item).strip()
                    ),
                    runtime_object_symbols=tuple(
                        str(item).strip()
                        for item in raw_binding.get("runtime_object_symbols", [])
                        if str(item).strip()
                    ),
                    missing_capability_ids=tuple(
                        str(item).strip()
                        for item in raw_binding.get("missing_capability_ids", [])
                        if str(item).strip()
                    ),
                    residue_ids=tuple(
                        str(item).strip()
                        for item in raw_binding.get("residue_ids", [])
                        if str(item).strip()
                    ),
                    evidence_paths=tuple(
                        _normalize_rel_path(root, item)
                        for item in raw_binding.get("evidence_paths", [])
                        if _normalize_rel_path(root, item)
                    ),
                    capabilities=tuple(capabilities),
                )
            )

    residues: list[KernelVmResidue] = []
    raw_residues = payload.get("residues", [])
    if isinstance(raw_residues, list):
        for residue_index, raw_residue in enumerate(raw_residues, start=1):
            if not isinstance(raw_residue, Mapping):
                continue
            residue_id = str(raw_residue.get("residue_id", "")).strip()
            if not residue_id:
                residue_id = f"residue-{residue_index}"
            residue_identity = identities.item_id(
                artifact_kind=StructuredArtifactKind.KERNEL_VM_ALIGNMENT,
                source_path=rel_path,
                item_kind="residue",
                item_key=residue_id,
                label=str(raw_residue.get("title", "")).strip()
                or residue_id,
            )
            residues.append(
                KernelVmResidue(
                    identity=residue_identity,
                    residue_id=residue_id,
                    binding_id=str(raw_residue.get("binding_id", "")).strip(),
                    fragment_id=str(raw_residue.get("fragment_id", "")).strip(),
                    residue_kind=str(raw_residue.get("residue_kind", "")).strip(),
                    severity=str(raw_residue.get("severity", "")).strip(),
                    score=int(raw_residue.get("score", 0) or 0),
                    title=str(raw_residue.get("title", "")).strip(),
                    message=str(raw_residue.get("message", "")).strip(),
                    missing_capability_ids=tuple(
                        str(item).strip()
                        for item in raw_residue.get("missing_capability_ids", [])
                        if str(item).strip()
                    ),
                    kernel_terms=tuple(
                        str(item).strip()
                        for item in raw_residue.get("kernel_terms", [])
                        if str(item).strip()
                    ),
                    runtime_surface_symbols=tuple(
                        str(item).strip()
                        for item in raw_residue.get("runtime_surface_symbols", [])
                        if str(item).strip()
                    ),
                    realizer_symbols=tuple(
                        str(item).strip()
                        for item in raw_residue.get("realizer_symbols", [])
                        if str(item).strip()
                    ),
                    runtime_object_symbols=tuple(
                        str(item).strip()
                        for item in raw_residue.get("runtime_object_symbols", [])
                        if str(item).strip()
                    ),
                    evidence_paths=tuple(
                        _normalize_rel_path(root, item)
                        for item in raw_residue.get("evidence_paths", [])
                        if _normalize_rel_path(root, item)
                    ),
                )
            )

    summary = payload.get("summary", {})
    summary_mapping = summary if isinstance(summary, Mapping) else {}
    return KernelVmAlignmentArtifact(
        identity=identity,
        source=source,
        fragment_id=str(payload.get("fragment_id", "")).strip(),
        binding_count=int(summary_mapping.get("binding_count", len(bindings)) or 0),
        pass_count=int(summary_mapping.get("pass_count", 0) or 0),
        partial_count=int(summary_mapping.get("partial_count", 0) or 0),
        fail_count=int(summary_mapping.get("fail_count", 0) or 0),
        residue_count=int(summary_mapping.get("residue_count", len(residues)) or 0),
        bindings=tuple(bindings),
        residues=tuple(residues),
    )


def load_identity_grammar_completion_artifact(
    *,
    root: Path,
    rel_path: str,
    identities: StructuredArtifactIdentitySpace,
) -> IdentityGrammarCompletionArtifact | None:
    payload = _load_json_mapping_artifact(root / rel_path)
    if payload is None:
        return None
    if (
        payload.get("artifact_kind")
        != StructuredArtifactKind.IDENTITY_GRAMMAR_COMPLETION.value
    ):
        return None
    source = StructuredArtifactSource(
        rel_path=rel_path,
        schema_version=int(payload.get("schema_version", 0) or 0),
        producer=str(payload.get("generated_by", "")).strip(),
    )
    identity = identities.artifact_id(
        artifact_kind=StructuredArtifactKind.IDENTITY_GRAMMAR_COMPLETION,
        source_path=rel_path,
        label=rel_path,
    )
    summary_mapping = (
        payload.get("summary", {}) if isinstance(payload.get("summary"), Mapping) else {}
    )
    surfaces: list[IdentityGrammarCompletionSurface] = []
    raw_surfaces = payload.get("surfaces", [])
    if isinstance(raw_surfaces, list):
        for index, raw_surface in enumerate(raw_surfaces, start=1):
            if not isinstance(raw_surface, Mapping):
                continue
            surface_id = str(raw_surface.get("surface_id", "")).strip()
            title = str(raw_surface.get("title", "")).strip()
            surfaces.append(
                IdentityGrammarCompletionSurface(
                    identity=identities.item_id(
                        artifact_kind=StructuredArtifactKind.IDENTITY_GRAMMAR_COMPLETION,
                        source_path=rel_path,
                        item_kind="surface",
                        item_key=surface_id or f"surface-{index}",
                        label=title or surface_id or f"surface-{index}",
                    ),
                    surface_id=surface_id,
                    title=title,
                    status=str(raw_surface.get("status", "")).strip(),
                    summary=str(raw_surface.get("summary", "")).strip(),
                    residue_ids=tuple(
                        str(item).strip()
                        for item in raw_surface.get("residue_ids", [])
                        if str(item).strip()
                    ),
                    evidence_paths=tuple(
                        _normalize_rel_path(root, item)
                        for item in raw_surface.get("evidence_paths", [])
                        if str(item).strip()
                    ),
                )
            )
    residues: list[IdentityGrammarCompletionResidue] = []
    raw_residues = payload.get("residues", [])
    if isinstance(raw_residues, list):
        for index, raw_residue in enumerate(raw_residues, start=1):
            if not isinstance(raw_residue, Mapping):
                continue
            residue_id = str(raw_residue.get("residue_id", "")).strip()
            title = str(raw_residue.get("title", "")).strip()
            residues.append(
                IdentityGrammarCompletionResidue(
                    identity=identities.item_id(
                        artifact_kind=StructuredArtifactKind.IDENTITY_GRAMMAR_COMPLETION,
                        source_path=rel_path,
                        item_kind="residue",
                        item_key=residue_id or f"residue-{index}",
                        label=title or residue_id or f"residue-{index}",
                    ),
                    residue_id=residue_id,
                    surface_id=str(raw_residue.get("surface_id", "")).strip(),
                    residue_kind=str(raw_residue.get("residue_kind", "")).strip(),
                    severity=str(raw_residue.get("severity", "")).strip(),
                    score=int(raw_residue.get("score", 0) or 0),
                    title=title,
                    message=str(raw_residue.get("message", "")).strip(),
                    evidence_paths=tuple(
                        _normalize_rel_path(root, item)
                        for item in raw_residue.get("evidence_paths", [])
                        if str(item).strip()
                    ),
                )
            )
    return IdentityGrammarCompletionArtifact(
        identity=identity,
        source=source,
        surface_count=int(summary_mapping.get("surface_count", len(surfaces)) or 0),
        pass_count=int(summary_mapping.get("pass_count", 0) or 0),
        fail_count=int(summary_mapping.get("fail_count", 0) or 0),
        residue_count=int(summary_mapping.get("residue_count", len(residues)) or 0),
        highest_severity=str(summary_mapping.get("highest_severity", "")).strip(),
        surfaces=tuple(surfaces),
        residues=tuple(residues),
    )


__all__ = [
    "ControllerDriftArtifact",
    "ControllerDriftFinding",
    "CrossOriginWitnessContractArtifact",
    "CrossOriginWitnessContractCase",
    "CrossOriginWitnessFieldCheck",
    "CrossOriginWitnessRow",
    "DocflowCommit",
    "DocflowComplianceArtifact",
    "DocflowComplianceRow",
    "DocflowIssueLifecycle",
    "DocflowIssueReference",
    "DocflowObligationEntry",
    "DocflowPacket",
    "DocflowPacketEnforcementArtifact",
    "DocflowPacketRow",
    "GitStateLineSpan",
    "IdentityGrammarCompletionArtifact",
    "IdentityGrammarCompletionResidue",
    "IdentityGrammarCompletionSurface",
    "IngressMergeParityArtifact",
    "IngressMergeParityCase",
    "IngressMergeParityFieldCheck",
    "JUnitFailureArtifact",
    "JUnitFailureCase",
    "KernelVmAlignmentArtifact",
    "KernelVmBinding",
    "KernelVmCapability",
    "KernelVmEvidenceRef",
    "KernelVmResidue",
    "LocalCiReproContractArtifact",
    "LocalCiReproCapability",
    "LocalCiReproRelation",
    "LocalCiReproSurface",
    "LocalReproClosureEntry",
    "LocalReproClosureLedgerArtifact",
    "StructuredArtifactDecompositionIdentity",
    "StructuredArtifactDecompositionKind",
    "StructuredArtifactDecompositionRelation",
    "StructuredArtifactDecompositionRelationKind",
    "StructuredArtifactIdentity",
    "StructuredArtifactIdentityNamespace",
    "StructuredArtifactIdentitySpace",
    "StructuredArtifactKind",
    "StructuredArtifactSource",
    "TestEvidenceArtifact",
    "TestEvidenceCase",
    "TestEvidenceSite",
    "build_ingress_merge_parity_artifact",
    "load_cross_origin_witness_contract_artifact",
    "load_controller_drift_artifact",
    "load_docflow_compliance_artifact",
    "load_docflow_packet_enforcement_artifact",
    "load_git_state_artifact",
    "load_identity_grammar_completion_artifact",
    "load_ingress_merge_parity_artifact",
    "load_junit_failure_artifact",
    "load_kernel_vm_alignment_artifact",
    "load_local_ci_repro_contract_artifact",
    "load_local_repro_closure_ledger_artifact",
    "load_test_evidence_artifact",
    "write_ingress_merge_parity_artifact",
]
