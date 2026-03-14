from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

from gabion.order_contract import ordered_or_sorted

_ARTIFACT_KIND = "kernel_vm_alignment"
_SCHEMA_VERSION = 1
_GENERATED_BY = (
    "gabion.tooling.runtime.kernel_vm_alignment_artifact."
    "build_kernel_vm_alignment_artifact_payload"
)

_TTL_ONTOLOGY_PATH = "in/lg_kernel_ontology_cut_elim-1.ttl"
_TTL_SEMANTICS_DOC_PATH = "docs/ttl_kernel_semantics.md"
_ASPF_RUNTIME_PATH = "src/gabion/analysis/aspf/aspf_lattice_algebra.py"
_SEMANTIC_FRAGMENT_PATH = "src/gabion/analysis/projection/semantic_fragment.py"
_SEMANTIC_LOWERING_PATH = (
    "src/gabion/analysis/projection/projection_semantic_lowering.py"
)
_SEMANTIC_COMPILE_PATH = (
    "src/gabion/analysis/projection/semantic_fragment_compile.py"
)


def _sorted[T](values: list[T], *, key=None) -> list[T]:
    return ordered_or_sorted(
        values,
        source="tooling.runtime.kernel_vm_alignment_artifact",
        key=key,
    )


class KernelVmEvidenceRefPayload(TypedDict):
    rel_path: str
    evidence_kind: str
    symbol: str
    present: bool


class KernelVmCapabilityPayload(TypedDict):
    capability_id: str
    requirement_kind: str
    status: str
    match_mode: str
    description: str
    residue_kind: str
    severity: str
    score: int
    expected_refs: list[KernelVmEvidenceRefPayload]
    matched_refs: list[KernelVmEvidenceRefPayload]
    missing_refs: list[KernelVmEvidenceRefPayload]


class KernelVmBindingPayload(TypedDict):
    binding_id: str
    fragment_id: str
    title: str
    status: str
    summary: str
    kernel_terms: list[str]
    runtime_surface_symbols: list[str]
    realizer_symbols: list[str]
    runtime_object_symbols: list[str]
    missing_capability_ids: list[str]
    residue_ids: list[str]
    evidence_paths: list[str]
    capabilities: list[KernelVmCapabilityPayload]


class KernelVmResiduePayload(TypedDict):
    residue_id: str
    binding_id: str
    fragment_id: str
    residue_kind: str
    severity: str
    score: int
    title: str
    message: str
    missing_capability_ids: list[str]
    kernel_terms: list[str]
    runtime_surface_symbols: list[str]
    realizer_symbols: list[str]
    runtime_object_symbols: list[str]
    evidence_paths: list[str]


class KernelVmAlignmentArtifactPayload(TypedDict):
    artifact_kind: str
    schema_version: int
    generated_by: str
    fragment_id: str
    summary: dict[str, object]
    bindings: list[KernelVmBindingPayload]
    residues: list[KernelVmResiduePayload]


@dataclass(frozen=True)
class _EvidenceRef:
    rel_path: str
    evidence_kind: Literal["ttl_term", "doc_text", "python_symbol"]
    symbol: str


@dataclass(frozen=True)
class _CapabilitySpec:
    capability_id: str
    requirement_kind: str
    description: str
    match_mode: Literal["all", "any"]
    evidence_refs: tuple[_EvidenceRef, ...]
    residue_kind: str
    severity: str
    score: int


@dataclass(frozen=True)
class _BindingSpec:
    binding_id: str
    fragment_id: str
    title: str
    kernel_terms: tuple[str, ...]
    runtime_surface_symbols: tuple[str, ...]
    realizer_symbols: tuple[str, ...]
    runtime_object_symbols: tuple[str, ...]
    capabilities: tuple[_CapabilitySpec, ...]


@dataclass(frozen=True)
class _EvaluatedCapability:
    spec: _CapabilitySpec
    matched_refs: tuple[_EvidenceRef, ...]
    missing_refs: tuple[_EvidenceRef, ...]

    @property
    def status(self) -> str:
        return "pass" if not self.missing_refs else "fail"


_RUNTIME_OBJECT_IMAGE_PATHS = (
    _ASPF_RUNTIME_PATH,
    _SEMANTIC_FRAGMENT_PATH,
    _SEMANTIC_LOWERING_PATH,
    _SEMANTIC_COMPILE_PATH,
)

_FRAGMENT_ID = "ttl_kernel_vm.fragment.augmented_rule_polarity_query_ast"


def _evidence_refs_for_symbol(
    *,
    evidence_kind: Literal["ttl_term", "doc_text", "python_symbol"],
    symbol: str,
    rel_paths: tuple[str, ...],
) -> tuple[_EvidenceRef, ...]:
    return tuple(
        _EvidenceRef(rel_path=rel_path, evidence_kind=evidence_kind, symbol=symbol)
        for rel_path in rel_paths
    )


_BINDING_SPECS = (
    _BindingSpec(
        binding_id="kernel_vm.augmented_rule_core",
        fragment_id=_FRAGMENT_ID,
        title="AugmentedRule core object over semantic-row reflection",
        kernel_terms=(
            "lg:AugmentedRule",
            "lg:hasSyntaxClause",
            "lg:hasTypingClause",
            "lg:hasCategoricalClause",
        ),
        runtime_surface_symbols=(
            "CanonicalWitnessedSemanticRow",
            "reflect_projection_fiber_witness",
        ),
        realizer_symbols=(),
        runtime_object_symbols=("AugmentedRule",),
        capabilities=(
            _CapabilitySpec(
                capability_id="law_source",
                requirement_kind="law_source",
                description="TTL kernel law source for AugmentedRule core",
                match_mode="all",
                evidence_refs=(
                    *_evidence_refs_for_symbol(
                        evidence_kind="ttl_term",
                        symbol="lg:AugmentedRule",
                        rel_paths=(_TTL_ONTOLOGY_PATH,),
                    ),
                    *_evidence_refs_for_symbol(
                        evidence_kind="ttl_term",
                        symbol="lg:hasSyntaxClause",
                        rel_paths=(_TTL_ONTOLOGY_PATH,),
                    ),
                    *_evidence_refs_for_symbol(
                        evidence_kind="ttl_term",
                        symbol="lg:hasTypingClause",
                        rel_paths=(_TTL_ONTOLOGY_PATH,),
                    ),
                    *_evidence_refs_for_symbol(
                        evidence_kind="ttl_term",
                        symbol="lg:hasCategoricalClause",
                        rel_paths=(_TTL_ONTOLOGY_PATH,),
                    ),
                    *_evidence_refs_for_symbol(
                        evidence_kind="doc_text",
                        symbol="AugmentedRule",
                        rel_paths=(_TTL_SEMANTICS_DOC_PATH,),
                    ),
                ),
                residue_kind="missing_law_source",
                severity="warning",
                score=8,
            ),
            _CapabilitySpec(
                capability_id="runtime_surface",
                requirement_kind="runtime_surface",
                description="runtime semantic-row surface for AugmentedRule core",
                match_mode="all",
                evidence_refs=(
                    *_evidence_refs_for_symbol(
                        evidence_kind="python_symbol",
                        symbol="CanonicalWitnessedSemanticRow",
                        rel_paths=(_SEMANTIC_FRAGMENT_PATH,),
                    ),
                    *_evidence_refs_for_symbol(
                        evidence_kind="python_symbol",
                        symbol="reflect_projection_fiber_witness",
                        rel_paths=(_SEMANTIC_FRAGMENT_PATH,),
                    ),
                ),
                residue_kind="missing_runtime_surface",
                severity="warning",
                score=7,
            ),
            _CapabilitySpec(
                capability_id="runtime_object_image",
                requirement_kind="runtime_object_image",
                description="explicit runtime object image for AugmentedRule",
                match_mode="any",
                evidence_refs=_evidence_refs_for_symbol(
                    evidence_kind="python_symbol",
                    symbol="AugmentedRule",
                    rel_paths=_RUNTIME_OBJECT_IMAGE_PATHS,
                ),
                residue_kind="missing_runtime_object_image",
                severity="warning",
                score=6,
            ),
        ),
    ),
    _BindingSpec(
        binding_id="kernel_vm.rule_polarity_package",
        fragment_id=_FRAGMENT_ID,
        title="RulePolarity package over ASPF witnesses and support reflection",
        kernel_terms=(
            "lg:RulePolarity",
            "lg:WitnessDomain",
            "lg:PredicateDomain",
            "lg:SupportReflection",
        ),
        runtime_surface_symbols=(
            "NaturalityWitness",
            "FrontierWitness",
            "SemanticOpKind",
        ),
        realizer_symbols=(
            "compile_projection_fiber_support_reflect_to_shacl",
            "compile_projection_fiber_support_reflect_to_sparql",
        ),
        runtime_object_symbols=(
            "RulePolarity",
            "WitnessDomain",
            "PredicateDomain",
            "SupportReflection",
        ),
        capabilities=(
            _CapabilitySpec(
                capability_id="law_source",
                requirement_kind="law_source",
                description="TTL kernel polarity package",
                match_mode="all",
                evidence_refs=(
                    *_evidence_refs_for_symbol(
                        evidence_kind="ttl_term",
                        symbol="lg:RulePolarity",
                        rel_paths=(_TTL_ONTOLOGY_PATH,),
                    ),
                    *_evidence_refs_for_symbol(
                        evidence_kind="ttl_term",
                        symbol="lg:WitnessDomain",
                        rel_paths=(_TTL_ONTOLOGY_PATH,),
                    ),
                    *_evidence_refs_for_symbol(
                        evidence_kind="ttl_term",
                        symbol="lg:PredicateDomain",
                        rel_paths=(_TTL_ONTOLOGY_PATH,),
                    ),
                    *_evidence_refs_for_symbol(
                        evidence_kind="ttl_term",
                        symbol="lg:SupportReflection",
                        rel_paths=(_TTL_ONTOLOGY_PATH,),
                    ),
                    *_evidence_refs_for_symbol(
                        evidence_kind="doc_text",
                        symbol="RulePolarity",
                        rel_paths=(_TTL_SEMANTICS_DOC_PATH,),
                    ),
                ),
                residue_kind="missing_law_source",
                severity="warning",
                score=8,
            ),
            _CapabilitySpec(
                capability_id="runtime_surface",
                requirement_kind="runtime_surface",
                description="runtime witness substrate for polarity package",
                match_mode="all",
                evidence_refs=(
                    *_evidence_refs_for_symbol(
                        evidence_kind="python_symbol",
                        symbol="NaturalityWitness",
                        rel_paths=(_ASPF_RUNTIME_PATH,),
                    ),
                    *_evidence_refs_for_symbol(
                        evidence_kind="python_symbol",
                        symbol="FrontierWitness",
                        rel_paths=(_ASPF_RUNTIME_PATH,),
                    ),
                    *_evidence_refs_for_symbol(
                        evidence_kind="python_symbol",
                        symbol="SemanticOpKind",
                        rel_paths=(_SEMANTIC_FRAGMENT_PATH,),
                    ),
                ),
                residue_kind="missing_runtime_surface",
                severity="warning",
                score=7,
            ),
            _CapabilitySpec(
                capability_id="realizer",
                requirement_kind="runtime_realizer",
                description="support-reflection realizer surfaces",
                match_mode="all",
                evidence_refs=(
                    *_evidence_refs_for_symbol(
                        evidence_kind="python_symbol",
                        symbol="compile_projection_fiber_support_reflect_to_shacl",
                        rel_paths=(_SEMANTIC_COMPILE_PATH,),
                    ),
                    *_evidence_refs_for_symbol(
                        evidence_kind="python_symbol",
                        symbol="compile_projection_fiber_support_reflect_to_sparql",
                        rel_paths=(_SEMANTIC_COMPILE_PATH,),
                    ),
                ),
                residue_kind="missing_runtime_realizer",
                severity="warning",
                score=5,
            ),
            _CapabilitySpec(
                capability_id="runtime_object_image",
                requirement_kind="runtime_object_image",
                description="explicit runtime object image for polarity package",
                match_mode="any",
                evidence_refs=tuple(
                    ref
                    for symbol in ("RulePolarity", "WitnessDomain", "PredicateDomain", "SupportReflection")
                    for ref in _evidence_refs_for_symbol(
                        evidence_kind="python_symbol",
                        symbol=symbol,
                        rel_paths=_RUNTIME_OBJECT_IMAGE_PATHS,
                    )
                ),
                residue_kind="missing_runtime_object_image",
                severity="warning",
                score=6,
            ),
        ),
    ),
    _BindingSpec(
        binding_id="kernel_vm.closed_rule_cell_quotient_recovery",
        fragment_id=_FRAGMENT_ID,
        title="ClosedRuleCell quotient recovery over semantic lowering",
        kernel_terms=(
            "lg:ClosedRuleCell",
            "lg:hasExtentClosure",
            "lg:hasIntentClosure",
        ),
        runtime_surface_symbols=(
            "ProjectionSemanticLoweringPlan",
            "SemanticOpKind",
        ),
        realizer_symbols=(
            "compile_projection_fiber_quotient_face_to_shacl",
            "compile_projection_fiber_quotient_face_to_sparql",
        ),
        runtime_object_symbols=("ClosedRuleCell",),
        capabilities=(
            _CapabilitySpec(
                capability_id="law_source",
                requirement_kind="law_source",
                description="TTL closed rule-cell and quotient recovery law source",
                match_mode="all",
                evidence_refs=(
                    *_evidence_refs_for_symbol(
                        evidence_kind="ttl_term",
                        symbol="lg:ClosedRuleCell",
                        rel_paths=(_TTL_ONTOLOGY_PATH,),
                    ),
                    *_evidence_refs_for_symbol(
                        evidence_kind="ttl_term",
                        symbol="lg:hasExtentClosure",
                        rel_paths=(_TTL_ONTOLOGY_PATH,),
                    ),
                    *_evidence_refs_for_symbol(
                        evidence_kind="ttl_term",
                        symbol="lg:hasIntentClosure",
                        rel_paths=(_TTL_ONTOLOGY_PATH,),
                    ),
                    *_evidence_refs_for_symbol(
                        evidence_kind="doc_text",
                        symbol="ClosedRuleCell",
                        rel_paths=(_TTL_SEMANTICS_DOC_PATH,),
                    ),
                ),
                residue_kind="missing_law_source",
                severity="warning",
                score=8,
            ),
            _CapabilitySpec(
                capability_id="runtime_surface",
                requirement_kind="runtime_surface",
                description="runtime lowering surface for quotient recovery",
                match_mode="all",
                evidence_refs=(
                    *_evidence_refs_for_symbol(
                        evidence_kind="python_symbol",
                        symbol="ProjectionSemanticLoweringPlan",
                        rel_paths=(_SEMANTIC_LOWERING_PATH,),
                    ),
                    *_evidence_refs_for_symbol(
                        evidence_kind="python_symbol",
                        symbol="SemanticOpKind",
                        rel_paths=(_SEMANTIC_FRAGMENT_PATH,),
                    ),
                ),
                residue_kind="missing_runtime_surface",
                severity="warning",
                score=7,
            ),
            _CapabilitySpec(
                capability_id="realizer",
                requirement_kind="runtime_realizer",
                description="quotient-face realizer surfaces",
                match_mode="all",
                evidence_refs=(
                    *_evidence_refs_for_symbol(
                        evidence_kind="python_symbol",
                        symbol="compile_projection_fiber_quotient_face_to_shacl",
                        rel_paths=(_SEMANTIC_COMPILE_PATH,),
                    ),
                    *_evidence_refs_for_symbol(
                        evidence_kind="python_symbol",
                        symbol="compile_projection_fiber_quotient_face_to_sparql",
                        rel_paths=(_SEMANTIC_COMPILE_PATH,),
                    ),
                ),
                residue_kind="missing_runtime_realizer",
                severity="warning",
                score=5,
            ),
            _CapabilitySpec(
                capability_id="runtime_object_image",
                requirement_kind="runtime_object_image",
                description="explicit runtime object image for ClosedRuleCell",
                match_mode="any",
                evidence_refs=_evidence_refs_for_symbol(
                    evidence_kind="python_symbol",
                    symbol="ClosedRuleCell",
                    rel_paths=_RUNTIME_OBJECT_IMAGE_PATHS,
                ),
                residue_kind="missing_runtime_object_image",
                severity="warning",
                score=6,
            ),
        ),
    ),
    _BindingSpec(
        binding_id="kernel_vm.query_ast_reflective_boundary",
        fragment_id=_FRAGMENT_ID,
        title="Query AST reflective boundary over compiled SHACL/SPARQL plans",
        kernel_terms=(
            "lg:SelectQuery",
            "lg:TriplePattern",
            "lg:JoinPattern",
            "lg:AntiJoinPattern",
        ),
        runtime_surface_symbols=(
            "CompiledShaclPlan",
            "CompiledSparqlPlan",
        ),
        realizer_symbols=(
            "compile_projection_fiber_reflect_to_shacl",
            "compile_projection_fiber_reflect_to_sparql",
            "compile_projection_fiber_negate_to_sparql",
            "compile_projection_fiber_existential_image_to_sparql",
        ),
        runtime_object_symbols=(
            "SelectQuery",
            "TriplePattern",
            "JoinPattern",
            "AntiJoinPattern",
        ),
        capabilities=(
            _CapabilitySpec(
                capability_id="law_source",
                requirement_kind="law_source",
                description="TTL query AST and reflective boundary law source",
                match_mode="all",
                evidence_refs=(
                    *_evidence_refs_for_symbol(
                        evidence_kind="ttl_term",
                        symbol="lg:SelectQuery",
                        rel_paths=(_TTL_ONTOLOGY_PATH,),
                    ),
                    *_evidence_refs_for_symbol(
                        evidence_kind="ttl_term",
                        symbol="lg:TriplePattern",
                        rel_paths=(_TTL_ONTOLOGY_PATH,),
                    ),
                    *_evidence_refs_for_symbol(
                        evidence_kind="ttl_term",
                        symbol="lg:JoinPattern",
                        rel_paths=(_TTL_ONTOLOGY_PATH,),
                    ),
                    *_evidence_refs_for_symbol(
                        evidence_kind="ttl_term",
                        symbol="lg:AntiJoinPattern",
                        rel_paths=(_TTL_ONTOLOGY_PATH,),
                    ),
                    *_evidence_refs_for_symbol(
                        evidence_kind="doc_text",
                        symbol="Query AST",
                        rel_paths=(_TTL_SEMANTICS_DOC_PATH,),
                    ),
                ),
                residue_kind="missing_law_source",
                severity="warning",
                score=8,
            ),
            _CapabilitySpec(
                capability_id="runtime_surface",
                requirement_kind="runtime_surface",
                description="compiled runtime surface for reflective query boundaries",
                match_mode="all",
                evidence_refs=(
                    *_evidence_refs_for_symbol(
                        evidence_kind="python_symbol",
                        symbol="CompiledShaclPlan",
                        rel_paths=(_SEMANTIC_COMPILE_PATH,),
                    ),
                    *_evidence_refs_for_symbol(
                        evidence_kind="python_symbol",
                        symbol="CompiledSparqlPlan",
                        rel_paths=(_SEMANTIC_COMPILE_PATH,),
                    ),
                ),
                residue_kind="missing_runtime_surface",
                severity="warning",
                score=7,
            ),
            _CapabilitySpec(
                capability_id="realizer",
                requirement_kind="runtime_realizer",
                description="reflective query realizer surfaces",
                match_mode="all",
                evidence_refs=(
                    *_evidence_refs_for_symbol(
                        evidence_kind="python_symbol",
                        symbol="compile_projection_fiber_reflect_to_shacl",
                        rel_paths=(_SEMANTIC_COMPILE_PATH,),
                    ),
                    *_evidence_refs_for_symbol(
                        evidence_kind="python_symbol",
                        symbol="compile_projection_fiber_reflect_to_sparql",
                        rel_paths=(_SEMANTIC_COMPILE_PATH,),
                    ),
                    *_evidence_refs_for_symbol(
                        evidence_kind="python_symbol",
                        symbol="compile_projection_fiber_negate_to_sparql",
                        rel_paths=(_SEMANTIC_COMPILE_PATH,),
                    ),
                    *_evidence_refs_for_symbol(
                        evidence_kind="python_symbol",
                        symbol="compile_projection_fiber_existential_image_to_sparql",
                        rel_paths=(_SEMANTIC_COMPILE_PATH,),
                    ),
                ),
                residue_kind="missing_runtime_realizer",
                severity="warning",
                score=5,
            ),
            _CapabilitySpec(
                capability_id="runtime_object_image",
                requirement_kind="runtime_object_image",
                description="explicit runtime query-AST object image",
                match_mode="any",
                evidence_refs=tuple(
                    ref
                    for symbol in (
                        "SelectQuery",
                        "TriplePattern",
                        "JoinPattern",
                        "AntiJoinPattern",
                    )
                    for ref in _evidence_refs_for_symbol(
                        evidence_kind="python_symbol",
                        symbol=symbol,
                        rel_paths=_RUNTIME_OBJECT_IMAGE_PATHS,
                    )
                ),
                residue_kind="missing_runtime_object_image",
                severity="warning",
                score=6,
            ),
        ),
    ),
)


def _read_text(root: Path, rel_path: str) -> str:
    path = root / rel_path
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _python_symbol_index(root: Path, rel_path: str) -> set[str]:
    path = root / rel_path
    if not path.exists():
        return set()
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return set()
    try:
        module = ast.parse(source, filename=str(path))
    except SyntaxError:
        return set()
    symbols: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            symbols.add(node.name)
    return symbols


def _evidence_ref_present(
    *,
    root: Path,
    ref: _EvidenceRef,
    python_symbols_by_path: dict[str, set[str]],
    text_by_path: dict[str, str],
) -> bool:
    if ref.evidence_kind == "python_symbol":
        return ref.symbol in python_symbols_by_path.setdefault(
            ref.rel_path,
            _python_symbol_index(root, ref.rel_path),
        )
    text = text_by_path.setdefault(ref.rel_path, _read_text(root, ref.rel_path))
    return ref.symbol in text


def _evaluated_capability(
    *,
    root: Path,
    spec: _CapabilitySpec,
    python_symbols_by_path: dict[str, set[str]],
    text_by_path: dict[str, str],
) -> _EvaluatedCapability:
    matched_refs: list[_EvidenceRef] = []
    missing_refs: list[_EvidenceRef] = []
    for ref in spec.evidence_refs:
        if _evidence_ref_present(
            root=root,
            ref=ref,
            python_symbols_by_path=python_symbols_by_path,
            text_by_path=text_by_path,
        ):
            matched_refs.append(ref)
        else:
            missing_refs.append(ref)
    if spec.match_mode == "any":
        if matched_refs:
            missing_refs = []
        else:
            matched_refs = []
    return _EvaluatedCapability(
        spec=spec,
        matched_refs=tuple(matched_refs),
        missing_refs=tuple(missing_refs),
    )


def _capability_payload(result: _EvaluatedCapability) -> KernelVmCapabilityPayload:
    def _render(ref: _EvidenceRef, *, present: bool) -> KernelVmEvidenceRefPayload:
        return {
            "rel_path": ref.rel_path,
            "evidence_kind": ref.evidence_kind,
            "symbol": ref.symbol,
            "present": present,
        }

    return {
        "capability_id": result.spec.capability_id,
        "requirement_kind": result.spec.requirement_kind,
        "status": result.status,
        "match_mode": result.spec.match_mode,
        "description": result.spec.description,
        "residue_kind": result.spec.residue_kind,
        "severity": result.spec.severity,
        "score": result.spec.score,
        "expected_refs": [
            _render(ref, present=ref in result.matched_refs)
            for ref in result.spec.evidence_refs
        ],
        "matched_refs": [_render(ref, present=True) for ref in result.matched_refs],
        "missing_refs": [_render(ref, present=False) for ref in result.missing_refs],
    }


def _binding_status(capabilities: tuple[_EvaluatedCapability, ...]) -> str:
    failed_ids = {item.spec.capability_id for item in capabilities if item.status != "pass"}
    if "law_source" in failed_ids or "runtime_surface" in failed_ids:
        return "fail"
    if failed_ids:
        return "partial"
    return "pass"


def build_kernel_vm_alignment_artifact_payload(*, root: Path) -> KernelVmAlignmentArtifactPayload:
    python_symbols_by_path: dict[str, set[str]] = {}
    text_by_path: dict[str, str] = {}
    bindings: list[KernelVmBindingPayload] = []
    residues: list[KernelVmResiduePayload] = []

    for binding_spec in _BINDING_SPECS:
        capability_results = tuple(
            _evaluated_capability(
                root=root,
                spec=capability_spec,
                python_symbols_by_path=python_symbols_by_path,
                text_by_path=text_by_path,
            )
            for capability_spec in binding_spec.capabilities
        )
        missing_capability_ids = tuple(
            item.spec.capability_id
            for item in capability_results
            if item.status != "pass"
        )
        residue_ids: list[str] = []
        for capability in capability_results:
            if capability.status == "pass":
                continue
            missing_refs = capability.missing_refs or capability.spec.evidence_refs
            residue_id = (
                f"{binding_spec.binding_id}:{capability.spec.residue_kind}"
            )
            residue_ids.append(residue_id)
            evidence_paths = tuple(
                dict.fromkeys(ref.rel_path for ref in (*capability.matched_refs, *missing_refs))
            )
            missing_evidence_text = ", ".join(
                f"{ref.rel_path}::{ref.symbol}" for ref in missing_refs
            )
            residues.append(
                KernelVmResiduePayload(
                    residue_id=residue_id,
                    binding_id=binding_spec.binding_id,
                    fragment_id=binding_spec.fragment_id,
                    residue_kind=capability.spec.residue_kind,
                    severity=capability.spec.severity,
                    score=capability.spec.score,
                    title=binding_spec.title,
                    message=(
                        f"{binding_spec.title} lacks {capability.spec.description}; "
                        f"missing evidence: {missing_evidence_text or '<none>'}"
                    ),
                    missing_capability_ids=[capability.spec.capability_id],
                    kernel_terms=list(binding_spec.kernel_terms),
                    runtime_surface_symbols=list(binding_spec.runtime_surface_symbols),
                    realizer_symbols=list(binding_spec.realizer_symbols),
                    runtime_object_symbols=list(binding_spec.runtime_object_symbols),
                    evidence_paths=list(evidence_paths),
                )
            )
        evidence_paths = tuple(
            dict.fromkeys(
                ref.rel_path
                for capability in capability_results
                for ref in (*capability.matched_refs, *capability.missing_refs)
            )
        )
        binding_status = _binding_status(capability_results)
        bindings.append(
            KernelVmBindingPayload(
                binding_id=binding_spec.binding_id,
                fragment_id=binding_spec.fragment_id,
                title=binding_spec.title,
                status=binding_status,
                summary=(
                    f"{binding_spec.title} "
                    f"capabilities="
                    f"{sum(1 for item in capability_results if item.status == 'pass')}/"
                    f"{len(capability_results)}"
                ),
                kernel_terms=list(binding_spec.kernel_terms),
                runtime_surface_symbols=list(binding_spec.runtime_surface_symbols),
                realizer_symbols=list(binding_spec.realizer_symbols),
                runtime_object_symbols=list(binding_spec.runtime_object_symbols),
                missing_capability_ids=list(missing_capability_ids),
                residue_ids=list(residue_ids),
                evidence_paths=list(evidence_paths),
                capabilities=[
                    _capability_payload(item) for item in capability_results
                ],
            )
        )

    pass_count = sum(1 for item in bindings if item["status"] == "pass")
    partial_count = sum(1 for item in bindings if item["status"] == "partial")
    fail_count = sum(1 for item in bindings if item["status"] == "fail")
    return KernelVmAlignmentArtifactPayload(
        artifact_kind=_ARTIFACT_KIND,
        schema_version=_SCHEMA_VERSION,
        generated_by=_GENERATED_BY,
        fragment_id=_FRAGMENT_ID,
        summary={
            "binding_count": len(bindings),
            "pass_count": pass_count,
            "partial_count": partial_count,
            "fail_count": fail_count,
            "residue_count": len(residues),
            "runtime_object_image_gap_count": sum(
                1
                for item in residues
                if item["residue_kind"] == "missing_runtime_object_image"
            ),
            "runtime_realizer_gap_count": sum(
                1
                for item in residues
                if item["residue_kind"] == "missing_runtime_realizer"
            ),
            "law_source_gap_count": sum(
                1 for item in residues if item["residue_kind"] == "missing_law_source"
            ),
        },
        bindings=_sorted(bindings, key=lambda item: item["binding_id"]),
        residues=_sorted(residues, key=lambda item: item["residue_id"]),
    )


def write_kernel_vm_alignment_artifact(*, path: Path, root: Path) -> Path:
    payload = build_kernel_vm_alignment_artifact_payload(root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


__all__ = [
    "KernelVmAlignmentArtifactPayload",
    "KernelVmBindingPayload",
    "KernelVmCapabilityPayload",
    "KernelVmEvidenceRefPayload",
    "KernelVmResiduePayload",
    "build_kernel_vm_alignment_artifact_payload",
    "write_kernel_vm_alignment_artifact",
]
