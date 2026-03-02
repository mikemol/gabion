# gabion:decision_protocol_module
from __future__ import annotations

"""Dataflow analysis contracts owned outside the runtime monolith.

This module centralizes shared carrier types consumed across analysis/runtime
boundaries so call sites do not import directly from the legacy monolith.
"""

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from gabion.analysis.aspf import Forest
from gabion.analysis.json_types import JSONObject
from gabion.analysis.semantic_primitives import (
    CallArgumentMapping,
    CallableId,
    ParameterId,
)
from gabion.analysis.type_fingerprints import (
    Fingerprint,
    PrimeRegistry,
    SynthRegistry,
    TypeConstructorRegistry,
)
from gabion.order_contract import sort_once

from .deprecated_substrate import DeprecatedFiber
from .forest_spec import ForestSpec
from .timeout_context import check_deadline

OptionalPath = Path | None
OptionalString = str | None
OptionalStringSet = set[str] | None
OptionalFloat = float | None
OptionalJsonObject = dict[str, object] | None
OptionalPrimeRegistry = PrimeRegistry | None
OptionalTypeConstructorRegistry = TypeConstructorRegistry | None
OptionalSynthRegistry = SynthRegistry | None
OptionalForestSpec = ForestSpec | None
ParamAnnotationMap = dict[str, str | None]
OptionalSpan4 = tuple[int, int, int, int] | None
FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


@dataclass
class ParamUse:
    direct_forward: set[tuple[str, str]]
    non_forward: bool
    current_aliases: set[str]
    forward_sites: dict[tuple[str, str], set[tuple[int, int, int, int]]] = field(
        default_factory=dict
    )
    unknown_key_carrier: bool = False
    unknown_key_sites: set[tuple[int, int, int, int]] = field(default_factory=set)


@dataclass(frozen=True)
class CallArgs:
    callee: str
    pos_map: dict[str, str]
    kw_map: dict[str, str]
    const_pos: dict[str, str]
    const_kw: dict[str, str]
    non_const_pos: set[str]
    non_const_kw: set[str]
    star_pos: list[tuple[int, str]]
    star_kw: list[str]
    is_test: bool
    span: OptionalSpan4 = None
    callable_kind: str = "function"
    callable_source: str = "symbol"

    def callable_id(self) -> CallableId:
        return CallableId.from_raw(self.callee)

    def argument_mapping(self) -> CallArgumentMapping:
        positional = {
            int(idx): ParameterId.from_raw(param) for idx, param in self.pos_map.items()
        }
        keywords = {
            key: ParameterId.from_raw(param) for key, param in self.kw_map.items()
        }
        return CallArgumentMapping(
            positional=positional,
            keywords=keywords,
            star_positional=tuple(
                (idx, ParameterId.from_raw(param)) for idx, param in self.star_pos
            ),
            star_keywords=tuple(ParameterId.from_raw(param) for param in self.star_kw),
        )


@dataclass(frozen=True)
class InvariantProposition:
    form: str
    terms: tuple[str, ...]
    scope: OptionalString = None
    source: OptionalString = None
    invariant_id: OptionalString = None
    confidence: OptionalFloat = None
    evidence_keys: tuple[str, ...] = ()

    def as_dict(self) -> JSONObject:
        payload: JSONObject = {"form": self.form, "terms": list(self.terms)}
        if self.scope is not None:
            payload["scope"] = self.scope
        if self.source is not None:
            payload["source"] = self.source
        if self.invariant_id is not None:
            payload["invariant_id"] = self.invariant_id
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        if self.evidence_keys:
            payload["evidence_keys"] = list(self.evidence_keys)
        return payload


@dataclass
class SymbolTable:
    imports: dict[tuple[str, str], str] = field(default_factory=dict)
    internal_roots: set[str] = field(default_factory=set)
    external_filter: bool = True
    star_imports: dict[str, set[str]] = field(default_factory=dict)
    module_exports: dict[str, set[str]] = field(default_factory=dict)
    module_export_map: dict[str, dict[str, str]] = field(default_factory=dict)

    def resolve(self, current_module: str, name: str) -> OptionalString:
        if (current_module, name) in self.imports:
            fqn = self.imports[(current_module, name)]
            if self.external_filter:
                root = fqn.split(".")[0]
                if root not in self.internal_roots:
                    return None
            return fqn
        return f"{current_module}.{name}"

    def resolve_star(self, current_module: str, name: str) -> OptionalString:
        check_deadline()
        candidates = self.star_imports.get(current_module, set())
        if not candidates:
            return None
        for module in sort_once(
            candidates,
            source="dataflow_contracts.SymbolTable.resolve_star.candidates",
        ):
            check_deadline()
            exports = self.module_exports.get(module)
            if exports is not None and name in exports:
                export_map = self.module_export_map.get(module, {})
                mapped = export_map.get(name)
                if mapped:
                    if self.external_filter:
                        root = mapped.split(".")[0]
                        if root in self.internal_roots:
                            return mapped
                    else:
                        return mapped
                resolved = f"{module}.{name}".strip(".")
                if not module:
                    return resolved
                if self.external_filter:
                    root = module.split(".")[0]
                    if root in self.internal_roots:
                        return resolved
                    continue
                return resolved
        return None


@dataclass
class FunctionInfo:
    name: str
    qual: str
    path: Path
    params: list[str]
    annots: ParamAnnotationMap
    calls: list[CallArgs]
    unused_params: set[str]
    unknown_key_carriers: set[str] = field(default_factory=set)
    defaults: set[str] = field(default_factory=set)
    transparent: bool = True
    class_name: OptionalString = None
    scope: tuple[str, ...] = ()
    lexical_scope: tuple[str, ...] = ()
    decision_params: set[str] = field(default_factory=set)
    decision_surface_reasons: dict[str, set[str]] = field(default_factory=dict)
    value_decision_params: set[str] = field(default_factory=set)
    value_decision_reasons: set[str] = field(default_factory=set)
    positional_params: tuple[str, ...] = ()
    kwonly_params: tuple[str, ...] = ()
    vararg: OptionalString = None
    kwarg: OptionalString = None
    param_spans: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)
    function_span: OptionalSpan4 = None
    local_lambda_bindings: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass
class ClassInfo:
    qual: str
    module: str
    bases: list[str]
    methods: set[str]


@dataclass
class AuditConfig:
    project_root: OptionalPath = None
    exclude_dirs: set[str] = field(default_factory=set)
    ignore_params: set[str] = field(default_factory=set)
    decision_ignore_params: set[str] = field(default_factory=set)
    external_filter: bool = True
    strictness: str = "high"
    transparent_decorators: OptionalStringSet = None
    decision_tiers: dict[str, int] = field(default_factory=dict)
    decision_require_tiers: bool = False
    never_exceptions: set[str] = field(default_factory=set)
    deadline_roots: set[str] = field(default_factory=set)
    fingerprint_registry: OptionalPrimeRegistry = None
    fingerprint_index: dict[Fingerprint, set[str]] = field(default_factory=dict)
    constructor_registry: OptionalTypeConstructorRegistry = None
    fingerprint_seed_revision: OptionalString = None
    fingerprint_synth_min_occurrences: int = 0
    fingerprint_synth_version: str = "synth@1"
    fingerprint_synth_registry: OptionalSynthRegistry = None
    invariant_emitters: tuple[
        Callable[[ast.FunctionDef], object],
        ...,
    ] = field(default_factory=tuple)
    adapter_contract: OptionalJsonObject = None
    required_analysis_surfaces: set[str] = field(default_factory=set)

    def is_ignored_path(self, path: Path) -> bool:
        parts = set(path.parts)
        return bool(self.exclude_dirs & parts)


@dataclass
class AnalysisResult:
    groups_by_path: dict[Path, dict[str, list[set[str]]]]
    param_spans_by_path: dict[Path, dict[str, dict[str, tuple[int, int, int, int]]]]
    bundle_sites_by_path: dict[Path, dict[str, list[list[JSONObject]]]]
    type_suggestions: list[str]
    type_ambiguities: list[str]
    type_callsite_evidence: list[str]
    constant_smells: list[str]
    unused_arg_smells: list[str]
    forest: Forest
    lint_lines: list[str] = field(default_factory=list)
    deadness_witnesses: list[JSONObject] = field(default_factory=list)
    coherence_witnesses: list[JSONObject] = field(default_factory=list)
    rewrite_plans: list[JSONObject] = field(default_factory=list)
    exception_obligations: list[JSONObject] = field(default_factory=list)
    never_invariants: list[JSONObject] = field(default_factory=list)
    handledness_witnesses: list[JSONObject] = field(default_factory=list)
    decision_surfaces: list[str] = field(default_factory=list)
    value_decision_surfaces: list[str] = field(default_factory=list)
    decision_warnings: list[str] = field(default_factory=list)
    fingerprint_warnings: list[str] = field(default_factory=list)
    fingerprint_matches: list[str] = field(default_factory=list)
    fingerprint_synth: list[str] = field(default_factory=list)
    fingerprint_synth_registry: OptionalJsonObject = None
    fingerprint_provenance: list[JSONObject] = field(default_factory=list)
    context_suggestions: list[str] = field(default_factory=list)
    invariant_propositions: list[InvariantProposition] = field(default_factory=list)
    value_decision_rewrites: list[str] = field(default_factory=list)
    ambiguity_witnesses: list[JSONObject] = field(default_factory=list)
    deadline_obligations: list[JSONObject] = field(default_factory=list)
    parse_failure_witnesses: list[JSONObject] = field(default_factory=list)
    forest_spec: OptionalForestSpec = None
    profiling_v1: OptionalJsonObject = None
    deprecated_artifacts: object = None
    deprecated_fibers: list[DeprecatedFiber] = field(default_factory=list)
    unsupported_by_adapter: list[JSONObject] = field(default_factory=list)


@dataclass
class ReportCarrier:
    forest: Forest
    bundle_sites_by_path: dict[Path, dict[str, list[list[JSONObject]]]] = field(
        default_factory=dict
    )
    type_suggestions: list[str] = field(default_factory=list)
    type_ambiguities: list[str] = field(default_factory=list)
    type_callsite_evidence: list[str] = field(default_factory=list)
    constant_smells: list[str] = field(default_factory=list)
    unused_arg_smells: list[str] = field(default_factory=list)
    deadness_witnesses: list[JSONObject] = field(default_factory=list)
    coherence_witnesses: list[JSONObject] = field(default_factory=list)
    rewrite_plans: list[JSONObject] = field(default_factory=list)
    exception_obligations: list[JSONObject] = field(default_factory=list)
    never_invariants: list[JSONObject] = field(default_factory=list)
    ambiguity_witnesses: list[JSONObject] = field(default_factory=list)
    handledness_witnesses: list[JSONObject] = field(default_factory=list)
    decision_surfaces: list[str] = field(default_factory=list)
    value_decision_surfaces: list[str] = field(default_factory=list)
    decision_warnings: list[str] = field(default_factory=list)
    fingerprint_warnings: list[str] = field(default_factory=list)
    fingerprint_matches: list[str] = field(default_factory=list)
    fingerprint_synth: list[str] = field(default_factory=list)
    fingerprint_provenance: list[JSONObject] = field(default_factory=list)
    context_suggestions: list[str] = field(default_factory=list)
    invariant_propositions: list[InvariantProposition] = field(default_factory=list)
    value_decision_rewrites: list[str] = field(default_factory=list)
    deadline_obligations: list[JSONObject] = field(default_factory=list)
    parse_failure_witnesses: list[JSONObject] = field(default_factory=list)
    resumability_obligations: list[JSONObject] = field(default_factory=list)
    incremental_report_obligations: list[JSONObject] = field(default_factory=list)
    unsupported_by_adapter: list[JSONObject] = field(default_factory=list)
    progress_marker: str = ""
    phase_progress_v2: OptionalJsonObject = None
    deprecated_signals: tuple[str, ...] = ()

    @classmethod
    def from_analysis_result(
        cls,
        analysis: AnalysisResult,
        *,
        include_type_audit: bool = True,
    ) -> "ReportCarrier":
        return cls(
            forest=analysis.forest,
            bundle_sites_by_path=analysis.bundle_sites_by_path,
            type_suggestions=analysis.type_suggestions if include_type_audit else [],
            type_ambiguities=analysis.type_ambiguities if include_type_audit else [],
            type_callsite_evidence=(
                analysis.type_callsite_evidence if include_type_audit else []
            ),
            constant_smells=analysis.constant_smells,
            unused_arg_smells=analysis.unused_arg_smells,
            deadness_witnesses=analysis.deadness_witnesses,
            coherence_witnesses=analysis.coherence_witnesses,
            rewrite_plans=analysis.rewrite_plans,
            exception_obligations=analysis.exception_obligations,
            never_invariants=analysis.never_invariants,
            ambiguity_witnesses=analysis.ambiguity_witnesses,
            handledness_witnesses=analysis.handledness_witnesses,
            decision_surfaces=analysis.decision_surfaces,
            value_decision_surfaces=analysis.value_decision_surfaces,
            decision_warnings=analysis.decision_warnings,
            fingerprint_warnings=analysis.fingerprint_warnings,
            fingerprint_matches=analysis.fingerprint_matches,
            fingerprint_synth=analysis.fingerprint_synth,
            fingerprint_provenance=analysis.fingerprint_provenance,
            context_suggestions=analysis.context_suggestions,
            invariant_propositions=analysis.invariant_propositions,
            value_decision_rewrites=analysis.value_decision_rewrites,
            deadline_obligations=analysis.deadline_obligations,
            parse_failure_witnesses=analysis.parse_failure_witnesses,
            unsupported_by_adapter=analysis.unsupported_by_adapter,
            deprecated_signals=(
                analysis.deprecated_artifacts.informational_signals
                if analysis.deprecated_artifacts is not None
                else ()
            ),
        )

__all__ = [
    "AnalysisResult",
    "AuditConfig",
    "CallArgs",
    "ClassInfo",
    "FunctionInfo",
    "InvariantProposition",
    "ParamUse",
    "ReportCarrier",
    "SymbolTable",
]
