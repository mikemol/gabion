#!/usr/bin/env python3
# gabion:decision_protocol_module
# gabion:boundary_normalization_module
"""Infer forwarding-based parameter bundles and propagate them across calls.

This script performs a two-stage analysis:
  1) Local grouping: within a function, parameters used *only* as direct
     call arguments are grouped by identical forwarding signatures.
  2) Propagation: if a function f calls g, and g has local bundles, then
     f's parameters passed into g's bundled positions are linked as a
     candidate bundle. This is iterated to a fixed point.

The goal is to surface "dataflow grammar" candidates for config dataclasses.

It can also emit a DOT graph (see --dot) so downstream tooling can render
bundle candidates as a dependency graph.
"""
from __future__ import annotations

import argparse
import ast
import json
import hashlib
import os
import sys
import time
from collections import Counter, defaultdict, deque
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field, replace
from enum import StrEnum
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import Callable, Generic, Hashable, Iterable, Iterator, Literal, Mapping, Sequence, TypeVar, cast
import re
from gabion.analysis.pattern_schema import (
    PatternAxis,
    PatternInstance,
    PatternResidue,
    PatternSchema,
    mismatch_residue_payload,
)

from gabion.analysis.visitors import ImportVisitor, ParentAnnotator, UseVisitor
from gabion.analysis.evidence import (
    Site,
    exception_obligation_summary_for_site,
    normalize_bundle_key,
)
from gabion.analysis.json_types import JSONObject, JSONValue
from gabion.analysis.schema_audit import find_anonymous_schema_surfaces
from gabion.analysis.aspf import Alt, Forest, Node, NodeId
from gabion.analysis.derivation_cache import get_global_derivation_cache
from gabion.analysis.derivation_contract import DerivationOp
from gabion.analysis import evidence_keys
from gabion.invariants import never, require_not_none
from gabion.order_contract import OrderPolicy, sort_once
from gabion.config import (
    dataflow_defaults,
    dataflow_deadline_roots,
    decision_defaults,
    decision_ignore_list,
    decision_require_tiers,
    decision_tier_map,
    exception_defaults,
    exception_never_list,
    fingerprint_defaults,
    merge_payload,
    synthesis_defaults,
)
from gabion.analysis.type_fingerprints import (
    Fingerprint,
    FingerprintDimension,
    PrimeRegistry,
    TypeConstructorRegistry,
    _collect_base_atoms,
    _collect_constructors,
    SynthRegistry,
    build_synth_registry,
    build_fingerprint_registry,
    build_synth_registry_from_payload,
    bundle_fingerprint_dimensional,
    format_fingerprint,
    fingerprint_carrier_soundness,
    fingerprint_identity_payload,
    synth_registry_payload,
)
from .forest_signature import (
    build_forest_signature,
    build_forest_signature_from_groups,
)
from .forest_spec import (
    ForestSpec,
    build_forest_spec,
    default_forest_spec,
    forest_spec_metadata,
)
from .timeout_context import (
    Deadline,
    GasMeter,
    TimeoutExceeded,
    build_timeout_context_from_stack,
    check_deadline,
    deadline_loop_iter,
    deadline_clock_scope,
    deadline_scope,
    forest_scope,
    reset_forest,
    set_forest,
)
from .projection_exec import apply_spec
from .projection_normalize import spec_hash as projection_spec_hash
from .baseline_io import load_json
from .decision_flow import (
    build_decision_tables,
    detect_repeated_guard_bundles,
    enforce_decision_protocol_contracts,
)
from .resume_codec import (
    allowed_path_lookup,
    int_str_pairs_from_sequence,
    int_tuple4_or_none,
    iter_valid_key_entries,
    load_resume_map,
    load_allowed_paths_from_sequence,
    mapping_payload,
    mapping_sections,
    mapping_or_empty,
    mapping_or_none,
    payload_with_format,
    payload_with_phase,
    sequence_or_none,
    str_list_from_sequence,
    str_map_from_mapping,
    str_pair_set_from_sequence,
    str_set_from_sequence,
    str_tuple_from_sequence,
)
from .projection_registry import (
    AMBIGUITY_SUMMARY_SPEC,
    AMBIGUITY_SUITE_AGG_SPEC,
    AMBIGUITY_VIRTUAL_SET_SPEC,
    DEADLINE_OBLIGATIONS_SUMMARY_SPEC,
    LINT_FINDINGS_SPEC,
    NEVER_INVARIANTS_SPEC,
    REPORT_SECTION_LINES_SPEC,
    SUITE_ORDER_SPEC,
    WL_REFINEMENT_SPEC,
    spec_metadata_lines_from_payload,
    spec_metadata_payload,
)
from .wl_refinement import emit_wl_refinement_facets
from .aspf_core import parse_2cell_witness
from .deprecated_substrate import (
    DeprecatedExtractionArtifacts,
    DeprecatedFiber,
    detect_report_section_extinction,
)
from .structure_reuse_classes import build_structure_class, structure_class_payload
from .aspf_decision_surface import classify_drift_by_homotopy
from .dataflow_decision_surfaces import (
    compute_fingerprint_coherence as _ds_compute_fingerprint_coherence,
    compute_fingerprint_rewrite_plans as _ds_compute_fingerprint_rewrite_plans,
    extract_smell_sample as _ds_extract_smell_sample,
    lint_lines_from_bundle_evidence as _ds_lint_lines_from_bundle_evidence,
    lint_lines_from_constant_smells as _ds_lint_lines_from_constant_smells,
    lint_lines_from_type_evidence as _ds_lint_lines_from_type_evidence,
    lint_lines_from_unused_arg_smells as _ds_lint_lines_from_unused_arg_smells,
    parse_lint_location as _ds_parse_lint_location,
    summarize_coherence_witnesses as _ds_summarize_coherence_witnesses,
    summarize_deadness_witnesses as _ds_summarize_deadness_witnesses,
    summarize_rewrite_plans as _ds_summarize_rewrite_plans,
)
from .dataflow_exception_obligations import (
    exception_handler_compatibility as _exc_exception_handler_compatibility,
    exception_param_names as _exc_exception_param_names,
    handler_type_names as _exc_handler_type_names,
    exception_type_name as _exc_exception_type_name,
    handler_is_broad as _exc_handler_is_broad,
    handler_label as _exc_handler_label,
    node_in_try_body as _exc_node_in_try_body,
    _builtin_exception_class as _exc_builtin_exception_class,
)
from .dataflow_report_rendering import (
    render_synthesis_section as _report_render_synthesis_section,
)
from .dataflow_bundle_iteration import (
    iter_dataclass_call_bundle_effects as _iter_dataclass_call_bundle_effects_impl,
)
from .dataflow_callee_resolution import (
    CalleeResolutionContext as _CalleeResolutionContextCore,
    collect_callee_resolution_effects as _collect_callee_resolution_effects_impl,
    resolve_callee_with_effects as _resolve_callee_with_effects_impl,
)
from .dataflow_run_outputs import (
    DataflowRunOutputContext as _RunImplOutputContextCore,
    finalize_run_outputs as _finalize_run_outputs_impl,
)
from gabion.schema import SynthesisResponse
from gabion.refactor.rewrite_plan import rewrite_plan_schema, validate_rewrite_plan_payload
from gabion.synthesis import NamingContext, SynthesisConfig, Synthesizer
from gabion.synthesis.emission import render_protocol_stubs as _render_protocol_stubs
from gabion.synthesis.merge import merge_bundles
from gabion.synthesis.schedule import topological_schedule


_AST_UNPARSE_ERROR_TYPES = (
    AttributeError,
    TypeError,
    ValueError,
    RecursionError,
)
_LITERAL_EVAL_ERROR_TYPES = (
    SyntaxError,
    ValueError,
    TypeError,
    MemoryError,
    RecursionError,
)
_PARSE_MODULE_ERROR_TYPES = (
    OSError,
    UnicodeError,
    SyntaxError,
    ValueError,
    TypeError,
    MemoryError,
    RecursionError,
)

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef
OptionalIgnoredParams = set[str] | None
ParamAnnotationMap = dict[str, str | None]
ReturnAliasMap = dict[str, tuple[list[str], list[str]]]
OptionalReturnAliasMap = ReturnAliasMap | None
OptionalClassName = str | None
Span4 = tuple[int, int, int, int]
OptionalSpan4 = Span4 | None
OptionalString = str | None
OptionalFloat = float | None
OptionalPath = Path | None
OptionalStringSet = set[str] | None
OptionalPrimeRegistry = PrimeRegistry | None
OptionalTypeConstructorRegistry = TypeConstructorRegistry | None
OptionalSynthRegistry = SynthRegistry | None
OptionalJsonObject = JSONObject | None
OptionalForestSpec = ForestSpec | None
OptionalDeprecatedExtractionArtifacts = DeprecatedExtractionArtifacts | None
OptionalFunctionNode = FunctionNode | None
OptionalJsonObjectList = list[JSONObject] | None
OptionalAstNode = ast.AST | None
OptionalAstCall = ast.Call | None
NodeIdOrNone = NodeId | None
ParseCacheValue = ast.Module | BaseException

_FORBID_RAW_SORTED_ENV = "GABION_FORBID_RAW_SORTED"
_RAW_SORTED_BASELINE_COUNTS: dict[str, int] = {
    "src/gabion/analysis/ambiguity_delta.py": 2,
    "src/gabion/analysis/aspf.py": 3,
    "src/gabion/analysis/call_cluster_consolidation.py": 3,
    "src/gabion/analysis/call_clusters.py": 1,
    "src/gabion/analysis/dataflow_audit.py": 146,
    "src/gabion/analysis/evidence.py": 2,
    "src/gabion/analysis/evidence_keys.py": 2,
    "src/gabion/analysis/forest_signature.py": 4,
    "src/gabion/analysis/forest_spec.py": 5,
    "src/gabion/analysis/schema_audit.py": 1,
    "src/gabion/analysis/test_annotation_drift_delta.py": 2,
    "src/gabion/analysis/test_evidence.py": 6,
    "src/gabion/analysis/test_evidence_suggestions.py": 14,
    "src/gabion/analysis/test_obsolescence.py": 6,
    "src/gabion/analysis/test_obsolescence_delta.py": 8,
    "src/gabion/analysis/timeout_context.py": 5,
    "src/gabion/analysis/type_fingerprints.py": 0,
    "src/gabion/lsp_client.py": 1,
    "src/gabion/order_contract.py": 2,
    "src/gabion/refactor/engine.py": 1,
    "src/gabion/server.py": 13,
    "src/gabion/synthesis/merge.py": 2,
    "src/gabion/synthesis/naming.py": 1,
    "src/gabion/synthesis/protocols.py": 1,
    "src/gabion/synthesis/schedule.py": 2,
}


class _ParseModuleStage(StrEnum):
    PARAM_ANNOTATIONS = "param_annotations"
    DEADLINE_FUNCTION_FACTS = "deadline_function_facts"
    CALL_NODES = "call_nodes"
    SUITE_CONTAINMENT = "suite_containment"
    SYMBOL_TABLE = "symbol_table"
    CLASS_INDEX = "class_index"
    FUNCTION_INDEX = "function_index"
    CONFIG_FIELDS = "config_fields"
    DATACLASS_REGISTRY = "dataclass_registry"
    DATACLASS_CALL_BUNDLES = "dataclass_call_bundles"
    RAW_SORTED_AUDIT = "raw_sorted_audit"


ReportProjectionPhase = Literal["collection", "forest", "edge", "post"]


@dataclass(frozen=True)
class _PhaseWorkProgress:
    work_done: int
    work_total: int


def _phase_work_progress(*, work_done: int, work_total: int) -> _PhaseWorkProgress:
    check_deadline()
    normalized_total = max(int(work_total), 0)
    normalized_done = max(int(work_done), 0)
    if normalized_total:
        normalized_done = min(normalized_done, normalized_total)
    return _PhaseWorkProgress(work_done=normalized_done, work_total=normalized_total)


@dataclass(frozen=True)
class _FunctionSuiteKey:
    # dataflow-bundle: path, qual
    path: str
    qual: str


@dataclass(frozen=True)
class _ReportSectionKey:
    # dataflow-bundle: run_id, section
    run_id: str
    section: str


@dataclass(frozen=True)
class _ExecutionPatternMatch:
    pattern_id: str
    kind: str
    members: tuple[str, ...]
    suggestion: str


@dataclass(frozen=True)
class _ExecutionPatternRule:
    pattern_id: str
    kind: str
    description: str


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
        payload: JSONObject = {
            "form": self.form,
            "terms": list(self.terms),
        }
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


def _invariant_digest(payload: Mapping[str, object], *, prefix: str) -> str:
    encoded = json.dumps(payload, sort_keys=False, separators=(",", ":")).encode("utf-8")
    digest = hashlib.blake2s(encoded, digest_size=12).hexdigest()
    return f"{prefix}:{digest}"


def _invariant_confidence(value: OptionalFloat) -> float:
    if value is None:
        return 1.0
    return max(0.0, min(1.0, float(value)))


def _compute_invariant_id(
    *,
    form: str,
    terms: tuple[str, ...],
    scope: str,
    source: str,
) -> str:
    payload = {
        "form": form,
        "terms": list(terms),
        "scope": scope,
        "source": source,
    }
    return _invariant_digest(payload, prefix="inv")


def _compute_invariant_evidence_key(
    *,
    invariant_id: str,
    form: str,
    terms: tuple[str, ...],
    scope: str,
) -> str:
    term_display = ",".join(terms)
    return f"E:invariant::{scope}::{form}::{term_display}::{invariant_id}"


def _normalize_invariant_proposition(
    proposition: InvariantProposition,
    *,
    default_scope: str,
    default_source: str,
) -> InvariantProposition:
    scope = proposition.scope or default_scope
    source = proposition.source or default_source
    invariant_id = proposition.invariant_id or _compute_invariant_id(
        form=proposition.form,
        terms=proposition.terms,
        scope=scope,
        source=source,
    )
    evidence_keys = proposition.evidence_keys or (
        _compute_invariant_evidence_key(
            invariant_id=invariant_id,
            form=proposition.form,
            terms=proposition.terms,
            scope=scope,
        ),
    )
    return InvariantProposition(
        form=proposition.form,
        terms=proposition.terms,
        scope=scope,
        source=source,
        invariant_id=invariant_id,
        confidence=_invariant_confidence(proposition.confidence),
        evidence_keys=tuple(str(key) for key in evidence_keys),
    )

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
            source="SymbolTable.resolve_star.candidates",
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
        Callable[[ast.FunctionDef], Iterable[InvariantProposition]],
        ...,
    ] = field(default_factory=tuple)

    def is_ignored_path(self, path: Path) -> bool:
        parts = set(path.parts)
        return bool(self.exclude_dirs & parts)


def _call_context(node: ast.AST, parents: dict[ast.AST, ast.AST]):
    check_deadline()
    child = node
    parent = parents.get(child)
    while parent is not None:
        check_deadline()
        if type(parent) is ast.Call:
            call_parent = cast(ast.Call, parent)
            if child in call_parent.args:
                return call_parent, True
            for kw in call_parent.keywords:
                check_deadline()
                if child is kw or child is kw.value:
                    return call_parent, True
            return call_parent, False
        child = parent
        parent = parents.get(child)
    return None, False


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
    deprecated_artifacts: OptionalDeprecatedExtractionArtifacts = None
    deprecated_fibers: list[DeprecatedFiber] = field(default_factory=list)


_ANALYSIS_PROFILING_FORMAT_VERSION = 1


def _profiling_v1_payload(*, stage_ns: Mapping[str, int], counters: Mapping[str, int]) -> JSONObject:
    return {
        "format_version": _ANALYSIS_PROFILING_FORMAT_VERSION,
        "stage_ns": {str(key): int(stage_ns[key]) for key in stage_ns},
        "counters": {str(key): int(counters[key]) for key in counters},
    }


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
            deprecated_signals=(
                analysis.deprecated_artifacts.informational_signals
                if analysis.deprecated_artifacts is not None
                else ()
            ),
        )


_ReportSectionValue = TypeVar("_ReportSectionValue")


@dataclass(frozen=True)
class ReportProjectionSpec(Generic[_ReportSectionValue]):
    section_id: str
    phase: ReportProjectionPhase
    deps: tuple[str, ...]
    build: Callable[
        [ReportCarrier, dict[Path, dict[str, list[set[str]]]]],
        _ReportSectionValue,
    ]
    render: Callable[[_ReportSectionValue], list[str]]
    violation_extract: Callable[[_ReportSectionValue], list[str]]
    preview_build: object = None


def _report_section_identity_render(lines: list[str]) -> list[str]:
    return lines


def _report_section_no_violations(_lines: list[str]) -> list[str]:
    return []


def _preview_components_section(
    report: ReportCarrier,
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
) -> list[str]:
    check_deadline()
    path_count = len(groups_by_path)
    function_count = sum(len(groups) for groups in groups_by_path.values())
    bundle_alternatives = 0
    for groups in groups_by_path.values():
        check_deadline()
        for bundles in groups.values():
            check_deadline()
            bundle_alternatives += len(bundles)
    lines = [
        "Component preview (provisional).",
        f"- `paths_with_groups`: `{path_count}`",
        f"- `functions_with_groups`: `{function_count}`",
        f"- `bundle_alternatives`: `{bundle_alternatives}`",
    ]
    if report.forest.alts:
        # Keep preview cheap under tight timeout budgets; full component
        # materialization is emitted in forest/post projection.
        lines.append(f"- `forest_alternatives`: `{len(report.forest.alts)}`")
        lines.append("- `component_count`: `deferred_until_post_phase`")
    return lines


def _known_violation_lines(
    report: ReportCarrier,
) -> list[str]:
    check_deadline()
    lines: list[str] = []
    lines.extend(_runtime_obligation_violation_lines(report.resumability_obligations))
    lines.extend(
        _runtime_obligation_violation_lines(report.incremental_report_obligations)
    )
    lines.extend(_parse_failure_violation_lines(report.parse_failure_witnesses))
    lines.extend(report.decision_warnings)
    seen: set[str] = set()
    unique: list[str] = []
    for line in lines:
        check_deadline()
        if line in seen:
            continue
        seen.add(line)
        unique.append(line)
    return unique


def _preview_violations_section(
    report: ReportCarrier,
    _groups_by_path: dict[Path, dict[str, list[set[str]]]],
) -> list[str]:
    check_deadline()
    known = _known_violation_lines(report)
    lines = [
        "Violations preview (provisional).",
        f"- `known_violations`: `{len(known)}`",
    ]
    if not known:
        lines.append("- none observed yet")
        return lines
    lines.append("Top known violations:")
    for line in known[:10]:
        check_deadline()
        lines.append(f"- {line}")
    return lines


def _preview_type_flow_section(
    report: ReportCarrier,
    _groups_by_path: dict[Path, dict[str, list[set[str]]]],
) -> list[str]:
    check_deadline()
    lines = [
        "Type-flow audit preview (provisional).",
        f"- `type_suggestions`: `{len(report.type_suggestions)}`",
        f"- `type_ambiguities`: `{len(report.type_ambiguities)}`",
        f"- `type_callsite_evidence`: `{len(report.type_callsite_evidence)}`",
    ]
    if report.type_ambiguities:
        lines.append(f"- `sample_type_ambiguity`: `{report.type_ambiguities[0]}`")
    return lines


def _preview_deadline_summary_section(
    report: ReportCarrier,
    _groups_by_path: dict[Path, dict[str, list[set[str]]]],
) -> list[str]:
    check_deadline()
    if not report.deadline_obligations:
        return [
            "Deadline propagation preview (provisional).",
            "- no deadline obligations yet",
        ]
    summary = _summarize_deadline_obligations(
        report.deadline_obligations,
        forest=report.forest,
    )
    lines = ["Deadline propagation preview (provisional)."]
    lines.extend(summary[:20])
    return lines


def _make_list_section_preview(
    *,
    title: str,
    count_label: str,
    values: Callable[[ReportCarrier], Sequence[str]],
    sample_label = None,
    extra_count_labels: tuple[tuple[str, Callable[[ReportCarrier], int]], ...] = (),
) -> Callable[[ReportCarrier, dict[Path, dict[str, list[set[str]]]]], list[str]]:
    def _preview(
        report: ReportCarrier,
        _groups_by_path: dict[Path, dict[str, list[set[str]]]],
    ) -> list[str]:
        check_deadline()
        series = values(report)
        lines = [
            f"{title} preview (provisional).",
            f"- `{count_label}`: `{len(series)}`",
        ]
        for label, getter in extra_count_labels:
            check_deadline()
            lines.append(f"- `{label}`: `{getter(report)}`")
        if sample_label and series:
            lines.append(f"- `{sample_label}`: `{series[0]}`")
        return lines

    return _preview


_preview_constant_smells_section = _make_list_section_preview(
    title="Constant-propagation smells",
    count_label="constant_smells",
    values=lambda report: report.constant_smells,
    sample_label="sample_constant_smell",
)

_preview_unused_arg_smells_section = _make_list_section_preview(
    title="Unused-argument smells",
    count_label="unused_arg_smells",
    values=lambda report: report.unused_arg_smells,
    sample_label="sample_unused_arg_smell",
)


def _preview_runtime_obligations_section(
    *,
    title: str,
    obligations: list[JSONObject],
) -> list[str]:
    check_deadline()
    violated = 0
    satisfied = 0
    pending = 0
    for entry in obligations:
        check_deadline()
        status = entry.get("status")
        if status == "VIOLATION":
            violated += 1
        elif status == "SATISFIED":
            satisfied += 1
        else:
            pending += 1
    lines = [
        f"{title} preview (provisional).",
        f"- `obligations`: `{len(obligations)}`",
        f"- `violations`: `{violated}`",
        f"- `satisfied`: `{satisfied}`",
        f"- `pending`: `{pending}`",
    ]
    for entry in obligations:
        check_deadline()
        if entry.get("status") != "VIOLATION":
            continue
        contract = str(entry.get("contract", "runtime_contract"))
        kind = str(entry.get("kind", "unknown"))
        detail = str(entry.get("detail", ""))
        lines.append(f"- `sample_violation`: `{contract}/{kind} {detail}`")
        break
    return lines


def _preview_resumability_obligations_section(
    report: ReportCarrier,
    _groups_by_path: dict[Path, dict[str, list[set[str]]]],
) -> list[str]:
    return _preview_runtime_obligations_section(
        title="Resumability obligations",
        obligations=report.resumability_obligations,
    )


def _preview_incremental_report_obligations_section(
    report: ReportCarrier,
    _groups_by_path: dict[Path, dict[str, list[set[str]]]],
) -> list[str]:
    return _preview_runtime_obligations_section(
        title="Incremental report obligations",
        obligations=report.incremental_report_obligations,
    )


def _preview_parse_failure_witnesses_section(
    report: ReportCarrier,
    _groups_by_path: dict[Path, dict[str, list[set[str]]]],
) -> list[str]:
    check_deadline()
    stage_counts: dict[str, int] = defaultdict(int)
    for witness in report.parse_failure_witnesses:
        check_deadline()
        stage = witness.get("stage")
        if type(stage) is str and stage:
            stage_counts[stage] += 1
            continue
        stage_counts["unknown"] += 1
    lines = [
        "Parse failure witnesses preview (provisional).",
        f"- `parse_failure_witnesses`: `{len(report.parse_failure_witnesses)}`",
    ]
    for stage, count in sort_once(
        stage_counts.items(),
        source="_preview_parse_failure_witnesses_section.stage_counts",
        key=lambda item: item[0],
    ):
        check_deadline()
        lines.append(f"- `stage[{stage}]`: `{count}`")
    return lines


def _preview_execution_pattern_suggestions_section(
    report: ReportCarrier,
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
) -> list[str]:
    check_deadline()
    function_count = sum(len(groups) for groups in groups_by_path.values())
    lines = [
        "Execution pattern opportunities preview (provisional).",
        f"- `paths_with_groups`: `{len(groups_by_path)}`",
        f"- `functions_with_groups`: `{function_count}`",
        f"- `decision_surfaces_seen`: `{len(report.decision_surfaces)}`",
        (
            "- `note`: `full execution-pattern synthesis is materialized in post-phase "
            "projection`"
        ),
    ]
    return lines


def _preview_pattern_schema_residue_section(
    report: ReportCarrier,
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
) -> list[str]:
    check_deadline()
    bundle_alternatives = 0
    for groups in groups_by_path.values():
        check_deadline()
        for bundles in groups.values():
            check_deadline()
            bundle_alternatives += len(bundles)
    lines = [
        "Pattern schema residue preview (provisional).",
        f"- `paths_with_groups`: `{len(groups_by_path)}`",
        f"- `bundle_alternatives`: `{bundle_alternatives}`",
        f"- `decision_surfaces_seen`: `{len(report.decision_surfaces)}`",
        f"- `value_decision_surfaces_seen`: `{len(report.value_decision_surfaces)}`",
    ]
    return lines


_preview_decision_surfaces_section = _make_list_section_preview(
    title="Decision surfaces",
    count_label="decision_surfaces",
    values=lambda report: report.decision_surfaces,
    sample_label="sample_decision_surface",
    extra_count_labels=(("decision_warnings", lambda report: len(report.decision_warnings)),),
)

_preview_value_decision_surfaces_section = _make_list_section_preview(
    title="Value-encoded decision surfaces",
    count_label="value_decision_surfaces",
    values=lambda report: report.value_decision_surfaces,
    sample_label="sample_value_decision_surface",
    extra_count_labels=(
        ("value_decision_rewrites", lambda report: len(report.value_decision_rewrites)),
    ),
)

_preview_fingerprint_warnings_section = _make_list_section_preview(
    title="Fingerprint warnings",
    count_label="fingerprint_warnings",
    values=lambda report: report.fingerprint_warnings,
    sample_label="sample_fingerprint_warning",
)

_preview_fingerprint_matches_section = _make_list_section_preview(
    title="Fingerprint matches",
    count_label="fingerprint_matches",
    values=lambda report: report.fingerprint_matches,
    sample_label="sample_fingerprint_match",
)

_preview_fingerprint_synthesis_section = _make_list_section_preview(
    title="Fingerprint synthesis",
    count_label="fingerprint_synth",
    values=lambda report: report.fingerprint_synth,
    sample_label="sample_fingerprint_synth",
    extra_count_labels=(
        ("fingerprint_provenance", lambda report: len(report.fingerprint_provenance)),
    ),
)

_preview_context_suggestions_section = _make_list_section_preview(
    title="Context suggestions",
    count_label="context_suggestions",
    values=lambda report: report.context_suggestions,
    sample_label="sample_context_suggestion",
)


def _preview_schema_surfaces_section(
    _report: ReportCarrier,
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
) -> list[str]:
    check_deadline()
    return [
        "Schema surfaces preview (provisional).",
        f"- `paths_with_groups`: `{len(groups_by_path)}`",
        "- `note`: `full schema-surface discovery is materialized in post-phase projection`",
    ]


def _preview_deprecated_substrate_section(
    report: ReportCarrier,
    _groups_by_path: dict[Path, dict[str, list[set[str]]]],
) -> list[str]:
    check_deadline()
    lines = [
        "Deprecated substrate preview (provisional).",
        f"- `informational_signals`: `{len(report.deprecated_signals)}`",
    ]
    for signal in report.deprecated_signals[:5]:
        check_deadline()
        lines.append(f"- {signal}")
    return lines


def _report_section_text(
    report: ReportCarrier,
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    *,
    section_id: str,
) -> list[str]:
    rendered, _ = _emit_report(
        groups_by_path,
        max_components=10,
        report=report,
    )
    return extract_report_sections(rendered).get(section_id, [])


def _report_section_spec(
    *,
    section_id: str,
    phase: ReportProjectionPhase,
    deps: tuple[str, ...] = (),
    preview_build = None,
) -> ReportProjectionSpec[list[str]]:
    return ReportProjectionSpec[list[str]](
        section_id=section_id,
        phase=phase,
        deps=deps,
        build=lambda report, groups_by_path, _section_id=section_id: _report_section_text(
            report,
            groups_by_path,
            section_id=_section_id,
        ),
        render=_report_section_identity_render,
        violation_extract=_report_section_no_violations,
        preview_build=preview_build,
    )


_REPORT_PROJECTION_DECLARED_SPECS: tuple[ReportProjectionSpec[list[str]], ...] = (
    _report_section_spec(section_id="intro", phase="collection"),
    _report_section_spec(
        section_id="components",
        phase="forest",
        deps=("intro",),
        preview_build=_preview_components_section,
    ),
    _report_section_spec(
        section_id="violations",
        phase="post",
        deps=("components",),
        preview_build=_preview_violations_section,
    ),
    _report_section_spec(
        section_id="type_flow",
        phase="edge",
        deps=("components",),
        preview_build=_preview_type_flow_section,
    ),
    _report_section_spec(
        section_id="constant_smells",
        phase="edge",
        deps=("type_flow",),
        preview_build=_preview_constant_smells_section,
    ),
    _report_section_spec(
        section_id="unused_arg_smells",
        phase="edge",
        deps=("type_flow",),
        preview_build=_preview_unused_arg_smells_section,
    ),
    _report_section_spec(
        section_id="deadline_summary",
        phase="post",
        deps=("components",),
        preview_build=_preview_deadline_summary_section,
    ),
    _report_section_spec(
        section_id="resumability_obligations",
        phase="post",
        deps=("components",),
        preview_build=_preview_resumability_obligations_section,
    ),
    _report_section_spec(
        section_id="incremental_report_obligations",
        phase="post",
        deps=("components",),
        preview_build=_preview_incremental_report_obligations_section,
    ),
    _report_section_spec(
        section_id="parse_failure_witnesses",
        phase="post",
        deps=("components",),
        preview_build=_preview_parse_failure_witnesses_section,
    ),
    _report_section_spec(
        section_id="execution_pattern_suggestions",
        phase="post",
        deps=("components",),
        preview_build=_preview_execution_pattern_suggestions_section,
    ),
    _report_section_spec(
        section_id="pattern_schema_residue",
        phase="post",
        deps=("components",),
        preview_build=_preview_pattern_schema_residue_section,
    ),
    _report_section_spec(
        section_id="decision_surfaces",
        phase="post",
        deps=("components",),
        preview_build=_preview_decision_surfaces_section,
    ),
    _report_section_spec(
        section_id="value_decision_surfaces",
        phase="post",
        deps=("decision_surfaces",),
        preview_build=_preview_value_decision_surfaces_section,
    ),
    _report_section_spec(
        section_id="fingerprint_warnings",
        phase="post",
        deps=("components",),
        preview_build=_preview_fingerprint_warnings_section,
    ),
    _report_section_spec(
        section_id="fingerprint_matches",
        phase="post",
        deps=("components",),
        preview_build=_preview_fingerprint_matches_section,
    ),
    _report_section_spec(
        section_id="fingerprint_synthesis",
        phase="post",
        deps=("components",),
        preview_build=_preview_fingerprint_synthesis_section,
    ),
    _report_section_spec(
        section_id="context_suggestions",
        phase="post",
        deps=("decision_surfaces",),
        preview_build=_preview_context_suggestions_section,
    ),
    _report_section_spec(
        section_id="schema_surfaces",
        phase="post",
        deps=("components",),
        preview_build=_preview_schema_surfaces_section,
    ),
    _report_section_spec(
        section_id="deprecated_substrate",
        phase="post",
        deps=("components",),
        preview_build=_preview_deprecated_substrate_section,
    ),
)

_REPORT_PROJECTION_PHASE_RANKS: dict[ReportProjectionPhase, int] = {
    "collection": 0,
    "forest": 1,
    "edge": 2,
    "post": 3,
}


def report_projection_phase_rank(phase: ReportProjectionPhase) -> int:
    return _REPORT_PROJECTION_PHASE_RANKS[phase]


def _topologically_order_report_projection_specs(
    specs: tuple[ReportProjectionSpec[list[str]], ...],
) -> tuple[ReportProjectionSpec[list[str]], ...]:
    by_id = {spec.section_id: spec for spec in specs}
    if len(by_id) != len(specs):
        duplicate_id = next(
            section_id
            for section_id, count in Counter(spec.section_id for spec in specs).items()
            if count > 1
        )
        never("duplicate report projection section_id", section_id=duplicate_id)
    declaration_index = {
        spec.section_id: idx for idx, spec in enumerate(specs)
    }
    dep_pairs = tuple(
        (spec.section_id, dep)
        for spec in specs
        for dep in spec.deps
    )
    missing_dep = next(
        ((section_id, dep) for section_id, dep in dep_pairs if dep not in by_id),
        None,
    )
    if missing_dep is not None:
        section_id, dep = missing_dep
        never(
            "report projection dependency missing",
            section_id=section_id,
            missing_dep=dep,
        )
    self_dep_section = next(
        (section_id for section_id, dep in dep_pairs if dep == section_id),
        None,
    )
    if self_dep_section is not None:
        never(
            "report projection self dependency",
            section_id=self_dep_section,
        )

    def _order_key(section_id: str) -> tuple[int, int, str]:
        spec = by_id[section_id]
        return (
            report_projection_phase_rank(spec.phase),
            declaration_index[section_id],
            section_id,
        )

    prioritized_ids = tuple(sort_once(by_id, key=_order_key, source = 'src/gabion/analysis/dataflow_audit.py:1213'))
    predecessor_graph = {
        section_id: tuple(
            dict.fromkeys(sort_once(by_id[section_id].deps, key=_order_key, source = 'src/gabion/analysis/dataflow_audit.py:1216'))
        )
        for section_id in prioritized_ids
    }
    try:
        ordered_ids = tuple(TopologicalSorter(predecessor_graph).static_order())
    except CycleError as exc:
        cycle = exc.args[1] if len(exc.args) > 1 else ()
        cycle_entries = tuple(sequence_or_none(cycle) or ())
        unresolved = [
            str(item)
            for item in cycle_entries
        ]
        never(
            "report projection dependency cycle",
            unresolved=unresolved,
        )
    return tuple(by_id[section_id] for section_id in ordered_ids)


_REPORT_PROJECTION_SPECS = _topologically_order_report_projection_specs(
    _REPORT_PROJECTION_DECLARED_SPECS
)


def report_projection_specs() -> tuple[ReportProjectionSpec[list[str]], ...]:
    return _REPORT_PROJECTION_SPECS


def report_projection_spec_rows() -> list[JSONObject]:
    return [
        {
            "section_id": spec.section_id,
            "phase": spec.phase,
            "deps": list(spec.deps),
            "has_preview": spec.preview_build is not None,
        }
        for spec in _REPORT_PROJECTION_SPECS
    ]


def project_report_sections(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    report: ReportCarrier,
    *,
    max_phase = None,
    include_previews: bool = False,
    preview_only: bool = False,
) -> dict[str, list[str]]:
    extracted: dict[str, list[str]] = {}
    if not preview_only:
        rendered, _ = _emit_report(
            groups_by_path,
            max_components=10,
            report=report,
        )
        extracted = extract_report_sections(rendered)
    max_rank = None
    if max_phase is not None:
        max_rank = report_projection_phase_rank(max_phase)

    def _include_spec(spec: ReportProjectionSpec[list[str]]) -> bool:
        if max_rank is None:
            return True
        return report_projection_phase_rank(spec.phase) <= max_rank

    eligible_specs = tuple(spec for spec in _REPORT_PROJECTION_SPECS if _include_spec(spec))

    def _resolve_spec_lines(spec: ReportProjectionSpec[list[str]]) -> list[str]:
        lines = extracted.get(spec.section_id, [])
        if not lines and include_previews and spec.preview_build is not None:
            preview = spec.preview_build(report, groups_by_path)
            lines = spec.render(preview or [])
        return lines

    return {
        spec.section_id: lines
        for spec in eligible_specs
        if (lines := _resolve_spec_lines(spec))
    }


@dataclass(frozen=True)
class CallAmbiguity:
    kind: str
    caller: FunctionInfo
    call: "CallArgs | None"
    callee_key: str
    candidates: tuple[FunctionInfo, ...]
    phase: str


def _callee_name(call: ast.Call) -> str:
    try:
        return ast.unparse(call.func)
    except _AST_UNPARSE_ERROR_TYPES:
        return "<call>"


def _normalize_callee(name: str, class_name) -> str:
    if not class_name:
        return name
    if name.startswith("self.") or name.startswith("cls."):
        parts = name.split(".")
        if len(parts) == 2:
            return f"{class_name}.{parts[1]}"
    return name


def _iter_paths(paths: Iterable[str], config: AuditConfig) -> list[Path]:
    """Expand input paths to python files, pruning ignored directories early."""
    check_deadline()
    out: list[Path] = []
    for p in paths:
        check_deadline()
        path = Path(p)
        if path.is_dir():
            for root, dirnames, filenames in os.walk(path, topdown=True):
                check_deadline()
                if config.exclude_dirs:
                    # Prune excluded dirs before descending to avoid scanning
                    # large env/vendor trees like `.venv/`.
                    dirnames[:] = [d for d in dirnames if d not in config.exclude_dirs]
                dirnames[:] = sort_once(
                    dirnames,
                    source="_iter_paths.dirnames",
                    # Lexical directory names for deterministic traversal order.
                    key=lambda name: name,
                )
                for filename in sort_once(
                    filenames,
                    source="_iter_paths.filenames",
                ):
                    check_deadline()
                    if not filename.endswith(".py"):
                        continue
                    candidate = Path(root) / filename
                    if config.is_ignored_path(candidate):
                        continue
                    out.append(candidate)
        else:
            if config.is_ignored_path(path):
                continue
            out.append(path)
    return sort_once(
        out,
        source="_iter_paths.out",
    )


def resolve_analysis_paths(paths: Iterable[object], *, config: AuditConfig) -> list[Path]:
    check_deadline()
    return _iter_paths([str(path) for path in paths], config)


def _collect_functions(tree: ast.AST) -> list[FunctionNode]:
    check_deadline()
    funcs: list[FunctionNode] = []
    for idx, node in enumerate(ast.walk(tree), start=1):
        if (idx & 63) == 0:
            check_deadline()
        node_type = type(node)
        if node_type is ast.FunctionDef or node_type is ast.AsyncFunctionDef:
            funcs.append(cast(FunctionNode, node))
    return funcs


def _invariant_term(expr: ast.AST, params: set[str]):
    expr_type = type(expr)
    if expr_type is ast.Name:
        name_expr = cast(ast.Name, expr)
        return next(iter(params.intersection({name_expr.id})), None)
    if expr_type is ast.Call:
        call_expr = cast(ast.Call, expr)
        if type(call_expr.func) is ast.Name:
            func_name = cast(ast.Name, call_expr.func)
            if func_name.id == "len" and len(call_expr.args) == 1:
                arg = call_expr.args[0]
                if type(arg) is ast.Name:
                    arg_id = cast(ast.Name, arg).id
                    return next((f"{entry}.length" for entry in params.intersection({arg_id})), None)
    return None


def _extract_invariant_from_expr(
    expr: ast.AST,
    params: set[str],
    *,
    scope: str,
    source: str = "assert",
) -> object:
    if type(expr) is not ast.Compare:
        return None
    compare_expr = cast(ast.Compare, expr)
    if len(compare_expr.ops) != 1 or len(compare_expr.comparators) != 1:
        return None
    if type(compare_expr.ops[0]) is not ast.Eq:
        return None
    left = _invariant_term(compare_expr.left, params)
    right = _invariant_term(compare_expr.comparators[0], params)
    if left is not None and right is not None:
        return InvariantProposition(
            form="Equal",
            terms=(left, right),
            scope=scope,
            source=source,
        )
    return None


class _InvariantCollector(ast.NodeVisitor):
    # dataflow-bundle: params, scope
    def __init__(self, params: set[str], scope: str) -> None:
        self._params = params
        self._scope = scope
        self.propositions: list[InvariantProposition] = []
        self._seen: set[tuple[str, tuple[str, ...], str]] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_Assert(self, node: ast.Assert) -> None:
        prop = _extract_invariant_from_expr(
            node.test,
            self._params,
            scope=self._scope,
        )
        if prop is not None:
            normalized = _normalize_invariant_proposition(
                prop,
                default_scope=self._scope,
                default_source="assert",
            )
            key = (normalized.form, normalized.terms, normalized.scope or "")
            if key not in self._seen:
                self._seen.add(key)
                self.propositions.append(normalized)
        self.generic_visit(node)


def _scope_path(path: Path, root) -> str:
    if root is not None:
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return str(path)


def _collect_invariant_propositions(
    path: Path,
    *,
    ignore_params: set[str],
    project_root,
    emitters: Iterable[
        Callable[[ast.FunctionDef], Iterable[InvariantProposition]]
    ] = (),
) -> list[InvariantProposition]:
    check_deadline()
    tree = ast.parse(path.read_text())
    propositions: list[InvariantProposition] = []
    for fn in _collect_functions(tree):
        check_deadline()
        params = set(_param_names(fn, ignore_params))
        if not params:
            continue
        scope = f"{_scope_path(path, project_root)}:{fn.name}"
        collector = _InvariantCollector(params, scope)
        for stmt in fn.body:
            check_deadline()
            collector.visit(stmt)
        propositions.extend(collector.propositions)
        for emitter in emitters:
            check_deadline()
            emitted = emitter(fn)
            for prop in emitted:
                check_deadline()
                if type(prop) is not InvariantProposition:
                    raise TypeError(
                        "Invariant emitters must yield InvariantProposition instances."
                    )
                propositions.append(
                    _normalize_invariant_proposition(
                        prop,
                        default_scope=scope,
                        default_source="emitter",
                    )
                )
    return propositions


def _format_invariant_proposition(prop: InvariantProposition) -> str:
    if prop.form == "Equal" and len(prop.terms) == 2:
        rendered = f"{prop.terms[0]} == {prop.terms[1]}"
    else:
        rendered = f"{prop.form}({', '.join(prop.terms)})"
    prefix = f"{prop.scope}: " if prop.scope else ""
    suffix = f" [{prop.source}]" if prop.source else ""
    return f"{prefix}{rendered}{suffix}"


def _format_invariant_propositions(
    props: list[InvariantProposition],
) -> list[str]:
    return [
        _format_invariant_proposition(prop)
        for prop in props
    ]


def generate_property_hook_manifest(
    invariants: Sequence[InvariantProposition],
    *,
    min_confidence: float = 0.7,
    emit_hypothesis_templates: bool = False,
) -> JSONObject:
    threshold = max(0.0, min(1.0, min_confidence))
    hooks: list[JSONObject] = []
    for proposition in sort_once(
        invariants,
        key=lambda prop: (
            prop.scope or "",
            prop.form,
            prop.terms,
            prop.invariant_id or "",
        ),
    source = 'src/gabion/analysis/dataflow_audit.py:1535'):
        check_deadline()
        scope = proposition.scope or ""
        if not scope or ":" not in scope:
            continue
        confidence = _invariant_confidence(proposition.confidence)
        if confidence < threshold:
            continue
        normalized = _normalize_invariant_proposition(
            proposition,
            default_scope=scope,
            default_source=proposition.source or "inferred",
        )
        hook_id = _invariant_digest(
            {
                "invariant_id": normalized.invariant_id,
                "scope": normalized.scope,
            },
            prefix="hook",
        )
        path, callable_name = scope.rsplit(":", 1)
        hook_payload: JSONObject = {
            "hook_id": hook_id,
            "invariant_id": normalized.invariant_id or "",
            "callable": {
                "path": path,
                "qual": callable_name,
            },
            "form": normalized.form,
            "terms": list(normalized.terms),
            "confidence": confidence,
            "source": normalized.source or "",
            "source_invariant_evidence_keys": list(normalized.evidence_keys),
        }
        if emit_hypothesis_templates:
            params = ", ".join(normalized.terms)
            hypothesis_name = (
                f"test_{callable_name}_{(normalized.invariant_id or '').replace(':', '_')}"
                .replace("-", "_")
            )
            hook_payload["hypothesis_template"] = "\n".join(
                [
                    "from hypothesis import given",
                    "",
                    f"def {hypothesis_name}():",
                    f"    # invariant: {normalized.form}({params})",
                    "    # TODO: provide strategies and callable invocation.",
                    "    pass",
                ]
            )
        hooks.append(hook_payload)
    hooks = [
        hooks[idx]
        for idx in sort_once(
            range(len(hooks)),
            key=lambda idx: (
                str(hooks[idx].get("hook_id", "")),
                str(hooks[idx].get("invariant_id", "")),
            ),
        source = 'src/gabion/analysis/dataflow_audit.py:1596')
    ]
    callable_index = _build_property_hook_callable_index(hooks)
    return {
        "format_version": 1,
        "kind": "property_hook_manifest",
        "min_confidence": threshold,
        "emit_hypothesis_templates": emit_hypothesis_templates,
        "hooks": hooks,
        "callable_index": callable_index,
    }


def _build_property_hook_callable_index(hooks: Sequence[JSONValue]) -> list[JSONObject]:
    callables: dict[str, list[str]] = defaultdict(list)
    for hook in hooks:
        check_deadline()
        if type(hook) is not dict:
            continue
        hook_payload = cast(Mapping[str, JSONValue], hook)
        callable_payload = hook_payload.get("callable")
        if type(callable_payload) is not dict:
            continue
        callable_mapping = cast(Mapping[str, JSONValue], callable_payload)
        path = str(callable_mapping.get("path", "") or "")
        qual = str(callable_mapping.get("qual", "") or "")
        if not path or not qual:
            continue
        callables[f"{path}:{qual}"].append(str(hook_payload.get("hook_id", "") or ""))
    return [
        {
            "scope": scope,
            "hook_ids": sort_once(hook_ids, source = 'src/gabion/analysis/dataflow_audit.py:1632'),
        }
        for scope, hook_ids in sort_once(callables.items(), source = 'src/gabion/analysis/dataflow_audit.py:1634')
    ]


def _decorator_name(node: ast.AST):
    check_deadline()
    node_type = type(node)
    if node_type is ast.Name:
        return cast(ast.Name, node).id
    if node_type is ast.Attribute:
        parts: list[str] = []
        current: ast.AST = node
        while type(current) is ast.Attribute:
            check_deadline()
            attribute_node = cast(ast.Attribute, current)
            parts.append(attribute_node.attr)
            current = attribute_node.value
        if type(current) is ast.Name:
            parts.append(cast(ast.Name, current).id)
            return ".".join(reversed(parts))
        return None
    if node_type is ast.Call:
        return _decorator_name(cast(ast.Call, node).func)
    return None


def _decorator_matches(name: str, allowlist: set[str]) -> bool:
    if name in allowlist:
        return True
    if "." in name and name.split(".")[-1] in allowlist:
        return True
    return False


_NEVER_MARKERS = {"never", "gabion.never", "gabion.invariants.never"}
_NEVER_STATUS_ORDER = {"VIOLATION": 0, "OBLIGATION": 1, "PROVEN_UNREACHABLE": 2}


def _is_never_call(call: ast.Call) -> bool:
    name = _decorator_name(call.func)
    if not name:
        return False
    return _decorator_matches(name, _NEVER_MARKERS)


def _is_never_marker_raise(
    function: str,
    exception_name,
    never_exceptions: set[str],
) -> bool:
    if not exception_name or not never_exceptions:
        return False
    if not _decorator_matches(exception_name, never_exceptions):
        return False
    return function == "never" or function.endswith(".never")


def _never_sort_key(entry: JSONObject) -> tuple:
    status = str(entry.get("status", "UNKNOWN"))
    order = _NEVER_STATUS_ORDER.get(status, 3)
    raw_site = entry.get("site")
    site = mapping_or_empty(raw_site)
    path = str(site.get("path", ""))
    function = str(site.get("function", ""))
    span = entry.get("span")
    line = -1
    col = -1
    span_entries = sequence_or_none(span)
    if span_entries is not None and len(span_entries) == 4:
        try:
            line = int(span_entries[0])
            col = int(span_entries[1])
        except (TypeError, ValueError):
            line = -1
            col = -1
    return (order, path, function, line, col, str(entry.get("never_id", "")))


def _decorators_transparent(
    fn: FunctionNode,
    transparent_decorators,
) -> bool:
    check_deadline()
    if not fn.decorator_list:
        return True
    if not transparent_decorators:
        return True
    for deco in fn.decorator_list:
        check_deadline()
        name = _decorator_name(deco)
        if not name:
            return False
        if not _decorator_matches(name, transparent_decorators):
            return False
    return True


def _collect_local_class_bases(
    tree: ast.AST, parents: dict[ast.AST, ast.AST]
) -> dict[str, list[str]]:
    check_deadline()
    class_bases: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        check_deadline()
        if type(node) is ast.ClassDef:
            class_node = cast(ast.ClassDef, node)
            scopes = _enclosing_class_scopes(class_node, parents)
            qual_parts = list(scopes)
            qual_parts.append(class_node.name)
            qual = ".".join(qual_parts)
            bases: list[str] = []
            for base in class_node.bases:
                check_deadline()
                base_name = _base_identifier(base)
                if base_name:
                    bases.append(base_name)
            class_bases[qual] = bases
    return class_bases


def _local_class_name(base: str, class_bases: dict[str, list[str]]):
    if base in class_bases:
        return base
    if "." in base:
        tail = base.split(".")[-1]
        if tail in class_bases:
            return tail
    return None


def _resolve_local_method_in_hierarchy(
    class_name: str,
    method: str,
    *,
    class_bases: dict[str, list[str]],
    local_functions: set[str],
    seen: set[str],
):
    check_deadline()
    if class_name in seen:
        return None
    seen.add(class_name)
    candidate = f"{class_name}.{method}"
    if candidate in local_functions:
        return candidate
    for base in class_bases.get(class_name, []):
        check_deadline()
        base_name = _local_class_name(base, class_bases)
        if base_name is not None:
            resolved = _resolve_local_method_in_hierarchy(
                base_name,
                method,
                class_bases=class_bases,
                local_functions=local_functions,
                seen=seen,
            )
            if resolved is not None:
                return resolved
    return None


def _param_names(
    fn: FunctionNode,
    ignore_params: OptionalIgnoredParams = None,
) -> list[str]:
    args = (
        fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
    )
    names = [a.arg for a in args]
    if fn.args.vararg:
        names.append(fn.args.vararg.arg)
    if fn.args.kwarg:
        names.append(fn.args.kwarg.arg)
    if names and names[0] in {"self", "cls"}:
        names = names[1:]
    if ignore_params:
        names = [name for name in names if name not in ignore_params]
    return names


def _decision_root_name(node: ast.AST):
    check_deadline()
    current = node
    while True:
        check_deadline()
        current_type = type(current)
        if current_type is ast.Attribute:
            current = cast(ast.Attribute, current).value
        elif current_type is ast.Subscript:
            current = cast(ast.Subscript, current).value
        else:
            break
    if type(current) is ast.Name:
        return cast(ast.Name, current).id
    return None


def is_decision_surface(node: ast.AST) -> bool:
    node_type = type(node)
    return (
        node_type is ast.If
        or node_type is ast.While
        or node_type is ast.Assert
        or node_type is ast.IfExp
        or node_type is ast.Match
        or node_type is ast.comprehension
    )


def _decision_surface_form_entries(
    fn: ast.AST,
) -> list[tuple[str, ast.AST]]:
    check_deadline()
    entries: list[tuple[str, ast.AST]] = []
    for node in ast.walk(fn):
        check_deadline()
        if not is_decision_surface(node):
            continue
        node_type = type(node)
        if node_type is ast.If:
            entries.append(("if", cast(ast.If, node).test))
            continue
        if node_type is ast.While:
            entries.append(("while", cast(ast.While, node).test))
            continue
        if node_type is ast.Assert:
            entries.append(("assert", cast(ast.Assert, node).test))
            continue
        if node_type is ast.IfExp:
            entries.append(("ifexp", cast(ast.IfExp, node).test))
            continue
        if node_type is ast.Match:
            match_node = cast(ast.Match, node)
            entries.append(("match_subject", match_node.subject))
            for case in match_node.cases:
                check_deadline()
                if case.guard is not None:
                    entries.append(("match_guard", case.guard))
            continue
        for guard in cast(ast.comprehension, node).ifs:
            check_deadline()
            entries.append(("comprehension_guard", guard))
    return entries


def _decision_surface_reason_map(
    fn: FunctionNode,
    ignore_params: OptionalIgnoredParams = None,
) -> dict[str, set[str]]:
    check_deadline()
    params = set(_param_names(fn, ignore_params))
    if not params:
        return {}
    reason_map: dict[str, set[str]] = defaultdict(set)
    for reason, expr in _decision_surface_form_entries(fn):
        check_deadline()
        found = _collect_param_roots(expr, params)
        for param in found:
            check_deadline()
            reason_map[param].add(reason)
    return reason_map


def _decision_surface_params(
    fn: FunctionNode,
    ignore_params: OptionalIgnoredParams = None,
) -> set[str]:
    check_deadline()
    reason_map = _decision_surface_reason_map(fn, ignore_params)
    return set(reason_map)


def _mark_param_roots(expr: ast.AST, params: set[str], out: set[str]) -> None:
    check_deadline()
    for node in ast.walk(expr):
        check_deadline()
        node_type = type(node)
        if node_type is ast.Name and cast(ast.Name, node).id in params:
            out.add(cast(ast.Name, node).id)
            continue
        if node_type is ast.Attribute or node_type is ast.Subscript:
            root = _decision_root_name(node)
            if root in params:
                out.add(root)


def _collect_param_roots(expr: ast.AST, params: set[str]) -> set[str]:
    found: set[str] = set()
    _mark_param_roots(expr, params, found)
    return found


def _contains_boolish(expr: ast.AST) -> bool:
    check_deadline()
    for node in ast.walk(expr):
        check_deadline()
        node_type = type(node)
        if node_type is ast.Compare or node_type is ast.BoolOp:
            return True
        if node_type is ast.UnaryOp and type(cast(ast.UnaryOp, node).op) is ast.Not:
            return True
    return False


def _value_encoded_decision_params(
    fn: ast.AST,
    ignore_params = None,
) -> tuple[set[str], set[str]]:
    check_deadline()
    params = set(_param_names(fn, ignore_params))
    if not params:
        return set(), set()
    flagged: set[str] = set()
    reasons: set[str] = set()
    for node in ast.walk(fn):
        check_deadline()
        node_type = type(node)
        if node_type is ast.Call:
            call_node = cast(ast.Call, node)
            func = call_node.func
            func_type = type(func)
            if func_type is ast.Name and cast(ast.Name, func).id in {"min", "max"}:
                reasons.add("min/max")
                _mark_param_roots(call_node, params, flagged)
            elif func_type is ast.Attribute and cast(ast.Attribute, func).attr in {"min", "max"}:
                reasons.add("min/max")
                _mark_param_roots(call_node, params, flagged)
        elif node_type is ast.BinOp:
            binop_node = cast(ast.BinOp, node)
            op_type = type(binop_node.op)
            left_bool = _contains_boolish(binop_node.left)
            right_bool = _contains_boolish(binop_node.right)
            if op_type in {
                ast.Mult,
                ast.Add,
                ast.Sub,
                ast.FloorDiv,
                ast.Mod,
                ast.BitAnd,
                ast.BitOr,
                ast.BitXor,
                ast.LShift,
                ast.RShift,
            }:
                if left_bool or right_bool:
                    reasons.add("boolean arithmetic")
                    if left_bool:
                        flagged |= _collect_param_roots(binop_node.left, params)
                    if right_bool:
                        flagged |= _collect_param_roots(binop_node.right, params)
                if op_type in {
                    ast.BitAnd,
                    ast.BitOr,
                    ast.BitXor,
                    ast.LShift,
                    ast.RShift,
                } and not (left_bool or right_bool):
                    left_roots = _collect_param_roots(binop_node.left, params)
                    right_roots = _collect_param_roots(binop_node.right, params)
                    if left_roots or right_roots:
                        reasons.add("bitmask")
                        flagged |= left_roots | right_roots
    return flagged, reasons


@dataclass(frozen=True)
class _DecisionSurfaceSpec:
    pass_id: str
    alt_kind: str
    surface_label: str
    params: Callable[[FunctionInfo], set[str]]
    descriptor: Callable[[FunctionInfo, str], str]
    alt_evidence: Callable[[str, str], JSONObject]
    surface_lint_code: str
    surface_lint_message: Callable[[str, str, str], str]
    emit_surface_lint: Callable[[int, object], bool]
    tier_lint_code: str
    tier_missing_message: Callable[[str, str], str]
    tier_internal_message: Callable[[str, int, str, str], str]
    rewrite_line: object = None


def _decision_reason_summary(info: FunctionInfo, params: Iterable[str]) -> str:
    labels: set[str] = set()
    for param in params:
        check_deadline()
        labels.update(info.decision_surface_reasons.get(param, set()))
    if not labels:
        return "heuristic"
    return ", ".join(
        sort_once(labels, source="_decision_reason_summary.labels")
    )


def _boundary_tier_obligation(caller_count: int) -> str:
    if caller_count > 0:
        return "tier-2:decision-bundle-elevation"
    return "tier-3:decision-table-boundary"


def _decision_surface_alt_evidence(
    *,
    spec: _DecisionSurfaceSpec,
    boundary: str,
    descriptor: str,
    params: Iterable[str],
    caller_count: int,
    reason_summary: str,
) -> JSONObject:
    base_evidence = dict(spec.alt_evidence(boundary, descriptor))
    payload: JSONObject = {
        "boundary": base_evidence.get("boundary", boundary),
        "classification_descriptor": descriptor,
        "classification_reason": reason_summary,
        "decision_params": sort_once(
            set(params),
            source="_decision_surface_alt_evidence.params",
        ),
    }
    if "meta" in base_evidence:
        payload["meta"] = base_evidence["meta"]
    for key in sort_once(
        (str(k) for k in base_evidence if str(k) not in {"boundary", "meta"}),
        source="_decision_surface_alt_evidence.base_evidence",
    ):
        payload[key] = base_evidence[key]
    payload["tier_obligation"] = _boundary_tier_obligation(caller_count)
    payload["tier_pathway"] = "internal" if caller_count > 0 else "boundary"
    return payload

_DIRECT_DECISION_SURFACE_SPEC = _DecisionSurfaceSpec(
    pass_id="decision_surfaces",
    alt_kind="DecisionSurface",
    surface_label="decision surface params",
    params=lambda info: info.decision_params,
    descriptor=lambda info, boundary: (
        f"{boundary}; reason={_decision_reason_summary(info, info.decision_params)}"
    ),
    alt_evidence=lambda boundary, _descriptor: {
        "meta": boundary,
        "boundary": boundary,
    },
    surface_lint_code="GABION_DECISION_SURFACE",
    surface_lint_message=lambda param, boundary, _descriptor: (
        f"decision surface param '{param}' ({boundary})"
    ),
    emit_surface_lint=lambda caller_count, tier: caller_count == 0 and tier is None,
    tier_lint_code="GABION_DECISION_TIER",
    tier_missing_message=lambda param, _descriptor: (
        f"decision param '{param}' missing decision tier metadata"
    ),
    tier_internal_message=lambda param, tier, boundary, _descriptor: (
        f"tier-{tier} decision param '{param}' used below boundary ({boundary})"
    ),
)


_VALUE_DECISION_SURFACE_SPEC = _DecisionSurfaceSpec(
    pass_id="value_encoded_decisions",
    alt_kind="ValueDecisionSurface",
    surface_label="value-encoded decision params",
    params=lambda info: info.value_decision_params,
    descriptor=lambda info, _boundary: ", ".join(
        sort_once(
            info.value_decision_reasons,
            source="_VALUE_DECISION_SURFACE_SPEC.descriptor",
        )
    )
    or "heuristic",
    alt_evidence=lambda boundary, descriptor: {
        "meta": descriptor,
        "boundary": boundary,
        "reasons": descriptor,
    },
    surface_lint_code="GABION_VALUE_DECISION_SURFACE",
    surface_lint_message=lambda param, boundary, descriptor: (
        f"value-encoded decision param '{param}' ({boundary}; {descriptor})"
    ),
    emit_surface_lint=lambda _caller_count, tier: tier is None,
    tier_lint_code="GABION_VALUE_DECISION_TIER",
    tier_missing_message=lambda param, descriptor: (
        f"value-encoded decision param '{param}' missing decision tier metadata ({descriptor})"
    ),
    tier_internal_message=lambda param, tier, boundary, descriptor: (
        f"tier-{tier} value-encoded decision param '{param}' used below boundary ({boundary}; {descriptor})"
    ),
    rewrite_line=lambda info, params, descriptor: (
        f"{info.path.name}:{info.qual} consider rebranching value-encoded decision params: "
        + ", ".join(params)
        + f" ({descriptor})"
    ),
)


def _analyze_decision_surface_indexed(
    context: _IndexedPassContext,
    *,
    spec: _DecisionSurfaceSpec,
    decision_tiers,
    require_tiers: bool,
    forest: Forest,
) -> tuple[list[str], list[str], list[str], list[str]]:
    _, by_qual, transitive_callers = _build_call_graph(
        context.paths,
        project_root=context.project_root,
        ignore_params=context.ignore_params,
        strictness=context.strictness,
        external_filter=context.external_filter,
        transparent_decorators=context.transparent_decorators,
        parse_failure_witnesses=context.parse_failure_witnesses,
        analysis_index=context.analysis_index,
    )
    surfaces: list[str] = []
    warnings: list[str] = []
    rewrites: list[str] = []
    lint_lines: list[str] = []
    tier_map = decision_tiers or {}
    for info in by_qual.values():
        check_deadline()
        if _is_test_path(info.path):
            continue
        params = sort_once(
            spec.params(info),
            source=f"_analyze_decision_surface_indexed.{spec.pass_id}.params",
        )
        if not params:
            continue
        caller_count = len(transitive_callers.get(info.qual, set()))
        boundary = (
            "boundary"
            if caller_count == 0
            else f"internal callers (transitive): {caller_count}"
        )
        descriptor = spec.descriptor(info, boundary)
        site_id = forest.add_site(info.path.name, info.qual)
        paramset_id = forest.add_paramset(params)
        reason_summary = (
            _decision_reason_summary(info, params)
            if spec.pass_id == "decision_surfaces"
            else descriptor
        )
        forest.add_alt(
            spec.alt_kind,
            (site_id, paramset_id),
            evidence=_decision_surface_alt_evidence(
                spec=spec,
                boundary=boundary,
                descriptor=descriptor,
                params=params,
                caller_count=caller_count,
                reason_summary=reason_summary,
            ),
        )
        surfaces.append(
            f"{info.path.name}:{info.qual} {spec.surface_label}: "
            + ", ".join(params)
            + f" ({descriptor})"
        )
        if spec.rewrite_line is not None:
            rewrites.append(spec.rewrite_line(info, params, descriptor))
        for param in params:
            check_deadline()
            tier = _decision_tier_for(
                info,
                param,
                tier_map=tier_map,
                project_root=context.project_root,
            )
            if spec.emit_surface_lint(caller_count, tier):
                lint = _decision_param_lint_line(
                    info,
                    param,
                    project_root=context.project_root,
                    code=spec.surface_lint_code,
                    message=spec.surface_lint_message(param, boundary, descriptor),
                )
                if lint is not None:
                    lint_lines.append(lint)
            if not tier_map:
                continue
            if tier is None:
                if require_tiers:
                    message = spec.tier_missing_message(param, descriptor)
                    warnings.append(f"{info.path.name}:{info.qual} {message}")
                    lint = _decision_param_lint_line(
                        info,
                        param,
                        project_root=context.project_root,
                        code=spec.tier_lint_code,
                        message=message,
                    )
                    if lint is not None:
                        lint_lines.append(lint)
                continue
            if tier in {2, 3} and caller_count > 0:
                message = spec.tier_internal_message(param, tier, boundary, descriptor)
                warnings.append(f"{info.path.name}:{info.qual} {message}")
                lint = _decision_param_lint_line(
                    info,
                    param,
                    project_root=context.project_root,
                    code=spec.tier_lint_code,
                    message=message,
                )
                if lint is not None:
                    lint_lines.append(lint)
    return (
        sort_once(
            surfaces,
            source="_analyze_decision_surface_indexed.surfaces",
        ),
        sort_once(
            set(warnings),
            source="_analyze_decision_surface_indexed.warnings",
        ),
        sort_once(
            rewrites,
            source="_analyze_decision_surface_indexed.rewrites",
        ),
        sort_once(
            set(lint_lines),
            source="_analyze_decision_surface_indexed.lint_lines",
        ),
    )


def _analyze_decision_surfaces_indexed(
    context: _IndexedPassContext,
    *,
    decision_tiers,
    require_tiers: bool,
    forest: Forest,
    run_fn: Callable[..., tuple[list[str], list[str], list[str], list[str]]] = _analyze_decision_surface_indexed,
) -> tuple[list[str], list[str], list[str]]:
    surfaces, warnings, rewrites, lint_lines = run_fn(
        context,
        spec=_DIRECT_DECISION_SURFACE_SPEC,
        decision_tiers=decision_tiers,
        require_tiers=require_tiers,
        forest=forest,
    )
    if rewrites:
        never(
            "decision_surfaces rewrites must be empty",
            pass_id=_DIRECT_DECISION_SURFACE_SPEC.pass_id,
        )
    return surfaces, warnings, lint_lines


def analyze_decision_surfaces_repo(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators = None,
    decision_tiers = None,
    require_tiers: bool = False,
    forest: Forest,
    parse_failure_witnesses = None,
    analysis_index = None,
) -> tuple[list[str], list[str], list[str]]:
    check_deadline()
    return _run_indexed_pass(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=analysis_index,
        spec=_IndexedPassSpec(
            pass_id="decision_surfaces",
            run=lambda context: _analyze_decision_surfaces_indexed(
                context,
                decision_tiers=decision_tiers,
                require_tiers=require_tiers,
                forest=forest,
            ),
        ),
    )


def _analyze_value_encoded_decisions_indexed(
    context: _IndexedPassContext,
    *,
    decision_tiers,
    require_tiers: bool,
    forest: Forest,
) -> tuple[list[str], list[str], list[str], list[str]]:
    return _analyze_decision_surface_indexed(
        context,
        spec=_VALUE_DECISION_SURFACE_SPEC,
        decision_tiers=decision_tiers,
        require_tiers=require_tiers,
        forest=forest,
    )


def analyze_value_encoded_decisions_repo(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators = None,
    decision_tiers = None,
    require_tiers: bool = False,
    forest: Forest,
    parse_failure_witnesses = None,
    analysis_index = None,
) -> tuple[list[str], list[str], list[str], list[str]]:
    check_deadline()
    return _run_indexed_pass(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=analysis_index,
        spec=_IndexedPassSpec(
            pass_id="value_encoded_decisions",
            run=lambda context: _analyze_value_encoded_decisions_indexed(
                context,
                decision_tiers=decision_tiers,
                require_tiers=require_tiers,
                forest=forest,
            ),
        ),
    )


def _internal_broad_type_lint_lines_indexed(
    context: _IndexedPassContext,
) -> list[str]:
    _, by_qual, transitive_callers = _build_call_graph(
        context.paths,
        project_root=context.project_root,
        ignore_params=context.ignore_params,
        strictness=context.strictness,
        external_filter=context.external_filter,
        transparent_decorators=context.transparent_decorators,
        parse_failure_witnesses=context.parse_failure_witnesses,
        analysis_index=context.analysis_index,
    )
    lines: list[str] = []
    for info in by_qual.values():
        check_deadline()
        if _is_test_path(info.path):
            continue
        caller_count = len(transitive_callers.get(info.qual, set()))
        if caller_count == 0:
            continue
        for param, annot in info.annots.items():
            check_deadline()
            if not _is_broad_internal_type(annot):
                continue
            message = (
                f"internal param '{param}' uses broad type '{annot}' "
                f"(internal callers: {caller_count})"
            )
            lint = _decision_param_lint_line(
                info,
                param,
                project_root=context.project_root,
                code="GABION_BROAD_TYPE",
                message=message,
            )
            if lint is not None:
                lines.append(lint)
    return sort_once(
        set(lines),
        source="_internal_broad_type_lint_lines_indexed.lines",
    )


def _internal_broad_type_lint_lines(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators = None,
    parse_failure_witnesses: list[JSONObject],
    analysis_index = None,
) -> list[str]:
    check_deadline()
    return _run_indexed_pass(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=analysis_index,
        spec=_IndexedPassSpec(
            pass_id="internal_broad_type_lint_lines",
            run=_internal_broad_type_lint_lines_indexed,
        ),
    )


def _node_span(node: ast.AST):
    if not hasattr(node, "lineno") or not hasattr(node, "col_offset"):
        return None
    start_line = max(getattr(node, "lineno", 1) - 1, 0)
    start_col = max(getattr(node, "col_offset", 0), 0)
    end_line = max(getattr(node, "end_lineno", getattr(node, "lineno", 1)) - 1, 0)
    end_col = getattr(node, "end_col_offset", start_col + 1)
    if end_line == start_line and end_col <= start_col:
        end_col = start_col + 1
    return (start_line, start_col, end_line, end_col)


def _param_spans(
    fn: FunctionNode,
    ignore_params: OptionalIgnoredParams = None,
) -> dict[str, tuple[int, int, int, int]]:
    check_deadline()
    spans: dict[str, tuple[int, int, int, int]] = {}
    args = fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
    names = [a.arg for a in args]
    if names and names[0] in {"self", "cls"}:
        args = args[1:]
        names = names[1:]
    for arg in args:
        check_deadline()
        if ignore_params and arg.arg in ignore_params:
            continue
        span = _node_span(arg)
        if span is not None:
            spans[arg.arg] = span
    if fn.args.vararg:
        name = fn.args.vararg.arg
        if not ignore_params or name not in ignore_params:
            span = _node_span(fn.args.vararg)
            if span is not None:
                spans[name] = span
    if fn.args.kwarg:
        name = fn.args.kwarg.arg
        if not ignore_params or name not in ignore_params:
            span = _node_span(fn.args.kwarg)
            if span is not None:
                spans[name] = span
    return spans


def _function_key(scope: Iterable[str], name: str) -> str:
    parts = list(scope)
    parts.append(name)
    return ".".join(parts)


def _enclosing_class(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
):
    check_deadline()
    current = parents.get(node)
    while current is not None:
        check_deadline()
        if type(current) is ast.ClassDef:
            return cast(ast.ClassDef, current).name
        current = parents.get(current)
    return None


def _enclosing_scopes(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> list[str]:
    check_deadline()
    scopes: list[str] = []
    current = parents.get(node)
    while current is not None:
        check_deadline()
        current_type = type(current)
        if current_type is ast.ClassDef:
            scopes.append(cast(ast.ClassDef, current).name)
        elif current_type is ast.FunctionDef or current_type is ast.AsyncFunctionDef:
            scopes.append(cast(FunctionNode, current).name)
        current = parents.get(current)
    return list(reversed(scopes))


def _enclosing_class_scopes(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> list[str]:
    check_deadline()
    scopes: list[str] = []
    current = parents.get(node)
    while current is not None:
        check_deadline()
        if type(current) is ast.ClassDef:
            scopes.append(cast(ast.ClassDef, current).name)
        current = parents.get(current)
    return list(reversed(scopes))


def _enclosing_function_scopes(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> list[str]:
    check_deadline()
    scopes: list[str] = []
    current = parents.get(node)
    while current is not None:
        check_deadline()
        current_type = type(current)
        if current_type is ast.FunctionDef or current_type is ast.AsyncFunctionDef:
            scopes.append(cast(FunctionNode, current).name)
        current = parents.get(current)
    return list(reversed(scopes))


def _param_annotations(
    fn: FunctionNode,
    ignore_params: OptionalIgnoredParams = None,
) -> ParamAnnotationMap:
    check_deadline()
    args = fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
    names = [a.arg for a in args]
    annots: ParamAnnotationMap = {}
    for name, arg in zip(names, args):
        check_deadline()
        if arg.annotation is None:
            annots[name] = None
        else:
            try:
                annots[name] = ast.unparse(arg.annotation)
            except _AST_UNPARSE_ERROR_TYPES:
                annots[name] = None
    if fn.args.vararg:
        vararg = fn.args.vararg
        if vararg.annotation is None:
            annots[vararg.arg] = None
        else:
            try:
                annots[vararg.arg] = ast.unparse(vararg.annotation)
            except _AST_UNPARSE_ERROR_TYPES:  # pragma: no cover - defensive against malformed AST nodes
                annots[vararg.arg] = None  # pragma: no cover
    if fn.args.kwarg:
        kwarg = fn.args.kwarg
        if kwarg.annotation is None:
            annots[kwarg.arg] = None
        else:
            try:
                annots[kwarg.arg] = ast.unparse(kwarg.annotation)
            except _AST_UNPARSE_ERROR_TYPES:  # pragma: no cover - defensive against malformed AST nodes
                annots[kwarg.arg] = None  # pragma: no cover
    if names and names[0] in {"self", "cls"}:
        annots.pop(names[0], None)
    if ignore_params:
        for name in list(annots.keys()):
            check_deadline()
            if name in ignore_params:
                annots.pop(name, None)
    return annots


def _param_defaults(
    fn: FunctionNode,
    ignore_params: OptionalIgnoredParams = None,
) -> set[str]:
    check_deadline()
    defaults: set[str] = set()
    args = fn.args.posonlyargs + fn.args.args
    names = [a.arg for a in args]
    if fn.args.defaults:
        defaulted = names[-len(fn.args.defaults) :]
        defaults.update(defaulted)
    for kw_arg, default in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
        check_deadline()
        if default is not None:
            defaults.add(kw_arg.arg)
    if names and names[0] in {"self", "cls"}:
        defaults.discard(names[0])
    if ignore_params:
        defaults = {name for name in defaults if name not in ignore_params}
    return defaults


def _parse_failure_witness(
    *,
    path: Path,
    stage,
    error: Exception,
) -> JSONObject:
    stage_value = stage.value if type(stage) is _ParseModuleStage else stage
    return {
        "path": str(path),
        "stage": stage_value,
        "error_type": type(error).__name__,
        "error": str(error),
    }


def _record_parse_failure_witness(
    *,
    sink: list[JSONObject],
    path: Path,
    stage,
    error: Exception,
) -> None:
    sink.append(_parse_failure_witness(path=path, stage=stage, error=error))


def _parse_failure_sink(
    parse_failure_witnesses,
) -> list[JSONObject]:
    sink = parse_failure_witnesses
    if sink is None:
        sink = []
    return sink


_NON_NULL_PARSE_WITNESS_HELPERS = frozenset(
    {
        "_internal_broad_type_lint_lines",
        "_collect_deadline_obligations",
        "_build_call_graph",
        "_collect_call_ambiguities",
        "_populate_bundle_forest",
        "_infer_type_flow",
        "_collect_constant_flow_details",
    }
)

_PARSE_CONTRACT_PROGRESS_INTERVAL = 128
_PARSE_CONTRACT_MAPS_OP = DerivationOp(
    name="parse_contract.maps",
    version=1,
    scope="dataflow_audit",
)
_PARSE_CONTRACT_IMPORTED_FUNCTIONS_OP = DerivationOp(
    name="parse_contract.imported_functions",
    version=1,
    scope="dataflow_audit",
)
_ANALYSIS_INDEX_STAGE_CACHE_OP = DerivationOp(
    name="analysis_index.stage_cache",
    version=1,
    scope="dataflow_audit",
)


def _annotation_allows_none(
    annotation,
    *,
    unparse_fn: Callable[[ast.AST], str] = ast.unparse,
) -> bool:
    if annotation is None:
        return True
    try:
        text = unparse_fn(annotation)
    except _AST_UNPARSE_ERROR_TYPES:
        return True
    normalized = text.replace(" ", "")
    return "None" in normalized or "Optional[" in normalized


def _parameter_default_map(
    node: ast.FunctionDef,
):
    check_deadline()
    mapping = {}
    positional = list(node.args.posonlyargs) + list(node.args.args)
    defaults = list(node.args.defaults)
    if defaults:
        defaults_checked = False
        for arg_node, default in zip(positional[-len(defaults) :], defaults):
            if not defaults_checked:
                check_deadline()
                defaults_checked = True
            mapping[arg_node.arg] = default
    kw_defaults_checked = False
    for arg_node, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        if not kw_defaults_checked:
            check_deadline()
            kw_defaults_checked = True
        mapping[arg_node.arg] = default
    return mapping


def _imported_helper_targets(
    tree: ast.Module,
) -> dict[str, tuple[str, str]]:
    targets: dict[str, tuple[str, str]] = {}
    for index, node in enumerate(tree.body):
        if index % _PARSE_CONTRACT_PROGRESS_INTERVAL == 0:
            check_deadline()
        if type(node) is ast.ImportFrom:
            import_from = cast(ast.ImportFrom, node)
            if not import_from.module:
                continue
            for alias in import_from.names:
                local_name = alias.asname or alias.name
                targets[local_name] = (import_from.module, alias.name)
    return targets


def _resolve_repo_module_path(
    module_name: str,
):
    if not module_name.startswith("gabion."):
        return None
    src_root = Path(__file__).resolve().parents[2]
    module_parts = module_name.split(".")
    module_path = src_root.joinpath(*module_parts).with_suffix(".py")
    if module_path.exists():
        return module_path
    package_init = src_root.joinpath(*module_parts, "__init__.py")
    if package_init.exists():
        return package_init
    return None


def _module_dependency_payload(
    module_path: Path,
) -> dict[str, object]:
    resolved = module_path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "mtime_ns": int(stat.st_mtime_ns),
        "size": int(stat.st_size),
    }


def _path_dependency_payload(
    path: Path,
) -> dict[str, object]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "mtime_ns": int(stat.st_mtime_ns),
        "size": int(stat.st_size),
    }


def _parse_contract_maps_from_source(
    source: str,
) -> tuple[dict[str, ast.FunctionDef], dict[str, tuple[str, str]]]:
    tree = ast.parse(source)
    functions: dict[str, ast.FunctionDef] = {}
    for index, node in enumerate(tree.body):
        if index % _PARSE_CONTRACT_PROGRESS_INTERVAL == 0:
            check_deadline()
        if type(node) is ast.FunctionDef:
            function_node = cast(ast.FunctionDef, node)
            functions[function_node.name] = function_node
    imported_helpers = _imported_helper_targets(tree)
    return functions, imported_helpers


def _module_function_map(
    module_path: Path,
) -> dict[str, ast.FunctionDef]:
    runtime = get_global_derivation_cache()
    dependencies = _module_dependency_payload(module_path)

    def _compute_functions() -> dict[str, ast.FunctionDef]:
        source = module_path.read_text()
        functions, _ = _parse_contract_maps_from_source(source)
        return functions

    result = runtime.derive(
        op=_PARSE_CONTRACT_IMPORTED_FUNCTIONS_OP,
        structural_inputs={"module_path": str(module_path.resolve())},
        dependencies=dependencies,
        params={"kind": "module_functions"},
        compute_fn=_compute_functions,
        source="dataflow_audit._module_function_map",
    )
    return cast(dict[str, ast.FunctionDef], result)


def _parse_witness_contract_violations(
    *,
    source = None,
    source_path = None,
    target_helpers = None,
    module_function_map_fn: Callable[[Path], dict[str, ast.FunctionDef]] = _module_function_map,
) -> list[str]:
    helpers = (
        _NON_NULL_PARSE_WITNESS_HELPERS if target_helpers is None else target_helpers
    )
    module_path = source_path or Path(__file__)
    functions: dict[str, ast.FunctionDef]
    imported_helpers: dict[str, tuple[str, str]]
    if source is None:
        try:
            runtime = get_global_derivation_cache()
            dependencies = _module_dependency_payload(module_path)

            def _compute_maps() -> tuple[
                dict[str, ast.FunctionDef],
                dict[str, tuple[str, str]],
            ]:
                source_text = module_path.read_text()
                return _parse_contract_maps_from_source(source_text)

            cached_maps = runtime.derive(
                op=_PARSE_CONTRACT_MAPS_OP,
                structural_inputs={"module_path": str(module_path.resolve())},
                dependencies=dependencies,
                params={"kind": "functions_and_imported_helpers"},
                compute_fn=_compute_maps,
                source="dataflow_audit._parse_witness_contract_violations",
            )
            functions, imported_helpers = cast(
                tuple[
                    dict[str, ast.FunctionDef],
                    dict[str, tuple[str, str]],
                ],
                cached_maps,
            )
        except OSError as exc:
            return [f"{module_path} parse_sink_contract read_error: {type(exc).__name__}"]
        except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError) as exc:
            return [f"{module_path} parse_sink_contract parse_error: {type(exc).__name__}"]
    else:
        try:
            functions, imported_helpers = _parse_contract_maps_from_source(source)
        except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError) as exc:
            return [f"{module_path} parse_sink_contract parse_error: {type(exc).__name__}"]
    violations: list[str] = []
    for helper_name in sort_once(
        helpers,
        source="_parse_witness_contract_violations.helpers",
    ):
        check_deadline()
        node = functions.get(helper_name)
        helper_module_path = module_path
        if node is None:
            imported_target = imported_helpers.get(helper_name)
            if imported_target is not None:
                imported_module_name, imported_symbol = imported_target
                imported_module_path = _resolve_repo_module_path(imported_module_name)
                if imported_module_path is not None:
                    try:
                        imported_functions = module_function_map_fn(imported_module_path)
                    except (
                        OSError,
                        SyntaxError,
                        ValueError,
                        TypeError,
                        MemoryError,
                        RecursionError,
                    ) as exc:
                        violations.append(
                            f"{imported_module_path}:{imported_symbol} parse_sink_contract import_parse_error: {type(exc).__name__}"
                        )
                        continue
                    node = imported_functions.get(imported_symbol)
                    if node is not None:
                        helper_module_path = imported_module_path
        if node is None:
            violations.append(
                f"{module_path}:{helper_name} parse_sink_contract missing helper definition"
            )
            continue
        params = list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
        param_node = next(
            (candidate for candidate in params if candidate.arg == "parse_failure_witnesses"),
            None,
        )
        if param_node is None:
            violations.append(
                f"{helper_module_path}:{helper_name} parse_sink_contract missing parse_failure_witnesses"
            )
            continue
        if _annotation_allows_none(param_node.annotation):
            violations.append(
                f"{helper_module_path}:{helper_name} parse_sink_contract parse_failure_witnesses must be total list[JSONObject]"
            )
        default_map = _parameter_default_map(node)
        default_node = default_map.get("parse_failure_witnesses")
        if (
            type(default_node) is ast.Constant
            and cast(ast.Constant, default_node).value is None
        ):
            violations.append(
                f"{helper_module_path}:{helper_name} parse_sink_contract parse_failure_witnesses must not default to None"
            )
    return violations


def _raw_sorted_baseline_key(path: Path) -> str:
    parts = path.parts
    if "src" in parts:
        start = parts.index("src")
        return str(Path(*parts[start:]))
    return str(path)


def _raw_sorted_callsite_counts(
    paths: Iterable[Path],
    *,
    parse_failure_witnesses: list[JSONObject],
) -> dict[str, list[tuple[int, int]]]:
    counts: dict[str, list[tuple[int, int]]] = {}
    for path in _iter_monotonic_paths(
        paths,
        source="_raw_sorted_callsite_counts.paths",
    ):
        check_deadline()
        if path.suffix != ".py":
            continue
        tree = _parse_module_tree(
            path,
            stage=_ParseModuleStage.RAW_SORTED_AUDIT,
            parse_failure_witnesses=parse_failure_witnesses,
        )
        if tree is not None:
            locations: list[tuple[int, int]] = []
            for node in ast.walk(tree):
                check_deadline()
                if type(node) is not ast.Call:
                    continue
                call_node = cast(ast.Call, node)
                if type(call_node.func) is not ast.Name:
                    continue
                if cast(ast.Name, call_node.func).id != "sorted":
                    continue
                line = int(getattr(call_node, "lineno", 1))
                col = int(getattr(call_node, "col_offset", 0)) + 1
                locations.append((line, col))
            if locations:
                counts[_raw_sorted_baseline_key(path)] = locations
    return counts


def _raw_sorted_contract_violations(
    paths: Iterable[Path],
    *,
    parse_failure_witnesses: list[JSONObject],
    strict_forbid = None,
    baseline_counts = None,
) -> list[str]:
    counts = _raw_sorted_callsite_counts(
        paths,
        parse_failure_witnesses=parse_failure_witnesses,
    )
    strict_forbid_flag = (
        os.environ.get(_FORBID_RAW_SORTED_ENV) == "1"
        if strict_forbid is None
        else bool(strict_forbid)
    )
    baseline_counts_map: Mapping[str, int] = (
        _RAW_SORTED_BASELINE_COUNTS if baseline_counts is None else baseline_counts
    )
    violations: list[str] = []
    for path in sort_once(
        counts,
        source="_raw_sorted_contract_violations.counts",
    ):
        check_deadline()
        baseline = baseline_counts_map.get(path)
        current = len(counts[path])
        if strict_forbid_flag:
            for line, col in counts[path]:
                check_deadline()
                violations.append(
                    f"{path}:{line}:{col} order_contract raw sorted() forbidden; use sort_once(...)"
                )
            continue
        if baseline is None:
            violations.append(
                f"{path} order_contract raw_sorted introduced count={current} baseline=0"
            )
            continue
        if current > baseline:
            violations.append(
                f"{path} order_contract raw_sorted exceeded baseline current={current} baseline={baseline}"
            )
    return violations


_INDEXED_PASS_INGRESS_RULE = _ExecutionPatternRule(
    pattern_id="indexed_pass_ingress",
    kind="execution_pattern",
    description=(
        "Functions sharing the indexed-pass ingress contract should be reified "
        "behind a typed pass metafactory."
    ),
)
_INDEXED_PASS_INGRESS_CORE_PARAMS = frozenset(
    {
        "paths",
        "project_root",
        "ignore_params",
        "strictness",
        "external_filter",
        "transparent_decorators",
        "parse_failure_witnesses",
        "analysis_index",
    }
)
_EXECUTION_PATTERN_RULES: tuple[_ExecutionPatternRule, ...] = (
    _INDEXED_PASS_INGRESS_RULE,
)


def _function_param_names(node: ast.FunctionDef) -> tuple[str, ...]:
    params: list[str] = []
    params.extend(arg.arg for arg in node.args.posonlyargs)
    params.extend(arg.arg for arg in node.args.args)
    params.extend(arg.arg for arg in node.args.kwonlyargs)
    return tuple(params)


def _indexed_pass_ingress_members(
    *,
    source = None,
    source_path = None,
) -> tuple[str, ...]:
    module_path = source_path or Path(__file__)
    if source is None:
        try:
            source = module_path.read_text()
        except OSError:
            return ()
    try:
        tree = ast.parse(source)
    except _PARSE_MODULE_ERROR_TYPES:
        return ()
    indexed_members: list[str] = []
    for node in tree.body:
        check_deadline()
        if type(node) is not ast.FunctionDef:
            continue
        function_node = cast(ast.FunctionDef, node)
        param_names = _function_param_names(function_node)
        if not _INDEXED_PASS_INGRESS_CORE_PARAMS.issubset(set(param_names)):
            continue
        calls_index_ingress = False
        for index, child in enumerate(ast.walk(function_node), start=1):
            if index % 64 == 0:
                check_deadline()
            if type(child) is not ast.Call:
                continue
            call_node = cast(ast.Call, child)
            if type(call_node.func) is not ast.Name:
                continue
            call_name = cast(ast.Name, call_node.func)
            if call_name.id in {"_build_analysis_index", "_build_call_graph"}:
                calls_index_ingress = True
                break
        if not calls_index_ingress:
            continue
        indexed_members.append(function_node.name)
    return tuple(
        sort_once(
            indexed_members,
            source="_indexed_pass_ingress_members.indexed_members",
        )
    )


def _detect_execution_pattern_matches(
    *,
    source = None,
    source_path = None,
) -> list[_ExecutionPatternMatch]:
    matches: list[_ExecutionPatternMatch] = []
    members = _indexed_pass_ingress_members(source=source, source_path=source_path)
    if len(members) >= 3:
        matches.append(
            _ExecutionPatternMatch(
                pattern_id=_INDEXED_PASS_INGRESS_RULE.pattern_id,
                kind=_INDEXED_PASS_INGRESS_RULE.kind,
                members=members,
                suggestion=(
                    f"{_INDEXED_PASS_INGRESS_RULE.pattern_id} members={len(members)} "
                    + ", ".join(members[:8])
                    + (" ..." if len(members) > 8 else "")
                    + "; candidate=IndexedPassSpec[T] metafactory"
                ),
            )
        )
    return matches


def _execution_pattern_instances(
    *,
    source = None,
    source_path = None,
) -> list[PatternInstance]:
    instances: list[PatternInstance] = []
    for match in _detect_execution_pattern_matches(
        source=source,
        source_path=source_path,
    ):
        check_deadline()
        signature: JSONObject = {
            "pattern_id": match.pattern_id,
            "members": list(match.members),
        }
        schema = PatternSchema.build(
            axis=PatternAxis.EXECUTION,
            kind=match.pattern_id,
            signature=signature,
            normalization={"members": list(match.members)},
        )
        instances.append(
            PatternInstance.build(
                schema=schema,
                members=match.members,
                suggestion=(
                    f"execution_pattern {match.suggestion}"
                ),
                residue=(
                    PatternResidue(
                        schema_id=schema.schema_id,
                        reason="unreified_metafactory",
                        payload=mismatch_residue_payload(
                            axis=PatternAxis.EXECUTION,
                            kind=match.pattern_id,
                            expected={"min_members": 3, "candidate": "IndexedPassSpec[T]"},
                            observed={"members": list(match.members), "member_count": len(match.members)},
                        ),
                    ),
                ),
            )
        )
    near_miss_members = _indexed_pass_ingress_members(
        source=source,
        source_path=source_path,
    )
    if len(near_miss_members) == 2:
        signature: JSONObject = {
            "pattern_id": _INDEXED_PASS_INGRESS_RULE.pattern_id,
            "members": list(near_miss_members),
        }
        schema = PatternSchema.build(
            axis=PatternAxis.EXECUTION,
            kind=_INDEXED_PASS_INGRESS_RULE.pattern_id,
            signature=signature,
            normalization={"members": list(near_miss_members)},
        )
        instances.append(
            PatternInstance.build(
                schema=schema,
                members=near_miss_members,
                suggestion=(
                    "execution_pattern near_miss "
                    + f"{_INDEXED_PASS_INGRESS_RULE.pattern_id} members={len(near_miss_members)}"
                ),
                residue=(
                    PatternResidue(
                        schema_id=schema.schema_id,
                        reason="schema_contract_mismatch",
                        payload=mismatch_residue_payload(
                            axis=PatternAxis.EXECUTION,
                            kind=_INDEXED_PASS_INGRESS_RULE.pattern_id,
                            expected={"min_members": 3},
                            observed={"members": list(near_miss_members), "member_count": 2},
                        ),
                    ),
                ),
            )
        )
    return instances


def _bundle_pattern_instances(
    *,
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
) -> list[PatternInstance]:
    if not groups_by_path:
        return []
    occurrences: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for path, by_fn in groups_by_path.items():
        check_deadline()
        for fn_name, bundles in by_fn.items():
            check_deadline()
            for bundle in bundles:
                check_deadline()
                key = tuple(
                    sort_once(
                        bundle,
                        source="_bundle_pattern_instances.bundle",
                    )
                )
                if len(key) < 2:
                    continue
                occurrences[key].append(f"{path.name}:{fn_name}")
    instances: list[PatternInstance] = []
    for bundle_key in sort_once(
        occurrences,
        source="_bundle_pattern_instances.occurrences",
    ):
        check_deadline()
        members = tuple(
            sort_once(
                set(occurrences[bundle_key]),
                source="_bundle_pattern_instances.members",
            )
        )
        count = len(members)
        if count <= 1:
            schema = PatternSchema.build(
                axis=PatternAxis.DATAFLOW,
                kind="bundle_signature",
                signature={
                    "bundle": list(bundle_key),
                    "tier": 3,
                    "site_count": count,
                },
                normalization={"bundle": list(bundle_key)},
            )
            instances.append(
                PatternInstance.build(
                    schema=schema,
                    members=members,
                    suggestion=(
                        "dataflow_pattern near_miss "
                        + f"bundle={','.join(bundle_key)} sites={count}"
                    ),
                    residue=(
                        PatternResidue(
                            schema_id=schema.schema_id,
                            reason="schema_contract_mismatch",
                            payload=mismatch_residue_payload(
                                axis=PatternAxis.DATAFLOW,
                                kind="bundle_signature",
                                expected={"min_sites": 2, "tier": 2},
                                observed={"site_count": count, "tier": 3, "bundle": list(bundle_key)},
                            ),
                        ),
                    ),
                )
            )
            continue
        signature: JSONObject = {
            "bundle": list(bundle_key),
            "tier": 2 if count > 1 else 3,
            "site_count": count,
        }
        schema = PatternSchema.build(
            axis=PatternAxis.DATAFLOW,
            kind="bundle_signature",
            signature=signature,
            normalization={"bundle": list(bundle_key)},
        )
        instances.append(
            PatternInstance.build(
                schema=schema,
                members=members,
                suggestion=(
                    "dataflow_pattern "
                    + f"bundle={','.join(bundle_key)} sites={count}; "
                    + "candidate=Protocol/dataclass reification"
                ),
                residue=(
                    PatternResidue(
                        schema_id=schema.schema_id,
                        reason="unreified_protocol",
                        payload=mismatch_residue_payload(
                            axis=PatternAxis.DATAFLOW,
                            kind="bundle_signature",
                            expected={
                                "candidate": "Protocol/dataclass reification",
                                "tier": 2,
                            },
                            observed={
                                "bundle": list(bundle_key),
                                "site_count": count,
                                "tier": 2 if count > 1 else 3,
                            },
                        ),
                    ),
                ),
            )
        )
    return instances


def _pattern_schema_matches(
    *,
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    include_execution: bool = True,
    source = None,
    source_path = None,
) -> list[PatternInstance]:
    instances: list[PatternInstance] = []
    if include_execution:
        instances.extend(
            _execution_pattern_instances(
                source=source,
                source_path=source_path,
            )
        )
    instances.extend(
        _bundle_pattern_instances(
            groups_by_path=groups_by_path,
        )
    )
    return sort_once(
        instances,
        source="_pattern_schema_matches.instances",
        key=lambda entry: (
            entry.schema.axis.value,
            entry.schema.kind,
            entry.schema.schema_id,
            entry.suggestion,
        ),
    )


def _pattern_schema_suggestions(
    *,
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    include_execution: bool = True,
    source = None,
    source_path = None,
) -> list[str]:
    instances = _pattern_schema_matches(
        groups_by_path=groups_by_path,
        include_execution=include_execution,
        source=source,
        source_path=source_path,
    )
    return _pattern_schema_suggestions_from_instances(instances)


def _pattern_schema_suggestions_from_instances(
    instances: Sequence[PatternInstance],
) -> list[str]:
    suggestions: list[str] = []
    for instance in instances:
        check_deadline()
        suggestions.append(
            f"pattern_schema axis={instance.schema.axis.value} {instance.suggestion}"
        )
    return sort_once(
        set(suggestions),
        source="_pattern_schema_suggestions_from_instances.suggestions",
    )


def _pattern_schema_residue_entries(
    instances: Sequence[PatternInstance],
) -> list[PatternResidue]:
    entries: list[PatternResidue] = []
    for instance in instances:
        check_deadline()
        entries.extend(instance.residue)
    return sort_once(
        entries,
        source="_pattern_schema_residue_entries.entries",
        key=lambda entry: (
            entry.schema_id,
            entry.reason,
            json.dumps(entry.payload, sort_keys=False, separators=(",", ":")),
        ),
    )


def _pattern_schema_residue_lines(entries: Sequence[PatternResidue]) -> list[str]:
    lines: list[str] = []
    for entry in entries:
        check_deadline()
        payload = json.dumps(entry.payload, sort_keys=False)
        lines.append(
            f"schema_id={entry.schema_id} reason={entry.reason} payload={payload}"
        )
    return lines


def _pattern_schema_snapshot_entries(
    instances: Sequence[PatternInstance],
) -> tuple[list[JSONObject], list[JSONObject]]:
    serialized_instances: list[JSONObject] = []
    for instance in instances:
        check_deadline()
        serialized_instances.append(
            {
                    "schema": {
                        "schema_id": instance.schema.schema_id,
                        "legacy_schema_id": instance.schema.legacy_schema_id,
                        "schema_contract": instance.schema.schema_contract,
                        "axis": instance.schema.axis.value,
                        "kind": instance.schema.kind,
                        "signature": instance.schema.normalized_signature,
                        "normalization": instance.schema.normalization,
                    },
                "members": list(instance.members),
                "suggestion": instance.suggestion,
                "residue": [
                    {
                        "schema_id": residue.schema_id,
                        "reason": residue.reason,
                        "payload": residue.payload,
                    }
                    for residue in sort_once(
                        instance.residue,
                        source="_pattern_schema_snapshot_entries.instance_residue",
                        key=lambda entry: (
                            entry.schema_id,
                            entry.reason,
                            json.dumps(entry.payload, sort_keys=False, separators=(",", ":")),
                        ),
                    )
                ],
            }
        )
    residues = _pattern_schema_residue_entries(instances)
    serialized_residue: list[JSONObject] = []
    for entry in residues:
        check_deadline()
        serialized_residue.append(
            {
                "schema_id": entry.schema_id,
                "reason": entry.reason,
                "payload": entry.payload,
            }
        )
    return serialized_instances, serialized_residue


def _execution_pattern_suggestions(
    *,
    source = None,
    source_path = None,
) -> list[str]:
    suggestions: list[str] = []
    for instance in _execution_pattern_instances(
        source=source,
        source_path=source_path,
    ):
        check_deadline()
        suggestions.append(instance.suggestion)
    return sort_once(
        set(suggestions),
        source="_execution_pattern_suggestions.suggestions",
    )


def _parse_module_tree(
    path: Path,
    *,
    stage: _ParseModuleStage,
    parse_failure_witnesses: list[JSONObject],
):
    try:
        return ast.parse(path.read_text())
    except _PARSE_MODULE_ERROR_TYPES as exc:
        _record_parse_failure_witness(
            sink=parse_failure_witnesses,
            path=path,
            stage=stage,
            error=exc,
        )
        return None


def _param_annotations_by_path(
    paths: list[Path],
    *,
    ignore_params: set[str],
    parse_failure_witnesses: list[JSONObject],
) -> dict[Path, dict[str, ParamAnnotationMap]]:
    check_deadline()
    annotations: dict[Path, dict[str, ParamAnnotationMap]] = {}
    for path in paths:
        check_deadline()
        tree = _parse_module_tree(
            path,
            stage=_ParseModuleStage.PARAM_ANNOTATIONS,
            parse_failure_witnesses=parse_failure_witnesses,
        )
        if tree is not None:
            parent = ParentAnnotator()
            parent.visit(tree)
            parents = parent.parents
            by_fn: dict[str, ParamAnnotationMap] = {}
            for fn in _collect_functions(tree):
                check_deadline()
                scopes = _enclosing_scopes(fn, parents)
                fn_key = _function_key(scopes, fn.name)
                by_fn[fn_key] = _param_annotations(fn, ignore_params)
            annotations[path] = by_fn
    return annotations


def _compute_fingerprint_warnings(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    annotations_by_path: dict[Path, dict[str, ParamAnnotationMap]],
    *,
    registry: PrimeRegistry,
    index: dict[Fingerprint, set[str]],
    ctor_registry = None,
) -> list[str]:
    check_deadline()
    warnings: list[str] = []
    if not index:
        return warnings
    for path, groups in groups_by_path.items():
        check_deadline()
        annots_by_fn = annotations_by_path.get(path, {})
        for fn_name, bundles in groups.items():
            check_deadline()
            fn_annots = annots_by_fn.get(fn_name, {})
            for bundle in bundles:
                check_deadline()
                missing = [param for param in bundle if not fn_annots.get(param)]
                bundle_params = sort_once(
                    bundle,
                    source="_compute_fingerprint_warnings.bundle",
                )
                if missing:
                    warnings.append(
                        f"{path.name}:{fn_name} bundle {bundle_params} missing type annotations: "
                        + ", ".join(
                            sort_once(
                                missing,
                                source="_compute_fingerprint_warnings.missing",
                            )
                        )
                    )
                else:
                    types = [fn_annots[param] for param in bundle_params]
                    if not any(t is None for t in types):
                        hint_list = [t for t in types if t is not None]
                        fingerprint = bundle_fingerprint_dimensional(
                            hint_list,
                            registry,
                            ctor_registry,
                        )
                        soundness_issues = _fingerprint_soundness_issues(fingerprint)
                        names = index.get(fingerprint)

                        base_keys, base_remaining = fingerprint.base.keys_with_remainder(
                            registry
                        )
                        ctor_keys, ctor_remaining = fingerprint.ctor.keys_with_remainder(
                            registry
                        )
                        ctor_keys = [
                            key[len("ctor:") :] if key.startswith("ctor:") else key
                            for key in ctor_keys
                        ]
                        base_keys_sorted = sort_once(
                            base_keys,
                            source="_compute_fingerprint_warnings.base_keys",
                        )
                        ctor_keys_sorted = sort_once(
                            ctor_keys,
                            source="_compute_fingerprint_warnings.ctor_keys",
                        )
                        details = f" base={base_keys_sorted}"
                        if ctor_keys:
                            details += f" ctor={ctor_keys_sorted}"
                        if base_remaining not in (0, 1) or ctor_remaining not in (0, 1):
                            details += f" remainder=({base_remaining},{ctor_remaining})"
                        if soundness_issues:
                            warnings.append(
                                f"{path.name}:{fn_name} bundle {bundle_params} fingerprint carrier soundness failed for "
                                + ", ".join(soundness_issues)
                                + details
                            )
                        if not names:
                            warnings.append(
                                f"{path.name}:{fn_name} bundle {bundle_params} fingerprint missing glossary match{details}"
                            )
    return sort_once(
        set(warnings),
        source="_compute_fingerprint_warnings.warnings",
    )


def _collect_fingerprint_atom_keys(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    annotations_by_path: dict[Path, dict[str, ParamAnnotationMap]],
) -> tuple[list[str], list[str]]:
    check_deadline()
    base_keys: set[str] = set()
    ctor_keys: set[str] = set()
    for path, groups in groups_by_path.items():
        check_deadline()
        annots_by_fn = annotations_by_path.get(path, {})
        for fn_name, bundles in groups.items():
            check_deadline()
            fn_annots = annots_by_fn.get(fn_name, {})
            for bundle in bundles:
                check_deadline()
                for param_name in bundle:
                    check_deadline()
                    hint = fn_annots.get(param_name)
                    if hint is not None:
                        atoms: list[str] = []
                        _collect_base_atoms(hint, atoms)
                        base_keys.update(atom for atom in atoms if atom)
                        _collect_constructors(hint, ctor_keys)
    return (
        sort_once(base_keys, source="_collect_fingerprint_atom_keys.base_keys"),
        sort_once(ctor_keys, source="_collect_fingerprint_atom_keys.ctor_keys"),
    )


def _compute_fingerprint_matches(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    annotations_by_path: dict[Path, dict[str, ParamAnnotationMap]],
    *,
    registry: PrimeRegistry,
    index: dict[Fingerprint, set[str]],
    ctor_registry = None,
) -> list[str]:
    check_deadline()
    matches: list[str] = []
    if not index:
        return matches
    for path, groups in groups_by_path.items():
        check_deadline()
        annots_by_fn = annotations_by_path.get(path, {})
        for fn_name, bundles in groups.items():
            check_deadline()
            fn_annots = annots_by_fn.get(fn_name, {})
            for bundle in bundles:
                check_deadline()
                missing = [param for param in bundle if param not in fn_annots]
                if not missing:
                    bundle_params = sort_once(
                        bundle,
                        source="_compute_fingerprint_matches.bundle",
                    )
                    types = [fn_annots[param] for param in bundle_params]
                    if not any(t is None for t in types):
                        hint_list = [t for t in types if t is not None]
                        fingerprint = bundle_fingerprint_dimensional(
                            hint_list,
                            registry,
                            ctor_registry,
                        )
                        names = index.get(fingerprint)
                        if names:
                            base_keys, base_remaining = fingerprint.base.keys_with_remainder(
                                registry
                            )
                            ctor_keys, ctor_remaining = fingerprint.ctor.keys_with_remainder(
                                registry
                            )
                            ctor_keys = [
                                key[len("ctor:") :] if key.startswith("ctor:") else key
                                for key in ctor_keys
                            ]
                            base_keys_sorted = sort_once(
                                base_keys,
                                source="_compute_fingerprint_matches.base_keys",
                            )
                            ctor_keys_sorted = sort_once(
                                ctor_keys,
                                source="_compute_fingerprint_matches.ctor_keys",
                            )
                            details = f" base={base_keys_sorted}"
                            if ctor_keys:
                                details += f" ctor={ctor_keys_sorted}"
                            if (
                                base_remaining not in (0, 1)
                                or ctor_remaining not in (0, 1)
                            ):
                                details += f" remainder=({base_remaining},{ctor_remaining})"
                            matches.append(
                                f"{path.name}:{fn_name} bundle {bundle_params} fingerprint {format_fingerprint(fingerprint)} matches: "
                                + ", ".join(
                                    sort_once(
                                        names,
                                        source="_compute_fingerprint_matches.names",
                                    )
                                )
                                + details
                            )
    return sort_once(
        set(matches),
        source="_compute_fingerprint_matches.matches",
    )


def _fingerprint_soundness_issues(
    fingerprint: Fingerprint,
) -> list[str]:
    check_deadline()
    def _is_empty(dim: FingerprintDimension) -> bool:
        return dim.product in (0, 1) and dim.mask == 0

    pairs = [
        ("base/ctor", fingerprint.base, fingerprint.ctor),
        ("base/provenance", fingerprint.base, fingerprint.provenance),
        ("base/synth", fingerprint.base, fingerprint.synth),
        ("ctor/provenance", fingerprint.ctor, fingerprint.provenance),
        ("ctor/synth", fingerprint.ctor, fingerprint.synth),
        ("provenance/synth", fingerprint.provenance, fingerprint.synth),
    ]
    issues: list[str] = []
    for label, left, right in pairs:
        check_deadline()
        if _is_empty(left) or _is_empty(right):
            continue
        if not fingerprint_carrier_soundness(left, right):
            issues.append(label)
    return issues


def _compute_fingerprint_provenance(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    annotations_by_path: dict[Path, dict[str, dict[str, object]]],
    *,
    registry: PrimeRegistry,
    project_root = None,
    index = None,
    ctor_registry = None,
) -> list[JSONObject]:
    check_deadline()
    entries: list[JSONObject] = []
    for path, groups in groups_by_path.items():
        check_deadline()
        path_value = _normalize_snapshot_path(path, project_root)
        annots_by_fn = annotations_by_path.get(path, {})
        for fn_name, bundles in groups.items():
            check_deadline()
            fn_annots = annots_by_fn.get(fn_name, {})
            for bundle in bundles:
                check_deadline()
                missing = [param for param in bundle if param not in fn_annots]
                if missing:
                    continue
                bundle_params = sort_once(
                    bundle,
                    source="_compute_fingerprint_provenance.bundle",
                )
                types = [fn_annots[param] for param in bundle_params]
                hint_list = [str(value) for value in types if value is not None]
                if len(hint_list) != len(types):
                    continue
                fingerprint = bundle_fingerprint_dimensional(
                    hint_list,
                    registry,
                    ctor_registry,
                )
                soundness_issues = _fingerprint_soundness_issues(fingerprint)
                base_keys, base_remaining = fingerprint.base.keys_with_remainder(registry)
                ctor_keys, ctor_remaining = fingerprint.ctor.keys_with_remainder(registry)
                ctor_keys = [
                    key[len("ctor:") :] if key.startswith("ctor:") else key
                    for key in ctor_keys
                ]
                matches = []
                if index:
                    matches = sort_once(
                        index.get(fingerprint, set()),
                        source="_compute_fingerprint_provenance.matches",
                    )
                identity_payload = fingerprint_identity_payload(fingerprint)
                representative = str(
                    identity_payload["identity_layers"]["canonical"]["representative"]
                )
                basis_repr = "|".join(
                    str(item)
                    for item in identity_payload["identity_layers"]["canonical"].get(
                        "basis_path", []
                    )
                )
                higher_path_payload = identity_payload.get("witness_carriers", {}).get(
                    "higher_path_witness"
                )
                higher_path_witness = (
                    parse_2cell_witness(higher_path_payload)
                    if type(higher_path_payload) is dict
                    else None
                )
                drift_classification = classify_drift_by_homotopy(
                    baseline_representative=representative,
                    current_representative=basis_repr,
                    equivalence_witness=higher_path_witness,
                )
                bundle_key = ",".join(bundle_params)
                entries.append(
                    {
                        "provenance_id": f"{path_value}:{fn_name}:{bundle_key}",
                        "path": path_value,
                        "function": fn_name,
                        "bundle": bundle_params,
                        "fingerprint": {
                            "base": {
                                "product": fingerprint.base.product,
                                "mask": fingerprint.base.mask,
                            },
                            "ctor": {
                                "product": fingerprint.ctor.product,
                                "mask": fingerprint.ctor.mask,
                            },
                            "provenance": {
                                "product": fingerprint.provenance.product,
                                "mask": fingerprint.provenance.mask,
                            },
                            "synth": {
                                "product": fingerprint.synth.product,
                                "mask": fingerprint.synth.mask,
                            },
                        },
                        "base_keys": sort_once(
                            base_keys,
                            source="_compute_fingerprint_provenance.base_keys",
                        ),
                        "ctor_keys": sort_once(
                            ctor_keys,
                            source="_compute_fingerprint_provenance.ctor_keys",
                        ),
                        "remainder": {
                            "base": base_remaining,
                            "ctor": ctor_remaining,
                        },
                        "soundness_issues": soundness_issues,
                        "glossary_matches": matches,
                        "canonical_identity_contract": identity_payload[
                            "canonical_identity_contract"
                        ],
                        "identity_layers": identity_payload["identity_layers"],
                        "representative_selection": identity_payload[
                            "representative_selection"
                        ],
                        "witness_carriers": identity_payload["witness_carriers"],
                        "derived_aliases": identity_payload["derived_aliases"],
                        "cofibration_witness": identity_payload.get(
                            "cofibration_witness", {"entries": []}
                        ),
                        "drift_classification": drift_classification,
                    }
                )
    return entries


def _summarize_fingerprint_provenance(
    entries: list[JSONObject],
    *,
    max_groups: int = 20,
    max_examples: int = 3,
) -> list[str]:
    check_deadline()
    if not entries:
        return []
    grouped: dict[tuple[object, ...], list[JSONObject]] = {}
    for entry in entries:
        check_deadline()
        matches = str_tuple_from_sequence(entry.get("glossary_matches")) or tuple()
        if matches:
            key = ("glossary", matches)
        else:
            base_keys = str_tuple_from_sequence(entry.get("base_keys")) or tuple()
            ctor_keys = str_tuple_from_sequence(entry.get("ctor_keys")) or tuple()
            key = ("types", base_keys, ctor_keys)
        grouped.setdefault(key, []).append(entry)
    lines: list[str] = []
    grouped_entries = sort_once(
        grouped.items(),
        source="_summarize_fingerprint_provenance.grouped",
        key=lambda item: (-len(item[1]), item[0]),
    )
    for key, group in grouped_entries[:max_groups]:
        check_deadline()
        label = ""
        if key and key[0] == "glossary":
            label = "glossary=" + ", ".join(key[1])
        else:
            base_keys = list(key[1])
            ctor_keys = list(key[2])
            label = f"base={base_keys}"
            if ctor_keys:
                label += f" ctor={ctor_keys}"
        lines.append(f"- {label} occurrences={len(group)}")
        for entry in group[:max_examples]:
            check_deadline()
            path = entry.get("path")
            fn_name = entry.get("function")
            bundle = entry.get("bundle")
            lines.append(f"  - {path}:{fn_name} bundle={bundle}")
        if len(group) > max_examples:
            lines.append(f"  - ... ({len(group) - max_examples} more)")
    return lines


def _summarize_deadness_witnesses(
    entries: list[JSONObject],
    *,
    max_entries: int = 10,
) -> list[str]:
    return _ds_summarize_deadness_witnesses(
        entries,
        max_entries=max_entries,
        check_deadline=check_deadline,
    )


def _compute_fingerprint_coherence(
    entries: list[JSONObject],
    *,
    synth_version: str,
) -> list[JSONObject]:
    return _ds_compute_fingerprint_coherence(
        entries,
        synth_version=synth_version,
        check_deadline=check_deadline,
        ordered_or_sorted=sort_once,
    )


def _summarize_coherence_witnesses(
    entries: list[JSONObject],
    *,
    max_entries: int = 10,
) -> list[str]:
    return _ds_summarize_coherence_witnesses(
        entries,
        max_entries=max_entries,
        check_deadline=check_deadline,
    )


def _compute_fingerprint_rewrite_plans(
    provenance: list[JSONObject],
    coherence: list[JSONObject],
    *,
    synth_version: str,
    exception_obligations = None,
) -> list[JSONObject]:
    return _ds_compute_fingerprint_rewrite_plans(
        provenance,
        coherence,
        synth_version=synth_version,
        exception_obligations=exception_obligations,
        check_deadline=check_deadline,
        ordered_or_sorted=sort_once,
        site_from_payload=Site.from_payload,
    )

def _glossary_match_strata(matches: Sequence[object]) -> str:
    if not matches:
        return "none"
    if len(matches) == 1:
        return "exact"
    return "ambiguous"


def _find_provenance_entry_for_site(
    provenance: list[JSONObject],
    *,
    site: Site,
):
    check_deadline()
    target_key = site.key()
    for entry in provenance:
        check_deadline()
        entry_site = Site.from_payload(entry)
        if entry_site is not None and entry_site.key() == target_key:
            return entry
    return None


def _exception_obligation_summary_for_site(
    obligations: list[JSONObject],
    *,
    site: Site,
) -> dict[str, int]:
    return exception_obligation_summary_for_site(obligations, site=site)


@dataclass(frozen=True)
class _RewritePredicateContext:
    expected_base: list[object]
    expected_ctor: list[object]
    expected_remainder: Mapping[str, object]
    expected_strata: str
    expected_candidates: list[str]
    post_base: list[object]
    post_ctor: list[object]
    post_remainder: Mapping[str, object]
    post_matches: tuple[str, ...]
    post_strata: str
    post_exception_obligations: OptionalJsonObjectList
    pre: Mapping[str, object]
    plan_evidence: Mapping[str, object]
    post_entry: Mapping[str, object]
    site: Site


def _evaluate_base_conservation_predicate(
    predicate: JSONObject,
    context: _RewritePredicateContext,
) -> JSONObject:
    kind = str(predicate.get("kind", ""))
    base_ok = context.post_base == context.expected_base
    return {
        "kind": kind,
        "passed": base_ok,
        "expected": context.expected_base,
        "observed": context.post_base,
    }


def _evaluate_ctor_coherence_predicate(
    predicate: JSONObject,
    context: _RewritePredicateContext,
) -> JSONObject:
    kind = str(predicate.get("kind", ""))
    ctor_ok = context.post_ctor == context.expected_ctor
    return {
        "kind": kind,
        "passed": ctor_ok,
        "expected": context.expected_ctor,
        "observed": context.post_ctor,
    }


def _evaluate_match_strata_predicate(
    predicate: JSONObject,
    context: _RewritePredicateContext,
) -> JSONObject:
    kind = str(predicate.get("kind", ""))
    strata_expect = str(predicate.get("expect", context.expected_strata) or "")
    candidates = [
        str(item)
        for item in (predicate.get("candidates") or context.expected_candidates)
        if item
    ]
    strata_ok = True
    if strata_expect:
        strata_ok = context.post_strata == strata_expect
    if (
        strata_expect == "exact"
        and len(context.post_matches) == 1
    ):
        strata_ok = strata_ok and (str(context.post_matches[0]) in set(candidates))
    return {
        "kind": kind,
        "passed": strata_ok,
        "expected": strata_expect,
        "observed": context.post_strata,
        "candidates": candidates,
        "observed_matches": list(context.post_matches),
    }


def _remainder_clean(value: int) -> bool:
    return value in (0, 1)


def _evaluate_remainder_non_regression_predicate(
    predicate: JSONObject,
    context: _RewritePredicateContext,
) -> JSONObject:
    kind = str(predicate.get("kind", ""))
    pre_base_rem = int(context.expected_remainder.get("base", 1) or 1)
    pre_ctor_rem = int(context.expected_remainder.get("ctor", 1) or 1)
    post_base_rem = int(context.post_remainder.get("base", 1) or 1)
    post_ctor_rem = int(context.post_remainder.get("ctor", 1) or 1)
    rem_ok = True
    if _remainder_clean(pre_base_rem):
        rem_ok = rem_ok and _remainder_clean(post_base_rem)
    if _remainder_clean(pre_ctor_rem):
        rem_ok = rem_ok and _remainder_clean(post_ctor_rem)
    return {
        "kind": kind,
        "passed": rem_ok,
        "expected": {"base": pre_base_rem, "ctor": pre_ctor_rem},
        "observed": {"base": post_base_rem, "ctor": post_ctor_rem},
    }




def _evaluate_witness_obligation_non_regression_predicate(
    predicate: JSONObject,
    context: _RewritePredicateContext,
) -> JSONObject:
    kind = str(predicate.get("kind", ""))
    evidence = context.plan_evidence
    obligations: list[Mapping[str, JSONValue]] = []
    raw_obligations = sequence_or_none(evidence.get("witness_obligations"))
    if raw_obligations is not None:
        for item in raw_obligations:
            mapped_item = mapping_or_none(item)
            if mapped_item is not None:
                obligations.append(mapped_item)
    missing_required: list[str] = []
    for item in obligations:
        required = bool(item.get("required"))
        witness_ref = str(item.get("witness_ref", "") or "")
        witness_kind = str(item.get("kind", "witness") or "witness")
        if required and not witness_ref:
            missing_required.append(f"{witness_kind}:missing")
    post_identity = context.post_entry.get("canonical_identity_contract")
    pre_identity = context.pre.get("canonical_identity_contract")
    identity_ok = True
    if pre_identity is not None:
        identity_ok = pre_identity == post_identity
    elif post_identity is None:
        identity_ok = False
    passed = (not missing_required) and identity_ok
    return {
        "kind": kind,
        "passed": passed,
        "expected": {
            "required_witnesses": [
                item for item in obligations if bool(item.get("required"))
            ],
            "identity_contract": pre_identity,
        },
        "observed": {
            "missing_required": missing_required,
            "identity_contract": post_identity,
        },
    }

def _summary_unknown_and_discharged(summary: Mapping[str, object]) -> tuple[int, int]:
    try:
        unknown = int(summary.get("UNKNOWN", 0) or 0)
        discharged = int(summary.get("DEAD", 0) or 0) + int(summary.get("HANDLED", 0) or 0)
    except (TypeError, ValueError):
        unknown = 0
        discharged = 0
    return unknown, discharged


def _evaluate_exception_obligation_non_regression_predicate(
    predicate: JSONObject,
    context: _RewritePredicateContext,
) -> JSONObject:
    kind = str(predicate.get("kind", ""))
    raw_pre_summary = context.pre.get("exception_obligations_summary")
    pre_summary = cast(Mapping[str, object] | None, mapping_or_none(raw_pre_summary))
    if context.post_exception_obligations is None:
        return {
            "kind": kind,
            "passed": False,
            "expected": pre_summary,
            "observed": None,
            "issue": "missing post exception obligations",
        }
    if pre_summary is None:
        return {
            "kind": kind,
            "passed": False,
            "expected": None,
            "observed": None,
            "issue": "missing pre exception obligations summary",
        }
    post_summary = _exception_obligation_summary_for_site(
        context.post_exception_obligations,
        site=context.site,
    )
    pre_unknown, pre_discharged = _summary_unknown_and_discharged(pre_summary)
    post_unknown, post_discharged = _summary_unknown_and_discharged(post_summary)
    exc_ok = (post_unknown <= pre_unknown) and (post_discharged >= pre_discharged)
    return {
        "kind": kind,
        "passed": exc_ok,
        "expected": {"UNKNOWN": pre_unknown, "DISCHARGED": pre_discharged},
        "observed": {"UNKNOWN": post_unknown, "DISCHARGED": post_discharged},
        "pre_summary": pre_summary,
        "post_summary": post_summary,
    }


_REWRITE_PREDICATE_EVALUATORS: Mapping[
    str,
    Callable[[JSONObject, _RewritePredicateContext], JSONObject],
] = {
    "base_conservation": _evaluate_base_conservation_predicate,
    "ctor_coherence": _evaluate_ctor_coherence_predicate,
    "match_strata": _evaluate_match_strata_predicate,
    "remainder_non_regression": _evaluate_remainder_non_regression_predicate,
    "witness_obligation_non_regression": _evaluate_witness_obligation_non_regression_predicate,
    "exception_obligation_non_regression": _evaluate_exception_obligation_non_regression_predicate,
}


def _evaluate_rewrite_predicate(
    predicate: JSONObject,
    *,
    context: _RewritePredicateContext,
) -> JSONObject:
    kind = str(predicate.get("kind", ""))
    evaluator = _REWRITE_PREDICATE_EVALUATORS.get(kind)
    if evaluator is None:
        return {
            "kind": kind,
            "passed": False,
            "expected": predicate.get("expect"),
            "observed": None,
            "issue": "unknown predicate kind",
        }
    return evaluator(predicate, context)


def verify_rewrite_plan(
    plan: JSONObject,
    *,
    post_provenance: list[JSONObject],
    post_exception_obligations = None,
) -> JSONObject:
    """Verify a single rewrite plan using a post-state provenance artifact.

    The pre-state is taken from the plan's embedded boundary evidence; the
    evaluator only needs the post provenance entry for the plan's site.
    """
    check_deadline()
    plan_id = str(plan.get("plan_id", ""))
    raw_site = plan.get("site", {}) or {}
    site = Site.from_payload(raw_site)
    if site is None or not site.path or not site.function:
        return {
            "plan_id": plan_id,
            "accepted": False,
            "issues": ["missing or invalid plan site"],
            "predicate_results": [],
        }
    path = site.path
    function = site.function
    bundle = list(site.bundle)

    issues: list[str] = []
    status = str(plan.get("status", "") or "")
    if status == "ABSTAINED":
        issues.append("plan abstained: preconditions not satisfied")
        abstention = mapping_or_empty(plan.get("abstention"))
        if abstention.get("reason"):
            issues.append(f"abstention reason: {abstention.get('reason')}")
        return {
            "plan_id": plan_id,
            "accepted": False,
            "issues": issues,
            "predicate_results": [],
        }

    schema_issues = [
        issue
        for issue in validate_rewrite_plan_payload(plan)
        if not issue.startswith("missing predicate:")
        and not issue.startswith("missing evidence refs:")
        and not issue.startswith("unknown rewrite kind:")
    ]
    if schema_issues:
        issues.extend(schema_issues)

    post_entry = _find_provenance_entry_for_site(
        post_provenance,
        site=site,
    )
    if post_entry is None:
        issues.append("missing post provenance entry for site")
        return {
            "plan_id": plan_id,
            "accepted": False,
            "issues": issues,
            "predicate_results": [],
        }

    pre = mapping_or_empty(plan.get("pre"))
    evidence = mapping_or_empty(plan.get("evidence"))
    raw_pre_remainder = pre.get("remainder")
    if raw_pre_remainder not in (None, {}) and mapping_or_none(raw_pre_remainder) is None:
        issues.append("invalid pre remainder payload")
    expected_base = list(pre.get("base_keys") or [])
    expected_ctor = list(pre.get("ctor_keys") or [])
    expected_remainder = mapping_or_empty(pre.get("remainder"))
    post_expectation = mapping_or_empty(plan.get("post_expectation"))
    expected_strata = str(post_expectation.get("match_strata", ""))

    post_base = list(post_entry.get("base_keys") or [])
    post_ctor = list(post_entry.get("ctor_keys") or [])
    post_remainder = mapping_or_empty(post_entry.get("remainder"))
    post_matches = str_tuple_from_sequence(post_entry.get("glossary_matches")) or tuple()
    post_strata = _glossary_match_strata(post_matches)

    predicate_results: list[JSONObject] = []

    expected_candidates: list[str] = []
    rewrite = mapping_or_empty(plan.get("rewrite"))
    rewrite_kind = str(rewrite.get("kind", "") or "")
    raw_params_payload = rewrite.get("parameters")
    if raw_params_payload not in (None, {}) and mapping_or_none(raw_params_payload) is None:
        issues.append("invalid rewrite parameters payload")
    params = mapping_or_empty(rewrite.get("parameters"))
    expected_candidates = [str(v) for v in (params.get("candidates") or []) if v]

    requested_predicates: list[JSONObject] = []
    verification = mapping_or_empty(plan.get("verification"))
    predicates = sequence_or_none(verification.get("predicates")) or ()
    requested_predicates = [
        {str(key): mapped_predicate[key] for key in mapped_predicate}
        for predicate in predicates
        for mapped_predicate in (mapping_or_none(predicate),)
        if mapped_predicate is not None and mapped_predicate.get("kind")
    ]
    if not requested_predicates:
        schema_lookup = rewrite_plan_schema(rewrite_kind)
        defaults = (
            list(schema_lookup.schema.required_predicates)
            if schema_lookup.is_known
            else [
            "base_conservation",
            "ctor_coherence",
            "match_strata",
            "remainder_non_regression",
            ]
        )
        requested_predicates = []
        for kind in defaults:
            if kind == "match_strata":
                requested_predicates.append(
                    {
                        "kind": kind,
                        "expect": expected_strata,
                        "candidates": expected_candidates,
                    }
                )
            elif kind == "remainder_non_regression":
                requested_predicates.append(
                    {"kind": kind, "expect": "no-new-remainder"}
                )
            else:
                requested_predicates.append({"kind": kind, "expect": True})

    predicate_context = _RewritePredicateContext(
        expected_base=expected_base,
        expected_ctor=expected_ctor,
        expected_remainder=expected_remainder,
        expected_strata=expected_strata,
        expected_candidates=expected_candidates,
        post_base=post_base,
        post_ctor=post_ctor,
        post_remainder=post_remainder,
        post_matches=post_matches,
        post_strata=post_strata,
        post_exception_obligations=post_exception_obligations,
        pre=pre,
        plan_evidence=evidence,
        post_entry=post_entry,
        site=site,
    )

    for predicate in requested_predicates:
        check_deadline()
        predicate_results.append(
            _evaluate_rewrite_predicate(
                predicate,
                context=predicate_context,
            )
        )

    accepted = (not issues) and all(bool(result.get("passed")) for result in predicate_results)
    if predicate_results and not all(bool(result.get("passed")) for result in predicate_results):
        issues.append("verification predicates failed")
    return {
        "plan_id": plan_id,
        "accepted": accepted,
        "issues": issues,
        "predicate_results": predicate_results,
    }


def verify_rewrite_plans(
    plans: list[JSONObject],
    *,
    post_provenance: list[JSONObject],
    post_exception_obligations = None,
) -> list[JSONObject]:
    return [
        verify_rewrite_plan(
            plan,
            post_provenance=post_provenance,
            post_exception_obligations=post_exception_obligations,
        )
        for plan in plans
    ]


def _summarize_rewrite_plans(
    entries: list[JSONObject],
    *,
    max_entries: int = 10,
) -> list[str]:
    return _ds_summarize_rewrite_plans(
        entries,
        max_entries=max_entries,
        check_deadline=check_deadline,
    )

def _enclosing_function_node(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
):
    check_deadline()
    current = parents.get(node)
    while current is not None:
        check_deadline()
        current_type = type(current)
        if current_type is ast.FunctionDef or current_type is ast.AsyncFunctionDef:
            return cast(ast.FunctionDef | ast.AsyncFunctionDef, current)
        current = parents.get(current)
    return None


def _exception_param_names(expr, params: set[str]) -> list[str]:
    return _exc_exception_param_names(expr, params, check_deadline=check_deadline)

def _exception_type_name(expr):
    return _exc_exception_type_name(expr, decorator_name=_decorator_name)


def _annotation_exception_candidates(annotation) -> tuple[str, ...]:
    check_deadline()
    if not annotation:
        return ()
    try:
        expr = ast.parse(annotation, mode="eval").body
    except SyntaxError:
        return ()
    candidates: set[str] = set()
    for node in ast.walk(expr):
        check_deadline()
        node_type = type(node)
        if node_type is ast.Name:
            node_name = cast(ast.Name, node)
            cls = _exc_builtin_exception_class(node_name.id)
            if cls is not None:
                candidates.add(node_name.id)
        elif node_type is ast.Attribute:
            node_attr = cast(ast.Attribute, node)
            cls = _exc_builtin_exception_class(node_attr.attr)
            if cls is not None:
                candidates.add(node_attr.attr)
    return tuple(
        sort_once(
            candidates,
            source="_annotation_exception_candidates.candidates",
            policy=OrderPolicy.SORT,
        )
    )


def _refine_exception_name_from_annotations(
    expr,
    *,
    param_annotations: ParamAnnotationMap,
):
    check_deadline()
    direct_name = _exception_type_name(expr)
    if type(expr) is not ast.Name:
        return direct_name, None, ()
    annotation = param_annotations.get(cast(ast.Name, expr).id)
    candidates = _annotation_exception_candidates(annotation)
    if not candidates:
        return direct_name, None, ()
    if len(candidates) == 1:
        return candidates[0], "PARAM_ANNOTATION", candidates
    return direct_name, "PARAM_ANNOTATION_AMBIGUOUS", candidates


def _handler_type_names(handler_type) -> tuple[str, ...]:
    return _exc_handler_type_names(
        handler_type,
        decorator_name=_decorator_name,
        check_deadline=check_deadline,
    )


def _exception_handler_compatibility(
    exception_name,
    handler_type,
) -> str:
    return _exc_exception_handler_compatibility(
        exception_name,
        handler_type,
        decorator_name=_decorator_name,
        check_deadline=check_deadline,
    )

def _exception_path_id(
    *,
    path: str,
    function: str,
    source_kind: str,
    lineno: int,
    col: int,
    kind: str,
) -> str:
    return f"{path}:{function}:{source_kind}:{lineno}:{col}:{kind}"


def _handler_is_broad(handler: ast.ExceptHandler) -> bool:
    return _exc_handler_is_broad(handler)

def _handler_label(handler: ast.ExceptHandler) -> str:
    return _exc_handler_label(handler)

def _node_in_try_body(node: ast.AST, try_node: ast.Try) -> bool:
    return _exc_node_in_try_body(node, try_node, check_deadline=check_deadline)

def _find_handling_try(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
):
    check_deadline()
    current = parents.get(node)
    try_ancestors: list[ast.Try] = []
    while current is not None:
        check_deadline()
        if type(current) is ast.Try:
            try_ancestors.append(cast(ast.Try, current))
        current = parents.get(current)
    return next(
        (try_node for try_node in try_ancestors if _node_in_try_body(node, try_node)),
        None,
    )


def _node_in_block(node: ast.AST, block: list[ast.stmt]) -> bool:
    check_deadline()
    for stmt in block:
        check_deadline()
        if node is stmt:
            return True
        for child in ast.walk(stmt):
            check_deadline()
            if node is child:
                return True
    return False


def _names_in_expr(expr: ast.AST) -> set[str]:
    check_deadline()
    names: set[str] = set()
    for node in ast.walk(expr):
        check_deadline()
        if type(node) is ast.Name:
            names.add(cast(ast.Name, node).id)
    return names


class _EvalDecision(StrEnum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class _BoolEvalOutcome:
    decision: _EvalDecision

    def is_unknown(self) -> bool:
        return self.decision is _EvalDecision.UNKNOWN

    def as_bool(self) -> bool:
        return self.decision is _EvalDecision.TRUE


@dataclass(frozen=True)
class _ValueEvalOutcome:
    decision: _EvalDecision
    value: JSONValue

    def is_unknown(self) -> bool:
        return self.decision is _EvalDecision.UNKNOWN


def _bool_outcome(value: bool) -> _BoolEvalOutcome:
    return _BoolEvalOutcome(
        _EvalDecision.TRUE if value else _EvalDecision.FALSE
    )


def _unknown_bool_outcome() -> _BoolEvalOutcome:
    return _BoolEvalOutcome(_EvalDecision.UNKNOWN)


def _known_value_outcome(value: JSONValue) -> _ValueEvalOutcome:
    return _ValueEvalOutcome(_EvalDecision.TRUE, value)


def _unknown_value_outcome() -> _ValueEvalOutcome:
    return _ValueEvalOutcome(_EvalDecision.UNKNOWN, False)


def _constant_scalar_outcome(expr: ast.Constant) -> _ValueEvalOutcome:
    value = expr.value
    value_type = type(value)
    if value is None or value_type in {str, int, float, bool}:
        return _known_value_outcome(cast(JSONValue, value))
    return _unknown_value_outcome()


def _is_numeric_value(value: JSONValue) -> bool:
    return issubclass(type(value), (int, float))


def _unary_numeric_outcome(
    expr: ast.UnaryOp,
    env: dict[str, JSONValue],
) -> _ValueEvalOutcome:
    op_type = type(expr.op)
    operand = _eval_value_expr(expr.operand, env)
    if operand.is_unknown():
        return _unknown_value_outcome()
    value = operand.value
    if not _is_numeric_value(value):
        return _unknown_value_outcome()
    if op_type is ast.USub:
        return _known_value_outcome(-value)
    return _known_value_outcome(value)


def _eval_value_expr(expr: ast.AST, env: dict[str, JSONValue]) -> _ValueEvalOutcome:
    check_deadline()
    expr_type = type(expr)
    if expr_type is ast.Constant:
        return _constant_scalar_outcome(cast(ast.Constant, expr))
    if expr_type is ast.Name:
        name_expr = cast(ast.Name, expr)
        if name_expr.id in env:
            return _known_value_outcome(env[name_expr.id])
        return _unknown_value_outcome()
    if expr_type is ast.UnaryOp:
        unary_expr = cast(ast.UnaryOp, expr)
        if type(unary_expr.op) is ast.USub or type(unary_expr.op) is ast.UAdd:
            return _unary_numeric_outcome(unary_expr, env)
        return _unknown_value_outcome()
    return _unknown_value_outcome()


def _eval_bool_not_expr(expr: ast.UnaryOp, env: dict[str, JSONValue]) -> _BoolEvalOutcome:
    inner = _eval_bool_expr(expr.operand, env)
    if inner.is_unknown():
        return _unknown_bool_outcome()
    return _bool_outcome(not inner.as_bool())


def _eval_bool_and_values(
    values: Sequence[ast.expr],
    env: dict[str, JSONValue],
) -> _BoolEvalOutcome:
    any_unknown = False
    for value in values:
        check_deadline()
        result = _eval_bool_expr(value, env)
        if result.decision is _EvalDecision.FALSE:
            return _bool_outcome(False)
        if result.is_unknown():
            any_unknown = True
    return _unknown_bool_outcome() if any_unknown else _bool_outcome(True)


def _eval_bool_or_values(
    values: Sequence[ast.expr],
    env: dict[str, JSONValue],
) -> _BoolEvalOutcome:
    any_unknown = False
    for value in values:
        check_deadline()
        result = _eval_bool_expr(value, env)
        if result.decision is _EvalDecision.TRUE:
            return _bool_outcome(True)
        if result.is_unknown():
            any_unknown = True
    return _unknown_bool_outcome() if any_unknown else _bool_outcome(False)


def _eval_bool_compare_expr(
    expr: ast.Compare,
    env: dict[str, JSONValue],
) -> _BoolEvalOutcome:
    left_outcome = _eval_value_expr(expr.left, env)
    right_outcome = _eval_value_expr(expr.comparators[0], env)
    if left_outcome.is_unknown() or right_outcome.is_unknown():
        return _unknown_bool_outcome()
    left = left_outcome.value
    right = right_outcome.value
    op_type = type(expr.ops[0])
    if op_type is ast.Eq:
        return _bool_outcome(left == right)
    if op_type is ast.NotEq:
        return _bool_outcome(left != right)
    if _is_numeric_value(left) and _is_numeric_value(right):
        if op_type is ast.Lt:
            return _bool_outcome(left < right)
        if op_type is ast.LtE:
            return _bool_outcome(left <= right)
        if op_type is ast.Gt:
            return _bool_outcome(left > right)
        if op_type is ast.GtE:
            return _bool_outcome(left >= right)
    return _unknown_bool_outcome()


def _eval_bool_boolop_expr(expr: ast.BoolOp, env: dict[str, JSONValue]) -> _BoolEvalOutcome:
    op_type = type(expr.op)
    if op_type is ast.And:
        return _eval_bool_and_values(expr.values, env)
    return _eval_bool_or_values(expr.values, env)


def _eval_bool_name_expr(expr: ast.Name, env: dict[str, JSONValue]) -> _BoolEvalOutcome:
    if expr.id not in env:
        return _unknown_bool_outcome()
    return _bool_outcome(bool(env[expr.id]))


def _eval_bool_expr(expr: ast.AST, env: dict[str, JSONValue]) -> _BoolEvalOutcome:
    check_deadline()
    expr_type = type(expr)
    if expr_type is ast.Constant:
        constant_expr = cast(ast.Constant, expr)
        return _bool_outcome(bool(constant_expr.value))
    if expr_type is ast.Name:
        return _eval_bool_name_expr(cast(ast.Name, expr), env)
    if expr_type is ast.UnaryOp:
        unary_expr = cast(ast.UnaryOp, expr)
        if type(unary_expr.op) is ast.Not:
            return _eval_bool_not_expr(unary_expr, env)
        return _unknown_bool_outcome()
    if expr_type is ast.BoolOp:
        boolop_expr = cast(ast.BoolOp, expr)
        if type(boolop_expr.op) is ast.And or type(boolop_expr.op) is ast.Or:
            return _eval_bool_boolop_expr(boolop_expr, env)
        return _unknown_bool_outcome()
    if expr_type is ast.Compare:
        compare_expr = cast(ast.Compare, expr)
        if len(compare_expr.ops) != 1 or len(compare_expr.comparators) != 1:
            return _unknown_bool_outcome()
        return _eval_bool_compare_expr(compare_expr, env)
    return _unknown_bool_outcome()


def _branch_reachability_under_env(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
    env: dict[str, JSONValue],
) -> _EvalDecision:
    """Conservatively evaluate nested-if constraints for `node` under `env`."""
    check_deadline()
    constraints: list[tuple[ast.AST, bool]] = []
    current_node: ast.AST = node
    current = parents.get(current_node)
    while current is not None:
        check_deadline()
        if type(current) is ast.If:
            if_node = cast(ast.If, current)
            if _node_in_block(current_node, if_node.body):
                constraints.append((if_node.test, True))
            elif _node_in_block(current_node, if_node.orelse):
                constraints.append((if_node.test, False))
        current_node = current
        current = parents.get(current_node)
    if not constraints:
        return _EvalDecision.UNKNOWN
    any_unknown = False
    for test, want_true in constraints:
        check_deadline()
        result = _eval_bool_expr(test, env)
        if result.is_unknown():
            any_unknown = True
            continue
        if result.as_bool() != want_true:
            return _EvalDecision.FALSE
    return _EvalDecision.UNKNOWN if any_unknown else _EvalDecision.TRUE


def _is_reachability_false(reachability: _EvalDecision) -> bool:
    return reachability is _EvalDecision.FALSE


def _is_reachability_true(reachability: _EvalDecision) -> bool:
    return reachability is _EvalDecision.TRUE


def _collect_handledness_witnesses(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
) -> list[JSONObject]:
    check_deadline()
    witnesses: list[JSONObject] = []
    raise_or_assert_types = {ast.Raise, ast.Assert}
    for path in paths:
        check_deadline()
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        parent = ParentAnnotator()
        parent.visit(tree)
        parents = parent.parents
        params_by_fn: dict[ast.AST, set[str]] = {}
        param_annotations_by_fn = {}
        for fn in _collect_functions(tree):
            check_deadline()
            params_by_fn[fn] = set(_param_names(fn, ignore_params))
            param_annotations_by_fn[fn] = _param_annotations(fn, ignore_params)
        path_value = _normalize_snapshot_path(path, project_root)
        for node in ast.walk(tree):
            check_deadline()
            if type(node) in raise_or_assert_types:
                raise_node = cast(ast.Raise | ast.Assert, node)
                try_node = _find_handling_try(raise_node, parents)
                source_kind = "E0"
                kind = "raise" if type(raise_node) is ast.Raise else "assert"
                fn_node = _enclosing_function_node(raise_node, parents)
                if fn_node is None:
                    function = "<module>"
                    params = set()
                    param_annotations: dict[str, JSONValue] = {}
                else:
                    scopes = _enclosing_scopes(fn_node, parents)
                    function = _function_key(scopes, fn_node.name)
                    params = params_by_fn.get(fn_node, set())
                    param_annotations = param_annotations_by_fn.get(fn_node, {})
                expr = (
                    cast(ast.Raise, raise_node).exc
                    if type(raise_node) is ast.Raise
                    else cast(ast.Assert, raise_node).test
                )
                (
                    exception_name,
                    exception_type_source,
                    exception_type_candidates,
                ) = _refine_exception_name_from_annotations(
                    expr,
                    param_annotations=param_annotations,
                )
                bundle = _exception_param_names(expr, params)
                lineno = getattr(raise_node, "lineno", 0)
                col = getattr(raise_node, "col_offset", 0)
                exception_id = _exception_path_id(
                    path=path_value,
                    function=function,
                    source_kind=source_kind,
                    lineno=lineno,
                    col=col,
                    kind=kind,
                )
                handledness_id = f"handled:{exception_id}"
                handler_kind = None
                handler_boundary = None
                compatibility = "incompatible"
                handledness_reason_code = "NO_HANDLER"
                handledness_reason = "no enclosing handler discharges this exception path"
                type_refinement_opportunity = ""
                if try_node is not None:
                    unknown_handler = None
                    first_incompatible_handler = None
                    for handler in try_node.handlers:
                        check_deadline()
                        compatibility = _exception_handler_compatibility(
                            exception_name,
                            handler.type,
                        )
                        if compatibility == "compatible":
                            handler_kind = "catch"
                            handler_boundary = _handler_label(handler)
                            if handler.type is None:
                                handledness_reason_code = "BROAD_EXCEPT"
                                handledness_reason = (
                                    "handled by broad except: without a typed match proof"
                                )
                            else:
                                handledness_reason_code = "TYPED_MATCH"
                                handledness_reason = (
                                    "raised exception type matches an explicit except clause"
                                )
                            break
                        if compatibility == "unknown" and unknown_handler is None:
                            unknown_handler = handler
                        if (
                            compatibility == "incompatible"
                            and first_incompatible_handler is None
                        ):
                            first_incompatible_handler = handler
                    if handler_kind is None and unknown_handler is not None:
                        handler_kind = "catch"
                        handler_boundary = _handler_label(unknown_handler)
                        compatibility = "unknown"
                        handledness_reason_code = "TYPE_UNRESOLVED"
                        handledness_reason = (
                            "exception or handler types are dynamic/unresolved; handledness is unknown"
                        )
                        if exception_type_candidates:
                            type_refinement_opportunity = (
                                "narrow raised exception type to a single concrete exception"
                            )
                    elif handler_kind is None and first_incompatible_handler is not None:
                        handler_kind = "catch"
                        handler_boundary = _handler_label(first_incompatible_handler)
                        compatibility = "incompatible"
                        handledness_reason_code = "TYPED_MISMATCH"
                        handledness_reason = (
                            "explicit except clauses do not match the raised exception type"
                        )
                        type_refinement_opportunity = (
                            f"consider except {exception_name} (or a supertype) to dominate this raise path"
                            if exception_name
                            else "consider a typed except clause to dominate this raise path"
                        )
                if handler_kind is None and exception_name == "SystemExit":
                    handler_kind = "convert"
                    handler_boundary = "process exit"
                    compatibility = "compatible"
                    handledness_reason_code = "SYSTEM_EXIT_CONVERT"
                    handledness_reason = "SystemExit is converted to process exit"
                if handler_kind is not None:
                    witness_result = "HANDLED" if compatibility == "compatible" else "UNKNOWN"
                    handler_type_names: tuple[str, ...] = ()
                    if try_node is not None and handler_kind == "catch":
                        handler_types_by_label: dict[str, tuple[str, ...]] = {}
                        for handler in try_node.handlers:
                            check_deadline()
                            handler_types_by_label[_handler_label(handler)] = _handler_type_names(
                                handler.type
                            )
                        handler_type_names = handler_types_by_label.get(
                            str(handler_boundary), ()
                        )
                    witnesses.append(
                        {
                            "handledness_id": handledness_id,
                            "exception_path_id": exception_id,
                            "site": {
                                "path": path_value,
                                "function": function,
                                "bundle": bundle,
                            },
                            "handler_kind": handler_kind,
                            "handler_boundary": handler_boundary,
                            "handler_types": list(handler_type_names),
                            "type_compatibility": compatibility,
                            "exception_type_source": exception_type_source,
                            "exception_type_candidates": list(exception_type_candidates),
                            "type_refinement_opportunity": type_refinement_opportunity,
                            "handledness_reason_code": handledness_reason_code,
                            "handledness_reason": handledness_reason,
                            "environment": {},
                            "core": (
                                [f"enclosed by {handler_boundary}"]
                                if handler_kind == "catch"
                                else ["converted to process exit"]
                            ),
                            "result": witness_result,
                        }
                    )
    return sort_once(
        witnesses,
        key=lambda entry: (
            str(entry.get("site", {}).get("path", "")),
            str(entry.get("site", {}).get("function", "")),
            ",".join(entry.get("site", {}).get("bundle", []) or []),
            str(entry.get("exception_path_id", "")),
        ),
    source = 'src/gabion/analysis/dataflow_audit.py:4462')


def _dead_env_map(
    deadness_witnesses,
) -> dict[tuple[str, str], dict[str, tuple[JSONValue, JSONObject]]]:
    check_deadline()
    dead_env_map: dict[tuple[str, str], dict[str, tuple[JSONValue, JSONObject]]] = {}
    if not deadness_witnesses:
        return dead_env_map
    for entry in deadness_witnesses:
        check_deadline()
        path_value = str(entry.get("path", ""))
        function_value = str(entry.get("function", ""))
        bundle = entry.get("bundle", []) or []
        bundle_values = sequence_or_none(cast(JSONValue, bundle))
        if bundle_values is not None and bundle_values:
            param = str(bundle_values[0])
            environment = mapping_or_none(entry.get("environment", {}))
            if environment is not None:
                value_str = environment.get(param)
                if type(value_str) is str:
                    try:
                        literal_value = ast.literal_eval(value_str)
                    except _LITERAL_EVAL_ERROR_TYPES:
                        literal_value = None
                    if literal_value is not None:
                        dead_env_map.setdefault((path_value, function_value), {})[param] = (
                            literal_value,
                            entry,
                        )
    return dead_env_map


def _collect_exception_obligations(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    handledness_witnesses=None,
    deadness_witnesses=None,
    never_exceptions=None,
) -> list[JSONObject]:
    check_deadline()
    obligations: list[JSONObject] = []
    never_exceptions_set = set(never_exceptions or [])
    handled_map: dict[str, JSONObject] = {}
    raise_or_assert_types = {ast.Raise, ast.Assert}
    if handledness_witnesses:
        for entry in handledness_witnesses:
            check_deadline()
            exception_id = str(entry.get("exception_path_id", ""))
            if exception_id:
                handled_map[exception_id] = entry
    dead_env_map = _dead_env_map(deadness_witnesses)
    for path in paths:
        check_deadline()
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        parent = ParentAnnotator()
        parent.visit(tree)
        parents = parent.parents
        params_by_fn: dict[ast.AST, set[str]] = {}
        for fn in _collect_functions(tree):
            check_deadline()
            params_by_fn[fn] = set(_param_names(fn, ignore_params))
        path_value = _normalize_snapshot_path(path, project_root)
        for node in ast.walk(tree):
            check_deadline()
            if type(node) in raise_or_assert_types:
                raise_node = cast(ast.Raise | ast.Assert, node)
                source_kind = "E0"
                kind = "raise" if type(raise_node) is ast.Raise else "assert"
                fn_node = _enclosing_function_node(raise_node, parents)
                if fn_node is None:
                    function = "<module>"
                    params = set()
                else:
                    scopes = _enclosing_scopes(fn_node, parents)
                    function = _function_key(scopes, fn_node.name)
                    params = params_by_fn.get(fn_node, set())
                expr = (
                    cast(ast.Raise, raise_node).exc
                    if type(raise_node) is ast.Raise
                    else cast(ast.Assert, raise_node).test
                )
                exception_name = _exception_type_name(expr)
                protocol = None
                if (
                    exception_name
                    and never_exceptions_set
                    and _decorator_matches(exception_name, never_exceptions_set)
                ):
                    protocol = "never"
                if not _is_never_marker_raise(function, exception_name, never_exceptions_set):
                    bundle = _exception_param_names(expr, params)
                    lineno = getattr(raise_node, "lineno", 0)
                    col = getattr(raise_node, "col_offset", 0)
                    exception_id = _exception_path_id(
                        path=path_value,
                        function=function,
                        source_kind=source_kind,
                        lineno=lineno,
                        col=col,
                        kind=kind,
                    )
                    handled = handled_map.get(exception_id)
                    status = "UNKNOWN"
                    witness_ref = None
                    remainder = {"exception_kind": kind}
                    environment_ref: JSONValue = None
                    handledness_reason_code = "NO_HANDLER"
                    handledness_reason = "no handledness witness"
                    exception_type_source: JSONValue = None
                    exception_type_candidates: list[str] = []
                    type_refinement_opportunity = ""
                    if handled:
                        witness_result = str(handled.get("result", ""))
                        handledness_reason_code = str(
                            handled.get("handledness_reason_code", "UNKNOWN_REASON")
                        )
                        handledness_reason = str(handled.get("handledness_reason", ""))
                        exception_type_source = handled.get("exception_type_source")
                        raw_candidates = sequence_or_none(handled.get("exception_type_candidates") or [])
                        if raw_candidates is not None:
                            exception_type_candidates = [str(v) for v in raw_candidates]
                        type_refinement_opportunity = str(
                            handled.get("type_refinement_opportunity", "")
                        )
                        if witness_result == "HANDLED":
                            status = "HANDLED"
                            remainder = {}
                        else:
                            remainder["handledness_result"] = witness_result or "UNKNOWN"
                            remainder["type_compatibility"] = str(
                                handled.get("type_compatibility", "unknown")
                            )
                            remainder["handledness_reason_code"] = handledness_reason_code
                            remainder["handledness_reason"] = handledness_reason
                            if exception_type_source:
                                remainder["exception_type_source"] = exception_type_source
                            if exception_type_candidates:
                                remainder["exception_type_candidates"] = exception_type_candidates
                            if type_refinement_opportunity:
                                remainder["type_refinement_opportunity"] = type_refinement_opportunity
                        witness_ref = handled.get("handledness_id")
                        environment_ref = handled.get("environment") or {}
                    if status != "HANDLED":
                        env_entries = dead_env_map.get((path_value, function), {})
                        if env_entries:
                            env = {name: value for name, (value, _) in env_entries.items()}
                            reachability = _branch_reachability_under_env(raise_node, parents, env)
                            if _is_reachability_false(reachability):
                                names: set[str] = set()
                                current = parents.get(raise_node)
                                while current is not None:
                                    check_deadline()
                                    if type(current) is ast.If:
                                        names.update(_names_in_expr(cast(ast.If, current).test))
                                    current = parents.get(current)
                                ordered_names = sort_once(
                                    names,
                                    source="_collect_exception_obligations.names.dead",
                                    policy=OrderPolicy.SORT,
                                )
                                for name in sort_once(
                                    ordered_names,
                                    source="_collect_exception_obligations.names.dead.enforce",
                                    policy=OrderPolicy.ENFORCE,
                                ):
                                    check_deadline()
                                    if name not in env_entries:
                                        continue
                                    _, witness = env_entries[name]
                                    status = "DEAD"
                                    witness_ref = witness.get("deadness_id")
                                    remainder = {}
                                    environment_ref = witness.get("environment") or {}
                                    break
                    if protocol == "never" and status != "DEAD":
                        status = "FORBIDDEN"
                    obligations.append(
                        {
                            "exception_path_id": exception_id,
                            "site": {
                                "path": path_value,
                                "function": function,
                                "bundle": bundle,
                            },
                            "source_kind": source_kind,
                            "status": status,
                            "handledness_reason_code": handledness_reason_code,
                            "handledness_reason": handledness_reason,
                            "exception_type_source": exception_type_source,
                            "exception_type_candidates": exception_type_candidates,
                            "type_refinement_opportunity": type_refinement_opportunity,
                            "witness_ref": witness_ref,
                            "remainder": remainder,
                            "environment_ref": environment_ref,
                            "exception_name": exception_name,
                            "protocol": protocol,
                        }
                    )
    return sort_once(
        obligations,
        key=lambda entry: (
            str(entry.get("site", {}).get("path", "")),
            str(entry.get("site", {}).get("function", "")),
            ",".join(entry.get("site", {}).get("bundle", []) or []),
            str(entry.get("source_kind", "")),
            str(entry.get("exception_path_id", "")),
        ),
    source = 'src/gabion/analysis/dataflow_audit.py:4672')


def _never_reason(call: ast.Call):
    check_deadline()
    if call.args:
        first = call.args[0]
        if type(first) is ast.Constant:
            first_value = cast(ast.Constant, first).value
            if type(first_value) is str:
                return first_value
    for kw in call.keywords:
        check_deadline()
        if kw.arg == "reason":
            kw_value = kw.value
            if type(kw_value) is ast.Constant:
                constant_value = cast(ast.Constant, kw_value).value
                if type(constant_value) is str:
                    return constant_value
    return None


def _collect_never_invariants(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    forest: Forest,
    deadness_witnesses=None,
) -> list[JSONObject]:
    check_deadline()
    invariants: list[JSONObject] = []
    dead_env_map = _dead_env_map(deadness_witnesses)
    for path in paths:
        check_deadline()
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        parent = ParentAnnotator()
        parent.visit(tree)
        parents = parent.parents
        params_by_fn: dict[ast.AST, set[str]] = {}
        for fn in _collect_functions(tree):
            check_deadline()
            params_by_fn[fn] = set(_param_names(fn, ignore_params))
        path_value = _normalize_snapshot_path(path, project_root)
        for node in ast.walk(tree):
            check_deadline()
            if type(node) is ast.Call and _is_never_call(cast(ast.Call, node)):
                call_node = cast(ast.Call, node)
                fn_node = _enclosing_function_node(call_node, parents)
                if fn_node is None:
                    function = "<module>"
                    params = set()
                else:
                    scopes = _enclosing_scopes(fn_node, parents)
                    function = _function_key(scopes, fn_node.name)
                    params = params_by_fn.get(fn_node, set())
                bundle = _exception_param_names(call_node, params)
                span = _node_span(call_node)
                lineno = getattr(call_node, "lineno", 0)
                col = getattr(call_node, "col_offset", 0)
                never_id = f"never:{path_value}:{function}:{lineno}:{col}"
                reason = _never_reason(call_node) or ""
                status = "OBLIGATION"
                witness_ref = None
                environment_ref: JSONValue = None
                undecidable_reason = None
                env_entries = dead_env_map.get((path_value, function), {})
                if env_entries:
                    env = {name: value for name, (value, _) in env_entries.items()}
                    reachability = _branch_reachability_under_env(call_node, parents, env)
                    if _is_reachability_false(reachability):
                        names: set[str] = set()
                        current = parents.get(call_node)
                        while current is not None:
                            check_deadline()
                            if type(current) is ast.If:
                                names.update(_names_in_expr(cast(ast.If, current).test))
                            current = parents.get(current)
                        ordered_names = sort_once(
                            names,
                            source="_collect_never_invariants.names.proven_unreachable",
                            policy=OrderPolicy.SORT,
                        )
                        for name in sort_once(
                            ordered_names,
                            source="_collect_never_invariants.names.proven_unreachable.enforce",
                            policy=OrderPolicy.ENFORCE,
                        ):
                            check_deadline()
                            if name not in env_entries:
                                continue
                            _, witness = env_entries[name]
                            status = "PROVEN_UNREACHABLE"
                            witness_ref = witness.get("deadness_id")
                            environment_ref = witness.get("environment") or {}
                            break
                        if status == "PROVEN_UNREACHABLE" and not environment_ref:
                            environment_ref = env
                    elif _is_reachability_true(reachability):
                        status = "VIOLATION"
                        environment_ref = env
                    else:
                        names: set[str] = set()
                        current = parents.get(call_node)
                        while current is not None:
                            check_deadline()
                            if type(current) is ast.If:
                                names.update(_names_in_expr(cast(ast.If, current).test))
                            current = parents.get(current)
                        undecidable_params = sort_once(
                            (n for n in names if n not in env_entries),
                            source="_collect_never_invariants.undecidable_params",
                            policy=OrderPolicy.SORT,
                        )
                        if undecidable_params:
                            undecidable_reason = f"depends on params: {', '.join(undecidable_params)}"
                entry: JSONObject = {
                    "never_id": never_id,
                    "site": {
                        "path": path_value,
                        "function": function,
                        "bundle": bundle,
                    },
                    "status": status,
                    "reason": reason,
                }
                normalized_span = span or (lineno, col, lineno, col)
                if undecidable_reason:
                    entry["undecidable_reason"] = undecidable_reason
                if witness_ref is not None:
                    entry["witness_ref"] = witness_ref
                if environment_ref is not None:
                    entry["environment_ref"] = environment_ref
                entry["span"] = list(normalized_span)
                invariants.append(entry)
                site_id = forest.add_site(path.name, function)
                paramset_id = forest.add_paramset(bundle)
                evidence: dict[str, object] = {"path": path.name, "qual": function}
                if reason:
                    evidence["reason"] = reason
                evidence["span"] = list(normalized_span)
                forest.add_alt("NeverInvariantSink", (site_id, paramset_id), evidence=evidence)
    return sort_once(
        invariants,
        key=lambda entry: (
            str(entry.get("site", {}).get("path", "")),
            str(entry.get("site", {}).get("function", "")),
            ",".join(entry.get("site", {}).get("bundle", []) or []),
            str(entry.get("never_id", "")),
        ),
    source = 'src/gabion/analysis/dataflow_audit.py:4823')


_DEADLINE_CHECK_METHODS = {"check", "expired"}
_DEADLINE_HELPER_QUALS = {
    "gabion.analysis.timeout_context.check_deadline",
    "gabion.analysis.timeout_context.deadline_loop_iter",
    "gabion.analysis.timeout_context.set_deadline",
    "gabion.analysis.timeout_context.reset_deadline",
    "gabion.analysis.timeout_context.get_deadline",
    "gabion.analysis.timeout_context.deadline_scope",
}
_DEADLINE_EXEMPT_PREFIXES = ("gabion.analysis.timeout_context.",)


def _is_deadline_annot(annot) -> bool:
    if not annot:
        return False
    return bool(re.search(r"\bDeadline\b", annot))


def _is_deadline_param(name: str, annot) -> bool:
    if _is_deadline_annot(annot):
        return True
    if annot is None and name.lower() == "deadline":
        return True
    return False


def _is_deadline_origin_call(expr: ast.AST) -> bool:
    if type(expr) is not ast.Call:
        return False
    call_expr = cast(ast.Call, expr)
    try:
        name = ast.unparse(call_expr.func)
    except _AST_UNPARSE_ERROR_TYPES:
        return False
    if name == "Deadline" or name.endswith(".Deadline"):
        return True
    if name in {
        "Deadline.from_timeout",
        "Deadline.from_timeout_ms",
        "Deadline.from_timeout_ticks",
    }:
        return True
    if name.endswith(".Deadline.from_timeout"):
        return True
    if name.endswith(".Deadline.from_timeout_ms"):
        return True
    if name.endswith(".Deadline.from_timeout_ticks"):
        return True
    return False


def _target_names(target: ast.AST) -> set[str]:
    check_deadline()
    names: set[str] = set()
    for node in ast.walk(target):
        check_deadline()
        if type(node) is ast.Name:
            name_node = cast(ast.Name, node)
            if type(name_node.ctx) is ast.Store:
                names.add(name_node.id)
    return names


def _simple_store_name(target: ast.AST) -> OptionalString:
    if type(target) is ast.Name:
        return cast(ast.Name, target).id
    return None


class _DeadlineFunctionCollector(ast.NodeVisitor):
    def __init__(self, root: ast.AST, params: set[str]) -> None:
        self._root = root
        self._params = params
        self.loop = False
        self.check_params: set[str] = set()
        self.ambient_check = False
        self.loop_sites: list[_DeadlineLoopFacts] = []
        self._loop_stack: list[_DeadlineLoopFacts] = []
        self.assignments: list[tuple[list[ast.AST], OptionalAstNode, OptionalSpan4]] = []

    def _mark_param_check(self, name: str) -> None:
        if self._loop_stack:
            self._loop_stack[-1].check_params.add(name)
        else:
            self.check_params.add(name)

    def _mark_ambient_check(self) -> None:
        if self._loop_stack:
            self._loop_stack[-1].ambient_check = True
        else:
            self.ambient_check = True

    def _record_call_span(self, node: ast.AST) -> None:
        if self._loop_stack:
            span = _node_span(node)
            if span is not None:
                self._loop_stack[-1].call_spans.add(span)

    def _iter_marks_ambient(self, expr: ast.AST) -> bool:
        if type(expr) is ast.Call:
            func = cast(ast.Call, expr).func
            func_type = type(func)
            if func_type is ast.Name:
                return cast(ast.Name, func).id == "deadline_loop_iter"
            if func_type is ast.Attribute:
                return cast(ast.Attribute, func).attr == "deadline_loop_iter"
        return False

    def _visit_loop_body(
        self,
        node: ast.AST,
        kind: str,
        *,
        ambient_check: bool = False,
    ) -> None:
        self.loop = True
        loop_fact = _DeadlineLoopFacts(
            span=_node_span(node),
            kind=kind,
            depth=len(self._loop_stack) + 1,
            ambient_check=ambient_check,
        )
        self._loop_stack.append(loop_fact)
        for stmt in getattr(node, "body", []):
            check_deadline()
            self.visit(stmt)
        self._loop_stack.pop()
        self.loop_sites.append(loop_fact)
        for stmt in getattr(node, "orelse", []):
            check_deadline()
            self.visit(stmt)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node is not self._root:
            return
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node is not self._root:
            return
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_For(self, node: ast.For) -> None:
        self.loop = True
        ambient_check = self._iter_marks_ambient(node.iter)
        self.visit(node.target)
        self.visit(node.iter)
        self._visit_loop_body(node, "for", ambient_check=ambient_check)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.loop = True
        ambient_check = self._iter_marks_ambient(node.iter)
        self.visit(node.target)
        self.visit(node.iter)
        self._visit_loop_body(node, "async_for", ambient_check=ambient_check)

    def visit_While(self, node: ast.While) -> None:
        self.loop = True
        self.visit(node.test)
        self._visit_loop_body(node, "while")

    def visit_Call(self, node: ast.Call) -> None:
        self._record_call_span(node)
        func = node.func
        func_type = type(func)
        if func_type is ast.Attribute:
            attribute_func = cast(ast.Attribute, func)
            if attribute_func.attr == "deadline_loop_iter":
                self._mark_ambient_check()
            if (
                attribute_func.attr in _DEADLINE_CHECK_METHODS
                and type(attribute_func.value) is ast.Name
                and cast(ast.Name, attribute_func.value).id in self._params
            ):
                self._mark_param_check(cast(ast.Name, attribute_func.value).id)
            if attribute_func.attr == "check_deadline" and node.args:
                first = node.args[0]
                if type(first) is ast.Name and cast(ast.Name, first).id in self._params:
                    self._mark_param_check(cast(ast.Name, first).id)
            if attribute_func.attr in {"check_deadline", "require_deadline"} and not node.args:
                self._mark_ambient_check()
        elif func_type is ast.Name:
            name_func = cast(ast.Name, func)
            if name_func.id == "deadline_loop_iter":
                self._mark_ambient_check()
            if name_func.id == "check_deadline" and node.args:
                first = node.args[0]
                if type(first) is ast.Name and cast(ast.Name, first).id in self._params:
                    self._mark_param_check(cast(ast.Name, first).id)
            if name_func.id in {"check_deadline", "require_deadline"} and not node.args:
                self._mark_ambient_check()
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.assignments.append((node.targets, node.value, _node_span(node)))
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.assignments.append(([node.target], node.value, _node_span(node)))
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.assignments.append(([node.target], node.value, _node_span(node)))
        self.generic_visit(node)


@dataclass
class _DeadlineLoopFacts:
    span: OptionalSpan4
    kind: str
    depth: int = 1
    check_params: set[str] = field(default_factory=set)
    ambient_check: bool = False
    call_spans: set[tuple[int, int, int, int]] = field(default_factory=set)


@dataclass(frozen=True)
class _DeadlineLocalInfo:
    origin_vars: set[str]
    origin_spans: dict[str, tuple[int, int, int, int]]
    alias_to_param: dict[str, str]


@dataclass(frozen=True)
class _DeadlineFunctionFacts:
    path: Path
    qual: str
    span: OptionalSpan4
    loop: bool
    check_params: set[str]
    ambient_check: bool
    loop_sites: list[_DeadlineLoopFacts]
    local_info: _DeadlineLocalInfo


def _collect_deadline_local_info(
    assignments: list[tuple[list[ast.AST], OptionalAstNode, OptionalSpan4]],
    params: set[str],
) -> _DeadlineLocalInfo:
    check_deadline()
    origin_assign: set[str] = set()
    origin_spans: dict[str, tuple[int, int, int, int]] = {}
    for targets, value, span in assignments:
        check_deadline()
        if value is not None and _is_deadline_origin_call(value):
            for target in targets:
                check_deadline()
                for name in _target_names(target):
                    check_deadline()
                    origin_assign.add(name)
                    if span is not None and name not in origin_spans:
                        origin_spans[name] = span

    alias_assign: dict[str, set[str]] = defaultdict(set)
    origin_alias: set[str] = set()
    unknown_assign: set[str] = set()
    for targets, value, _ in assignments:
        check_deadline()
        if value is None:
            for target in targets:
                check_deadline()
                unknown_assign.update(_target_names(target))
        elif not _is_deadline_origin_call(value):
            alias_source = None
            propagate_origin_alias = False
            if type(value) is ast.Name:
                value_name = cast(ast.Name, value)
                if value_name.id in params:
                    alias_source = value_name.id
                elif value_name.id in origin_assign:
                    propagate_origin_alias = True
            for target in targets:
                check_deadline()
                for name in _target_names(target):
                    check_deadline()
                    if propagate_origin_alias:
                        origin_alias.add(name)
                    elif alias_source is not None:
                        alias_assign[name].add(alias_source)
                    else:
                        unknown_assign.add(name)

    origin_candidates = origin_assign | origin_alias
    origin_vars = {
        name
        for name in origin_candidates
        if name not in unknown_assign and name not in alias_assign
    }
    alias_to_param: dict[str, str] = {}
    for name, sources in alias_assign.items():
        check_deadline()
        if name in unknown_assign or name in origin_candidates:
            continue
        if len(sources) == 1:
            alias_to_param[name] = next(iter(sources))
    for param in params:
        check_deadline()
        alias_to_param[param] = param
    return _DeadlineLocalInfo(
        origin_vars=origin_vars,
        origin_spans=origin_spans,
        alias_to_param=alias_to_param,
    )


def _collect_deadline_function_facts(
    paths: list[Path],
    *,
    project_root = None,
    ignore_params: set[str],
    parse_failure_witnesses: list[JSONObject],
    trees = None,
    analysis_index = None,
    stage_cache_fn = None,
) -> dict[str, _DeadlineFunctionFacts]:
    check_deadline()
    ignore_param_names = set(ignore_params or ())
    if stage_cache_fn is None:
        stage_cache_fn = _analysis_index_stage_cache
    if analysis_index is not None and trees is None:
        facts_by_path = stage_cache_fn(
            analysis_index,
            paths,
            spec=_StageCacheSpec(
                stage=_ParseModuleStage.DEADLINE_FUNCTION_FACTS,
                cache_key=_parse_stage_cache_key(
                    stage=_ParseModuleStage.DEADLINE_FUNCTION_FACTS,
                    cache_context=_EMPTY_CACHE_SEMANTIC_CONTEXT,
                    config_subset={
                        "project_root": str(project_root) if project_root is not None else "",
                        "ignore_params": list(_sorted_text(ignore_param_names)),
                    },
                    detail="deadline_function_facts",
                ),
                build=lambda tree, path: _deadline_function_facts_for_tree(
                    path,
                    tree,
                    project_root=project_root,
                    ignore_params=ignore_param_names,
                ),
            ),
            parse_failure_witnesses=parse_failure_witnesses,
        )
        facts: dict[str, _DeadlineFunctionFacts] = {}
        for entry in facts_by_path.values():
            check_deadline()
            if entry is not None:
                facts.update(entry)
        return facts
    facts: dict[str, _DeadlineFunctionFacts] = {}
    for path in paths:
        check_deadline()
        if trees is not None and path in trees:
            tree = trees[path]
        else:
            tree = _parse_module_tree(
                path,
                stage=_ParseModuleStage.DEADLINE_FUNCTION_FACTS,
                parse_failure_witnesses=parse_failure_witnesses,
            )
        if tree is not None:
            facts.update(
                _deadline_function_facts_for_tree(
                    path,
                    tree,
                    project_root=project_root,
                    ignore_params=ignore_param_names,
                )
            )
    return facts


def _deadline_function_facts_for_tree(
    path: Path,
    tree: ast.AST,
    *,
    project_root,
    ignore_params: set[str],
) -> dict[str, _DeadlineFunctionFacts]:
    check_deadline()
    parents = ParentAnnotator()
    parents.visit(tree)
    module = _module_name(path, project_root)
    facts: dict[str, _DeadlineFunctionFacts] = {}
    for fn in _collect_functions(tree):
        check_deadline()
        scopes = _enclosing_scopes(fn, parents.parents)
        qual_parts = [module] if module else []
        if scopes:
            qual_parts.extend(scopes)
        qual_parts.append(fn.name)
        qual = ".".join(qual_parts)
        params = set(_param_names(fn, ignore_params))
        collector = _DeadlineFunctionCollector(fn, params)
        collector.visit(fn)
        local_info = _collect_deadline_local_info(collector.assignments, params)
        facts[qual] = _DeadlineFunctionFacts(
            path=path,
            qual=qual,
            span=_node_span(fn),
            loop=collector.loop,
            check_params=set(collector.check_params),
            ambient_check=collector.ambient_check,
            loop_sites=list(collector.loop_sites),
            local_info=local_info,
        )
    return facts


def _collect_call_nodes_by_path(
    paths: list[Path],
    *,
    trees = None,
    parse_failure_witnesses: list[JSONObject],
    analysis_index = None,
) -> dict[Path, dict[tuple[int, int, int, int], list[ast.Call]]]:
    check_deadline()
    if analysis_index is not None and trees is None:
        cached_by_path = _analysis_index_stage_cache(
            analysis_index,
            paths,
            spec=_StageCacheSpec(
                stage=_ParseModuleStage.CALL_NODES,
                cache_key=_parse_stage_cache_key(
                    stage=_ParseModuleStage.CALL_NODES,
                    cache_context=_EMPTY_CACHE_SEMANTIC_CONTEXT,
                    config_subset={},
                    detail="call_nodes",
                ),
                build=lambda tree, _path: _call_nodes_for_tree(tree),
            ),
            parse_failure_witnesses=parse_failure_witnesses,
        )
        return {
            path: nodes
            for path, nodes in cached_by_path.items()
            if nodes is not None
        }
    call_nodes: dict[Path, dict[tuple[int, int, int, int], list[ast.Call]]] = {}
    for path in paths:
        check_deadline()
        if trees is not None and path in trees:
            tree = trees[path]
        else:
            tree = _parse_module_tree(
                path,
                stage=_ParseModuleStage.CALL_NODES,
                parse_failure_witnesses=parse_failure_witnesses,
            )
        if tree is not None:
            call_nodes[path] = _call_nodes_for_tree(tree)
    return call_nodes


def _call_nodes_for_tree(
    tree: ast.AST,
) -> dict[tuple[int, int, int, int], list[ast.Call]]:
    check_deadline()
    span_map: dict[tuple[int, int, int, int], list[ast.Call]] = defaultdict(list)
    for node in ast.walk(tree):
        check_deadline()
        if type(node) is ast.Call:
            call_node = cast(ast.Call, node)
            span = _node_span(call_node)
            if span is not None:
                span_map[span].append(call_node)
    return span_map


def _collect_call_edges(
    *,
    by_name: dict[str, list[FunctionInfo]],
    by_qual: dict[str, FunctionInfo],
    symbol_table: SymbolTable,
    project_root,
    class_index: dict[str, ClassInfo],
    resolve_callee_outcome_fn = None,
) -> dict[str, set[str]]:
    check_deadline()
    if resolve_callee_outcome_fn is None:
        resolve_callee_outcome_fn = _resolve_callee_outcome
    edges: dict[str, set[str]] = defaultdict(set)
    for infos in by_name.values():
        check_deadline()
        for info in infos:
            check_deadline()
            if _is_test_path(info.path):
                continue
            for call in info.calls:
                check_deadline()
                if call.is_test:
                    continue
                resolution = resolve_callee_outcome_fn(
                    call.callee,
                    info,
                    by_name,
                    by_qual,
                    symbol_table=symbol_table,
                    project_root=project_root,
                    class_index=class_index,
                    call=call,
                )
                if not resolution.candidates:
                    continue
                for candidate in resolution.candidates:
                    check_deadline()
                    edges[info.qual].add(candidate.qual)
    return edges


def _function_suite_key(path: str, qual: str) -> _FunctionSuiteKey:
    return _FunctionSuiteKey(path=path, qual=qual)


def _function_suite_id(key: _FunctionSuiteKey) -> NodeId:
    return NodeId("SuiteSite", (key.path, key.qual, "function"))


class _FunctionSuiteLookupStatus(StrEnum):
    RESOLVED = "resolved"
    NODE_MISSING = "node_missing"
    SUITE_KIND_UNSUPPORTED = "suite_kind_unsupported"


@dataclass(frozen=True)
class _FunctionSuiteLookupOutcome:
    status: _FunctionSuiteLookupStatus
    suite_id: NodeId


def _node_to_function_suite_lookup_outcome(
    forest: Forest,
    node_id: NodeId,
) -> _FunctionSuiteLookupOutcome:
    missing_suite_id = NodeId("MissingSuiteSite", ("", "", ""))
    node = forest.nodes.get(node_id)
    if node is None:
        return _FunctionSuiteLookupOutcome(
            _FunctionSuiteLookupStatus.NODE_MISSING,
            missing_suite_id,
        )
    if node.kind == "FunctionSite":
        path = str(node.meta.get("path", "") or "")
        qual = str(node.meta.get("qual", "") or "")
        if not path or not qual:
            never("function site missing identity", path=path, qual=qual)
        return _FunctionSuiteLookupOutcome(
            _FunctionSuiteLookupStatus.RESOLVED,
            _function_suite_id(_function_suite_key(path, qual)),
        )
    if node.kind == "SuiteSite":
        suite_kind = str(node.meta.get("suite_kind", "") or "")
        if suite_kind in {"function", "function_body"}:
            path = str(node.meta.get("path", "") or "")
            qual = str(node.meta.get("qual", "") or "")
            if not path or not qual:
                never("function suite missing identity", path=path, qual=qual)
            return _FunctionSuiteLookupOutcome(
                _FunctionSuiteLookupStatus.RESOLVED,
                _function_suite_id(_function_suite_key(path, qual)),
            )
    return _FunctionSuiteLookupOutcome(
        _FunctionSuiteLookupStatus.SUITE_KIND_UNSUPPORTED,
        missing_suite_id,
    )


def _suite_caller_function_id(
    suite_node: Node,
) -> NodeId:
    path = str(suite_node.meta.get("path", "") or "")
    qual = str(suite_node.meta.get("qual", "") or "")
    if not path or not qual:
        never(
            "suite site missing caller identity",
            suite_kind=str(suite_node.meta.get("suite_kind", "") or ""),
            path=path,
            qual=qual,
        )
    return _function_suite_id(_function_suite_key(path, qual))


def _node_to_function_suite_id(
    forest: Forest,
    node_id: NodeId,
):
    outcome = _node_to_function_suite_lookup_outcome(forest, node_id)
    if outcome.status is _FunctionSuiteLookupStatus.RESOLVED:
        return outcome.suite_id
    unresolved: dict[str, NodeId] = {}
    return unresolved.get("suite_id")


def _obligation_candidate_suite_ids(
    *,
    by_name: dict[str, list[FunctionInfo]],
    callee_key: str,
) -> set[NodeId]:
    key = _callee_key(callee_key)
    candidates: set[NodeId] = set()
    for info in by_name.get(key, []):
        check_deadline()
        if _is_test_path(info.path):
            continue
        candidates.add(_function_suite_id(_function_suite_key(info.path.name, info.qual)))
    return candidates


def _collect_call_edges_from_forest(
    forest: Forest,
    *,
    by_name: dict[str, list[FunctionInfo]],
) -> dict[NodeId, set[NodeId]]:
    check_deadline()
    edges: dict[NodeId, set[NodeId]] = defaultdict(set)
    for alt in forest.alts:
        check_deadline()
        if not alt.inputs:
            continue
        suite_id = alt.inputs[0]
        suite_node = forest.nodes.get(suite_id)
        if suite_node is not None and suite_node.kind == "SuiteSite":
            suite_kind = str(suite_node.meta.get("suite_kind", "") or "")
            if suite_kind == "call":
                caller_id = _suite_caller_function_id(suite_node)
                if alt.kind == "CallCandidate":
                    if len(alt.inputs) >= 2:
                        candidate_id = _node_to_function_suite_id(forest, alt.inputs[1])
                        if candidate_id is not None:
                            edges[caller_id].add(candidate_id)
                elif alt.kind == "CallResolutionObligation":
                    callee_key = str(alt.evidence.get("callee", "") or "")
                    if callee_key:
                        for candidate_id in _obligation_candidate_suite_ids(
                            by_name=by_name,
                            callee_key=callee_key,
                        ):
                            check_deadline()
                            edges[caller_id].add(candidate_id)
    return edges


def _collect_call_resolution_obligations_from_forest(
    forest: Forest,
) -> list[tuple[NodeId, NodeId, tuple[int, int, int, int], str]]:
    check_deadline()
    obligations: list[tuple[NodeId, NodeId, tuple[int, int, int, int], str]] = []
    seen: set[tuple[NodeId, NodeId, tuple[int, int, int, int], str]] = set()
    for alt in forest.alts:
        check_deadline()
        if alt.kind != "CallResolutionObligation":
            continue
        if not alt.inputs:
            continue
        suite_id = alt.inputs[0]
        suite_node = forest.nodes.get(suite_id)
        suite_kind = suite_node.kind if suite_node is not None else ""
        if suite_kind != "SuiteSite":
            continue
        node_suite_kind = str(suite_node.meta.get("suite_kind", "") or "")
        if node_suite_kind != "call":
            continue
        caller_id = _suite_caller_function_id(suite_node)
        raw_span = suite_node.meta.get("span")
        span = (0, 0, 0, 0)
        span_valid = False
        if type(raw_span) is list and len(raw_span) == 4:
            coerced: list[int] = []
            valid = True
            for value in raw_span:
                check_deadline()
                if type(value) is not int:
                    valid = False
                    break
                coerced.append(cast(int, value))
            if valid:
                span = (coerced[0], coerced[1], coerced[2], coerced[3])
                span_valid = True
        if not span_valid:
            caller_path = str(suite_node.meta.get("path", "") or "")
            caller_qual = str(suite_node.meta.get("qual", "") or "")
            never(
                "call resolution obligation requires span",
                path=caller_path,
                qual=caller_qual,
            )
        callee_key = str(alt.evidence.get("callee", "") or "")
        if not callee_key:
            continue
        record = (caller_id, suite_id, span, callee_key)
        if record in seen:
            continue
        seen.add(record)
        obligations.append(record)
    return obligations


def _collect_call_resolution_obligation_details_from_forest(
    forest: Forest,
) -> list[tuple[NodeId, NodeId, tuple[int, int, int, int], str, str]]:
    evidence_by_key: dict[tuple[NodeId, str], JSONObject] = {}
    for alt in forest.alts:
        check_deadline()
        if alt.kind != "CallResolutionObligation" or not alt.inputs:
            continue
        suite_id = alt.inputs[0]
        callee_value = alt.evidence.get("callee")
        callee_key = str(callee_value or "")
        if not callee_key:
            continue
        key = (suite_id, callee_key)
        if key not in evidence_by_key:
            evidence_by_key[key] = alt.evidence

    records: list[tuple[NodeId, NodeId, tuple[int, int, int, int], str, str]] = []
    for caller_id, suite_id, span, callee_key in _collect_call_resolution_obligations_from_forest(
        forest
    ):
        check_deadline()
        evidence = evidence_by_key.get((suite_id, callee_key), {})
        obligation_kind = str(evidence.get("kind", "") or "")
        if not obligation_kind:
            obligation_kind = "unresolved_internal_callee"
        records.append((caller_id, suite_id, span, callee_key, obligation_kind))
    return records


def _call_resolution_obligation_evidence(
    forest: Forest,
    *,
    suite_id: NodeId,
    callee_key: str,
) -> JSONObject:
    check_deadline()
    for alt in forest.alts:
        check_deadline()
        if alt.kind != "CallResolutionObligation" or not alt.inputs:
            continue
        if alt.inputs[0] != suite_id:
            continue
        if str(alt.evidence.get("callee", "") or "") != callee_key:
            continue
        return alt.evidence
    return {}


def _collect_unresolved_call_sites_from_forest(
    forest: Forest,
    collect_call_resolution_obligations_from_forest_fn: Callable[
        [Forest], list[tuple[NodeId, NodeId, tuple[int, int, int, int], str]]
    ] = _collect_call_resolution_obligations_from_forest,
) -> list[tuple[str, str, tuple[int, int, int, int], str]]:
    """Backward-compatible alias for call resolution obligations."""
    out: list[tuple[str, str, tuple[int, int, int, int], str]] = []
    for caller_id, _, span, callee_key in collect_call_resolution_obligations_from_forest_fn(
        forest
    ):
        check_deadline()
        if caller_id.kind != "SuiteSite" or len(caller_id.key) < 2:
            continue
        caller_path = str(caller_id.key[0] or "")
        caller_qual = str(caller_id.key[1] or "")
        if not caller_path or not caller_qual:
            continue
        out.append((caller_path, caller_qual, span, callee_key))
    return out


def _call_candidate_target_site(
    *,
    forest: Forest,
    candidate: FunctionInfo,
) -> NodeId:
    check_deadline()
    if candidate.function_span is None:
        never(
            "call candidate target requires function span",
            path=candidate.path.name,
            qual=candidate.qual,
        )
    return forest.add_suite_site(
        candidate.path.name,
        candidate.qual,
        "function",
        span=candidate.function_span,
    )


def _materialize_call_candidates(
    *,
    forest: Forest,
    by_name: dict[str, list[FunctionInfo]],
    by_qual: dict[str, FunctionInfo],
    symbol_table: SymbolTable,
    project_root,
    class_index: dict[str, ClassInfo],
    resolve_callee_outcome_fn = None,
) -> None:
    check_deadline()
    if resolve_callee_outcome_fn is None:
        resolve_callee_outcome_fn = _resolve_callee_outcome
    seen: set[tuple[NodeId, NodeId]] = set()
    obligation_seen: set[NodeId] = set()
    seen_loop_checked = False
    for alt in forest.alts:
        if not seen_loop_checked:
            check_deadline()
            seen_loop_checked = True
        if alt.kind != "CallCandidate" or len(alt.inputs) < 2:
            if alt.kind == "CallResolutionObligation" and alt.inputs:
                obligation_seen.add(alt.inputs[0])
            continue
        seen.add((alt.inputs[0], alt.inputs[1]))
    for infos in by_name.values():
        check_deadline()
        for info in infos:
            check_deadline()
            if _is_test_path(info.path):
                continue
            for call in info.calls:
                check_deadline()
                if call.is_test:
                    continue
                resolution = resolve_callee_outcome_fn(
                    call.callee,
                    info,
                    by_name,
                    by_qual,
                    symbol_table=symbol_table,
                    project_root=project_root,
                    class_index=class_index,
                    call=call,
                )
                if call.span is None:
                    if resolution.status != "unresolved_external":
                        never(
                            "call candidate requires span",
                            path=_normalize_snapshot_path(info.path, project_root),
                            qual=info.qual,
                            callee=call.callee,
                        )
                    continue
                suite_id = forest.add_suite_site(
                    info.path.name,
                    info.qual,
                    "call",
                    span=call.span,
                )
                if resolution.status in {"resolved", "ambiguous"}:
                    for candidate in resolution.candidates:
                        check_deadline()
                        candidate_id = _call_candidate_target_site(
                            forest=forest,
                            candidate=candidate,
                        )
                        key = (suite_id, candidate_id)
                        if key in seen:
                            continue
                        seen.add(key)
                        forest.add_alt(
                            "CallCandidate",
                            (suite_id, candidate_id),
                            evidence={
                                "resolution": resolution.status,
                                "phase": resolution.phase,
                                "callee": resolution.callee_key,
                            },
                        )
                    continue
                obligation_kind_by_status = {
                    "unresolved_internal": "unresolved_internal_callee",
                    "unresolved_dynamic": "unresolved_dynamic_callee",
                }
                obligation_kind = obligation_kind_by_status.get(resolution.status)
                if obligation_kind is not None and suite_id not in obligation_seen:
                    obligation_seen.add(suite_id)
                    forest.add_alt(
                        "CallResolutionObligation",
                        (suite_id,),
                        evidence={
                            "phase": resolution.phase,
                            "callee": call.callee,
                            "kind": obligation_kind,
                        },
                    )


_GraphNode = TypeVar("_GraphNode", bound=Hashable)


def _sorted_graph_nodes(
    nodes: Iterable[_GraphNode],
) -> list[_GraphNode]:
    try:
        return sort_once(nodes, source = 'src/gabion/analysis/dataflow_audit.py:5702')
    except TypeError:
        return sort_once(nodes, key=lambda item: repr(item), source = 'src/gabion/analysis/dataflow_audit.py:5704')


def _collect_recursive_nodes(
    edges: Mapping[_GraphNode, set[_GraphNode]],
) -> set[_GraphNode]:
    check_deadline()
    index = 0
    stack: list[_GraphNode] = []
    on_stack: set[_GraphNode] = set()
    indices: dict[_GraphNode, int] = {}
    lowlink: dict[_GraphNode, int] = {}
    recursive: set[_GraphNode] = set()

    def _strongconnect(node: _GraphNode) -> None:
        check_deadline()
        nonlocal index
        indices[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for succ in edges.get(node, set()):
            check_deadline()
            if succ not in indices:
                _strongconnect(succ)
                lowlink[node] = min(lowlink[node], lowlink.get(succ, lowlink[node]))
            elif succ in on_stack:
                lowlink[node] = min(lowlink[node], indices.get(succ, lowlink[node]))
        if lowlink.get(node) == indices.get(node):
            scc: list[str] = []
            while True:
                check_deadline()
                w = stack.pop()
                on_stack.discard(w)
                scc.append(w)
                if w == node:
                    break
            if len(scc) > 1:
                recursive.update(scc)
            else:
                if node in edges.get(node, set()):
                    recursive.add(node)

    for node in edges:
        check_deadline()
        if node not in indices:
            _strongconnect(node)
    return recursive


def _collect_recursive_functions(edges: Mapping[str, set[str]]) -> set[str]:
    return _collect_recursive_nodes(edges)


def _reachable_from_roots(
    edges: Mapping[_GraphNode, set[_GraphNode]],
    roots: set[_GraphNode],
) -> set[_GraphNode]:
    check_deadline()
    reachable: set[_GraphNode] = set()
    queue: deque[_GraphNode] = deque(_sorted_graph_nodes(roots))
    while queue:
        check_deadline()
        node = queue.popleft()
        if node in reachable:
            continue
        reachable.add(node)
        for succ in _sorted_graph_nodes(edges.get(node, set())):
            check_deadline()
            if succ not in reachable:
                queue.append(succ)
    return reachable


@dataclass(frozen=True)
class _DeadlineArgInfo:
    kind: str
    param: OptionalString = None
    const: OptionalString = None


def _bind_call_args(
    call_node: ast.Call,
    callee: FunctionInfo,
    *,
    strictness: str,
) -> dict[str, ast.AST]:
    check_deadline()
    pos_params = (
        list(callee.positional_params)
        if callee.positional_params
        else list(callee.params)
    )
    kwonly_params = set(callee.kwonly_params or ())
    named_params = set(pos_params) | kwonly_params
    mapping: dict[str, ast.AST] = {}
    star_args: list[ast.AST] = []
    star_kwargs: list[ast.AST] = []
    for idx, arg in enumerate(call_node.args):
        check_deadline()
        if type(arg) is ast.Starred:
            star_args.append(cast(ast.Starred, arg).value)
            continue
        if idx < len(pos_params):
            mapping[pos_params[idx]] = arg
        elif callee.vararg is not None:
            mapping.setdefault(callee.vararg, arg)
    for kw in call_node.keywords:
        check_deadline()
        if kw.arg is None:
            star_kwargs.append(kw.value)
            continue
        if kw.arg in named_params:
            mapping[kw.arg] = kw.value
        elif callee.kwarg is not None:
            mapping.setdefault(callee.kwarg, kw.value)
    if strictness == "low":
        remaining = [p for p in sort_once(named_params, source = 'src/gabion/analysis/dataflow_audit.py:5822') if p not in mapping]
        if len(star_args) == 1 and type(star_args[0]) is ast.Name:
            for param in remaining:
                check_deadline()
                mapping.setdefault(param, star_args[0])
        if len(star_kwargs) == 1 and type(star_kwargs[0]) is ast.Name:
            for param in remaining:
                check_deadline()
                mapping.setdefault(param, star_kwargs[0])
    return mapping


def _caller_param_bindings_for_call(
    call: CallArgs,
    callee: FunctionInfo,
    *,
    strictness: str,
) -> dict[str, set[str]]:
    check_deadline()
    pos_params = (
        list(callee.positional_params)
        if callee.positional_params
        else list(callee.params)
    )
    kwonly_params = set(callee.kwonly_params or ())
    named_params = set(pos_params) | kwonly_params
    mapping: dict[str, set[str]] = defaultdict(set)
    mapped_params: set[str] = set()
    for pos_idx, caller_param in call.pos_map.items():
        check_deadline()
        idx = int(pos_idx)
        if idx < len(pos_params):
            callee_param = pos_params[idx]
        elif callee.vararg is not None:
            callee_param = callee.vararg
        else:
            continue
        mapped_params.add(callee_param)
        mapping[callee_param].add(caller_param)
    for kw_name, caller_param in call.kw_map.items():
        check_deadline()
        if kw_name in named_params:
            mapped_params.add(kw_name)
            mapping[kw_name].add(caller_param)
        elif callee.kwarg is not None:
            mapped_params.add(callee.kwarg)
            mapping[callee.kwarg].add(caller_param)
    if strictness == "low":
        remaining = [p for p in sort_once(named_params, source = 'src/gabion/analysis/dataflow_audit.py:5870') if p not in mapped_params]
        if callee.vararg is not None and callee.vararg not in mapped_params:
            remaining.append(callee.vararg)
        if callee.kwarg is not None and callee.kwarg not in mapped_params:
            remaining.append(callee.kwarg)
        if len(call.star_pos) == 1:
            _, star_param = call.star_pos[0]
            for param in remaining:
                check_deadline()
                mapping[param].add(star_param)
        if len(call.star_kw) == 1:
            star_param = call.star_kw[0]
            for param in remaining:
                check_deadline()
                mapping[param].add(star_param)
    return mapping


def _classify_deadline_expr(
    expr: ast.AST,
    *,
    alias_to_param: Mapping[str, str],
    origin_vars: set[str],
) -> _DeadlineArgInfo:
    expr_type = type(expr)
    if expr_type is ast.Name:
        name = cast(ast.Name, expr).id
        if name in alias_to_param:
            return _DeadlineArgInfo(kind="param", param=alias_to_param[name])
        if name in origin_vars:
            return _DeadlineArgInfo(kind="origin", param=name)
    if _is_deadline_origin_call(expr):
        return _DeadlineArgInfo(kind="origin")
    if expr_type is ast.Constant:
        constant_value = cast(ast.Constant, expr).value
        if constant_value is None:
            return _DeadlineArgInfo(kind="none")
        return _DeadlineArgInfo(kind="const", const=repr(constant_value))
    return _DeadlineArgInfo(kind="unknown")


def _fallback_deadline_arg_info(
    call: CallArgs,
    callee: FunctionInfo,
    *,
    strictness: str,
) -> dict[str, _DeadlineArgInfo]:
    check_deadline()
    pos_params = (
        list(callee.positional_params)
        if callee.positional_params
        else list(callee.params)
    )
    kwonly_params = set(callee.kwonly_params or ())
    named_params = set(pos_params) | kwonly_params
    mapping: dict[str, _DeadlineArgInfo] = {}
    for idx_str, caller_param in call.pos_map.items():
        check_deadline()
        idx = int(idx_str)
        if idx < len(pos_params):
            mapping[pos_params[idx]] = _DeadlineArgInfo(kind="param", param=caller_param)
        elif callee.vararg is not None:
            mapping.setdefault(callee.vararg, _DeadlineArgInfo(kind="param", param=caller_param))
    for idx_str, const_val in call.const_pos.items():
        check_deadline()
        idx = int(idx_str)
        if idx < len(pos_params):
            kind = "none" if const_val == "None" else "const"
            mapping[pos_params[idx]] = _DeadlineArgInfo(kind=kind, const=const_val)
        elif callee.vararg is not None:
            kind = "none" if const_val == "None" else "const"
            mapping.setdefault(callee.vararg, _DeadlineArgInfo(kind=kind, const=const_val))
    for idx_str in call.non_const_pos:
        check_deadline()
        idx = int(idx_str)
        if idx < len(pos_params):
            mapping[pos_params[idx]] = _DeadlineArgInfo(kind="unknown")
        elif callee.vararg is not None:
            mapping.setdefault(callee.vararg, _DeadlineArgInfo(kind="unknown"))
    for kw_name, caller_param in call.kw_map.items():
        check_deadline()
        if kw_name in named_params:
            mapping[kw_name] = _DeadlineArgInfo(kind="param", param=caller_param)
        elif callee.kwarg is not None:
            mapping.setdefault(callee.kwarg, _DeadlineArgInfo(kind="param", param=caller_param))
    for kw_name, const_val in call.const_kw.items():
        check_deadline()
        if kw_name in named_params:
            kind = "none" if const_val == "None" else "const"
            mapping[kw_name] = _DeadlineArgInfo(kind=kind, const=const_val)
        elif callee.kwarg is not None:
            kind = "none" if const_val == "None" else "const"
            mapping.setdefault(callee.kwarg, _DeadlineArgInfo(kind=kind, const=const_val))
    for kw_name in call.non_const_kw:
        check_deadline()
        if kw_name in named_params:
            mapping[kw_name] = _DeadlineArgInfo(kind="unknown")
        elif callee.kwarg is not None:
            mapping.setdefault(callee.kwarg, _DeadlineArgInfo(kind="unknown"))
    if strictness == "low":
        remaining = [p for p in sort_once(named_params, source = 'src/gabion/analysis/dataflow_audit.py:5968') if p not in mapping]
        if len(call.star_pos) == 1:
            _, star_param = call.star_pos[0]
            for param in remaining:
                check_deadline()
                mapping.setdefault(param, _DeadlineArgInfo(kind="param", param=star_param))
        if len(call.star_kw) == 1:
            star_param = call.star_kw[0]
            for param in remaining:
                check_deadline()
                mapping.setdefault(param, _DeadlineArgInfo(kind="param", param=star_param))
    return mapping


def _deadline_arg_info_map(
    call: CallArgs,
    callee: FunctionInfo,
    *,
    call_node: OptionalAstCall,
    alias_to_param: Mapping[str, str],
    origin_vars: set[str],
    strictness: str,
) -> dict[str, _DeadlineArgInfo]:
    check_deadline()
    if call_node is None:
        return _fallback_deadline_arg_info(call, callee, strictness=strictness)
    expr_map = _bind_call_args(call_node, callee, strictness=strictness)
    info_map: dict[str, _DeadlineArgInfo] = {}
    for param, expr in expr_map.items():
        check_deadline()
        info_map[param] = _classify_deadline_expr(
            expr,
            alias_to_param=alias_to_param,
            origin_vars=origin_vars,
        )
    return info_map


def _deadline_loop_forwarded_params(
    *,
    qual: str,
    loop_fact: _DeadlineLoopFacts,
    deadline_params: Mapping[str, set[str]],
    call_infos: Mapping[str, list[tuple[CallArgs, FunctionInfo, dict[str, "_DeadlineArgInfo"]]]],
) -> set[str]:
    forwarded: set[str] = set()
    caller_params = deadline_params.get(qual, set())
    if not caller_params:
        return forwarded
    for call, callee, arg_info in call_infos.get(qual, []):
        check_deadline()
        if call.span is not None and call.span in loop_fact.call_spans:
            for callee_param in deadline_params.get(callee.qual, set()):
                check_deadline()
                info = arg_info.get(callee_param)
                if info is not None and info.kind == "param" and info.param in caller_params:
                    forwarded.add(info.param)
    return forwarded


from gabion.analysis.dataflow_obligations import (
    collect_deadline_obligations as _collect_deadline_obligations,
)


def _spec_row_span(row: Mapping[str, JSONValue]):
    def _coerce(name: str, value: JSONValue) -> int:
        if value is None:
            never(
                f"projection spec missing {name}",
                field=name,
            )
        try:
            return int(value)
        except (TypeError, ValueError):
            never(
                f"projection spec {name} must be an int",
                field=name,
                value=value,
            )

    line = _coerce("span_line", row.get("span_line"))
    col = _coerce("span_col", row.get("span_col"))
    end_line = _coerce("span_end_line", row.get("span_end_line"))
    end_col = _coerce("span_end_col", row.get("span_end_col"))
    if line < 0 or col < 0 or end_line < 0 or end_col < 0:
        never(
            "projection spec span fields must be non-negative",
            span_line=line,
            span_col=col,
            span_end_line=end_line,
            span_end_col=end_col,
        )
    return (line, col, end_line, end_col)


def _materialize_projection_spec_rows(
    *,
    spec: ProjectionSpec,
    projected: Iterable[Mapping[str, JSONValue]],
    forest: Forest,
    row_to_site: Callable[[Mapping[str, JSONValue]], NodeIdOrNone],
) -> None:
    spec_identity = projection_spec_hash(spec)
    spec_site = forest.add_spec_site(
        spec_hash=spec_identity,
        spec_name=str(spec.name),
        spec_domain=str(spec.domain),
        spec_version=int(spec.spec_version) if spec.spec_version else None,
    )
    for row in projected:
        check_deadline()
        site_id = row_to_site(row)
        if site_id is not None:
            evidence: dict[str, object] = {
                "spec_name": str(spec.name),
                "spec_hash": spec_identity,
            }
            for key, value in row.items():
                check_deadline()
                evidence[str(key)] = value
            forest.add_alt("SpecFacet", (spec_site, site_id), evidence=evidence)


def _suite_order_depth(suite_kind: str) -> int:
    if suite_kind in {"function", "spec"}:
        return 0
    return 1


def _suite_order_relation(
    forest: Forest,
) -> tuple[list[dict[str, JSONValue]], dict[tuple[object, ...], NodeId]]:
    alt_degree: Counter[NodeId] = Counter()
    for alt in forest.alts:
        check_deadline()
        if alt.kind == "SpecFacet":
            continue
        for node_id in alt.inputs:
            check_deadline()
            alt_degree[node_id] += 1
    relation: list[dict[str, JSONValue]] = []
    suite_index: dict[tuple[object, ...], NodeId] = {}
    for node_id, node in forest.nodes.items():
        check_deadline()
        if node_id.kind != "SuiteSite":
            continue
        suite_kind = str(node.meta.get("suite_kind", "") or "")
        if suite_kind == "spec":
            continue
        path = str(node.meta.get("path", "") or "")
        qual = str(node.meta.get("qual", "") or "")
        if not path or not qual:
            never(
                "suite order requires path/qual",
                path=path,
                qual=qual,
                suite_kind=suite_kind,
            )
        span = node.meta.get("span")
        parsed_span = int_tuple4_or_none(span)
        if parsed_span is None:
            never(
                "suite order requires span",
                path=path,
                qual=qual,
                suite_kind=suite_kind,
                span=span,
            )
        span_line, span_col, span_end_line, span_end_col = parsed_span
        depth = _suite_order_depth(suite_kind)
        complexity = int(alt_degree.get(node_id, 0))
        order_key: list[JSONValue] = [
            depth,
            complexity,
            path,
            qual,
            span_line,
            span_col,
            span_end_line,
            span_end_col,
        ]
        relation.append(
            {
                "suite_path": path,
                "suite_qual": qual,
                "suite_kind": suite_kind,
                "span_line": span_line,
                "span_col": span_col,
                "span_end_line": span_end_line,
                "span_end_col": span_end_col,
                "depth": depth,
                "complexity": complexity,
                "order_key": order_key,
            }
        )
        suite_index[
            (path, qual, suite_kind, span_line, span_col, span_end_line, span_end_col)
        ] = node_id
    relation = sort_once(
        relation,
        key=lambda row: (
            int(row.get("depth", 0) or 0),
            int(row.get("complexity", 0) or 0),
            str(row.get("suite_path", "") or ""),
            str(row.get("suite_qual", "") or ""),
            int(row.get("span_line", -1) or -1),
            int(row.get("span_col", -1) or -1),
            int(row.get("span_end_line", -1) or -1),
            int(row.get("span_end_col", -1) or -1),
        ),
        source="src/gabion/analysis/dataflow_audit.py:_suite_order_relation.relation",
    )
    return relation, suite_index


def _suite_order_row_to_site(
    row: Mapping[str, JSONValue],
    suite_index: Mapping[tuple[object, ...], NodeId],
):
    path = str(row.get("suite_path", "") or "")
    qual = str(row.get("suite_qual", "") or "")
    suite_kind = str(row.get("suite_kind", "") or "")
    if not path or not qual or not suite_kind:
        return None
    try:
        span_line = int(row.get("span_line", -1))
        span_col = int(row.get("span_col", -1))
        span_end_line = int(row.get("span_end_line", -1))
        span_end_col = int(row.get("span_end_col", -1))
    except (TypeError, ValueError):
        return None
    key = (
        path,
        qual,
        suite_kind,
        span_line,
        span_col,
        span_end_line,
        span_end_col,
    )
    return suite_index.get(key)


def _materialize_suite_order_spec(
    *,
    forest: Forest,
) -> None:
    relation, suite_index = _suite_order_relation(forest)
    if not relation:
        return
    projected = apply_spec(SUITE_ORDER_SPEC, relation)

    _materialize_projection_spec_rows(
        spec=SUITE_ORDER_SPEC,
        projected=projected,
        forest=forest,
        row_to_site=lambda row: _suite_order_row_to_site(row, suite_index),
    )


def _ambiguity_suite_relation(
    forest: Forest,
) -> list[dict[str, JSONValue]]:
    relation: list[dict[str, JSONValue]] = []
    for alt in forest.alts:
        check_deadline()
        if alt.kind != "CallCandidate":
            continue
        if len(alt.inputs) < 2:
            continue
        suite_id = alt.inputs[0]
        suite_node = forest.nodes.get(suite_id)
        if suite_node is not None and suite_node.kind == "SuiteSite":
            suite_kind = str(suite_node.meta.get("suite_kind", "") or "")
            if suite_kind == "call":
                path = str(suite_node.meta.get("path", "") or "")
                qual = str(suite_node.meta.get("qual", "") or "")
                if not path or not qual:
                    never(
                        "ambiguity suite requires path/qual",
                        path=path,
                        qual=qual,
                        suite_kind=suite_kind,
                    )
                span = suite_node.meta.get("span")
                parsed_span = int_tuple4_or_none(span)
                if parsed_span is None:
                    never(
                        "ambiguity suite requires span",
                        path=path,
                        qual=qual,
                        suite_kind=suite_kind,
                        span=span,
                    )
                span_line, span_col, span_end_line, span_end_col = parsed_span
                relation.append(
                    {
                        "suite_path": path,
                        "suite_qual": qual,
                        "suite_kind": suite_kind,
                        "span_line": span_line,
                        "span_col": span_col,
                        "span_end_line": span_end_line,
                        "span_end_col": span_end_col,
                        "kind": str(alt.evidence.get("kind", "") or ""),
                        "phase": str(alt.evidence.get("phase", "") or ""),
                    }
                )
    return relation


def _ambiguity_suite_row_to_suite(
    row: Mapping[str, JSONValue],
    forest: Forest,
) -> NodeId:
    path = str(row.get("suite_path", "") or "")
    qual = str(row.get("suite_qual", "") or "")
    suite_kind = str(row.get("suite_kind", "") or "")
    if not path or not qual or not suite_kind:
        never(
            "ambiguity suite row missing suite identity",
            path=path,
            qual=qual,
            suite_kind=suite_kind,
        )
    span = _spec_row_span(row)
    require_not_none(
        span,
        reason="ambiguity suite row missing span",
        strict=True,
    )
    return forest.add_suite_site(
        path,
        qual,
        suite_kind,
        span=span,
    )


def _ambiguity_virtual_count_gt_1(
    row: Mapping[str, JSONValue],
    _params: Mapping[str, JSONValue],
) -> bool:
    try:
        return int(row.get("count", 0) or 0) > 1
    except (TypeError, ValueError):
        return False


def _materialize_ambiguity_suite_agg_spec(
    *,
    forest: Forest,
) -> None:
    relation = _ambiguity_suite_relation(forest)
    if not relation:
        return
    projected = apply_spec(AMBIGUITY_SUITE_AGG_SPEC, relation)
    _materialize_projection_spec_rows(
        spec=AMBIGUITY_SUITE_AGG_SPEC,
        projected=projected,
        forest=forest,
        row_to_site=lambda row: _ambiguity_suite_row_to_suite(row, forest),
    )


def _materialize_ambiguity_virtual_set_spec(
    *,
    forest: Forest,
) -> None:
    relation = _ambiguity_suite_relation(forest)
    if not relation:
        return

    projected = apply_spec(
        AMBIGUITY_VIRTUAL_SET_SPEC,
        relation,
        op_registry={"count_gt_1": _ambiguity_virtual_count_gt_1},
    )
    _materialize_projection_spec_rows(
        spec=AMBIGUITY_VIRTUAL_SET_SPEC,
        projected=projected,
        forest=forest,
        row_to_site=lambda row: _ambiguity_suite_row_to_suite(row, forest),
    )


def _summarize_deadline_obligations(
    entries: list[JSONObject],
    *,
    max_entries: int = 20,
    forest: Forest,
) -> list[str]:
    check_deadline()
    if not entries:
        return []
    relation: list[dict[str, JSONValue]] = []
    for entry in entries:
        check_deadline()
        site = mapping_or_empty(entry.get("site"))
        path = str(site.get("path", "") or "")
        function = str(site.get("function", "") or "")
        span = entry.get("span")
        line = col = end_line = end_col = -1
        span_entries = sequence_or_none(span)
        if span_entries is not None and len(span_entries) == 4:
            try:
                line = int(span_entries[0])
                col = int(span_entries[1])
                end_line = int(span_entries[2])
                end_col = int(span_entries[3])
            except (TypeError, ValueError):
                line = col = end_line = end_col = -1
        relation.append(
            {
                "status": str(entry.get("status", "UNKNOWN") or "UNKNOWN"),
                "kind": str(entry.get("kind", "") or ""),
                "detail": str(entry.get("detail", "") or ""),
                "site_path": path,
                "site_function": function,
                "site_suite_id": str(site.get("suite_id", "") or ""),
                "site_suite_kind": str(site.get("suite_kind", "") or ""),
                "span_line": line,
                "span_col": col,
                "span_end_line": end_line,
                "span_end_col": end_col,
                "deadline_id": str(entry.get("deadline_id", "") or ""),
            }
        )

    projected = apply_spec(DEADLINE_OBLIGATIONS_SUMMARY_SPEC, relation)
    def _row_to_site(row: Mapping[str, JSONValue]):
        path = str(row.get("site_path", "") or "")
        function = str(row.get("site_function", "") or "")
        if not path or not function:
            return None
        span = _spec_row_span(row)
        path_name = Path(path).name
        suite_kind = str(row.get("site_suite_kind", "") or "")
        require_not_none(
            span,
            reason="projection spec row missing span",
            strict=True,
            path=path,
            function=function,
        )
        if suite_kind:
            return forest.add_suite_site(path_name, function, suite_kind, span=span)
        return forest.add_site(path_name, function, span)

    _materialize_projection_spec_rows(
        spec=DEADLINE_OBLIGATIONS_SUMMARY_SPEC,
        projected=projected,
        forest=forest,
        row_to_site=_row_to_site,
    )
    lines: list[str] = []
    lines.extend(
        spec_metadata_lines_from_payload(
            spec_metadata_payload(DEADLINE_OBLIGATIONS_SUMMARY_SPEC)
        )
    )
    for entry in projected[:max_entries]:
        check_deadline()
        path = entry.get("site_path") or "?"
        function = entry.get("site_function") or "?"
        span = _format_span_fields(
            entry.get("span_line", -1),
            entry.get("span_col", -1),
            entry.get("span_end_line", -1),
            entry.get("span_end_col", -1),
        )
        status = entry.get("status", "UNKNOWN")
        kind = entry.get("kind", "?")
        detail = entry.get("detail", "")
        suffix = f"@{span}" if span else ""
        lines.append(
            f"{path}:{function}{suffix} status={status} kind={kind} {detail}".strip()
        )
    if len(projected) > max_entries:
        lines.append(f"... {len(projected) - max_entries} more")
    return lines


def _span_line_col(span):
    parsed = int_tuple4_or_none(span)
    if parsed is None:
        return None, None
    return parsed[0] + 1, parsed[1] + 1


def _deadline_lint_lines(entries: list[JSONObject]) -> list[str]:
    check_deadline()
    lines: list[str] = []
    for entry in entries:
        check_deadline()
        site = mapping_or_none(entry.get("site")) or {}
        path = str(site.get("path", "") or "")
        line, col = _span_line_col(entry.get("span"))
        if not path:
            continue
        status = entry.get("status", "UNKNOWN")
        kind = entry.get("kind", "?")
        detail = entry.get("detail", "")
        message = f"{status} {kind} {detail}".strip()
        lines.append(_lint_line(path, line or 1, col or 1, "GABION_DEADLINE", message))
    return lines


def _summarize_exception_obligations(
    entries: list[JSONObject],
    *,
    max_entries: int = 10,
) -> list[str]:
    check_deadline()
    if not entries:
        return []
    lines: list[str] = []
    for entry in entries[:max_entries]:
        check_deadline()
        site = entry.get("site", {})
        path = site.get("path", "?")
        function = site.get("function", "?")
        bundle = site.get("bundle", [])
        status = entry.get("status", "UNKNOWN")
        source = entry.get("source_kind", "?")
        exception_name = entry.get("exception_name")
        protocol = entry.get("protocol")
        reason_code = entry.get("handledness_reason_code", "UNKNOWN_REASON")
        refinement = str(entry.get("type_refinement_opportunity", "") or "")
        suffix = f" reason={reason_code}"
        if exception_name:
            suffix += f" exception={exception_name}"
        if protocol:
            suffix += f" protocol={protocol}"
        if refinement:
            suffix += f" refine={refinement}"
        lines.append(
            f"{path}:{function} bundle={bundle} source={source} status={status}{suffix}"
        )
    if len(entries) > max_entries:
        lines.append(f"... {len(entries) - max_entries} more")
    return lines


def _summarize_call_ambiguities(
    entries: list[JSONObject],
    *,
    max_entries: int = 20,
) -> list[str]:
    check_deadline()
    if not entries:
        return []
    relation: list[dict[str, JSONValue]] = []
    for entry in entries:
        check_deadline()
        if type(entry) is not dict:
            continue
        entry_payload = cast(Mapping[str, JSONValue], entry)
        kind = str(entry_payload.get("kind", "") or "unknown")
        site_payload = entry_payload.get("site", {})
        site = site_payload if type(site_payload) is dict else {}
        site_mapping = cast(Mapping[str, JSONValue], site)
        path = str(site_mapping.get("path", "") or "")
        function = str(site_mapping.get("function", "") or "")
        span = site_mapping.get("span")
        line = col = end_line = end_col = -1
        if type(span) is list and len(cast(list[JSONValue], span)) == 4:
            span_values = cast(list[JSONValue], span)
            try:
                line = int(span_values[0])
                col = int(span_values[1])
                end_line = int(span_values[2])
                end_col = int(span_values[3])
            except (TypeError, ValueError):
                line = col = end_line = end_col = -1
        candidate_count = entry_payload.get("candidate_count")
        try:
            candidate_count = int(candidate_count) if candidate_count is not None else 0
        except (TypeError, ValueError):
            candidate_count = 0
        relation.append(
            {
                "kind": kind,
                "site_path": path,
                "site_function": function,
                "span_line": line,
                "span_col": col,
                "span_end_line": end_line,
                "span_end_col": end_col,
                "candidate_count": candidate_count,
            }
        )
    projected = apply_spec(AMBIGUITY_SUMMARY_SPEC, relation)
    counts: dict[str, int] = {}
    for row in relation:
        check_deadline()
        kind = str(row.get("kind", "") or "unknown")
        counts[kind] = counts.get(kind, 0) + 1
    lines: list[str] = []
    lines.extend(
        spec_metadata_lines_from_payload(spec_metadata_payload(AMBIGUITY_SUMMARY_SPEC))
    )
    lines.append("Counts by witness kind:")
    for kind in sort_once(counts, source = 'src/gabion/analysis/dataflow_audit.py:7425'):
        check_deadline()
        lines.append(f"- {kind}: {counts[kind]}")
    lines.append("Top ambiguous sites:")
    for row in projected[:max_entries]:
        check_deadline()
        path = row.get("site_path") or "?"
        function = row.get("site_function") or "?"
        span = _format_span_fields(
            row.get("span_line", -1),
            row.get("span_col", -1),
            row.get("span_end_line", -1),
            row.get("span_end_col", -1),
        )
        count = row.get("candidate_count", 0)
        suffix = f"@{span}" if span else ""
        lines.append(f"- {path}:{function}{suffix} candidates={count}")
    if len(projected) > max_entries:
        lines.append(f"... {len(projected) - max_entries} more")
    return lines


def _format_span_fields(
    line: object,
    col: object,
    end_line: object,
    end_col: object,
) -> str:
    # dataflow-bundle: col, end_col, end_line, line
    try:
        line_value = int(line)
        col_value = int(col)
        end_line_value = int(end_line)
        end_col_value = int(end_col)
    except (TypeError, ValueError):
        return ""
    if (
        line_value < 0
        or col_value < 0
        or end_line_value < 0
        or end_col_value < 0
    ):
        return ""
    return (
        f"{line_value + 1}:{col_value + 1}-"
        f"{end_line_value + 1}:{end_col_value + 1}"
    )


def _summarize_never_invariants(
    entries: list[JSONObject],
    *,
    max_entries: int = 50,
    include_proven_unreachable: bool = True,
) -> list[str]:
    check_deadline()
    if not entries:
        return []
    def _format_span(row: Mapping[str, JSONValue]) -> str:
        return _format_span_fields(
            row.get("span_line", -1),
            row.get("span_col", -1),
            row.get("span_end_line", -1),
            row.get("span_end_col", -1),
        )

    def _format_site(row: Mapping[str, JSONValue]) -> str:
        path = row.get("site_path") or "?"
        function = row.get("site_function") or "?"
        span = _format_span(row)
        if span:
            return f"{path}:{function}@{span}"
        return f"{path}:{function}"

    def _format_evidence(row: Mapping[str, JSONValue], status: str) -> str:
        witness_ref = row.get("witness_ref")
        env = row.get("environment_ref")
        undecidable = row.get("undecidable_reason") or ""
        parts: list[str] = []
        if status == "VIOLATION":
            if witness_ref:
                parts.append(f"witness={witness_ref}")
            if env:
                parts.append(f"env={json.dumps(env, sort_keys=False)}")
        elif status == "PROVEN_UNREACHABLE":
            if witness_ref:
                parts.append(f"deadness={witness_ref}")
            if env:
                parts.append(f"env={json.dumps(env, sort_keys=False)}")
        else:
            if undecidable:
                parts.append(f"why={undecidable}")
            else:
                parts.append("why=no witness env available")
        return "; ".join(parts)

    def _never_status_allowed(
        row: Mapping[str, JSONValue], params: Mapping[str, JSONValue]
    ) -> bool:
        status = str(row.get("status", "UNKNOWN") or "UNKNOWN")
        if status == "PROVEN_UNREACHABLE":
            include = params.get("include_proven_unreachable", True)
            return bool(include)
        return True

    relation: list[dict[str, JSONValue]] = []
    for entry in entries:
        check_deadline()
        if type(entry) is not dict:
            continue
        entry_payload = cast(Mapping[str, JSONValue], entry)
        status = str(entry_payload.get("status", "UNKNOWN") or "UNKNOWN")
        raw_site = entry_payload.get("site")
        site = raw_site if type(raw_site) is dict else {}
        site_mapping = cast(Mapping[str, JSONValue], site)
        path = str(site_mapping.get("path", "") or "")
        function = str(site_mapping.get("function", "") or "")
        span = entry_payload.get("span")
        line = col = end_line = end_col = -1
        if type(span) is list and len(cast(list[JSONValue], span)) == 4:
            span_values = cast(list[JSONValue], span)
            try:
                line = int(span_values[0])
                col = int(span_values[1])
                end_line = int(span_values[2])
                end_col = int(span_values[3])
            except (TypeError, ValueError):
                line = col = end_line = end_col = -1
        relation.append(
            {
                "status": status,
                "status_rank": _NEVER_STATUS_ORDER.get(status, 3),
                "site_path": path,
                "site_function": function,
                "span_line": line,
                "span_col": col,
                "span_end_line": end_line,
                "span_end_col": end_col,
                "never_id": str(entry.get("never_id", "") or ""),
                "reason": str(entry.get("reason", "") or ""),
                "witness_ref": entry.get("witness_ref"),
                "environment_ref": entry.get("environment_ref"),
                "undecidable_reason": str(entry.get("undecidable_reason", "") or ""),
            }
        )

    params = dict(NEVER_INVARIANTS_SPEC.params)
    params.update(
        {
            "max_entries": max_entries,
            "include_proven_unreachable": include_proven_unreachable,
        }
    )
    projected = apply_spec(
        NEVER_INVARIANTS_SPEC,
        relation,
        op_registry={"never_status_allowed": _never_status_allowed},
        params_override=params,
    )
    ordered_statuses = list(params.get("ordered_statuses") or [])
    grouped: dict[str, list[dict[str, JSONValue]]] = {}
    for row in projected:
        check_deadline()
        status = str(row.get("status", "UNKNOWN") or "UNKNOWN")
        grouped.setdefault(status, []).append(row)
    extra_statuses = sort_once(
        (status for status in grouped.keys() if status not in ordered_statuses),
        source="src/gabion/analysis/dataflow_audit.py:7586",
    )
    lines: list[str] = []
    lines.extend(
        spec_metadata_lines_from_payload(spec_metadata_payload(NEVER_INVARIANTS_SPEC))
    )
    for status in ordered_statuses + extra_statuses:
        check_deadline()
        if status == "PROVEN_UNREACHABLE" and not include_proven_unreachable:
            continue
        items = grouped.get(status) or []
        if not items:
            continue
        lines.append(f"{status}:")
        for entry in items[:max_entries]:
            check_deadline()
            reason = entry.get("reason") or ""
            evidence = _format_evidence(entry, status)
            bits: list[str] = []
            if reason:
                bits.append(f"reason={reason}")
            if evidence:
                bits.append(evidence)
            suffix = f" ({'; '.join(bits)})" if bits else ""
            lines.append(f"- {_format_site(entry)} never(){suffix}")
        if len(items) > max_entries:
            lines.append(f"... {len(items) - max_entries} more")
    return lines


def _exception_protocol_warnings(entries: list[JSONObject]) -> list[str]:
    check_deadline()
    warnings: list[str] = []
    for entry in entries:
        check_deadline()
        if entry.get("protocol") != "never":
            continue
        if entry.get("status") == "DEAD":
            continue
        site = entry.get("site", {}) or {}
        path = site.get("path", "?")
        function = site.get("function", "?")
        exception_name = entry.get("exception_name") or "?"
        status = entry.get("status", "UNKNOWN")
        warnings.append(
            f"{path}:{function} raises {exception_name} (protocol=never, status={status})"
        )
    return warnings


def _exception_protocol_evidence(entries: list[JSONObject]) -> list[str]:
    check_deadline()
    lines: list[str] = []
    for entry in entries:
        check_deadline()
        if entry.get("protocol") != "never":
            continue
        exception_id = entry.get("exception_path_id", "?")
        exception_name = entry.get("exception_name") or "?"
        status = entry.get("status", "UNKNOWN")
        lines.append(
            f"{exception_id} exception={exception_name} protocol=never status={status}"
        )
    return lines


def _parse_lint_location(line: str):
    return _ds_parse_lint_location(line)

def _lint_line(path: str, line: int, col: int, code: str, message: str) -> str:
    return f"{path}:{line}:{col}: {code} {message}".strip()


def _parse_lint_remainder(remainder: str) -> tuple[str, str]:
    text = remainder.strip()
    if not text:
        return ("GABION_UNKNOWN", "")
    head, *tail = text.split(maxsplit=1)
    code = head.strip() or "GABION_UNKNOWN"
    message = tail[0].strip() if tail else ""
    return (code, message)


def _lint_rows_from_lines(
    lines: Iterable[str],
    *,
    source: str,
) -> list[dict[str, JSONValue]]:
    check_deadline()
    rows: list[dict[str, JSONValue]] = []
    for line in lines:
        check_deadline()
        parsed = _parse_lint_location(line)
        if parsed is not None:
            path, lineno, col, remainder = parsed
            code, message = _parse_lint_remainder(remainder)
            rows.append(
                {
                    "path": path,
                    "line": int(lineno),
                    "col": int(col),
                    "code": code,
                    "message": message,
                    "source": source,
                }
            )
    return rows


def _add_interned_alt(
    *,
    forest: Forest,
    kind: str,
    inputs: Iterable[NodeId],
    evidence = None,
) -> Alt:
    return forest.add_alt(kind, inputs, evidence=evidence)


def _materialize_lint_rows(
    *,
    forest: Forest,
    rows: Iterable[Mapping[str, JSONValue]],
) -> None:
    check_deadline()
    for row in rows:
        check_deadline()
        path = str(row.get("path", "") or "")
        if not path:
            continue
        try:
            lineno = int(row.get("line", 1) or 1)
            col = int(row.get("col", 1) or 1)
        except (TypeError, ValueError):
            continue
        code = str(row.get("code", "") or "")
        if not code:
            continue
        message = str(row.get("message", "") or "")
        source = str(row.get("source", "") or "")
        lint_node = forest.add_node(
            "LintFinding",
            (
                path,
                lineno,
                col,
                code,
                message,
            ),
            meta={
                "path": path,
                "line": lineno,
                "col": col,
                "code": code,
                "message": message,
            },
        )
        file_node = forest.add_file_site(path)
        _add_interned_alt(
            forest=forest,
            kind="LintFinding",
            inputs=(file_node, lint_node),
            evidence={"source": source},
        )


def _lint_relation_from_forest(forest: Forest) -> list[dict[str, JSONValue]]:
    check_deadline()
    by_identity: dict[tuple[str, int, int, str, str], set[str]] = {}
    for alt in forest.alts:
        check_deadline()
        if alt.kind != "LintFinding" or len(alt.inputs) < 2:
            continue
        lint_node_id = alt.inputs[1]
        if lint_node_id.kind != "LintFinding":
            continue
        lint_node = forest.nodes.get(lint_node_id)
        if lint_node is not None:
            path = str(lint_node.meta.get("path", "") or "")
            code = str(lint_node.meta.get("code", "") or "")
            message = str(lint_node.meta.get("message", "") or "")
            if not path or not code:
                continue
            try:
                line = int(lint_node.meta.get("line", 1) or 1)
                col = int(lint_node.meta.get("col", 1) or 1)
            except (TypeError, ValueError):
                continue
            key = (path, line, col, code, message)
            source = str(alt.evidence.get("source", "") or "")
            bucket = by_identity.setdefault(key, set())
            if source:
                bucket.add(source)
    relation: list[dict[str, JSONValue]] = []
    for key in sort_once(by_identity, source = 'src/gabion/analysis/dataflow_audit.py:7783'):
        check_deadline()
        path, line, col, code, message = key
        relation.append(
            {
                "path": path,
                "line": line,
                "col": col,
                "code": code,
                "message": message,
                "sources": sort_once(by_identity[key], source = 'src/gabion/analysis/dataflow_audit.py:7793'),
            }
        )
    return relation


def _project_lint_rows_from_forest(
    *,
    forest: Forest,
    relation_fn: Callable[[Forest], list[dict[str, JSONValue]]] = _lint_relation_from_forest,
    apply_spec_fn: Callable[[ProjectionSpec, list[dict[str, JSONValue]]], list[dict[str, JSONValue]]] = apply_spec,
) -> list[dict[str, JSONValue]]:
    relation = relation_fn(forest)
    if not relation:
        return []
    projected = apply_spec_fn(LINT_FINDINGS_SPEC, relation)

    def _row_to_file_site(row: Mapping[str, JSONValue]):
        path = str(row.get("path", "") or "")
        if not path:
            return None
        return forest.add_file_site(path)

    _materialize_projection_spec_rows(
        spec=LINT_FINDINGS_SPEC,
        projected=projected,
        forest=forest,
        row_to_site=_row_to_file_site,
    )
    return projected


def _materialize_report_section_lines(
    *,
    forest: Forest,
    section_key: _ReportSectionKey,
    lines: Iterable[str],
) -> None:
    check_deadline()
    report_file = forest.add_file_site("<report>")
    for idx, text in enumerate(lines):
        check_deadline()
        text_value = str(text)
        line_node = forest.add_node(
            "ReportSectionLine",
            (section_key.run_id, section_key.section, idx, text_value),
            meta={
                "run_id": section_key.run_id,
                "section": section_key.section,
                "line_index": idx,
                "text": text_value,
            },
        )
        _add_interned_alt(
            forest=forest,
            kind="ReportSectionLine",
            inputs=(report_file, line_node),
            evidence={
                "run_id": section_key.run_id,
                "section": section_key.section,
            },
        )


def _report_section_line_relation(
    *,
    forest: Forest,
    section_key: _ReportSectionKey,
) -> list[dict[str, JSONValue]]:
    check_deadline()
    relation: list[dict[str, JSONValue]] = []
    for alt in forest.alts:
        check_deadline()
        if alt.kind != "ReportSectionLine" or len(alt.inputs) < 2:
            continue
        if str(alt.evidence.get("run_id", "") or "") != section_key.run_id:
            continue
        if str(alt.evidence.get("section", "") or "") != section_key.section:
            continue
        node_id = alt.inputs[1]
        if node_id.kind != "ReportSectionLine":
            continue
        node = forest.nodes.get(node_id)
        if node is not None:
            node_run_id = str(node.meta.get("run_id", "") or "")
            node_section = str(node.meta.get("section", "") or "")
            if (
                node_run_id != section_key.run_id
                or node_section != section_key.section
            ):
                continue
            try:
                line_index = int(node.meta.get("line_index", 0) or 0)
            except (TypeError, ValueError):
                continue
            relation.append(
                {
                    "section": section_key.section,
                    "line_index": line_index,
                    "text": str(node.meta.get("text", "") or ""),
                }
            )
    return relation


def _project_report_section_lines(
    *,
    forest: Forest,
    section_key: _ReportSectionKey,
    lines: Iterable[str],
) -> list[str]:
    check_deadline()
    _materialize_report_section_lines(
        forest=forest,
        section_key=section_key,
        lines=lines,
    )
    relation = _report_section_line_relation(
        forest=forest,
        section_key=section_key,
    )
    if not relation:
        return []
    projected = apply_spec(REPORT_SECTION_LINES_SPEC, relation)
    _materialize_projection_spec_rows(
        spec=REPORT_SECTION_LINES_SPEC,
        projected=projected,
        forest=forest,
        row_to_site=lambda _row: forest.add_file_site("<report>"),
    )
    rendered: list[str] = []
    for row in projected:
        check_deadline()
        rendered.append(str(row.get("text", "") or ""))
    return rendered


def _decision_param_lint_line(
    info: "FunctionInfo",
    param: str,
    *,
    project_root,
    code: str,
    message: str,
):
    span = info.param_spans.get(param)
    if span is not None:
        path = _normalize_snapshot_path(info.path, project_root)
        line, col, _, _ = span
        return _lint_line(path, line + 1, col + 1, code, message)
    missing_lines: dict[str, str] = {}
    return missing_lines.get("lint_line")


def _decision_tier_for(
    info: "FunctionInfo",
    param: str,
    *,
    tier_map: dict[str, int],
    project_root,
):
    check_deadline()
    if not tier_map:
        return None
    span = info.param_spans.get(param)
    if span is not None:
        path = _normalize_snapshot_path(info.path, project_root)
        line, col, _, _ = span
        location = f"{path}:{line + 1}:{col + 1}"
        for key in (location, f"{location}:{param}"):
            check_deadline()
            if key in tier_map:
                return tier_map[key]
    for key in (f"{info.qual}:{param}", f"{info.qual}.{param}", param):
        check_deadline()
        if key in tier_map:
            return tier_map[key]
    return None


def _collect_transitive_callers(
    callers_by_qual: dict[str, set[str]],
    by_qual: dict[str, FunctionInfo],
) -> dict[str, set[str]]:
    check_deadline()
    transitive: dict[str, set[str]] = {}
    for qual in by_qual:
        check_deadline()
        seen: set[str] = set()
        stack = list(callers_by_qual.get(qual, set()))
        while stack:
            check_deadline()
            caller = stack.pop()
            if caller in seen:
                continue
            seen.add(caller)
            stack.extend(callers_by_qual.get(caller, set()))
        transitive[qual] = seen
    return transitive


@dataclass
class AnalysisIndex:
    by_name: dict[str, list[FunctionInfo]]
    by_qual: dict[str, FunctionInfo]
    symbol_table: SymbolTable
    class_index: dict[str, ClassInfo]
    parsed_modules_by_path: dict[Path, ast.Module] = field(default_factory=dict)
    module_parse_errors_by_path: dict[Path, Exception] = field(default_factory=dict)
    stage_cache_by_key: dict[Hashable, dict[Path, object]] = field(default_factory=dict)
    index_cache_identity: str = ""
    projection_cache_identity: str = ""
    transitive_callers: "dict[str, set[str]] | None" = None
    resolved_call_edges: 'tuple["_ResolvedCallEdge", ...] | None' = None
    resolved_transparent_call_edges: 'tuple["_ResolvedCallEdge", ...] | None' = None
    resolved_transparent_edges_by_caller: 'dict[str, tuple["_ResolvedCallEdge", ...]] | None' = None


@dataclass(frozen=True)
class _ResolvedCallEdge:
    caller: FunctionInfo
    call: CallArgs
    callee: FunctionInfo


_StageCacheValue = TypeVar("_StageCacheValue")
_IndexedPassResult = TypeVar("_IndexedPassResult")
_ResolvedEdgeAcc = TypeVar("_ResolvedEdgeAcc")
_ResolvedEdgeOut = TypeVar("_ResolvedEdgeOut")
_ModuleArtifactAcc = TypeVar("_ModuleArtifactAcc")
_ModuleArtifactOut = TypeVar("_ModuleArtifactOut")
OptionalProjectRoot = Path | None
OptionalDecorators = set[str] | None
OptionalParseFailures = list[JSONObject] | None
OptionalAnalysisIndex = AnalysisIndex | None


@dataclass(frozen=True)
class _IndexedPassContext:
    paths: list[Path]
    project_root: OptionalProjectRoot
    ignore_params: set[str]
    strictness: str
    external_filter: bool
    transparent_decorators: OptionalDecorators
    parse_failure_witnesses: list[JSONObject]
    analysis_index: AnalysisIndex


@dataclass(frozen=True)
class _IndexedPassSpec(Generic[_IndexedPassResult]):
    pass_id: str
    run: Callable[[_IndexedPassContext], _IndexedPassResult]


@dataclass(frozen=True)
class _ResolvedEdgeReducerSpec(Generic[_ResolvedEdgeAcc, _ResolvedEdgeOut]):
    reducer_id: str
    init: Callable[[], _ResolvedEdgeAcc]
    fold: Callable[[_ResolvedEdgeAcc, _ResolvedCallEdge], None]
    finish: Callable[[_ResolvedEdgeAcc], _ResolvedEdgeOut]


@dataclass(frozen=True)
class _ModuleArtifactSpec(Generic[_ModuleArtifactAcc, _ModuleArtifactOut]):
    artifact_id: str
    stage: _ParseModuleStage
    init: Callable[[], _ModuleArtifactAcc]
    fold: Callable[[_ModuleArtifactAcc, Path, ast.Module], None]
    finish: Callable[[_ModuleArtifactAcc], _ModuleArtifactOut]


@dataclass(frozen=True)
class _ResolvedEdgeParamEvent:
    kind: str
    param: str
    value: OptionalString
    countable: bool


@dataclass(frozen=True)
class _StageCacheSpec(Generic[_StageCacheValue]):
    stage: _ParseModuleStage
    cache_key: Hashable
    build: Callable[[ast.Module, Path], _StageCacheValue]


@dataclass(frozen=True)
class _CacheSemanticContext:
    forest_spec_id: OptionalString = None
    fingerprint_seed_revision: OptionalString = None


_EMPTY_CACHE_SEMANTIC_CONTEXT = _CacheSemanticContext()
_ANALYSIS_INDEX_RESUME_VARIANTS_KEY = "resume_variants"
_ANALYSIS_INDEX_RESUME_MAX_VARIANTS = 4


def _sorted_text(values = None) -> tuple[str, ...]:
    if values is None:
        return ()
    cleaned = {str(value).strip() for value in values if str(value).strip()}
    return tuple(sort_once(cleaned, source = 'src/gabion/analysis/dataflow_audit.py:8091'))


def _canonical_cache_identity(
    *,
    stage: Literal["parse", "index", "projection"],
    cache_context: _CacheSemanticContext,
    config_subset: Mapping[str, JSONValue],
) -> str:
    payload: dict[str, JSONValue] = {
        "stage": stage,
        "forest_spec_id": str(cache_context.forest_spec_id or ""),
        "fingerprint_seed_revision": str(cache_context.fingerprint_seed_revision or ""),
        "config_subset": {str(key): config_subset[key] for key in sort_once(config_subset, source = 'src/gabion/analysis/dataflow_audit.py:8104')},
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"aspf:sha1:{digest}"


def _cache_identity_aliases(identity: str) -> tuple[str, ...]:
    value = str(identity or "")
    if not value:
        return ("",)
    if value.startswith("aspf:sha1:"):
        legacy = value[len("aspf:sha1:") :]
        if legacy:
            return (value, legacy)
        return (value,)
    if len(value) == 40 and all(ch in "0123456789abcdef" for ch in value.lower()):
        return (value, f"aspf:sha1:{value}")
    return (value,)


def _cache_identity_matches(actual: str, expected: str) -> bool:
    actual_aliases = set(_cache_identity_aliases(actual))
    expected_aliases = set(_cache_identity_aliases(expected))
    return bool(actual_aliases & expected_aliases)


def _resume_variant_for_identity(
    variants: Mapping[str, JSONObject],
    expected_identity: str,
):
    direct = variants.get(expected_identity)
    if direct is not None:
        return direct
    expected_aliases = set(_cache_identity_aliases(expected_identity))
    for variant_identity, variant in variants.items():
        check_deadline()
        if expected_aliases & set(_cache_identity_aliases(variant_identity)):
            return variant
    return None


def _parse_stage_cache_key(
    *,
    stage: _ParseModuleStage,
    cache_context: _CacheSemanticContext,
    config_subset: Mapping[str, JSONValue],
    detail: Hashable,
) -> tuple[str, str, str, Hashable]:
    return (
        "parse",
        stage.value,
        _canonical_cache_identity(
            stage="parse",
            cache_context=cache_context,
            config_subset=config_subset,
        ),
        detail,
    )


def _index_stage_cache_identity(
    *,
    cache_context: _CacheSemanticContext,
    config_subset: Mapping[str, JSONValue],
) -> str:
    return _canonical_cache_identity(
        stage="index",
        cache_context=cache_context,
        config_subset=config_subset,
    )


def _projection_stage_cache_identity(
    *,
    cache_context: _CacheSemanticContext,
    config_subset: Mapping[str, JSONValue],
) -> str:
    return _canonical_cache_identity(
        stage="projection",
        cache_context=cache_context,
        config_subset=config_subset,
    )


def _parse_module_source(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _build_module_artifacts(
    paths: list[Path],
    *,
    specs: tuple[_ModuleArtifactSpec[object, object], ...],
    parse_failure_witnesses: list[JSONObject],
    parse_module: Callable[[Path], ast.Module] = _parse_module_source,
) -> tuple[object, ...]:
    check_deadline()
    if not specs:
        return ()
    parse_cache: dict[Path, ParseCacheValue] = {}
    accumulators = [spec.init() for spec in specs]
    for path in paths:
        check_deadline()
        parsed = parse_cache.get(path)
        if parsed is None:
            try:
                parsed = parse_module(path)
            except _PARSE_MODULE_ERROR_TYPES as exc:
                parsed = exc
            parse_cache[path] = parsed
        if type(parsed) is not ast.Module:
            parsed_error = cast(BaseException, parsed)
            for spec in specs:
                check_deadline()
                _record_parse_failure_witness(
                    sink=parse_failure_witnesses,
                    path=path,
                    stage=spec.stage,
                    error=cast(Exception, parsed_error),
                )
            continue
        parsed_module = cast(ast.Module, parsed)
        for idx, spec in enumerate(specs):
            check_deadline()
            spec.fold(accumulators[idx], path, parsed_module)
    return tuple(
        spec.finish(accumulator) for spec, accumulator in zip(specs, accumulators)
    )


def _build_analysis_index(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators=None,
    parse_failure_witnesses: list[JSONObject],
    resume_payload=None,
    on_progress=None,
    accumulate_function_index_for_tree_fn=None,
    forest_spec_id=None,
    fingerprint_seed_revision=None,
    decision_ignore_params=None,
    decision_require_tiers: bool = False,
) -> AnalysisIndex:
    check_deadline()
    if accumulate_function_index_for_tree_fn is None:
        accumulate_function_index_for_tree_fn = _accumulate_function_index_for_tree
    normalized_ignore = _sorted_text(ignore_params)
    normalized_transparent = _sorted_text(transparent_decorators)
    normalized_decision_ignore = _sorted_text(decision_ignore_params)
    cache_context = _CacheSemanticContext(
        forest_spec_id=forest_spec_id,
        fingerprint_seed_revision=fingerprint_seed_revision,
    )
    # dataflow-bundle: decision_require_tiers, external_filter
    index_config_subset: dict[str, JSONValue] = {
        "ignore_params": list(normalized_ignore),
        "strictness": str(strictness),
        "transparent_decorators": list(normalized_transparent),
        "external_filter": external_filter,
        "decision_ignore_params": list(normalized_decision_ignore),
        "decision_require_tiers": decision_require_tiers,
    }
    index_cache_identity = _index_stage_cache_identity(
        cache_context=cache_context,
        config_subset=index_config_subset,
    )
    projection_cache_identity = _projection_stage_cache_identity(
        cache_context=cache_context,
        config_subset={
            "strictness": str(strictness),
            "external_filter": external_filter,
            "decision_require_tiers": decision_require_tiers,
        },
    )
    ordered_paths = _iter_monotonic_paths(
        paths,
        source="_build_analysis_index.paths",
    )
    (
        hydrated_paths,
        by_qual,
        symbol_table,
        class_index,
    ) = _load_analysis_index_resume_payload(
        payload=resume_payload,
        file_paths=ordered_paths,
        expected_index_cache_identity=index_cache_identity,
        expected_projection_cache_identity=projection_cache_identity,
    )
    symbol_table.external_filter = external_filter
    function_index_acc = _FunctionIndexAccumulator(
        by_name=defaultdict(list),
        by_qual={},
    )
    for qual in sort_once(
        by_qual,
        source="_build_analysis_index.resume.by_qual",
    ):
        check_deadline()
        info = by_qual[qual]
        function_index_acc.by_qual[qual] = info
        function_index_acc.by_name[info.name].append(info)
    progress_since_emit = 0
    last_progress_emit_monotonic = None
    profile_stage_ns: dict[str, int] = {
        "analysis_index.parse_module": 0,
        "analysis_index.function_index": 0,
        "analysis_index.symbol_table": 0,
        "analysis_index.class_index": 0,
    }
    profile_counters: Counter[str] = Counter(
        {
            "analysis_index.paths_total": len(ordered_paths),
            "analysis_index.paths_hydrated": len(hydrated_paths),
            "analysis_index.paths_parsed": 0,
            "analysis_index.parse_errors": 0,
        }
    )

    def _index_profile_payload() -> JSONObject:
        return _profiling_v1_payload(stage_ns=profile_stage_ns, counters=profile_counters)

    def _emit_index_progress(*, force: bool = False) -> None:
        nonlocal progress_since_emit
        nonlocal last_progress_emit_monotonic
        progress_callback = on_progress
        if progress_callback is not None:
            progress_since_emit += 1
            now = time.monotonic()
            emit_allowed = (
                force
                or last_progress_emit_monotonic is None
                or (
                    now - last_progress_emit_monotonic
                    >= _PROGRESS_EMIT_MIN_INTERVAL_SECONDS
                )
            )
            if emit_allowed:
                progress_since_emit = 0
                last_progress_emit_monotonic = now
                progress_callback(
                    _serialize_analysis_index_resume_payload(
                        hydrated_paths=hydrated_paths,
                        by_qual=function_index_acc.by_qual,
                        symbol_table=symbol_table,
                        class_index=class_index,
                        index_cache_identity=index_cache_identity,
                        projection_cache_identity=projection_cache_identity,
                        profiling_v1=_index_profile_payload(),
                        previous_payload=resume_payload,
                    )
                )

    try:
        for path in ordered_paths:
            check_deadline()
            if path in hydrated_paths:
                continue
            parse_started_ns = time.monotonic_ns()
            try:
                tree = _parse_module_source(path)
            except _PARSE_MODULE_ERROR_TYPES as exc:
                profile_stage_ns["analysis_index.parse_module"] += (
                    time.monotonic_ns() - parse_started_ns
                )
                profile_counters["analysis_index.parse_errors"] += 1
                _record_parse_failure_witness(
                    sink=parse_failure_witnesses,
                    path=path,
                    stage=_ParseModuleStage.FUNCTION_INDEX,
                    error=exc,
                )
                _record_parse_failure_witness(
                    sink=parse_failure_witnesses,
                    path=path,
                    stage=_ParseModuleStage.SYMBOL_TABLE,
                    error=exc,
                )
                _record_parse_failure_witness(
                    sink=parse_failure_witnesses,
                    path=path,
                    stage=_ParseModuleStage.CLASS_INDEX,
                    error=exc,
                )
                continue
            profile_stage_ns["analysis_index.parse_module"] += (
                time.monotonic_ns() - parse_started_ns
            )
            profile_counters["analysis_index.paths_parsed"] += 1
            function_started_ns = time.monotonic_ns()
            accumulate_function_index_for_tree_fn(
                function_index_acc,
                path,
                tree,
                project_root=project_root,
                ignore_params=ignore_params,
                strictness=strictness,
                transparent_decorators=transparent_decorators,
            )
            profile_stage_ns["analysis_index.function_index"] += (
                time.monotonic_ns() - function_started_ns
            )
            symbol_started_ns = time.monotonic_ns()
            _accumulate_symbol_table_for_tree(
                symbol_table,
                path,
                tree,
                project_root=project_root,
            )
            profile_stage_ns["analysis_index.symbol_table"] += (
                time.monotonic_ns() - symbol_started_ns
            )
            class_started_ns = time.monotonic_ns()
            _accumulate_class_index_for_tree(
                class_index,
                path,
                tree,
                project_root=project_root,
            )
            profile_stage_ns["analysis_index.class_index"] += (
                time.monotonic_ns() - class_started_ns
            )
            hydrated_paths.add(path)
            profile_counters["analysis_index.paths_hydrated"] = len(hydrated_paths)
            _emit_index_progress()
    except TimeoutExceeded:
        _emit_index_progress(force=True)
        raise
    _emit_index_progress(force=True)
    return AnalysisIndex(
        by_name=dict(function_index_acc.by_name),
        by_qual=function_index_acc.by_qual,
        symbol_table=symbol_table,
        class_index=class_index,
        index_cache_identity=index_cache_identity,
        projection_cache_identity=projection_cache_identity,
    )


def _run_indexed_pass(
    paths: list[Path],
    *,
    project_root: OptionalProjectRoot,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators: OptionalDecorators = None,
    parse_failure_witnesses: OptionalParseFailures = None,
    analysis_index: OptionalAnalysisIndex = None,
    spec: _IndexedPassSpec[_IndexedPassResult],
    build_index: Callable[..., AnalysisIndex] = _build_analysis_index,
) -> _IndexedPassResult:
    check_deadline()
    sink = _parse_failure_sink(parse_failure_witnesses)
    index = analysis_index
    if index is None:
        index = build_index(
            paths,
            project_root=project_root,
            ignore_params=ignore_params,
            strictness=strictness,
            external_filter=external_filter,
            transparent_decorators=transparent_decorators,
            parse_failure_witnesses=sink,
        )
    context = _IndexedPassContext(
        paths=paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
        parse_failure_witnesses=sink,
        analysis_index=index,
    )
    return spec.run(context)


def _analysis_index_module_trees(
    analysis_index: AnalysisIndex,
    paths: list[Path],
    *,
    stage: _ParseModuleStage,
    parse_failure_witnesses: list[JSONObject],
):
    check_deadline()
    trees = {}
    for path in paths:
        check_deadline()
        cached_tree = analysis_index.parsed_modules_by_path.get(path)
        if cached_tree is not None:
            trees[path] = cached_tree
            continue
        cached_error = analysis_index.module_parse_errors_by_path.get(path)
        if cached_error is not None:
            _record_parse_failure_witness(
                sink=parse_failure_witnesses,
                path=path,
                stage=stage,
                error=cached_error,
            )
            trees[path] = None
            continue
        try:
            tree = ast.parse(path.read_text())
        except _PARSE_MODULE_ERROR_TYPES as exc:
            analysis_index.module_parse_errors_by_path[path] = exc
            _record_parse_failure_witness(
                sink=parse_failure_witnesses,
                path=path,
                stage=stage,
                error=exc,
            )
            trees[path] = None
            continue
        analysis_index.parsed_modules_by_path[path] = tree
        trees[path] = tree
    return trees


def _analysis_index_stage_cache(
    analysis_index: AnalysisIndex,
    paths: list[Path],
    *,
    spec: _StageCacheSpec[_StageCacheValue],
    parse_failure_witnesses: list[JSONObject],
    module_trees_fn = None,
):
    check_deadline()
    if module_trees_fn is None:
        module_trees_fn = _analysis_index_module_trees
    derivation_runtime = get_global_derivation_cache()
    trees = module_trees_fn(
        analysis_index,
        paths,
        stage=spec.stage,
        parse_failure_witnesses=parse_failure_witnesses,
    )
    scoped_cache_key = (analysis_index.index_cache_identity, spec.cache_key)
    cache = analysis_index.stage_cache_by_key.setdefault(scoped_cache_key, {})
    results = {}
    for path in paths:
        check_deadline()
        tree = trees.get(path)
        if tree is None:
            results[path] = None
            continue
        if path not in cache:
            try:
                dependencies = _path_dependency_payload(path)
            except OSError:
                dependencies = {
                    "path": str(path),
                    "mtime_ns": 0,
                    "size": 0,
                }

            def _compute_stage_value() -> _StageCacheValue:
                return spec.build(tree, path)

            cache[path] = cast(
                object,
                derivation_runtime.derive(
                    op=_ANALYSIS_INDEX_STAGE_CACHE_OP,
                    structural_inputs={
                        "index_cache_identity": analysis_index.index_cache_identity,
                        "projection_cache_identity": analysis_index.projection_cache_identity,
                        "stage": spec.stage.value,
                        "cache_key": spec.cache_key,
                        "path": str(path.resolve()),
                    },
                    dependencies=dependencies,
                    params={"cache_scope": "analysis_index_stage_cache"},
                    compute_fn=_compute_stage_value,
                    source="dataflow_audit._analysis_index_stage_cache",
                ),
            )
        results[path] = cast(_StageCacheValue, cache[path])
    return results


def _analysis_index_transitive_callers(
    analysis_index: AnalysisIndex,
    *,
    project_root,
) -> dict[str, set[str]]:
    check_deadline()
    if analysis_index.transitive_callers is not None:
        return analysis_index.transitive_callers
    callers_by_qual: dict[str, set[str]] = defaultdict(set)
    for edge in _analysis_index_resolved_call_edges(
        analysis_index,
        project_root=project_root,
        require_transparent=False,
    ):
        check_deadline()
        callers_by_qual[edge.callee.qual].add(edge.caller.qual)
    analysis_index.transitive_callers = _collect_transitive_callers(
        callers_by_qual,
        analysis_index.by_qual,
    )
    return analysis_index.transitive_callers


def _analysis_index_resolved_call_edges(
    analysis_index: AnalysisIndex,
    *,
    project_root,
    require_transparent: bool,
) -> tuple[_ResolvedCallEdge, ...]:
    check_deadline()
    if require_transparent:
        cached_edges = analysis_index.resolved_transparent_call_edges
    else:
        cached_edges = analysis_index.resolved_call_edges
    if cached_edges is not None:
        return cached_edges
    edges: list[_ResolvedCallEdge] = []
    for infos in analysis_index.by_name.values():
        check_deadline()
        for info in infos:
            check_deadline()
            for call in info.calls:
                check_deadline()
                if not call.is_test:
                    callee = _resolve_callee(
                        call.callee,
                        info,
                        analysis_index.by_name,
                        analysis_index.by_qual,
                        analysis_index.symbol_table,
                        project_root,
                        analysis_index.class_index,
                    )
                    if callee is not None and (not require_transparent or callee.transparent):
                        edges.append(_ResolvedCallEdge(caller=info, call=call, callee=callee))
    frozen_edges = tuple(edges)
    if require_transparent:
        analysis_index.resolved_transparent_call_edges = frozen_edges
    else:
        analysis_index.resolved_call_edges = frozen_edges
    return frozen_edges


def _analysis_index_resolved_call_edges_by_caller(
    analysis_index: AnalysisIndex,
    *,
    project_root,
    require_transparent: bool,
) -> dict[str, tuple[_ResolvedCallEdge, ...]]:
    check_deadline()
    if require_transparent and analysis_index.resolved_transparent_edges_by_caller is not None:
        return analysis_index.resolved_transparent_edges_by_caller
    grouped: dict[str, list[_ResolvedCallEdge]] = defaultdict(list)
    for edge in _analysis_index_resolved_call_edges(
        analysis_index,
        project_root=project_root,
        require_transparent=require_transparent,
    ):
        check_deadline()
        grouped[edge.caller.qual].append(edge)
    frozen_grouped = {qual: tuple(edges) for qual, edges in grouped.items()}
    if require_transparent:
        analysis_index.resolved_transparent_edges_by_caller = frozen_grouped
    return frozen_grouped


def _reduce_resolved_call_edges(
    analysis_index: AnalysisIndex,
    *,
    project_root,
    require_transparent: bool,
    spec: _ResolvedEdgeReducerSpec[_ResolvedEdgeAcc, _ResolvedEdgeOut],
) -> _ResolvedEdgeOut:
    check_deadline()
    acc = spec.init()
    for edge in _analysis_index_resolved_call_edges(
        analysis_index,
        project_root=project_root,
        require_transparent=require_transparent,
    ):
        check_deadline()
        spec.fold(acc, edge)
    return spec.finish(acc)


def _iter_resolved_edge_param_events(
    edge: _ResolvedCallEdge,
    *,
    strictness: str,
    include_variadics_in_low_star: bool,
) -> Iterator[_ResolvedEdgeParamEvent]:
    check_deadline()
    call = edge.call
    callee = edge.callee
    pos_params = (
        list(callee.positional_params)
        if callee.positional_params
        else list(callee.params)
    )
    kwonly_params = set(callee.kwonly_params or ())
    named_params = set(pos_params) | kwonly_params
    mapped_params: set[str] = set()

    for idx_str in call.pos_map:
        check_deadline()
        idx = int(idx_str)
        if idx < len(pos_params):
            param = pos_params[idx]
            mapped_params.add(param)
            yield _ResolvedEdgeParamEvent(
                kind="non_const",
                param=param,
                value=None,
                countable=True,
            )
        elif callee.vararg is not None:
            mapped_params.add(callee.vararg)
            yield _ResolvedEdgeParamEvent(
                kind="non_const",
                param=callee.vararg,
                value=None,
                countable=False,
            )
    for kw in call.kw_map:
        check_deadline()
        if kw in named_params:
            mapped_params.add(kw)
            yield _ResolvedEdgeParamEvent(
                kind="non_const",
                param=kw,
                value=None,
                countable=True,
            )
        elif callee.kwarg is not None:
            mapped_params.add(callee.kwarg)
            yield _ResolvedEdgeParamEvent(
                kind="non_const",
                param=callee.kwarg,
                value=None,
                countable=False,
            )

    for idx_str, value in call.const_pos.items():
        check_deadline()
        idx = int(idx_str)
        if idx < len(pos_params):
            yield _ResolvedEdgeParamEvent(
                kind="const",
                param=pos_params[idx],
                value=value,
                countable=True,
            )
        elif callee.vararg is not None:
            yield _ResolvedEdgeParamEvent(
                kind="non_const",
                param=callee.vararg,
                value=None,
                countable=False,
            )
    for idx_str in call.non_const_pos:
        check_deadline()
        idx = int(idx_str)
        if idx < len(pos_params):
            yield _ResolvedEdgeParamEvent(
                kind="non_const",
                param=pos_params[idx],
                value=None,
                countable=True,
            )
        elif callee.vararg is not None:
            yield _ResolvedEdgeParamEvent(
                kind="non_const",
                param=callee.vararg,
                value=None,
                countable=False,
            )
    for kw, value in call.const_kw.items():
        check_deadline()
        if kw in named_params:
            yield _ResolvedEdgeParamEvent(
                kind="const",
                param=kw,
                value=value,
                countable=True,
            )
        elif callee.kwarg is not None:
            yield _ResolvedEdgeParamEvent(
                kind="non_const",
                param=callee.kwarg,
                value=None,
                countable=False,
            )
    for kw in call.non_const_kw:
        check_deadline()
        if kw in named_params:
            yield _ResolvedEdgeParamEvent(
                kind="non_const",
                param=kw,
                value=None,
                countable=True,
            )
        elif callee.kwarg is not None:
            yield _ResolvedEdgeParamEvent(
                kind="non_const",
                param=callee.kwarg,
                value=None,
                countable=False,
            )

    if strictness != "low":
        return

    remaining = [p for p in named_params if p not in mapped_params]
    if include_variadics_in_low_star:
        if callee.vararg is not None and callee.vararg not in mapped_params:
            remaining.append(callee.vararg)
        if callee.kwarg is not None and callee.kwarg not in mapped_params:
            remaining.append(callee.kwarg)

    if len(call.star_pos) == 1:
        for param in remaining:
            check_deadline()
            yield _ResolvedEdgeParamEvent(
                kind="non_const",
                param=param,
                value=None,
                countable=param in named_params,
            )
        if not include_variadics_in_low_star and callee.vararg is not None:
            yield _ResolvedEdgeParamEvent(
                kind="non_const",
                param=callee.vararg,
                value=None,
                countable=False,
            )

    if len(call.star_kw) == 1:
        for param in remaining:
            check_deadline()
            yield _ResolvedEdgeParamEvent(
                kind="non_const",
                param=param,
                value=None,
                countable=param in named_params,
            )
        if not include_variadics_in_low_star and callee.kwarg is not None:
            yield _ResolvedEdgeParamEvent(
                kind="non_const",
                param=callee.kwarg,
                value=None,
                countable=False,
            )


def _build_call_graph(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators = None,
    parse_failure_witnesses: list[JSONObject],
    analysis_index = None,
) -> tuple[dict[str, list[FunctionInfo]], dict[str, FunctionInfo], dict[str, set[str]]]:
    check_deadline()
    index = require_not_none(
        analysis_index,
        reason="_build_call_graph requires prebuilt analysis_index",
        strict=True,
    )
    transitive_callers = _analysis_index_transitive_callers(
        index,
        project_root=project_root,
    )
    return index.by_name, index.by_qual, transitive_callers


def _collect_call_ambiguities_indexed(
    context: _IndexedPassContext,
    *,
    resolve_callee_fn = None,
) -> list[CallAmbiguity]:
    ambiguities: list[CallAmbiguity] = []
    resolve_callee = _resolve_callee if resolve_callee_fn is None else resolve_callee_fn

    def _sink(
        caller: FunctionInfo,
        call,
        candidates: list[FunctionInfo],
        phase: str,
        callee_key: str,
    ) -> None:
        ordered = tuple(sort_once(candidates, key=lambda info: info.qual, source = 'src/gabion/analysis/dataflow_audit.py:8837'))
        ambiguities.append(
            CallAmbiguity(
                kind="local_resolution_ambiguous",
                caller=caller,
                call=call,
                callee_key=callee_key,
                candidates=ordered,
                phase=phase,
            )
        )

    for infos in context.analysis_index.by_name.values():
        check_deadline()
        for info in infos:
            check_deadline()
            for call in info.calls:
                check_deadline()
                if call.is_test:
                    continue
                resolve_callee(
                    call.callee,
                    info,
                    context.analysis_index.by_name,
                    context.analysis_index.by_qual,
                    context.analysis_index.symbol_table,
                    context.project_root,
                    context.analysis_index.class_index,
                    call=call,
                    ambiguity_sink=_sink,
                )
    return ambiguities


def _collect_call_ambiguities(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators = None,
    parse_failure_witnesses: list[JSONObject],
    analysis_index = None,
) -> list[CallAmbiguity]:
    check_deadline()
    return _run_indexed_pass(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=analysis_index,
        spec=_IndexedPassSpec(
            pass_id="collect_call_ambiguities",
            run=_collect_call_ambiguities_indexed,
        ),
    )


def _dedupe_call_ambiguities(
    ambiguities: Iterable[CallAmbiguity],
) -> list[CallAmbiguity]:
    check_deadline()
    seen: set[tuple[object, ...]] = set()
    ordered: list[CallAmbiguity] = []
    for entry in ambiguities:
        check_deadline()
        span = entry.call.span if entry.call is not None else None
        candidate_keys = tuple(
            (candidate.path, candidate.qual) for candidate in entry.candidates
        )
        key = (
            entry.kind,
            entry.caller.path,
            entry.caller.qual,
            span,
            entry.callee_key,
            candidate_keys,
        )
        if key in seen:
            continue
        seen.add(key)
        ordered.append(entry)
    return ordered


def _emit_call_ambiguities(
    ambiguities: Iterable[CallAmbiguity],
    *,
    project_root,
    forest: Forest,
) -> list[JSONObject]:
    check_deadline()
    entries: list[JSONObject] = []
    for entry in ambiguities:
        check_deadline()
        call_span = entry.call.span if entry.call is not None else None
        site_path = _normalize_snapshot_path(entry.caller.path, project_root)
        site_payload: JSONObject = {
            "path": site_path,
            "function": entry.caller.qual,
        }
        if call_span is not None:
            site_payload["span"] = list(call_span)
        candidate_targets: list[dict[str, str]] = []
        for candidate in entry.candidates:
            check_deadline()
            candidate_targets.append(
                {
                    "path": _normalize_snapshot_path(candidate.path, project_root),
                    "qual": candidate.qual,
                }
            )
        candidate_targets = evidence_keys.normalize_targets(candidate_targets)
        payload: JSONObject = {
            "kind": entry.kind,
            "site": site_payload,
            "candidates": candidate_targets,
            "candidate_count": len(candidate_targets),
            "phase": entry.phase,
        }
        entries.append(payload)
        if call_span is None:
            never(
                "call ambiguity requires span",
                path=site_path,
                qual=entry.caller.qual,
                kind=entry.kind,
                phase=entry.phase,
            )
        suite_id = forest.add_suite_site(
            entry.caller.path.name,
            entry.caller.qual,
            "call",
            span=call_span,
        )
        ambiguity_key = evidence_keys.make_ambiguity_set_key(
            path=site_path,
            qual=entry.caller.qual,
            span=call_span,
            candidates=candidate_targets,
        )
        ambiguity_key = evidence_keys.normalize_key(ambiguity_key)
        for candidate in entry.candidates:
            check_deadline()
            candidate_id = _call_candidate_target_site(
                forest=forest,
                candidate=candidate,
            )
            _add_interned_alt(
                forest=forest,
                kind="CallCandidate",
                inputs=(suite_id, candidate_id),
                evidence={
                    "kind": entry.kind,
                    "phase": entry.phase,
                    "ambiguity_key": ambiguity_key,
                },
            )
        witness_key = evidence_keys.make_partition_witness_key(
            kind=entry.kind,
            site=ambiguity_key.get("site", {}),
            ambiguity=ambiguity_key,
            support={
                "phase": entry.phase,
                "reason": "multiple local candidates",
            },
            collapse={
                "hint": "add explicit qualifier or disambiguating annotation",
            },
        )
        witness_key = evidence_keys.normalize_key(witness_key)
        witness_identity = evidence_keys.key_identity(witness_key)
        witness_node = forest.add_node(
            "PartitionWitness",
            (witness_identity,),
            meta={"evidence_key": witness_key},
        )
        _add_interned_alt(
            forest=forest,
            kind="PartitionWitness",
            inputs=(suite_id, witness_node),
            evidence={
                "kind": entry.kind,
                "phase": entry.phase,
            },
        )
    return entries


def _lint_lines_from_bundle_evidence(evidence: Iterable[str]) -> list[str]:
    return _ds_lint_lines_from_bundle_evidence(
        evidence,
        check_deadline=check_deadline,
        lint_line=_lint_line,
    )

def _lint_lines_from_type_evidence(evidence: Iterable[str]) -> list[str]:
    return _ds_lint_lines_from_type_evidence(
        evidence,
        check_deadline=check_deadline,
        lint_line=_lint_line,
    )

def _lint_lines_from_call_ambiguities(entries: Iterable[JSONObject]) -> list[str]:
    check_deadline()
    lines: list[str] = []
    for entry in entries:
        check_deadline()
        if type(entry) is not dict:
            continue
        entry_payload = cast(Mapping[str, JSONValue], entry)
        site = entry_payload.get("site", {})
        if type(site) is not dict:
            continue
        site_mapping = cast(Mapping[str, JSONValue], site)
        path = str(site_mapping.get("path", "") or "")
        if not path:
            continue
        lineno, col = _span_line_col(site_mapping.get("span"))
        candidate_count = entry_payload.get("candidate_count")
        try:
            count_value = int(candidate_count) if candidate_count is not None else 0
        except (TypeError, ValueError):
            count_value = 0
        kind = str(entry_payload.get("kind", "") or "ambiguity")
        message = f"{kind} candidates={count_value}"
        lines.append(_lint_line(path, lineno or 1, col or 1, "GABION_AMBIGUITY", message))
    return lines


def _lint_lines_from_unused_arg_smells(smells: Iterable[str]) -> list[str]:
    return _ds_lint_lines_from_unused_arg_smells(
        smells,
        check_deadline=check_deadline,
        lint_line=_lint_line,
    )

def _extract_smell_sample(entry: str):
    return _ds_extract_smell_sample(entry)

def _lint_lines_from_constant_smells(smells: Iterable[str]) -> list[str]:
    return _ds_lint_lines_from_constant_smells(
        smells,
        check_deadline=check_deadline,
        lint_line=_lint_line,
    )

def _parse_exception_path_id(value: str):
    parts = value.split(":", 5)
    if len(parts) != 6:
        return None
    path = parts[0]
    try:
        lineno = int(parts[3])
        col = int(parts[4])
    except ValueError:
        return None
    return path, lineno, col


def _exception_protocol_lint_lines(entries: list[JSONObject]) -> list[str]:
    check_deadline()
    lines: list[str] = []
    for entry in entries:
        check_deadline()
        if entry.get("protocol") != "never":
            continue
        if entry.get("status") == "DEAD":
            continue
        exception_id = str(entry.get("exception_path_id", ""))
        parsed = _parse_exception_path_id(exception_id)
        if not parsed:
            continue
        path, lineno, col = parsed
        exception_name = entry.get("exception_name") or "?"
        status = entry.get("status", "UNKNOWN")
        message = f"never-throw exception {exception_name} (status={status})"
        lines.append(_lint_line(path, lineno, col, "GABION_EXC_NEVER", message))
    return lines


def _never_invariant_lint_lines(entries: list[JSONObject]) -> list[str]:
    check_deadline()
    lines: list[str] = []
    for entry in sort_once(entries, key=_never_sort_key, source = 'src/gabion/analysis/dataflow_audit.py:9123'):
        check_deadline()
        status = entry.get("status", "UNKNOWN")
        if status != "PROVEN_UNREACHABLE":
            span = entry.get("span")
            span_entries = sequence_or_none(span)
            if span_entries is not None and len(span_entries) == 4:
                site = mapping_or_empty(entry.get("site"))
                path = str(site.get("path", "?"))
                reason = entry.get("reason") or ""
                witness_ref = entry.get("witness_ref")
                env = entry.get("environment_ref")
                undecidable = entry.get("undecidable_reason") or ""
                line, col, _, _ = span_entries
                bits: list[str] = [f"status={status}"]
                if reason:
                    bits.append(f"reason={reason}")
                if witness_ref:
                    bits.append(f"witness={witness_ref}")
                if env:
                    bits.append(f"env={json.dumps(env, sort_keys=False)}")
                if status == "OBLIGATION":
                    if undecidable:
                        bits.append(f"why={undecidable}")
                    else:
                        bits.append("why=no witness env available")
                message = f"never() invariant ({'; '.join(bits)})"
                lines.append(
                    _lint_line(
                        path,
                        int(line) + 1,
                        int(col) + 1,
                        "GABION_NEVER_INVARIANT",
                        message,
                    )
                )
    return lines


def _has_bundles(groups_by_path: dict[Path, dict[str, list[set[str]]]]) -> bool:
    check_deadline()
    for groups in groups_by_path.values():
        check_deadline()
        for bundles in groups.values():
            check_deadline()
            if bundles:
                return True
    return False


def _forbid_adhoc_bundle_discovery(reason: str) -> None:
    if os.environ.get("GABION_FORBID_ADHOC_BUNDLES") == "1":
        raise AssertionError(
            f"Ad-hoc bundle discovery invoked while forest-only invariant active: {reason}"
        )


def _collect_bundle_evidence_lines(
    *,
    forest: Forest,
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    bundle_sites_by_path: dict[Path, dict[str, list[list[JSONObject]]]],
) -> list[str]:
    check_deadline()
    if not groups_by_path or not _has_bundles(groups_by_path):
        return []
    file_paths = _iter_monotonic_paths(
        groups_by_path.keys(),
        source="_collect_bundle_evidence_lines.groups_by_path",
    )
    projection = _bundle_projection_from_forest(forest, file_paths=file_paths)
    components = _connected_components(projection.nodes, projection.adj)
    bundle_site_index = _bundle_site_index(groups_by_path, bundle_sites_by_path)
    evidence_lines: list[str] = []
    for comp in components:
        check_deadline()
        evidence = _render_component_callsite_evidence(
            component=comp,
            nodes=projection.nodes,
            bundle_map=projection.bundle_map,
            bundle_counts=projection.bundle_counts,
            adj=projection.adj,
            documented_by_path=projection.documented_by_path,
            declared_global=projection.declared_global,
            bundle_site_index=bundle_site_index,
            root=projection.root,
            path_lookup=projection.path_lookup,
        )
        evidence_lines.extend(evidence)
    return evidence_lines


class _SuiteSpanStatus(StrEnum):
    PRESENT = "present"
    MISSING = "missing"


@dataclass(frozen=True)
class _SuiteSpanOutcome:
    status: _SuiteSpanStatus
    span: tuple[int, int, int, int]


def _suite_span_from_statements_outcome(
    statements: Sequence[ast.stmt],
) -> _SuiteSpanOutcome:
    check_deadline()
    missing_span = (0, 0, 0, 0)
    if not statements:
        return _SuiteSpanOutcome(_SuiteSpanStatus.MISSING, missing_span)
    first_span = _node_span(statements[0])
    if first_span is not None:
        last_span = first_span
        for stmt in statements[1:]:
            check_deadline()
            candidate = _node_span(stmt)
            if candidate is not None:
                last_span = candidate
        return _SuiteSpanOutcome(
            _SuiteSpanStatus.PRESENT,
            (first_span[0], first_span[1], last_span[2], last_span[3]),
        )
    return _SuiteSpanOutcome(_SuiteSpanStatus.MISSING, missing_span)


def _suite_span_from_statements(
    statements: Sequence[ast.stmt],
):
    outcome = _suite_span_from_statements_outcome(statements)
    if outcome.status is _SuiteSpanStatus.PRESENT:
        return outcome.span
    missing_spans: dict[str, tuple[int, int, int, int]] = {}
    return missing_spans.get("span")


def _materialize_statement_suite_contains(
    *,
    forest: Forest,
    path_name: str,
    qual: str,
    statements: Sequence[ast.stmt],
    parent_suite: NodeId,
) -> None:
    check_deadline()

    def _emit_body_suite(
        suite_kind: str,
        body: Sequence[ast.stmt],
    ):
        check_deadline()
        span_outcome = _suite_span_from_statements_outcome(body)
        if span_outcome.status is _SuiteSpanStatus.PRESENT:
            return forest.add_suite_site(
                path_name,
                qual,
                suite_kind,
                span=span_outcome.span,
                parent=parent_suite,
            )
        missing_suite_ids: dict[str, NodeId] = {}
        return missing_suite_ids.get("suite_id")

    for stmt in statements:
        check_deadline()
        stmt_type = type(stmt)
        if stmt_type is ast.If:
            if_stmt = cast(ast.If, stmt)
            if_suite = _emit_body_suite("if_body", if_stmt.body)
            if if_suite is not None:
                _materialize_statement_suite_contains(
                    forest=forest,
                    path_name=path_name,
                    qual=qual,
                    statements=if_stmt.body,
                    parent_suite=if_suite,
                )
            if if_stmt.orelse:
                else_suite = _emit_body_suite("if_else", if_stmt.orelse)
                if else_suite is not None:
                    _materialize_statement_suite_contains(
                        forest=forest,
                        path_name=path_name,
                        qual=qual,
                        statements=if_stmt.orelse,
                        parent_suite=else_suite,
                    )
            continue
        if stmt_type is ast.For:
            for_stmt = cast(ast.For, stmt)
            for_suite = _emit_body_suite("for_body", for_stmt.body)
            if for_suite is not None:
                _materialize_statement_suite_contains(
                    forest=forest,
                    path_name=path_name,
                    qual=qual,
                    statements=for_stmt.body,
                    parent_suite=for_suite,
                )
            if for_stmt.orelse:
                for_else_suite = _emit_body_suite("for_else", for_stmt.orelse)
                if for_else_suite is not None:
                    _materialize_statement_suite_contains(
                        forest=forest,
                        path_name=path_name,
                        qual=qual,
                        statements=for_stmt.orelse,
                        parent_suite=for_else_suite,
                    )
            continue
        if stmt_type is ast.AsyncFor:
            async_for_stmt = cast(ast.AsyncFor, stmt)
            async_for_suite = _emit_body_suite("async_for_body", async_for_stmt.body)
            if async_for_suite is not None:
                _materialize_statement_suite_contains(
                    forest=forest,
                    path_name=path_name,
                    qual=qual,
                    statements=async_for_stmt.body,
                    parent_suite=async_for_suite,
                )
            if async_for_stmt.orelse:
                async_for_else_suite = _emit_body_suite("async_for_else", async_for_stmt.orelse)
                if async_for_else_suite is not None:
                    _materialize_statement_suite_contains(
                        forest=forest,
                        path_name=path_name,
                        qual=qual,
                        statements=async_for_stmt.orelse,
                        parent_suite=async_for_else_suite,
                    )
            continue
        if stmt_type is ast.While:
            while_stmt = cast(ast.While, stmt)
            while_suite = _emit_body_suite("while_body", while_stmt.body)
            if while_suite is not None:
                _materialize_statement_suite_contains(
                    forest=forest,
                    path_name=path_name,
                    qual=qual,
                    statements=while_stmt.body,
                    parent_suite=while_suite,
                )
            if while_stmt.orelse:
                while_else_suite = _emit_body_suite("while_else", while_stmt.orelse)
                if while_else_suite is not None:
                    _materialize_statement_suite_contains(
                        forest=forest,
                        path_name=path_name,
                        qual=qual,
                        statements=while_stmt.orelse,
                        parent_suite=while_else_suite,
                    )
            continue
        if stmt_type is ast.Try:
            try_stmt = cast(ast.Try, stmt)
            try_body_suite = _emit_body_suite("try_body", try_stmt.body)
            if try_body_suite is not None:
                _materialize_statement_suite_contains(
                    forest=forest,
                    path_name=path_name,
                    qual=qual,
                    statements=try_stmt.body,
                    parent_suite=try_body_suite,
                )
            for handler in try_stmt.handlers:
                check_deadline()
                except_suite = _emit_body_suite("except_body", handler.body)
                if except_suite is not None:
                    _materialize_statement_suite_contains(
                        forest=forest,
                        path_name=path_name,
                        qual=qual,
                        statements=handler.body,
                        parent_suite=except_suite,
                    )
            if stmt.orelse:
                try_else_suite = _emit_body_suite("try_else", stmt.orelse)
                if try_else_suite is not None:
                    _materialize_statement_suite_contains(
                        forest=forest,
                        path_name=path_name,
                        qual=qual,
                        statements=stmt.orelse,
                        parent_suite=try_else_suite,
                    )
            if stmt.finalbody:
                try_finally_suite = _emit_body_suite("try_finally", stmt.finalbody)
                if try_finally_suite is not None:
                    _materialize_statement_suite_contains(
                        forest=forest,
                        path_name=path_name,
                        qual=qual,
                        statements=stmt.finalbody,
                        parent_suite=try_finally_suite,
                    )


def _materialize_structured_suite_sites_for_tree(
    *,
    forest: Forest,
    path: Path,
    tree: ast.Module,
    project_root,
) -> None:
    check_deadline()
    parent_annotator = ParentAnnotator()
    parent_annotator.visit(tree)
    parent_map = parent_annotator.parents
    module = _module_name(path, project_root)
    path_name = path.name
    for fn in _collect_functions(tree):
        check_deadline()
        scopes = _enclosing_scopes(fn, parent_map)
        qual_parts = [module] if module else []
        if scopes:
            qual_parts.extend(scopes)
        qual_parts.append(fn.name)
        qual = ".".join(qual_parts)
        function_span = _node_span(fn)
        function_suite = forest.add_suite_site(
            path_name,
            qual,
            "function",
            span=function_span,
        )
        parent_suite = function_suite
        if function_span is not None:
            parent_suite = forest.add_suite_site(
                path_name,
                qual,
                "function_body",
                span=function_span,
                parent=function_suite,
            )
        _materialize_statement_suite_contains(
            forest=forest,
            path_name=path_name,
            qual=qual,
            statements=fn.body,
            parent_suite=parent_suite,
        )


def _materialize_structured_suite_sites(
    *,
    forest: Forest,
    file_paths: list[Path],
    project_root,
    parse_failure_witnesses: list[JSONObject],
    analysis_index = None,
) -> None:
    check_deadline()
    ordered_file_paths = _iter_monotonic_paths(
        file_paths,
        source="_materialize_structured_suite_sites.file_paths",
    )
    if analysis_index is not None:
        trees = _analysis_index_module_trees(
            analysis_index,
            ordered_file_paths,
            stage=_ParseModuleStage.SUITE_CONTAINMENT,
            parse_failure_witnesses=parse_failure_witnesses,
        )
    else:
        trees = {}
        for path in ordered_file_paths:
            check_deadline()
            tree = _parse_module_tree(
                path,
                stage=_ParseModuleStage.SUITE_CONTAINMENT,
                parse_failure_witnesses=parse_failure_witnesses,
            )
            trees[path] = tree
    for path in _iter_monotonic_paths(
        trees,
        source="_materialize_structured_suite_sites.trees",
    ):
        check_deadline()
        tree = trees[path]
        if tree is not None:
            _materialize_structured_suite_sites_for_tree(
                forest=forest,
                path=path,
                tree=tree,
                project_root=project_root,
            )


def _populate_bundle_forest(
    forest: Forest,
    *,
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    file_paths: list[Path],
    project_root = None,
    include_all_sites: bool = True,
    ignore_params = None,
    strictness: str = "high",
    transparent_decorators = None,
    parse_failure_witnesses: list[JSONObject],
    analysis_index = None,
    on_progress = None,
) -> None:
    check_deadline()
    if not groups_by_path:
        return
    ordered_file_paths = _iter_monotonic_paths(
        file_paths,
        source="_populate_bundle_forest.file_paths",
    )
    index = analysis_index
    if include_all_sites and index is None:
        index = _build_analysis_index(
            ordered_file_paths,
            project_root=project_root,
            ignore_params=ignore_params or set(),
            strictness=strictness,
            external_filter=True,
            transparent_decorators=transparent_decorators,
            parse_failure_witnesses=parse_failure_witnesses,
        )
    callsite_inventory_total = 0
    if index is not None:
        callsite_inventory_total = sum(len(info.calls) for info in index.by_qual.values())
    site_materialization_total = 0
    site_materialization_done = 0
    structured_suite_total = 0
    structured_suite_done = 0
    group_paths_total = len(groups_by_path)
    group_paths_done = 0
    config_paths_total = 0
    config_paths_done = 0
    dataclass_quals_total = 0
    dataclass_quals_done = 0
    marker_paths_total = len(ordered_file_paths)
    marker_paths_done = 0
    progress_since_emit = 0
    progress_accepts_payload = None
    last_progress_emit_monotonic = None

    def _forest_progress_snapshot(*, marker: str) -> JSONObject:
        mutable_done = (
            site_materialization_done
            + structured_suite_done
            + group_paths_done
            + config_paths_done
            + dataclass_quals_done
            + marker_paths_done
        )
        mutable_total = (
            site_materialization_total
            + structured_suite_total
            + group_paths_total
            + config_paths_total
            + dataclass_quals_total
            + marker_paths_total
        )
        return {
            "format_version": 1,
            "schema": "gabion/forest_progress_v2",
            "primary_unit": "forest_mutable_steps",
            "primary_done": mutable_done,
            "primary_total": mutable_total,
            "dimensions": {
                "site_materialization": {
                    "done": site_materialization_done,
                    "total": site_materialization_total,
                },
                "structured_suite_materialization": {
                    "done": structured_suite_done,
                    "total": structured_suite_total,
                },
                "group_paths": {
                    "done": group_paths_done,
                    "total": group_paths_total,
                },
                "config_paths": {
                    "done": config_paths_done,
                    "total": config_paths_total,
                },
                "dataclass_quals": {
                    "done": dataclass_quals_done,
                    "total": dataclass_quals_total,
                },
                "marker_paths": {
                    "done": marker_paths_done,
                    "total": marker_paths_total,
                },
            },
            "inventory": {
                "callsites_total": callsite_inventory_total,
                "input_file_paths_total": len(ordered_file_paths),
            },
            "marker": marker,
        }

    def _notify_progress(progress_delta: int, *, marker: str) -> None:
        nonlocal progress_accepts_payload
        if on_progress is not None:
            snapshot = _forest_progress_snapshot(marker=marker)
            normalized_delta = max(int(progress_delta), 0)
            if progress_accepts_payload is True:
                on_progress(snapshot)
            elif progress_accepts_payload is False:
                try:
                    on_progress(normalized_delta)
                except TypeError:
                    on_progress()
            else:
                try:
                    on_progress(snapshot)
                    progress_accepts_payload = True
                except TypeError:
                    progress_accepts_payload = False
                    try:
                        on_progress(normalized_delta)
                    except TypeError:
                        on_progress()

    def _emit_progress(*, force: bool = False, marker: str) -> None:
        nonlocal last_progress_emit_monotonic
        if on_progress is not None:
            now = time.monotonic()
            min_interval_elapsed = (
                last_progress_emit_monotonic is None
                or now - last_progress_emit_monotonic
                >= _PROGRESS_EMIT_MIN_INTERVAL_SECONDS
            )
            if force or min_interval_elapsed:
                last_progress_emit_monotonic = now
                _notify_progress(1, marker=marker)

    _notify_progress(0, marker="start")
    if include_all_sites:
        non_test_quals = [
            qual
            for qual in sort_once(
                index.by_qual,
                source="_populate_bundle_forest.index.by_qual",
            )
            if not _is_test_path(index.by_qual[qual].path)
        ]
        site_materialization_total = len(non_test_quals)
        for qual in non_test_quals:
            check_deadline()
            info = index.by_qual[qual]
            forest.add_site(info.path.name, info.qual)
            site_materialization_done += 1
            _emit_progress(marker="site_materialization")
        non_test_file_paths = [
            path for path in ordered_file_paths if not _is_test_path(path)
        ]
        structured_suite_total = 1
        _materialize_structured_suite_sites(
            forest=forest,
            file_paths=non_test_file_paths,
            project_root=project_root,
            parse_failure_witnesses=parse_failure_witnesses,
            analysis_index=index,
        )
        structured_suite_done = 1
        _emit_progress(force=True, marker="structured_suite_materialization")

    def _add_alt(
        kind: str,
        inputs: Iterable[NodeId],
        evidence = None,
    ) -> None:
        _add_interned_alt(forest=forest, kind=kind, inputs=inputs, evidence=evidence)

    for path in sort_once(
        groups_by_path,
        source="_populate_bundle_forest.groups_by_path",
        key=lambda candidate: str(candidate),
    ):
        check_deadline()
        groups = groups_by_path[path]
        for fn_name in sort_once(groups, source = 'src/gabion/analysis/dataflow_audit.py:9664'):
            check_deadline()
            site_id = forest.add_site(path.name, fn_name)
            for bundle in groups[fn_name]:
                check_deadline()
                paramset_id = forest.add_paramset(bundle)
                _add_alt(
                    "SignatureBundle",
                    (site_id, paramset_id),
                    evidence={"path": path.name, "qual": fn_name},
                )
        group_paths_done += 1
        _emit_progress(marker="group_paths")

    config_bundles_by_path = _collect_config_bundles(
        ordered_file_paths,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=index,
    )
    config_paths_total = len(config_bundles_by_path)
    _emit_progress(force=True, marker="config_paths_discovered")
    for path in _iter_monotonic_paths(
        config_bundles_by_path,
        source="_populate_bundle_forest.config_bundles_by_path",
    ):
        check_deadline()
        bundles = config_bundles_by_path[path]
        for name in sort_once(bundles, source = 'src/gabion/analysis/dataflow_audit.py:9691'):
            check_deadline()
            paramset_id = forest.add_paramset(bundles[name])
            _add_alt(
                "ConfigBundle",
                (paramset_id,),
                evidence={"path": path.name, "name": name},
            )
        config_paths_done += 1
        _emit_progress(marker="config_paths")

    dataclass_registry = _collect_dataclass_registry(
        ordered_file_paths,
        project_root=project_root,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=index,
    )
    dataclass_quals_total = len(dataclass_registry)
    _emit_progress(force=True, marker="dataclass_quals_discovered")
    for qual_name in sort_once(dataclass_registry, source = 'src/gabion/analysis/dataflow_audit.py:9710'):
        check_deadline()
        paramset_id = forest.add_paramset(dataclass_registry[qual_name])
        _add_alt(
            "DataclassBundle",
            (paramset_id,),
            evidence={"qual": qual_name},
        )
        dataclass_quals_done += 1
        _emit_progress(marker="dataclass_quals")

    if index is None or not index.symbol_table.external_filter:
        symbol_table = _build_symbol_table(
            ordered_file_paths,
            project_root,
            external_filter=True,
            parse_failure_witnesses=parse_failure_witnesses,
        )
    else:
        symbol_table = index.symbol_table
    for path in ordered_file_paths:
        check_deadline()
        for bundle in sort_once(_iter_documented_bundles(path), source = 'src/gabion/analysis/dataflow_audit.py:9732'):
            check_deadline()
            paramset_id = forest.add_paramset(bundle)
            _add_alt("MarkerBundle", (paramset_id,), evidence={"path": path.name})
        for bundle in sort_once(
            _iter_dataclass_call_bundles(
                path,
                project_root=project_root,
                symbol_table=symbol_table,
                dataclass_registry=dataclass_registry,
                parse_failure_witnesses=parse_failure_witnesses,
            ), 
        source = 'src/gabion/analysis/dataflow_audit.py:9736'):
            check_deadline()
            paramset_id = forest.add_paramset(bundle)
            _add_alt(
                "DataclassCallBundle",
                (paramset_id,),
                evidence={"path": path.name},
            )
        marker_paths_done += 1
        _emit_progress(marker="marker_paths")
    _emit_progress(force=True, marker="complete")


# dataflow-bundle: decision_lint_lines, broad_type_lint_lines
def _compute_lint_lines(
    *,
    forest: Forest,
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    bundle_sites_by_path: dict[Path, dict[str, list[list[JSONObject]]]],
    type_callsite_evidence: list[str],
    ambiguity_witnesses: list[JSONObject],
    exception_obligations: list[JSONObject],
    never_invariants: list[JSONObject],
    deadline_obligations: list[JSONObject],
    decision_lint_lines: list[str],
    broad_type_lint_lines: list[str],
    constant_smells: list[str],
    unused_arg_smells: list[str],
    project_lint_rows_from_forest_fn = None,
) -> list[str]:
    if project_lint_rows_from_forest_fn is None:
        project_lint_rows_from_forest_fn = _project_lint_rows_from_forest
    bundle_evidence = _collect_bundle_evidence_lines(
        forest=forest,
        groups_by_path=groups_by_path,
        bundle_sites_by_path=bundle_sites_by_path,
    )
    bundle_lint_lines = _lint_lines_from_bundle_evidence(bundle_evidence)
    type_lint_lines = _lint_lines_from_type_evidence(type_callsite_evidence)
    ambiguity_lint_lines = _lint_lines_from_call_ambiguities(ambiguity_witnesses)
    exception_lint_lines = _exception_protocol_lint_lines(exception_obligations)
    never_lint_lines = _never_invariant_lint_lines(never_invariants)
    deadline_lint_lines = _deadline_lint_lines(deadline_obligations)
    constant_lint_lines = _lint_lines_from_constant_smells(constant_smells)
    unused_arg_lint_lines = _lint_lines_from_unused_arg_smells(unused_arg_smells)

    lint_rows: list[dict[str, JSONValue]] = []
    lint_rows.extend(
        _lint_rows_from_lines(bundle_lint_lines, source="bundle_evidence")
    )
    lint_rows.extend(
        _lint_rows_from_lines(type_lint_lines, source="type_evidence")
    )
    lint_rows.extend(
        _lint_rows_from_lines(ambiguity_lint_lines, source="ambiguity_witnesses")
    )
    lint_rows.extend(
        _lint_rows_from_lines(exception_lint_lines, source="exception_obligations")
    )
    lint_rows.extend(
        _lint_rows_from_lines(never_lint_lines, source="never_invariants")
    )
    lint_rows.extend(
        _lint_rows_from_lines(deadline_lint_lines, source="deadline_obligations")
    )
    lint_rows.extend(
        _lint_rows_from_lines(decision_lint_lines, source="decision_surfaces")
    )
    lint_rows.extend(
        _lint_rows_from_lines(broad_type_lint_lines, source="broad_type")
    )
    lint_rows.extend(
        _lint_rows_from_lines(constant_lint_lines, source="constant_smells")
    )
    lint_rows.extend(
        _lint_rows_from_lines(unused_arg_lint_lines, source="unused_arg_smells")
    )

    _materialize_lint_rows(forest=forest, rows=lint_rows)
    projected = project_lint_rows_from_forest_fn(forest=forest)
    if not projected:
        return []

    rendered: list[str] = []
    for row in projected:
        check_deadline()
        path = str(row.get("path", "") or "")
        code = str(row.get("code", "") or "")
        message = str(row.get("message", "") or "")
        if not path or not code:
            continue
        try:
            line = int(row.get("line", 1) or 1)
            col = int(row.get("col", 1) or 1)
        except (TypeError, ValueError):
            continue
        rendered.append(_lint_line(path, line, col, code, message))
    return rendered


def _summarize_handledness_witnesses(
    entries: list[JSONObject],
    *,
    max_entries: int = 10,
) -> list[str]:
    check_deadline()
    if not entries:
        return []
    lines: list[str] = []
    for entry in entries[:max_entries]:
        check_deadline()
        site = entry.get("site", {})
        path = site.get("path", "?")
        function = site.get("function", "?")
        bundle = site.get("bundle", [])
        handler = entry.get("handler_boundary", "?")
        result = entry.get("result", "UNKNOWN")
        reason_code = entry.get("handledness_reason_code", "UNKNOWN_REASON")
        refinement = str(entry.get("type_refinement_opportunity", "") or "")
        suffix = f" reason={reason_code}"
        if refinement:
            suffix += f" refine={refinement}"
        lines.append(
            f"{path}:{function} bundle={bundle} handler={handler} result={result}{suffix}"
        )
    if len(entries) > max_entries:
        lines.append(f"... {len(entries) - max_entries} more")
    return lines


def _summarize_parse_failure_witnesses(
    entries: list[JSONObject],
    *,
    max_entries: int = 25,
) -> list[str]:
    check_deadline()
    if not entries:
        return []
    lines: list[str] = []
    ordered = sort_once(
        entries,
        key=lambda entry: (
            str(entry.get("path", "")),
            str(entry.get("stage", "")),
            str(entry.get("error_type", "")),
            str(entry.get("error", "")),
        ),
    source = 'src/gabion/analysis/dataflow_audit.py:9885')
    for entry in ordered[:max_entries]:
        check_deadline()
        path = str(entry.get("path", "?"))
        stage = str(entry.get("stage", "?"))
        error_type = str(entry.get("error_type", "Error"))
        error = str(entry.get("error", "")).strip()
        if error:
            lines.append(f"{path} stage={stage} {error_type}: {error}")
        else:
            lines.append(f"{path} stage={stage} {error_type}")
    if len(ordered) > max_entries:
        lines.append(f"... {len(ordered) - max_entries} more")
    return lines


def _parse_failure_violation_lines(entries: list[JSONObject]) -> list[str]:
    check_deadline()
    if not entries:
        return []
    lines: list[str] = []
    for entry in sort_once(
        entries,
        key=lambda item: (
            str(item.get("path", "")),
            str(item.get("stage", "")),
            str(item.get("error_type", "")),
            str(item.get("error", "")),
        ),
    source = 'src/gabion/analysis/dataflow_audit.py:9914'):
        check_deadline()
        path = str(entry.get("path", "?"))
        stage = str(entry.get("stage", "?"))
        error_type = str(entry.get("error_type", "Error"))
        error = str(entry.get("error", "")).strip()
        if error:
            lines.append(f"{path} parse_failure stage={stage} {error_type}: {error}")
        else:
            lines.append(f"{path} parse_failure stage={stage} {error_type}")
    return lines


def _summarize_runtime_obligations(
    entries: list[JSONObject],
    *,
    max_entries: int = 50,
) -> list[str]:
    check_deadline()
    if not entries:
        return []
    lines: list[str] = []
    ordered = sort_once(
        entries,
        key=lambda entry: (
            str(entry.get("status", "")),
            str(entry.get("contract", "")),
            str(entry.get("kind", "")),
            str(entry.get("section_id", "")),
            str(entry.get("detail", "")),
        ),
    source = 'src/gabion/analysis/dataflow_audit.py:9944')
    for entry in ordered[:max_entries]:
        check_deadline()
        status = str(entry.get("status", "OBLIGATION"))
        contract = str(entry.get("contract", "runtime_contract"))
        kind = str(entry.get("kind", "unknown"))
        section_id = entry.get("section_id")
        phase = entry.get("phase")
        detail = str(entry.get("detail", "")).strip()
        section_part = ""
        if type(section_id) is str and section_id:
            section_part = f" section={section_id}"
        phase_part = ""
        if type(phase) is str and phase:
            phase_part = f" phase={phase}"
        line = f"{status} {contract} {kind}{section_part}{phase_part}".strip()
        if detail:
            line = f"{line} detail={detail}"
        lines.append(line)
    if len(ordered) > max_entries:
        lines.append(f"... {len(ordered) - max_entries} more")
    return lines


def _runtime_obligation_violation_lines(entries: list[JSONObject]) -> list[str]:
    check_deadline()
    violations: list[str] = []
    for entry in sort_once(
        entries,
        key=lambda item: (
            str(item.get("contract", "")),
            str(item.get("kind", "")),
            str(item.get("section_id", "")),
            str(item.get("phase", "")),
            str(item.get("detail", "")),
        ),
    source = 'src/gabion/analysis/dataflow_audit.py:9980'):
        check_deadline()
        if str(entry.get("status", "")).upper() != "VIOLATION":
            continue
        contract = str(entry.get("contract", "runtime_contract"))
        kind = str(entry.get("kind", "unknown"))
        section_id = entry.get("section_id")
        phase = entry.get("phase")
        detail = str(entry.get("detail", "")).strip()
        section_part = (
            f" section={section_id}"
            if type(section_id) is str and section_id
            else ""
        )
        phase_part = f" phase={phase}" if type(phase) is str and phase else ""
        text = f"{contract} {kind}{section_part}{phase_part}".strip()
        if detail:
            text = f"{text} detail={detail}"
        violations.append(text)
    return violations


def _compute_fingerprint_synth(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    annotations_by_path: dict[Path, dict[str, dict[str, object]]],
    *,
    registry: PrimeRegistry,
    ctor_registry,
    min_occurrences: int,
    version: str,
    existing = None,
):
    check_deadline()
    if min_occurrences < 2 and existing is None:
        return [], None
    fingerprints: list[Fingerprint] = []
    for path, groups in groups_by_path.items():
        check_deadline()
        annots_by_fn = annotations_by_path.get(path, {})
        for fn_name, bundles in groups.items():
            check_deadline()
            fn_annots = annots_by_fn.get(fn_name, {})
            for bundle in bundles:
                check_deadline()
                if any(param not in fn_annots for param in bundle):
                    continue
                types = [fn_annots[param] for param in sort_once(bundle, source = 'src/gabion/analysis/dataflow_audit.py:10035')]
                hint_list = [str(value) for value in types if value is not None]
                if len(hint_list) != len(types):
                    continue
                fingerprint = bundle_fingerprint_dimensional(
                    hint_list,
                    registry,
                    ctor_registry,
                )
                fingerprints.append(fingerprint)
    if not fingerprints and existing is None:
        return [], None
    if existing is not None:
        synth_registry = existing
        payload = synth_registry_payload(
            synth_registry,
            registry,
            min_occurrences=min_occurrences,
        )
    else:
        synth_registry = build_synth_registry(
            fingerprints,
            registry,
            min_occurrences=min_occurrences,
            version=version,
        )
        if not synth_registry.tails:
            return [], None
        payload = synth_registry_payload(
            synth_registry,
            registry,
            min_occurrences=min_occurrences,
        )
    lines: list[str] = [f"synth registry {synth_registry.version}:"]
    for entry in payload.get("entries", []):
        check_deadline()
        tail = entry.get("tail", {})
        base_keys = entry.get("base_keys", [])
        ctor_keys = entry.get("ctor_keys", [])
        remainder = entry.get("remainder", {})
        details = f"base={base_keys}"
        if ctor_keys:
            details += f" ctor={ctor_keys}"
        if remainder.get("base") not in (0, 1) or remainder.get("ctor") not in (0, 1):
            details += f" remainder=({remainder.get('base')},{remainder.get('ctor')})"
        lines.append(
            f"- synth_prime={entry.get('prime')} tail="
            f"{{base={tail.get('base', {}).get('product')}, "
            f"ctor={tail.get('ctor', {}).get('product')}}} "
            f"{details}"
        )
    return lines, payload


def _build_synth_registry_payload(
    synth_registry: "SynthRegistry",
    registry: PrimeRegistry,
    *,
    min_occurrences: int,
) -> JSONObject:
    check_deadline()
    entries: list[JSONObject] = []
    for prime, tail in sort_once(synth_registry.tails.items(), source = 'src/gabion/analysis/dataflow_audit.py:10097'):
        check_deadline()
        base_keys, base_remaining = tail.base.keys_with_remainder(registry)
        ctor_keys, ctor_remaining = tail.ctor.keys_with_remainder(registry)
        ctor_keys = [
            key[len("ctor:") :] if key.startswith("ctor:") else key
            for key in ctor_keys
        ]
        entries.append(
            {
                "prime": prime,
                "tail": {
                    "base": {
                        "product": tail.base.product,
                        "mask": tail.base.mask,
                    },
                    "ctor": {
                        "product": tail.ctor.product,
                        "mask": tail.ctor.mask,
                    },
                    "provenance": {
                        "product": tail.provenance.product,
                        "mask": tail.provenance.mask,
                    },
                    "synth": {
                        "product": tail.synth.product,
                        "mask": tail.synth.mask,
                    },
                },
                "base_keys": sort_once(base_keys, source = 'src/gabion/analysis/dataflow_audit.py:10126'),
                "ctor_keys": sort_once(ctor_keys, source = 'src/gabion/analysis/dataflow_audit.py:10127'),
                "remainder": {
                    "base": base_remaining,
                    "ctor": ctor_remaining,
                },
            }
        )
    return {
        "version": synth_registry.version,
        "min_occurrences": min_occurrences,
        "entries": entries,
    }


class _ReturnAliasCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.returns: list[OptionalAstNode] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_Return(self, node: ast.Return) -> None:
        self.returns.append(node.value)


def _return_aliases(
    fn: ast.AST,
    ignore_params = None,
):
    check_deadline()
    params = _param_names(fn, ignore_params)
    if not params:
        return None
    param_set = set(params)
    collector = _ReturnAliasCollector()
    for stmt in fn.body:
        check_deadline()
        collector.visit(stmt)
    if not collector.returns:
        return None
    alias = None

    def _alias_from_expr(expr = None):
        check_deadline()
        if expr is not None:
            expr_type = type(expr)
            if expr_type is ast.Name:
                name_node = cast(ast.Name, expr)
                if name_node.id in param_set:
                    return [name_node.id]
            if expr_type in {ast.Tuple, ast.List}:
                sequence_node = cast(ast.Tuple | ast.List, expr)
                names: list[str] = []
                for elt in sequence_node.elts:
                    check_deadline()
                    if type(elt) is ast.Name and cast(ast.Name, elt).id in param_set:
                        names.append(cast(ast.Name, elt).id)
                    else:
                        return None
                return names
        return None

    for expr in collector.returns:
        check_deadline()
        candidate = _alias_from_expr(expr)
        if candidate is not None:
            if alias is None:
                alias = candidate
                continue
            if alias != candidate:
                return None
            continue
        return None
    return alias


def _collect_return_aliases(
    funcs: list[FunctionNode],
    parents: dict[ast.AST, ast.AST],
    *,
    ignore_params,
) -> dict[str, tuple[list[str], list[str]]]:
    check_deadline()
    aliases: dict[str, tuple[list[str], list[str]]] = {}
    conflicts: set[str] = set()
    for fn in funcs:
        check_deadline()
        alias = _return_aliases(fn, ignore_params)
        if not alias:
            continue
        params = _param_names(fn, ignore_params)
        class_name = _enclosing_class(fn, parents)
        scopes = _enclosing_scopes(fn, parents)
        keys = {fn.name}
        if class_name:
            keys.add(f"{class_name}.{fn.name}")
        if scopes:
            keys.add(_function_key(scopes, fn.name))
        info = (params, alias)
        for key in keys:
            check_deadline()
            if key in conflicts:
                continue
            if key in aliases:
                aliases.pop(key, None)
                conflicts.add(key)
                continue
            aliases[key] = info
    return aliases


def _const_repr(node: ast.AST):
    node_type = type(node)
    if node_type is ast.Constant:
        return repr(cast(ast.Constant, node).value)
    if node_type is ast.UnaryOp:
        unary_node = cast(ast.UnaryOp, node)
        if type(unary_node.op) in {ast.USub, ast.UAdd} and type(unary_node.operand) is ast.Constant:
            try:
                return ast.unparse(unary_node)
            except _AST_UNPARSE_ERROR_TYPES:
                return None
    if node_type is ast.Attribute:
        attribute_node = cast(ast.Attribute, node)
        if attribute_node.attr.isupper():
            try:
                return ast.unparse(attribute_node)
            except _AST_UNPARSE_ERROR_TYPES:
                return None
        return None
    return None


def _normalize_key_expr(
    node: ast.AST,
    *,
    const_bindings: Mapping[str, ast.AST],
):
    """Normalize deterministic subscript key forms.

    Recognizes literal string/int keys, constant-bound names resolving to
    literals, and literal tuples composed from those forms.
    """
    check_deadline()
    node_type = type(node)
    normalized_key = None
    if node_type is ast.Constant:
        value = cast(ast.Constant, node).value
        value_type = type(value)
        if value_type in {str, int}:
            normalized_key = ("literal", value_type.__name__, value)
    elif node_type is ast.UnaryOp and type(cast(ast.UnaryOp, node).op) in {
        ast.USub,
        ast.UAdd,
    }:
        evaluated_value = None
        try:
            evaluated_value = ast.literal_eval(node)
        except _LITERAL_EVAL_ERROR_TYPES:
            pass
        if type(evaluated_value) is int:
            normalized_key = ("literal", "int", evaluated_value)
    elif node_type is ast.Name:
        bound = const_bindings.get(cast(ast.Name, node).id)
        if bound is not None:
            normalized_key = _normalize_key_expr(bound, const_bindings=const_bindings)
    elif node_type is ast.Tuple:
        tuple_node = cast(ast.Tuple, node)
        items: list[Hashable] = []
        complete = True
        for elt in tuple_node.elts:
            check_deadline()
            normalized_item = _normalize_key_expr(elt, const_bindings=const_bindings)
            if normalized_item is None:
                complete = False
            else:
                items.append(normalized_item)
        if complete:
            normalized_key = ("tuple", tuple(items))
    return normalized_key


def _type_from_const_repr(value: str):
    try:
        literal = ast.literal_eval(value)
    except _LITERAL_EVAL_ERROR_TYPES:
        return None
    if literal is None:
        return "None"
    literal_type = type(literal)
    if literal_type is bool:
        return "bool"
    if literal_type is int:
        return "int"
    if literal_type is float:
        return "float"
    if literal_type is complex:
        return "complex"
    if literal_type is str:
        return "str"
    if literal_type is bytes:
        return "bytes"
    if literal_type is list:
        return "list"
    if literal_type is tuple:
        return "tuple"
    if literal_type is set:
        return "set"
    if literal_type is dict:
        return "dict"
    return None


def _is_test_path(path: Path) -> bool:
    if "tests" in path.parts:
        return True
    return path.name.startswith("test_")


def _analyze_function(
    fn: FunctionNode,
    parents: dict[ast.AST, ast.AST],
    *,
    is_test: bool,
    ignore_params: OptionalIgnoredParams = None,
    strictness: str = "high",
    class_name: OptionalClassName = None,
    return_aliases: OptionalReturnAliasMap = None,
) -> tuple[dict[str, ParamUse], list[CallArgs]]:
    params = _param_names(fn, ignore_params)
    use_map = {p: ParamUse(set(), False, {p}) for p in params}
    alias_to_param: dict[str, str] = {p: p for p in params}
    call_args: list[CallArgs] = []

    visitor = UseVisitor(
        parents=parents,
        use_map=use_map,
        call_args=call_args,
        alias_to_param=alias_to_param,
        is_test=is_test,
        strictness=strictness,
        const_repr=_const_repr,
        callee_name=lambda call: _normalize_callee(_callee_name(call), class_name),
        call_args_factory=CallArgs,
        call_context=_call_context,
        return_aliases=return_aliases,
        normalize_key_expr=_normalize_key_expr,
    )
    visitor.visit(fn)
    return use_map, call_args


def _unused_params(use_map: dict[str, ParamUse]) -> tuple[set[str], set[str]]:
    check_deadline()
    unused: set[str] = set()
    unknown_key_carriers: set[str] = set()
    for name, info in use_map.items():
        check_deadline()
        if info.non_forward:
            continue
        if info.direct_forward:
            continue
        if info.unknown_key_carrier:
            unknown_key_carriers.add(name)
            continue
        unused.add(name)
    return unused, unknown_key_carriers


def _group_by_signature(use_map: dict[str, ParamUse]) -> list[set[str]]:
    check_deadline()
    sig_map: dict[tuple[tuple[str, str], ...], list[str]] = defaultdict(list)
    for name, info in use_map.items():
        check_deadline()
        if info.non_forward:
            continue
        sig = tuple(sort_once(info.direct_forward, source = 'src/gabion/analysis/dataflow_audit.py:10391'))
        # Empty forwarding signatures are usually just unused params; treating them as
        # bundles creates noisy Tier-3 violations and unstable fingerprint baselines.
        if not sig:
            continue
        sig_map[sig].append(name)
    groups = [set(names) for names in sig_map.values() if len(names) > 1]
    return groups


def _union_groups(groups: list[set[str]]) -> list[set[str]]:
    check_deadline()
    changed = True
    while changed:
        check_deadline()
        changed = False
        out = []
        while groups:
            check_deadline()
            base = groups.pop()
            merged = True
            while merged:
                check_deadline()
                merged = False
                for i, other in enumerate(groups):
                    check_deadline()
                    if base & other:
                        base |= other
                        groups.pop(i)
                        merged = True
                        changed = True
                        break
            out.append(base)
        groups = out
    return groups


def _propagate_groups(
    call_args: list[CallArgs],
    callee_groups: dict[str, list[set[str]]],
    callee_param_orders: dict[str, list[str]],
    strictness: str,
    opaque_callees = None,
) -> list[set[str]]:
    check_deadline()
    groups: list[set[str]] = []
    for call in call_args:
        check_deadline()
        if opaque_callees and call.callee in opaque_callees:
            continue
        if call.callee not in callee_groups:
            continue
        callee_params = callee_param_orders[call.callee]
        # Build mapping from callee param to caller param.
        callee_to_caller: dict[str, str] = {}
        for idx, pname in enumerate(callee_params):
            check_deadline()
            key = str(idx)
            if key in call.pos_map:
                callee_to_caller[pname] = call.pos_map[key]
        for kw, caller_name in call.kw_map.items():
            check_deadline()
            callee_to_caller[kw] = caller_name
        if strictness == "low":
            mapped = set(callee_to_caller.keys())
            remaining = [p for p in callee_params if p not in mapped]
            if len(call.star_pos) == 1:
                _, star_param = call.star_pos[0]
                for param in remaining:
                    check_deadline()
                    callee_to_caller.setdefault(param, star_param)
            if len(call.star_kw) == 1:
                star_param = call.star_kw[0]
                for param in remaining:
                    check_deadline()
                    callee_to_caller.setdefault(param, star_param)
        for group in callee_groups[call.callee]:
            check_deadline()
            mapped = {callee_to_caller.get(p) for p in group}
            mapped.discard(None)
            if len(mapped) > 1:
                groups.append(set(mapped))
    return groups


def _callsite_evidence_for_bundle(
    calls: list[CallArgs],
    bundle: set[str],
    *,
    limit: int = 12,
) -> list[JSONObject]:
    """Collect callsite evidence for where bundle params are forwarded.

    A bundle can be induced either by co-forwarding in a single callsite or by
    repeated forwarding to identical callee/slot pairs across distinct callsites.
    """
    check_deadline()
    out: list[JSONObject] = []
    seen: set[tuple[tuple[int, int, int, int], str, tuple[str, ...], tuple[str, ...]]] = set()
    for call in calls:
        check_deadline()
        if call.span is not None:
            params_in_call: list[str] = []
            slots: list[str] = []
            for idx_str, param in call.pos_map.items():
                check_deadline()
                if param in bundle:
                    params_in_call.append(param)
                    slots.append(f"arg[{idx_str}]")
            for name, param in call.kw_map.items():
                check_deadline()
                if param in bundle:
                    params_in_call.append(param)
                    slots.append(f"kw[{name}]")
            for idx, param in call.star_pos:
                check_deadline()
                if param in bundle:
                    params_in_call.append(param)
                    slots.append(f"arg[{idx}]*")
            for param in call.star_kw:
                check_deadline()
                if param in bundle:
                    params_in_call.append(param)
                    slots.append("kw[**]")
            distinct = tuple(
                sort_once(
                    set(params_in_call),
                    source="src/gabion/analysis/dataflow_audit.py:10516",
                )
            )
            if distinct:
                slot_list = tuple(
                    sort_once(
                        set(slots),
                        source="src/gabion/analysis/dataflow_audit.py:10519",
                    )
                )
                key = (call.span, call.callee, distinct, slot_list)
                if key not in seen:
                    seen.add(key)
                    out.append(
                        {
                            "callee": call.callee,
                            "span": list(call.span),
                            "params": list(distinct),
                            "slots": list(slot_list),
                            "callable_kind": call.callable_kind,
                            "callable_source": call.callable_source,
                        }
                    )
    out = sort_once(
        out,
        source="_ranked_callargs_evidence.out",
        # Non-lexical tuple key: arity desc, span, callee, then params.
        key=lambda entry: (
            -len(entry.get("params") or []),
            tuple(entry.get("span") or []),
            str(entry.get("callee") or ""),
            tuple(entry.get("params") or []),
        ),
    )
    return out[:limit]


def _analyze_file_internal(
    path: Path,
    recursive: bool = True,
    *,
    config = None,
    resume_state = None,
    on_progress = None,
    on_profile = None,
    analyze_function_fn = None,
) -> tuple[
    dict[str, list[set[str]]],
    dict[str, dict[str, tuple[int, int, int, int]]],
    dict[str, list[list[JSONObject]]],
]:
    check_deadline()
    if analyze_function_fn is None:
        analyze_function_fn = _analyze_function
    if config is None:
        config = AuditConfig()
    profile_stage_ns: dict[str, int] = {
        "file_scan.read_parse": 0,
        "file_scan.parent_annotation": 0,
        "file_scan.collect_functions": 0,
        "file_scan.function_scan": 0,
        "file_scan.resolve_local_calls": 0,
        "file_scan.resolve_local_methods": 0,
        "file_scan.grouping": 0,
        "file_scan.propagation": 0,
        "file_scan.bundle_sites": 0,
    }
    profile_counters: Counter[str] = Counter()
    parse_started_ns = time.monotonic_ns()
    tree = ast.parse(path.read_text())
    profile_stage_ns["file_scan.read_parse"] += time.monotonic_ns() - parse_started_ns
    parent_started_ns = time.monotonic_ns()
    parent = ParentAnnotator()
    parent.visit(tree)
    profile_stage_ns["file_scan.parent_annotation"] += (
        time.monotonic_ns() - parent_started_ns
    )
    parents = parent.parents
    is_test = _is_test_path(path)

    collect_started_ns = time.monotonic_ns()
    funcs = _collect_functions(tree)
    profile_stage_ns["file_scan.collect_functions"] += (
        time.monotonic_ns() - collect_started_ns
    )
    profile_counters["file_scan.functions_total"] = len(funcs)
    fn_keys_in_file: set[str] = set()
    for function_node in funcs:
        check_deadline()
        scopes = _enclosing_scopes(function_node, parents)
        fn_keys_in_file.add(_function_key(scopes, function_node.name))
    return_aliases = _collect_return_aliases(
        funcs, parents, ignore_params=config.ignore_params
    )
    (
        fn_use,
        fn_calls,
        fn_param_orders,
        fn_param_spans,
        fn_names,
        fn_lexical_scopes,
        fn_class_names,
        opaque_callees,
    ) = _load_file_scan_resume_state(
        payload=resume_state,
        valid_fn_keys=fn_keys_in_file,
    )
    scanned_since_emit = 0
    last_scan_progress_emit_monotonic = None

    def _emit_scan_progress(*, force: bool = False) -> bool:
        nonlocal last_scan_progress_emit_monotonic
        if on_progress is None:
            return False
        now = time.monotonic()
        if (
            not force
            and last_scan_progress_emit_monotonic is not None
            and now - last_scan_progress_emit_monotonic < _PROGRESS_EMIT_MIN_INTERVAL_SECONDS
        ):
            return False
        progress_payload = _serialize_file_scan_resume_state(
            fn_use=fn_use,
            fn_calls=fn_calls,
            fn_param_orders=fn_param_orders,
            fn_param_spans=fn_param_spans,
            fn_names=fn_names,
            fn_lexical_scopes=fn_lexical_scopes,
            fn_class_names=fn_class_names,
            opaque_callees=opaque_callees,
        )
        progress_payload["profiling_v1"] = _profiling_v1_payload(
            stage_ns=profile_stage_ns,
            counters=profile_counters,
        )
        on_progress(progress_payload)
        last_scan_progress_emit_monotonic = now
        return True

    def _emit_file_profile() -> None:
        if on_profile is not None:
            on_profile(
                _profiling_v1_payload(
                    stage_ns=profile_stage_ns,
                    counters=profile_counters,
                )
            )

    try:
        scan_started_ns = time.monotonic_ns()
        for f in funcs:
            check_deadline()
            class_name = _enclosing_class(f, parents)
            scopes = _enclosing_scopes(f, parents)
            lexical_scopes = _enclosing_function_scopes(f, parents)
            fn_key = _function_key(scopes, f.name)
            if (
                fn_key in fn_use
                and fn_key in fn_calls
                and fn_key in fn_param_orders
                and fn_key in fn_param_spans
                and fn_key in fn_names
                and fn_key in fn_lexical_scopes
                and fn_key in fn_class_names
            ):
                continue
            if not _decorators_transparent(f, config.transparent_decorators):
                opaque_callees.add(fn_key)
            use_map, call_args = analyze_function_fn(
                f,
                parents,
                is_test=is_test,
                ignore_params=config.ignore_params,
                strictness=config.strictness,
                class_name=class_name,
                return_aliases=return_aliases,
            )
            fn_use[fn_key] = use_map
            fn_calls[fn_key] = call_args
            fn_param_orders[fn_key] = _param_names(f, config.ignore_params)
            fn_param_spans[fn_key] = _param_spans(f, config.ignore_params)
            fn_names[fn_key] = f.name
            fn_lexical_scopes[fn_key] = tuple(lexical_scopes)
            fn_class_names[fn_key] = class_name
            scanned_since_emit += 1
            if (
                scanned_since_emit >= _FILE_SCAN_PROGRESS_EMIT_INTERVAL
                and _emit_scan_progress()
            ):
                scanned_since_emit = 0
        profile_stage_ns["file_scan.function_scan"] += time.monotonic_ns() - scan_started_ns
    except TimeoutExceeded:
        _emit_scan_progress(force=True)
        _emit_file_profile()
        raise
    if scanned_since_emit > 0:
        _emit_scan_progress(force=True)

    local_by_name: dict[str, list[str]] = defaultdict(list)
    for key, name in fn_names.items():
        check_deadline()
        local_by_name[name].append(key)

    def _resolve_local_callee(callee: str, caller_key: str):
        check_deadline()
        if "." in callee:
            return None
        candidates = local_by_name.get(callee, [])
        if not candidates:
            return None
        effective_scope = list(fn_lexical_scopes.get(caller_key, ())) + [fn_names[caller_key]]
        while True:
            check_deadline()
            scoped = [
                key
                for key in candidates
                if fn_lexical_scopes.get(key, ()) == tuple(effective_scope)
                and not (fn_class_names.get(key) and not fn_lexical_scopes.get(key))
            ]
            if len(scoped) == 1:
                return scoped[0]
            if len(scoped) > 1:
                return None
            if not effective_scope:
                break
            effective_scope = effective_scope[:-1]
        return None

    local_resolve_started_ns = time.monotonic_ns()
    for caller_key, calls in list(fn_calls.items()):
        check_deadline()
        resolved_calls: list[CallArgs] = []
        for call in calls:
            check_deadline()
            resolved = _resolve_local_callee(call.callee, caller_key)
            if resolved:
                resolved_calls.append(replace(call, callee=resolved))
            else:
                resolved_calls.append(call)
        fn_calls[caller_key] = resolved_calls
    profile_stage_ns["file_scan.resolve_local_calls"] += (
        time.monotonic_ns() - local_resolve_started_ns
    )

    class_bases = _collect_local_class_bases(tree, parents)
    profile_counters["file_scan.class_bases_count"] = len(class_bases)
    if class_bases:
        method_resolve_started_ns = time.monotonic_ns()
        local_functions = set(fn_use.keys())

        def _resolve_local_method(callee: str):
            class_part, method = callee.rsplit(".", 1)
            return _resolve_local_method_in_hierarchy(
                class_part,
                method,
                class_bases=class_bases,
                local_functions=local_functions,
                seen=set(),
            )

        for caller_key, calls in list(fn_calls.items()):
            check_deadline()
            resolved_calls = []
            for call in calls:
                check_deadline()
                if "." in call.callee:
                    resolved = _resolve_local_method(call.callee)
                    if resolved and resolved != call.callee:
                        resolved_calls.append(replace(call, callee=resolved))
                        continue
                resolved_calls.append(call)
            fn_calls[caller_key] = resolved_calls
        profile_stage_ns["file_scan.resolve_local_methods"] += (
            time.monotonic_ns() - method_resolve_started_ns
        )

    grouping_started_ns = time.monotonic_ns()
    groups_by_fn = {fn: _group_by_signature(use_map) for fn, use_map in fn_use.items()}
    profile_stage_ns["file_scan.grouping"] += time.monotonic_ns() - grouping_started_ns
    profile_counters["file_scan.functions_scanned"] = len(fn_use)

    if not recursive:
        bundle_started_ns = time.monotonic_ns()
        bundle_sites_by_fn: dict[str, list[list[JSONObject]]] = {}
        for fn_key, bundles in groups_by_fn.items():
            check_deadline()
            calls = fn_calls.get(fn_key, [])
            bundle_sites_by_fn[fn_key] = [
                _callsite_evidence_for_bundle(calls, bundle) for bundle in bundles
            ]
        profile_stage_ns["file_scan.bundle_sites"] += time.monotonic_ns() - bundle_started_ns
        _emit_file_profile()
        return groups_by_fn, fn_param_spans, bundle_sites_by_fn

    propagation_started_ns = time.monotonic_ns()
    changed = True
    while changed:
        check_deadline()
        changed = False
        for fn in fn_use:
            check_deadline()
            propagated = _propagate_groups(
                fn_calls[fn],
                groups_by_fn,
                fn_param_orders,
                config.strictness,
                opaque_callees,
            )
            if not propagated:
                continue
            combined = _union_groups(groups_by_fn.get(fn, []) + propagated)
            if combined != groups_by_fn.get(fn, []):
                groups_by_fn[fn] = combined
                changed = True
    profile_stage_ns["file_scan.propagation"] += (
        time.monotonic_ns() - propagation_started_ns
    )
    bundle_started_ns = time.monotonic_ns()
    bundle_sites_by_fn: dict[str, list[list[JSONObject]]] = {}
    for fn_key, bundles in groups_by_fn.items():
        check_deadline()
        calls = fn_calls.get(fn_key, [])
        bundle_sites_by_fn[fn_key] = [
            _callsite_evidence_for_bundle(calls, bundle) for bundle in bundles
        ]
    profile_stage_ns["file_scan.bundle_sites"] += time.monotonic_ns() - bundle_started_ns
    _emit_file_profile()
    return groups_by_fn, fn_param_spans, bundle_sites_by_fn


def analyze_file(
    path: Path,
    recursive: bool = True,
    *,
    config = None,
) -> tuple[dict[str, list[set[str]]], dict[str, dict[str, tuple[int, int, int, int]]]]:
    groups, spans, _ = _analyze_file_internal(path, recursive=recursive, config=config)
    return groups, spans


def _callee_key(name: str) -> str:
    if not name:
        return name
    return name.split(".")[-1]


def _is_broad_type(annot) -> bool:
    if annot is None:
        return True
    base = annot.replace("typing.", "")
    return base in {"Any", "object"}


def _normalize_type_name(value: str) -> str:
    value = value.strip()
    if value.startswith("typing."):
        value = value[len("typing.") :]
    if value.startswith("builtins."):
        value = value[len("builtins.") :]
    return value


_BROAD_SCALAR_TYPES = {
    "str",
    "int",
    "float",
    "bool",
    "bytes",
    "bytearray",
    "complex",
}


def _is_node_id_type(value: str) -> bool:
    return value == "NodeId" or value.endswith(".NodeId")


def _is_literal_type(value: str) -> bool:
    return value.startswith("Literal[")


def _is_broad_internal_type(annot) -> bool:
    if annot is None:
        return False
    normalized = annot.replace("typing.", "")
    expanded = {_normalize_type_name(t) for t in _expand_type_hint(normalized)}
    non_none = {t for t in expanded if t not in _NONE_TYPES}
    if not non_none:
        return False
    if all(_is_node_id_type(t) for t in non_none):
        return False
    if any(_is_literal_type(t) for t in non_none):
        return True
    if "Any" in non_none or "object" in non_none:
        return True
    if _BROAD_SCALAR_TYPES & non_none:
        return True
    return False


_NONE_TYPES = {"None", "NoneType", "type(None)"}


def _split_top_level(value: str, sep: str) -> list[str]:
    check_deadline()
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in value:
        check_deadline()
        if ch in "[({":
            depth += 1
        elif ch in "])}":
            depth = max(depth - 1, 0)
        if ch == sep and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _expand_type_hint(hint: str) -> set[str]:
    hint = hint.strip()
    if not hint:
        return set()
    if hint.startswith("Optional[") and hint.endswith("]"):
        inner = hint[len("Optional[") : -1]
        return {_strip_type(t) for t in _split_top_level(inner, ",")} | {"None"}
    if hint.startswith("Union[") and hint.endswith("]"):
        inner = hint[len("Union[") : -1]
        return {_strip_type(t) for t in _split_top_level(inner, ",")}
    if "|" in hint:
        return {_strip_type(t) for t in _split_top_level(hint, "|")}
    return {hint}


def _strip_type(value: str) -> str:
    return value.strip()


def _combine_type_hints(types: set[str]) -> tuple[str, bool]:
    check_deadline()
    normalized_sets = []
    for hint in types:
        check_deadline()
        expanded = _expand_type_hint(hint)
        normalized_sets.append(
            tuple(
                sort_once(
                    (t for t in expanded if t not in _NONE_TYPES),
                    source="src/gabion/analysis/dataflow_audit.py:10962",
                )
            )
        )
    unique_normalized = {norm for norm in normalized_sets if norm}
    expanded: set[str] = set()
    for hint in types:
        check_deadline()
        expanded.update(_expand_type_hint(hint))
    none_types = {t for t in expanded if t in _NONE_TYPES}
    expanded -= none_types
    if not expanded:
        return "Any", bool(types)
    sorted_types = sort_once(expanded, source = 'src/gabion/analysis/dataflow_audit.py:10973')
    if len(sorted_types) == 1:
        base = sorted_types[0]
        if none_types:
            conflicted = len(unique_normalized) > 1
            return f"Optional[{base}]", conflicted
        return base, len(unique_normalized) > 1
    union = f"Union[{', '.join(sorted_types)}]"
    if none_types:
        return f"Optional[{union}]", len(unique_normalized) > 1
    return union, len(unique_normalized) > 1


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


def _module_name(path: Path, project_root = None) -> str:
    rel = path.with_suffix("")
    if project_root is not None:
        try:
            rel = rel.relative_to(project_root)
        except ValueError:
            pass
    parts = list(rel.parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    return ".".join(parts)


def _string_list(node: ast.AST):
    check_deadline()
    node_type = type(node)
    if node_type is ast.List or node_type is ast.Tuple:
        container = cast(ast.List | ast.Tuple, node)
        values: list[str] = []
        for elt in container.elts:
            check_deadline()
            if type(elt) is ast.Constant and type(cast(ast.Constant, elt).value) is str:
                values.append(cast(str, cast(ast.Constant, elt).value))
            else:
                return None
        return values
    return None


def _base_identifier(node: ast.AST):
    check_deadline()
    node_type = type(node)
    if node_type is ast.Name:
        return cast(ast.Name, node).id
    if node_type is ast.Attribute:
        try:
            return ast.unparse(node)
        except _AST_UNPARSE_ERROR_TYPES:
            return None
    if node_type is ast.Subscript:
        return _base_identifier(cast(ast.Subscript, node).value)
    if node_type is ast.Call:
        return _base_identifier(cast(ast.Call, node).func)
    return None


def _collect_module_exports(
    tree: ast.AST,
    *,
    module_name: str,
    import_map: dict[str, str],
) -> tuple[set[str], dict[str, str]]:
    check_deadline()
    explicit_all: list[str] = []
    has_explicit_all = False
    for stmt in getattr(tree, "body", []):
        check_deadline()
        stmt_type = type(stmt)
        if stmt_type is ast.Assign:
            assign_stmt = cast(ast.Assign, stmt)
            targets = assign_stmt.targets
            if any(type(target) is ast.Name and cast(ast.Name, target).id == "__all__" for target in targets):
                values = _string_list(assign_stmt.value)
                if values is not None:
                    explicit_all = list(values)
                    has_explicit_all = True
        elif stmt_type is ast.AnnAssign:
            ann_assign = cast(ast.AnnAssign, stmt)
            target = ann_assign.target
            if type(target) is ast.Name and cast(ast.Name, target).id == "__all__":
                values = _string_list(ann_assign.value) if ann_assign.value is not None else None
                if values is not None:
                    explicit_all = list(values)
                    has_explicit_all = True
        elif stmt_type is ast.AugAssign:
            aug_assign = cast(ast.AugAssign, stmt)
            target = aug_assign.target
            if (
                type(target) is ast.Name
                and cast(ast.Name, target).id == "__all__"
                and type(aug_assign.op) is ast.Add
            ):
                values = _string_list(aug_assign.value)
                if values is not None:
                    if not has_explicit_all:
                        has_explicit_all = True
                        explicit_all = []
                    explicit_all.extend(values)

    local_defs: set[str] = set()
    for stmt in getattr(tree, "body", []):
        check_deadline()
        stmt_type = type(stmt)
        if stmt_type in {ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef}:
            stmt_name = str(getattr(stmt, "name", ""))
            if stmt_name and not stmt_name.startswith("_"):
                local_defs.add(stmt_name)
        elif stmt_type is ast.Assign:
            assign_stmt = cast(ast.Assign, stmt)
            for target in assign_stmt.targets:
                check_deadline()
                local_defs.update(
                    name
                    for name in _target_names(target)
                    if not name.startswith("_")
                )
        elif stmt_type is ast.AnnAssign:
            ann_assign = cast(ast.AnnAssign, stmt)
            local_defs.update(
                name
                for name in _target_names(ann_assign.target)
                if not name.startswith("_")
            )

    if has_explicit_all:
        export_names = set(explicit_all)
    else:
        export_names = set(local_defs) | {
            name for name in import_map.keys() if not name.startswith("_")
        }
        export_names = {name for name in export_names if not name.startswith("_")}

    export_map: dict[str, str] = {}
    for name in export_names:
        check_deadline()
        if name in import_map:
            export_map[name] = import_map[name]
        elif name in local_defs:
            export_map[name] = f"{module_name}.{name}" if module_name else name
    return export_names, export_map


def _accumulate_symbol_table_for_tree(
    table: SymbolTable,
    path: Path,
    tree: ast.Module,
    *,
    project_root,
) -> None:
    check_deadline()
    module = _module_name(path, project_root)
    table.internal_roots.add(module.split(".")[0])
    visitor = ImportVisitor(module, table)
    visitor.visit(tree)
    import_map = {
        local: fqn for (mod, local), fqn in table.imports.items() if mod == module
    }
    exports, export_map = _collect_module_exports(
        tree,
        module_name=module,
        import_map=import_map,
    )
    table.module_exports[module] = exports
    table.module_export_map[module] = export_map


def _symbol_table_module_artifact_spec(
    *,
    project_root,
    external_filter: bool,
) -> _ModuleArtifactSpec[SymbolTable, SymbolTable]:
    return _ModuleArtifactSpec[SymbolTable, SymbolTable](
        artifact_id="symbol_table",
        stage=_ParseModuleStage.SYMBOL_TABLE,
        init=lambda: SymbolTable(external_filter=external_filter),
        fold=lambda table, path, tree: _accumulate_symbol_table_for_tree(
            table,
            path,
            tree,
            project_root=project_root,
        ),
        finish=lambda table: table,
    )


def _build_symbol_table(
    paths: list[Path],
    project_root,
    *,
    external_filter: bool,
    parse_failure_witnesses: list[JSONObject],
) -> SymbolTable:
    check_deadline()
    raw_table, = _build_module_artifacts(
        paths,
        specs=(
            cast(
                _ModuleArtifactSpec[object, object],
                _symbol_table_module_artifact_spec(
                    project_root=project_root,
                    external_filter=external_filter,
                ),
            ),
        ),
        parse_failure_witnesses=parse_failure_witnesses,
    )
    return cast(SymbolTable, raw_table)


def _accumulate_class_index_for_tree(
    class_index: dict[str, ClassInfo],
    path: Path,
    tree: ast.Module,
    *,
    project_root,
) -> None:
    check_deadline()
    parents = ParentAnnotator()
    parents.visit(tree)
    module = _module_name(path, project_root)
    for node in ast.walk(tree):
        check_deadline()
        if type(node) is not ast.ClassDef:
            continue
        class_node = cast(ast.ClassDef, node)
        scopes = _enclosing_class_scopes(class_node, parents.parents)
        qual_parts = [module] if module else []
        qual_parts.extend(scopes)
        qual_parts.append(class_node.name)
        qual = ".".join(qual_parts)
        bases: list[str] = []
        for base in class_node.bases:
            check_deadline()
            base_name = _base_identifier(base)
            if base_name:
                bases.append(base_name)
        methods: set[str] = set()
        for stmt in class_node.body:
            check_deadline()
            stmt_type = type(stmt)
            if stmt_type in {ast.FunctionDef, ast.AsyncFunctionDef}:
                methods.add(cast(ast.FunctionDef | ast.AsyncFunctionDef, stmt).name)
        class_index[qual] = ClassInfo(
            qual=qual,
            module=module,
            bases=bases,
            methods=methods,
        )


def _class_index_module_artifact_spec(
    *,
    project_root,
) -> _ModuleArtifactSpec[dict[str, ClassInfo], dict[str, ClassInfo]]:
    return _ModuleArtifactSpec[dict[str, ClassInfo], dict[str, ClassInfo]](
        artifact_id="class_index",
        stage=_ParseModuleStage.CLASS_INDEX,
        init=dict,
        fold=lambda class_index, path, tree: _accumulate_class_index_for_tree(
            class_index,
            path,
            tree,
            project_root=project_root,
        ),
        finish=lambda class_index: class_index,
    )


def _collect_class_index(
    paths: list[Path],
    project_root,
    *,
    parse_failure_witnesses: list[JSONObject],
) -> dict[str, ClassInfo]:
    check_deadline()
    raw_class_index, = _build_module_artifacts(
        paths,
        specs=(
            cast(
                _ModuleArtifactSpec[object, object],
                _class_index_module_artifact_spec(project_root=project_root),
            ),
        ),
        parse_failure_witnesses=parse_failure_witnesses,
    )
    return cast(dict[str, ClassInfo], raw_class_index)


def _resolve_class_candidates(
    base: str,
    *,
    module: str,
    symbol_table,
    class_index: dict[str, ClassInfo],
) -> list[str]:
    check_deadline()
    if not base:
        return []
    candidates: list[str] = []
    if "." in base:
        parts = base.split(".")
        head = parts[0]
        tail = ".".join(parts[1:])
        if symbol_table is not None:
            resolved_head = symbol_table.resolve(module, head)
            if resolved_head:
                candidates.append(f"{resolved_head}.{tail}")
        if module:
            candidates.append(f"{module}.{base}")
        candidates.append(base)
    else:
        if symbol_table is not None:
            resolved = symbol_table.resolve(module, base)
            if resolved:
                candidates.append(resolved)
            resolved_star = symbol_table.resolve_star(module, base)
            if resolved_star:
                candidates.append(resolved_star)
        if module:
            candidates.append(f"{module}.{base}")
        candidates.append(base)
    seen: set[str] = set()
    resolved: list[str] = []
    for candidate in candidates:
        check_deadline()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate in class_index:
            resolved.append(candidate)
    return resolved


class _MethodHierarchyResolutionStatus(StrEnum):
    FOUND = "found"
    NOT_FOUND = "not_found"


@dataclass(frozen=True)
class _MethodHierarchyResolutionOutcome:
    status: _MethodHierarchyResolutionStatus
    method_qual: str


def _resolve_method_in_hierarchy_outcome(
    class_qual: str,
    method: str,
    *,
    class_index: dict[str, ClassInfo],
    by_qual: dict[str, FunctionInfo],
    symbol_table,
    seen: set[str],
) -> _MethodHierarchyResolutionOutcome:
    check_deadline()
    if class_qual in seen:
        return _MethodHierarchyResolutionOutcome(
            _MethodHierarchyResolutionStatus.NOT_FOUND,
            "",
        )
    seen.add(class_qual)
    candidate = f"{class_qual}.{method}"
    if candidate in by_qual:
        return _MethodHierarchyResolutionOutcome(
            _MethodHierarchyResolutionStatus.FOUND,
            candidate,
        )
    info = class_index.get(class_qual)
    if info is not None:
        for base in info.bases:
            check_deadline()
            for base_qual in _resolve_class_candidates(
                base,
                module=info.module,
                symbol_table=symbol_table,
                class_index=class_index,
            ):
                check_deadline()
                resolved = _resolve_method_in_hierarchy_outcome(
                    base_qual,
                    method,
                    class_index=class_index,
                    by_qual=by_qual,
                    symbol_table=symbol_table,
                    seen=seen,
                )
                if resolved.status is _MethodHierarchyResolutionStatus.FOUND:
                    return resolved
    return _MethodHierarchyResolutionOutcome(
        _MethodHierarchyResolutionStatus.NOT_FOUND,
        "",
    )


def _resolve_method_in_hierarchy(
    class_qual: str,
    method: str,
    *,
    class_index: dict[str, ClassInfo],
    by_qual: dict[str, FunctionInfo],
    symbol_table,
    seen: set[str],
):
    outcome = _resolve_method_in_hierarchy_outcome(
        class_qual,
        method,
        class_index=class_index,
        by_qual=by_qual,
        symbol_table=symbol_table,
        seen=seen,
    )
    return by_qual.get(outcome.method_qual)


@dataclass
class _FunctionIndexAccumulator:
    by_name: dict[str, list[FunctionInfo]] = field(
        default_factory=lambda: defaultdict(list)
    )
    by_qual: dict[str, FunctionInfo] = field(default_factory=dict)


def _accumulate_function_index_for_tree(
    acc: _FunctionIndexAccumulator,
    path: Path,
    tree: ast.Module,
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    transparent_decorators,
) -> None:
    check_deadline()
    funcs = _collect_functions(tree)
    if not funcs:
        return
    parents = ParentAnnotator()
    parents.visit(tree)
    parent_map = parents.parents
    module = _module_name(path, project_root)
    lambda_infos = _collect_lambda_function_infos(
        tree,
        path=path,
        module=module,
        parent_map=parent_map,
        ignore_params=ignore_params,
    )
    lambda_bindings_by_caller = _collect_lambda_bindings_by_caller(
        tree,
        module=module,
        parent_map=parent_map,
        lambda_infos=lambda_infos,
    )
    direct_lambda_callee_by_call_span = _direct_lambda_callee_by_call_span(
        tree,
        lambda_infos=lambda_infos,
    )
    return_aliases = _collect_return_aliases(funcs, parent_map, ignore_params=ignore_params)
    for fn in funcs:
        check_deadline()
        class_name = _enclosing_class(fn, parent_map)
        scopes = _enclosing_scopes(fn, parent_map)
        lexical_scopes = _enclosing_function_scopes(fn, parent_map)
        use_map, raw_call_args = _analyze_function(
            fn,
            parent_map,
            is_test=_is_test_path(path),
            ignore_params=ignore_params,
            strictness=strictness,
            class_name=class_name,
            return_aliases=return_aliases,
        )
        call_args = _materialize_direct_lambda_callees(
            raw_call_args,
            direct_lambda_callee_by_call_span=direct_lambda_callee_by_call_span,
        )
        unused_params, unknown_key_carriers = _unused_params(use_map)
        decision_reason_map = _decision_surface_reason_map(fn, ignore_params)
        value_params, value_reasons = _value_encoded_decision_params(fn, ignore_params)
        pos_args = [a.arg for a in (fn.args.posonlyargs + fn.args.args)]
        kwonly_args = [a.arg for a in fn.args.kwonlyargs]
        if pos_args and pos_args[0] in {"self", "cls"}:
            pos_args = pos_args[1:]
        if ignore_params:
            pos_args = [name for name in pos_args if name not in ignore_params]
            kwonly_args = [name for name in kwonly_args if name not in ignore_params]
        vararg = None
        if fn.args.vararg is not None:
            candidate = fn.args.vararg.arg
            if not ignore_params or candidate not in ignore_params:
                vararg = candidate
        kwarg = None
        if fn.args.kwarg is not None:
            candidate = fn.args.kwarg.arg
            if not ignore_params or candidate not in ignore_params:
                kwarg = candidate
        qual_parts = [module] if module else []
        if scopes:
            qual_parts.extend(scopes)
        qual_parts.append(fn.name)
        qual = ".".join(qual_parts)
        info = FunctionInfo(
            name=fn.name,
            qual=qual,
            path=path,
            params=_param_names(fn, ignore_params),
            annots=_param_annotations(fn, ignore_params),
            defaults=_param_defaults(fn, ignore_params),
            calls=call_args,
            unused_params=unused_params,
            unknown_key_carriers=unknown_key_carriers,
            transparent=_decorators_transparent(fn, transparent_decorators),
            class_name=class_name,
            scope=tuple(scopes),
            lexical_scope=tuple(lexical_scopes),
            decision_params=set(decision_reason_map),
            decision_surface_reasons=decision_reason_map,
            value_decision_params=value_params,
            value_decision_reasons=value_reasons,
            positional_params=tuple(pos_args),
            kwonly_params=tuple(kwonly_args),
            vararg=vararg,
            kwarg=kwarg,
            param_spans=_param_spans(fn, ignore_params),
            function_span=_node_span(fn),
            local_lambda_bindings=lambda_bindings_by_caller.get(qual, {}),
        )
        acc.by_name[fn.name].append(info)
        acc.by_qual[info.qual] = info
    for info in lambda_infos:
        check_deadline()
        acc.by_name[info.name].append(info)
        acc.by_qual[info.qual] = info


def _synthetic_lambda_name(
    *,
    module: str,
    lexical_scope: Sequence[str],
    span: tuple[int, int, int, int],
) -> str:
    check_deadline()
    lexical = ".".join(lexical_scope) if lexical_scope else "<module>"
    stable_payload = f"{module}|{lexical}|{span[0]}:{span[1]}:{span[2]}:{span[3]}"
    digest = hashlib.sha1(stable_payload.encode("utf-8")).hexdigest()[:12]
    return f"<lambda:{digest}>"


def _collect_lambda_function_infos(
    tree: ast.AST,
    *,
    path: Path,
    module: str,
    parent_map: Mapping[ast.AST, ast.AST],
    ignore_params,
) -> list[FunctionInfo]:
    check_deadline()
    lambda_infos: list[FunctionInfo] = []
    for node in ast.walk(tree):
        check_deadline()
        if type(node) is ast.Lambda:
            lambda_node = cast(ast.Lambda, node)
            span = _node_span(lambda_node)
            if span is not None:
                lexical_scopes = _enclosing_function_scopes(lambda_node, parent_map)
                scopes = _enclosing_scopes(lambda_node, parent_map)
                class_name = _enclosing_class(lambda_node, parent_map)
                synthetic_name = _synthetic_lambda_name(
                    module=module,
                    lexical_scope=lexical_scopes,
                    span=span,
                )
                qual_parts = [module] if module else []
                if scopes:
                    qual_parts.extend(scopes)
                qual_parts.append(synthetic_name)
                qual = ".".join(qual_parts)
                params = [
                    arg.arg
                    for arg in (
                        lambda_node.args.posonlyargs
                        + lambda_node.args.args
                        + lambda_node.args.kwonlyargs
                    )
                ]
                if ignore_params:
                    params = [name for name in params if name not in ignore_params]
                lambda_infos.append(
                    FunctionInfo(
                        name=synthetic_name,
                        qual=qual,
                        path=path,
                        params=params,
                        annots={name: None for name in params},
                        calls=[],
                        unused_params=set(),
                        class_name=class_name,
                        scope=tuple(scopes),
                        lexical_scope=tuple(lexical_scopes),
                        positional_params=tuple(params),
                        function_span=span,
                    )
                )
    return lambda_infos


def _collect_lambda_bindings_by_caller(
    tree: ast.AST,
    *,
    module: str,
    parent_map: dict[ast.AST, ast.AST],
    lambda_infos: Sequence[FunctionInfo],
) -> dict[str, dict[str, tuple[str, ...]]]:
    check_deadline()
    lambda_qual_by_span: dict[tuple[int, int, int, int], str] = {
        require_not_none(info.function_span, reason="lambda function site requires span", strict=True): info.qual
        for info in lambda_infos
    }
    binding_sets: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    closure_factories = _collect_closure_lambda_factories(
        tree,
        module=module,
        parent_map=parent_map,
        lambda_qual_by_span=lambda_qual_by_span,
    )

    for node in ast.walk(tree):
        check_deadline()
        assignment_node = None
        node_type = type(node)
        if node_type is ast.Assign or node_type is ast.AnnAssign:
            assignment_node = cast(ast.Assign | ast.AnnAssign, node)
        if assignment_node is not None and assignment_node.value is not None:
            fn_scope = _enclosing_scopes(assignment_node, parent_map)
            if fn_scope:
                targets = (
                    assignment_node.targets
                    if type(assignment_node) is ast.Assign
                    else [assignment_node.target]
                )
                qual_parts = [module] if module else []
                qual_parts.extend(fn_scope)
                caller_key = ".".join(qual_parts)
                value = assignment_node.value

                assigned_quals: set[str] = set()
                value_span = _node_span(value)
                value_type = type(value)
                if value_type is ast.Lambda and value_span is not None:
                    qual = lambda_qual_by_span.get(value_span)
                    if qual is not None:
                        assigned_quals.add(qual)
                elif value_type is ast.Name:
                    assigned_quals.update(
                        binding_sets.get(caller_key, {}).get(cast(ast.Name, value).id, set())
                    )
                elif value_type is ast.Call:
                    call_value = cast(ast.Call, value)
                    if type(call_value.func) is ast.Name:
                        assigned_quals.update(
                            closure_factories.get(cast(ast.Name, call_value.func).id, set())
                        )

                for target in targets:
                    check_deadline()
                    target_names = list(_target_names(target))
                    if type(target) is ast.Attribute:
                        attribute_target = cast(ast.Attribute, target)
                        target_value = attribute_target.value
                        if type(target_value) is ast.Name:
                            target_names.append(
                                f"{cast(ast.Name, target_value).id}.{attribute_target.attr}"
                            )
                    for name in target_names:
                        check_deadline()
                        if assigned_quals:
                            binding_sets[caller_key][name].update(assigned_quals)
                        else:
                            binding_sets[caller_key].pop(name, None)

    out: dict[str, dict[str, tuple[str, ...]]] = {}
    for caller_key, mapping in binding_sets.items():
        check_deadline()
        non_empty = {
            symbol: tuple(sort_once(quals, source = 'src/gabion/analysis/dataflow_audit.py:11609'))
            for symbol, quals in mapping.items()
            if quals
        }
        if non_empty:
            out[caller_key] = non_empty
    return out


def _collect_closure_lambda_factories(
    tree: ast.AST,
    *,
    module: str,
    parent_map: dict[ast.AST, ast.AST],
    lambda_qual_by_span: Mapping[tuple[int, int, int, int], str],
) -> dict[str, set[str]]:
    check_deadline()
    factories: dict[str, set[str]] = defaultdict(set)
    for node in ast.walk(tree):
        check_deadline()
        function_node = None
        node_type = type(node)
        if node_type is ast.FunctionDef or node_type is ast.AsyncFunctionDef:
            function_node = cast(ast.FunctionDef | ast.AsyncFunctionDef, node)
        if function_node is not None:
            local_bindings: dict[str, set[str]] = {}
            for statement in function_node.body:
                check_deadline()
                statement_type = type(statement)
                if statement_type is ast.Assign or statement_type is ast.AnnAssign:
                    assignment = cast(ast.Assign | ast.AnnAssign, statement)
                    value = assignment.value
                    if value is not None:
                        assigned_quals: set[str] = set()
                        value_span = _node_span(value)
                        value_type = type(value)
                        if value_type is ast.Lambda and value_span is not None:
                            qual = lambda_qual_by_span.get(value_span)
                            if qual is not None:
                                assigned_quals.add(qual)
                        elif value_type is ast.Name:
                            assigned_quals.update(
                                local_bindings.get(cast(ast.Name, value).id, set())
                            )
                        targets = (
                            assignment.targets
                            if type(assignment) is ast.Assign
                            else [assignment.target]
                        )
                        for target in targets:
                            check_deadline()
                            for name in _target_names(target):
                                check_deadline()
                                if assigned_quals:
                                    local_bindings[name] = set(assigned_quals)
                                else:
                                    local_bindings.pop(name, None)
                elif statement_type is ast.Return:
                    return_statement = cast(ast.Return, statement)
                    return_value = return_statement.value
                    if type(return_value) is ast.Name:
                        returned = local_bindings.get(cast(ast.Name, return_value).id, set())
                        if returned:
                            scopes = _enclosing_scopes(function_node, parent_map)
                            keys = {function_node.name}
                            if scopes:
                                keys.add(_function_key(scopes, function_node.name))
                            qual_parts = [module] if module else []
                            if scopes:
                                qual_parts.extend(scopes)
                            qual_parts.append(function_node.name)
                            keys.add(".".join(qual_parts))
                            for key in keys:
                                check_deadline()
                                factories[key].update(returned)
    return factories


def _direct_lambda_callee_by_call_span(
    tree: ast.AST,
    *,
    lambda_infos: Sequence[FunctionInfo],
) -> dict[tuple[int, int, int, int], str]:
    check_deadline()
    lambda_qual_by_span = {
        info.function_span: info.qual
        for info in lambda_infos
        if info.function_span is not None
    }
    mapping: dict[tuple[int, int, int, int], str] = {}
    for node in ast.walk(tree):
        check_deadline()
        if type(node) is ast.Call:
            call_node = cast(ast.Call, node)
            if type(call_node.func) is ast.Lambda:
                call_span = _node_span(call_node)
                lambda_span = _node_span(call_node.func)
                if call_span is not None and lambda_span is not None:
                    callee = lambda_qual_by_span.get(lambda_span)
                    if callee is not None:
                        mapping[call_span] = callee
    return mapping


def _materialize_direct_lambda_callees(
    call_args: Sequence[CallArgs],
    *,
    direct_lambda_callee_by_call_span: Mapping[tuple[int, int, int, int], str],
) -> list[CallArgs]:
    out: list[CallArgs] = []
    for call in call_args:
        check_deadline()
        if call.span is not None and call.span in direct_lambda_callee_by_call_span:
            out.append(replace(call, callee=direct_lambda_callee_by_call_span[call.span]))
            continue
        out.append(call)
    return out


def _function_index_module_artifact_spec(
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    transparent_decorators,
) -> _ModuleArtifactSpec[
    _FunctionIndexAccumulator,
    tuple[dict[str, list[FunctionInfo]], dict[str, FunctionInfo]],
]:
    return _ModuleArtifactSpec[
        _FunctionIndexAccumulator,
        tuple[dict[str, list[FunctionInfo]], dict[str, FunctionInfo]],
    ](
        artifact_id="function_index",
        stage=_ParseModuleStage.FUNCTION_INDEX,
        init=_FunctionIndexAccumulator,
        fold=lambda acc, path, tree: _accumulate_function_index_for_tree(
            acc,
            path,
            tree,
            project_root=project_root,
            ignore_params=ignore_params,
            strictness=strictness,
            transparent_decorators=transparent_decorators,
        ),
        finish=lambda acc: (acc.by_name, acc.by_qual),
    )


def _build_function_index(
    paths: list[Path],
    project_root,
    ignore_params: set[str],
    strictness: str,
    transparent_decorators = None,
    *,
    parse_failure_witnesses: list[JSONObject],
) -> tuple[dict[str, list[FunctionInfo]], dict[str, FunctionInfo]]:
    check_deadline()
    raw_index, = _build_module_artifacts(
        paths,
        specs=(
            cast(
                _ModuleArtifactSpec[object, object],
                _function_index_module_artifact_spec(
                    project_root=project_root,
                    ignore_params=ignore_params,
                    strictness=strictness,
                    transparent_decorators=transparent_decorators,
                ),
            ),
        ),
        parse_failure_witnesses=parse_failure_witnesses,
    )
    return cast(
        tuple[dict[str, list[FunctionInfo]], dict[str, FunctionInfo]],
        raw_index,
    )


def _resolve_callee(
    callee_key: str,
    caller: FunctionInfo,
    by_name: dict[str, list[FunctionInfo]],
    by_qual: dict[str, FunctionInfo],
    symbol_table = None,
    project_root = None,
    class_index = None,
    call = None,
    ambiguity_sink = None,
    local_lambda_bindings = None,
):
    check_deadline()
    lambda_bindings = local_lambda_bindings
    if lambda_bindings is None:
        lambda_bindings = caller.local_lambda_bindings
    context = _CalleeResolutionContextCore(
        callee_key=callee_key,
        caller=caller,
        by_name=by_name,
        by_qual=by_qual,
        symbol_table=symbol_table,
        project_root=project_root,
        class_index=class_index,
        call=call,
        local_lambda_bindings=lambda_bindings,
        caller_module=_module_name(caller.path, project_root=project_root),
    )
    resolution = _resolve_callee_with_effects_impl(context)
    if ambiguity_sink is not None:
        for effect in _collect_callee_resolution_effects_impl(resolution):
            check_deadline()
            ambiguity_sink(
                caller,
                call,
                list(effect.candidates),
                effect.phase,
                effect.callee_key,
            )
    return resolution.resolved


def _is_dynamic_dispatch_callee_key(callee_key: str) -> bool:
    """Classify obvious syntax-level dynamic-dispatch call shapes."""
    check_deadline()
    text = callee_key.strip()
    if not text:
        return False
    if text.startswith("getattr("):
        return True
    if "." not in text:
        return False
    base, _, _ = text.partition(".")
    base = base.strip()
    if not base or base in {"self", "cls"}:
        return False
    if any(token in base for token in ("(", "[", "{")):
        return True
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", base) is None:
        return True
    return False


@dataclass(frozen=True)
class _CalleeResolutionOutcome:
    status: str
    phase: str
    callee_key: str
    candidates: tuple[FunctionInfo, ...] = ()


def _dedupe_resolution_candidates(
    candidates: Iterable[FunctionInfo],
) -> tuple[FunctionInfo, ...]:
    deduped: dict[str, FunctionInfo] = {}
    for candidate in candidates:
        check_deadline()
        if _is_test_path(candidate.path):
            continue
        deduped[candidate.qual] = candidate
    return tuple(sort_once(deduped.values(), key=lambda info: info.qual, source = 'src/gabion/analysis/dataflow_audit.py:11956'))


def _resolve_callee_outcome(
    callee_key: str,
    caller: FunctionInfo,
    by_name: dict[str, list[FunctionInfo]],
    by_qual: dict[str, FunctionInfo],
    *,
    symbol_table = None,
    project_root = None,
    class_index = None,
    call = None,
    local_lambda_bindings = None,
    resolve_callee_fn = _resolve_callee,
) -> _CalleeResolutionOutcome:
    check_deadline()
    ambiguous_candidates: list[FunctionInfo] = []
    ambiguity_phase = "unresolved"
    ambiguity_callee_key = callee_key

    def _sink(
        sink_caller: FunctionInfo,
        sink_call,
        candidates: list[FunctionInfo],
        phase: str,
        sink_callee_key: str,
    ) -> None:
        check_deadline()
        del sink_caller, sink_call
        ambiguous_candidates.extend(candidates)
        nonlocal ambiguity_phase, ambiguity_callee_key
        ambiguity_phase = phase
        ambiguity_callee_key = sink_callee_key

    if resolve_callee_fn is _resolve_callee:
        lambda_bindings = local_lambda_bindings
        if lambda_bindings is None:
            lambda_bindings = caller.local_lambda_bindings
        context = _CalleeResolutionContextCore(
            callee_key=callee_key,
            caller=caller,
            by_name=by_name,
            by_qual=by_qual,
            symbol_table=symbol_table,
            project_root=project_root,
            class_index=class_index,
            call=call,
            local_lambda_bindings=lambda_bindings,
            caller_module=_module_name(caller.path, project_root=project_root),
        )
        resolution = _resolve_callee_with_effects_impl(context)
        for effect in _collect_callee_resolution_effects_impl(resolution):
            check_deadline()
            _sink(caller, call, list(effect.candidates), effect.phase, effect.callee_key)
        resolved = resolution.resolved
    else:
        resolved = resolve_callee_fn(
            callee_key,
            caller,
            by_name,
            by_qual,
            symbol_table=symbol_table,
            project_root=project_root,
            class_index=class_index,
            call=call,
            ambiguity_sink=_sink,
            local_lambda_bindings=local_lambda_bindings,
        )
    if resolved is not None:
        return _CalleeResolutionOutcome(
            status="resolved",
            phase="resolved",
            callee_key=callee_key,
            candidates=(resolved,),
        )
    ambiguous = _dedupe_resolution_candidates(ambiguous_candidates)
    if ambiguous:
        return _CalleeResolutionOutcome(
            status="ambiguous",
            phase=ambiguity_phase,
            callee_key=ambiguity_callee_key,
            candidates=ambiguous,
        )
    internal_pool = list(by_name.get(_callee_key(callee_key), []))
    lambda_bindings = local_lambda_bindings
    if lambda_bindings is None:
        lambda_bindings = caller.local_lambda_bindings
    if "." not in callee_key:
        for qual in lambda_bindings.get(callee_key, ()):
            check_deadline()
            candidate = by_qual.get(qual)
            if candidate is not None:
                internal_pool.append(candidate)
    internal_candidates = _dedupe_resolution_candidates(internal_pool)
    if internal_candidates:
        return _CalleeResolutionOutcome(
            status="unresolved_internal",
            phase="unresolved_internal",
            callee_key=callee_key,
            candidates=internal_candidates,
        )
    if _is_dynamic_dispatch_callee_key(callee_key):
        return _CalleeResolutionOutcome(
            status="unresolved_dynamic",
            phase="unresolved_dynamic",
            callee_key=callee_key,
            candidates=(),
        )
    return _CalleeResolutionOutcome(
        status="unresolved_external",
        phase="unresolved_external",
        callee_key=callee_key,
        candidates=(),
    )


def _format_type_flow_site(
    *,
    caller: FunctionInfo,
    call: CallArgs,
    callee: FunctionInfo,
    caller_param: str,
    callee_param: str,
    annot: str,
    project_root,
) -> str:
    """Format a stable, machine-actionable callsite for type-flow evidence."""
    caller_name = _function_key(caller.scope, caller.name)
    caller_path = _normalize_snapshot_path(caller.path, project_root)
    if call.span is None:
        loc = f"{caller_path}:{caller_name}"
    else:
        line, col, _, _ = call.span
        loc = f"{caller_path}:{line + 1}:{col + 1}"
    return (
        f"{loc}: {caller_name}.{caller_param} -> {callee.qual}.{callee_param} expects {annot}"
    )


def _infer_type_flow(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators = None,
    max_sites_per_param: int = 3,
    parse_failure_witnesses: list[JSONObject],
    analysis_index = None,
):
    """Repo-wide fixed-point pass for downstream type tightening + evidence."""
    check_deadline()
    index = require_not_none(
        analysis_index,
        reason="_infer_type_flow requires prebuilt analysis_index",
        strict=True,
    )
    by_name = index.by_name
    by_qual = index.by_qual
    resolved_edges_by_caller = _analysis_index_resolved_call_edges_by_caller(
        index,
        project_root=project_root,
        require_transparent=True,
    )
    inferred: dict[str, dict[str, object]] = {}
    for infos in by_name.values():
        check_deadline()
        for info in infos:
            check_deadline()
            inferred[info.qual] = dict(info.annots)

    def _get_annot(info: FunctionInfo, param: str):
        value = inferred.get(info.qual, {}).get(param)
        if type(value) is str:
            return value
        return None

    def _downstream_for(info: FunctionInfo) -> tuple[dict[str, set[str]], dict[str, dict[str, set[str]]]]:
        check_deadline()
        downstream: dict[str, set[str]] = defaultdict(set)
        sites: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        for edge in resolved_edges_by_caller.get(info.qual, ()):
            check_deadline()
            callee = edge.callee
            call = edge.call
            callee_to_caller = _caller_param_bindings_for_call(
                call,
                callee,
                strictness=strictness,
            )
            for callee_param, callers in callee_to_caller.items():
                check_deadline()
                annot = _get_annot(callee, callee_param)
                if not annot:
                    continue
                for caller_param in callers:
                    check_deadline()
                    downstream[caller_param].add(annot)
                    sites[caller_param][annot].add(
                        _format_type_flow_site(
                            caller=info,
                            call=call,
                            callee=callee,
                            caller_param=caller_param,
                            callee_param=callee_param,
                            annot=annot,
                            project_root=project_root,
                        )
                    )
        return downstream, sites

    # Fixed-point inference pass.
    changed = True
    while changed:
        check_deadline()
        changed = False
        for infos in by_name.values():
            check_deadline()
            for info in infos:
                check_deadline()
                if _is_test_path(info.path):
                    continue
                downstream, _ = _downstream_for(info)
                for param, annots in downstream.items():
                    check_deadline()
                    if len(annots) != 1:
                        continue
                    downstream_annot = next(iter(annots))
                    current = _get_annot(info, param)
                    if _is_broad_type(current) and downstream_annot:
                        if inferred[info.qual].get(param) != downstream_annot:
                            inferred[info.qual][param] = downstream_annot
                            changed = True

    suggestions: set[str] = set()
    ambiguities: set[str] = set()
    evidence_lines: set[str] = set()
    for infos in by_name.values():
        check_deadline()
        for info in infos:
            check_deadline()
            if _is_test_path(info.path):
                continue
            downstream, sites = _downstream_for(info)
            fn_key = _function_key(info.scope, info.name)
            path_key = _normalize_snapshot_path(info.path, project_root)
            for param, annots in downstream.items():
                check_deadline()
                if len(annots) > 1:
                    ambiguities.add(
                        f"{path_key}:{fn_key}.{param} downstream types conflict: {sort_once(annots, source = 'src/gabion/analysis/dataflow_audit.py:12183')}"
                    )
                    for annot in sort_once(annots, source = 'src/gabion/analysis/dataflow_audit.py:12185'):
                        check_deadline()
                        for site in sort_once(sites.get(param, {}).get(annot, set()), source = 'src/gabion/analysis/dataflow_audit.py:12187')[
                            :max_sites_per_param
                        ]:
                            check_deadline()
                            evidence_lines.add(site)
                    continue
                downstream_annot = next(iter(annots))
                original = info.annots.get(param)
                final = inferred.get(info.qual, {}).get(param)
                if _is_broad_type(original) and final == downstream_annot and downstream_annot:
                    suggestions.add(
                        f"{path_key}:{fn_key}.{param} can tighten to {downstream_annot}"
                    )
                    for site in sort_once(
                        sites.get(param, {}).get(downstream_annot, set()), 
                    source = 'src/gabion/analysis/dataflow_audit.py:12200')[:max_sites_per_param]:
                        check_deadline()
                        evidence_lines.add(site)
    return inferred, sort_once(suggestions, source = 'src/gabion/analysis/dataflow_audit.py:12205'), sort_once(ambiguities, source = 'src/gabion/analysis/dataflow_audit.py:12205'), sort_once(evidence_lines, source = 'src/gabion/analysis/dataflow_audit.py:12205')


def analyze_type_flow_repo_with_map(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators = None,
    parse_failure_witnesses = None,
    analysis_index = None,
):
    """Repo-wide fixed-point pass for downstream type tightening."""
    check_deadline()
    return _run_indexed_pass(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=analysis_index,
        spec=_IndexedPassSpec(
            pass_id="type_flow_with_map",
            run=lambda context: _infer_type_flow(
                context.paths,
                project_root=context.project_root,
                ignore_params=context.ignore_params,
                strictness=context.strictness,
                external_filter=context.external_filter,
                transparent_decorators=context.transparent_decorators,
                parse_failure_witnesses=context.parse_failure_witnesses,
                analysis_index=context.analysis_index,
            )[:3],
        ),
    )


def analyze_type_flow_repo_with_evidence(
    paths: list[Path],
    *,
    project_root: OptionalProjectRoot,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators: OptionalDecorators = None,
    max_sites_per_param: int = 3,
    parse_failure_witnesses: OptionalParseFailures = None,
    analysis_index: OptionalAnalysisIndex = None,
) -> tuple[list[str], list[str], list[str]]:
    check_deadline()
    return _run_indexed_pass(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=analysis_index,
        spec=_IndexedPassSpec(
            pass_id="type_flow_with_evidence",
            run=lambda context: _infer_type_flow(
                context.paths,
                project_root=context.project_root,
                ignore_params=context.ignore_params,
                strictness=context.strictness,
                external_filter=context.external_filter,
                transparent_decorators=context.transparent_decorators,
                max_sites_per_param=max_sites_per_param,
                parse_failure_witnesses=context.parse_failure_witnesses,
                analysis_index=context.analysis_index,
            )[1:],
        ),
    )


def analyze_type_flow_repo(
    paths: list[Path],
    *,
    project_root: OptionalProjectRoot,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators: OptionalDecorators = None,
    parse_failure_witnesses: OptionalParseFailures = None,
    analysis_index: OptionalAnalysisIndex = None,
) -> tuple[list[str], list[str]]:
    inferred, suggestions, ambiguities = analyze_type_flow_repo_with_map(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=analysis_index,
    )
    return suggestions, ambiguities


def analyze_constant_flow_repo(
    paths: list[Path],
    *,
    project_root: OptionalProjectRoot,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators: OptionalDecorators = None,
    parse_failure_witnesses: OptionalParseFailures = None,
    analysis_index: OptionalAnalysisIndex = None,
) -> list[str]:
    """Detect parameters that only receive a single constant value (non-test)."""
    return _run_indexed_pass(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=analysis_index,
        spec=_IndexedPassSpec(
            pass_id="constant_flow",
            run=lambda context: _constant_smells_from_details(
                _collect_constant_flow_details(
                    context.paths,
                    project_root=context.project_root,
                    ignore_params=context.ignore_params,
                    strictness=context.strictness,
                    external_filter=context.external_filter,
                    transparent_decorators=context.transparent_decorators,
                    parse_failure_witnesses=context.parse_failure_witnesses,
                    analysis_index=context.analysis_index,
                )
            ),
        ),
    )


@dataclass(frozen=True)
class ConstantFlowDetail:
    path: Path
    qual: str
    name: str
    param: str
    value: str
    count: int
    sites: tuple[str, ...] = ()


def _constant_smells_from_details(
    details: Iterable[ConstantFlowDetail],
) -> list[str]:
    check_deadline()
    smells: list[str] = []
    for detail in details:
        check_deadline()
        path_name = detail.path.name
        site_suffix = ""
        if detail.sites:
            sample = ", ".join(detail.sites[:3])
            site_suffix = f" (e.g. {sample})"
        smells.append(
            f"{path_name}:{detail.name}.{detail.param} only observed constant {detail.value} across {detail.count} non-test call(s){site_suffix}"
        )
    return sort_once(smells, source = 'src/gabion/analysis/dataflow_audit.py:12374')


def _deadness_witnesses_from_constant_details(
    details: Iterable[ConstantFlowDetail],
    *,
    project_root,
) -> list[JSONObject]:
    check_deadline()
    witnesses: list[JSONObject] = []
    for detail in details:
        check_deadline()
        path_value = _normalize_snapshot_path(detail.path, project_root)
        predicate = f"{detail.param} != {detail.value}"
        core = [
            f"observed constant {detail.value} across {detail.count} non-test call(s)"
        ]
        deadness_id = f"deadness:{path_value}:{detail.name}:{detail.param}:{detail.value}"
        witnesses.append(
            {
                "deadness_id": deadness_id,
                "path": path_value,
                "function": detail.name,
                "bundle": [detail.param],
                "environment": {detail.param: detail.value},
                "predicate": predicate,
                "core": core,
                "result": "UNREACHABLE",
                "call_sites": list(detail.sites[:10]),
                "projection": (
                    f"{detail.name}.{detail.param} constant {detail.value} across "
                    f"{detail.count} non-test call(s)"
                ),
            }
        )
    return sort_once(
        witnesses,
        key=lambda entry: (
            str(entry.get("path", "")),
            str(entry.get("function", "")),
            ",".join(entry.get("bundle", [])),
            str(entry.get("predicate", "")),
        ),
    source = 'src/gabion/analysis/dataflow_audit.py:12409')


def _format_call_site(caller: FunctionInfo, call: CallArgs) -> str:
    """Render a stable, human-friendly call site identifier.

    Spans are stored 0-based; we report 1-based line/col for readability.
    """
    caller_name = _function_key(caller.scope, caller.name)
    span = call.span
    if span is None:
        return f"{caller.path.name}:{caller_name}"
    line, col, _, _ = span
    return f"{caller.path.name}:{line + 1}:{col + 1}:{caller_name}"


@dataclass
class _ConstantFlowFoldAccumulator:
    const_values: dict[tuple[str, str], set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    non_const: dict[tuple[str, str], bool] = field(
        default_factory=lambda: defaultdict(bool)
    )
    call_counts: dict[tuple[str, str], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    call_sites: dict[tuple[str, str], set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )


@dataclass
class _KnobFlowFoldAccumulator:
    const_values: dict[tuple[str, str], set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    non_const: dict[tuple[str, str], bool] = field(
        default_factory=lambda: defaultdict(bool)
    )
    explicit_passed: dict[tuple[str, str], bool] = field(
        default_factory=lambda: defaultdict(bool)
    )
    call_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))


def _collect_constant_flow_details(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators = None,
    parse_failure_witnesses: list[JSONObject],
    analysis_index = None,
    iter_resolved_edge_param_events_fn: Callable[..., Iterable[_ResolvedEdgeParamEvent]] = _iter_resolved_edge_param_events,
    reduce_resolved_call_edges_fn: Callable[..., _ConstantFlowFoldAccumulator] = _reduce_resolved_call_edges,
) -> list[ConstantFlowDetail]:
    check_deadline()
    index = require_not_none(
        analysis_index,
        reason="_collect_constant_flow_details requires prebuilt analysis_index",
        strict=True,
    )
    by_qual = index.by_qual
    def _fold(acc: _ConstantFlowFoldAccumulator, edge: _ResolvedCallEdge) -> None:
        for event in iter_resolved_edge_param_events_fn(
            edge,
            strictness=strictness,
            include_variadics_in_low_star=False,
        ):
            check_deadline()
            key = (edge.callee.qual, event.param)
            if event.kind == "const":
                if event.value is not None:
                    acc.const_values[key].add(event.value)
                    if event.countable:
                        acc.call_counts[key] += 1
                        acc.call_sites[key].add(
                            _format_call_site(edge.caller, edge.call)
                        )
            else:
                acc.non_const[key] = True
                if event.countable:
                    acc.call_counts[key] += 1

    folded = reduce_resolved_call_edges_fn(
        index,
        project_root=project_root,
        require_transparent=True,
        spec=_ResolvedEdgeReducerSpec[
            _ConstantFlowFoldAccumulator, _ConstantFlowFoldAccumulator
        ](
            reducer_id="constant_flow",
            init=_ConstantFlowFoldAccumulator,
            fold=_fold,
            finish=lambda acc: acc,
        ),
    )

    details: list[ConstantFlowDetail] = []
    for key, values in folded.const_values.items():
        check_deadline()
        if folded.non_const.get(key):
            continue
        if len(values) != 1:
            continue
        qual, param = key
        info = by_qual.get(qual)
        path = info.path if info is not None else Path(qual)
        # Use the same scope-aware function key used elsewhere in the audit so
        # cross-artifact joins (e.g., deadness ↔ exception obligations) work.
        name = (
            _function_key(info.scope, info.name)
            if info is not None
            else qual.split(".")[-1]
        )
        count = folded.call_counts.get(key, 0)
        details.append(
            ConstantFlowDetail(
                path=path,
                qual=qual,
                name=name,
                param=param,
                value=next(iter(values)),
                count=count,
                sites=tuple(sort_once(folded.call_sites.get(key, set()), source = 'src/gabion/analysis/dataflow_audit.py:12543')),
            )
        )
    return sort_once(details, key=lambda entry: (str(entry.path), entry.name, entry.param), source = 'src/gabion/analysis/dataflow_audit.py:12546')


def analyze_deadness_flow_repo(
    paths: list[Path],
    *,
    project_root: OptionalProjectRoot,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators: OptionalDecorators = None,
    parse_failure_witnesses: OptionalParseFailures = None,
    analysis_index: OptionalAnalysisIndex = None,
) -> list[JSONObject]:
    """Emit deadness witnesses based on constant-only parameter flows."""
    return _run_indexed_pass(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=analysis_index,
        spec=_IndexedPassSpec(
            pass_id="deadness_flow",
            run=lambda context: _deadness_witnesses_from_constant_details(
                _collect_constant_flow_details(
                    context.paths,
                    project_root=context.project_root,
                    ignore_params=context.ignore_params,
                    strictness=context.strictness,
                    external_filter=context.external_filter,
                    transparent_decorators=context.transparent_decorators,
                    parse_failure_witnesses=context.parse_failure_witnesses,
                    analysis_index=context.analysis_index,
                ),
                project_root=context.project_root,
            ),
        ),
    )


def _compute_knob_param_names(
    *,
    by_name: dict[str, list[FunctionInfo]],
    by_qual: dict[str, FunctionInfo],
    symbol_table: SymbolTable,
    project_root,
    class_index: dict[str, ClassInfo],
    strictness: str,
    analysis_index = None,
) -> set[str]:
    check_deadline()
    index = analysis_index
    if index is None:
        index = AnalysisIndex(
            by_name=by_name,
            by_qual=by_qual,
            symbol_table=symbol_table,
            class_index=class_index,
        )
    def _fold(acc: _KnobFlowFoldAccumulator, edge: _ResolvedCallEdge) -> None:
        acc.call_counts[edge.callee.qual] += 1
        for event in _iter_resolved_edge_param_events(
            edge,
            strictness=strictness,
            include_variadics_in_low_star=True,
        ):
            check_deadline()
            key = (edge.callee.qual, event.param)
            if event.kind == "const":
                if event.value is not None:
                    acc.const_values[key].add(event.value)
            else:
                acc.non_const[key] = True
            acc.explicit_passed[key] = True

    folded = _reduce_resolved_call_edges(
        index,
        project_root=project_root,
        require_transparent=True,
        spec=_ResolvedEdgeReducerSpec[
            _KnobFlowFoldAccumulator, _KnobFlowFoldAccumulator
        ](
            reducer_id="knob_flow",
            init=_KnobFlowFoldAccumulator,
            fold=_fold,
            finish=lambda acc: acc,
        ),
    )
    knob_names: set[str] = set()
    for key, values in folded.const_values.items():
        check_deadline()
        if folded.non_const.get(key):
            continue
        if len(values) == 1:
            knob_names.add(key[1])
    for qual, info in by_qual.items():
        check_deadline()
        if folded.call_counts.get(qual, 0) == 0:
            continue
        for param in info.defaults:
            check_deadline()
            if not folded.explicit_passed.get((qual, param), False):
                knob_names.add(param)
    return knob_names


def _analyze_unused_arg_flow_indexed(
    context: _IndexedPassContext,
) -> list[str]:
    resolved_edges = _analysis_index_resolved_call_edges(
        context.analysis_index,
        project_root=context.project_root,
        require_transparent=True,
    )
    smells: set[str] = set()

    def _format(
        caller: FunctionInfo,
        callee_info: FunctionInfo,
        callee_param: str,
        arg_desc: str,
        *,
        category: Literal["unused", "unknown_key_carrier"] = "unused",
        call = None,
    ) -> str:
        # dataflow-bundle: callee_info, caller
        prefix = f"{caller.path.name}:{caller.name}"
        if call is not None and call.span is not None:
            line, col, _, _ = call.span
            prefix = f"{caller.path.name}:{line + 1}:{col + 1}:{caller.name}"
        if category == "unknown_key_carrier":
            return (
                f"{prefix} passes {arg_desc} to {callee_info.path.name}:{callee_info.name}.{callee_param} "
                f"(unknown key carrier)"
            )
        return (
            f"{prefix} passes {arg_desc} "
            f"to unused {callee_info.path.name}:{callee_info.name}.{callee_param} "
            f"(no forwarding use)"
        )

    for edge in resolved_edges:
        check_deadline()
        info = edge.caller
        call = edge.call
        callee = edge.callee
        if not callee.unused_params and not callee.unknown_key_carriers:
            continue
        callee_params = callee.params
        mapped_params = set()
        for idx_str in call.pos_map:
            check_deadline()
            idx = int(idx_str)
            if idx >= len(callee_params):
                continue
            mapped_params.add(callee_params[idx])
        for kw in call.kw_map:
            check_deadline()
            if kw in callee_params:
                mapped_params.add(kw)
        remaining = [
            (idx, name)
            for idx, name in enumerate(callee_params)
            if name not in mapped_params
        ]

        for idx_str, caller_param in call.pos_map.items():
            check_deadline()
            idx = int(idx_str)
            if idx >= len(callee_params):
                continue
            callee_param = callee_params[idx]
            if callee_param in callee.unused_params | callee.unknown_key_carriers:
                smells.add(
                    _format(
                        info,
                        callee,
                        callee_param,
                        f"param {caller_param}",
                        category=(
                            "unknown_key_carrier"
                            if callee_param in callee.unknown_key_carriers
                            else "unused"
                        ),
                        call=call,
                    )
                )
        for idx_str in call.non_const_pos:
            check_deadline()
            idx = int(idx_str)
            if idx >= len(callee_params):
                continue
            callee_param = callee_params[idx]
            if callee_param in callee.unused_params | callee.unknown_key_carriers:
                smells.add(
                    _format(
                        info,
                        callee,
                        callee_param,
                        f"non-constant arg at position {idx}",
                        category=(
                            "unknown_key_carrier"
                            if callee_param in callee.unknown_key_carriers
                            else "unused"
                        ),
                        call=call,
                    )
                )
        for kw, caller_param in call.kw_map.items():
            check_deadline()
            if kw not in callee_params:
                continue
            if kw in callee.unused_params | callee.unknown_key_carriers:
                smells.add(
                    _format(
                        info,
                        callee,
                        kw,
                        f"param {caller_param}",
                        category=(
                            "unknown_key_carrier"
                            if kw in callee.unknown_key_carriers
                            else "unused"
                        ),
                        call=call,
                    )
                )
        for kw in call.non_const_kw:
            check_deadline()
            if kw not in callee_params:
                continue
            if kw in callee.unused_params | callee.unknown_key_carriers:
                smells.add(
                    _format(
                        info,
                        callee,
                        kw,
                        f"non-constant kw '{kw}'",
                        category=(
                            "unknown_key_carrier"
                            if kw in callee.unknown_key_carriers
                            else "unused"
                        ),
                        call=call,
                    )
                )
        if context.strictness == "low":
            if len(call.star_pos) == 1:
                for idx, param in remaining:
                    check_deadline()
                    if param in callee.unused_params | callee.unknown_key_carriers:
                        smells.add(
                            _format(
                                info,
                                callee,
                                param,
                                f"non-constant arg at position {idx}",
                                category=(
                                    "unknown_key_carrier"
                                    if param in callee.unknown_key_carriers
                                    else "unused"
                                ),
                                call=call,
                            )
                        )
            if len(call.star_kw) == 1:
                for _, param in remaining:
                    check_deadline()
                    if param in callee.unused_params | callee.unknown_key_carriers:
                        smells.add(
                            _format(
                                info,
                                callee,
                                param,
                                f"non-constant kw '{param}'",
                                category=(
                                    "unknown_key_carrier"
                                    if param in callee.unknown_key_carriers
                                    else "unused"
                                ),
                                call=call,
                            )
                        )
    return sort_once(smells, source = 'src/gabion/analysis/dataflow_audit.py:12832')


def analyze_unused_arg_flow_repo(
    paths: list[Path],
    *,
    project_root: OptionalProjectRoot,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators: OptionalDecorators = None,
    parse_failure_witnesses: OptionalParseFailures = None,
    analysis_index: OptionalAnalysisIndex = None,
) -> list[str]:
    """Detect non-constant arguments passed into unused callee parameters."""
    check_deadline()
    return _run_indexed_pass(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=analysis_index,
        spec=_IndexedPassSpec(
            pass_id="unused_arg_flow",
            run=_analyze_unused_arg_flow_indexed,
        ),
    )


def _iter_config_fields(
    path: Path,
    *,
    tree = None,
    parse_failure_witnesses: list[JSONObject],
) -> dict[str, set[str]]:
    """Best-effort extraction of config bundles from dataclasses."""
    check_deadline()
    module_tree = tree
    if module_tree is None:
        module_tree = _parse_module_tree(
            path,
            stage=_ParseModuleStage.CONFIG_FIELDS,
            parse_failure_witnesses=parse_failure_witnesses,
        )
    if module_tree is None:
        return {}
    bundles: dict[str, set[str]] = {}
    for node in ast.walk(module_tree):
        check_deadline()
        if type(node) is not ast.ClassDef:
            continue
        class_node = cast(ast.ClassDef, node)
        decorators = {getattr(d, "id", None) for d in class_node.decorator_list}
        is_dataclass = "dataclass" in decorators
        is_config = class_node.name.endswith("Config")
        if not is_dataclass and not is_config:
            continue
        fields: set[str] = set()
        for stmt in class_node.body:
            check_deadline()
            stmt_type = type(stmt)
            if stmt_type is ast.AnnAssign:
                ann_stmt = cast(ast.AnnAssign, stmt)
                name = _simple_store_name(ann_stmt.target)
                if name is not None and (is_config or name.endswith("_fn")):
                    fields.add(name)
            elif stmt_type is ast.Assign:
                assign_stmt = cast(ast.Assign, stmt)
                for target in assign_stmt.targets:
                    check_deadline()
                    name = _simple_store_name(target)
                    if name is not None and (is_config or name.endswith("_fn")):
                        fields.add(name)
        if fields:
            bundles[class_node.name] = fields
    return bundles


def _collect_config_bundles(
    paths: list[Path],
    *,
    parse_failure_witnesses: list[JSONObject],
    analysis_index = None,
) -> dict[Path, dict[str, set[str]]]:
    check_deadline()
    _forbid_adhoc_bundle_discovery("_collect_config_bundles")
    bundles_by_path: dict[Path, dict[str, set[str]]] = {}
    if analysis_index is not None:
        config_fields_by_path = _analysis_index_stage_cache(
            analysis_index,
            paths,
            spec=_StageCacheSpec(
                stage=_ParseModuleStage.CONFIG_FIELDS,
                cache_key=_parse_stage_cache_key(
                    stage=_ParseModuleStage.CONFIG_FIELDS,
                    cache_context=_EMPTY_CACHE_SEMANTIC_CONTEXT,
                    config_subset={},
                    detail="config_fields",
                ),
                build=lambda tree, path: _iter_config_fields(
                    path,
                    tree=tree,
                    parse_failure_witnesses=parse_failure_witnesses,
                ),
            ),
            parse_failure_witnesses=parse_failure_witnesses,
        )
        for path, bundles in config_fields_by_path.items():
            check_deadline()
            if bundles:
                bundles_by_path[path] = bundles
        return bundles_by_path
    for path in paths:
        check_deadline()
        bundles = _iter_config_fields(
            path,
            parse_failure_witnesses=parse_failure_witnesses,
        )
        if bundles:
            bundles_by_path[path] = bundles
    return bundles_by_path


_BUNDLE_MARKER = re.compile(r"dataflow-bundle:\s*(.*)")


def _iter_documented_bundles(path: Path) -> set[tuple[str, ...]]:
    """Return bundles documented via '# dataflow-bundle: a, b' markers."""
    check_deadline()
    _forbid_adhoc_bundle_discovery("_iter_documented_bundles")
    bundles: set[tuple[str, ...]] = set()
    try:
        text = path.read_text()
    except (OSError, UnicodeError):
        return bundles
    for line in text.splitlines():
        check_deadline()
        match = _BUNDLE_MARKER.search(line)
        if not match:
            continue
        payload = match.group(1)
        if not payload:
            continue
        parts = [p.strip() for p in re.split(r"[,\s]+", payload) if p.strip()]
        if len(parts) < 2:
            continue
        bundles.add(tuple(sort_once(parts, source = 'src/gabion/analysis/dataflow_audit.py:12976')))
    return bundles


def _collect_dataclass_registry(
    paths: list[Path],
    *,
    project_root,
    parse_failure_witnesses: list[JSONObject],
    analysis_index = None,
    stage_cache_fn = None,
) -> dict[str, list[str]]:
    check_deadline()
    if stage_cache_fn is None:
        stage_cache_fn = _analysis_index_stage_cache
    registry: dict[str, list[str]] = {}
    if analysis_index is not None:
        registry_by_path = stage_cache_fn(
            analysis_index,
            paths,
            spec=_StageCacheSpec(
                stage=_ParseModuleStage.DATACLASS_REGISTRY,
                cache_key=_parse_stage_cache_key(
                    stage=_ParseModuleStage.DATACLASS_REGISTRY,
                    cache_context=_EMPTY_CACHE_SEMANTIC_CONTEXT,
                    config_subset={
                        "project_root": str(project_root) if project_root is not None else "",
                    },
                    detail="dataclass_registry",
                ),
                build=lambda tree, path: _dataclass_registry_for_tree(
                    path,
                    tree,
                    project_root=project_root,
                ),
            ),
            parse_failure_witnesses=parse_failure_witnesses,
        )
        for entries in registry_by_path.values():
            check_deadline()
            if entries is not None:
                registry.update(entries)
        return registry
    for path in paths:
        check_deadline()
        tree = _parse_module_tree(
            path,
            stage=_ParseModuleStage.DATACLASS_REGISTRY,
            parse_failure_witnesses=parse_failure_witnesses,
        )
        if tree is not None:
            registry.update(_dataclass_registry_for_tree(path, tree, project_root=project_root))
    return registry


def _dataclass_registry_for_tree(
    path: Path,
    tree: ast.AST,
    *,
    project_root = None,
) -> dict[str, list[str]]:
    check_deadline()
    registry: dict[str, list[str]] = {}
    module = _module_name(path, project_root)
    for node in ast.walk(tree):
        check_deadline()
        if type(node) is not ast.ClassDef:
            continue
        class_node = cast(ast.ClassDef, node)
        decorators = {
            ast.unparse(dec) if hasattr(ast, "unparse") else ""
            for dec in class_node.decorator_list
        }
        if not any("dataclass" in dec for dec in decorators):
            continue
        fields: list[str] = []
        for stmt in class_node.body:
            check_deadline()
            stmt_type = type(stmt)
            if stmt_type is ast.AnnAssign:
                ann_stmt = cast(ast.AnnAssign, stmt)
                name = _simple_store_name(ann_stmt.target)
                if name is not None:
                    fields.append(name)
            elif stmt_type is ast.Assign:
                assign_stmt = cast(ast.Assign, stmt)
                for target in assign_stmt.targets:
                    check_deadline()
                    name = _simple_store_name(target)
                    if name is not None:
                        fields.append(name)
        if not fields:
            continue
        if module:
            registry[f"{module}.{class_node.name}"] = fields
        else:  # pragma: no cover - module name is always non-empty for file paths
            registry[class_node.name] = fields
    return registry


def _bundle_name_registry(root: Path) -> dict[tuple[str, ...], set[str]]:
    check_deadline()
    file_paths = sort_once(
        root.rglob("*.py"),
        source="_bundle_name_registry.file_paths",
        key=lambda path: str(path),
    )
    parse_failure_witnesses: list[JSONObject] = []
    config_bundles_by_path = _collect_config_bundles(
        file_paths,
        parse_failure_witnesses=parse_failure_witnesses,
    )
    dataclass_registry = _collect_dataclass_registry(
        file_paths,
        project_root=root,
        parse_failure_witnesses=parse_failure_witnesses,
    )
    name_map: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for bundles in config_bundles_by_path.values():
        check_deadline()
        for name, fields in bundles.items():
            check_deadline()
            key = tuple(sort_once(fields, source = 'src/gabion/analysis/dataflow_audit.py:13093'))
            name_map[key].add(name)
    for qual_name, fields in dataclass_registry.items():
        check_deadline()
        key = tuple(sort_once(fields, source = 'src/gabion/analysis/dataflow_audit.py:13097'))
        name_map[key].add(qual_name.split(".")[-1])
    return name_map


def _iter_dataclass_call_bundles(
    path: Path,
    *,
    project_root = None,
    symbol_table = None,
    dataclass_registry = None,
    parse_failure_witnesses: list[JSONObject],
) -> set[tuple[str, ...]]:
    """Return bundles promoted via @dataclass constructor calls."""
    check_deadline()
    outcome = _iter_dataclass_call_bundle_effects_impl(
        path,
        project_root=project_root,
        symbol_table=symbol_table,
        dataclass_registry=dataclass_registry,
        parse_failure_witnesses=parse_failure_witnesses,
    )
    parse_failure_witnesses.extend(outcome.witness_effects)
    return set(outcome.bundles)


@dataclass(frozen=True)
class BundleProjection:
    nodes: dict[NodeId, dict[str, str]]
    adj: dict[NodeId, set[NodeId]]
    bundle_map: dict[NodeId, tuple[str, ...]]
    bundle_counts: dict[tuple[str, ...], int]
    declared_global: set[tuple[str, ...]]
    declared_by_path: dict[str, set[tuple[str, ...]]]
    documented_by_path: dict[str, set[tuple[str, ...]]]
    root: Path
    path_lookup: dict[str, Path]


def _alt_input(alt: Alt, kind: str):
    for node_id in deadline_loop_iter(alt.inputs):
        if node_id.kind == kind:
            return node_id
    return None


def _paramset_key(forest: Forest, paramset_id: NodeId) -> tuple[str, ...]:
    node = forest.nodes.get(paramset_id)
    if node is not None:
        params = node.meta.get("params")
        if type(params) is list:
            return tuple(str(p) for p in cast(list[JSONValue], params))
    return tuple(str(p) for p in paramset_id.key)


def _bundle_projection_from_forest(
    forest: Forest,
    *,
    file_paths: list[Path],
) -> BundleProjection:
    check_deadline()
    nodes: dict[NodeId, dict[str, str]] = {}
    adj: dict[NodeId, set[NodeId]] = defaultdict(set)
    bundle_map: dict[NodeId, tuple[str, ...]] = {}
    bundle_counts: dict[tuple[str, ...], int] = defaultdict(int)
    for alt in forest.alts:
        check_deadline()
        if alt.kind == "SignatureBundle":
            site_id = _alt_input(alt, "FunctionSite")
            paramset_id = _alt_input(alt, "ParamSet")
            if site_id is not None and paramset_id is not None:
                site_node = forest.nodes.get(site_id)
                if site_node is not None:
                    path = str(site_node.meta.get("path", "?"))
                    qual = str(site_node.meta.get("qual", "?"))
                    nodes[site_id] = {
                        "kind": "fn",
                        "label": f"{path}:{qual}",
                        "path": path,
                        "qual": qual,
                    }
                    bundle_key = _paramset_key(forest, paramset_id)
                    nodes[paramset_id] = {
                        "kind": "bundle",
                        "label": ", ".join(bundle_key),
                    }
                    bundle_map[paramset_id] = bundle_key
                    adj[site_id].add(paramset_id)
                    adj[paramset_id].add(site_id)
                    bundle_counts[bundle_key] += 1

    declared_global: set[tuple[str, ...]] = set()
    declared_by_path: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    documented_by_path: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for alt in forest.alts:
        check_deadline()
        paramset_id = _alt_input(alt, "ParamSet")
        if paramset_id is not None:
            bundle_key = _paramset_key(forest, paramset_id)
            if alt.kind == "ConfigBundle":
                declared_global.add(bundle_key)
                path = str(alt.evidence.get("path") or "")
                if path:
                    declared_by_path[path].add(bundle_key)
            elif alt.kind in ("MarkerBundle", "DataclassCallBundle"):
                path = str(alt.evidence.get("path") or "")
                if path:
                    documented_by_path[path].add(bundle_key)

    if file_paths:
        root = Path(os.path.commonpath([str(p) for p in file_paths]))
    else:
        root = Path(".")
    path_lookup: dict[str, Path] = {}
    for path in file_paths:
        check_deadline()
        path_lookup.setdefault(path.name, path)
    return BundleProjection(
        nodes=nodes,
        adj=adj,
        bundle_map=bundle_map,
        bundle_counts=bundle_counts,
        declared_global=declared_global,
        declared_by_path=declared_by_path,
        documented_by_path=documented_by_path,
        root=root,
        path_lookup=path_lookup,
    )


def _bundle_site_index(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    bundle_sites_by_path: dict[Path, dict[str, list[list[JSONObject]]]],
) -> dict[tuple[str, str, tuple[str, ...]], list[list[JSONObject]]]:
    check_deadline()
    index: dict[tuple[str, str, tuple[str, ...]], list[list[JSONObject]]] = {}
    for path, groups in groups_by_path.items():
        check_deadline()
        fn_sites = bundle_sites_by_path.get(path, {})
        for fn_name, bundles in groups.items():
            check_deadline()
            sites = fn_sites.get(fn_name, [])
            for idx, bundle in enumerate(bundles):
                check_deadline()
                bundle_key = tuple(sort_once(bundle, source = 'src/gabion/analysis/dataflow_audit.py:13427'))
                entry = index.setdefault((path.name, fn_name, bundle_key), [])
                if idx < len(sites):
                    entry.append(sites[idx])
    return index


def _emit_dot(forest: Forest) -> str:
    check_deadline()
    if type(forest) is not Forest:
        raise RuntimeError("forest required for dataflow dot output")
    projection = _bundle_projection_from_forest(forest, file_paths=[])
    lines = [
        "digraph dataflow_grammar {",
        "  rankdir=LR;",
        "  node [fontsize=10];",
    ]
    for node_id, meta in projection.nodes.items():
        check_deadline()
        label = meta["label"].replace('"', "'")
        if meta["kind"] == "fn":
            lines.append(f'  {abs(hash(node_id.sort_key()))} [shape=box,label="{label}"];')
        else:
            lines.append(f'  {abs(hash(node_id.sort_key()))} [shape=ellipse,label="{label}"];')
    for src, targets in projection.adj.items():
        check_deadline()
        for dst in targets:
            check_deadline()
            if projection.nodes.get(src, {}).get("kind") == "fn":
                lines.append(
                    f"  {abs(hash(src.sort_key()))} -> {abs(hash(dst.sort_key()))};"
                )
    lines.append("}")
    return "\n".join(lines)


def _connected_components(
    nodes: dict[NodeId, dict[str, str]],
    adj: dict[NodeId, set[NodeId]],
) -> list[list[NodeId]]:
    check_deadline()
    seen: set[NodeId] = set()
    comps: list[list[NodeId]] = []
    for node in nodes:
        check_deadline()
        if node in seen:
            continue
        q: deque[NodeId] = deque([node])
        seen.add(node)
        comp: list[NodeId] = []
        while q:
            check_deadline()
            curr = q.popleft()
            comp.append(curr)
            for nxt in adj.get(curr, ()):
                check_deadline()
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        comps.append(sort_once(comp, key=lambda node_id: node_id.sort_key(), source = 'src/gabion/analysis/dataflow_audit.py:13486'))
    return comps


def _render_mermaid_component(
    nodes: dict[NodeId, dict[str, str]],
    bundle_map: dict[NodeId, tuple[str, ...]],
    bundle_counts: dict[tuple[str, ...], int],
    adj: dict[NodeId, set[NodeId]],
    component: list[NodeId],
    declared_global: set[tuple[str, ...]],
    declared_by_path: dict[str, set[tuple[str, ...]]],
    documented_by_path: dict[str, set[tuple[str, ...]]],
) -> tuple[str, str]:
    check_deadline()
    # dataflow-bundle: adj, declared_by_path, documented_by_path, nodes
    lines = ["```mermaid", "flowchart LR"]
    fn_nodes = [n for n in component if nodes[n]["kind"] == "fn"]
    bundle_nodes = [n for n in component if nodes[n]["kind"] == "bundle"]
    for n in fn_nodes:
        check_deadline()
        label = nodes[n]["label"].replace('"', "'")
        lines.append(f'  {abs(hash(n.sort_key()))}["{label}"]')
    for n in bundle_nodes:
        check_deadline()
        label = nodes[n]["label"].replace('"', "'")
        lines.append(f'  {abs(hash(n.sort_key()))}(({label}))')
    for n in component:
        check_deadline()
        for nxt in adj.get(n, ()):
            check_deadline()
            if nxt in component and nodes[n]["kind"] == "fn":
                lines.append(
                    f"  {abs(hash(n.sort_key()))} --> {abs(hash(nxt.sort_key()))}"
                )
    lines.append("  classDef fn fill:#cfe8ff,stroke:#2b6cb0,stroke-width:1px;")
    lines.append("  classDef bundle fill:#ffe9c6,stroke:#c05621,stroke-width:1px;")
    if fn_nodes:
        lines.append(
            "  class "
            + ",".join(str(abs(hash(n.sort_key()))) for n in fn_nodes)
            + " fn;"
        )
    if bundle_nodes:
        lines.append(
            "  class "
            + ",".join(str(abs(hash(n.sort_key()))) for n in bundle_nodes)
            + " bundle;"
        )
    lines.append("```")

    observed = [bundle_map[n] for n in bundle_nodes if n in bundle_map]
    component_paths: set[str] = set()
    for n in fn_nodes:
        check_deadline()
        component_paths.add(nodes[n]["path"])
    declared_local = set()
    documented = set()
    for path in component_paths:
        check_deadline()
        declared_local |= declared_by_path.get(path, set())
        documented |= documented_by_path.get(path, set())
    observed_norm = {tuple(sort_once(b, source = 'src/gabion/analysis/dataflow_audit.py:13548')) for b in observed}
    observed_only = (
        sort_once(observed_norm - declared_global, source = 'src/gabion/analysis/dataflow_audit.py:13550')
        if declared_global
        else sort_once(observed_norm, source = 'src/gabion/analysis/dataflow_audit.py:13552')
    )
    declared_only = sort_once(declared_local - observed_norm, source = 'src/gabion/analysis/dataflow_audit.py:13554')
    documented_only = sort_once(observed_norm & documented, source = 'src/gabion/analysis/dataflow_audit.py:13555')

    def _tier(bundle: tuple[str, ...]) -> str:
        count = bundle_counts.get(bundle, 1)
        if count > 1:
            return "tier-2"
        return "tier-3"

    summary_lines = [
        f"Functions: {len(fn_nodes)}",
        f"Observed bundles: {len(observed_norm)}",
    ]
    if not declared_local:
        summary_lines.append("Declared Config bundles: none found for this component.")
    if observed_only:
        summary_lines.append("Observed-only bundles (not declared in Configs):")
        for bundle in observed_only:
            check_deadline()
            tier = _tier(bundle)
            documented_flag = "documented" if bundle in documented else "undocumented"
            summary_lines.append(
                f"  - {', '.join(bundle)} ({tier}, {documented_flag})"
            )
    if documented_only:
        summary_lines.append(
            "Documented bundles (dataflow-bundle markers or local dataclass calls):"
        )
        summary_lines.extend(f"  - {', '.join(bundle)}" for bundle in documented_only)
    if declared_only:
        summary_lines.append("Declared Config bundles not observed in this component:")
        summary_lines.extend(f"  - {', '.join(bundle)}" for bundle in declared_only)
    summary = "\n".join(summary_lines)
    return "\n".join(lines), summary


def _render_component_callsite_evidence(
    *,
    component: list[NodeId],
    nodes: dict[NodeId, dict[str, str]],
    bundle_map: dict[NodeId, tuple[str, ...]],
    bundle_counts: dict[tuple[str, ...], int],
    adj: dict[NodeId, set[NodeId]],
    documented_by_path: dict[str, set[tuple[str, ...]]],
    declared_global: set[tuple[str, ...]],
    bundle_site_index: dict[tuple[str, str, tuple[str, ...]], list[list[JSONObject]]],
    root: Path,
    path_lookup: dict[str, Path],
    max_sites_per_bundle: int = 5,
) -> list[str]:
    """Render machine-actionable callsite evidence for undocumented bundles in a component.

    This assumes internal IDs and evidence payloads are well-formed; drift should
    fail loudly to force reification rather than silently degrade fidelity.
    """
    check_deadline()
    fn_nodes = [n for n in component if nodes[n]["kind"] == "fn"]
    bundle_nodes = [n for n in component if nodes[n]["kind"] == "bundle"]
    component_paths: set[str] = set()
    for n in fn_nodes:
        check_deadline()
        component_paths.add(nodes[n]["path"])
    documented: set[tuple[str, ...]] = set()
    for path in component_paths:
        check_deadline()
        documented |= documented_by_path.get(path, set())

    bundle_key_by_node: dict[NodeId, tuple[str, ...]] = {}
    for n in bundle_nodes:
        check_deadline()
        key = tuple(sort_once(bundle_map[n], source = 'src/gabion/analysis/dataflow_audit.py:13624'))
        bundle_key_by_node[n] = key

    # Keep output deterministic and review-friendly.
    ordered_nodes = sort_once(
        bundle_key_by_node,
        key=lambda node_id: (node_id.sort_key(), bundle_key_by_node.get(node_id, ())),
    source = 'src/gabion/analysis/dataflow_audit.py:13628')

    lines: list[str] = []
    for bundle_id in ordered_nodes:
        check_deadline()
        bundle_key = bundle_key_by_node[bundle_id]
        observed_only = (not declared_global) or (bundle_key not in declared_global)
        if not observed_only or bundle_key in documented:
            continue
        tier = "tier-2" if bundle_counts.get(bundle_key, 1) > 1 else "tier-3"
        adjacent_sites = [
            node_id
            for node_id in sort_once(adj.get(bundle_id, set()), key=lambda node: node.sort_key(), source = 'src/gabion/analysis/dataflow_audit.py:13643')
            if nodes.get(node_id, {}).get("kind") == "fn"
        ]
        for site_id in adjacent_sites:
            check_deadline()
            path_name = nodes[site_id]["path"]
            fn_name = nodes[site_id]["qual"]
            evidence_sets = bundle_site_index.get((path_name, fn_name, bundle_key), [])
            if not evidence_sets:
                continue
            path = path_lookup.get(path_name, Path(path_name))
            evidence_entries: list[JSONObject] = []
            for entry in evidence_sets:
                check_deadline()
                evidence_entries.extend(entry)
            for site in evidence_entries[:max_sites_per_bundle]:
                check_deadline()
                start_line, start_col, end_line, end_col = site["span"]
                loc = f"{start_line + 1}:{start_col + 1}-{end_line + 1}:{end_col + 1}"
                rel = _normalize_snapshot_path(path, root)
                callee = str(site.get("callee") or "")
                params = ", ".join(site.get("params") or [])
                slots = ", ".join(site.get("slots") or [])
                bundle_label = ", ".join(bundle_key)
                lines.append(
                    f"{rel}:{loc}: {fn_name} -> {callee} forwards {params} "
                    f"({tier}, undocumented bundle: {bundle_label}; slots: {slots})"
                )
    return lines


_REPORT_SECTION_MARKER_PREFIX = "<!-- report-section:"
_REPORT_SECTION_MARKER_SUFFIX = "-->"


def _report_section_marker(section_id: str) -> str:
    return f"{_REPORT_SECTION_MARKER_PREFIX}{section_id}{_REPORT_SECTION_MARKER_SUFFIX}"


def _parse_report_section_marker(line: str):
    text = line.strip()
    if not text.startswith(_REPORT_SECTION_MARKER_PREFIX):
        return None
    if not text.endswith(_REPORT_SECTION_MARKER_SUFFIX):
        return None
    section_id = text[
        len(_REPORT_SECTION_MARKER_PREFIX) : -len(_REPORT_SECTION_MARKER_SUFFIX)
    ].strip()
    if not section_id:
        return None
    return section_id


def extract_report_sections(markdown: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    active_section_id: OptionalString = None
    for raw_line in markdown.splitlines():
        check_deadline()
        section_id = _parse_report_section_marker(raw_line)
        if section_id is not None:
            active_section_id = section_id
            sections.setdefault(section_id, [])
        elif active_section_id is not None:
            sections[active_section_id].append(raw_line)
    return sections


def detect_report_section_extinctions(
    *,
    previous_markdown: str,
    current_markdown: str,
) -> tuple[str, ...]:
    check_deadline()
    previous_sections = tuple(extract_report_sections(previous_markdown))
    current_sections = tuple(extract_report_sections(current_markdown))
    return detect_report_section_extinction(
        previous_sections=previous_sections,
        current_sections=current_sections,
    )


from gabion.analysis.dataflow_reporting import emit_report as _emit_report


def _infer_root(groups_by_path: dict[Path, dict[str, list[set[str]]]]) -> Path:
    if groups_by_path:
        common = os.path.commonpath([str(p) for p in groups_by_path])
        return Path(common)
    return Path(".")


def _normalize_snapshot_path(path: Path, root) -> str:
    if root is not None:
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return str(path)


def load_structure_snapshot(path: Path) -> JSONObject:
    try:
        data = load_json(path)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid snapshot JSON: {path}") from exc
    except ValueError as exc:
        raise ValueError(f"Snapshot must be a JSON object: {path}") from exc
    return {str(key): data[key] for key in data}


def compute_structure_metrics(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    *,
    forest: Forest,
) -> JSONObject:
    check_deadline()
    file_count = len(groups_by_path)
    function_count = sum(len(groups) for groups in groups_by_path.values())
    bundle_sizes: list[int] = []
    for groups in groups_by_path.values():
        check_deadline()
        for bundles in groups.values():
            check_deadline()
            for bundle in bundles:
                check_deadline()
                bundle_sizes.append(len(bundle))
    bundle_count = len(bundle_sizes)
    mean_bundle_size = (sum(bundle_sizes) / bundle_count) if bundle_count else 0.0
    max_bundle_size = max(bundle_sizes) if bundle_sizes else 0
    size_histogram: dict[int, int] = defaultdict(int)
    for size in bundle_sizes:
        check_deadline()
        size_histogram[size] += 1
    metrics: JSONObject = {
        "files": file_count,
        "functions": function_count,
        "bundles": bundle_count,
        "mean_bundle_size": mean_bundle_size,
        "max_bundle_size": max_bundle_size,
        # JSON object keys are strings; use explicit conversion for stability.
        "bundle_size_histogram": {
            str(size): count for size, count in sort_once(size_histogram.items(), source = 'src/gabion/analysis/dataflow_audit.py:14196')
        },
    }
    metrics["forest_signature"] = build_forest_signature(forest)
    return metrics


def _partial_forest_signature_metadata(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    *,
    basis: str = "bundles_only",
) -> JSONObject:
    return {
        "forest_signature": build_forest_signature_from_groups(groups_by_path),
        "forest_signature_partial": True,
        "forest_signature_basis": basis,
    }


def _copy_forest_signature_metadata(
    payload: JSONObject,
    snapshot: JSONObject,
    *,
    prefix: str = "",
) -> None:
    signature = snapshot.get("forest_signature")
    if signature is not None:
        payload[f"{prefix}forest_signature"] = signature
    partial = snapshot.get("forest_signature_partial")
    if partial is not None:
        payload[f"{prefix}forest_signature_partial"] = partial
    basis = snapshot.get("forest_signature_basis")
    if basis is not None:
        payload[f"{prefix}forest_signature_basis"] = basis
    if signature is None:
        payload[f"{prefix}forest_signature_partial"] = True
        if basis is None:
            payload[f"{prefix}forest_signature_basis"] = "missing"


def render_structure_snapshot(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    *,
    project_root = None,
    forest: Forest,
    forest_spec = None,
    invariant_propositions = None,
) -> JSONObject:
    check_deadline()
    root = project_root or _infer_root(groups_by_path)
    invariant_map: dict[tuple[str, str], list[InvariantProposition]] = {}
    if invariant_propositions:
        for prop in invariant_propositions:
            check_deadline()
            if not prop.scope or ":" not in prop.scope:
                continue
            scope_path, fn_name = prop.scope.rsplit(":", 1)
            invariant_map.setdefault((scope_path, fn_name), []).append(prop)
    files: list[JSONObject] = []
    for path in sort_once(
        groups_by_path, key=lambda p: _normalize_snapshot_path(p, root), 
    source = 'src/gabion/analysis/dataflow_audit.py:14255'):
        check_deadline()
        groups = groups_by_path[path]
        functions: list[JSONObject] = []
        path_key = _normalize_snapshot_path(path, root)
        for fn_name in sort_once(groups, source = 'src/gabion/analysis/dataflow_audit.py:14262'):
            check_deadline()
            bundles = groups[fn_name]
            normalized = [sort_once(bundle, source = 'src/gabion/analysis/dataflow_audit.py:14265') for bundle in bundles]
            normalized = sort_once(
                normalized,
                source="snapshot_from_groups_by_path.functions.normalized",
                # Primary by bundle length then lexicalized bundle tuple.
                key=lambda bundle: (len(bundle), bundle),
            )
            entry: JSONObject = {"name": fn_name, "bundles": normalized}
            invariants = invariant_map.get((path_key, fn_name))
            if invariants:
                entry["invariants"] = [
                    prop.as_dict()
                    for prop in sort_once(
                        invariants,
                        key=lambda prop: (
                            prop.form,
                            prop.terms,
                            prop.source or "",
                            prop.scope or "",
                        ),
                    source = 'src/gabion/analysis/dataflow_audit.py:14277')
                ]
            functions.append(entry)
        files.append({"path": _normalize_snapshot_path(path, root), "functions": functions})
    snapshot: JSONObject = {
        "format_version": 1,
        "root": str(root) if root is not None else None,
        "files": files,
    }
    spec = forest_spec or default_forest_spec(include_bundle_forest=True)
    snapshot.update(forest_spec_metadata(spec))
    snapshot["forest_signature"] = build_forest_signature(forest)
    return snapshot


@dataclass(frozen=True)
class DecisionSnapshotSurfaces:
    decision_surfaces: list[str]
    value_decision_surfaces: list[str]


@dataclass(frozen=True)
class StructureSnapshotDiffRequest:
    baseline_path: Path
    current_path: Path

# dataflow-bundle: surfaces
def render_decision_snapshot(
    *,
    surfaces: DecisionSnapshotSurfaces,
    project_root = None,
    forest: Forest,
    forest_spec = None,
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    pattern_schema_instances = None,
) -> JSONObject:
    if type(forest) is not Forest:
        never("decision snapshot requires forest carrier")
    instances = pattern_schema_instances
    if instances is None:
        instances = _pattern_schema_matches(
            groups_by_path=groups_by_path,
            include_execution=True,
        )
    schema_instances, schema_residue = _pattern_schema_snapshot_entries(instances)
    decision_tables = build_decision_tables(
        decision_surfaces=surfaces.decision_surfaces,
        value_decision_surfaces=surfaces.value_decision_surfaces,
    )
    decision_bundles = detect_repeated_guard_bundles(decision_tables)
    decision_protocol_violations = enforce_decision_protocol_contracts(
        decision_tables=decision_tables,
        decision_bundles=decision_bundles,
    )
    snapshot: JSONObject = {
        "format_version": 1,
        "root": str(project_root) if project_root is not None else None,
        "decision_surfaces": sort_once(surfaces.decision_surfaces, source = 'src/gabion/analysis/dataflow_audit.py:14342'),
        "value_decision_surfaces": sort_once(surfaces.value_decision_surfaces, source = 'src/gabion/analysis/dataflow_audit.py:14343'),
        "pattern_schema_instances": schema_instances,
        "pattern_schema_residue": schema_residue,
        "decision_tables": decision_tables,
        "decision_bundles": decision_bundles,
        "decision_protocol_violations": decision_protocol_violations,
        "summary": {
            "decision_surfaces": len(surfaces.decision_surfaces),
            "value_decision_surfaces": len(surfaces.value_decision_surfaces),
            "pattern_schema_instances": len(schema_instances),
            "pattern_schema_residue": len(schema_residue),
            "decision_tables": len(decision_tables),
            "decision_bundles": len(decision_bundles),
            "decision_protocol_violations": len(decision_protocol_violations),
        },
    }
    snapshot["forest"] = forest.to_json()
    snapshot["forest_signature"] = build_forest_signature(forest)
    spec = forest_spec or default_forest_spec(
        include_bundle_forest=True,
        include_decision_surfaces=True,
        include_value_decision_surfaces=True,
    )
    snapshot.update(forest_spec_metadata(spec))
    return snapshot


def load_decision_snapshot(path: Path) -> JSONObject:
    try:
        data = load_json(path)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid decision snapshot JSON: {path}") from exc
    except ValueError as exc:
        raise ValueError(f"Decision snapshot must be a JSON object: {path}") from exc
    return {str(key): data[key] for key in data}


def diff_decision_snapshots(
    baseline_snapshot: JSONObject,
    current_snapshot: JSONObject,
) -> JSONObject:
    base_decisions = set(baseline_snapshot.get("decision_surfaces") or [])
    curr_decisions = set(current_snapshot.get("decision_surfaces") or [])
    base_value = set(baseline_snapshot.get("value_decision_surfaces") or [])
    curr_value = set(current_snapshot.get("value_decision_surfaces") or [])
    diff: JSONObject = {
        "format_version": 1,
        "baseline_root": baseline_snapshot.get("root"),
        "current_root": current_snapshot.get("root"),
        "decision_surfaces": {
            "added": sort_once(curr_decisions - base_decisions, source = 'src/gabion/analysis/dataflow_audit.py:14393'),
            "removed": sort_once(base_decisions - curr_decisions, source = 'src/gabion/analysis/dataflow_audit.py:14394'),
        },
        "value_decision_surfaces": {
            "added": sort_once(curr_value - base_value, source = 'src/gabion/analysis/dataflow_audit.py:14397'),
            "removed": sort_once(base_value - curr_value, source = 'src/gabion/analysis/dataflow_audit.py:14398'),
        },
    }
    _copy_forest_signature_metadata(diff, baseline_snapshot, prefix="baseline_")
    _copy_forest_signature_metadata(diff, current_snapshot, prefix="current_")
    return diff


def _bundle_counts_from_snapshot(snapshot: JSONObject) -> dict[tuple[str, ...], int]:
    check_deadline()
    _forbid_adhoc_bundle_discovery("_bundle_counts_from_snapshot")
    counts: dict[tuple[str, ...], int] = defaultdict(int)
    files = snapshot.get("files") or []
    for file_entry in files:
        check_deadline()
        if type(file_entry) is not dict:
            continue
        file_entry_obj = cast(JSONObject, file_entry)
        functions = file_entry_obj.get("functions") or []
        for fn_entry in functions:
            check_deadline()
            if type(fn_entry) is not dict:
                continue
            fn_entry_obj = cast(JSONObject, fn_entry)
            bundles = fn_entry_obj.get("bundles") or []
            for bundle in bundles:
                check_deadline()
                if type(bundle) is not list:
                    continue
                counts[tuple(cast(list[object], bundle))] += 1
    return counts


# dataflow-bundle: baseline_snapshot, current_snapshot
def diff_structure_snapshots(
    baseline_snapshot: JSONObject,
    current_snapshot: JSONObject,
) -> JSONObject:
    check_deadline()
    baseline_counts = _bundle_counts_from_snapshot(baseline_snapshot)
    current_counts = _bundle_counts_from_snapshot(current_snapshot)
    all_bundles = sort_once(
        set(baseline_counts) | set(current_counts),
        key=lambda bundle: (len(bundle), list(bundle)),
    source = 'src/gabion/analysis/dataflow_audit.py:14437')
    added: list[JSONObject] = []
    removed: list[JSONObject] = []
    changed: list[JSONObject] = []
    for bundle in all_bundles:
        check_deadline()
        before = baseline_counts.get(bundle, 0)
        after = current_counts.get(bundle, 0)
        entry = {
            "bundle": list(bundle),
            "before": before,
            "after": after,
            "delta": after - before,
        }
        if before == 0:
            added.append(entry)
        elif after == 0:
            removed.append(entry)
        elif before != after:
            changed.append(entry)
    diff: JSONObject = {
        "format_version": 1,
        "baseline_root": baseline_snapshot.get("root"),
        "current_root": current_snapshot.get("root"),
        "added": added,
        "removed": removed,
        "changed": changed,
        "summary": {
            "added": len(added),
            "removed": len(removed),
            "changed": len(changed),
            "baseline_total": sum(baseline_counts.values()),
            "current_total": sum(current_counts.values()),
        },
    }
    _copy_forest_signature_metadata(diff, baseline_snapshot, prefix="baseline_")
    _copy_forest_signature_metadata(diff, current_snapshot, prefix="current_")
    return diff


def diff_structure_snapshot_files(
    request: StructureSnapshotDiffRequest,
) -> JSONObject:
    # dataflow-bundle: request
    baseline = load_structure_snapshot(request.baseline_path)
    current = load_structure_snapshot(request.current_path)
    return diff_structure_snapshots(baseline, current)


def compute_structure_reuse(
    snapshot: JSONObject,
    *,
    min_count: int = 2,
    hash_fn = None,
) -> JSONObject:
    check_deadline()
    if min_count < 2:
        min_count = 2
    files = snapshot.get("files") or []
    root_value = snapshot.get("root")
    root_path = Path(root_value) if type(root_value) is str else None
    bundle_name_map: dict[tuple[str, ...], set[str]] = {}
    if root_path is not None and root_path.exists():
        bundle_name_map = _bundle_name_registry(root_path)
    reuse_map: dict[str, JSONObject] = {}
    warnings: list[str] = []

    def _hash_node(kind: str, value, child_hashes: list[str]) -> str:
        structure_class = build_structure_class(
            kind=kind,
            value=value,
            child_hashes=sort_once(
                child_hashes,
                source="compute_structure_reuse._hash_node.child_hashes",
            ),
        )
        return structure_class.digest()

    hasher = cast(Callable[[str, object, list[str]], str], hash_fn) if callable(hash_fn) else _hash_node

    def _record(
        *,
        node_hash: str,
        kind: str,
        location: str,
        value = None,
        child_count = -1,
    ) -> None:
        entry = reuse_map.get(node_hash)
        if entry is None:
            structure_class = build_structure_class(
                kind=kind,
                value=value,
                child_hashes=(),
            )
            entry = {
                "hash": node_hash,
                "kind": kind,
                "count": 0,
                "locations": [],
                "aspf_structure_class": structure_class_payload(structure_class),
            }
            if value is not None:
                entry["value"] = value
            if child_count >= 0:
                entry["child_count"] = child_count
            reuse_map[node_hash] = entry
        entry["count"] += 1
        entry["locations"].append(location)

    file_hashes: list[str] = []
    for file_entry in files:
        check_deadline()
        if type(file_entry) is dict:
            file_entry_obj = cast(JSONObject, file_entry)
            file_path = file_entry_obj.get("path")
            if type(file_path) is str:
                function_hashes: list[str] = []
                functions = file_entry_obj.get("functions") or []
                for fn_entry in functions:
                    check_deadline()
                    if type(fn_entry) is dict:
                        fn_entry_obj = cast(JSONObject, fn_entry)
                        fn_name = fn_entry_obj.get("name")
                        if type(fn_name) is str:
                            bundle_hashes: list[str] = []
                            bundles = fn_entry_obj.get("bundles") or []
                            for bundle in bundles:
                                check_deadline()
                                if type(bundle) is list:
                                    bundle_items = cast(list[object], bundle)
                                    normalized = tuple(
                                        sort_once(
                                            (str(item) for item in bundle_items),
                                            source="src/gabion/analysis/dataflow_audit.py:14567",
                                        )
                                    )
                                    bundle_hash = hasher("bundle", normalized, [])
                                    bundle_hashes.append(bundle_hash)
                                    _record(
                                        node_hash=bundle_hash,
                                        kind="bundle",
                                        location=f"{file_path}::{fn_name}::bundle:{','.join(normalized)}",
                                        value=list(normalized),
                                    )
                            fn_hash = hasher("function", None, bundle_hashes)
                            function_hashes.append(fn_hash)
                            _record(
                                node_hash=fn_hash,
                                kind="function",
                                location=f"{file_path}::{fn_name}",
                                child_count=len(bundle_hashes),
                            )
                file_hash = hasher("file", None, function_hashes)
                file_hashes.append(file_hash)
                _record(node_hash=file_hash, kind="file", location=f"{file_path}")

    root_hash = hasher("root", None, file_hashes)
    _record(
        node_hash=root_hash,
        kind="root",
        location="root",
        child_count=len(file_hashes),
    )

    reused = [
        entry
        for entry in reuse_map.values()
        if type(entry.get("count")) is int and entry["count"] >= min_count
    ]
    reused = sort_once(
        reused,
        source="extract_rewrite_plan_candidates.reused",
        # Lexical kind, descending count, lexical hash for deterministic suggestions.
        key=lambda entry: (
            entry.get("kind", ""),
            -int(entry.get("count", 0)),
            entry.get("hash", ""),
        ),
    )
    suggested: list[JSONObject] = []
    replacement_map: dict[str, list[JSONObject]] = {}
    for entry in reused:
        check_deadline()
        kind = entry.get("kind")
        if kind in {"bundle", "function"}:
            count = int(entry.get("count", 0))
            hash_value = entry.get("hash")
            if type(hash_value) is str and hash_value:
                suggestion = {
                    "hash": hash_value,
                    "kind": kind,
                    "count": count,
                    "suggested_name": f"_gabion_{kind}_lemma_{hash_value[:8]}",
                    "locations": entry.get("locations", []),
                }
                if "value" in entry:
                    suggestion["value"] = entry.get("value")
                if "child_count" in entry:
                    suggestion["child_count"] = entry.get("child_count")
                if kind == "bundle" and "value" in entry:
                    value = entry.get("value")
                    key = tuple(
                        sort_once(
                            (str(item) for item in cast(list[object], value)),
                            source="src/gabion/analysis/dataflow_audit.py:14635",
                        )
                    )
                    name_candidates = bundle_name_map.get(key)
                    if name_candidates:
                        sorted_names = sort_once(name_candidates, source = 'src/gabion/analysis/dataflow_audit.py:14638')
                        if len(sorted_names) == 1:
                            suggestion["suggested_name"] = sorted_names[0]
                            suggestion["name_source"] = "declared_bundle"
                        else:
                            suggestion["name_candidates"] = sorted_names
                    else:
                        warnings.append(
                            f"Missing declared bundle name for {list(key)}"
                        )
                suggested.append(suggestion)
    replacement_map = _build_reuse_replacement_map(suggested)
    reuse_payload: JSONObject = {
        "format_version": 1,
        "min_count": min_count,
        "reused": reused,
        "suggested_lemmas": suggested,
        "replacement_map": replacement_map,
        "warnings": warnings,
    }
    _copy_forest_signature_metadata(reuse_payload, snapshot)
    return reuse_payload


def _build_reuse_replacement_map(
    suggested: list[JSONObject],
) -> dict[str, list[JSONObject]]:
    check_deadline()
    replacement_map: dict[str, list[JSONObject]] = {}
    for suggestion in suggested:
        check_deadline()
        locations = suggestion.get("locations") or []
        location_entries = sequence_or_none(locations)
        if location_entries is not None:
            for location in location_entries:
                check_deadline()
                if type(location) is str:
                    replacement_map.setdefault(location, []).append(
                        {
                            "kind": suggestion.get("kind"),
                            "hash": suggestion.get("hash"),
                            "suggested_name": suggestion.get("suggested_name"),
                        }
                    )
    return replacement_map


def render_reuse_lemma_stubs(reuse: JSONObject) -> str:
    check_deadline()
    suggested = reuse.get("suggested_lemmas") or []
    lines = [
        "# Generated by gabion structure-reuse",
        "# TODO: replace stubs with actual lemma definitions.",
        "",
    ]
    if not suggested:
        lines.append("# No lemma suggestions available.")
        lines.append("")
        return "\n".join(lines)
    suggested_entries = [
        mapping_or_empty(raw_entry)
        for raw_entry in suggested
    ]
    for entry in sort_once(
        suggested_entries,
        key=lambda e: (str(e.get("kind", "")), str(e.get("suggested_name", ""))),
    source = 'src/gabion/analysis/dataflow_audit.py:14698'):
        check_deadline()
        name = entry.get("suggested_name")
        if type(name) is str and name:
            kind = entry.get("kind", "lemma")
            count = entry.get("count", 0)
            value = entry.get("value")
            child_count = entry.get("child_count")
            lines.append(f"def {name}() -> None:")
            lines.append('    """Auto-generated lemma stub."""')
            lines.append(f"    # kind: {kind}")
            lines.append(f"    # count: {count}")
            if value is not None:
                lines.append(f"    # value: {value}")
            if child_count is not None:
                lines.append(f"    # child_count: {child_count}")
            lines.append("    ...")
            lines.append("")
    return "\n".join(lines)


_ANALYSIS_COLLECTION_RESUME_FORMAT_VERSION = 2
_FILE_SCAN_PROGRESS_EMIT_INTERVAL = 1
_BUNDLE_FOREST_PROGRESS_EMIT_INTERVAL = 1
_COLLECTION_PROGRESS_EMIT_INTERVAL = 1
_ANALYSIS_INDEX_PROGRESS_EMIT_INTERVAL = 1
_DEADLINE_OBLIGATIONS_PROGRESS_EMIT_INTERVAL = 1
_PROGRESS_EMIT_MIN_INTERVAL_SECONDS = 1.0


@dataclass(frozen=True)
class _PartialWorkerCarrierOutput:
    violations: tuple[str, ...] = ()
    witnesses: tuple[JSONObject, ...] = ()
    deltas: tuple[JSONObject, ...] = ()
    snapshots: tuple[JSONObject, ...] = ()


def _canonical_json_bytes(value: JSONValue) -> str:
    return json.dumps(value, sort_keys=False, separators=(",", ":"), ensure_ascii=False)


def _path_key_from_violation(violation: str):
    path_prefix, separator, _ = violation.partition(":")
    if separator and path_prefix.endswith(".py"):
        return path_prefix
    return None


def _path_key_from_payload(payload: Mapping[str, JSONValue]):
    for key in deadline_loop_iter(
        ("path", "module_path", "file", "baseline_path", "current_path")
    ):
        raw_value = payload.get(key)
        if type(raw_value) is str and raw_value:
            return raw_value
    return None


def _canonicalize_worker_violations(
    violations: Iterable[str],
    *,
    source: str,
) -> list[str]:
    by_path: dict[str, dict[str, str]] = defaultdict(dict)
    no_path: dict[str, str] = {}
    for violation in violations:
        check_deadline()
        canonical = _canonical_json_bytes(violation)
        path_key = _path_key_from_violation(violation)
        if path_key is None:
            no_path.setdefault(canonical, violation)
            continue
        by_path[path_key].setdefault(canonical, violation)
    ordered_paths = _iter_monotonic_paths(
        sort_once(
            (Path(path_key) for path_key in by_path),
            source=f"{source}.paths",
            key=str,
        ),
        source=f"{source}.paths_monotonic",
    )
    ordered: list[str] = []
    for path in ordered_paths:
        check_deadline()
        path_key = str(path)
        entries = by_path.get(path_key, {})
        for canonical in sort_once(
            entries,
            source=f"{source}.path_entries",
        ):
            check_deadline()
            ordered.append(entries[canonical])
    for canonical in sort_once(
        no_path,
        source=f"{source}.no_path_entries",
    ):
        check_deadline()
        ordered.append(no_path[canonical])
    return ordered


def _canonicalize_worker_structured_entries(
    entries: Iterable[JSONObject],
    *,
    source: str,
) -> list[JSONObject]:
    by_path: dict[str, dict[str, JSONObject]] = defaultdict(dict)
    no_path: dict[str, JSONObject] = {}
    for entry in entries:
        check_deadline()
        canonical = _canonical_json_bytes(entry)
        path_key = _path_key_from_payload(entry)
        if path_key is None:
            no_path.setdefault(canonical, entry)
            continue
        by_path[path_key].setdefault(canonical, entry)
    ordered_paths = _iter_monotonic_paths(
        sort_once(
            (Path(path_key) for path_key in by_path),
            source=f"{source}.paths",
            key=str,
        ),
        source=f"{source}.paths_monotonic",
    )
    ordered: list[JSONObject] = []
    for path in ordered_paths:
        check_deadline()
        path_key = str(path)
        path_entries = by_path.get(path_key, {})
        for canonical in sort_once(
            path_entries,
            source=f"{source}.path_entries",
        ):
            check_deadline()
            ordered.append(path_entries[canonical])
    for canonical in sort_once(
        no_path,
        source=f"{source}.no_path_entries",
    ):
        check_deadline()
        ordered.append(no_path[canonical])
    return ordered


def _merge_worker_carriers(
    partials: Sequence[_PartialWorkerCarrierOutput],
) -> _PartialWorkerCarrierOutput:
    combined_violations: list[str] = []
    combined_witnesses: list[JSONObject] = []
    combined_deltas: list[JSONObject] = []
    combined_snapshots: list[JSONObject] = []
    for partial in partials:
        check_deadline()
        combined_violations.extend(partial.violations)
        combined_witnesses.extend(partial.witnesses)
        combined_deltas.extend(partial.deltas)
        combined_snapshots.extend(partial.snapshots)
    return _PartialWorkerCarrierOutput(
        violations=tuple(
            _canonicalize_worker_violations(
                combined_violations,
                source="_merge_worker_carriers.violations",
            )
        ),
        witnesses=tuple(
            _canonicalize_worker_structured_entries(
                combined_witnesses,
                source="_merge_worker_carriers.witnesses",
            )
        ),
        deltas=tuple(
            _canonicalize_worker_structured_entries(
                combined_deltas,
                source="_merge_worker_carriers.deltas",
            )
        ),
        snapshots=tuple(
            _canonicalize_worker_structured_entries(
                combined_snapshots,
                source="_merge_worker_carriers.snapshots",
            )
        ),
    )


def _analysis_collection_resume_path_key(path: Path) -> str:
    return str(path)


def _iter_monotonic_paths(
    paths: Iterable[Path],
    *,
    source: str,
) -> list[Path]:
    ordered: list[Path] = []
    previous_path_key: OptionalString = None
    for path in paths:
        check_deadline()
        path_key = _analysis_collection_resume_path_key(path)
        if previous_path_key is not None and previous_path_key > path_key:
            never(
                "path order regression",
                source=source,
                previous_path=previous_path_key,
                current_path=path_key,
            )
        previous_path_key = path_key
        ordered.append(path)
    return ordered


def _serialize_param_use(value: ParamUse) -> JSONObject:
    return {
        "direct_forward": [
            [callee, slot] for callee, slot in sort_once(value.direct_forward, source = 'src/gabion/analysis/dataflow_audit.py:14917')
        ],
        "non_forward": bool(value.non_forward),
        "current_aliases": sort_once(value.current_aliases, source = 'src/gabion/analysis/dataflow_audit.py:14920'),
        "forward_sites": [
            {
                "callee": callee,
                "slot": slot,
                "spans": [list(span) for span in sort_once(spans, source = 'src/gabion/analysis/dataflow_audit.py:14925')],
            }
            for (callee, slot), spans in sort_once(value.forward_sites.items(), source = 'src/gabion/analysis/dataflow_audit.py:14927')
        ],
        "unknown_key_carrier": bool(value.unknown_key_carrier),
        "unknown_key_sites": [list(span) for span in sort_once(value.unknown_key_sites, source = 'src/gabion/analysis/dataflow_audit.py:14930')],
    }


def _deserialize_param_use(payload: Mapping[str, JSONValue]) -> ParamUse:
    direct_forward = str_pair_set_from_sequence(payload.get("direct_forward"))
    current_aliases = str_set_from_sequence(payload.get("current_aliases"))
    forward_sites: dict[tuple[str, str], set[tuple[int, int, int, int]]] = {}
    for raw_entry in sequence_or_none(payload.get("forward_sites")) or ():
        check_deadline()
        entry = mapping_or_none(raw_entry)
        if entry is not None:
            callee = entry.get("callee")
            slot = entry.get("slot")
            if type(callee) is str and type(slot) is str:
                span_set: set[tuple[int, int, int, int]] = set()
                for raw_span in sequence_or_none(entry.get("spans")) or ():
                    check_deadline()
                    span = int_tuple4_or_none(raw_span)
                    if span is not None:
                        span_set.add(span)
                forward_sites[(callee, slot)] = span_set
    non_forward = bool(payload.get("non_forward"))
    unknown_key_carrier = bool(payload.get("unknown_key_carrier"))
    unknown_key_sites: set[tuple[int, int, int, int]] = set()
    for raw_span in sequence_or_none(payload.get("unknown_key_sites")) or ():
        check_deadline()
        span = int_tuple4_or_none(raw_span)
        if span is not None:
            unknown_key_sites.add(span)
    return ParamUse(
        direct_forward=direct_forward,
        non_forward=non_forward,
        current_aliases=current_aliases,
        forward_sites=forward_sites,
        unknown_key_carrier=unknown_key_carrier,
        unknown_key_sites=unknown_key_sites,
    )


def _serialize_param_use_map(
    use_map: Mapping[str, ParamUse],
) -> JSONObject:
    payload: JSONObject = {}
    for param_name in sort_once(use_map, source = 'src/gabion/analysis/dataflow_audit.py:14978'):
        check_deadline()
        payload[param_name] = _serialize_param_use(use_map[param_name])
    return payload


def _deserialize_param_use_map(
    payload: Mapping[str, JSONValue],
) -> dict[str, ParamUse]:
    use_map: dict[str, ParamUse] = {}
    for param_name, raw_value in payload.items():
        check_deadline()
        raw_mapping = mapping_or_none(raw_value)
        if type(param_name) is str and raw_mapping is not None:
            use_map[param_name] = _deserialize_param_use(raw_mapping)
    return use_map


def _serialize_call_args(call: CallArgs) -> JSONObject:
    payload: JSONObject = {
        "callee": call.callee,
        "pos_map": {key: call.pos_map[key] for key in sort_once(call.pos_map, source = 'src/gabion/analysis/dataflow_audit.py:14999')},
        "kw_map": {key: call.kw_map[key] for key in sort_once(call.kw_map, source = 'src/gabion/analysis/dataflow_audit.py:15000')},
        "const_pos": {key: call.const_pos[key] for key in sort_once(call.const_pos, source = 'src/gabion/analysis/dataflow_audit.py:15001')},
        "const_kw": {key: call.const_kw[key] for key in sort_once(call.const_kw, source = 'src/gabion/analysis/dataflow_audit.py:15002')},
        "non_const_pos": sort_once(call.non_const_pos, source = 'src/gabion/analysis/dataflow_audit.py:15003'),
        "non_const_kw": sort_once(call.non_const_kw, source = 'src/gabion/analysis/dataflow_audit.py:15004'),
        "star_pos": [[idx, name] for idx, name in call.star_pos],
        "star_kw": list(call.star_kw),
        "is_test": call.is_test,
        "callable_kind": call.callable_kind,
        "callable_source": call.callable_source,
    }
    if call.span is not None:
        payload["span"] = list(call.span)
    return payload


def _deserialize_call_args(payload: Mapping[str, JSONValue]):
    callee = payload.get("callee")
    if type(callee) is not str:
        return None
    star_pos = int_str_pairs_from_sequence(payload.get("star_pos"))
    span = int_tuple4_or_none(payload.get("span"))

    return CallArgs(
        callee=callee,
        pos_map=str_map_from_mapping(payload.get("pos_map")),
        kw_map=str_map_from_mapping(payload.get("kw_map")),
        const_pos=str_map_from_mapping(payload.get("const_pos")),
        const_kw=str_map_from_mapping(payload.get("const_kw")),
        non_const_pos=str_set_from_sequence(payload.get("non_const_pos")),
        non_const_kw=str_set_from_sequence(payload.get("non_const_kw")),
        star_pos=star_pos,
        star_kw=sort_once(str_set_from_sequence(payload.get("star_kw")), source = 'src/gabion/analysis/dataflow_audit.py:15032'),
        is_test=bool(payload.get("is_test")),
        span=span,
        callable_kind=str(payload.get("callable_kind") or "function"),
        callable_source=str(payload.get("callable_source") or "symbol"),
    )


def _serialize_call_args_list(call_args: Sequence[CallArgs]) -> list[JSONObject]:
    return [_serialize_call_args(call) for call in call_args]


def _deserialize_call_args_list(payload: Sequence[JSONValue]) -> list[CallArgs]:
    call_args: list[CallArgs] = []
    for raw_entry in payload:
        check_deadline()
        entry_mapping = mapping_or_none(raw_entry)
        if entry_mapping is not None:
            call = _deserialize_call_args(entry_mapping)
            if call is not None:
                call_args.append(call)
    return call_args


def _serialize_function_info_for_resume(info: FunctionInfo) -> JSONObject:
    payload: JSONObject = {
        "name": info.name,
        "qual": info.qual,
        "path": str(info.path),
        "params": list(info.params),
        "annots": {param: info.annots[param] for param in sort_once(info.annots, source = 'src/gabion/analysis/dataflow_audit.py:15062')},
        "calls": _serialize_call_args_list(info.calls),
        "unused_params": sort_once(info.unused_params, source = 'src/gabion/analysis/dataflow_audit.py:15064'),
        "unknown_key_carriers": sort_once(info.unknown_key_carriers, source = 'src/gabion/analysis/dataflow_audit.py:15065'),
        "defaults": sort_once(info.defaults, source = 'src/gabion/analysis/dataflow_audit.py:15066'),
        "transparent": bool(info.transparent),
        "class_name": info.class_name,
        "scope": list(info.scope),
        "lexical_scope": list(info.lexical_scope),
        "decision_params": sort_once(info.decision_params, source = 'src/gabion/analysis/dataflow_audit.py:15071'),
        "decision_surface_reasons": {
            param: sort_once(
                info.decision_surface_reasons.get(param, set()),
                source="_serialize_function_info_for_resume.decision_surface_reasons",
            )
            for param in sort_once(
                info.decision_surface_reasons,
                source="_serialize_function_info_for_resume.decision_surface_reason_keys",
            )
        },
        "value_decision_params": sort_once(info.value_decision_params, source = 'src/gabion/analysis/dataflow_audit.py:15082'),
        "value_decision_reasons": sort_once(info.value_decision_reasons, source = 'src/gabion/analysis/dataflow_audit.py:15083'),
        "positional_params": list(info.positional_params),
        "kwonly_params": list(info.kwonly_params),
        "vararg": info.vararg,
        "kwarg": info.kwarg,
        "param_spans": {
            param: [int(value) for value in info.param_spans[param]]
            for param in sort_once(info.param_spans, source = 'src/gabion/analysis/dataflow_audit.py:15090')
        },
    }
    if info.function_span is not None:
        payload["function_span"] = [int(value) for value in info.function_span]
    return payload


def _deserialize_function_info_for_resume(
    payload: Mapping[str, JSONValue],
    *,
    allowed_paths: Mapping[str, Path],
):
    name = payload.get("name")
    qual = payload.get("qual")
    path_key = payload.get("path")
    raw_params = payload.get("params")
    params_payload = sequence_or_none(raw_params)
    path = allowed_paths.get(path_key) if type(path_key) is str else None
    if (
        type(name) is str
        and type(qual) is str
        and path is not None
        and params_payload is not None
    ):
        params = str_list_from_sequence(params_payload)
        raw_annots = payload.get("annots")
        annots: dict[str, JSONValue] = {}
        for param, annot in mapping_or_empty(raw_annots).items():
            check_deadline()
            if type(param) is str and (annot is None or type(annot) is str):
                annots[param] = annot
        raw_calls = payload.get("calls")
        calls = _deserialize_call_args_list(sequence_or_none(raw_calls) or [])
        unused_params = str_set_from_sequence(payload.get("unused_params"))
        unknown_key_carriers = str_set_from_sequence(payload.get("unknown_key_carriers"))
        defaults = str_set_from_sequence(payload.get("defaults"))
        class_name = payload.get("class_name")
        if class_name is not None and type(class_name) is not str:
            class_name = None
        scope = str_tuple_from_sequence(payload.get("scope"))
        lexical_scope = str_tuple_from_sequence(payload.get("lexical_scope"))
        decision_params = str_set_from_sequence(payload.get("decision_params"))
        decision_surface_reasons: dict[str, set[str]] = {}
        for param, raw_reasons in mapping_or_empty(
            payload.get("decision_surface_reasons")
        ).items():
            check_deadline()
            if type(param) is str:
                reasons = str_set_from_sequence(raw_reasons)
                if reasons:
                    decision_surface_reasons[param] = reasons
        value_decision_params = str_set_from_sequence(payload.get("value_decision_params"))
        value_decision_reasons = str_set_from_sequence(payload.get("value_decision_reasons"))
        positional_params = str_tuple_from_sequence(payload.get("positional_params"))
        kwonly_params = str_tuple_from_sequence(payload.get("kwonly_params"))
        raw_vararg = payload.get("vararg")
        vararg = raw_vararg if type(raw_vararg) is str else None
        raw_kwarg = payload.get("kwarg")
        kwarg = raw_kwarg if type(raw_kwarg) is str else None
        param_spans: dict[str, tuple[int, int, int, int]] = {}
        for param, raw_span in mapping_or_empty(payload.get("param_spans")).items():
            check_deadline()
            if type(param) is str:
                span = int_tuple4_or_none(raw_span)
                if span is not None:
                    param_spans[param] = span
        function_span = int_tuple4_or_none(payload.get("function_span"))
        return FunctionInfo(
            name=cast(str, name),
            qual=cast(str, qual),
            path=path,
            params=params,
            annots=annots,
            calls=calls,
            unused_params=unused_params,
            unknown_key_carriers=unknown_key_carriers,
            defaults=defaults,
            transparent=bool(payload.get("transparent", True)),
            class_name=cast(str | None, class_name),
            scope=scope,
            lexical_scope=lexical_scope,
            decision_params=decision_params,
            decision_surface_reasons=decision_surface_reasons,
            value_decision_params=value_decision_params,
            value_decision_reasons=value_decision_reasons,
            positional_params=positional_params,
            kwonly_params=kwonly_params,
            vararg=vararg,
            kwarg=kwarg,
            param_spans=param_spans,
            function_span=function_span,
        )
    return None


def _serialize_class_info_for_resume(class_info: ClassInfo) -> JSONObject:
    return {
        "qual": class_info.qual,
        "module": class_info.module,
        "bases": list(class_info.bases),
        "methods": sort_once(class_info.methods, source = 'src/gabion/analysis/dataflow_audit.py:15196'),
    }


def _deserialize_class_info_for_resume(
    payload: Mapping[str, JSONValue],
):
    qual = payload.get("qual")
    module = payload.get("module")
    if type(qual) is not str or type(module) is not str:
        return None
    bases = str_list_from_sequence(payload.get("bases"))
    methods = str_set_from_sequence(payload.get("methods"))
    return ClassInfo(
        qual=qual,
        module=module,
        bases=bases,
        methods=methods,
    )


def _serialize_symbol_table_for_resume(table: SymbolTable) -> JSONObject:
    return {
        "imports": [
            [module, name, fqn]
            for (module, name), fqn in sort_once(
                table.imports.items(),
                source="_serialize_symbol_table_for_resume.imports",
            )
        ],
        "internal_roots": sort_once(
            table.internal_roots,
            source="_serialize_symbol_table_for_resume.internal_roots",
        ),
        "external_filter": bool(table.external_filter),
        "star_imports": {
            module: sort_once(
                names,
                source=f"_serialize_symbol_table_for_resume.star_imports.{module}",
            )
            for module, names in sort_once(
                table.star_imports.items(),
                source="_serialize_symbol_table_for_resume.star_imports",
            )
        },
        "module_exports": {
            module: sort_once(
                names,
                source=f"_serialize_symbol_table_for_resume.module_exports.{module}",
            )
            for module, names in sort_once(
                table.module_exports.items(),
                source="_serialize_symbol_table_for_resume.module_exports",
            )
        },
        "module_export_map": {
            module: {
                name: mapping[name]
                for name in sort_once(
                    mapping,
                    source=(
                        "_serialize_symbol_table_for_resume.module_export_map."
                        f"{module}"
                    ),
                )
            }
            for module, mapping in sort_once(
                table.module_export_map.items(),
                source="_serialize_symbol_table_for_resume.module_export_map",
            )
        },
    }


def _deserialize_symbol_table_for_resume(payload: Mapping[str, JSONValue]) -> SymbolTable:
    table = SymbolTable(external_filter=bool(payload.get("external_filter", True)))
    raw_imports = sequence_or_none(payload.get("imports"))
    if raw_imports is not None:
        for entry in raw_imports:
            check_deadline()
            entry_sequence = sequence_or_none(entry)
            if entry_sequence is not None and len(entry_sequence) == 3:
                module, name, fqn = entry_sequence
                if type(module) is str and type(name) is str and type(fqn) is str:
                    table.imports[(module, name)] = fqn
    raw_internal_roots = sequence_or_none(payload.get("internal_roots"))
    if raw_internal_roots is not None:
        for entry in raw_internal_roots:
            check_deadline()
            if type(entry) is str:
                table.internal_roots.add(entry)
    raw_star_imports = mapping_or_none(payload.get("star_imports"))
    if raw_star_imports is not None:
        for module, raw_names in raw_star_imports.items():
            check_deadline()
            if type(module) is str:
                names = str_set_from_sequence(raw_names)
                table.star_imports[module] = names
    raw_module_exports = mapping_or_none(payload.get("module_exports"))
    if raw_module_exports is not None:
        for module, raw_names in raw_module_exports.items():
            check_deadline()
            if type(module) is str:
                names = str_set_from_sequence(raw_names)
                table.module_exports[module] = names
    raw_module_export_map = mapping_or_none(payload.get("module_export_map"))
    if raw_module_export_map is not None:
        for module, raw_mapping in raw_module_export_map.items():
            check_deadline()
            if type(module) is str:
                mapping: dict[str, str] = {}
                mapping_payload = mapping_or_empty(raw_mapping)
                for name, mapped in mapping_payload.items():
                    check_deadline()
                    if type(name) is str and type(mapped) is str:
                        mapping[name] = mapped
                table.module_export_map[module] = mapping
    return table


def _analysis_index_resume_variant_payload(payload: Mapping[str, JSONValue]) -> JSONObject:
    return {
        str(key): payload[key]
        for key in payload
        if str(key) != _ANALYSIS_INDEX_RESUME_VARIANTS_KEY
    }


def _analysis_index_resume_variants(
    payload = None,
) -> dict[str, JSONObject]:
    variants: dict[str, JSONObject] = {}
    if payload is None:
        return variants
    raw_variants = payload.get(_ANALYSIS_INDEX_RESUME_VARIANTS_KEY)
    raw_variants_mapping = mapping_or_none(raw_variants)
    if raw_variants_mapping is not None:
        for identity, raw_variant in raw_variants_mapping.items():
            check_deadline()
            raw_variant_mapping = mapping_or_none(raw_variant)
            if type(identity) is str and raw_variant_mapping is not None:
                variant_payload = payload_with_format(raw_variant_mapping, format_version=1)
                if variant_payload is not None:
                    variants[identity] = _analysis_index_resume_variant_payload(
                        variant_payload
                    )
    return variants


def _with_analysis_index_resume_variants(
    *,
    payload: JSONObject,
    previous_payload,
) -> JSONObject:
    current_identity = str(payload.get("index_cache_identity", "") or "")
    variants = _analysis_index_resume_variants(previous_payload)
    if current_identity:
        variants[current_identity] = _analysis_index_resume_variant_payload(payload)
    if not variants:
        return payload
    ordered_variant_keys = sort_once(variants.keys(), source = 'src/gabion/analysis/dataflow_audit.py:15360')
    if current_identity and current_identity in ordered_variant_keys:
        ordered_variant_keys.remove(current_identity)
        ordered_variant_keys.append(current_identity)
    if len(ordered_variant_keys) > _ANALYSIS_INDEX_RESUME_MAX_VARIANTS:
        ordered_variant_keys = ordered_variant_keys[-_ANALYSIS_INDEX_RESUME_MAX_VARIANTS :]
    payload[_ANALYSIS_INDEX_RESUME_VARIANTS_KEY] = {
        key: variants[key] for key in ordered_variant_keys
    }
    return payload


def _serialize_analysis_index_resume_payload(
    *,
    hydrated_paths: set[Path],
    by_qual: Mapping[str, FunctionInfo],
    symbol_table: SymbolTable,
    class_index: Mapping[str, ClassInfo],
    index_cache_identity: str,
    projection_cache_identity: str,
    profiling_v1 = None,
    previous_payload = None,
) -> JSONObject:
    hydrated_path_keys = sort_once(
        (
            _analysis_collection_resume_path_key(path)
            for path in hydrated_paths
        ),
        source="_serialize_analysis_index_resume_payload.hydrated_paths",
    )
    ordered_function_items = list(
        sort_once(
            by_qual.items(),
            source="_serialize_analysis_index_resume_payload.functions_by_qual",
        )
    )
    ordered_class_items = list(
        sort_once(
            class_index.items(),
            source="_serialize_analysis_index_resume_payload.class_index",
        )
    )
    resume_digest = hashlib.sha1(
        json.dumps(
            {
                "hydrated_paths": hydrated_path_keys,
                "function_quals": [qual for qual, _ in ordered_function_items],
                "class_quals": [qual for qual, _ in ordered_class_items],
            },
            sort_keys=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    payload: JSONObject = {
        "format_version": 1,
        "phase": "analysis_index_hydration",
        "resume_digest": resume_digest,
        "index_cache_identity": index_cache_identity,
        "projection_cache_identity": projection_cache_identity,
        "hydrated_paths": hydrated_path_keys,
        "hydrated_paths_count": len(hydrated_path_keys),
        "function_count": len(by_qual),
        "class_count": len(class_index),
        "functions_by_qual": {
            qual: _serialize_function_info_for_resume(info)
            for qual, info in ordered_function_items
        },
        "symbol_table": _serialize_symbol_table_for_resume(symbol_table),
        "class_index": {
            qual: _serialize_class_info_for_resume(class_info)
            for qual, class_info in ordered_class_items
        },
    }
    profiling_payload = mapping_or_none(cast(JSONValue, profiling_v1))
    if profiling_payload is not None:
        payload["profiling_v1"] = {str(key): profiling_payload[key] for key in profiling_payload}
    return _with_analysis_index_resume_variants(
        payload=payload,
        previous_payload=previous_payload,
    )


def _load_analysis_index_resume_payload(
    *,
    payload,
    file_paths: Sequence[Path],
    expected_index_cache_identity: str = "",
    expected_projection_cache_identity: str = "",
) -> tuple[set[Path], dict[str, FunctionInfo], SymbolTable, dict[str, ClassInfo]]:
    hydrated_paths: set[Path] = set()
    by_qual: dict[str, FunctionInfo] = {}
    symbol_table = SymbolTable()
    class_index: dict[str, ClassInfo] = {}
    payload = payload_with_format(payload, format_version=1)
    if payload is None:
        return hydrated_paths, by_qual, symbol_table, class_index
    selected_payload: Mapping[str, JSONValue] = payload
    if expected_index_cache_identity:
        resume_identity = str(payload.get("index_cache_identity", "") or "")
        if not _cache_identity_matches(resume_identity, expected_index_cache_identity):
            variants = _analysis_index_resume_variants(payload)
            variant = _resume_variant_for_identity(variants, expected_index_cache_identity)
            if variant is None:
                return hydrated_paths, by_qual, symbol_table, class_index
            selected_payload = variant
    if expected_projection_cache_identity:
        projection_identity = str(selected_payload.get("projection_cache_identity", "") or "")
        if not _cache_identity_matches(projection_identity, expected_projection_cache_identity):
            return hydrated_paths, by_qual, symbol_table, class_index
    allowed_paths = allowed_path_lookup(
        file_paths,
        key_fn=_analysis_collection_resume_path_key,
    )
    hydrated_paths = set(
        load_allowed_paths_from_sequence(
            selected_payload.get("hydrated_paths"),
            allowed_paths=allowed_paths,
        )
    )
    raw_functions = selected_payload.get("functions_by_qual")
    raw_functions_mapping = mapping_or_none(raw_functions)
    if raw_functions_mapping is not None:
        for qual, raw_info in raw_functions_mapping.items():
            check_deadline()
            raw_info_mapping = mapping_or_none(raw_info)
            if type(qual) is str and raw_info_mapping is not None:
                info = _deserialize_function_info_for_resume(
                    raw_info_mapping,
                    allowed_paths=allowed_paths,
                )
                if info is not None:
                    by_qual[qual] = info
    raw_symbol_table = selected_payload.get("symbol_table")
    raw_symbol_table_mapping = mapping_or_none(raw_symbol_table)
    if raw_symbol_table_mapping is not None:
        symbol_table = _deserialize_symbol_table_for_resume(raw_symbol_table_mapping)
    raw_class_index = selected_payload.get("class_index")
    raw_class_index_mapping = mapping_or_none(raw_class_index)
    if raw_class_index_mapping is not None:
        for qual, raw_class in raw_class_index_mapping.items():
            check_deadline()
            raw_class_mapping = mapping_or_none(raw_class)
            if type(qual) is str and raw_class_mapping is not None:
                class_info = _deserialize_class_info_for_resume(raw_class_mapping)
                if class_info is not None:
                    class_index[qual] = class_info
    return hydrated_paths, by_qual, symbol_table, class_index


def _serialize_groups_for_resume(
    groups: dict[str, list[set[str]]],
) -> dict[str, list[list[str]]]:
    payload: dict[str, list[list[str]]] = {}
    for fn_name in sort_once(groups, source = 'src/gabion/analysis/dataflow_audit.py:15511'):
        check_deadline()
        bundles = groups[fn_name]
        normalized = [
            sort_once(
                (str(param) for param in bundle),
                source="src/gabion/analysis/dataflow_audit.py:15514",
            )
            for bundle in bundles
        ]
        normalized = sort_once(
            normalized,
            source="_serialize_groups_for_resume.normalized",
            # Primary by bundle length then lexicalized bundle tuple.
            key=lambda bundle: (len(bundle), bundle),
        )
        payload[fn_name] = normalized
    return payload


def _deserialize_groups_for_resume(
    payload: Mapping[str, JSONValue],
) -> dict[str, list[set[str]]]:
    groups: dict[str, list[set[str]]] = {}
    for fn_name, bundles in payload.items():
        check_deadline()
        bundle_entries = sequence_or_none(bundles)
        if type(fn_name) is str and bundle_entries is not None:
            normalized: list[set[str]] = []
            for bundle in bundle_entries:
                check_deadline()
                bundle_params = sequence_or_none(bundle)
                if bundle_params is not None:
                    normalized.append({str(param) for param in bundle_params})
            groups[fn_name] = normalized
    return groups


def _serialize_param_spans_for_resume(
    spans: dict[str, dict[str, tuple[int, int, int, int]]],
) -> dict[str, dict[str, list[int]]]:
    payload: dict[str, dict[str, list[int]]] = {}
    for fn_name in sort_once(spans, source = 'src/gabion/analysis/dataflow_audit.py:15547'):
        check_deadline()
        param_spans = spans[fn_name]
        payload[fn_name] = {}
        for param_name in sort_once(param_spans, source = 'src/gabion/analysis/dataflow_audit.py:15551'):
            check_deadline()
            span = param_spans[param_name]
            payload[fn_name][param_name] = [int(part) for part in span]
    return payload


def _deserialize_param_spans_for_resume(
    payload: Mapping[str, JSONValue],
) -> dict[str, dict[str, tuple[int, int, int, int]]]:
    spans: dict[str, dict[str, tuple[int, int, int, int]]] = {}
    for fn_name, raw_map in payload.items():
        check_deadline()
        param_map = mapping_or_none(raw_map)
        if type(fn_name) is str and param_map is not None:
            fn_spans: dict[str, tuple[int, int, int, int]] = {}
            for param_name, raw_span in param_map.items():
                check_deadline()
                span_parts = sequence_or_none(raw_span)
                if type(param_name) is str and span_parts is not None and len(span_parts) == 4:
                    try:
                        span = tuple(int(part) for part in span_parts)
                    except (TypeError, ValueError):
                        span = None
                    if span is not None:
                        fn_spans[param_name] = cast(tuple[int, int, int, int], span)
            spans[fn_name] = fn_spans
    return spans


def _serialize_bundle_sites_for_resume(
    bundle_sites: dict[str, list[list[JSONObject]]],
) -> dict[str, list[list[JSONObject]]]:
    payload: dict[str, list[list[JSONObject]]] = {}
    for fn_name in sort_once(bundle_sites, source = 'src/gabion/analysis/dataflow_audit.py:15586'):
        check_deadline()
        fn_sites = bundle_sites[fn_name]
        encoded_fn_sites: list[list[JSONObject]] = []
        for bundle in fn_sites:
            check_deadline()
            encoded_bundle: list[JSONObject] = []
            bundle_entries = sequence_or_none(cast(JSONValue, bundle))
            if bundle_entries is not None:
                for site in bundle_entries:
                    check_deadline()
                    site_mapping = mapping_or_none(site)
                    if site_mapping is not None:
                        encoded_bundle.append(
                            {str(key): site_mapping[key] for key in site_mapping}
                        )
                encoded_fn_sites.append(encoded_bundle)
        payload[fn_name] = encoded_fn_sites
    return payload


def _deserialize_bundle_sites_for_resume(
    payload: Mapping[str, JSONValue],
) -> dict[str, list[list[JSONObject]]]:
    bundle_sites: dict[str, list[list[JSONObject]]] = {}
    for fn_name, raw_sites in payload.items():
        check_deadline()
        site_groups = sequence_or_none(raw_sites)
        if type(fn_name) is str and site_groups is not None:
            fn_sites: list[list[JSONObject]] = []
            for raw_bundle in site_groups:
                check_deadline()
                bundle_entries = sequence_or_none(raw_bundle)
                if bundle_entries is not None:
                    bundle: list[JSONObject] = []
                    for site in bundle_entries:
                        check_deadline()
                        site_mapping = mapping_or_none(site)
                        if site_mapping is not None:
                            bundle.append({str(key): site_mapping[key] for key in site_mapping})
                    fn_sites.append(bundle)
            bundle_sites[fn_name] = fn_sites
    return bundle_sites


def _serialize_invariants_for_resume(
    invariants: Sequence[InvariantProposition],
) -> list[JSONObject]:
    payload: list[JSONObject] = []
    for proposition in sort_once(
        invariants,
        key=lambda proposition: (
            proposition.form,
            proposition.terms,
            proposition.scope or "",
            proposition.source or "",
        ),
    source = 'src/gabion/analysis/dataflow_audit.py:15631'):
        check_deadline()
        payload.append(proposition.as_dict())
    return payload


def _deserialize_invariants_for_resume(
    payload: Sequence[JSONValue],
) -> list[InvariantProposition]:
    invariants: list[InvariantProposition] = []
    for entry in payload:
        check_deadline()
        entry_mapping = mapping_or_none(entry)
        if entry_mapping is not None:
            form = entry_mapping.get("form")
            terms = sequence_or_none(entry_mapping.get("terms"))
            if type(form) is str and terms is not None:
                normalized_terms: list[str] = []
                for term in terms:
                    check_deadline()
                    if type(term) is str:
                        normalized_terms.append(term)
                scope = entry_mapping.get("scope")
                source = entry_mapping.get("source")
                invariant_id = entry_mapping.get("invariant_id")
                confidence_raw = entry_mapping.get("confidence")
                confidence = (
                    float(confidence_raw)
                    if type(confidence_raw) in {int, float}
                    else None
                )
                raw_evidence = entry_mapping.get("evidence_keys")
                evidence_keys: tuple[str, ...] = ()
                evidence_sequence = sequence_or_none(raw_evidence)
                if evidence_sequence is not None:
                    evidence_keys = tuple(
                        str(item) for item in evidence_sequence if str(item).strip()
                    )
                normalized = _normalize_invariant_proposition(
                    InvariantProposition(
                        form=form,
                        terms=tuple(normalized_terms),
                        scope=scope if type(scope) is str else None,
                        source=source if type(source) is str else None,
                        invariant_id=invariant_id if type(invariant_id) is str else None,
                        confidence=confidence,
                        evidence_keys=evidence_keys,
                    ),
                    default_scope=scope if type(scope) is str else "",
                    default_source=source if type(source) is str else "resume",
                )
                invariants.append(normalized)
    return invariants


def _serialize_file_scan_resume_state(
    *,
    fn_use: Mapping[str, Mapping[str, ParamUse]],
    fn_calls: Mapping[str, Sequence[CallArgs]],
    fn_param_orders: Mapping[str, Sequence[str]],
    fn_param_spans: Mapping[str, Mapping[str, tuple[int, int, int, int]]],
    fn_names: Mapping[str, str],
    fn_lexical_scopes: Mapping[str, Sequence[str]],
    fn_class_names: Mapping[str, object],
    opaque_callees: set[str],
) -> JSONObject:
    fn_use_payload: JSONObject = {}
    fn_calls_payload: JSONObject = {}
    fn_param_orders_payload: JSONObject = {}
    fn_param_spans_payload: JSONObject = {}
    fn_names_payload: JSONObject = {}
    fn_lexical_scopes_payload: JSONObject = {}
    fn_class_names_payload: JSONObject = {}
    for fn_key in sort_once(fn_use, source = 'src/gabion/analysis/dataflow_audit.py:15706'):
        check_deadline()
        fn_use_payload[fn_key] = _serialize_param_use_map(fn_use[fn_key])
    for fn_key in sort_once(fn_calls, source = 'src/gabion/analysis/dataflow_audit.py:15709'):
        check_deadline()
        fn_calls_payload[fn_key] = _serialize_call_args_list(fn_calls[fn_key])
    for fn_key in sort_once(fn_param_orders, source = 'src/gabion/analysis/dataflow_audit.py:15712'):
        check_deadline()
        fn_param_orders_payload[fn_key] = list(fn_param_orders[fn_key])
    for fn_key in sort_once(fn_param_spans, source = 'src/gabion/analysis/dataflow_audit.py:15715'):
        check_deadline()
        fn_param_spans_payload[fn_key] = _serialize_param_spans_for_resume(
            {fn_key: dict(fn_param_spans[fn_key])}
        ).get(fn_key, {})
    for fn_key in sort_once(fn_names, source = 'src/gabion/analysis/dataflow_audit.py:15720'):
        check_deadline()
        fn_names_payload[fn_key] = fn_names[fn_key]
    for fn_key in sort_once(fn_lexical_scopes, source = 'src/gabion/analysis/dataflow_audit.py:15723'):
        check_deadline()
        fn_lexical_scopes_payload[fn_key] = list(fn_lexical_scopes[fn_key])
    for fn_key in sort_once(fn_class_names, source = 'src/gabion/analysis/dataflow_audit.py:15726'):
        check_deadline()
        fn_class_names_payload[fn_key] = fn_class_names[fn_key]
    return {
        "phase": "function_scan",
        "fn_use": fn_use_payload,
        "fn_calls": fn_calls_payload,
        "fn_param_orders": fn_param_orders_payload,
        "fn_param_spans": fn_param_spans_payload,
        "fn_names": fn_names_payload,
        "fn_lexical_scopes": fn_lexical_scopes_payload,
        "fn_class_names": fn_class_names_payload,
        "opaque_callees": sort_once(opaque_callees, source = 'src/gabion/analysis/dataflow_audit.py:15738'),
        "processed_functions": sort_once(fn_use.keys(), source = 'src/gabion/analysis/dataflow_audit.py:15739'),
    }


def _empty_file_scan_resume_state():
    return ({}, {}, {}, {}, {}, {}, {}, set())


def _load_file_scan_resume_state(
    *,
    payload,
    valid_fn_keys: set[str],
):
    (
        fn_use,
        fn_calls,
        fn_param_orders,
        fn_param_spans,
        fn_names,
        fn_lexical_scopes,
        fn_class_names,
        opaque_callees,
    ) = _empty_file_scan_resume_state()
    payload = payload_with_phase(payload, phase="function_scan")
    if payload is None:
        return _empty_file_scan_resume_state()
    sections = mapping_sections(
        payload,
        section_keys=(
            "fn_use",
            "fn_calls",
            "fn_param_orders",
            "fn_param_spans",
            "fn_names",
            "fn_lexical_scopes",
            "fn_class_names",
        ),
    )
    if sections is None:
        return _empty_file_scan_resume_state()
    (
        raw_use,
        raw_calls,
        raw_param_orders,
        raw_param_spans,
        raw_names,
        raw_scopes,
        raw_class_names,
    ) = sections
    fn_use = load_resume_map(
        payload=raw_use,
        valid_keys=valid_fn_keys,
        parser=lambda raw_value: (
            _deserialize_param_use_map(raw_mapping)
            if (raw_mapping := mapping_or_none(raw_value)) is not None
            else None
        ),
    )
    fn_calls = load_resume_map(
        payload=raw_calls,
        valid_keys=valid_fn_keys,
        parser=lambda raw_value: (
            _deserialize_call_args_list(raw_sequence)
            if (raw_sequence := sequence_or_none(raw_value)) is not None
            else None
        ),
    )
    fn_param_orders = load_resume_map(
        payload=raw_param_orders,
        valid_keys=valid_fn_keys,
        parser=lambda raw_value: (
            str_list_from_sequence(raw_value)
            if sequence_or_none(raw_value) is not None
            else None
        ),
    )
    fn_param_spans = load_resume_map(
        payload=raw_param_spans,
        valid_keys=valid_fn_keys,
        parser=lambda raw_value: (
            _deserialize_param_spans_for_resume({"_": raw_mapping}).get("_", {})
            if (raw_mapping := mapping_or_none(raw_value)) is not None
            else None
        ),
    )
    fn_names = load_resume_map(
        payload=raw_names,
        valid_keys=valid_fn_keys,
        parser=lambda raw_value: raw_value if type(raw_value) is str else None,
    )
    fn_lexical_scopes = load_resume_map(
        payload=raw_scopes,
        valid_keys=valid_fn_keys,
        parser=lambda raw_value: (
            str_tuple_from_sequence(raw_value)
            if sequence_or_none(raw_value) is not None
            else None
        ),
    )
    fn_class_names = {}
    for fn_key, raw_value in deadline_loop_iter(
        iter_valid_key_entries(
            payload=raw_class_names,
            valid_keys=valid_fn_keys,
        )
    ):
        if raw_value is None or type(raw_value) is str:
            fn_class_names[fn_key] = cast(str | None, raw_value)
    raw_opaque = payload.get("opaque_callees")
    raw_opaque_entries = sequence_or_none(raw_opaque)
    if raw_opaque_entries is not None:
        for entry in deadline_loop_iter(raw_opaque_entries):
            if type(entry) is str and entry in valid_fn_keys:
                opaque_callees.add(entry)
    return (
        fn_use,
        fn_calls,
        fn_param_orders,
        fn_param_spans,
        fn_names,
        fn_lexical_scopes,
        fn_class_names,
        opaque_callees,
    )


def _build_analysis_collection_resume_payload(
    *,
    groups_by_path: Mapping[Path, dict[str, list[set[str]]]],
    param_spans_by_path: Mapping[Path, dict[str, dict[str, tuple[int, int, int, int]]]],
    bundle_sites_by_path: Mapping[Path, dict[str, list[list[JSONObject]]]],
    invariant_propositions: Sequence[InvariantProposition],
    completed_paths: set[Path],
    in_progress_scan_by_path: Mapping[Path, JSONObject],
    analysis_index_resume = None,
    file_stage_timings_v1_by_path = None,
) -> JSONObject:
    check_deadline()
    groups_payload: JSONObject = {}
    spans_payload: JSONObject = {}
    sites_payload: JSONObject = {}
    in_progress_scan_payload: JSONObject = {}
    completed_keys = sort_once(
        (_analysis_collection_resume_path_key(path) for path in completed_paths),
        source="src/gabion/analysis/dataflow_audit.py:15898",
    )
    for path_key in completed_keys:
        check_deadline()
        path = Path(path_key)
        groups_payload[path_key] = _serialize_groups_for_resume(
            groups_by_path.get(path, {})
        )
        spans_payload[path_key] = _serialize_param_spans_for_resume(
            param_spans_by_path.get(path, {})
        )
        sites_payload[path_key] = _serialize_bundle_sites_for_resume(
            bundle_sites_by_path.get(path, {})
        )
    previous_path_key = None
    for path in in_progress_scan_by_path:
        check_deadline()
        path_key = _analysis_collection_resume_path_key(path)
        if previous_path_key is not None and previous_path_key > path_key:
            never(
                "in_progress_scan_by_path path order regression",
                previous_path=previous_path_key,
                current_path=path_key,
            )
        previous_path_key = path_key
        in_progress_scan_payload[path_key] = {
            str(key): in_progress_scan_by_path[path][key]
            for key in in_progress_scan_by_path[path]
        }
    payload: JSONObject = {
        "format_version": _ANALYSIS_COLLECTION_RESUME_FORMAT_VERSION,
        "completed_paths": completed_keys,
        "groups_by_path": groups_payload,
        "param_spans_by_path": spans_payload,
        "bundle_sites_by_path": sites_payload,
        "in_progress_scan_by_path": in_progress_scan_payload,
        "invariant_propositions": _serialize_invariants_for_resume(
            invariant_propositions
        ),
    }
    if file_stage_timings_v1_by_path:
        payload["file_stage_timings_v1_by_path"] = {
            _analysis_collection_resume_path_key(path): {
                str(key): value
                for key, value in file_stage_timings_v1_by_path[path].items()
            }
            for path in sort_once(
                file_stage_timings_v1_by_path,
                key=_analysis_collection_resume_path_key,
            source = 'src/gabion/analysis/dataflow_audit.py:15945')
        }
    analysis_index_resume_mapping = mapping_or_none(analysis_index_resume)
    if analysis_index_resume_mapping is not None:
        payload["analysis_index_resume"] = {
            str(key): analysis_index_resume_mapping[key]
            for key in analysis_index_resume_mapping
        }
    return payload


def build_analysis_collection_resume_seed(
    *,
    in_progress_paths: Sequence[Path] = (),
) -> JSONObject:
    """Build an empty collection-resume payload seeded with pending paths."""
    check_deadline()
    in_progress_scan_by_path: dict[Path, JSONObject] = {
        path: {"phase": "scan_pending"} for path in in_progress_paths
    }
    return _build_analysis_collection_resume_payload(
        groups_by_path={},
        param_spans_by_path={},
        bundle_sites_by_path={},
        invariant_propositions=[],
        completed_paths=set(),
        in_progress_scan_by_path=in_progress_scan_by_path,
        analysis_index_resume=None,
    )


def _empty_analysis_collection_resume_payload():
    return ({}, {}, {}, [], set(), {}, None)


def _load_analysis_collection_resume_payload(
    *,
    payload,
    file_paths: Sequence[Path],
    include_invariant_propositions: bool,
):
    (
        groups_by_path,
        param_spans_by_path,
        bundle_sites_by_path,
        invariant_propositions,
        completed_paths,
        in_progress_scan_by_path,
        analysis_index_resume,
    ) = _empty_analysis_collection_resume_payload()
    payload = payload_with_format(
        payload,
        format_version=_ANALYSIS_COLLECTION_RESUME_FORMAT_VERSION,
    )
    if payload is None:
        return _empty_analysis_collection_resume_payload()
    sections = mapping_sections(
        payload,
        section_keys=(
            "groups_by_path",
            "param_spans_by_path",
            "bundle_sites_by_path",
        ),
    )
    if sections is None:
        return _empty_analysis_collection_resume_payload()
    groups_payload, spans_payload, sites_payload = sections
    in_progress_scan_payload = mapping_payload(payload.get("in_progress_scan_by_path"))
    completed_payload = payload.get("completed_paths")
    if in_progress_scan_payload is None:
        in_progress_scan_payload = {}
    allowed_paths = allowed_path_lookup(
        file_paths,
        key_fn=_analysis_collection_resume_path_key,
    )
    for path in load_allowed_paths_from_sequence(
        completed_payload,
        allowed_paths=allowed_paths,
    ):
        check_deadline()
        path_key = _analysis_collection_resume_path_key(path)
        raw_groups = mapping_or_none(groups_payload.get(path_key))
        raw_spans = mapping_or_none(spans_payload.get(path_key))
        raw_sites = mapping_or_none(sites_payload.get(path_key))
        if raw_groups is not None and raw_spans is not None and raw_sites is not None:
            groups_by_path[path] = _deserialize_groups_for_resume(raw_groups)
            param_spans_by_path[path] = _deserialize_param_spans_for_resume(raw_spans)
            bundle_sites_by_path[path] = _deserialize_bundle_sites_for_resume(raw_sites)
            completed_paths.add(path)
    if include_invariant_propositions:
        raw_invariants = sequence_or_none(payload.get("invariant_propositions"))
        if raw_invariants is not None:
            invariant_propositions = _deserialize_invariants_for_resume(raw_invariants)
    for raw_path, raw_state in in_progress_scan_payload.items():
        check_deadline()
        raw_state_mapping = mapping_or_none(raw_state)
        path = allowed_paths.get(raw_path)
        if raw_state_mapping is not None and path is not None and path not in completed_paths:
            in_progress_scan_by_path[path] = {
                str(key): raw_state_mapping[key] for key in raw_state_mapping
            }
    raw_analysis_index_resume = payload.get("analysis_index_resume")
    raw_analysis_index_mapping = mapping_or_none(raw_analysis_index_resume)
    if raw_analysis_index_mapping is not None:
        analysis_index_resume = {
            str(key): raw_analysis_index_mapping[key]
            for key in raw_analysis_index_mapping
        }
    return (
        groups_by_path,
        param_spans_by_path,
        bundle_sites_by_path,
        invariant_propositions,
        completed_paths,
        in_progress_scan_by_path,
        analysis_index_resume,
    )


def _bundle_counts(
    groups_by_path: dict[Path, dict[str, list[set[str]]]]
) -> dict[tuple[str, ...], int]:
    check_deadline()
    _forbid_adhoc_bundle_discovery("_bundle_counts")
    counts: dict[tuple[str, ...], int] = defaultdict(int)
    for groups in groups_by_path.values():
        check_deadline()
        for bundles in groups.values():
            check_deadline()
            for bundle in bundles:
                check_deadline()
                counts[tuple(sort_once(bundle, source = 'src/gabion/analysis/dataflow_audit.py:16093'))] += 1
    return counts


def _merge_counts_by_knobs(
    counts: dict[tuple[str, ...], int],
    knob_names: set[str],
) -> dict[tuple[str, ...], int]:
    check_deadline()
    if not knob_names:
        return counts
    bundles = [set(bundle) for bundle in counts]
    merged: dict[tuple[str, ...], int] = defaultdict(int)
    for bundle_key, count in counts.items():
        check_deadline()
        bundle = set(bundle_key)
        target = bundle
        for other in bundles:
            check_deadline()
            if bundle and bundle.issubset(other):
                extra = set(other) - bundle
                if extra and extra.issubset(knob_names):
                    if len(other) < len(target) or target == bundle:
                        target = set(other)
        merged[tuple(sort_once(target, source = 'src/gabion/analysis/dataflow_audit.py:16117'))] += count
    return merged


def _collect_declared_bundles(root: Path) -> set[tuple[str, ...]]:
    check_deadline()
    _forbid_adhoc_bundle_discovery("_collect_declared_bundles")
    declared: set[tuple[str, ...]] = set()
    file_paths = sort_once(
        root.rglob("*.py"),
        source="_collect_declared_bundles.file_paths",
        key=lambda path: str(path),
    )
    parse_failure_witnesses: list[JSONObject] = []
    bundles_by_path = _collect_config_bundles(
        file_paths,
        parse_failure_witnesses=parse_failure_witnesses,
    )
    for bundles in bundles_by_path.values():
        check_deadline()
        for fields in bundles.values():
            check_deadline()
            declared.add(tuple(sort_once(fields, source = 'src/gabion/analysis/dataflow_audit.py:16139')))
    return declared


@dataclass(frozen=True)
class _SynthesisPlanContext:
    audit_config: AuditConfig
    root: Path
    signature_meta: JSONObject
    path_list: list[Path]
    parse_failure_witnesses: list[JSONObject]
    analysis_index: AnalysisIndex
    by_name: dict[str, list[FunctionInfo]]
    by_qual: dict[str, FunctionInfo]
    symbol_table: SymbolTable
    class_index: dict[str, ClassInfo]
    transitive_callers: dict[str, set[str]]


def _build_synthesis_plan_context(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    *,
    project_root,
    config,
) -> _SynthesisPlanContext:
    check_deadline()
    parse_failure_witnesses: list[JSONObject] = []
    audit_config = config or AuditConfig(
        project_root=project_root or _infer_root(groups_by_path)
    )
    root = project_root or audit_config.project_root or _infer_root(groups_by_path)
    signature_meta = _partial_forest_signature_metadata(groups_by_path)
    path_list = list(groups_by_path.keys())
    analysis_index = _build_analysis_index(
        path_list,
        project_root=root,
        ignore_params=audit_config.ignore_params,
        strictness=audit_config.strictness,
        external_filter=audit_config.external_filter,
        transparent_decorators=audit_config.transparent_decorators,
        parse_failure_witnesses=parse_failure_witnesses,
    )
    _, _, transitive_callers = _build_call_graph(
        path_list,
        project_root=root,
        ignore_params=audit_config.ignore_params,
        strictness=audit_config.strictness,
        external_filter=audit_config.external_filter,
        transparent_decorators=audit_config.transparent_decorators,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=analysis_index,
    )
    return _SynthesisPlanContext(
        audit_config=audit_config,
        root=root,
        signature_meta=signature_meta,
        path_list=path_list,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=analysis_index,
        by_name=analysis_index.by_name,
        by_qual=analysis_index.by_qual,
        symbol_table=analysis_index.symbol_table,
        class_index=analysis_index.class_index,
        transitive_callers=transitive_callers,
    )


def _collect_synthesis_counts_and_evidence(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    *,
    context: _SynthesisPlanContext,
) -> tuple[dict[tuple[str, ...], int], dict[frozenset[str], set[str]]]:
    check_deadline()
    knob_names = _compute_knob_param_names(
        by_name=context.by_name,
        by_qual=context.by_qual,
        symbol_table=context.symbol_table,
        project_root=context.root,
        class_index=context.class_index,
        strictness=context.audit_config.strictness,
        analysis_index=context.analysis_index,
    )
    counts = _bundle_counts(groups_by_path)
    counts = _merge_counts_by_knobs(counts, knob_names)
    bundle_evidence: dict[frozenset[str], set[str]] = defaultdict(set)
    for bundle in counts:
        check_deadline()
        bundle_evidence[frozenset(bundle)].add("dataflow")

    decision_params_by_fn: dict[tuple[Path, str], set[str]] = {}
    decision_ignore = (
        context.audit_config.decision_ignore_params or context.audit_config.ignore_params
    )
    for info in context.by_qual.values():
        check_deadline()
        if not info.decision_params and not info.value_decision_params:
            continue
        fn_key = _function_key(info.scope, info.name)
        params = (set(info.decision_params) | set(info.value_decision_params)) - decision_ignore
        decision_params_by_fn[(info.path, fn_key)] = params

    for path, groups in groups_by_path.items():
        check_deadline()
        for fn_key, bundles in groups.items():
            check_deadline()
            decision_params = decision_params_by_fn.get((path, fn_key))
            if not decision_params:
                continue
            for bundle in bundles:
                check_deadline()
                bundle_evidence[frozenset(bundle)].add("control_context")

    decision_counts: dict[tuple[str, ...], int] = defaultdict(int)
    value_decision_counts: dict[tuple[str, ...], int] = defaultdict(int)
    for info in context.by_qual.values():
        check_deadline()
        caller_count = len(context.transitive_callers.get(info.qual, set()))
        if info.decision_params:
            bundle = tuple(sort_once(info.decision_params, source = 'src/gabion/analysis/dataflow_audit.py:16232'))
            decision_counts[bundle] += 1
            evidence = bundle_evidence[frozenset(bundle)]
            evidence.add("decision_surface")
            if caller_count > 0:
                evidence.add("tier-2:decision-bundle-elevation")
            else:
                evidence.add("tier-3:decision-table-boundary")
        if info.value_decision_params:
            bundle = tuple(sort_once(info.value_decision_params, source = 'src/gabion/analysis/dataflow_audit.py:16241'))
            value_decision_counts[bundle] += 1
            bundle_evidence[frozenset(bundle)].add("value_decision_surface")

    for bundle, count in decision_counts.items():
        check_deadline()
        if bundle not in counts:
            counts[bundle] = count
    for bundle, count in value_decision_counts.items():
        check_deadline()
        if bundle not in counts:
            counts[bundle] = count
    return counts, bundle_evidence


def _synthesis_empty_payload(
    *,
    signature_meta: Mapping[str, object],
    invariant_propositions: Sequence[InvariantProposition],
    property_hook_min_confidence: float,
    emit_hypothesis_templates: bool,
) -> JSONObject:
    response = SynthesisResponse(
        protocols=[],
        warnings=["No bundles observed for synthesis."],
        errors=[],
    )
    payload = response.model_dump()
    payload.update(signature_meta)
    payload["property_hook_manifest"] = generate_property_hook_manifest(
        invariant_propositions,
        min_confidence=property_hook_min_confidence,
        emit_hypothesis_templates=emit_hypothesis_templates,
    )
    return payload


def _compute_synthesis_tiers_and_merge(
    *,
    counts: Mapping[tuple[str, ...], int],
    bundle_evidence: Mapping[frozenset[str], set[str]],
    root: Path,
    max_tier: int,
    min_bundle_size: int,
    allow_singletons: bool,
    merge_overlap_threshold,
) -> tuple[
    dict[frozenset[str], int],
    dict[frozenset[str], set[str]],
    NamingContext,
    SynthesisConfig,
    set[str],
]:
    check_deadline()
    declared = _collect_declared_bundles(root)
    bundle_tiers: dict[frozenset[str], int] = {}
    frequency: dict[str, int] = defaultdict(int)
    bundle_fields: set[str] = set()
    for bundle, count in counts.items():
        check_deadline()
        tier = 1 if bundle in declared else (2 if count > 1 else 3)
        bundle_tiers[frozenset(bundle)] = tier
        for field in bundle:
            check_deadline()
            frequency[field] += count
            bundle_fields.add(field)

    original_bundles = [set(bundle) for bundle in counts]
    synth_config = SynthesisConfig(
        max_tier=max_tier,
        min_bundle_size=min_bundle_size,
        allow_singletons=allow_singletons,
        merge_overlap_threshold=(
            merge_overlap_threshold
            if merge_overlap_threshold is not None
            else SynthesisConfig().merge_overlap_threshold
        ),
    )
    merged_bundles = merge_bundles(
        original_bundles, min_overlap=synth_config.merge_overlap_threshold
    )
    merged_bundle_tiers: dict[frozenset[str], int] = {}
    for merged in merged_bundles:
        check_deadline()
        members = [
            bundle
            for bundle in original_bundles
            if bundle and bundle.issubset(merged)
        ]
        if not members:
            continue
        merged_bundle_tiers[frozenset(merged)] = min(
            bundle_tiers[frozenset(member)] for member in members
        )
    merged_bundle_evidence: dict[frozenset[str], set[str]] = {}
    if merged_bundle_tiers:
        bundle_tiers = merged_bundle_tiers
        for merged in merged_bundles:
            check_deadline()
            members = [
                bundle
                for bundle in original_bundles
                if bundle and bundle.issubset(merged)
            ]
            if not members:
                continue
            evidence: set[str] = set()
            for member in members:
                check_deadline()
                evidence.update(bundle_evidence.get(frozenset(member), set()))
            merged_bundle_evidence[frozenset(merged)] = evidence
    else:
        merged_bundle_evidence = {
            key: set(values) for key, values in bundle_evidence.items()
        }

    naming_context = NamingContext(frequency=dict(frequency))
    return (
        bundle_tiers,
        merged_bundle_evidence,
        naming_context,
        synth_config,
        bundle_fields,
    )


def _infer_synthesis_field_types(
    *,
    bundle_fields: set[str],
    context: _SynthesisPlanContext,
) -> tuple[dict[str, str], list[str]]:
    field_types: dict[str, str] = {}
    type_warnings: list[str] = []
    if not bundle_fields:
        return field_types, type_warnings

    inferred, _, _ = analyze_type_flow_repo_with_map(
        context.path_list,
        project_root=context.root,
        ignore_params=context.audit_config.ignore_params,
        strictness=context.audit_config.strictness,
        external_filter=context.audit_config.external_filter,
        transparent_decorators=context.audit_config.transparent_decorators,
        parse_failure_witnesses=context.parse_failure_witnesses,
        analysis_index=context.analysis_index,
    )
    type_sets: dict[str, set[str]] = defaultdict(set)
    for annots in inferred.values():
        check_deadline()
        for name, annot in annots.items():
            check_deadline()
            if name not in bundle_fields or not annot:
                continue
            type_sets[name].add(annot)
    for infos in context.by_name.values():
        check_deadline()
        for info in infos:
            check_deadline()
            for call in info.calls:
                check_deadline()
                if not call.is_test:
                    callee = _resolve_callee(
                        call.callee,
                        info,
                        context.by_name,
                        context.by_qual,
                        context.symbol_table,
                        context.root,
                        context.class_index,
                    )
                    if callee is not None and callee.transparent:
                        callee_params = callee.params
                        for idx_str, value in call.const_pos.items():
                            check_deadline()
                            idx = int(idx_str)
                            if idx < len(callee_params):
                                param = callee_params[idx]
                                if param in bundle_fields:
                                    hint = _type_from_const_repr(value)
                                    if hint:
                                        type_sets[param].add(hint)
                        for kw, value in call.const_kw.items():
                            check_deadline()
                            if kw in callee_params and kw in bundle_fields:
                                hint = _type_from_const_repr(value)
                                if hint:
                                    type_sets[kw].add(hint)
    for name, types in type_sets.items():
        check_deadline()
        combined, conflicted = _combine_type_hints(types)
        field_types[name] = combined
        if conflicted and len(types) > 1:
            type_warnings.append(
                f"Conflicting type hints for '{name}': {sort_once(types, source = 'src/gabion/analysis/dataflow_audit.py:16393')} -> {combined}"
            )
    return field_types, type_warnings


def _synthesis_payload_from_plan(
    *,
    plan: SynthesisPlan,
    bundle_evidence: Mapping[frozenset[str], set[str]],
    type_warnings: list[str],
    signature_meta: Mapping[str, object],
    invariant_propositions: Sequence[InvariantProposition],
    property_hook_min_confidence: float,
    emit_hypothesis_templates: bool,
) -> JSONObject:
    response = SynthesisResponse(
        protocols=[
            {
                "name": spec.name,
                "fields": [
                    {
                        "name": field.name,
                        "type_hint": field.type_hint,
                        "source_params": sort_once(field.source_params, source = 'src/gabion/analysis/dataflow_audit.py:16408'),
                    }
                    for field in spec.fields
                ],
                "bundle": sort_once(spec.bundle, source = 'src/gabion/analysis/dataflow_audit.py:16412'),
                "tier": spec.tier,
                "rationale": spec.rationale,
                "evidence": sort_once(bundle_evidence.get(frozenset(spec.bundle), set()), source = 'src/gabion/analysis/dataflow_audit.py:16415'),
            }
            for spec in plan.protocols
        ],
        warnings=plan.warnings + type_warnings,
        errors=plan.errors,
    )
    payload = response.model_dump()
    payload.update(signature_meta)
    payload["property_hook_manifest"] = generate_property_hook_manifest(
        invariant_propositions,
        min_confidence=property_hook_min_confidence,
        emit_hypothesis_templates=emit_hypothesis_templates,
    )
    return payload


def build_synthesis_plan(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    *,
    project_root = None,
    max_tier: int = 2,
    min_bundle_size: int = 2,
    allow_singletons: bool = False,
    merge_overlap_threshold = None,
    config = None,
    invariant_propositions: Sequence[InvariantProposition] = (),
    property_hook_min_confidence: float = 0.7,
    emit_hypothesis_templates: bool = False,
) -> JSONObject:
    check_deadline()
    context = _build_synthesis_plan_context(
        groups_by_path,
        project_root=project_root,
        config=config,
    )
    counts, bundle_evidence = _collect_synthesis_counts_and_evidence(
        groups_by_path,
        context=context,
    )
    if not counts:
        return _synthesis_empty_payload(
            signature_meta=context.signature_meta,
            invariant_propositions=invariant_propositions,
            property_hook_min_confidence=property_hook_min_confidence,
            emit_hypothesis_templates=emit_hypothesis_templates,
        )
    (
        bundle_tiers,
        bundle_evidence,
        naming_context,
        synth_config,
        bundle_fields,
    ) = _compute_synthesis_tiers_and_merge(
        counts=counts,
        bundle_evidence=bundle_evidence,
        root=context.root,
        max_tier=max_tier,
        min_bundle_size=min_bundle_size,
        allow_singletons=allow_singletons,
        merge_overlap_threshold=merge_overlap_threshold,
    )
    field_types, type_warnings = _infer_synthesis_field_types(
        bundle_fields=bundle_fields,
        context=context,
    )
    plan = Synthesizer(config=synth_config).plan(
        bundle_tiers=bundle_tiers,
        field_types=field_types,
        naming_context=naming_context,
    )
    return _synthesis_payload_from_plan(
        plan=plan,
        bundle_evidence=bundle_evidence,
        type_warnings=type_warnings,
        signature_meta=context.signature_meta,
        invariant_propositions=invariant_propositions,
        property_hook_min_confidence=property_hook_min_confidence,
        emit_hypothesis_templates=emit_hypothesis_templates,
    )


def render_synthesis_section(plan: JSONObject) -> str:
    return _report_render_synthesis_section(plan, check_deadline=check_deadline)

def render_protocol_stubs(plan: JSONObject, kind: str = "dataclass") -> str:
    return _render_protocol_stubs(plan, kind=kind)


def build_refactor_plan(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    paths: list[Path],
    *,
    config: AuditConfig,
) -> JSONObject:
    check_deadline()
    parse_failure_witnesses: list[JSONObject] = []
    signature_meta = _partial_forest_signature_metadata(groups_by_path)
    file_paths = _iter_paths([str(p) for p in paths], config)
    if not file_paths:
        payload = {"bundles": [], "warnings": ["No files available for refactor plan."]}
        payload.update(signature_meta)
        return payload

    analysis_index = _build_analysis_index(
        file_paths,
        project_root=config.project_root,
        ignore_params=config.ignore_params,
        strictness=config.strictness,
        external_filter=config.external_filter,
        transparent_decorators=config.transparent_decorators,
        parse_failure_witnesses=parse_failure_witnesses,
    )
    by_name = analysis_index.by_name
    by_qual = analysis_index.by_qual
    symbol_table = analysis_index.symbol_table
    class_index = analysis_index.class_index
    info_by_path_name: dict[tuple[Path, str], FunctionInfo] = {}
    for infos in by_name.values():
        check_deadline()
        for info in infos:
            check_deadline()
            key = _function_key(info.scope, info.name)
            info_by_path_name[(info.path, key)] = info

    bundle_map: dict[tuple[str, ...], dict[str, FunctionInfo]] = defaultdict(dict)
    for path, groups in groups_by_path.items():
        check_deadline()
        for fn, bundles in groups.items():
            check_deadline()
            for bundle in bundles:
                check_deadline()
                key = tuple(sort_once(bundle, source = 'src/gabion/analysis/dataflow_audit.py:16482'))
                info = info_by_path_name.get((path, fn))
                if info is not None:
                    bundle_map[key][info.qual] = info

    plans: list[JSONObject] = []
    for bundle, infos in sort_once(bundle_map.items(), key=lambda item: (len(item[0]), item[0]), source = 'src/gabion/analysis/dataflow_audit.py:16488'):
        check_deadline()
        comp = dict(infos)
        deps: dict[str, set[str]] = {qual: set() for qual in comp}
        for info in infos.values():
            check_deadline()
            for call in info.calls:
                check_deadline()
                callee = _resolve_callee(
                    call.callee,
                    info,
                    by_name,
                    by_qual,
                    symbol_table,
                    config.project_root,
                    class_index,
                )
                if (
                    callee is not None
                    and callee.transparent
                    and callee.qual in comp
                ):
                    deps[info.qual].add(callee.qual)
        schedule = topological_schedule(deps)
        plans.append(
            {
                "bundle": list(bundle),
                "functions": sort_once(comp.keys(), source = 'src/gabion/analysis/dataflow_audit.py:16515'),
                "order": schedule.order,
                "cycles": [sort_once(list(cycle), source = 'src/gabion/analysis/dataflow_audit.py:16517') for cycle in schedule.cycles],
            }
        )

    warnings: list[str] = []
    if not plans:
        warnings.append("No bundle components available for refactor plan.")
    payload = {"bundles": plans, "warnings": warnings}
    payload.update(signature_meta)
    return payload


def render_refactor_plan(plan: JSONObject) -> str:
    check_deadline()
    bundles = plan.get("bundles", [])
    warnings = plan.get("warnings", [])
    lines = ["", "## Refactoring plan (prototype)", ""]
    if not bundles:
        lines.append("No refactoring plan available.")
    else:
        for entry in bundles:
            check_deadline()
            bundle = entry.get("bundle", [])
            title = ", ".join(bundle) if bundle else "(unknown bundle)"
            lines.append(f"### Bundle: {title}")
            order = entry.get("order", [])
            if order:
                lines.append("Order (callee-first):")
                lines.append("```")
                for item in order:
                    check_deadline()
                    lines.append(f"- {item}")
                lines.append("```")
            cycles = entry.get("cycles", [])
            if cycles:
                lines.append("Cycles:")
                lines.append("```")
                for cycle in cycles:
                    check_deadline()
                    lines.append(", ".join(cycle))
                lines.append("```")
    if warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.append("```")
        lines.extend(str(w) for w in warnings)
        lines.append("```")
    return "\n".join(lines)


def _render_type_mermaid(
    suggestions: list[str],
    ambiguities: list[str],
) -> str:
    check_deadline()
    lines = ["```mermaid", "flowchart LR"]
    node_id = 0
    def _node(label: str) -> str:
        nonlocal node_id
        node_id += 1
        node = f"type_{node_id}"
        safe = label.replace('"', "'")
        lines.append(f'  {node}["{safe}"]')
        return node

    for entry in suggestions:
        # Format: file:func.param can tighten to Type
        check_deadline()
        if " can tighten to " not in entry:
            continue
        lhs, rhs = entry.split(" can tighten to ", 1)
        src = _node(lhs)
        dst = _node(rhs)
        lines.append(f"  {src} --> {dst}")
    for entry in ambiguities:
        check_deadline()
        if " downstream types conflict: " not in entry:
            continue
        lhs, rhs = entry.split(" downstream types conflict: ", 1)
        src = _node(lhs)
        # rhs is a repr of list; keep as string nodes per type
        rhs = rhs.strip()
        if rhs.startswith("[") and rhs.endswith("]"):
            rhs = rhs[1:-1]
        type_names = []
        for item in rhs.split(","):
            check_deadline()
            item = item.strip()
            if not item:
                continue
            item = item.strip("'\"")
            type_names.append(item)
        for type_name in type_names:
            check_deadline()
            dst = _node(type_name)
            lines.append(f"  {src} -.-> {dst}")
    lines.append("```")
    return "\n".join(lines)


def _compute_violations(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    max_components: int,
    *,
    report: ReportCarrier,
) -> list[str]:
    _, violations = _emit_report(
        groups_by_path,
        max_components,
        report=report,
    )
    return sort_once(
        set(violations),
        source="_compute_violations.violations",
    )


def _resolve_baseline_path(path, root: Path):
    if not path:
        return None
    baseline = Path(path)
    if not baseline.is_absolute():
        baseline = root / baseline
    return baseline


def _resolve_synth_registry_path(path, root: Path):
    if not path:
        return None
    value = str(path).strip()
    if not value:
        return None
    if value.endswith("/LATEST/fingerprint_synth.json"):
        marker = Path(root) / value.replace(
            "/LATEST/fingerprint_synth.json", "/LATEST.txt"
        )
        try:
            stamp = marker.read_text().strip()
        except OSError:
            return None
        return (marker.parent / stamp / "fingerprint_synth.json").resolve()
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def _write_text_or_stdout(path: str, text: str) -> None:
    if path.strip() == "-":
        print(text)
        return
    Path(path).write_text(text)


def _write_json_or_stdout(path: str, payload: object) -> None:
    _write_text_or_stdout(path, json.dumps(payload, indent=2, sort_keys=False))


def _has_followup_actions(
    args: argparse.Namespace,
    *,
    include_type_audit: bool = True,
    include_structure_tree: bool = False,
    include_structure_metrics: bool = False,
    include_decision_snapshot: bool = False,
) -> bool:
    # dataflow-bundle: include_decision_snapshot, include_structure_metrics, include_structure_tree, include_type_audit
    return bool(
        (include_type_audit and args.type_audit)
        or args.synthesis_plan
        or args.synthesis_report
        or args.synthesis_protocols
        or args.refactor_plan
        or args.refactor_plan_json
        or include_structure_tree
        or include_structure_metrics
        or include_decision_snapshot
    )


def _emit_sidecar_outputs(
    *,
    args: argparse.Namespace,
    analysis: AnalysisResult,
    fingerprint_deadness_json,
    fingerprint_coherence_json,
    fingerprint_rewrite_plans_json,
    fingerprint_exception_obligations_json,
    fingerprint_handledness_json,
) -> None:
    # dataflow-bundle: fingerprint_coherence_json, fingerprint_deadness_json, fingerprint_exception_obligations_json, fingerprint_handledness_json, fingerprint_rewrite_plans_json
    for path, payload, require_content in deadline_loop_iter(
        (
            (args.fingerprint_synth_json, analysis.fingerprint_synth_registry, True),
            (args.fingerprint_provenance_json, analysis.fingerprint_provenance, True),
            (fingerprint_deadness_json, analysis.deadness_witnesses, False),
            (fingerprint_coherence_json, analysis.coherence_witnesses, False),
            (fingerprint_rewrite_plans_json, analysis.rewrite_plans, False),
            (
                fingerprint_exception_obligations_json,
                analysis.exception_obligations,
                False,
            ),
            (fingerprint_handledness_json, analysis.handledness_witnesses, False),
        )
    ):
        if not path:
            continue
        if require_content and not payload:
            continue
        _write_json_or_stdout(path, payload)

    if args.lint:
        for line in deadline_loop_iter(analysis.lint_lines):
            print(line)


def _load_baseline(path: Path) -> set[str]:
    check_deadline()
    if not path.exists():
        return set()
    try:
        raw = path.read_text()
    except OSError:
        return set()
    entries: set[str] = set()
    for line in raw.splitlines():
        check_deadline()
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entries.add(line)
    return entries


def _write_baseline(path: Path, violations: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unique = sort_once(
        set(violations),
        source="_write_baseline.violations",
    )
    header = [
        "# gabion baseline (ratchet)",
        "# Lines list known violations to allow; new ones should fail.",
        "",
    ]
    path.write_text("\n".join(header + unique) + "\n")


def _apply_baseline(
    violations: list[str], baseline_allowlist: set[str]
) -> tuple[list[str], list[str]]:
    if not baseline_allowlist:
        return violations, []
    new = [line for line in violations if line not in baseline_allowlist]
    suppressed = [line for line in violations if line in baseline_allowlist]
    return new, suppressed


def resolve_baseline_path(path, root: Path):
    return _resolve_baseline_path(path, root)


def load_baseline(path: Path) -> set[str]:
    return _load_baseline(path)


def write_baseline(path: Path, violations: list[str]) -> None:
    _write_baseline(path, violations)


def apply_baseline(
    violations: list[str], baseline_allowlist: set[str]
) -> tuple[list[str], list[str]]:
    return _apply_baseline(violations, baseline_allowlist)


def render_dot(forest: Forest) -> str:
    return _emit_dot(forest)


def render_report(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    max_components: int,
    *,
    report: ReportCarrier,
) -> tuple[str, list[str]]:
    return _emit_report(
        groups_by_path,
        max_components,
        report=report,
    )


def compute_violations(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    max_components: int,
    *,
    report: ReportCarrier,
) -> list[str]:
    return _compute_violations(
        groups_by_path,
        max_components,
        report=report,
    )


from gabion.analysis.dataflow_pipeline import analyze_paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--root", default=".", help="Project root for module resolution.")
    parser.add_argument("--config", default=None, help="Path to gabion.toml.")
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        help="Comma-separated directory names to exclude (repeatable).",
    )
    parser.add_argument(
        "--ignore-params",
        default=None,
        help="Comma-separated parameter names to ignore.",
    )
    parser.add_argument(
        "--transparent-decorators",
        default=None,
        help="Comma-separated decorator names treated as transparent.",
    )
    parser.add_argument(
        "--allow-external",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow resolving calls into external libraries.",
    )
    parser.add_argument(
        "--strictness",
        choices=["high", "low"],
        default=None,
        help="Wildcard forwarding strictness (default: high).",
    )
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--dot", default=None, help="Write DOT graph to file or '-' for stdout.")
    parser.add_argument(
        "--emit-structure-tree",
        default=None,
        help="Write canonical structure snapshot JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--emit-structure-metrics",
        default=None,
        help="Write structure metrics JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--fingerprint-synth-json",
        default=None,
        help="Write fingerprint synth registry JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--fingerprint-provenance-json",
        default=None,
        help="Write fingerprint provenance JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--fingerprint-deadness-json",
        default=None,
        help="Write fingerprint deadness JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--fingerprint-coherence-json",
        default=None,
        help="Write fingerprint coherence JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--fingerprint-rewrite-plans-json",
        default=None,
        help="Write fingerprint rewrite plans JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--fingerprint-exception-obligations-json",
        default=None,
        help="Write fingerprint exception obligations JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--fingerprint-handledness-json",
        default=None,
        help="Write fingerprint handledness JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--emit-decision-snapshot",
        default=None,
        help="Write decision surface snapshot JSON to file or '-' for stdout.",
    )
    parser.add_argument("--report", default=None, help="Write Markdown report (mermaid) to file.")
    parser.add_argument(
        "--lint",
        action="store_true",
        help="Emit lint-style lines (path:line:col: CODE message).",
    )
    parser.add_argument("--max-components", type=int, default=10, help="Max components in report.")
    parser.add_argument(
        "--type-audit",
        action="store_true",
        help="Emit type-tightening suggestions based on downstream annotations.",
    )
    parser.add_argument(
        "--type-audit-max",
        type=int,
        default=50,
        help="Max type-tightening entries to print.",
    )
    parser.add_argument(
        "--type-audit-report",
        action="store_true",
        help="Include type-flow audit summary in the markdown report.",
    )
    parser.add_argument(
        "--wl-refinement",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Emit WL refinement facets over SuiteSite containment.",
    )
    parser.add_argument(
        "--fail-on-type-ambiguities",
        action="store_true",
        help="Exit non-zero if type ambiguities are detected.",
    )
    parser.add_argument(
        "--fail-on-violations",
        action="store_true",
        help="Exit non-zero if undocumented/undeclared bundle violations are detected.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Baseline file of violations to allow (ratchet mode).",
    )
    parser.add_argument(
        "--baseline-write",
        action="store_true",
        help="Write the current violations to the baseline file and exit zero.",
    )
    parser.add_argument(
        "--synthesis-plan",
        default=None,
        help="Write synthesis plan JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--synthesis-report",
        action="store_true",
        help="Include synthesis plan summary in the markdown report.",
    )
    parser.add_argument(
        "--synthesis-protocols",
        default=None,
        help="Write protocol/dataclass stubs to file or '-' for stdout.",
    )
    parser.add_argument(
        "--synthesis-protocols-kind",
        choices=["dataclass", "protocol", "contextvar"],
        default="dataclass",
        help="Emit dataclass, typing.Protocol, or ContextVar stubs (default: dataclass).",
    )
    parser.add_argument(
        "--refactor-plan",
        action="store_true",
        help="Include refactoring plan summary in the markdown report.",
    )
    parser.add_argument(
        "--refactor-plan-json",
        default=None,
        help="Write refactoring plan JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--synthesis-max-tier",
        type=int,
        default=2,
        help="Max tier to include in synthesis plan.",
    )
    parser.add_argument(
        "--synthesis-min-bundle-size",
        type=int,
        default=2,
        help="Min bundle size to include in synthesis plan.",
    )
    parser.add_argument(
        "--synthesis-allow-singletons",
        action="store_true",
        help="Allow single-field bundles in synthesis plan.",
    )
    parser.add_argument(
        "--synthesis-merge-overlap",
        type=float,
        default=None,
        help="Jaccard overlap threshold for merging bundles (0.0-1.0).",
    )
    parser.add_argument(
        "--synthesis-property-hook-min-confidence",
        type=float,
        default=0.7,
        help="Minimum invariant confidence required for property-hook emission.",
    )
    parser.add_argument(
        "--synthesis-property-hook-hypothesis",
        action="store_true",
        help="Include optional Hypothesis skeletons in property-hook manifest output.",
    )
    parser.add_argument(
        "--analysis-timeout-ticks",
        type=int,
        default=60_000,
        help="Deadline budget in ticks for standalone analysis execution.",
    )
    parser.add_argument(
        "--analysis-timeout-tick-ns",
        type=int,
        default=1_000_000,
        help="Nanoseconds per timeout tick for standalone analysis execution.",
    )
    parser.add_argument(
        "--analysis-tick-limit",
        type=int,
        default=None,
        help="Optional deterministic logical gas budget (ticks).",
    )
    return parser


def _normalize_transparent_decorators(
    value: object,
) -> object:
    check_deadline()
    if value is not None:
        items: list[str] = []
        value_type = type(value)
        if value_type is str:
            items = [part.strip() for part in cast(str, value).split(",") if part.strip()]
        elif value_type in {list, tuple, set}:
            for item in cast(Iterable[object], value):
                check_deadline()
                if type(item) is str:
                    parts = [part.strip() for part in cast(str, item).split(",") if part.strip()]
                    items.extend(parts)
        if items:
            return set(items)
    return None


@contextmanager
def _analysis_deadline_scope(args: argparse.Namespace):
    timeout_ticks = int(args.analysis_timeout_ticks)
    timeout_tick_ns = int(args.analysis_timeout_tick_ns)
    if timeout_ticks <= 0:
        never("invalid analysis timeout ticks", analysis_timeout_ticks=timeout_ticks)
    if timeout_tick_ns <= 0:
        never("invalid analysis timeout tick_ns", analysis_timeout_tick_ns=timeout_tick_ns)
    tick_limit_value = args.analysis_tick_limit
    logical_limit = timeout_ticks
    if tick_limit_value is not None:
        tick_limit = int(tick_limit_value)
        if tick_limit <= 0:
            never("invalid analysis tick limit", analysis_tick_limit=tick_limit)
        logical_limit = min(logical_limit, tick_limit)
    with ExitStack() as stack:
        stack.enter_context(forest_scope(Forest()))
        stack.enter_context(
            deadline_scope(Deadline.from_timeout_ticks(timeout_ticks, timeout_tick_ns))
        )
        stack.enter_context(deadline_clock_scope(GasMeter(limit=logical_limit)))
        yield


def _run_impl(
    args: argparse.Namespace,
    *,
    analyze_paths_fn: Callable[..., AnalysisResult] = analyze_paths,
    emit_report_fn: Callable[..., tuple[str, list[str]]] = _emit_report,
    compute_violations_fn: Callable[..., list[str]] = _compute_violations,
) -> int:
    check_deadline()
    if args.fail_on_type_ambiguities:
        args.type_audit = True
    fingerprint_deadness_json = args.fingerprint_deadness_json
    fingerprint_coherence_json = args.fingerprint_coherence_json
    fingerprint_rewrite_plans_json = args.fingerprint_rewrite_plans_json
    fingerprint_exception_obligations_json = args.fingerprint_exception_obligations_json
    fingerprint_handledness_json = args.fingerprint_handledness_json
    exclude_dirs = None
    if args.exclude is not None:
        exclude_dirs = []
        for entry in args.exclude:
            check_deadline()
            for part in entry.split(","):
                check_deadline()
                part = part.strip()
                if part:
                    exclude_dirs.append(part)
    ignore_params = None
    if args.ignore_params is not None:
        ignore_params = [p.strip() for p in args.ignore_params.split(",") if p.strip()]
    transparent_decorators = None
    if args.transparent_decorators is not None:
        transparent_decorators = [
            p.strip() for p in args.transparent_decorators.split(",") if p.strip()
        ]
    config_path = Path(args.config) if args.config else None
    defaults = dataflow_defaults(Path(args.root), config_path)
    synth_defaults = synthesis_defaults(Path(args.root), config_path)
    decision_section = decision_defaults(Path(args.root), config_path)
    decision_tiers = decision_tier_map(decision_section)
    decision_require = decision_require_tiers(decision_section)
    exception_section = exception_defaults(Path(args.root), config_path)
    never_exceptions = set(exception_never_list(exception_section))
    fingerprint_section = fingerprint_defaults(Path(args.root), config_path)
    synth_min_occurrences = 0
    synth_version = "synth@1"
    synth_registry_path = None
    fingerprint_seed_path = None
    try:
        synth_min_occurrences = int(
            fingerprint_section.get("synth_min_occurrences", 0) or 0
        )
    except (TypeError, ValueError):
        synth_min_occurrences = 0
    synth_version = str(
        fingerprint_section.get("synth_version", synth_version) or synth_version
    )
    synth_registry_path = fingerprint_section.get("synth_registry_path")
    fingerprint_seed_path = fingerprint_section.get("seed_registry_path")
    if fingerprint_seed_path is None:
        fingerprint_seed_path = fingerprint_section.get("fingerprint_seed_path")
    fingerprint_registry = None
    fingerprint_index: dict[Fingerprint, set[str]] = {}
    fingerprint_seed_revision = None
    constructor_registry = None
    synth_registry = None
    # The [fingerprints] section mixes bundle specs with synth settings.
    # Filter out the settings so they do not pollute the registry/index.
    fingerprint_spec: dict[str, JSONValue] = {
        key: value
        for key, value in fingerprint_section.items()
        if (
            not str(key).startswith("synth_")
            and not str(key).startswith("seed_")
            and str(key) != "fingerprint_seed_path"
        )
    }
    seed_revision = fingerprint_section.get("seed_revision")
    if seed_revision is None:
        seed_revision = fingerprint_section.get("registry_seed_revision")
    if seed_revision is not None:
        fingerprint_seed_revision = str(seed_revision)
    if fingerprint_spec:
        seed_payload: object = None
        if fingerprint_seed_path:
            resolved_seed = _resolve_synth_registry_path(
                str(fingerprint_seed_path), Path(args.root)
            )
            if resolved_seed is not None:
                try:
                    seed_payload = load_json(resolved_seed)
                except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
                    seed_payload = None
        registry, index = build_fingerprint_registry(
            fingerprint_spec,
            registry_seed=seed_payload,
        )
        if index:
            fingerprint_registry = registry
            fingerprint_index = index
            constructor_registry = TypeConstructorRegistry(registry)
            if synth_registry_path:
                resolved = _resolve_synth_registry_path(
                    str(synth_registry_path), Path(args.root)
                )
                if resolved is not None:
                    try:
                        payload = load_json(resolved)
                    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
                        payload = None
                else:
                    payload = None
                payload_mapping = mapping_or_none(cast(JSONValue, payload))
                if payload_mapping is not None:
                    synth_registry = build_synth_registry_from_payload(
                        payload_mapping, registry
                    )
    merged = merge_payload(
        {
            "exclude": exclude_dirs,
            "ignore_params": ignore_params,
            "allow_external": args.allow_external,
            "strictness": args.strictness,
            "baseline": args.baseline,
            "transparent_decorators": transparent_decorators,
        },
        defaults,
    )
    exclude_dirs = set(merged.get("exclude", []) or [])
    ignore_params_set = set(merged.get("ignore_params", []) or [])
    decision_ignore_params = set(ignore_params_set)
    decision_ignore_params.update(decision_ignore_list(decision_section))
    allow_external = bool(merged.get("allow_external", False))
    strictness = merged.get("strictness") or "high"
    if strictness not in {"high", "low"}:
        strictness = "high"
    transparent_decorators = _normalize_transparent_decorators(
        merged.get("transparent_decorators")
    )
    deadline_roots = set(dataflow_deadline_roots(merged))
    config = AuditConfig(
        project_root=Path(args.root),
        exclude_dirs=exclude_dirs,
        ignore_params=ignore_params_set,
        decision_ignore_params=decision_ignore_params,
        external_filter=not allow_external,
        strictness=strictness,
        transparent_decorators=transparent_decorators,
        decision_tiers=decision_tiers,
        decision_require_tiers=decision_require,
        never_exceptions=never_exceptions,
        deadline_roots=deadline_roots,
        fingerprint_registry=fingerprint_registry,
        fingerprint_index=fingerprint_index,
        constructor_registry=constructor_registry,
        fingerprint_seed_revision=fingerprint_seed_revision,
        fingerprint_synth_min_occurrences=synth_min_occurrences,
        fingerprint_synth_version=synth_version,
        fingerprint_synth_registry=synth_registry,
    )
    baseline_path = _resolve_baseline_path(merged.get("baseline"), Path(args.root))
    baseline_write = args.baseline_write
    if baseline_write and baseline_path is None:
        print("Baseline path required for --baseline-write.", file=sys.stderr)
        return 2
    paths = _iter_paths(args.paths, config)
    decision_snapshot_path = args.emit_decision_snapshot
    include_decisions = bool(args.report) or bool(decision_snapshot_path) or bool(
        args.fail_on_violations
    )
    if decision_tiers:
        include_decisions = True
    include_rewrite_plans = bool(args.report) or bool(fingerprint_rewrite_plans_json)
    include_exception_obligations = bool(args.report) or bool(
        fingerprint_exception_obligations_json
    )
    include_handledness_witnesses = bool(args.report) or bool(
        fingerprint_handledness_json
    )
    include_never_invariants = bool(args.report)
    include_wl_refinement = bool(args.wl_refinement)
    include_ambiguities = bool(args.report) or bool(args.lint)
    include_coherence = (
        bool(args.report)
        or bool(fingerprint_coherence_json)
        or include_rewrite_plans
    )
    forest = Forest()
    analysis = analyze_paths_fn(
        paths,
        forest=forest,
        recursive=not args.no_recursive,
        type_audit=args.type_audit or args.type_audit_report,
        type_audit_report=args.type_audit_report,
        type_audit_max=args.type_audit_max,
        include_constant_smells=bool(args.report),
        include_unused_arg_smells=bool(args.report),
        include_deadness_witnesses=bool(args.report) or bool(fingerprint_deadness_json),
        include_coherence_witnesses=include_coherence,
        include_rewrite_plans=include_rewrite_plans,
        include_exception_obligations=include_exception_obligations,
        include_handledness_witnesses=include_handledness_witnesses,
        include_never_invariants=include_never_invariants,
        include_wl_refinement=include_wl_refinement,
        include_deadline_obligations=bool(args.report) or bool(args.lint),
        include_decision_surfaces=include_decisions,
        include_value_decision_surfaces=include_decisions,
        include_invariant_propositions=(
            bool(args.report)
            or bool(args.synthesis_plan)
            or bool(args.synthesis_report)
            or bool(args.synthesis_property_hook_hypothesis)
        ),
        include_lint_lines=bool(args.lint),
        include_ambiguities=include_ambiguities,
        include_bundle_forest=bool(args.report)
        or bool(args.dot)
        or bool(args.fail_on_violations)
        or bool(args.emit_structure_tree)
        or bool(args.emit_structure_metrics)
        or bool(args.emit_decision_snapshot),
        config=config,
    )

    return _finalize_run_outputs_impl(
        context=_RunImplOutputContextCore(
            args=args,
            analysis=analysis,
            paths=paths,
            config=config,
            synth_defaults=synth_defaults,
            baseline_path=baseline_path,
            baseline_write=baseline_write,
            decision_snapshot_path=decision_snapshot_path,
            fingerprint_deadness_json=fingerprint_deadness_json,
            fingerprint_coherence_json=fingerprint_coherence_json,
            fingerprint_rewrite_plans_json=fingerprint_rewrite_plans_json,
            fingerprint_exception_obligations_json=(
                fingerprint_exception_obligations_json
            ),
            fingerprint_handledness_json=fingerprint_handledness_json,
        ),
        emit_report_fn=emit_report_fn,
        compute_violations_fn=compute_violations_fn,
    ).exit_code


def run(argv = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    with _analysis_deadline_scope(args):
        check_deadline()
        return _run_impl(args)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
