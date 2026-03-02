# gabion:boundary_normalization_module
from __future__ import annotations

"""Owned indexed file-scan and helper surfaces extracted from runtime.

This module is the canonical owner for the indexed/file-scan helper graph
used by analysis, deadline, ambiguity, and reporting surfaces during
final runtime retirement.
"""


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
    execution_signature,
    mismatch_residue_payload,
)

from gabion.ingest.python_ingest import ingest_python_file, iter_python_paths

from gabion.analysis.visitors import ImportVisitor, ParentAnnotator, UseVisitor

from gabion.analysis.evidence import (
    Site,
    exception_obligation_summary_for_site,
    normalize_bundle_key,
)

from gabion.analysis.json_types import JSONObject, JSONValue

from gabion.analysis.schema_audit import find_anonymous_schema_surfaces

from gabion.analysis.aspf import Alt, Forest, Node, NodeId, structural_key_atom, structural_key_json

from gabion.analysis.derivation_cache import get_global_derivation_cache

from gabion.analysis.derivation_contract import DerivationOp

from gabion.analysis import evidence_keys

from gabion.exceptions import NeverThrown

from gabion.invariants import never, require_not_none

from gabion.order_contract import OrderPolicy, sort_once

from gabion.config import (
    dataflow_defaults,
    dataflow_adapter_payload,
    dataflow_deadline_roots,
    dataflow_required_surfaces,
    decision_defaults,
    decision_ignore_list,
    decision_require_tiers,
    decision_tier_map,
    exception_defaults,
    exception_marker_family,
    exception_never_list,
    fingerprint_defaults,
    merge_payload,
    synthesis_defaults,
)

from gabion.analysis.marker_protocol import (
    DEFAULT_MARKER_ALIASES,
    MarkerKind,
    MarkerLifecycleState,
    marker_identity,
    normalize_marker_payload,
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
    fingerprint_stage_cache_identity,
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
    TimeoutTickCarrier,
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

from .semantic_primitives import (
    AnalysisPassPrerequisites,
    CallArgumentMapping,
    CallableId,
    DecisionPredicateEvidence,
    ParameterId,
    SpanIdentity,
)
from .dataflow_contracts import InvariantProposition

from .dataflow_report_rendering import (
    render_unsupported_by_adapter_section as _report_render_unsupported_section,
    render_synthesis_section as _report_render_synthesis_section,
)

from .dataflow_snapshot_contracts import (
    DecisionSnapshotSurfaces,
    StructureSnapshotDiffRequest,
)

from .pattern_schema_projection import (
    bundle_pattern_instances as _bundle_pattern_instances_impl,
    detect_execution_pattern_matches as _detect_execution_pattern_matches_impl,
    execution_pattern_instances as _execution_pattern_instances_impl,
    execution_pattern_suggestions as _execution_pattern_suggestions_impl,
    pattern_schema_matches as _pattern_schema_matches_impl,
    pattern_schema_residue_entries as _pattern_schema_residue_entries_impl,
    pattern_schema_residue_lines as _pattern_schema_residue_lines_impl,
    pattern_schema_snapshot_entries as _pattern_schema_snapshot_entries_impl,
    pattern_schema_suggestions as _pattern_schema_suggestions_impl,
    pattern_schema_suggestions_from_instances as _pattern_schema_suggestions_from_instances_impl,
    tier2_unreified_residue_entries as _tier2_unreified_residue_entries_impl,
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

@dataclass(frozen=True)
class AnnotationValue:
    text: str
    parse_status: Literal["present", "missing", "unparse_failure"]

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

OptionalAstNode = ast.AST | None

OptionalAstCall = ast.Call | None

NodeIdOrNone = NodeId | None

ParseCacheValue = ast.Module | BaseException

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

    def __post_init__(self) -> None:
        if set(self.pos_map) & set(self.const_pos):
            never("positional slot cannot be both param and constant")  # pragma: no cover - invariant sink
        if set(self.pos_map) & set(self.non_const_pos):
            never("positional slot cannot be both param and non-const")  # pragma: no cover - invariant sink
        if set(self.const_pos) & set(self.non_const_pos):
            never("positional slot cannot be both const and non-const")  # pragma: no cover - invariant sink
        if set(self.kw_map) & set(self.const_kw):
            never("keyword slot cannot be both param and constant")  # pragma: no cover - invariant sink
        if set(self.kw_map) & set(self.non_const_kw):
            never("keyword slot cannot be both param and non-const")  # pragma: no cover - invariant sink
        if set(self.const_kw) & set(self.non_const_kw):
            never("keyword slot cannot be both const and non-const")  # pragma: no cover - invariant sink

    def callable_id(self) -> CallableId:
        return CallableId.from_raw(self.callee)

    def argument_mapping(self) -> CallArgumentMapping:
        positional = {
            int(idx): ParameterId.from_raw(param)
            for idx, param in self.pos_map.items()
        }
        keywords = {
            key: ParameterId.from_raw(param)
            for key, param in self.kw_map.items()
        }
        return CallArgumentMapping(
            positional=positional,
            keywords=keywords,
            star_positional=tuple(
                (idx, ParameterId.from_raw(param)) for idx, param in self.star_pos
            ),
            star_keywords=tuple(ParameterId.from_raw(param) for param in self.star_kw),
        )

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

@dataclass(frozen=True)
class AdapterCapabilities:
    bundle_inference: bool = True
    decision_surfaces: bool = True
    type_flow: bool = True
    exception_obligations: bool = True
    rewrite_plan_support: bool = True

def parse_adapter_capabilities(payload: object) -> AdapterCapabilities:
    if type(payload) is not dict:
        return AdapterCapabilities()
    raw = cast(dict[object, object], payload)

    def _read(name: str, default: bool = True) -> bool:
        value = raw.get(name)
        if type(value) is bool:
            return bool(value)
        return default

    return AdapterCapabilities(
        bundle_inference=_read("bundle_inference"),
        decision_surfaces=_read("decision_surfaces"),
        type_flow=_read("type_flow"),
        exception_obligations=_read("exception_obligations"),
        rewrite_plan_support=_read("rewrite_plan_support"),
    )

def normalize_adapter_contract(payload: object) -> JSONObject:
    if type(payload) is not dict:
        return {"name": "native", "capabilities": AdapterCapabilities().__dict__}
    raw = cast(dict[object, object], payload)
    name = str(raw.get("name", "native") or "native")
    capabilities = parse_adapter_capabilities(raw.get("capabilities")).__dict__
    return {"name": name, "capabilities": {str(key): bool(capabilities[key]) for key in capabilities}}

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
    adapter_contract: OptionalJsonObject = None
    required_analysis_surfaces: set[str] = field(default_factory=set)

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
    unsupported_by_adapter: list[JSONObject] = field(default_factory=list)

_ANALYSIS_PROFILING_FORMAT_VERSION = 1


def _iter_dataclass_call_bundle_effects_impl(*args, **kwargs):
    from .dataflow_bundle_iteration import iter_dataclass_call_bundle_effects

    return iter_dataclass_call_bundle_effects(*args, **kwargs)


def _CalleeResolutionContextCore(*args, **kwargs):
    from .dataflow_callee_resolution import CalleeResolutionContext

    return CalleeResolutionContext(*args, **kwargs)


def _collect_callee_resolution_effects_impl(*args, **kwargs):
    from .dataflow_callee_resolution import collect_callee_resolution_effects

    return collect_callee_resolution_effects(*args, **kwargs)


def _resolve_callee_with_effects_impl(*args, **kwargs):
    from .dataflow_callee_resolution import resolve_callee_with_effects

    return resolve_callee_with_effects(*args, **kwargs)


def _RunImplOutputContextCore(*args, **kwargs):
    from .dataflow_run_outputs import DataflowRunOutputContext

    return DataflowRunOutputContext(*args, **kwargs)


def _finalize_run_outputs_impl(*args, **kwargs):
    from .dataflow_run_outputs import finalize_run_outputs

    return finalize_run_outputs(*args, **kwargs)


def analyze_paths(*args, **kwargs):
    from .dataflow_pipeline import analyze_paths as _analyze_paths

    return _analyze_paths(*args, **kwargs)


def _emit_report(*args, **kwargs):
    from .dataflow_reporting import emit_report

    return emit_report(*args, **kwargs)


def _resolve_class_candidates(*args, **kwargs):
    from .dataflow_evidence_helpers import _resolve_class_candidates as _impl

    return _impl(*args, **kwargs)


def _resolve_method_in_hierarchy(*args, **kwargs):
    from .dataflow_evidence_helpers import _resolve_method_in_hierarchy as _impl

    resolution = _impl(*args, **kwargs)
    if getattr(resolution, "kind", "") == "found" and hasattr(resolution, "resolved"):
        return resolution.resolved
    return None


def _internal_broad_type_lint_lines(*args, **kwargs):
    from .dataflow_lint_helpers import _internal_broad_type_lint_lines as _impl

    if kwargs.get("analysis_index") is None and args:
        paths = list(args[0])
        kwargs["analysis_index"] = _build_analysis_index(
            paths,
            project_root=kwargs["project_root"],
            ignore_params=set(kwargs["ignore_params"]),
            strictness=kwargs["strictness"],
            external_filter=bool(kwargs["external_filter"]),
            transparent_decorators=kwargs.get("transparent_decorators"),
            parse_failure_witnesses=list(kwargs.get("parse_failure_witnesses", [])),
        )
    return _impl(*args, **kwargs)


def _is_broad_internal_type(*args, **kwargs):
    from .dataflow_lint_helpers import _is_broad_internal_type as _impl

    return _impl(*args, **kwargs)


def _normalize_type_name(*args, **kwargs):
    from .dataflow_lint_helpers import _normalize_type_name as _impl

    return _impl(*args, **kwargs)


def _exception_protocol_lint_lines(*args, **kwargs):
    from .dataflow_lint_helpers import _exception_protocol_lint_lines as _impl

    return _impl(*args, **kwargs)


def _lint_lines_from_bundle_evidence(*args, **kwargs):
    from .dataflow_lint_helpers import _lint_lines_from_bundle_evidence as _impl

    return _impl(*args, **kwargs)


def _lint_lines_from_constant_smells(*args, **kwargs):
    from .dataflow_lint_helpers import _lint_lines_from_constant_smells as _impl

    return _impl(*args, **kwargs)


def _lint_lines_from_type_evidence(*args, **kwargs):
    from .dataflow_lint_helpers import _lint_lines_from_type_evidence as _impl

    return _impl(*args, **kwargs)


def _lint_lines_from_unused_arg_smells(*args, **kwargs):
    from .dataflow_lint_helpers import _lint_lines_from_unused_arg_smells as _impl

    return _impl(*args, **kwargs)


def _parse_exception_path_id(*args, **kwargs):
    from .dataflow_lint_helpers import _parse_exception_path_id as _impl

    return _impl(*args, **kwargs)


def _parse_lint_location(*args, **kwargs):
    from .dataflow_lint_helpers import _parse_lint_location as _impl

    return _impl(*args, **kwargs)


def _compute_fingerprint_coherence(*args, **kwargs):
    from .dataflow_fingerprint_helpers import _compute_fingerprint_coherence as _impl

    return _impl(*args, **kwargs)


def _compute_fingerprint_rewrite_plans(*args, **kwargs):
    from .dataflow_fingerprint_helpers import _compute_fingerprint_rewrite_plans as _impl

    return _impl(*args, **kwargs)


def _find_provenance_entry_for_site(*args, **kwargs):
    from .dataflow_fingerprint_helpers import _find_provenance_entry_for_site as _impl

    return _impl(*args, **kwargs)


def _glossary_match_strata(*args, **kwargs):
    from .dataflow_fingerprint_helpers import _glossary_match_strata as _impl

    return _impl(*args, **kwargs)


def verify_rewrite_plan(*args, **kwargs):
    from .dataflow_fingerprint_helpers import verify_rewrite_plan as _impl

    return _impl(*args, **kwargs)


def verify_rewrite_plans(*args, **kwargs):
    from .dataflow_fingerprint_helpers import verify_rewrite_plans as _impl

    return _impl(*args, **kwargs)


def _merge_counts_by_knobs(*args, **kwargs):
    from .dataflow_synthesis import _merge_counts_by_knobs as _impl

    return _impl(*args, **kwargs)


def _render_mermaid_component(*args, **kwargs):
    from .dataflow_reporting_helpers import render_mermaid_component as _impl

    return _impl(*args, **kwargs)


def _topologically_order_report_projection_specs(*args, **kwargs):
    from .dataflow_projection_helpers import _topologically_order_report_projection_specs as _impl

    return _impl(*args, **kwargs)


def _build_synth_registry_payload(*args, **kwargs):
    from .dataflow_fingerprint_helpers import _build_synth_registry_payload as _impl

    return _impl(*args, **kwargs)


def _compute_fingerprint_matches(*args, **kwargs):
    from .dataflow_fingerprint_helpers import _compute_fingerprint_matches as _impl

    return _impl(*args, **kwargs)


def _compute_fingerprint_provenance(*args, **kwargs):
    from .dataflow_fingerprint_helpers import _compute_fingerprint_provenance as _impl

    return _impl(*args, **kwargs)


def _compute_fingerprint_synth(*args, **kwargs):
    from .dataflow_fingerprint_helpers import _compute_fingerprint_synth as _impl

    return _impl(*args, **kwargs)


def _compute_fingerprint_warnings(*args, **kwargs):
    from .dataflow_fingerprint_helpers import _compute_fingerprint_warnings as _impl

    return _impl(*args, **kwargs)


def _fingerprint_soundness_issues(*args, **kwargs):
    from .dataflow_fingerprint_helpers import _fingerprint_soundness_issues as _impl

    return _impl(*args, **kwargs)


def _summarize_fingerprint_provenance(*args, **kwargs):
    from .dataflow_fingerprint_helpers import _summarize_fingerprint_provenance as _impl

    return _impl(*args, **kwargs)


def _collect_fingerprint_atom_keys(*args, **kwargs):
    from .dataflow_fingerprint_helpers import _collect_fingerprint_atom_keys as _impl

    return _impl(*args, **kwargs)


def render_report(*args, **kwargs):
    from .dataflow_reporting import render_report as _impl

    return _impl(*args, **kwargs)


def _collect_deadline_obligations(*args, **kwargs):
    from .dataflow_obligations import collect_deadline_obligations as _impl

    return _impl(*args, **kwargs)


def _deadline_lint_lines(*args, **kwargs):
    from .dataflow_lint_helpers import _deadline_lint_lines as _impl

    return _impl(*args, **kwargs)


def _summarize_deadline_obligations(entries, *, max_entries=20, forest):
    check_deadline()
    if not entries:
        return []
    spec_hash = projection_spec_hash(DEADLINE_OBLIGATIONS_SUMMARY_SPEC)
    spec_site = forest.add_spec_site(
        spec_hash=spec_hash,
        spec_name=str(DEADLINE_OBLIGATIONS_SUMMARY_SPEC.name),
        spec_domain=str(DEADLINE_OBLIGATIONS_SUMMARY_SPEC.domain),
        spec_version=int(DEADLINE_OBLIGATIONS_SUMMARY_SPEC.spec_version),
    )
    lines: list[str] = []
    for entry in entries[:max_entries]:
        check_deadline()
        site_payload = entry.get("site", {}) if type(entry) is dict else {}
        site = cast(Mapping[str, JSONValue], site_payload if type(site_payload) is dict else {})
        path = str(site.get("path", "?") or "?")
        function = str(site.get("function", "?") or "?")
        parsed_span = require_not_none(
            int_tuple4_or_none(entry.get("span") if type(entry) is dict else None),
            reason="deadline summary requires valid span",
            strict=True,
        )
        suite_kind = str(site.get("suite_kind", "function") or "function")
        status = entry.get("status", "UNKNOWN") if type(entry) is dict else "UNKNOWN"
        kind = entry.get("kind", "?") if type(entry) is dict else "?"
        detail = entry.get("detail", "") if type(entry) is dict else ""
        suite_site = forest.add_suite_site(path, function, "spec", span=parsed_span)
        forest.add_alt(
            "SpecFacet",
            (spec_site, suite_site),
            evidence={
                "spec_hash": spec_hash,
                "spec_name": str(DEADLINE_OBLIGATIONS_SUMMARY_SPEC.name),
                "status": status,
                "kind": kind,
            },
        )
        span_text = _format_span_fields(*parsed_span)
        suffix = f"@{span_text}" if span_text else ""
        line = f"{path}:{function}{suffix} status={status} kind={kind} {detail}".strip()
        lines.append(line)
    if len(entries) > max_entries:
        lines.append(f"... {len(entries) - max_entries} more")
    return lines


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
    return iter_python_paths(
        paths,
        config=config,
        check_deadline=check_deadline,
        sort_once=sort_once,
    )

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
    source = 'gabion.analysis.dataflow_indexed_file_scan.generate_property_hook_manifest.site_1'):
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
        source = 'gabion.analysis.dataflow_indexed_file_scan.generate_property_hook_manifest.site_2')
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
            "hook_ids": sort_once(hook_ids, source = 'gabion.analysis.dataflow_indexed_file_scan._build_property_hook_callable_index.site_1'),
        }
        for scope, hook_ids in sort_once(callables.items(), source = 'gabion.analysis.dataflow_indexed_file_scan._build_property_hook_callable_index.site_2')
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

def _is_marker_call(call: ast.Call, aliases: set[str]) -> bool:
    name = _decorator_name(call.func)
    if not name:
        return False
    return _decorator_matches(name, aliases)

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

def _decision_predicate_evidence(
    info: FunctionInfo,
    param: str,
) -> DecisionPredicateEvidence:
    reasons = tuple(
        sort_once(
            info.decision_surface_reasons.get(param, set()),
            source="_decision_predicate_evidence.reasons",
        )
    )
    span = info.param_spans.get(param)
    return DecisionPredicateEvidence(
        parameter=ParameterId.from_raw(param),
        reasons=reasons,
        spans=(SpanIdentity.from_tuple(span),) if span is not None else (),
    )

def _decision_reason_summary(info: FunctionInfo, params: Iterable[str]) -> str:
    labels: set[str] = set()
    for param in params:
        check_deadline()
        evidence = _decision_predicate_evidence(info, param)
        labels.update(evidence.reasons)
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

def _suite_site_label(*, forest: Forest, suite_id: NodeId) -> str:
    suite_node = forest.nodes.get(suite_id)
    if suite_node is None:
        never("suite site missing during label projection", suite_id=str(suite_id))  # pragma: no cover - invariant sink
    path = str(suite_node.meta.get("path", "") or "")
    qual = str(suite_node.meta.get("qual", "") or "")
    suite_kind = str(suite_node.meta.get("suite_kind", "") or "")
    span = int_tuple4_or_none(suite_node.meta.get("span"))
    if not path or not qual or not suite_kind or span is None:
        never(  # pragma: no cover - invariant sink
            "suite site label projection missing identity",
            path=path,
            qual=qual,
            suite_kind=suite_kind,
            span=suite_node.meta.get("span"),
        )
    span_text = _format_span_fields(*span)
    return f"{path}:{qual}[{suite_kind}]@{span_text}" if span_text else f"{path}:{qual}[{suite_kind}]"

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
        suite_id = forest.add_suite_site(
            info.path.name,
            info.qual,
            "function_body",
            span=info.function_span,
        )
        paramset_id = forest.add_paramset(params)
        reason_summary = (
            _decision_reason_summary(info, params)
            if spec.pass_id == "decision_surfaces"
            else descriptor
        )
        forest.add_alt(
            spec.alt_kind,
            (suite_id, paramset_id),
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
            f"{_suite_site_label(forest=forest, suite_id=suite_id)} {spec.surface_label}: "
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
        annotation_value = _extract_annotation_value(arg.annotation)
        annots[name] = annotation_value.text if annotation_value.parse_status == "present" else None
    if fn.args.vararg:
        vararg = fn.args.vararg
        annotation_value = _extract_annotation_value(vararg.annotation)
        annots[vararg.arg] = annotation_value.text if annotation_value.parse_status == "present" else None
    if fn.args.kwarg:
        kwarg = fn.args.kwarg
        annotation_value = _extract_annotation_value(kwarg.annotation)
        annots[kwarg.arg] = annotation_value.text if annotation_value.parse_status == "present" else None
    if names and names[0] in {"self", "cls"}:
        annots.pop(names[0], None)
    if ignore_params:
        for name in list(annots.keys()):
            check_deadline()
            if name in ignore_params:
                annots.pop(name, None)
    return annots

def _extract_annotation_value(annotation: OptionalAstNode) -> AnnotationValue:
    check_deadline()
    if annotation is None:
        return AnnotationValue(text="", parse_status="missing")
    try:
        return AnnotationValue(
            text=ast.unparse(annotation),
            parse_status="present",
        )
    except _AST_UNPARSE_ERROR_TYPES:
        return AnnotationValue(text="", parse_status="unparse_failure")

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

_ANALYSIS_INDEX_STAGE_CACHE_OP = DerivationOp(
    name="analysis_index.stage_cache",
    version=1,
    scope="gabion.analysis.dataflow_indexed_file_scan",
)

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
    source = 'gabion.analysis.dataflow_indexed_file_scan._collect_handledness_witnesses.site_1')

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
    source = 'gabion.analysis.dataflow_indexed_file_scan._collect_exception_obligations.site_1')

def _keyword_string_literal(call: ast.Call, key: str) -> str:
    check_deadline()
    for kw in call.keywords:
        check_deadline()
        if kw.arg != key:
            continue
        kw_value = kw.value
        if type(kw_value) is ast.Constant:
            constant_value = cast(ast.Constant, kw_value).value
            if type(constant_value) is str:
                return constant_value
    return ""

def _keyword_links_literal(call: ast.Call) -> list[JSONObject]:
    check_deadline()
    for kw in call.keywords:
        check_deadline()
        if kw.arg != "links":
            continue
        kw_value = kw.value
        if type(kw_value) is not ast.List:
            return []
        links: list[JSONObject] = []
        for item in cast(ast.List, kw_value).elts:
            check_deadline()
            if type(item) is not ast.Dict:
                continue
            dict_node = cast(ast.Dict, item)
            payload: JSONObject = {}
            for raw_key, raw_value in zip(dict_node.keys, dict_node.values, strict=False):
                check_deadline()
                if type(raw_key) is not ast.Constant:
                    continue
                if type(raw_value) is not ast.Constant:
                    continue
                key_value = cast(ast.Constant, raw_key).value
                value_value = cast(ast.Constant, raw_value).value
                if type(key_value) is str and type(value_value) is str:
                    payload[key_value] = value_value
            kind = str(payload.get("kind", "")).strip()
            value = str(payload.get("value", "")).strip()
            if kind and value:
                links.append({"kind": kind, "value": value})
        return sort_once(
            links,
            key=lambda item: (str(item.get("kind", "")), str(item.get("value", ""))),
            source="_keyword_links_literal",
        )
    return []

def _never_marker_metadata(
    call: ast.Call,
    never_id: str,
    reason: str,
    *,
    marker_kind: MarkerKind = MarkerKind.NEVER,
) -> JSONObject:
    check_deadline()
    owner = _keyword_string_literal(call, "owner")
    expiry = _keyword_string_literal(call, "expiry")
    links = _keyword_links_literal(call)
    payload = normalize_marker_payload(
        reason=reason,
        env={},
        marker_kind=marker_kind,
        owner=owner,
        expiry=expiry,
        lifecycle_state=MarkerLifecycleState.ACTIVE,
        links=tuple(cast(dict[str, str], link) for link in links),
    )
    return {
        "marker_kind": marker_kind.value,
        "marker_id": marker_identity(payload),
        "marker_site_id": never_id,
        "owner": owner,
        "expiry": expiry,
        "links": links,
    }

def _marker_alias_kind_map(marker_aliases: Sequence[str]) -> tuple[set[str], dict[str, MarkerKind]]:
    check_deadline()
    alias_map: dict[str, MarkerKind] = {}
    for marker_kind, aliases in DEFAULT_MARKER_ALIASES.items():
        check_deadline()
        for alias in aliases:
            check_deadline()
            alias_map[alias] = marker_kind
            alias_map[alias.split(".")[-1]] = marker_kind
    active_aliases = set(marker_aliases)
    if not active_aliases:
        active_aliases = set(alias_map.keys())
    else:
        for alias in active_aliases:
            check_deadline()
            alias_map.setdefault(alias, MarkerKind.NEVER)
            if "." in alias:
                alias_map.setdefault(alias.split(".")[-1], MarkerKind.NEVER)
    return active_aliases, alias_map

def _marker_kind_for_call(call: ast.Call, alias_map: Mapping[str, MarkerKind]) -> MarkerKind:
    check_deadline()
    name = _decorator_name(call.func) or ""
    if not name:
        return MarkerKind.NEVER
    if name in alias_map:
        return alias_map[name]
    short = name.split(".")[-1]
    return alias_map.get(short, MarkerKind.NEVER)

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
    marker_aliases: Sequence[str] = (),
    deadness_witnesses=None,
) -> list[JSONObject]:
    check_deadline()
    invariants: list[JSONObject] = []
    effective_aliases, alias_kind_map = _marker_alias_kind_map(marker_aliases)
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
            if type(node) is ast.Call and _is_marker_call(cast(ast.Call, node), effective_aliases):
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
                marker_kind = _marker_kind_for_call(call_node, alias_kind_map)
                marker_metadata = _never_marker_metadata(call_node, never_id, reason, marker_kind=marker_kind)
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
                    "marker_kind": marker_metadata.get("marker_kind", MarkerKind.NEVER.value),
                    "marker_id": marker_metadata.get("marker_id", never_id),
                    "marker_site_id": marker_metadata.get("marker_site_id", never_id),
                    "owner": marker_metadata.get("owner", ""),
                    "expiry": marker_metadata.get("expiry", ""),
                    "links": marker_metadata.get("links", []),
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
                site_id = forest.add_suite_site(
                    path.name,
                    function,
                    "call",
                    span=normalized_span,
                )
                suite_node = require_not_none(
                    forest.nodes.get(site_id),
                    reason="suite site missing from forest",
                    strict=True,
                    path=path_value,
                    function=function,
                )
                site_payload = cast(dict[str, object], entry["site"])
                site_payload["suite_id"] = str(suite_node.meta.get("suite_id", "") or "")
                site_payload["suite_kind"] = "call"
                paramset_id = forest.add_paramset(bundle)
                evidence: dict[str, object] = {"path": path.name, "qual": function}
                if reason:
                    evidence["reason"] = reason
                evidence["marker_id"] = str(marker_metadata.get("marker_id", never_id))
                evidence["marker_site_id"] = str(marker_metadata.get("marker_site_id", never_id))
                marker_links = marker_metadata.get("links")
                if type(marker_links) is list and marker_links:
                    evidence["links"] = marker_links
                marker_owner = str(marker_metadata.get("owner", "")).strip()
                if marker_owner:
                    evidence["owner"] = marker_owner
                marker_expiry = str(marker_metadata.get("expiry", "")).strip()
                if marker_expiry:
                    evidence["expiry"] = marker_expiry
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
    source = 'gabion.analysis.dataflow_indexed_file_scan._collect_never_invariants.site_1')

_DEADLINE_CHECK_METHODS = {"check", "expired"}

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
        return sort_once(nodes, source = 'gabion.analysis.dataflow_indexed_file_scan._sorted_graph_nodes.site_1')
    except TypeError:
        return sort_once(nodes, key=lambda item: repr(item), source = 'gabion.analysis.dataflow_indexed_file_scan._sorted_graph_nodes.site_2')

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
        remaining = [p for p in sort_once(named_params, source = 'gabion.analysis.dataflow_indexed_file_scan._bind_call_args.site_1') if p not in mapping]
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
    call_mapping = call.argument_mapping()
    for pos_idx, caller_param in call_mapping.positional.items():
        check_deadline()
        if pos_idx < len(pos_params):
            callee_param = pos_params[pos_idx]
        elif callee.vararg is not None:
            callee_param = callee.vararg
        else:
            continue
        mapped_params.add(callee_param)
        mapping[callee_param].add(caller_param.value)
    for kw_name, caller_param in call_mapping.keywords.items():
        check_deadline()
        if kw_name in named_params:
            mapped_params.add(kw_name)
            mapping[kw_name].add(caller_param.value)
        elif callee.kwarg is not None:
            mapped_params.add(callee.kwarg)
            mapping[callee.kwarg].add(caller_param.value)
    if strictness == "low":
        remaining = [p for p in sort_once(named_params, source = 'gabion.analysis.dataflow_indexed_file_scan._caller_param_bindings_for_call.site_1') if p not in mapped_params]
        if callee.vararg is not None and callee.vararg not in mapped_params:
            remaining.append(callee.vararg)
        if callee.kwarg is not None and callee.kwarg not in mapped_params:
            remaining.append(callee.kwarg)
        if len(call_mapping.star_positional) == 1:
            _, star_param = call_mapping.star_positional[0]
            for param in remaining:
                check_deadline()
                mapping[param].add(star_param.value)
        if len(call_mapping.star_keywords) == 1:
            star_param = call_mapping.star_keywords[0]
            for param in remaining:
                check_deadline()
                mapping[param].add(star_param.value)
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
        remaining = [p for p in sort_once(named_params, source = 'gabion.analysis.dataflow_indexed_file_scan._fallback_deadline_arg_info.site_1') if p not in mapping]
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

@dataclass(frozen=True)
class _ProjectionSpan:
    line: int
    col: int
    end_line: int
    end_col: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.line, self.col, self.end_line, self.end_col)

@dataclass(frozen=True)
class _AmbiguitySuiteRow:
    path: str
    qual: str
    suite_kind: str
    span: _ProjectionSpan

def _decode_projection_span(row: Mapping[str, JSONValue]) -> _ProjectionSpan:
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
    return _ProjectionSpan(line=line, col=col, end_line=end_line, end_col=end_col)

def _spec_row_span(row: Mapping[str, JSONValue]):
    return _decode_projection_span(row).as_tuple()

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
        source="gabion.analysis.dataflow_indexed_file_scan._suite_order_relation.relation",
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

def _decode_ambiguity_suite_row(row: Mapping[str, JSONValue]) -> _AmbiguitySuiteRow:
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
    return _AmbiguitySuiteRow(
        path=path,
        qual=qual,
        suite_kind=suite_kind,
        span=_decode_projection_span(row),
    )

def _ambiguity_suite_row_to_suite(
    row: Mapping[str, JSONValue],
    forest: Forest,
) -> NodeId:
    decoded = _decode_ambiguity_suite_row(row)
    return forest.add_suite_site(
        decoded.path,
        decoded.qual,
        decoded.suite_kind,
        span=decoded.span.as_tuple(),
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

def _span_line_col(span):
    parsed = int_tuple4_or_none(span)
    if parsed is None:
        return None, None
    return parsed[0] + 1, parsed[1] + 1

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
    for kind in sort_once(counts, source = 'gabion.analysis.dataflow_indexed_file_scan._summarize_call_ambiguities.site_1'):
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

def _lint_line(path: str, line: int, col: int, code: str, message: str) -> str:
    return f"{path}:{line}:{col}: {code} {message}".strip()

def _add_interned_alt(
    *,
    forest: Forest,
    kind: str,
    inputs: Iterable[NodeId],
    evidence = None,
) -> Alt:
    return forest.add_alt(kind, inputs, evidence=evidence)

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

@dataclass(frozen=True)
class _StageCacheIdentitySpec:
    stage: Literal["parse", "index", "projection"]
    forest_spec_id: str
    fingerprint_seed_revision: str
    normalized_config: JSONValue

_EMPTY_CACHE_SEMANTIC_CONTEXT = _CacheSemanticContext()

_ANALYSIS_INDEX_RESUME_VARIANTS_KEY = "resume_variants"

_ANALYSIS_INDEX_RESUME_MAX_VARIANTS = 4

_CACHE_IDENTITY_PREFIX = "aspf:sha1:"

_CACHE_IDENTITY_DIGEST_HEX = re.compile(r"^[0-9a-f]{40}$")

@dataclass(frozen=True)
class _CacheIdentity:
    value: str

    @classmethod
    def from_digest(cls, digest: str) -> "_CacheIdentity | None":
        cleaned = str(digest or "").strip().lower()
        if not _CACHE_IDENTITY_DIGEST_HEX.fullmatch(cleaned):
            return None
        return cls(f"{_CACHE_IDENTITY_PREFIX}{cleaned}")

    @classmethod
    def from_boundary(cls, raw_identity) -> "_CacheIdentity | None":
        identity = str(raw_identity or "").strip()
        if not identity:
            return None
        if identity.startswith(_CACHE_IDENTITY_PREFIX):
            digest = identity[len(_CACHE_IDENTITY_PREFIX) :]
            return cls.from_digest(digest)
        return cls.from_digest(identity)

    @classmethod
    def from_boundary_required(cls, raw_identity, *, field: str) -> "_CacheIdentity":
        identity = cls.from_boundary(raw_identity)
        if identity is None:
            never("invalid cache identity", field=field)
            return cls(value="")  # pragma: no cover - never() raises
        return identity

@dataclass(frozen=True)
class _ResumeCacheIdentityPair:
    canonical_index: _CacheIdentity
    canonical_projection: _CacheIdentity

    def encode(self) -> dict[str, str]:
        return {
            "index_cache_identity": self.canonical_index.value,
            "projection_cache_identity": self.canonical_projection.value,
        }

    @classmethod
    def decode_required(cls, payload: Mapping[str, JSONValue]) -> "_ResumeCacheIdentityPair":
        return cls(
            canonical_index=_CacheIdentity.from_boundary_required(
                payload.get("index_cache_identity"),
                field="index_cache_identity",
            ),
            canonical_projection=_CacheIdentity.from_boundary_required(
                payload.get("projection_cache_identity"),
                field="projection_cache_identity",
            ),
        )

def _sorted_text(values = None) -> tuple[str, ...]:
    if values is None:
        return ()
    cleaned = {str(value).strip() for value in values if str(value).strip()}
    return tuple(sort_once(cleaned, source = 'gabion.analysis.dataflow_indexed_file_scan._sorted_text.site_1'))

def _normalize_cache_config(value: JSONValue) -> JSONValue:
    if type(value) is dict:
        mapping = cast(dict[object, JSONValue], value)
        normalized = {
            str(key): _normalize_cache_config(mapping[key])
            for key in sort_once(mapping, source="_normalize_cache_config.mapping")
        }
        return cast(JSONValue, normalized)
    if type(value) is list:
        return cast(JSONValue, [_normalize_cache_config(item) for item in value])
    return value

def _canonical_stage_cache_detail(detail: Hashable) -> str:
    structural_detail = structural_key_atom(
        detail,
        source="gabion.analysis.dataflow_indexed_file_scan._canonical_stage_cache_detail",
    )
    canonical_json = structural_key_json(structural_detail)
    return json.dumps(canonical_json, sort_keys=False, separators=(",", ":"))

def _build_stage_cache_identity_spec(
    *,
    stage: Literal["parse", "index", "projection"],
    cache_context: _CacheSemanticContext,
    config_subset: Mapping[str, JSONValue],
) -> _StageCacheIdentitySpec:
    normalized_config = _normalize_cache_config(cast(JSONValue, config_subset))
    return _StageCacheIdentitySpec(
        stage=stage,
        forest_spec_id=str(cache_context.forest_spec_id or ""),
        fingerprint_seed_revision=fingerprint_stage_cache_identity(cache_context.fingerprint_seed_revision),
        normalized_config=normalized_config,
    )

def _canonical_stage_cache_identity(spec: _StageCacheIdentitySpec) -> str:
    payload: dict[str, JSONValue] = {
        "stage": spec.stage,
        "forest_spec_id": spec.forest_spec_id,
        "fingerprint_seed_revision": spec.fingerprint_seed_revision,
        "config_subset": spec.normalized_config,
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"aspf:sha1:{digest}"

def _canonical_cache_identity(
    *,
    stage: Literal["parse", "index", "projection"],
    cache_context: _CacheSemanticContext,
    config_subset: Mapping[str, JSONValue],
) -> _CacheIdentity:
    spec = _build_stage_cache_identity_spec(
        stage=stage,
        cache_context=cache_context,
        config_subset=config_subset,
    )
    canonical = _CacheIdentity.from_boundary(_canonical_stage_cache_identity(spec))
    if canonical is None:
        never("failed to construct canonical cache identity", stage=stage)  # pragma: no cover - invariant sink
    return canonical

def _cache_identity_aliases(identity: str) -> tuple[str, ...]:
    canonical = _CacheIdentity.from_boundary(identity)
    if canonical is None:
        return ("",)
    return (canonical.value,)

def _resume_variant_for_identity(
    variants: Mapping[str, JSONObject],
    expected_identity: _CacheIdentity,
):
    direct = variants.get(expected_identity.value)
    if direct is not None:
        return direct
    return None

def _parse_stage_cache_key(
    *,
    stage: _ParseModuleStage,
    cache_context: _CacheSemanticContext,
    config_subset: Mapping[str, JSONValue],
    detail: Hashable,
) -> NodeId:
    identity = _canonical_cache_identity(
        stage="parse",
        cache_context=cache_context,
        config_subset=config_subset,
    )
    return NodeId(
        kind="ParseStageCacheIdentity",
        key=(
            stage.value,
            identity.value,
            _canonical_stage_cache_detail(detail),
        ),
    )

def _index_stage_cache_identity(
    *,
    cache_context: _CacheSemanticContext,
    config_subset: Mapping[str, JSONValue],
) -> _CacheIdentity:
    return _canonical_cache_identity(
        stage="index",
        cache_context=cache_context,
        config_subset=config_subset,
    )

def _projection_stage_cache_identity(
    *,
    cache_context: _CacheSemanticContext,
    config_subset: Mapping[str, JSONValue],
) -> _CacheIdentity:
    return _canonical_cache_identity(
        stage="projection",
        cache_context=cache_context,
        config_subset=config_subset,
    )

def _stage_cache_key_aliases(key: Hashable) -> tuple[Hashable, ...]:
    if (
        type(key) is tuple
        and len(key) == 2
        and type(key[1]) is tuple
    ):
        scoped_identity = key[0]
        parse_key = cast(Hashable, key[1])
        parse_aliases = _stage_cache_key_aliases(parse_key)
        if len(parse_aliases) > 1:
            return tuple((scoped_identity, alias) for alias in parse_aliases)
        return (key,)
    if (
        type(key) is tuple
        and len(key) == 4
        and key[0] == "parse"
        and type(key[2]) is str
    ):
        identity = key[2]
        aliases = _cache_identity_aliases(identity)
        identity_text = str(identity)
        if len(aliases) == 1 and identity_text.startswith(_CACHE_IDENTITY_PREFIX):
            digest = identity_text[len(_CACHE_IDENTITY_PREFIX) :]
            if _CACHE_IDENTITY_DIGEST_HEX.fullmatch(digest):
                aliases = (aliases[0], digest)
        if len(aliases) > 1:
            return tuple((key[0], key[1], alias, key[3]) for alias in aliases)
    if (
        type(key) is NodeId
        and key.kind == "ParseStageCacheIdentity"
        and len(key.key) == 3
    ):
        stage_value, identity, detail = key.key
        if type(stage_value) is str and type(identity) is str:
            legacy_key = ("parse", stage_value, identity, detail)
            aliases = _stage_cache_key_aliases(legacy_key)
            return (key, *aliases)
    return (key,)

def _get_stage_cache_bucket(
    analysis_index: AnalysisIndex,
    *,
    scoped_cache_key: Hashable,
) -> dict[Path, object]:
    stage_cache_by_key = analysis_index.stage_cache_by_key
    bucket = stage_cache_by_key.get(scoped_cache_key)
    if bucket is not None:
        return bucket
    for candidate_key in _stage_cache_key_aliases(scoped_cache_key):
        check_deadline()
        if candidate_key == scoped_cache_key:
            continue
        legacy_bucket = stage_cache_by_key.get(candidate_key)
        if legacy_bucket is not None:
            stage_cache_by_key[scoped_cache_key] = legacy_bucket
            return legacy_bucket
    return stage_cache_by_key.setdefault(scoped_cache_key, {})

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
        expected_index_cache_identity=index_cache_identity.value,
        expected_projection_cache_identity=projection_cache_identity.value,
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
                        index_cache_identity=index_cache_identity.value,
                        projection_cache_identity=projection_cache_identity.value,
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
        index_cache_identity=index_cache_identity.value,
        projection_cache_identity=projection_cache_identity.value,
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
    cache = _get_stage_cache_bucket(
        analysis_index,
        scoped_cache_key=scoped_cache_key,
    )
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
                    source="gabion.analysis.dataflow_indexed_file_scan._analysis_index_stage_cache",
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
    index = analysis_index
    if index is None:
        index = _build_analysis_index(
            list(paths),
            project_root=project_root,
            ignore_params=set(ignore_params),
            strictness=strictness,
            external_filter=external_filter,
            transparent_decorators=transparent_decorators,
            parse_failure_witnesses=list(parse_failure_witnesses),
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
        ordered = tuple(sort_once(candidates, key=lambda info: info.qual, source = 'gabion.analysis.dataflow_indexed_file_scan._sink.site_1'))
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

def _forbid_adhoc_bundle_discovery(reason: str) -> None:
    if os.environ.get("GABION_FORBID_ADHOC_BUNDLES") == "1":
        raise AssertionError(
            f"Ad-hoc bundle discovery invoked while forest-only invariant active: {reason}"
        )

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
        for fn_name in sort_once(groups, source = 'gabion.analysis.dataflow_indexed_file_scan._populate_bundle_forest.site_1'):
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
        for name in sort_once(bundles, source = 'gabion.analysis.dataflow_indexed_file_scan._populate_bundle_forest.site_2'):
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
    for qual_name in sort_once(dataclass_registry, source = 'gabion.analysis.dataflow_indexed_file_scan._populate_bundle_forest.site_3'):
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
        for bundle in sort_once(_iter_documented_bundles(path), source = 'gabion.analysis.dataflow_indexed_file_scan._populate_bundle_forest.site_4'):
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
        source = 'gabion.analysis.dataflow_indexed_file_scan._populate_bundle_forest.site_5'):
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
        sig = tuple(sort_once(info.direct_forward, source = 'gabion.analysis.dataflow_indexed_file_scan._group_by_signature.site_1'))
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
        mapping = call.argument_mapping()
        # Build mapping from callee param to caller param.
        callee_to_caller: dict[str, str] = {}
        for idx, pname in enumerate(callee_params):
            check_deadline()
            if idx in mapping.positional:
                callee_to_caller[pname] = mapping.positional[idx].value
        for kw, caller_param in mapping.keywords.items():
            check_deadline()
            callee_to_caller[kw] = caller_param.value
        if strictness == "low":
            mapped = set(callee_to_caller.keys())
            remaining = [p for p in callee_params if p not in mapped]
            if len(mapping.star_positional) == 1:
                _, star_param = mapping.star_positional[0]
                for param in remaining:
                    check_deadline()
                    callee_to_caller.setdefault(param, star_param.value)
            if len(mapping.star_keywords) == 1:
                star_param = mapping.star_keywords[0]
                for param in remaining:
                    check_deadline()
                    callee_to_caller.setdefault(param, star_param.value)
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
            mapping = call.argument_mapping()
            callable_id = call.callable_id()
            for idx, param in mapping.positional.items():
                check_deadline()
                if param.value in bundle:
                    params_in_call.append(param.value)
                    slots.append(f"arg[{idx}]")
            for name, param in mapping.keywords.items():
                check_deadline()
                if param.value in bundle:
                    params_in_call.append(param.value)
                    slots.append(f"kw[{name}]")
            for idx, param in mapping.star_positional:
                check_deadline()
                if param.value in bundle:
                    params_in_call.append(param.value)
                    slots.append(f"arg[{idx}]*")
            for param in mapping.star_keywords:
                check_deadline()
                if param.value in bundle:
                    params_in_call.append(param.value)
                    slots.append("kw[**]")
            distinct = tuple(
                sort_once(
                    set(params_in_call),
                    source="gabion.analysis.dataflow_indexed_file_scan._callsite_evidence_for_bundle.site_1",
                )
            )
            if distinct:
                slot_list = tuple(
                    sort_once(
                        set(slots),
                        source="gabion.analysis.dataflow_indexed_file_scan._callsite_evidence_for_bundle.site_2",
                    )
                )
                span_identity = SpanIdentity.from_tuple(
                    require_not_none(
                        call.span,
                        reason="callsite evidence requires span",
                        strict=True,
                    )
                )
                span_tuple = (
                    span_identity.start_line,
                    span_identity.start_col,
                    span_identity.end_line,
                    span_identity.end_col,
                )
                callable_id = call.callable_id()
                key = (span_tuple, callable_id.value, distinct, slot_list)
                if key not in seen:
                    seen.add(key)
                    out.append(
                        {
                            "callee": callable_id.value,
                            "span": list(span_tuple),
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

def _adapt_ingest_carrier_to_analysis_maps(ingest_carrier):
    return (
        dict(ingest_carrier.function_use),
        dict(ingest_carrier.function_calls),
        dict(ingest_carrier.function_param_orders),
        dict(ingest_carrier.function_param_spans),
        set(ingest_carrier.opaque_callees),
    )

def analyze_ingested_file(
    ingest_carrier,
    *,
    recursive: bool,
    config: AuditConfig,
    on_profile = None,
) -> tuple[
    dict[str, list[set[str]]],
    dict[str, dict[str, tuple[int, int, int, int]]],
    dict[str, list[list[JSONObject]]],
]:
    (
        fn_use,
        fn_calls,
        fn_param_orders,
        fn_param_spans,
        opaque_callees,
    ) = _adapt_ingest_carrier_to_analysis_maps(ingest_carrier)
    profile_stage_ns: dict[str, int] = {
        "file_scan.grouping": 0,
        "file_scan.propagation": 0,
        "file_scan.bundle_sites": 0,
    }

    def _emit_file_profile() -> None:
        if on_profile is not None:
            on_profile(_profiling_v1_payload(stage_ns=profile_stage_ns, counters=Counter()))

    grouping_started_ns = time.monotonic_ns()
    groups_by_fn = {fn: _group_by_signature(use_map) for fn, use_map in fn_use.items()}
    profile_stage_ns["file_scan.grouping"] += time.monotonic_ns() - grouping_started_ns

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
    ingest_carrier = ingest_python_file(
        path,
        config=config,
        recursive=recursive,
        parse_module=_parse_module_source,
        collect_functions=_collect_functions,
        collect_return_aliases=_collect_return_aliases,
        load_resume_state=_load_file_scan_resume_state,
        serialize_resume_state=_serialize_file_scan_resume_state,
        profiling_payload=_profiling_v1_payload,
        analyze_function=analyze_function_fn,
        enclosing_class=_enclosing_class,
        enclosing_scopes=_enclosing_scopes,
        enclosing_function_scopes=_enclosing_function_scopes,
        function_key=_function_key,
        decorators_transparent=_decorators_transparent,
        param_names=_param_names,
        param_spans=_param_spans,
        collect_local_class_bases=_collect_local_class_bases,
        resolve_local_method_in_hierarchy=_resolve_local_method_in_hierarchy,
        is_test_path=_is_test_path,
        check_deadline=check_deadline,
        parent_annotator_factory=ParentAnnotator,
        progress_emit_interval=_FILE_SCAN_PROGRESS_EMIT_INTERVAL,
        progress_min_interval_seconds=_PROGRESS_EMIT_MIN_INTERVAL_SECONDS,
        on_progress=on_progress,
        on_profile=on_profile,
        resume_state=resume_state,
    )
    return analyze_ingested_file(
        ingest_carrier,
        recursive=recursive,
        config=config,
        on_profile=on_profile,
    )

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
                    source="gabion.analysis.dataflow_indexed_file_scan._combine_type_hints.site_1",
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
    sorted_types = sort_once(expanded, source = 'gabion.analysis.dataflow_indexed_file_scan._combine_type_hints.site_2')
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
            symbol: tuple(sort_once(quals, source = 'gabion.analysis.dataflow_indexed_file_scan._collect_lambda_bindings_by_caller.site_1'))
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
    return tuple(sort_once(deduped.values(), key=lambda info: info.qual, source = 'gabion.analysis.dataflow_indexed_file_scan._dedupe_resolution_candidates.site_1'))

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
    AnalysisPassPrerequisites(
        bundle_inference=True,
        call_propagation=True,
        decision_surfaces=True,
        type_flow=True,
        lint_evidence=True,
    ).validate(pass_id="type_flow")
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
                        f"{path_key}:{fn_key}.{param} downstream types conflict: {sort_once(annots, source = 'gabion.analysis.dataflow_indexed_file_scan._infer_type_flow.site_1')}"
                    )
                    for annot in sort_once(annots, source = 'gabion.analysis.dataflow_indexed_file_scan._infer_type_flow.site_2'):
                        check_deadline()
                        for site in sort_once(sites.get(param, {}).get(annot, set()), source = 'gabion.analysis.dataflow_indexed_file_scan._infer_type_flow.site_3')[
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
                    source = 'gabion.analysis.dataflow_indexed_file_scan._infer_type_flow.site_4')[:max_sites_per_param]:
                        check_deadline()
                        evidence_lines.add(site)
    return inferred, sort_once(suggestions, source = 'gabion.analysis.dataflow_indexed_file_scan._infer_type_flow.site_5'), sort_once(ambiguities, source = 'gabion.analysis.dataflow_indexed_file_scan._infer_type_flow.site_6'), sort_once(evidence_lines, source = 'gabion.analysis.dataflow_indexed_file_scan._infer_type_flow.site_7')

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
    return sort_once(smells, source = 'gabion.analysis.dataflow_indexed_file_scan._constant_smells_from_details.site_1')

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
    source = 'gabion.analysis.dataflow_indexed_file_scan._deadness_witnesses_from_constant_details.site_1')

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
                sites=tuple(sort_once(folded.call_sites.get(key, set()), source = 'gabion.analysis.dataflow_indexed_file_scan._collect_constant_flow_details.site_1')),
            )
        )
    return sort_once(details, key=lambda entry: (str(entry.path), entry.name, entry.param), source = 'gabion.analysis.dataflow_indexed_file_scan._collect_constant_flow_details.site_2')

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
    return sort_once(smells, source = 'gabion.analysis.dataflow_indexed_file_scan._analyze_unused_arg_flow_indexed.site_1')

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
        bundles.add(tuple(sort_once(parts, source = 'gabion.analysis.dataflow_indexed_file_scan._iter_documented_bundles.site_1')))
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
        else:
            registry[class_node.name] = fields
    return registry

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

_REPORT_SECTION_MARKER_PREFIX = "<!-- report-section:"

_REPORT_SECTION_MARKER_SUFFIX = "-->"

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

def _normalize_snapshot_path(path: Path, root) -> str:
    if root is not None:
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return str(path)

_ANALYSIS_COLLECTION_RESUME_FORMAT_VERSION = 2

_FILE_SCAN_PROGRESS_EMIT_INTERVAL = 1

_PROGRESS_EMIT_MIN_INTERVAL_SECONDS = 1.0

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
            [callee, slot] for callee, slot in sort_once(value.direct_forward, source = 'gabion.analysis.dataflow_indexed_file_scan._serialize_param_use.site_1')
        ],
        "non_forward": bool(value.non_forward),
        "current_aliases": sort_once(value.current_aliases, source = 'gabion.analysis.dataflow_indexed_file_scan._serialize_param_use.site_2'),
        "forward_sites": [
            {
                "callee": callee,
                "slot": slot,
                "spans": [list(span) for span in sort_once(spans, source = 'gabion.analysis.dataflow_indexed_file_scan._serialize_param_use.site_3')],
            }
            for (callee, slot), spans in sort_once(value.forward_sites.items(), source = 'gabion.analysis.dataflow_indexed_file_scan._serialize_param_use.site_4')
        ],
        "unknown_key_carrier": bool(value.unknown_key_carrier),
        "unknown_key_sites": [list(span) for span in sort_once(value.unknown_key_sites, source = 'gabion.analysis.dataflow_indexed_file_scan._serialize_param_use.site_5')],
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
    for param_name in sort_once(use_map, source = 'gabion.analysis.dataflow_indexed_file_scan._serialize_param_use_map.site_1'):
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
        "pos_map": {key: call.pos_map[key] for key in sort_once(call.pos_map, source = 'gabion.analysis.dataflow_indexed_file_scan._serialize_call_args.site_1')},
        "kw_map": {key: call.kw_map[key] for key in sort_once(call.kw_map, source = 'gabion.analysis.dataflow_indexed_file_scan._serialize_call_args.site_2')},
        "const_pos": {key: call.const_pos[key] for key in sort_once(call.const_pos, source = 'gabion.analysis.dataflow_indexed_file_scan._serialize_call_args.site_3')},
        "const_kw": {key: call.const_kw[key] for key in sort_once(call.const_kw, source = 'gabion.analysis.dataflow_indexed_file_scan._serialize_call_args.site_4')},
        "non_const_pos": sort_once(call.non_const_pos, source = 'gabion.analysis.dataflow_indexed_file_scan._serialize_call_args.site_5'),
        "non_const_kw": sort_once(call.non_const_kw, source = 'gabion.analysis.dataflow_indexed_file_scan._serialize_call_args.site_6'),
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
        star_kw=sort_once(str_set_from_sequence(payload.get("star_kw")), source = 'gabion.analysis.dataflow_indexed_file_scan._deserialize_call_args.site_1'),
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
        "annots": {param: info.annots[param] for param in sort_once(info.annots, source = 'gabion.analysis.dataflow_indexed_file_scan._serialize_function_info_for_resume.site_1')},
        "calls": _serialize_call_args_list(info.calls),
        "unused_params": sort_once(info.unused_params, source = 'gabion.analysis.dataflow_indexed_file_scan._serialize_function_info_for_resume.site_2'),
        "unknown_key_carriers": sort_once(info.unknown_key_carriers, source = 'gabion.analysis.dataflow_indexed_file_scan._serialize_function_info_for_resume.site_3'),
        "defaults": sort_once(info.defaults, source = 'gabion.analysis.dataflow_indexed_file_scan._serialize_function_info_for_resume.site_4'),
        "transparent": bool(info.transparent),
        "class_name": info.class_name,
        "scope": list(info.scope),
        "lexical_scope": list(info.lexical_scope),
        "decision_params": sort_once(info.decision_params, source = 'gabion.analysis.dataflow_indexed_file_scan._serialize_function_info_for_resume.site_5'),
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
        "value_decision_params": sort_once(info.value_decision_params, source = 'gabion.analysis.dataflow_indexed_file_scan._serialize_function_info_for_resume.site_6'),
        "value_decision_reasons": sort_once(info.value_decision_reasons, source = 'gabion.analysis.dataflow_indexed_file_scan._serialize_function_info_for_resume.site_7'),
        "positional_params": list(info.positional_params),
        "kwonly_params": list(info.kwonly_params),
        "vararg": info.vararg,
        "kwarg": info.kwarg,
        "param_spans": {
            param: [int(value) for value in info.param_spans[param]]
            for param in sort_once(info.param_spans, source = 'gabion.analysis.dataflow_indexed_file_scan._serialize_function_info_for_resume.site_8')
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
        "methods": sort_once(class_info.methods, source = 'gabion.analysis.dataflow_indexed_file_scan._serialize_class_info_for_resume.site_1'),
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
    variant_payload = {
        str(key): payload[key]
        for key in payload
        if str(key) != _ANALYSIS_INDEX_RESUME_VARIANTS_KEY
    }
    try:
        identities = _ResumeCacheIdentityPair.decode_required(variant_payload)
    except NeverThrown:
        return variant_payload
    variant_payload.update(identities.encode())
    return variant_payload

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
            variant_identity = _CacheIdentity.from_boundary(identity)
            if variant_identity is not None and raw_variant_mapping is not None:
                variant_payload = payload_with_format(raw_variant_mapping, format_version=1)
                if variant_payload is not None:
                    variants[variant_identity.value] = _analysis_index_resume_variant_payload(
                        variant_payload
                    )
    return variants

def _with_analysis_index_resume_variants(
    *,
    payload: JSONObject,
    previous_payload,
) -> JSONObject:
    identities = _ResumeCacheIdentityPair.decode_required(payload)
    variants = _analysis_index_resume_variants(previous_payload)
    payload.update(identities.encode())
    variants[identities.canonical_index.value] = _analysis_index_resume_variant_payload(payload)
    ordered_variant_keys = [
        key
        for key in sort_once(
            variants.keys(), source = 'gabion.analysis.dataflow_indexed_file_scan._with_analysis_index_resume_variants.site_1'
        )
        if key != identities.canonical_index.value
    ]
    ordered_variant_keys.append(identities.canonical_index.value)
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
    identities = _ResumeCacheIdentityPair(
        canonical_index=_CacheIdentity.from_boundary_required(
            index_cache_identity,
            field="index_cache_identity",
        ),
        canonical_projection=_CacheIdentity.from_boundary_required(
            projection_cache_identity,
            field="projection_cache_identity",
        ),
    )
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
        **identities.encode(),
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
    expected_index_identity = _CacheIdentity.from_boundary(expected_index_cache_identity)
    expected_projection_identity = _CacheIdentity.from_boundary(expected_projection_cache_identity)
    selected_payload: Mapping[str, JSONValue] = payload
    if expected_index_identity is not None:
        selected_identity = _CacheIdentity.from_boundary(selected_payload.get("index_cache_identity"))
        if selected_identity != expected_index_identity:
            variants = _analysis_index_resume_variants(payload)
            variant = _resume_variant_for_identity(variants, expected_index_identity)
            if variant is None:
                return hydrated_paths, by_qual, symbol_table, class_index
            selected_payload = variant
    if expected_projection_identity is not None:
        projection_identity = _CacheIdentity.from_boundary(
            selected_payload.get("projection_cache_identity")
        )
        if projection_identity != expected_projection_identity:
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
    for fn_name in sort_once(groups, source = 'gabion.analysis.dataflow_indexed_file_scan._serialize_groups_for_resume.site_1'):
        check_deadline()
        bundles = groups[fn_name]
        normalized = [
            sort_once(
                (str(param) for param in bundle),
                source="gabion.analysis.dataflow_indexed_file_scan._serialize_groups_for_resume.site_2",
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
    for fn_name in sort_once(spans, source = 'gabion.analysis.dataflow_indexed_file_scan._serialize_param_spans_for_resume.site_1'):
        check_deadline()
        param_spans = spans[fn_name]
        payload[fn_name] = {}
        for param_name in sort_once(param_spans, source = 'gabion.analysis.dataflow_indexed_file_scan._serialize_param_spans_for_resume.site_2'):
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
    for fn_name in sort_once(bundle_sites, source = 'gabion.analysis.dataflow_indexed_file_scan._serialize_bundle_sites_for_resume.site_1'):
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
    source = 'gabion.analysis.dataflow_indexed_file_scan._serialize_invariants_for_resume.site_1'):
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
    for fn_key in sort_once(fn_use, source = 'gabion.analysis.dataflow_indexed_file_scan._serialize_file_scan_resume_state.site_1'):
        check_deadline()
        fn_use_payload[fn_key] = _serialize_param_use_map(fn_use[fn_key])
    for fn_key in sort_once(fn_calls, source = 'gabion.analysis.dataflow_indexed_file_scan._serialize_file_scan_resume_state.site_2'):
        check_deadline()
        fn_calls_payload[fn_key] = _serialize_call_args_list(fn_calls[fn_key])
    for fn_key in sort_once(fn_param_orders, source = 'gabion.analysis.dataflow_indexed_file_scan._serialize_file_scan_resume_state.site_3'):
        check_deadline()
        fn_param_orders_payload[fn_key] = list(fn_param_orders[fn_key])
    for fn_key in sort_once(fn_param_spans, source = 'gabion.analysis.dataflow_indexed_file_scan._serialize_file_scan_resume_state.site_4'):
        check_deadline()
        fn_param_spans_payload[fn_key] = _serialize_param_spans_for_resume(
            {fn_key: dict(fn_param_spans[fn_key])}
        ).get(fn_key, {})
    for fn_key in sort_once(fn_names, source = 'gabion.analysis.dataflow_indexed_file_scan._serialize_file_scan_resume_state.site_5'):
        check_deadline()
        fn_names_payload[fn_key] = fn_names[fn_key]
    for fn_key in sort_once(fn_lexical_scopes, source = 'gabion.analysis.dataflow_indexed_file_scan._serialize_file_scan_resume_state.site_6'):
        check_deadline()
        fn_lexical_scopes_payload[fn_key] = list(fn_lexical_scopes[fn_key])
    for fn_key in sort_once(fn_class_names, source = 'gabion.analysis.dataflow_indexed_file_scan._serialize_file_scan_resume_state.site_7'):
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
        "opaque_callees": sort_once(opaque_callees, source = 'gabion.analysis.dataflow_indexed_file_scan._serialize_file_scan_resume_state.site_8'),
        "processed_functions": sort_once(fn_use.keys(), source = 'gabion.analysis.dataflow_indexed_file_scan._serialize_file_scan_resume_state.site_9'),
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
        source="gabion.analysis.dataflow_indexed_file_scan._build_analysis_collection_resume_payload.site_1",
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
            source = 'gabion.analysis.dataflow_indexed_file_scan._build_analysis_collection_resume_payload.site_2')
        }
    analysis_index_resume_mapping = mapping_or_none(analysis_index_resume)
    if analysis_index_resume_mapping is not None:
        payload["analysis_index_resume"] = {
            str(key): analysis_index_resume_mapping[key]
            for key in analysis_index_resume_mapping
        }
    return payload

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
    timeout_carrier = TimeoutTickCarrier.from_ingress(
        ticks=args.analysis_timeout_ticks,
        tick_ns=args.analysis_timeout_tick_ns,
    )
    if timeout_carrier.ticks == 0:
        never(
            "invalid analysis timeout ticks",
            analysis_timeout_ticks=timeout_carrier.ticks,
        )
    tick_limit_value = args.analysis_tick_limit
    logical_limit = timeout_carrier.ticks
    if tick_limit_value is not None:
        tick_limit = int(tick_limit_value)
        if tick_limit <= 0:
            never("invalid analysis tick limit", analysis_tick_limit=tick_limit)
        logical_limit = min(logical_limit, tick_limit)
    with ExitStack() as stack:
        stack.enter_context(forest_scope(Forest()))
        stack.enter_context(
            deadline_scope(Deadline.from_timeout_ticks(timeout_carrier))
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
    never_exceptions = set(exception_marker_family(exception_section, "never"))
    never_exceptions.update(exception_never_list(exception_section))
    all_marker_aliases = {
        alias
        for aliases in DEFAULT_MARKER_ALIASES.values()
        for alias in aliases
    }
    never_exceptions.update(all_marker_aliases)
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
    adapter_payload = dataflow_adapter_payload(merged)
    required_analysis_surfaces = {
        str(item)
        for item in dataflow_required_surfaces(merged)
        if type(item) is str and str(item)
    }
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
        adapter_contract=normalize_adapter_contract(adapter_payload),
        required_analysis_surfaces=required_analysis_surfaces,
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
