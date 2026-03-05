# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from gabion.analysis.dataflow.engine.dataflow_bundle_merge import _merge_counts_by_knobs
from gabion.analysis.dataflow.engine.dataflow_analysis_index_owner import _build_analysis_index
from gabion.analysis.dataflow.engine.dataflow_contracts import (
    AuditConfig, ClassInfo, FunctionInfo, InvariantProposition, SymbolTable)
from gabion.analysis.dataflow.engine.dataflow_evidence_helpers import _resolve_callee
from gabion.analysis.dataflow.io.dataflow_parse_helpers import _forbid_adhoc_bundle_discovery
from gabion.analysis.dataflow.io.dataflow_synthesis_runtime_bridge import (
    _build_call_graph, _collect_config_bundles, _collect_dataclass_registry, _combine_type_hints, _compute_knob_param_names, _type_from_const_repr, analyze_type_flow_repo_with_map, generate_property_hook_manifest)
from gabion.analysis.core.forest_signature import build_forest_signature_from_groups
from gabion.analysis.foundation.json_types import JSONObject
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.order_contract import sort_once
from gabion.schema import SynthesisResponse
from gabion.synthesis import NamingContext, SynthesisConfig, Synthesizer
from gabion.synthesis.emission import render_protocol_stubs as _render_protocol_stubs
from gabion.synthesis.merge import merge_bundles


def _infer_root(groups_by_path: dict[Path, dict[str, list[set[str]]]]) -> Path:
    if groups_by_path:
        common = os.path.commonpath([str(path) for path in groups_by_path])
        return Path(common)
    return Path(".")


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


def _function_key(scope: Sequence[str], name: str) -> str:
    parts = list(scope)
    parts.append(name)
    return ".".join(parts)


def _bundle_counts(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
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
                counts[
                    tuple(
                        sort_once(
                            bundle,
                            source="dataflow_synthesis._bundle_counts.bundle",
                        )
                    )
                ] += 1
    return counts


def _collect_declared_bundles(root: Path) -> set[tuple[str, ...]]:
    check_deadline()
    _forbid_adhoc_bundle_discovery("_collect_declared_bundles")
    declared: set[tuple[str, ...]] = set()
    file_paths = sort_once(
        root.rglob("*.py"),
        source="dataflow_synthesis._collect_declared_bundles.file_paths",
        key=lambda path: str(path),
    )
    parse_failure_witnesses: list[JSONObject] = []
    bundles_by_path = _collect_config_bundles(
        list(file_paths),
        parse_failure_witnesses=parse_failure_witnesses,
    )
    dataclass_registry = _collect_dataclass_registry(
        list(file_paths),
        project_root=root,
        parse_failure_witnesses=parse_failure_witnesses,
    )
    for bundles in bundles_by_path.values():
        check_deadline()
        for fields in bundles.values():
            check_deadline()
            declared.add(
                tuple(
                    sort_once(
                        fields,
                        source="dataflow_synthesis._collect_declared_bundles.config",
                    )
                )
            )
    for fields in dataclass_registry.values():
        check_deadline()
        declared.add(
            tuple(
                sort_once(
                    fields,
                    source="dataflow_synthesis._collect_declared_bundles.dataclass",
                )
            )
        )
    return declared


@dataclass(frozen=True)
class _SynthesisPlanContext:
    audit_config: AuditConfig
    root: Path
    signature_meta: JSONObject
    path_list: list[Path]
    parse_failure_witnesses: list[JSONObject]
    analysis_index: object
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
            bundle = tuple(
                sort_once(
                    info.decision_params,
                    source="dataflow_synthesis._collect_synthesis_counts_and_evidence.decision",
                )
            )
            decision_counts[bundle] += 1
            evidence = bundle_evidence[frozenset(bundle)]
            evidence.add("decision_surface")
            if caller_count > 0:
                evidence.add("tier-2:decision-bundle-elevation")
            else:
                evidence.add("tier-3:decision-table-boundary")
        if info.value_decision_params:
            bundle = tuple(
                sort_once(
                    info.value_decision_params,
                    source="dataflow_synthesis._collect_synthesis_counts_and_evidence.value_decision",
                )
            )
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
        original_bundles,
        min_overlap=synth_config.merge_overlap_threshold,
    )
    merged_bundle_tiers: dict[frozenset[str], int] = {}
    for merged in merged_bundles:
        check_deadline()
        members = [
            bundle for bundle in original_bundles if bundle and bundle.issubset(merged)
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
                bundle for bundle in original_bundles if bundle and bundle.issubset(merged)
            ]
            if not members:
                continue
            evidence: set[str] = set()
            for member in members:
                check_deadline()
                evidence.update(bundle_evidence.get(frozenset(member), set()))
            merged_bundle_evidence[frozenset(merged)] = evidence
    else:
        merged_bundle_evidence = {key: set(values) for key, values in bundle_evidence.items()}

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
            if name in bundle_fields and annot:
                type_sets[name].add(str(annot))

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
                f"Conflicting type hints for '{name}': "
                f"{sort_once(types, source='dataflow_synthesis._infer_synthesis_field_types.types')} -> {combined}"
            )
    return field_types, type_warnings


def _synthesis_payload_from_plan(
    *,
    plan,
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
                        "source_params": sort_once(
                            field.source_params,
                            source="dataflow_synthesis._synthesis_payload_from_plan.source_params",
                        ),
                    }
                    for field in spec.fields
                ],
                "bundle": sort_once(
                    spec.bundle,
                    source="dataflow_synthesis._synthesis_payload_from_plan.bundle",
                ),
                "tier": spec.tier,
                "rationale": spec.rationale,
                "evidence": sort_once(
                    bundle_evidence.get(frozenset(spec.bundle), set()),
                    source="dataflow_synthesis._synthesis_payload_from_plan.evidence",
                ),
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
    project_root=None,
    max_tier: int = 2,
    min_bundle_size: int = 2,
    allow_singletons: bool = False,
    merge_overlap_threshold=None,
    config=None,
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


def render_protocol_stubs(plan, kind: str = "dataclass") -> str:
    return _render_protocol_stubs(plan, kind=kind)
