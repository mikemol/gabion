from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from gabion.commands import check_contract
from gabion.commands.check_contract import (
    DataflowFilterBundle, DataflowPayloadCommonOptions)
from gabion.json_types import JSONObject

NormalizeOptionalOutputTargetFn = Callable[[object], str | None]
BuildDataflowPayloadCommonFn = Callable[..., JSONObject]


@dataclass(frozen=True)
class SynthPayloadOptions:
    no_recursive: bool
    max_components: int
    type_audit_report: bool
    type_audit_max: int
    synthesis_plan: Path
    synthesis_report: bool
    synthesis_protocols: Path
    synthesis_protocols_kind: str
    synthesis_max_tier: int
    synthesis_min_bundle_size: int
    synthesis_allow_singletons: bool
    refactor_plan: bool
    refactor_plan_json: Path | None
    fingerprint_synth_json: Path
    fingerprint_provenance_json: Path
    fingerprint_coherence_json: Path
    fingerprint_rewrite_plans_json: Path
    fingerprint_exception_obligations_json: Path
    fingerprint_handledness_json: Path


def build_synth_payload(
    *,
    options: DataflowPayloadCommonOptions,
    synth_options: SynthPayloadOptions,
    build_dataflow_payload_common_fn: BuildDataflowPayloadCommonFn,
) -> JSONObject:
    payload = build_dataflow_payload_common_fn(options=options)
    payload.update(
        {
            "dot": None,
            "no_recursive": synth_options.no_recursive,
            "max_components": synth_options.max_components,
            "type_audit_report": synth_options.type_audit_report,
            "type_audit_max": synth_options.type_audit_max,
            "synthesis_plan": str(synth_options.synthesis_plan),
            "synthesis_report": synth_options.synthesis_report,
            "synthesis_protocols": str(synth_options.synthesis_protocols),
            "synthesis_protocols_kind": synth_options.synthesis_protocols_kind,
            "synthesis_max_tier": synth_options.synthesis_max_tier,
            "synthesis_min_bundle_size": synth_options.synthesis_min_bundle_size,
            "synthesis_allow_singletons": synth_options.synthesis_allow_singletons,
            "refactor_plan": synth_options.refactor_plan,
            "refactor_plan_json": str(synth_options.refactor_plan_json)
            if synth_options.refactor_plan_json is not None
            else None,
            "fingerprint_synth_json": str(synth_options.fingerprint_synth_json),
            "fingerprint_provenance_json": str(synth_options.fingerprint_provenance_json),
            "fingerprint_coherence_json": str(synth_options.fingerprint_coherence_json),
            "fingerprint_rewrite_plans_json": str(
                synth_options.fingerprint_rewrite_plans_json
            ),
            "fingerprint_exception_obligations_json": str(
                synth_options.fingerprint_exception_obligations_json
            ),
            "fingerprint_handledness_json": str(synth_options.fingerprint_handledness_json),
        }
    )
    return payload


def build_dataflow_payload(
    opts: argparse.Namespace,
    *,
    normalize_optional_output_target_fn: NormalizeOptionalOutputTargetFn,
    build_dataflow_payload_common_fn: BuildDataflowPayloadCommonFn,
) -> JSONObject:
    report_target = normalize_optional_output_target_fn(opts.report)
    decision_snapshot_target = normalize_optional_output_target_fn(
        opts.emit_decision_snapshot
    )
    dot_target = normalize_optional_output_target_fn(opts.dot)
    synthesis_plan_target = normalize_optional_output_target_fn(opts.synthesis_plan)
    synthesis_protocols_target = normalize_optional_output_target_fn(
        opts.synthesis_protocols
    )
    refactor_plan_json_target = normalize_optional_output_target_fn(opts.refactor_plan_json)
    fingerprint_synth_json_target = normalize_optional_output_target_fn(
        opts.fingerprint_synth_json
    )
    fingerprint_provenance_json_target = normalize_optional_output_target_fn(
        opts.fingerprint_provenance_json
    )
    fingerprint_deadness_json_target = normalize_optional_output_target_fn(
        opts.fingerprint_deadness_json
    )
    fingerprint_coherence_json_target = normalize_optional_output_target_fn(
        opts.fingerprint_coherence_json
    )
    fingerprint_rewrite_plans_json_target = normalize_optional_output_target_fn(
        opts.fingerprint_rewrite_plans_json
    )
    fingerprint_exception_obligations_json_target = normalize_optional_output_target_fn(
        opts.fingerprint_exception_obligations_json
    )
    fingerprint_handledness_json_target = normalize_optional_output_target_fn(
        opts.fingerprint_handledness_json
    )
    aspf_trace_json_target = normalize_optional_output_target_fn(opts.aspf_trace_json)
    aspf_opportunities_json_target = normalize_optional_output_target_fn(
        opts.aspf_opportunities_json
    )
    aspf_state_json_target = normalize_optional_output_target_fn(opts.aspf_state_json)
    aspf_delta_jsonl_target = normalize_optional_output_target_fn(opts.aspf_delta_jsonl)
    structure_tree_target = normalize_optional_output_target_fn(opts.emit_structure_tree)
    structure_metrics_target = normalize_optional_output_target_fn(
        opts.emit_structure_metrics
    )
    aspf_import_trace = (
        check_contract.split_csv_entries(opts.aspf_import_trace)
        if opts.aspf_import_trace
        else []
    )
    aspf_equivalence_against = (
        check_contract.split_csv_entries(opts.aspf_equivalence_against)
        if opts.aspf_equivalence_against
        else []
    )
    aspf_import_state = (
        check_contract.split_csv_entries(opts.aspf_import_state)
        if opts.aspf_import_state
        else []
    )
    aspf_semantic_surface = (
        check_contract.split_csv_entries(opts.aspf_semantic_surface)
        if opts.aspf_semantic_surface
        else []
    )
    payload = build_dataflow_payload_common_fn(
        options=DataflowPayloadCommonOptions(
            paths=opts.paths,
            root=Path(opts.root),
            config=Path(opts.config) if opts.config is not None else None,
            report=Path(report_target) if report_target else None,
            fail_on_violations=opts.fail_on_violations,
            fail_on_type_ambiguities=opts.fail_on_type_ambiguities,
            baseline=Path(opts.baseline) if opts.baseline else None,
            baseline_write=opts.baseline_write if opts.baseline else None,
            decision_snapshot=Path(decision_snapshot_target)
            if decision_snapshot_target
            else None,
            exclude=opts.exclude,
            filter_bundle=DataflowFilterBundle(
                ignore_params_csv=opts.ignore_params,
                transparent_decorators_csv=opts.transparent_decorators,
            ),
            allow_external=opts.allow_external,
            strictness=opts.strictness,
            lint=bool(opts.lint or opts.lint_jsonl or opts.lint_sarif),
            language=opts.language,
            ingest_profile=opts.ingest_profile,
            aspf_trace_json=Path(aspf_trace_json_target)
            if aspf_trace_json_target
            else None,
            aspf_import_trace=[Path(path) for path in aspf_import_trace],
            aspf_equivalence_against=[Path(path) for path in aspf_equivalence_against],
            aspf_opportunities_json=Path(aspf_opportunities_json_target)
            if aspf_opportunities_json_target
            else None,
            aspf_state_json=Path(aspf_state_json_target)
            if aspf_state_json_target
            else None,
            aspf_import_state=[Path(path) for path in aspf_import_state],
            aspf_delta_jsonl=Path(aspf_delta_jsonl_target)
            if aspf_delta_jsonl_target
            else None,
            aspf_semantic_surface=list(aspf_semantic_surface),
        )
    )
    payload.update(
        {
            "dot": dot_target,
            "no_recursive": opts.no_recursive,
            "max_components": opts.max_components,
            "type_audit": opts.type_audit,
            "type_audit_report": opts.type_audit_report,
            "type_audit_max": opts.type_audit_max,
            "synthesis_plan": synthesis_plan_target,
            "synthesis_report": opts.synthesis_report,
            "synthesis_max_tier": opts.synthesis_max_tier,
            "synthesis_min_bundle_size": opts.synthesis_min_bundle_size,
            "synthesis_allow_singletons": opts.synthesis_allow_singletons,
            "synthesis_protocols": synthesis_protocols_target,
            "synthesis_protocols_kind": opts.synthesis_protocols_kind,
            "refactor_plan": opts.refactor_plan,
            "refactor_plan_json": refactor_plan_json_target,
            "fingerprint_synth_json": fingerprint_synth_json_target,
            "fingerprint_provenance_json": fingerprint_provenance_json_target,
            "fingerprint_deadness_json": fingerprint_deadness_json_target,
            "fingerprint_coherence_json": fingerprint_coherence_json_target,
            "fingerprint_rewrite_plans_json": fingerprint_rewrite_plans_json_target,
            "fingerprint_exception_obligations_json": (
                fingerprint_exception_obligations_json_target
            ),
            "fingerprint_handledness_json": fingerprint_handledness_json_target,
            "synthesis_merge_overlap": opts.synthesis_merge_overlap,
            "structure_tree": structure_tree_target,
            "structure_metrics": structure_metrics_target,
            "order_policy": getattr(opts, "order_policy", None),
            "order_telemetry": getattr(opts, "order_telemetry", None),
            "order_enforce_canonical_allowlist": getattr(
                opts, "order_enforce_canonical_allowlist", None
            ),
            "order_deadline_probe": getattr(opts, "order_deadline_probe", None),
            "derivation_cache_max_entries": getattr(
                opts, "derivation_cache_max_entries", None
            ),
            "projection_registry_gas_limit": getattr(
                opts, "projection_registry_gas_limit", None
            ),
        }
    )
    return payload
