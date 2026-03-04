# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

from gabion.analysis.aspf.aspf import Forest
from gabion.analysis.dataflow.engine.dataflow_contracts import AnalysisResult, AuditConfig
from gabion.analysis.dataflow.engine.runtime_bootstrap import (
    build_runtime_bootstrap,
    normalize_transparent_decorators as _normalize_transparent_decorators_shared,
    resolve_baseline_path,
    resolve_synth_registry_path,
)
from gabion.analysis.foundation.marker_protocol import DEFAULT_MARKER_ALIASES
from gabion.analysis.foundation.timeout_context import (
    Deadline, GasMeter, TimeoutTickCarrier, check_deadline, deadline_clock_scope, deadline_scope, forest_scope)
from gabion.invariants import never


@dataclass(frozen=True)
class RunImplDeps:
    dataflow_defaults_fn: Callable[..., Mapping[str, object]]
    synthesis_defaults_fn: Callable[..., Mapping[str, object]]
    decision_defaults_fn: Callable[..., Mapping[str, object]]
    decision_tier_map_fn: Callable[..., dict[str, object]]
    decision_require_tiers_fn: Callable[..., bool]
    decision_ignore_list_fn: Callable[..., list[str]]
    exception_defaults_fn: Callable[..., Mapping[str, object]]
    exception_marker_family_fn: Callable[..., set[str]]
    exception_never_list_fn: Callable[..., set[str]]
    fingerprint_defaults_fn: Callable[..., Mapping[str, object]]
    merge_payload_fn: Callable[..., Mapping[str, object]]
    dataflow_deadline_roots_fn: Callable[..., set[str]]
    dataflow_adapter_payload_fn: Callable[..., object]
    dataflow_required_surfaces_fn: Callable[..., list[object]]
    normalize_adapter_contract_fn: Callable[[object], object]
    resolve_baseline_path_fn: Callable[[object, Path], object]
    resolve_synth_registry_path_fn: Callable[[object, Path], object]
    iter_paths_fn: Callable[[Iterable[str], AuditConfig], list[Path]]
    load_json_fn: Callable[[Path], object]
    build_fingerprint_registry_fn: Callable[..., tuple[object, dict[object, set[str]]]]
    build_synth_registry_from_payload_fn: Callable[..., object]
    coerce_synth_payload_fn: Callable[[object], object]
    fingerprint_json_error_types: tuple[type[BaseException], ...]
    type_constructor_registry_cls: Callable[[object], object]
    default_marker_aliases: Mapping[object, object]
    audit_config_cls: Callable[..., AuditConfig]
    forest_cls: Callable[[], Forest]
    run_output_context_factory: Callable[..., object]
    finalize_run_outputs_fn: Callable[..., object]




def normalize_transparent_decorators(
    value: object,
    *,
    check_deadline_fn: Callable[[], None] = check_deadline,
) -> object:
    return _normalize_transparent_decorators_shared(value, check_deadline_fn=check_deadline_fn)

@contextmanager
def analysis_deadline_scope(args: argparse.Namespace):
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
        stack.enter_context(deadline_scope(Deadline.from_timeout_ticks(timeout_carrier)))
        stack.enter_context(deadline_clock_scope(GasMeter(limit=logical_limit)))
        yield


def run_impl(
    args: argparse.Namespace,
    *,
    deps: RunImplDeps,
    analyze_paths_fn: Callable[..., AnalysisResult],
    emit_report_fn: Callable[..., tuple[str, list[str]]],
    compute_violations_fn: Callable[..., list[str]],
    check_deadline_fn: Callable[[], None] = check_deadline,
) -> int:
    check_deadline_fn()
    if args.fail_on_type_ambiguities:
        args.type_audit = True

    fingerprint_deadness_json = args.fingerprint_deadness_json
    fingerprint_coherence_json = args.fingerprint_coherence_json
    fingerprint_rewrite_plans_json = args.fingerprint_rewrite_plans_json
    fingerprint_exception_obligations_json = args.fingerprint_exception_obligations_json
    fingerprint_handledness_json = args.fingerprint_handledness_json

    bootstrap = build_runtime_bootstrap(
        args,
        strategy=deps,
        check_deadline_fn=check_deadline_fn,
    )
    baseline_write = args.baseline_write
    if baseline_write and bootstrap.baseline_path is None:
        print("Baseline path required for --baseline-write.", file=sys.stderr)
        return 2

    paths = deps.iter_paths_fn(args.paths, bootstrap.config)
    decision_snapshot_path = args.emit_decision_snapshot

    include_decisions = bool(args.report) or bool(decision_snapshot_path) or bool(args.fail_on_violations)
    if bootstrap.decision_tiers:
        include_decisions = True

    include_rewrite_plans = bool(args.report) or bool(fingerprint_rewrite_plans_json)
    include_exception_obligations = bool(args.report) or bool(fingerprint_exception_obligations_json)
    include_handledness_witnesses = bool(args.report) or bool(fingerprint_handledness_json)
    include_never_invariants = bool(args.report)
    include_wl_refinement = bool(args.wl_refinement)
    include_ambiguities = bool(args.report) or bool(args.lint)
    include_coherence = bool(args.report) or bool(fingerprint_coherence_json) or include_rewrite_plans

    forest = deps.forest_cls()
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
        include_bundle_forest=(
            bool(args.report)
            or bool(args.dot)
            or bool(args.fail_on_violations)
            or bool(args.emit_structure_tree)
            or bool(args.emit_structure_metrics)
            or bool(args.emit_decision_snapshot)
        ),
        config=bootstrap.config,
    )

    return deps.finalize_run_outputs_fn(
        context=deps.run_output_context_factory(
            args=args,
            analysis=analysis,
            paths=paths,
            config=bootstrap.config,
            synth_defaults=bootstrap.synth_defaults,
            baseline_path=bootstrap.baseline_path,
            baseline_write=baseline_write,
            decision_snapshot_path=decision_snapshot_path,
            fingerprint_deadness_json=fingerprint_deadness_json,
            fingerprint_coherence_json=fingerprint_coherence_json,
            fingerprint_rewrite_plans_json=fingerprint_rewrite_plans_json,
            fingerprint_exception_obligations_json=fingerprint_exception_obligations_json,
            fingerprint_handledness_json=fingerprint_handledness_json,
        ),
        emit_report_fn=emit_report_fn,
        compute_violations_fn=compute_violations_fn,
    ).exit_code


def run_impl_from_runtime_module(
    args: argparse.Namespace,
    *,
    runtime_module,
    analyze_paths_fn: Callable[..., AnalysisResult],
    emit_report_fn: Callable[..., tuple[str, list[str]]],
    compute_violations_fn: Callable[..., list[str]],
) -> int:
    return run_impl(
        args,
        deps=RunImplDeps(
            dataflow_defaults_fn=runtime_module.dataflow_defaults,
            synthesis_defaults_fn=runtime_module.synthesis_defaults,
            decision_defaults_fn=runtime_module.decision_defaults,
            decision_tier_map_fn=runtime_module.decision_tier_map,
            decision_require_tiers_fn=runtime_module.decision_require_tiers,
            decision_ignore_list_fn=runtime_module.decision_ignore_list,
            exception_defaults_fn=runtime_module.exception_defaults,
            exception_marker_family_fn=runtime_module.exception_marker_family,
            exception_never_list_fn=runtime_module.exception_never_list,
            fingerprint_defaults_fn=runtime_module.fingerprint_defaults,
            merge_payload_fn=runtime_module.merge_payload,
            dataflow_deadline_roots_fn=runtime_module.dataflow_deadline_roots,
            dataflow_adapter_payload_fn=runtime_module.dataflow_adapter_payload,
            dataflow_required_surfaces_fn=runtime_module.dataflow_required_surfaces,
            normalize_adapter_contract_fn=runtime_module.normalize_adapter_contract,
            resolve_baseline_path_fn=runtime_module._resolve_baseline_path,
            resolve_synth_registry_path_fn=runtime_module._resolve_synth_registry_path,
            iter_paths_fn=runtime_module._iter_paths,
            load_json_fn=runtime_module.load_json,
            build_fingerprint_registry_fn=runtime_module.build_fingerprint_registry,
            build_synth_registry_from_payload_fn=runtime_module.build_synth_registry_from_payload,
            coerce_synth_payload_fn=lambda payload: payload if type(payload) is dict else None,
            fingerprint_json_error_types=(OSError, UnicodeError, json.JSONDecodeError, ValueError),
            type_constructor_registry_cls=runtime_module.TypeConstructorRegistry,
            default_marker_aliases=runtime_module.DEFAULT_MARKER_ALIASES,
            audit_config_cls=runtime_module.AuditConfig,
            forest_cls=runtime_module.Forest,
            run_output_context_factory=runtime_module._RunImplOutputContextCore,
            finalize_run_outputs_fn=runtime_module._finalize_run_outputs_impl,
        ),
        analyze_paths_fn=analyze_paths_fn,
        emit_report_fn=emit_report_fn,
        compute_violations_fn=compute_violations_fn,
        check_deadline_fn=runtime_module.check_deadline,
    )


__all__ = [
    "RunImplDeps",
    "analysis_deadline_scope",
    "normalize_transparent_decorators",
    "resolve_baseline_path",
    "resolve_synth_registry_path",
    "run_impl",
    "run_impl_from_runtime_module",
]
