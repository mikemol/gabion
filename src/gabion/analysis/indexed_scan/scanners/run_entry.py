# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterable, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

from gabion.analysis.aspf.aspf import Forest
from gabion.analysis.dataflow.engine.dataflow_contracts import AnalysisResult, AuditConfig
from gabion.analysis.foundation.json_types import JSONValue
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
    type_constructor_registry_cls: Callable[[object], object]
    default_marker_aliases: Mapping[object, object]
    audit_config_cls: Callable[..., AuditConfig]
    forest_cls: Callable[[], Forest]
    run_output_context_factory: Callable[..., object]
    finalize_run_outputs_fn: Callable[..., object]


def resolve_baseline_path(path: object, root: Path) -> object:
    if not path:
        return None
    baseline = Path(str(path))
    if not baseline.is_absolute():
        baseline = root / baseline
    return baseline


def resolve_synth_registry_path(path: object, root: Path) -> object:
    if not path:
        return None
    value = str(path).strip()
    if not value:
        return None
    if value.endswith("/LATEST/fingerprint_synth.json"):
        marker = Path(root) / value.replace(
            "/LATEST/fingerprint_synth.json",
            "/LATEST.txt",
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


def normalize_transparent_decorators(
    value: object,
    *,
    check_deadline_fn: Callable[[], None] = check_deadline,
) -> object:
    check_deadline_fn()
    if value is not None:
        items: list[str] = []
        value_type = type(value)
        if value_type is str:
            items = [part.strip() for part in str(value).split(",") if part.strip()]
        elif value_type in {list, tuple, set}:
            for item in value:
                check_deadline_fn()
                if type(item) is str:
                    items.extend([part.strip() for part in str(item).split(",") if part.strip()])
        if items:
            return set(items)
    return None


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

    exclude_dirs = None
    if args.exclude is not None:
        exclude_dirs = []
        for entry in args.exclude:
            check_deadline_fn()
            for part in entry.split(","):
                check_deadline_fn()
                part = part.strip()
                if part:
                    exclude_dirs.append(part)

    ignore_params = None
    if args.ignore_params is not None:
        ignore_params = [part.strip() for part in args.ignore_params.split(",") if part.strip()]

    transparent_decorators = None
    if args.transparent_decorators is not None:
        transparent_decorators = [
            part.strip() for part in args.transparent_decorators.split(",") if part.strip()
        ]

    config_path = Path(args.config) if args.config else None
    root_path = Path(args.root)

    defaults = deps.dataflow_defaults_fn(root_path, config_path)
    synth_defaults = deps.synthesis_defaults_fn(root_path, config_path)
    decision_section = deps.decision_defaults_fn(root_path, config_path)
    decision_tiers = deps.decision_tier_map_fn(decision_section)
    decision_require = deps.decision_require_tiers_fn(decision_section)

    exception_section = deps.exception_defaults_fn(root_path, config_path)
    never_exceptions = set(deps.exception_marker_family_fn(exception_section, "never"))
    never_exceptions.update(deps.exception_never_list_fn(exception_section))
    all_marker_aliases = {
        alias
        for aliases in deps.default_marker_aliases.values()
        for alias in aliases
    }
    never_exceptions.update(all_marker_aliases)

    fingerprint_section = deps.fingerprint_defaults_fn(root_path, config_path)
    synth_min_occurrences = 0
    synth_version = "synth@1"
    synth_registry_path = None
    fingerprint_seed_path = None
    try:
        synth_min_occurrences = int(fingerprint_section.get("synth_min_occurrences", 0) or 0)
    except (TypeError, ValueError):
        synth_min_occurrences = 0
    synth_version = str(fingerprint_section.get("synth_version", synth_version) or synth_version)
    synth_registry_path = fingerprint_section.get("synth_registry_path")
    fingerprint_seed_path = fingerprint_section.get("seed_registry_path")
    if fingerprint_seed_path is None:
        fingerprint_seed_path = fingerprint_section.get("fingerprint_seed_path")

    fingerprint_registry = None
    fingerprint_index: dict[object, set[str]] = {}
    fingerprint_seed_revision = None
    constructor_registry = None
    synth_registry = None

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
        seed_payload = None
        if fingerprint_seed_path:
            resolved_seed = deps.resolve_synth_registry_path_fn(fingerprint_seed_path, root_path)
            if resolved_seed is not None:
                try:
                    seed_payload = deps.load_json_fn(resolved_seed)
                except (OSError, UnicodeError, ValueError):
                    seed_payload = None
        registry, index = deps.build_fingerprint_registry_fn(
            fingerprint_spec,
            registry_seed=seed_payload,
        )
        if index:
            fingerprint_registry = registry
            fingerprint_index = index
            constructor_registry = deps.type_constructor_registry_cls(registry)
            if synth_registry_path:
                resolved = deps.resolve_synth_registry_path_fn(synth_registry_path, root_path)
                if resolved is not None:
                    try:
                        payload = deps.load_json_fn(resolved)
                    except (OSError, UnicodeError, ValueError):
                        payload = None
                else:
                    payload = None
                if type(payload) is dict:
                    synth_registry = deps.build_synth_registry_from_payload_fn(payload, registry)

    merged = deps.merge_payload_fn(
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

    exclude_dirs_set = set(merged.get("exclude", []) or [])
    ignore_params_set = set(merged.get("ignore_params", []) or [])
    decision_ignore_params = set(ignore_params_set)
    decision_ignore_params.update(deps.decision_ignore_list_fn(decision_section))

    allow_external = bool(merged.get("allow_external", False))
    strictness = merged.get("strictness") or "high"
    if strictness not in {"high", "low"}:
        strictness = "high"

    transparent_decorators_normalized = normalize_transparent_decorators(
        merged.get("transparent_decorators"),
        check_deadline_fn=check_deadline_fn,
    )
    deadline_roots = set(deps.dataflow_deadline_roots_fn(merged))
    adapter_payload = deps.dataflow_adapter_payload_fn(merged)
    required_analysis_surfaces = {
        str(item)
        for item in deps.dataflow_required_surfaces_fn(merged)
        if type(item) is str and str(item)
    }

    config = deps.audit_config_cls(
        project_root=root_path,
        exclude_dirs=exclude_dirs_set,
        ignore_params=ignore_params_set,
        decision_ignore_params=decision_ignore_params,
        external_filter=not allow_external,
        strictness=strictness,
        transparent_decorators=transparent_decorators_normalized,
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
        adapter_contract=deps.normalize_adapter_contract_fn(adapter_payload),
        required_analysis_surfaces=required_analysis_surfaces,
    )

    baseline_path = deps.resolve_baseline_path_fn(merged.get("baseline"), root_path)
    baseline_write = args.baseline_write
    if baseline_write and baseline_path is None:
        print("Baseline path required for --baseline-write.", file=sys.stderr)
        return 2

    paths = deps.iter_paths_fn(args.paths, config)
    decision_snapshot_path = args.emit_decision_snapshot

    include_decisions = bool(args.report) or bool(decision_snapshot_path) or bool(args.fail_on_violations)
    if decision_tiers:
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
        config=config,
    )

    return deps.finalize_run_outputs_fn(
        context=deps.run_output_context_factory(
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
            fingerprint_exception_obligations_json=fingerprint_exception_obligations_json,
            fingerprint_handledness_json=fingerprint_handledness_json,
        ),
        emit_report_fn=emit_report_fn,
        compute_violations_fn=compute_violations_fn,
    ).exit_code


__all__ = [
    "RunImplDeps",
    "analysis_deadline_scope",
    "normalize_transparent_decorators",
    "resolve_baseline_path",
    "resolve_synth_registry_path",
    "run_impl",
]
