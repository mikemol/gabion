from __future__ import annotations
# gabion:boundary_normalization_module
# gabion:decision_protocol_module

"""Owned raw runtime entry surface extracted from legacy_dataflow_monolith."""

import argparse
import json
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, cast

from gabion.analysis.aspf.aspf import Forest
from gabion.analysis.foundation.baseline_io import load_json
from gabion.analysis.dataflow.io.dataflow_baseline_gates import _resolve_baseline_path
from gabion.analysis.dataflow.engine.dataflow_contracts import AnalysisResult, AuditConfig
from gabion.analysis.dataflow.engine.dataflow_ingest_helpers import resolve_analysis_paths
from gabion.analysis.dataflow.engine.dataflow_pipeline import analyze_paths
from gabion.analysis.dataflow.io.dataflow_reporting import (
    compute_violations as _compute_violations, render_report as _emit_report)
from gabion.analysis.dataflow.io.dataflow_run_outputs import (
    DataflowRunOutputContext as _RunImplOutputContextCore, finalize_run_outputs as _finalize_run_outputs_impl)
from gabion.analysis.foundation.json_types import JSONValue, JSONObject
from gabion.analysis.foundation.marker_protocol import DEFAULT_MARKER_ALIASES
from gabion.analysis.foundation.resume_codec import mapping_or_none
from gabion.analysis.foundation.timeout_context import (
    Deadline, GasMeter, TimeoutExceeded, TimeoutTickCarrier, check_deadline, deadline_clock_scope, deadline_scope, forest_scope)
from gabion.analysis.core.type_fingerprints import (
    Fingerprint, TypeConstructorRegistry, build_fingerprint_registry, build_synth_registry_from_payload)
from gabion.config import (
    dataflow_adapter_payload, dataflow_deadline_roots, dataflow_defaults, dataflow_required_surfaces, decision_defaults, decision_ignore_list, decision_require_tiers, decision_tier_map, exception_defaults, exception_marker_family, exception_never_list, fingerprint_defaults, merge_payload, synthesis_defaults)
from gabion.invariants import never


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
    return {
        "name": name,
        "capabilities": {str(key): bool(capabilities[key]) for key in capabilities},
    }


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


def _iter_paths(paths: Iterable[str], config: AuditConfig) -> list[Path]:
    return resolve_analysis_paths(paths, config=config)

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


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
