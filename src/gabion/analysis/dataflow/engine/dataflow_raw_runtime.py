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
from gabion.analysis.indexed_scan.scanners.run_entry import RunImplDeps, run_impl
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
    return run_impl(
        args,
        deps=RunImplDeps(
            dataflow_defaults_fn=dataflow_defaults,
            synthesis_defaults_fn=synthesis_defaults,
            decision_defaults_fn=decision_defaults,
            decision_tier_map_fn=decision_tier_map,
            decision_require_tiers_fn=decision_require_tiers,
            decision_ignore_list_fn=decision_ignore_list,
            exception_defaults_fn=exception_defaults,
            exception_marker_family_fn=exception_marker_family,
            exception_never_list_fn=exception_never_list,
            fingerprint_defaults_fn=fingerprint_defaults,
            merge_payload_fn=merge_payload,
            dataflow_deadline_roots_fn=dataflow_deadline_roots,
            dataflow_adapter_payload_fn=dataflow_adapter_payload,
            dataflow_required_surfaces_fn=dataflow_required_surfaces,
            normalize_adapter_contract_fn=normalize_adapter_contract,
            resolve_baseline_path_fn=_resolve_baseline_path,
            resolve_synth_registry_path_fn=_resolve_synth_registry_path,
            iter_paths_fn=_iter_paths,
            load_json_fn=load_json,
            build_fingerprint_registry_fn=build_fingerprint_registry,
            build_synth_registry_from_payload_fn=build_synth_registry_from_payload,
            coerce_synth_payload_fn=lambda payload: mapping_or_none(cast(JSONValue, payload)),
            fingerprint_json_error_types=(OSError, UnicodeError, json.JSONDecodeError, ValueError),
            type_constructor_registry_cls=TypeConstructorRegistry,
            default_marker_aliases=DEFAULT_MARKER_ALIASES,
            audit_config_cls=AuditConfig,
            forest_cls=Forest,
            run_output_context_factory=_RunImplOutputContextCore,
            finalize_run_outputs_fn=_finalize_run_outputs_impl,
        ),
        analyze_paths_fn=analyze_paths_fn,
        emit_report_fn=emit_report_fn,
        compute_violations_fn=compute_violations_fn,
        check_deadline_fn=check_deadline,
    )


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
