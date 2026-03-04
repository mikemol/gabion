from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from gabion.analysis.dataflow.engine.dataflow_contracts import AuditConfig
from gabion.analysis.foundation.json_types import JSONValue
from gabion.analysis.foundation.marker_protocol import DEFAULT_MARKER_ALIASES


@dataclass(frozen=True)
class RuntimeBoundaryInputs:
    exclude: object
    ignore_params: object
    transparent_decorators: object


@dataclass(frozen=True)
class RuntimeBootstrapState:
    project_root: Path
    synth_defaults: Mapping[str, object]
    decision_tiers: dict[str, object]
    config: AuditConfig
    baseline_path: object
    merged_payload: Mapping[str, object]


class RuntimeBootstrapStrategy:
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
    load_json_fn: Callable[[Path], object]
    build_fingerprint_registry_fn: Callable[..., tuple[object, dict[object, set[str]]]]
    build_synth_registry_from_payload_fn: Callable[..., object]
    coerce_synth_payload_fn: Callable[[object], object]
    fingerprint_json_error_types: tuple[type[BaseException], ...]
    type_constructor_registry_cls: Callable[[object], object]
    default_marker_aliases: Mapping[object, object]
    audit_config_cls: Callable[..., AuditConfig]


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


def resolve_baseline_path(path: object, root: Path) -> object:
    if not path:
        return None
    baseline = Path(str(path))
    if not baseline.is_absolute():
        baseline = root / baseline
    return baseline


def normalize_transparent_decorators(
    value: object,
    *,
    check_deadline_fn: Callable[[], None],
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


def normalize_runtime_boundary_inputs(
    args: argparse.Namespace,
    *,
    check_deadline_fn: Callable[[], None],
) -> RuntimeBoundaryInputs:
    exclude_dirs = None
    if args.exclude is not None:
        exclude_dirs = []
        for entry in args.exclude:
            check_deadline_fn()
            for part in entry.split(","):
                check_deadline_fn()
                stripped = part.strip()
                if stripped:
                    exclude_dirs.append(stripped)

    ignore_params = None
    if args.ignore_params is not None:
        ignore_params = [part.strip() for part in args.ignore_params.split(",") if part.strip()]

    transparent_decorators = None
    if args.transparent_decorators is not None:
        transparent_decorators = [
            part.strip() for part in args.transparent_decorators.split(",") if part.strip()
        ]
    return RuntimeBoundaryInputs(
        exclude=exclude_dirs,
        ignore_params=ignore_params,
        transparent_decorators=transparent_decorators,
    )


def build_runtime_bootstrap(
    args: argparse.Namespace,
    *,
    strategy: RuntimeBootstrapStrategy,
    check_deadline_fn: Callable[[], None],
) -> RuntimeBootstrapState:
    boundary_inputs = normalize_runtime_boundary_inputs(args, check_deadline_fn=check_deadline_fn)
    project_root = Path(args.root)
    config_path = Path(args.config) if args.config else None

    defaults = strategy.dataflow_defaults_fn(project_root, config_path)
    synth_defaults = strategy.synthesis_defaults_fn(project_root, config_path)
    decision_section = strategy.decision_defaults_fn(project_root, config_path)
    decision_tiers = strategy.decision_tier_map_fn(decision_section)
    decision_require = strategy.decision_require_tiers_fn(decision_section)

    exception_section = strategy.exception_defaults_fn(project_root, config_path)
    never_exceptions = set(strategy.exception_marker_family_fn(exception_section, "never"))
    never_exceptions.update(strategy.exception_never_list_fn(exception_section))
    all_marker_aliases = {alias for aliases in strategy.default_marker_aliases.values() for alias in aliases}
    never_exceptions.update(all_marker_aliases)

    fingerprint_section = strategy.fingerprint_defaults_fn(project_root, config_path)
    synth_min_occurrences = 0
    synth_version = "synth@1"
    synth_registry_path = fingerprint_section.get("synth_registry_path")
    fingerprint_seed_path = fingerprint_section.get("seed_registry_path")
    if fingerprint_seed_path is None:
        fingerprint_seed_path = fingerprint_section.get("fingerprint_seed_path")

    try:
        synth_min_occurrences = int(fingerprint_section.get("synth_min_occurrences", 0) or 0)
    except (TypeError, ValueError):
        synth_min_occurrences = 0
    synth_version = str(fingerprint_section.get("synth_version", synth_version) or synth_version)

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
            resolved_seed = strategy.resolve_synth_registry_path_fn(fingerprint_seed_path, project_root)
            if resolved_seed is not None:
                try:
                    seed_payload = strategy.load_json_fn(resolved_seed)
                except strategy.fingerprint_json_error_types:
                    seed_payload = None
        registry, index = strategy.build_fingerprint_registry_fn(
            fingerprint_spec,
            registry_seed=seed_payload,
        )
        if index:
            fingerprint_registry = registry
            fingerprint_index = index
            constructor_registry = strategy.type_constructor_registry_cls(registry)
            if synth_registry_path:
                resolved = strategy.resolve_synth_registry_path_fn(synth_registry_path, project_root)
                payload = None
                if resolved is not None:
                    try:
                        payload = strategy.load_json_fn(resolved)
                    except strategy.fingerprint_json_error_types:
                        payload = None
                payload_mapping = strategy.coerce_synth_payload_fn(payload)
                if payload_mapping is not None:
                    synth_registry = strategy.build_synth_registry_from_payload_fn(
                        payload_mapping,
                        registry,
                    )

    merged = strategy.merge_payload_fn(
        {
            "exclude": boundary_inputs.exclude,
            "ignore_params": boundary_inputs.ignore_params,
            "allow_external": args.allow_external,
            "strictness": args.strictness,
            "baseline": args.baseline,
            "transparent_decorators": boundary_inputs.transparent_decorators,
        },
        defaults,
    )

    exclude_dirs = set(merged.get("exclude", []) or [])
    ignore_params_set = set(merged.get("ignore_params", []) or [])
    decision_ignore_params = set(ignore_params_set)
    decision_ignore_params.update(strategy.decision_ignore_list_fn(decision_section))
    allow_external = bool(merged.get("allow_external", False))

    strictness = merged.get("strictness") or "high"
    if strictness not in {"high", "low"}:
        strictness = "high"

    transparent_decorators_normalized = normalize_transparent_decorators(
        merged.get("transparent_decorators"),
        check_deadline_fn=check_deadline_fn,
    )

    deadline_roots = set(strategy.dataflow_deadline_roots_fn(merged))
    adapter_payload = strategy.dataflow_adapter_payload_fn(merged)
    required_analysis_surfaces = {
        str(item)
        for item in strategy.dataflow_required_surfaces_fn(merged)
        if type(item) is str and str(item)
    }

    config = strategy.audit_config_cls(
        project_root=project_root,
        exclude_dirs=exclude_dirs,
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
        adapter_contract=strategy.normalize_adapter_contract_fn(adapter_payload),
        required_analysis_surfaces=required_analysis_surfaces,
    )

    baseline_path = strategy.resolve_baseline_path_fn(merged.get("baseline"), project_root)
    return RuntimeBootstrapState(
        project_root=project_root,
        synth_defaults=synth_defaults,
        decision_tiers=decision_tiers,
        config=config,
        baseline_path=baseline_path,
        merged_payload=merged,
    )


__all__ = [
    "DEFAULT_MARKER_ALIASES",
    "RuntimeBootstrapState",
    "RuntimeBootstrapStrategy",
    "RuntimeBoundaryInputs",
    "build_runtime_bootstrap",
    "normalize_runtime_boundary_inputs",
    "normalize_transparent_decorators",
    "resolve_baseline_path",
    "resolve_synth_registry_path",
]
