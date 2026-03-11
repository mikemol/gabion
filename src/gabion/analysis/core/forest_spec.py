from __future__ import annotations

import json
from dataclasses import dataclass, field
from collections.abc import Iterable, Mapping

from gabion.json_types import JSONValue
from gabion.analysis.foundation.resume_codec import mapping_optional
from gabion.order_contract import sort_once
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.invariants import never


@dataclass(frozen=True)
class ForestCollectorSpec:
    name: str
    outputs: tuple[str, ...] = ()
    params: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True)
class ForestSpec:
    spec_version: int
    name: str
    collectors: tuple[ForestCollectorSpec, ...] = ()
    params: dict[str, JSONValue] = field(default_factory=dict)
    declared_outputs: tuple[str, ...] = ()


def build_forest_spec(
    *,
    include_bundle_forest: bool,
    include_decision_surfaces: bool,
    include_value_decision_surfaces: bool,
    include_never_invariants: bool,
    include_taint_projections: bool = False,
    include_wl_refinement: bool = False,
    include_ambiguities: bool = False,
    include_deadline_obligations: bool = False,
    include_lint_findings: bool = False,
    include_all_sites: bool = True,
    ignore_params: Iterable[str] = (),
    decision_ignore_params: Iterable[str] = (),
    transparent_decorators: Iterable[str] = (),
    strictness: str = "high",
    decision_tiers: object = None,
    require_tiers: bool = False,
    external_filter: bool = True,
) -> ForestSpec:
    # dataflow-bundle: decision_ignore_params, ignore_params, transparent_decorators
    # dataflow-bundle: external_filter, include_all_sites, require_tiers
    collectors: list[ForestCollectorSpec] = []
    declared_outputs: set[str] = set()

    if include_bundle_forest:
        outputs = (
            "FunctionSite",
            "ParamSet",
            "SignatureBundle",
            "ConfigBundle",
            "DataclassBundle",
            "MarkerBundle",
            "DataclassCallBundle",
        )
        declared_outputs.update(outputs)
        collectors.append(
            ForestCollectorSpec(
                name="bundle_forest",
                outputs=outputs,
                params={
                    "include_all_sites": bool(include_all_sites),
                    "ignore_params": _sorted_strings(ignore_params),
                    "strictness": str(strictness),
                    "transparent_decorators": _sorted_strings(transparent_decorators),
                },
            )
        )

    if include_decision_surfaces:
        outputs = ("DecisionSurface",)
        declared_outputs.update(outputs)
        collectors.append(
            ForestCollectorSpec(
                name="decision_surface",
                outputs=outputs,
                params={
                    "ignore_params": _sorted_strings(decision_ignore_params),
                    "strictness": str(strictness),
                    "transparent_decorators": _sorted_strings(transparent_decorators),
                    "decision_tiers": _normalize_decision_tiers(decision_tiers),
                    "require_tiers": bool(require_tiers),
                    "external_filter": bool(external_filter),
                },
            )
        )

    if include_value_decision_surfaces:
        outputs = ("ValueDecisionSurface",)
        declared_outputs.update(outputs)
        collectors.append(
            ForestCollectorSpec(
                name="value_decision_surface",
                outputs=outputs,
                params={
                    "ignore_params": _sorted_strings(decision_ignore_params),
                    "strictness": str(strictness),
                    "transparent_decorators": _sorted_strings(transparent_decorators),
                    "decision_tiers": _normalize_decision_tiers(decision_tiers),
                    "require_tiers": bool(require_tiers),
                    "external_filter": bool(external_filter),
                },
            )
        )

    if include_never_invariants:
        outputs = ("NeverInvariantSink",)
        declared_outputs.update(outputs)
        collectors.append(
            ForestCollectorSpec(
                name="never_invariants",
                outputs=outputs,
                params={
                    "ignore_params": _sorted_strings(ignore_params),
                },
            )
        )

    if include_taint_projections:
        outputs = (
            "TaintBoundaryLocus",
            "TaintWitness",
            "TaintLedgerRecord",
            "TaintLifecycleDecision",
        )
        declared_outputs.update(outputs)
        collectors.append(
            ForestCollectorSpec(
                name="taint_projection",
                outputs=outputs,
                params={"strictness": str(strictness)},
            )
        )

    if include_ambiguities:
        outputs = ("SuiteSite", "SuiteSiteInFunction", "CallCandidate", "PartitionWitness")
        declared_outputs.update(outputs)
        collectors.append(
            ForestCollectorSpec(
                name="call_ambiguities",
                outputs=outputs,
                params={
                    "ignore_params": _sorted_strings(ignore_params),
                    "strictness": str(strictness),
                    "transparent_decorators": _sorted_strings(transparent_decorators),
                    "external_filter": bool(external_filter),
                },
            )
        )

    if include_deadline_obligations:
        outputs = (
            "SuiteSite",
            "SuiteSiteInFunction",
            "CallCandidate",
            "CallResolutionObligation",
            "DeadlineObligation",
            "SpecFacet",
        )
        declared_outputs.update(outputs)
        collectors.append(
            ForestCollectorSpec(
                name="deadline_obligations",
                outputs=outputs,
                params={
                    "ignore_params": _sorted_strings(ignore_params),
                },
            )
        )

    if include_lint_findings:
        outputs = ("LintFinding", "SpecFacet")
        declared_outputs.update(outputs)
        collectors.append(
            ForestCollectorSpec(
                name="lint_findings",
                outputs=outputs,
                params={},
            )
        )

    if include_wl_refinement:
        outputs = ("SuiteContains", "WLLabel", "SpecFacet", "NeverInvariantSink")
        declared_outputs.update(outputs)
        collectors.append(
            ForestCollectorSpec(
                name="wl_refinement",
                outputs=outputs,
                params={},
            )
        )

    return ForestSpec(
        spec_version=1,
        name="forest_v1",
        collectors=tuple(collectors),
        params={},
        declared_outputs=tuple(sort_once(declared_outputs, source = 'src/gabion/analysis/forest_spec.py:188')),
    )


def default_forest_spec(
    *,
    include_bundle_forest: bool = True,
    include_decision_surfaces: bool = False,
    include_value_decision_surfaces: bool = False,
    include_never_invariants: bool = False,
    include_taint_projections: bool = False,
    include_wl_refinement: bool = False,
    include_ambiguities: bool = False,
    include_deadline_obligations: bool = False,
    include_lint_findings: bool = False,
) -> ForestSpec:
    return build_forest_spec(
        include_bundle_forest=include_bundle_forest,
        include_decision_surfaces=include_decision_surfaces,
        include_value_decision_surfaces=include_value_decision_surfaces,
        include_never_invariants=include_never_invariants,
        include_taint_projections=include_taint_projections,
        include_wl_refinement=include_wl_refinement,
        include_ambiguities=include_ambiguities,
        include_deadline_obligations=include_deadline_obligations,
        include_lint_findings=include_lint_findings,
        include_all_sites=True,
        ignore_params=(),
        decision_ignore_params=(),
        transparent_decorators=(),
        strictness="high",
        decision_tiers=None,
        require_tiers=False,
        external_filter=True,
    )


def forest_spec_to_dict(spec: ForestSpec) -> dict[str, JSONValue]:
    return {
        "spec_version": spec.spec_version,
        "name": spec.name,
        "params": dict(spec.params),
        "declared_outputs": list(spec.declared_outputs),
        "collectors": [
            {
                "name": collector.name,
                "outputs": list(collector.outputs),
                "params": dict(collector.params),
            }
            for collector in spec.collectors
        ],
    }


def forest_spec_from_dict(payload: Mapping[str, JSONValue]) -> ForestSpec:
    check_deadline()
    spec_version = payload.get("spec_version", 1)
    try:
        version = int(spec_version) if spec_version is not None else 1
    except (TypeError, ValueError):
        version = 1
    spec_name = str(payload.get("name", "") or "")
    spec_params = _string_key_mapping(payload.get("params"))
    declared_outputs = _string_tuple_payload(payload.get("declared_outputs", []))
    collectors_payload = payload.get("collectors", [])
    collectors: list[ForestCollectorSpec] = []
    match collectors_payload:
        case list() as collector_entries:
            for entry in collector_entries:
                check_deadline()
                collector_payload = mapping_optional(entry)
                if collector_payload is not None:
                    collector_name = str(collector_payload.get("name", "") or "").strip()
                    if collector_name:
                        collectors.append(
                            ForestCollectorSpec(
                                name=collector_name,
                                outputs=_string_tuple_payload(collector_payload.get("outputs", [])),
                                params=_string_key_mapping(collector_payload.get("params")),
                            )
                        )
    return ForestSpec(
        spec_version=version,
        name=spec_name,
        collectors=tuple(collectors),
        params=spec_params,
        declared_outputs=declared_outputs,
    )


def normalize_forest_spec(spec: ForestSpec) -> dict[str, JSONValue]:
    collectors = sort_once(spec.collectors, key=lambda entry: entry.name, source = 'src/gabion/analysis/forest_spec.py:295')
    return {
        "spec_version": int(spec.spec_version) if spec.spec_version else 1,
        "name": str(spec.name),
        "params": _normalize_value(dict(spec.params)),
        "declared_outputs": _sorted_strings(spec.declared_outputs),
        "collectors": [
            {
                "name": collector.name,
                "outputs": _sorted_strings(collector.outputs),
                "params": _normalize_value(dict(collector.params)),
            }
            for collector in collectors
        ],
    }


def forest_spec_canonical_json(spec: ForestSpec) -> str:
    payload = normalize_forest_spec(spec)
    return json.dumps(payload, sort_keys=False, separators=(",", ":"))


def forest_spec_hash_spec(spec: ForestSpec) -> str:
    return forest_spec_canonical_json(spec)


def forest_spec_hash(spec: object) -> str:
    match spec:
        case str() as spec_hash:
            return spec_hash
        case ForestSpec() as spec_model:
            return forest_spec_hash_spec(spec_model)
    spec_payload = mapping_optional(spec) or {}
    return forest_spec_hash_spec(forest_spec_from_dict(spec_payload))


def forest_spec_metadata(spec: ForestSpec) -> dict[str, JSONValue]:
    return {
        "generated_by_forest_spec_id": forest_spec_hash_spec(spec),
        "generated_by_forest_spec": normalize_forest_spec(spec),
    }


def _normalize_decision_tiers(
    tiers: object,
) -> dict[str, int]:
    check_deadline()
    if not tiers:
        return {}
    normalized: dict[str, int] = {}
    for key, value in tiers.items():
        check_deadline()
        name = str(key).strip()
        if name:
            try:
                tier = int(value)
            except (TypeError, ValueError):
                tier = None
            if tier is not None:
                normalized[name] = tier
    return {key: normalized[key] for key in sort_once(normalized, source = 'src/gabion/analysis/forest_spec.py:353')}


def _iterable_values(values: object) -> Iterable[object]:
    match values:
        case Iterable() as iterable_values:
            return iterable_values
    return ()


def _sorted_strings(values: object) -> list[str]:
    cleaned = {
        str(value).strip()
        for value in _iterable_values(values)
        if str(value).strip()
    }
    return sort_once(cleaned, source = 'src/gabion/analysis/forest_spec.py:360')


def _is_string_value(value: JSONValue) -> bool:
    match value:
        case str():
            return True
    return False


def _normalize_value(value: JSONValue) -> JSONValue:
    check_deadline()
    match value:
        case dict() as value_mapping:
            return {
                str(k): _normalize_value(value_mapping[k])
                for k in sort_once(value_mapping, source = 'src/gabion/analysis/forest_spec.py:366')
            }
        case list() as value_list:
            all_strings = all(_is_string_value(entry) for entry in value_list)
            if value_list and all_strings:
                return _sorted_strings([str(entry) for entry in value_list])
            return [_normalize_value(entry) for entry in value_list]
    return value


def _string_key_mapping(value: object) -> dict[str, JSONValue]:
    payload = mapping_optional(value)
    if payload is None:
        return {}
    return {str(k): v for k, v in payload.items()}


def _string_tuple_payload(value: object) -> tuple[str, ...]:
    match value:
        case list() as value_list:
            return tuple(str(entry) for entry in value_list if str(entry).strip())
    return ()
