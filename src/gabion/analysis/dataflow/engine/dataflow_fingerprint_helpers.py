from __future__ import annotations

from gabion.json_types import JSONObject, JSONValue
from gabion.invariants import never, todo
"""Fingerprint helper ownership module during runtime retirement."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence, cast

from gabion.analysis.aspf.aspf_core import parse_2cell_witness
from gabion.analysis.aspf.aspf_decision_surface import classify_drift_by_homotopy
from gabion.analysis.dataflow.engine.dataflow_decision_surfaces import (
    compute_fingerprint_coherence as _ds_compute_fingerprint_coherence, compute_fingerprint_rewrite_plans as _ds_compute_fingerprint_rewrite_plans)
from gabion.analysis.dataflow.io.dataflow_snapshot_io import normalize_snapshot_path
from gabion.analysis.semantics.evidence import Site, exception_obligation_summary_for_site
from gabion.analysis.foundation.json_types import JSONObject, JSONValue
from gabion.analysis.foundation.resume_codec import (
    mapping_default_empty, mapping_optional, sequence_optional, str_tuple_from_sequence)
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.analysis.core.type_fingerprints import (
    Fingerprint, FingerprintDimension, PrimeRegistry, _collect_base_atoms, _collect_constructors, bundle_fingerprint_dimensional, fingerprint_carrier_soundness, fingerprint_identity_payload, format_fingerprint, synth_registry_payload)
from gabion.order_contract import sort_once
from gabion.refactor.rewrite_plan import rewrite_plan_schema, validate_rewrite_plan_payload

OptionalJsonObjectList = list[JSONObject] | None

_PR412_RAW_IDENTITY_LAYER_CONSUMER = todo(
    reasoning={
        "summary": "PR-412 canonical identity contract adoption still partial in raw identity layer consumers",
        "control": "pr412.identity_payload.raw_consumer",
        "blocking_dependencies": (
            "replace_raw_identity_layers_access_with_typed_projection",
        ),
    },
    owner="gabion.analysis.dataflow.engine",
    links=[{"kind": "object_id", "value": "pr:412"}],
)

_PR412_RAW_IDENTITY_PAYLOAD_PROPAGATION = todo(
    reasoning={
        "summary": "PR-412 canonical identity contract adoption still partial in raw identity payload propagation",
        "control": "pr412.identity_payload.raw_propagation",
        "blocking_dependencies": (
            "replace_raw_identity_payload_dict_propagation_with_typed_contract_carriers",
        ),
    },
    owner="gabion.analysis.dataflow.engine",
    links=[{"kind": "object_id", "value": "pr:412"}],
)

_PR412_RAW_IDENTITY_CONTRACT_EQUALITY = todo(
    reasoning={
        "summary": "PR-412 canonical identity contract adoption still partial in raw mapping equality checks",
        "control": "pr412.identity_payload.raw_equality",
        "blocking_dependencies": (
            "replace_mapping_equality_with_typed_canonical_identity_contract_comparison",
        ),
    },
    owner="gabion.analysis.dataflow.engine",
    links=[{"kind": "object_id", "value": "pr:412"}],
)


def _compute_fingerprint_warnings(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    annotations_by_path: dict[Path, dict[str, dict[str, object]]],
    *,
    registry: PrimeRegistry,
    index: dict[Fingerprint, set[str]],
    ctor_registry=None,
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
                        hint_list = [str(t) for t in types if t is not None]
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
    annotations_by_path: dict[Path, dict[str, dict[str, object]]],
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
                        _collect_base_atoms(str(hint), atoms)
                        base_keys.update(atom for atom in atoms if atom)
                        _collect_constructors(str(hint), ctor_keys)
    return (
        sort_once(base_keys, source="_collect_fingerprint_atom_keys.base_keys"),
        sort_once(ctor_keys, source="_collect_fingerprint_atom_keys.ctor_keys"),
    )


def _compute_fingerprint_matches(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    annotations_by_path: dict[Path, dict[str, dict[str, object]]],
    *,
    registry: PrimeRegistry,
    index: dict[Fingerprint, set[str]],
    ctor_registry=None,
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
                        hint_list = [str(t) for t in types if t is not None]
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
    project_root: Path,
    index=None,
    ctor_registry=None,
) -> list[JSONObject]:
    check_deadline()
    entries: list[JSONObject] = []
    for path, groups in groups_by_path.items():
        check_deadline()
        path_value = normalize_snapshot_path(path, project_root)
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
                match higher_path_payload:
                    case dict() as higher_path_witness_payload:
                        higher_path_witness = parse_2cell_witness(
                            higher_path_witness_payload
                        )
                    case _:
                        higher_path_witness = None
                        never("unreachable wildcard match fall-through")
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
        matches = tuple(str_tuple_from_sequence(entry.get("glossary_matches")))
        if matches:
            key = ("glossary", matches)
        else:
            base_keys = tuple(str_tuple_from_sequence(entry.get("base_keys")))
            ctor_keys = tuple(str_tuple_from_sequence(entry.get("ctor_keys")))
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


def _compute_fingerprint_rewrite_plans(
    provenance: list[JSONObject],
    coherence: list[JSONObject],
    *,
    synth_version: str,
    exception_obligations=None,
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
    if strata_expect == "exact" and len(context.post_matches) == 1:
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
    raw_obligations = sequence_optional(evidence.get("witness_obligations"))
    if raw_obligations is not None:
        for item in raw_obligations:
            mapped_item = mapping_optional(item)
            if mapped_item is not None:
                obligations.append(mapped_item)
    missing_required: list[str] = []
    aspf_identity_mismatches: list[str] = []
    post_aspf_structure_class = mapping_default_empty(
        context.post_entry.get("aspf_structure_class")
    )
    for item in obligations:
        required = bool(item.get("required"))
        witness_ref = str(item.get("witness_ref", "") or "")
        witness_kind = str(item.get("kind", "witness") or "witness")
        if required and not witness_ref:
            missing_required.append(f"{witness_kind}:missing")
        if witness_kind == "aspf_structure_class_equivalence":
            expected_identity = mapping_optional(item.get("canonical_identity_contract"))
            post_identity_payload = mapping_optional(
                context.post_entry.get("canonical_identity_contract")
            )
            if expected_identity is not None and expected_identity != post_identity_payload:
                aspf_identity_mismatches.append("canonical_identity_contract")
            expected_structure_class = mapping_optional(item.get("aspf_structure_class"))
            if (
                expected_structure_class is not None
                and expected_structure_class != post_aspf_structure_class
            ):
                aspf_identity_mismatches.append("aspf_structure_class")
    post_identity = context.post_entry.get("canonical_identity_contract")
    pre_identity = context.pre.get("canonical_identity_contract")
    identity_ok = True
    if pre_identity is not None:
        identity_ok = pre_identity == post_identity
    elif post_identity is None:
        identity_ok = False
    passed = (not missing_required) and identity_ok and (not aspf_identity_mismatches)
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
            "aspf_identity_mismatches": aspf_identity_mismatches,
            "identity_contract": post_identity,
            "aspf_structure_class": post_aspf_structure_class,
        },
    }


def _summary_unknown_and_discharged(summary: Mapping[str, JSONValue]) -> tuple[int, int]:
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
    pre_summary = mapping_optional(raw_pre_summary)
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
    post_exception_obligations=None,
) -> JSONObject:
    """Verify a single rewrite plan using a post-state provenance artifact."""
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
    issues: list[str] = []
    status = str(plan.get("status", "") or "")
    if status == "ABSTAINED":
        issues.append("plan abstained: preconditions not satisfied")
        abstention = mapping_default_empty(plan.get("abstention"))
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

    post_entry = _find_provenance_entry_for_site(post_provenance, site=site)
    if post_entry is None:
        issues.append("missing post provenance entry for site")
        return {
            "plan_id": plan_id,
            "accepted": False,
            "issues": issues,
            "predicate_results": [],
        }

    pre = mapping_default_empty(plan.get("pre"))
    evidence = mapping_default_empty(plan.get("evidence"))
    raw_pre_remainder = pre.get("remainder")
    if raw_pre_remainder not in (None, {}) and mapping_optional(raw_pre_remainder) is None:
        issues.append("invalid pre remainder payload")
    expected_base = list(pre.get("base_keys") or [])
    expected_ctor = list(pre.get("ctor_keys") or [])
    expected_remainder = mapping_default_empty(pre.get("remainder"))
    post_expectation = mapping_default_empty(plan.get("post_expectation"))
    expected_strata = str(post_expectation.get("match_strata", ""))

    post_base = list(post_entry.get("base_keys") or [])
    post_ctor = list(post_entry.get("ctor_keys") or [])
    post_remainder = mapping_default_empty(post_entry.get("remainder"))
    post_matches = tuple(str_tuple_from_sequence(post_entry.get("glossary_matches")))
    post_strata = _glossary_match_strata(post_matches)

    predicate_results: list[JSONObject] = []

    expected_candidates: list[str] = []
    rewrite = mapping_default_empty(plan.get("rewrite"))
    rewrite_kind = str(rewrite.get("kind", "") or "")
    raw_params_payload = rewrite.get("parameters")
    if raw_params_payload not in (None, {}) and mapping_optional(raw_params_payload) is None:
        issues.append("invalid rewrite parameters payload")
    params = mapping_default_empty(rewrite.get("parameters"))
    expected_candidates = [str(v) for v in (params.get("candidates") or []) if v]

    verification = mapping_default_empty(plan.get("verification"))
    predicates = sequence_optional(verification.get("predicates")) or ()
    requested_predicates: list[JSONObject] = [
        {str(key): mapped_predicate[key] for key in mapped_predicate}
        for predicate in predicates
        for mapped_predicate in (mapping_optional(predicate),)
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
                requested_predicates.append({"kind": kind, "expect": "no-new-remainder"})
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
    post_exception_obligations=None,
) -> list[JSONObject]:
    return [
        verify_rewrite_plan(
            plan,
            post_provenance=post_provenance,
            post_exception_obligations=post_exception_obligations,
        )
        for plan in plans
    ]


def _compute_fingerprint_synth(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    annotations_by_path: dict[Path, dict[str, dict[str, object]]],
    *,
    registry: PrimeRegistry,
    ctor_registry,
    min_occurrences: int,
    version: str,
    existing=None,
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
                types = [
                    fn_annots[param]
                    for param in sort_once(
                        bundle,
                        source="_compute_fingerprint_synth.bundle",
                    )
                ]
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
        from gabion.analysis.core.type_fingerprints import build_synth_registry

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
        tail = cast(Mapping[str, object], entry.get("tail", {}))
        base_keys = cast(list[object], entry.get("base_keys", []))
        ctor_keys = cast(list[object], entry.get("ctor_keys", []))
        remainder = cast(Mapping[str, object], entry.get("remainder", {}))
        details = f"base={base_keys}"
        if ctor_keys:
            details += f" ctor={ctor_keys}"
        if remainder.get("base") not in (0, 1) or remainder.get("ctor") not in (0, 1):
            details += f" remainder=({remainder.get('base')},{remainder.get('ctor')})"
        tail_base = cast(Mapping[str, object], tail.get("base", {}))
        tail_ctor = cast(Mapping[str, object], tail.get("ctor", {}))
        lines.append(
            f"- synth_prime={entry.get('prime')} tail="
            f"{{base={tail_base.get('product')}, "
            f"ctor={tail_ctor.get('product')}}} "
            f"{details}"
        )
    return lines, payload


def _build_synth_registry_payload(
    synth_registry,
    registry: PrimeRegistry,
    *,
    min_occurrences: int,
) -> JSONObject:
    check_deadline()
    entries: list[JSONObject] = []
    for prime, tail in sort_once(
        synth_registry.tails.items(),
        source="_build_synth_registry_payload.tails",
    ):
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
                "base_keys": sort_once(base_keys, source="_build_synth_registry_payload.base_keys"),
                "ctor_keys": sort_once(ctor_keys, source="_build_synth_registry_payload.ctor_keys"),
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


__all__ = [
    "_build_synth_registry_payload",
    "_collect_fingerprint_atom_keys",
    "_compute_fingerprint_coherence",
    "_compute_fingerprint_matches",
    "_compute_fingerprint_provenance",
    "_compute_fingerprint_rewrite_plans",
    "_compute_fingerprint_synth",
    "_compute_fingerprint_warnings",
    "_find_provenance_entry_for_site",
    "_fingerprint_soundness_issues",
    "_glossary_match_strata",
    "_summarize_fingerprint_provenance",
    "verify_rewrite_plan",
    "verify_rewrite_plans",
]
