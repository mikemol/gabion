# gabion:boundary_normalization_module
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable, cast

from gabion.analysis.dataflow_analysis_index import _build_analysis_collection_resume_payload
from gabion.analysis.dataflow_contracts import ReportCarrier
from gabion.analysis.dataflow_indexed_file_scan import _collect_dataclass_registry
from gabion.analysis.dataflow_parse_helpers import _forbid_adhoc_bundle_discovery
from gabion.analysis.dataflow_projection_helpers import (
    report_projection_phase_rank,
    report_projection_specs,
)
from gabion.analysis.dataflow_projection_preview_bridge import preview_section_lines
from gabion.analysis.dataflow_reporting import render_report
from gabion.analysis.dataflow_snapshot_io import extract_report_sections
from gabion.analysis.dataflow_synthesis_runtime_bridge import _collect_config_bundles
from gabion.analysis.json_types import JSONObject
from gabion.analysis.resume_codec import mapping_or_empty, mapping_or_none, sequence_or_none
from gabion.analysis.structure_reuse_classes import build_structure_class, structure_class_payload
from gabion.analysis.timeout_context import check_deadline
from gabion.order_contract import sort_once


def build_analysis_collection_resume_seed(
    *,
    in_progress_paths=(),
):
    in_progress_scan_by_path: dict[Path, object] = {
        path: {"phase": "scan_pending"} for path in in_progress_paths
    }
    return _build_analysis_collection_resume_payload(
        groups_by_path={},
        param_spans_by_path={},
        bundle_sites_by_path={},
        invariant_propositions=[],
        completed_paths=set(),
        in_progress_scan_by_path=in_progress_scan_by_path,
        analysis_index_resume=None,
        file_stage_timings_v1_by_path={},
    )


def compute_structure_reuse(
    snapshot,
    *,
    min_count: int = 2,
    hash_fn=None,
):
    check_deadline()
    if min_count < 2:
        min_count = 2
    files = snapshot.get("files") or []
    root_value = snapshot.get("root")
    root_path = Path(root_value) if type(root_value) is str else None
    bundle_name_map: dict[tuple[str, ...], set[str]] = {}
    if root_path is not None and root_path.exists():
        bundle_name_map = _bundle_name_registry(root_path)
    reuse_map: dict[str, JSONObject] = {}
    warnings: list[str] = []

    def _hash_node(kind: str, value, child_hashes: list[str]) -> str:
        structure_class = build_structure_class(
            kind=kind,
            value=value,
            child_hashes=sort_once(
                child_hashes,
                source="dataflow_structure_reuse.compute_structure_reuse.child_hashes",
            ),
        )
        return structure_class.digest()

    hasher = (
        cast(Callable[[str, object, list[str]], str], hash_fn)
        if callable(hash_fn)
        else _hash_node
    )

    def _record(
        *,
        node_hash: str,
        kind: str,
        location: str,
        value=None,
        child_count=-1,
    ) -> None:
        entry = reuse_map.get(node_hash)
        if entry is None:
            structure_class = build_structure_class(
                kind=kind,
                value=value,
                child_hashes=(),
            )
            entry = {
                "hash": node_hash,
                "kind": kind,
                "count": 0,
                "locations": [],
                "aspf_structure_class": structure_class_payload(structure_class),
            }
            if value is not None:
                entry["value"] = value
            if child_count >= 0:
                entry["child_count"] = child_count
            reuse_map[node_hash] = entry
        entry["count"] += 1
        entry["locations"].append(location)

    file_hashes: list[str] = []
    for file_entry in files:
        check_deadline()
        if type(file_entry) is not dict:
            continue
        file_entry_obj = cast(JSONObject, file_entry)
        file_path = file_entry_obj.get("path")
        if type(file_path) is not str:
            continue
        function_hashes: list[str] = []
        functions = file_entry_obj.get("functions") or []
        for fn_entry in functions:
            check_deadline()
            if type(fn_entry) is not dict:
                continue
            fn_entry_obj = cast(JSONObject, fn_entry)
            fn_name = fn_entry_obj.get("name")
            if type(fn_name) is not str:
                continue
            bundle_hashes: list[str] = []
            bundles = fn_entry_obj.get("bundles") or []
            for bundle in bundles:
                check_deadline()
                if type(bundle) is not list:
                    continue
                bundle_items = cast(list[object], bundle)
                normalized = tuple(
                    sort_once(
                        (str(item) for item in bundle_items),
                        source="dataflow_structure_reuse.compute_structure_reuse.bundle",
                    )
                )
                bundle_hash = hasher("bundle", normalized, [])
                bundle_hashes.append(bundle_hash)
                _record(
                    node_hash=bundle_hash,
                    kind="bundle",
                    location=f"{file_path}::{fn_name}::bundle:{','.join(normalized)}",
                    value=list(normalized),
                )
            fn_hash = hasher("function", None, bundle_hashes)
            function_hashes.append(fn_hash)
            _record(
                node_hash=fn_hash,
                kind="function",
                location=f"{file_path}::{fn_name}",
                child_count=len(bundle_hashes),
            )
        file_hash = hasher("file", None, function_hashes)
        file_hashes.append(file_hash)
        _record(node_hash=file_hash, kind="file", location=f"{file_path}")

    root_hash = hasher("root", None, file_hashes)
    _record(
        node_hash=root_hash,
        kind="root",
        location="root",
        child_count=len(file_hashes),
    )

    reused = [
        entry
        for entry in reuse_map.values()
        if type(entry.get("count")) is int and entry["count"] >= min_count
    ]
    reused = sort_once(
        reused,
        source="dataflow_structure_reuse.compute_structure_reuse.reused",
        key=lambda entry: (
            entry.get("kind", ""),
            -int(entry.get("count", 0)),
            entry.get("hash", ""),
        ),
    )
    suggested: list[JSONObject] = []

    def _reuse_site_from_location(
        *,
        location: str,
        fallback_value: object,
    ) -> JSONObject:
        location_parts = location.split("::")
        path_value = location_parts[0] if location_parts else ""
        function_value = location_parts[1] if len(location_parts) > 1 else ""
        bundle_payload: list[str] = []
        if len(location_parts) > 2 and location_parts[2].startswith("bundle:"):
            raw_bundle = location_parts[2][len("bundle:") :]
            bundle_payload = [part for part in raw_bundle.split(",") if part]
        return {
            "path": path_value,
            "function": function_value,
            "bundle": bundle_payload,
        }

    def _build_suggested_plan_artifact(
        *,
        suggestion: JSONObject,
    ) -> JSONObject:
        kind = str(suggestion.get("kind", ""))
        suggestion_name = str(suggestion.get("suggested_name", ""))
        hash_value = str(suggestion.get("hash", ""))
        locations = sequence_or_none(suggestion.get("locations")) or ()
        sorted_locations = sort_once(
            [str(location) for location in locations if type(location) is str],
            source="dataflow_structure_reuse.compute_structure_reuse.suggested_locations",
        )
        primary_location = sorted_locations[0] if sorted_locations else ""
        site = _reuse_site_from_location(
            location=primary_location,
            fallback_value=suggestion.get("value"),
        )
        witness_obligations = list(
            sequence_or_none(suggestion.get("witness_obligations")) or []
        )
        aspf_witness_requirements = mapping_or_empty(
            suggestion.get("aspf_witness_requirements")
        )
        return {
            "plan_id": f"reuse:{kind}:{hash_value}:{suggestion_name}",
            "status": "UNVERIFIED",
            "site": site,
            "pre": {
                "canonical_identity_contract": suggestion.get("canonical_identity_contract"),
                "aspf_structure_class": suggestion.get("aspf_structure_class"),
            },
            "rewrite": {
                "kind": "BUNDLE_ALIGN" if kind == "bundle" else "AMBIENT_REWRITE",
                "selector": {
                    "hash": hash_value,
                    "locations": sorted_locations,
                },
                "parameters": {
                    "candidates": [suggestion_name] if suggestion_name else [],
                    "strategy": "reuse-lemma" if kind != "bundle" else "reuse-align",
                },
            },
            "evidence": {
                "provenance_id": f"reuse:{hash_value}",
                "coherence_id": f"aspf:{hash_value}",
                "aspf_witness_requirements": aspf_witness_requirements,
                "witness_obligations": witness_obligations,
            },
            "post_expectation": {
                "match_strata": "exact",
                "canonical_structure_class_equivalent": True,
            },
            "verification": {
                "mode": "re-audit",
                "status": "UNVERIFIED",
                "predicates": [
                    {"kind": "base_conservation", "expect": True},
                    *(
                        [{"kind": "ctor_coherence", "expect": True}]
                        if kind == "bundle"
                        else []
                    ),
                    {
                        "kind": "match_strata",
                        "expect": "exact",
                        "candidates": [suggestion_name] if suggestion_name else [],
                    },
                    {"kind": "remainder_non_regression", "expect": "no-new-remainder"},
                    {"kind": "witness_obligation_non_regression", "expect": "stable"},
                ],
            },
        }

    for entry in reused:
        check_deadline()
        kind = entry.get("kind")
        if kind not in {"bundle", "function"}:
            continue
        count = int(entry.get("count", 0))
        hash_value = entry.get("hash")
        if type(hash_value) is not str or not hash_value:
            continue
        suggestion: JSONObject = {
            "hash": hash_value,
            "kind": kind,
            "count": count,
            "suggested_name": f"_gabion_{kind}_lemma_{hash_value[:8]}",
            "locations": entry.get("locations", []),
            "aspf_structure_class": entry.get("aspf_structure_class"),
            "canonical_identity_contract": {
                "identity_kind": "canonical_aspf_structure_class_equivalence",
                "representative": hash_value,
            },
        }
        if "value" in entry:
            suggestion["value"] = entry.get("value")
        if "child_count" in entry:
            suggestion["child_count"] = entry.get("child_count")
        if kind == "bundle" and "value" in entry:
            value = entry.get("value")
            key = tuple(
                sort_once(
                    (str(item) for item in cast(list[object], value)),
                    source="dataflow_structure_reuse.compute_structure_reuse.bundle_name_key",
                )
            )
            name_candidates = bundle_name_map.get(key)
            if name_candidates:
                sorted_names = sort_once(
                    name_candidates,
                    source="dataflow_structure_reuse.compute_structure_reuse.bundle_name_candidates",
                )
                if len(sorted_names) == 1:
                    suggestion["suggested_name"] = sorted_names[0]
                    suggestion["name_source"] = "declared_bundle"
                else:
                    suggestion["name_candidates"] = sorted_names
            else:
                warnings.append(f"Missing declared bundle name for {list(key)}")
        suggestion["witness_obligations"] = [
            {
                "kind": "reuse_suggestion_site",
                "required": True,
                "witness_ref": str(location),
            }
            for location in sequence_or_none(suggestion.get("locations")) or ()
            if type(location) is str
        ]
        suggestion["witness_obligations"].append(
            {
                "kind": "aspf_structure_class_equivalence",
                "required": True,
                "witness_ref": f"aspf:{hash_value}",
                "aspf_structure_class": suggestion.get("aspf_structure_class"),
                "canonical_identity_contract": suggestion.get(
                    "canonical_identity_contract"
                ),
            }
        )
        suggestion["witness_obligations"].append(
            {
                "kind": "aspf_structure_class_coherence",
                "required": True,
                "witness_ref": f"aspf:coherence:{hash_value}",
                "coherence_ref": f"aspf:{hash_value}",
            }
        )
        suggestion["aspf_witness_requirements"] = {
            "equivalence": {
                "kind": "aspf_structure_class_equivalence",
                "witness_ref": f"aspf:{hash_value}",
            },
            "coherence": {
                "kind": "aspf_structure_class_coherence",
                "witness_ref": f"aspf:coherence:{hash_value}",
                "coherence_ref": f"aspf:{hash_value}",
            },
        }
        suggestion["rewrite_plan_artifact"] = _build_suggested_plan_artifact(
            suggestion=suggestion
        )
        suggested.append(suggestion)

    replacement_map = _build_reuse_replacement_map(suggested)
    reuse_payload: JSONObject = {
        "format_version": 1,
        "min_count": min_count,
        "reused": reused,
        "suggested_lemmas": suggested,
        "heuristic_structural_repetition_candidates": [
            {
                "hash": entry.get("hash"),
                "kind": entry.get("kind"),
                "count": entry.get("count"),
                "source": "heuristic_structural_repetition",
            }
            for entry in reused
            if entry.get("kind") in {"bundle", "function"}
        ],
        "witness_validated_isomorphy_candidates": [
            {
                "hash": suggestion.get("hash"),
                "kind": suggestion.get("kind"),
                "suggested_name": suggestion.get("suggested_name"),
                "source": "aspf_witness_requirements",
                "aspf_witness_requirements": suggestion.get("aspf_witness_requirements"),
            }
            for suggestion in suggested
            if mapping_or_none(suggestion.get("aspf_witness_requirements")) is not None
        ],
        "replacement_map": replacement_map,
        "warnings": warnings,
    }
    _copy_forest_signature_metadata(reuse_payload, snapshot)
    return reuse_payload


def _build_reuse_replacement_map(
    suggested: list[object],
) -> dict[str, list[dict[str, object]]]:
    replacement_map: dict[str, list[dict[str, object]]] = {}
    for suggestion_raw in suggested:
        if type(suggestion_raw) is not dict:
            continue
        suggestion = suggestion_raw
        locations = suggestion.get("locations") or []
        if type(locations) is not list:
            continue
        for location in locations:
            if type(location) is not str:
                continue
            replacement_map.setdefault(location, []).append(
                {
                    "kind": suggestion.get("kind"),
                    "hash": suggestion.get("hash"),
                    "suggested_name": suggestion.get("suggested_name"),
                }
            )
    return replacement_map


def project_report_sections(
    groups_by_path,
    report,
    *,
    max_phase=None,
    include_previews: bool = False,
    preview_only: bool = False,
):
    check_deadline()
    extracted: dict[str, list[str]] = {}
    if not preview_only:
        rendered, _ = render_report(
            groups_by_path,
            max_components=10,
            report=report,
        )
        extracted = extract_report_sections(rendered)
    max_rank = None
    if max_phase is not None:
        max_rank = report_projection_phase_rank(max_phase)

    selected: dict[str, list[str]] = {}
    eligible_specs = tuple(
        spec
        for spec in report_projection_specs()
        if max_rank is None or report_projection_phase_rank(spec.phase) <= max_rank
    )
    for spec in eligible_specs:
        check_deadline()
        lines = extracted.get(spec.section_id, [])
        if not lines and include_previews and spec.has_preview:
            lines = preview_section_lines(
                spec.section_id,
                report=report,
                groups_by_path=groups_by_path,
            )
        if lines:
            selected[spec.section_id] = lines
    return selected


def _bundle_name_registry(root: Path) -> dict[tuple[str, ...], set[str]]:
    check_deadline()
    _forbid_adhoc_bundle_discovery("_bundle_name_registry")
    file_paths = sort_once(
        root.rglob("*.py"),
        source="dataflow_structure_reuse._bundle_name_registry.file_paths",
        key=lambda path: str(path),
    )
    parse_failure_witnesses: list[JSONObject] = []
    config_bundles_by_path = _collect_config_bundles(
        list(file_paths),
        parse_failure_witnesses=parse_failure_witnesses,
    )
    dataclass_registry = _collect_dataclass_registry(
        list(file_paths),
        project_root=root,
        parse_failure_witnesses=parse_failure_witnesses,
    )
    name_map: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for bundles in config_bundles_by_path.values():
        check_deadline()
        for name, fields in bundles.items():
            check_deadline()
            key = tuple(
                sort_once(
                    fields,
                    source="dataflow_structure_reuse._bundle_name_registry.config",
                )
            )
            name_map[key].add(str(name))
    for qual_name, fields in dataclass_registry.items():
        check_deadline()
        key = tuple(
            sort_once(
                fields,
                source="dataflow_structure_reuse._bundle_name_registry.dataclass",
            )
        )
        name_map[key].add(str(qual_name).split(".")[-1])
    return name_map


def _copy_forest_signature_metadata(
    payload: JSONObject,
    snapshot: JSONObject,
    *,
    prefix: str = "",
) -> None:
    signature = snapshot.get("forest_signature")
    if signature is not None:
        payload[f"{prefix}forest_signature"] = signature
    partial = snapshot.get("forest_signature_partial")
    if partial is not None:
        payload[f"{prefix}forest_signature_partial"] = partial
    basis = snapshot.get("forest_signature_basis")
    if basis is not None:
        payload[f"{prefix}forest_signature_basis"] = basis
    if signature is None:
        payload[f"{prefix}forest_signature_partial"] = True
        if basis is None:
            payload[f"{prefix}forest_signature_basis"] = "missing"
