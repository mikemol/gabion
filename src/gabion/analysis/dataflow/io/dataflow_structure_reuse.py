# gabion:ambiguity_boundary_module
# gabion:boundary_normalization_module
# gabion:grade_boundary kind=semantic_carrier_adapter name=dataflow_structure_reuse
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from functools import singledispatch
from itertools import tee
from pathlib import Path
from typing import Callable, cast

from gabion.analysis.dataflow.engine.dataflow_analysis_index import (
    _build_analysis_collection_resume_payload,
)
from gabion.analysis.dataflow.engine.dataflow_contracts import ReportCarrier
from gabion.analysis.dataflow.io.dataflow_parse_helpers import (
    forbid_adhoc_bundle_discovery,
)
from gabion.analysis.dataflow.io.dataflow_projection_helpers import (
    report_projection_phase_rank, report_projection_specs)
from gabion.analysis.dataflow.io.dataflow_projection_preview_bridge import preview_section_lines
from gabion.analysis.dataflow.io.dataflow_reporting import render_report
from gabion.analysis.dataflow.io.dataflow_snapshot_io import extract_report_sections
from gabion.analysis.dataflow.engine.dataflow_post_phase_analyses import (
    _collect_config_bundles,
    _collect_dataclass_registry,
)
from gabion.analysis.foundation.json_types import JSONObject, JSONValue
from gabion.analysis.dataflow.io.forest_signature_metadata import apply_forest_signature_metadata
from gabion.analysis.foundation.resume_codec import mapping_default_empty, mapping_optional, sequence_optional
from gabion.analysis.core.structure_reuse_classes import build_structure_class, structure_class_payload
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.invariants import grade_boundary, never, todo
from gabion.order_contract import sort_once
from gabion.server_core.command_contract import ReportSectionState

_NONE_TYPE = type(None)

_PR412_REPRESENTATIVE_ONLY_STRUCTURE_REUSE_IDENTITY = todo(
    reasoning={
        "summary": "PR-412 canonical identity contract adoption still partial in representative-only structure reuse payloads",
        "control": "pr412.identity_payload.representative_only_structure_reuse",
        "blocking_dependencies": (
            "replace_representative_only_structure_reuse_identity_with_typed_canonical_contract",
        ),
    },
    owner="gabion.analysis.dataflow.io",
    links=[{"kind": "object_id", "value": "pr:412"}],
)


@singledispatch
def _is_json_object(value: JSONValue) -> bool:
    never("unregistered runtime type", value_type=type(value).__name__)


@_is_json_object.register(dict)
def _sd_reg_1(value: JSONObject) -> bool:
    _ = value
    return True


def _is_not_json_object(value: JSONValue) -> bool:
    _ = value
    return False


for _runtime_type in (list, tuple, set, str, int, float, bool, _NONE_TYPE):
    _is_json_object.register(_runtime_type)(_is_not_json_object)


@singledispatch
def _json_object_value(value: JSONValue) -> JSONObject:
    never("unregistered runtime type", value_type=type(value).__name__)


@_json_object_value.register(dict)
def _sd_reg_2(value: JSONObject) -> JSONObject:
    return value


@singledispatch
def _is_json_list(value: JSONValue) -> bool:
    never("unregistered runtime type", value_type=type(value).__name__)


@_is_json_list.register(list)
def _sd_reg_3(value: list[JSONValue]) -> bool:
    _ = value
    return True


def _is_not_json_list(value: JSONValue) -> bool:
    _ = value
    return False


for _runtime_type in (tuple, set, dict, str, int, float, bool, _NONE_TYPE):
    _is_json_list.register(_runtime_type)(_is_not_json_list)


@singledispatch
def _json_list_value(value: JSONValue) -> list[JSONValue]:
    never("unregistered runtime type", value_type=type(value).__name__)


@_json_list_value.register(list)
def _sd_reg_4(value: list[JSONValue]) -> list[JSONValue]:
    return value


@singledispatch
def _is_string_value(value: JSONValue) -> bool:
    never("unregistered runtime type", value_type=type(value).__name__)


@_is_string_value.register(str)
def _sd_reg_5(value: str) -> bool:
    _ = value
    return True


for _runtime_type in (list, tuple, set, dict, int, float, bool, _NONE_TYPE):
    _is_string_value.register(_runtime_type)(_is_not_json_list)


@singledispatch
def _string_value(value: JSONValue) -> str:
    never("unregistered runtime type", value_type=type(value).__name__)


@_string_value.register(str)
def _sd_reg_6(value: str) -> str:
    return value


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


@grade_boundary(
    kind="semantic_carrier_adapter",
    name="dataflow_structure_reuse.compute_structure_reuse",
)
def compute_structure_reuse(
    snapshot: JSONObject,
    *,
    project_root: Path,
    min_count: int = 2,
    hash_fn=None,
) -> JSONObject:
    check_deadline()
    if min_count < 2:
        min_count = 2
    files = snapshot.get("files") or []
    bundle_name_map: dict[tuple[str, ...], set[str]] = {}
    if project_root.exists():
        bundle_name_map = _bundle_name_registry(project_root)
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
        if _is_json_object(file_entry):
            file_entry_obj = _json_object_value(file_entry)
            file_path = file_entry_obj.get("path")
            if _is_string_value(file_path):
                file_path_text = _string_value(file_path)
                function_hashes: list[str] = []
                functions = file_entry_obj.get("functions") or []
                function_entries = (
                    _json_list_value(functions) if _is_json_list(functions) else []
                )
                for fn_entry in function_entries:
                    check_deadline()
                    if _is_json_object(fn_entry):
                        fn_entry_obj = _json_object_value(fn_entry)
                        fn_name = fn_entry_obj.get("name")
                        if _is_string_value(fn_name):
                            fn_name_text = _string_value(fn_name)
                            bundle_hashes: list[str] = []
                            bundles = fn_entry_obj.get("bundles") or []
                            bundle_entries = (
                                _json_list_value(bundles) if _is_json_list(bundles) else []
                            )
                            for bundle in bundle_entries:
                                check_deadline()
                                if _is_json_list(bundle):
                                    bundle_items = _json_list_value(bundle)
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
                                        location=f"{file_path_text}::{fn_name_text}::bundle:{','.join(normalized)}",
                                        value=list(normalized),
                                    )
                            fn_hash = hasher("function", None, bundle_hashes)
                            function_hashes.append(fn_hash)
                            _record(
                                node_hash=fn_hash,
                                kind="function",
                                location=f"{file_path_text}::{fn_name_text}",
                                child_count=len(bundle_hashes),
                            )
                file_hash = hasher("file", None, function_hashes)
                file_hashes.append(file_hash)
                _record(node_hash=file_hash, kind="file", location=f"{file_path_text}")

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
        locations = sequence_optional(suggestion.get("locations")) or ()
        sorted_locations = sort_once(
            [
                _string_value(location)
                for location in locations
                if _is_string_value(location)
            ],
            source="dataflow_structure_reuse.compute_structure_reuse.suggested_locations",
        )
        primary_location = sorted_locations[0] if sorted_locations else ""
        site = _reuse_site_from_location(
            location=primary_location,
            fallback_value=suggestion.get("value"),
        )
        witness_obligations = list(
            sequence_optional(suggestion.get("witness_obligations")) or []
        )
        aspf_witness_requirements = mapping_default_empty(
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
        if kind in {"bundle", "function"}:
            count = int(entry.get("count", 0))
            hash_value = entry.get("hash")
            if _is_string_value(hash_value):
                hash_text = _string_value(hash_value)
                if hash_text:
                    suggestion: JSONObject = {
                        "hash": hash_text,
                        "kind": kind,
                        "count": count,
                        "suggested_name": f"_gabion_{kind}_lemma_{hash_text[:8]}",
                        "locations": entry.get("locations", []),
                        "aspf_structure_class": entry.get("aspf_structure_class"),
                        "canonical_identity_contract": {
                            "identity_kind": "canonical_aspf_structure_class_equivalence",
                            "representative": hash_text,
                        },
                    }
                    if "value" in entry:
                        suggestion["value"] = entry.get("value")
                    if "child_count" in entry:
                        suggestion["child_count"] = entry.get("child_count")
                    if kind == "bundle" and "value" in entry:
                        value = entry.get("value")
                        if _is_json_list(value):
                            key = tuple(
                                sort_once(
                                    (
                                        str(item)
                                        for item in _json_list_value(value)
                                    ),
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
                                warnings.append(
                                    f"Missing declared bundle name for {list(key)}"
                                )
                    suggestion["witness_obligations"] = [
                        {
                            "kind": "reuse_suggestion_site",
                            "required": True,
                            "witness_ref": str(location),
                        }
                        for location in sequence_optional(suggestion.get("locations")) or ()
                        if _is_string_value(location)
                    ]
                    suggestion["witness_obligations"].append(
                        {
                            "kind": "aspf_structure_class_equivalence",
                            "required": True,
                            "witness_ref": f"aspf:{hash_text}",
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
                            "witness_ref": f"aspf:coherence:{hash_text}",
                            "coherence_ref": f"aspf:{hash_text}",
                        }
                    )
                    suggestion["aspf_witness_requirements"] = {
                        "equivalence": {
                            "kind": "aspf_structure_class_equivalence",
                            "witness_ref": f"aspf:{hash_text}",
                        },
                        "coherence": {
                            "kind": "aspf_structure_class_coherence",
                            "witness_ref": f"aspf:coherence:{hash_text}",
                            "coherence_ref": f"aspf:{hash_text}",
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
            if mapping_optional(suggestion.get("aspf_witness_requirements")) is not None
        ],
        "replacement_map": replacement_map,
        "warnings": warnings,
    }
    apply_forest_signature_metadata(reuse_payload, snapshot)
    return reuse_payload


def _build_reuse_replacement_map(
    suggested: list[object],
) -> dict[str, list[dict[str, object]]]:
    replacement_map: dict[str, list[dict[str, object]]] = {}
    for suggestion_raw in suggested:
        if _is_json_object(suggestion_raw):
            suggestion = _json_object_value(suggestion_raw)
            locations = suggestion.get("locations") or []
            location_items = (
                _json_list_value(locations) if _is_json_list(locations) else []
            )
            for location in location_items:
                if _is_string_value(location):
                    location_text = _string_value(location)
                    replacement_map.setdefault(location_text, []).append(
                        {
                            "kind": suggestion.get("kind"),
                            "hash": suggestion.get("hash"),
                            "suggested_name": suggestion.get("suggested_name"),
                        }
                    )
    return replacement_map


def _tee_report_section_state_stream(
    entries: Iterator[ReportSectionState],
) -> Callable[[], Iterator[ReportSectionState]]:
    source = entries

    def iter_entries() -> Iterator[ReportSectionState]:
        nonlocal source
        source, clone = tee(source)
        return clone

    return iter_entries


def project_report_sections(
    groups_by_path,
    report,
    *,
    project_root: Path,
    max_phase=None,
    include_previews: bool = False,
    preview_only: bool = False,
) -> Callable[[], Iterator[ReportSectionState]]:
    check_deadline()
    extracted: dict[str, list[str]] = {}
    if not preview_only:
        rendered, _ = render_report(
            groups_by_path,
            max_components=10,
            project_root=project_root,
            report=report,
        )
        extracted = extract_report_sections(rendered)
    max_rank = None
    if max_phase is not None:
        max_rank = report_projection_phase_rank(max_phase)

    eligible_specs = tuple(
        spec
        for spec in report_projection_specs()
        if max_rank is None or report_projection_phase_rank(spec.phase) <= max_rank
    )

    def iter_selected_sections() -> Iterator[ReportSectionState]:
        for spec in eligible_specs:
            check_deadline()
            lines = extracted.get(spec.section_id, []) or (
                preview_section_lines(
                    spec.section_id,
                    report=report,
                    groups_by_path=groups_by_path,
                    project_root=project_root,
                )
                if include_previews and spec.has_preview
                else []
            )
            if lines:
                yield ReportSectionState(
                    section_id=spec.section_id,
                    _line_iterator_factory=(
                        lambda section_lines=lines: iter(section_lines)
                    ),
                )

    return _tee_report_section_state_stream(iter_selected_sections())


def _bundle_name_registry(root: Path) -> dict[tuple[str, ...], set[str]]:
    check_deadline()
    forbid_adhoc_bundle_discovery("_bundle_name_registry")
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
