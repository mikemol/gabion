# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from gabion.analysis.aspf.aspf import Forest
from gabion.analysis.foundation.baseline_io import load_json
from gabion.analysis.dataflow.engine.dataflow_contracts import InvariantProposition
from gabion.analysis.dataflow.io.dataflow_parse_helpers import _forbid_adhoc_bundle_discovery
from gabion.analysis.dataflow.io.forest_signature_metadata import (
    apply_forest_signature_metadata,
)
from gabion.analysis.dataflow.io.dataflow_snapshot_contracts import (
    DecisionSnapshotSurfaces, StructureSnapshotDiffRequest)
from gabion.analysis.projection.decision_flow import (
    build_decision_tables, detect_repeated_guard_bundles, enforce_decision_protocol_contracts)
from gabion.analysis.core.deprecated_substrate import detect_report_section_extinction
from gabion.analysis.core.forest_signature import (
    build_forest_signature, build_forest_signature_from_groups)
from gabion.analysis.core.forest_spec import default_forest_spec, forest_spec_metadata
from gabion.analysis.foundation.json_types import JSONObject
from gabion.analysis.projection.pattern_schema_projection import (
    pattern_schema_matches, pattern_schema_snapshot_entries)
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.invariants import never
from gabion.order_contract import sort_once

_REPORT_SECTION_MARKER_PREFIX = "<!-- report-section:"
_REPORT_SECTION_MARKER_SUFFIX = "-->"


def report_section_marker(section_id: str) -> str:
    return f"{_REPORT_SECTION_MARKER_PREFIX}{section_id}{_REPORT_SECTION_MARKER_SUFFIX}"


def parse_report_section_marker(line: str):
    text = line.strip()
    if not text.startswith(_REPORT_SECTION_MARKER_PREFIX):
        return None
    if not text.endswith(_REPORT_SECTION_MARKER_SUFFIX):
        return None
    section_id = text[
        len(_REPORT_SECTION_MARKER_PREFIX) : -len(_REPORT_SECTION_MARKER_SUFFIX)
    ].strip()
    if not section_id:
        return None
    return section_id


def extract_report_sections(markdown: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    active_section_id = ""
    has_active_section = False
    for raw_line in markdown.splitlines():
        check_deadline()
        section_id = parse_report_section_marker(raw_line)
        if section_id is not None:
            active_section_id = section_id
            has_active_section = True
            sections.setdefault(section_id, [])
        elif has_active_section:
            sections[active_section_id].append(raw_line)
    return sections


def detect_report_section_extinctions(
    *,
    previous_markdown: str,
    current_markdown: str,
) -> tuple[str, ...]:
    check_deadline()
    previous_sections = tuple(extract_report_sections(previous_markdown))
    current_sections = tuple(extract_report_sections(current_markdown))
    return detect_report_section_extinction(
        previous_sections=previous_sections,
        current_sections=current_sections,
    )


def _infer_root(groups_by_path: dict[Path, dict[str, list[set[str]]]]) -> Path:
    if groups_by_path:
        common = os.path.commonpath([str(path) for path in groups_by_path])
        return Path(common)
    return Path(".")


def _normalize_snapshot_path(path: Path, root: object) -> str:
    if root is not None:
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return str(path)


def load_structure_snapshot(path: Path) -> JSONObject:
    try:
        data = load_json(path)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid snapshot JSON: {path}") from exc
    except ValueError as exc:
        raise ValueError(f"Snapshot must be a JSON object: {path}") from exc
    return {str(key): data[key] for key in data}


def compute_structure_metrics(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    *,
    forest: Forest,
) -> JSONObject:
    check_deadline()
    file_count = len(groups_by_path)
    function_count = sum(len(groups) for groups in groups_by_path.values())
    bundle_sizes: list[int] = []
    for groups in groups_by_path.values():
        check_deadline()
        for bundles in groups.values():
            check_deadline()
            for bundle in bundles:
                check_deadline()
                bundle_sizes.append(len(bundle))

    bundle_count = len(bundle_sizes)
    mean_bundle_size = (sum(bundle_sizes) / bundle_count) if bundle_count else 0.0
    max_bundle_size = max(bundle_sizes) if bundle_sizes else 0
    size_histogram: dict[int, int] = defaultdict(int)
    for size in bundle_sizes:
        check_deadline()
        size_histogram[size] += 1

    metrics: JSONObject = {
        "files": file_count,
        "functions": function_count,
        "bundles": bundle_count,
        "mean_bundle_size": mean_bundle_size,
        "max_bundle_size": max_bundle_size,
        "bundle_size_histogram": {
            str(size): count
            for size, count in sort_once(
                size_histogram.items(),
                source="src/gabion/analysis/dataflow_snapshot_io.py:compute_structure_metrics",
            )
        },
    }
    metrics["forest_signature"] = build_forest_signature(forest)
    return metrics


def render_structure_snapshot(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    *,
    project_root=None,
    forest: Forest,
    forest_spec=None,
    invariant_propositions: Sequence[InvariantProposition] = (),
) -> JSONObject:
    check_deadline()
    root = project_root or _infer_root(groups_by_path)
    invariant_map: dict[tuple[str, str], list[InvariantProposition]] = {}
    for proposition in invariant_propositions:
        check_deadline()
        if not proposition.scope or ":" not in proposition.scope:
            continue
        scope_path, fn_name = proposition.scope.rsplit(":", 1)
        invariant_map.setdefault((scope_path, fn_name), []).append(proposition)

    files: list[JSONObject] = []
    for path in sort_once(
        groups_by_path,
        key=lambda candidate: _normalize_snapshot_path(candidate, root),
        source="src/gabion/analysis/dataflow_snapshot_io.py:render_structure_snapshot.paths",
    ):
        check_deadline()
        groups = groups_by_path[path]
        path_key = _normalize_snapshot_path(path, root)
        functions: list[JSONObject] = []
        for fn_name in sort_once(
            groups,
            source="src/gabion/analysis/dataflow_snapshot_io.py:render_structure_snapshot.functions",
        ):
            check_deadline()
            bundles = groups[fn_name]
            normalized = [
                sort_once(
                    bundle,
                    source="src/gabion/analysis/dataflow_snapshot_io.py:render_structure_snapshot.bundle",
                )
                for bundle in bundles
            ]
            normalized = sort_once(
                normalized,
                key=lambda bundle: (len(bundle), bundle),
                source="src/gabion/analysis/dataflow_snapshot_io.py:render_structure_snapshot.normalized",
            )
            entry: JSONObject = {"name": fn_name, "bundles": normalized}
            invariants = invariant_map.get((path_key, fn_name))
            if invariants:
                entry["invariants"] = [
                    proposition.as_dict()
                    for proposition in sort_once(
                        invariants,
                        key=lambda proposition: (
                            proposition.form,
                            proposition.terms,
                            proposition.source or "",
                            proposition.scope or "",
                        ),
                        source="src/gabion/analysis/dataflow_snapshot_io.py:render_structure_snapshot.invariants",
                    )
                ]
            functions.append(entry)
        files.append({"path": path_key, "functions": functions})

    snapshot: JSONObject = {
        "format_version": 1,
        "root": str(root) if root is not None else None,
        "files": files,
    }
    spec = forest_spec or default_forest_spec(include_bundle_forest=True)
    snapshot.update(forest_spec_metadata(spec))
    snapshot["forest_signature"] = build_forest_signature(forest)
    return snapshot


def render_decision_snapshot(
    *,
    surfaces: DecisionSnapshotSurfaces,
    project_root=None,
    forest: Forest,
    forest_spec=None,
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    pattern_schema_instances=None,
) -> JSONObject:
    if type(forest) is not Forest:
        never("decision snapshot requires forest carrier")

    instances = pattern_schema_instances
    if instances is None:
        instances = pattern_schema_matches(
            groups_by_path=groups_by_path,
            include_execution=True,
        )
    schema_instances, schema_residue = pattern_schema_snapshot_entries(instances)

    decision_tables = build_decision_tables(
        decision_surfaces=surfaces.decision_surfaces,
        value_decision_surfaces=surfaces.value_decision_surfaces,
    )
    decision_bundles = detect_repeated_guard_bundles(decision_tables)
    decision_protocol_violations = enforce_decision_protocol_contracts(
        decision_tables=decision_tables,
        decision_bundles=decision_bundles,
    )

    snapshot: JSONObject = {
        "format_version": 1,
        "root": str(project_root) if project_root is not None else None,
        "decision_surfaces": sort_once(
            surfaces.decision_surfaces,
            source="src/gabion/analysis/dataflow_snapshot_io.py:render_decision_snapshot.decision",
        ),
        "value_decision_surfaces": sort_once(
            surfaces.value_decision_surfaces,
            source="src/gabion/analysis/dataflow_snapshot_io.py:render_decision_snapshot.value_decision",
        ),
        "pattern_schema_instances": schema_instances,
        "pattern_schema_residue": schema_residue,
        "decision_tables": decision_tables,
        "decision_bundles": decision_bundles,
        "decision_protocol_violations": decision_protocol_violations,
        "summary": {
            "decision_surfaces": len(surfaces.decision_surfaces),
            "value_decision_surfaces": len(surfaces.value_decision_surfaces),
            "pattern_schema_instances": len(schema_instances),
            "pattern_schema_residue": len(schema_residue),
            "decision_tables": len(decision_tables),
            "decision_bundles": len(decision_bundles),
            "decision_protocol_violations": len(decision_protocol_violations),
        },
    }
    snapshot["forest"] = forest.to_json()
    snapshot["forest_signature"] = build_forest_signature(forest)
    spec = forest_spec or default_forest_spec(
        include_bundle_forest=True,
        include_decision_surfaces=True,
        include_value_decision_surfaces=True,
    )
    snapshot.update(forest_spec_metadata(spec))
    return snapshot


def load_decision_snapshot(path: Path) -> JSONObject:
    try:
        data = load_json(path)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid decision snapshot JSON: {path}") from exc
    except ValueError as exc:
        raise ValueError(f"Decision snapshot must be a JSON object: {path}") from exc
    return {str(key): data[key] for key in data}


def diff_decision_snapshots(
    baseline_snapshot: JSONObject,
    current_snapshot: JSONObject,
) -> JSONObject:
    base_decisions = set(baseline_snapshot.get("decision_surfaces") or [])
    curr_decisions = set(current_snapshot.get("decision_surfaces") or [])
    base_value = set(baseline_snapshot.get("value_decision_surfaces") or [])
    curr_value = set(current_snapshot.get("value_decision_surfaces") or [])
    diff: JSONObject = {
        "format_version": 1,
        "baseline_root": baseline_snapshot.get("root"),
        "current_root": current_snapshot.get("root"),
        "decision_surfaces": {
            "added": sort_once(
                curr_decisions - base_decisions,
                source="src/gabion/analysis/dataflow_snapshot_io.py:diff_decision_snapshots.added",
            ),
            "removed": sort_once(
                base_decisions - curr_decisions,
                source="src/gabion/analysis/dataflow_snapshot_io.py:diff_decision_snapshots.removed",
            ),
        },
        "value_decision_surfaces": {
            "added": sort_once(
                curr_value - base_value,
                source="src/gabion/analysis/dataflow_snapshot_io.py:diff_decision_snapshots.value_added",
            ),
            "removed": sort_once(
                base_value - curr_value,
                source="src/gabion/analysis/dataflow_snapshot_io.py:diff_decision_snapshots.value_removed",
            ),
        },
    }
    apply_forest_signature_metadata(diff, baseline_snapshot, prefix="baseline_")
    apply_forest_signature_metadata(diff, current_snapshot, prefix="current_")
    return diff


def _bundle_counts_from_snapshot(snapshot: JSONObject) -> dict[tuple[str, ...], int]:
    check_deadline()
    _forbid_adhoc_bundle_discovery("_bundle_counts_from_snapshot")
    counts: dict[tuple[str, ...], int] = defaultdict(int)
    files = snapshot.get("files") or []
    for file_entry in files:
        check_deadline()
        if type(file_entry) is not dict:
            continue
        file_entry_obj = cast(JSONObject, file_entry)
        functions = file_entry_obj.get("functions") or []
        for fn_entry in functions:
            check_deadline()
            if type(fn_entry) is not dict:
                continue
            fn_entry_obj = cast(JSONObject, fn_entry)
            bundles = fn_entry_obj.get("bundles") or []
            for bundle in bundles:
                check_deadline()
                if type(bundle) is not list:
                    continue
                counts[tuple(cast(list[object], bundle))] += 1
    return counts


def diff_structure_snapshots(
    baseline_snapshot: JSONObject,
    current_snapshot: JSONObject,
) -> JSONObject:
    check_deadline()
    baseline_counts = _bundle_counts_from_snapshot(baseline_snapshot)
    current_counts = _bundle_counts_from_snapshot(current_snapshot)
    all_bundles = sort_once(
        set(baseline_counts) | set(current_counts),
        key=lambda bundle: (len(bundle), list(bundle)),
        source="src/gabion/analysis/dataflow_snapshot_io.py:diff_structure_snapshots.all_bundles",
    )

    added: list[JSONObject] = []
    removed: list[JSONObject] = []
    changed: list[JSONObject] = []
    for bundle in all_bundles:
        check_deadline()
        before = baseline_counts.get(bundle, 0)
        after = current_counts.get(bundle, 0)
        entry: JSONObject = {
            "bundle": list(bundle),
            "before": before,
            "after": after,
            "delta": after - before,
        }
        if before == 0:
            added.append(entry)
        elif after == 0:
            removed.append(entry)
        elif before != after:
            changed.append(entry)

    diff: JSONObject = {
        "format_version": 1,
        "baseline_root": baseline_snapshot.get("root"),
        "current_root": current_snapshot.get("root"),
        "added": added,
        "removed": removed,
        "changed": changed,
        "summary": {
            "added": len(added),
            "removed": len(removed),
            "changed": len(changed),
            "baseline_total": sum(baseline_counts.values()),
            "current_total": sum(current_counts.values()),
        },
    }
    apply_forest_signature_metadata(diff, baseline_snapshot, prefix="baseline_")
    apply_forest_signature_metadata(diff, current_snapshot, prefix="current_")
    return diff


def diff_structure_snapshot_files(
    request: StructureSnapshotDiffRequest,
) -> JSONObject:
    baseline = load_structure_snapshot(request.baseline_path)
    current = load_structure_snapshot(request.current_path)
    return diff_structure_snapshots(baseline, current)


__all__ = [
    "compute_structure_metrics",
    "detect_report_section_extinctions",
    "diff_decision_snapshots",
    "diff_structure_snapshot_files",
    "diff_structure_snapshots",
    "extract_report_sections",
    "load_decision_snapshot",
    "load_structure_snapshot",
    "parse_report_section_marker",
    "render_decision_snapshot",
    "render_structure_snapshot",
    "report_section_marker",
]
