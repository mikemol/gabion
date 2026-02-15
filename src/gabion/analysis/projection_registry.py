from __future__ import annotations

import json
import os
from typing import Iterable, Mapping

from gabion.analysis.projection_normalize import (
    normalize_spec,
    spec_canonical_json,
    spec_hash,
)
from gabion.analysis.projection_spec import ProjectionOp, ProjectionSpec
from gabion.analysis.aspf import Forest
from gabion.analysis.timeout_context import (
    Deadline,
    deadline_clock_scope,
    deadline_scope,
    forest_scope,
)
from gabion.deadline_clock import GasMeter
from gabion.invariants import never
from gabion.json_types import JSONValue


_PROJECTION_REGISTRY_GAS_LIMIT_DEFAULT = 1_000_000


def _projection_registry_gas_limit() -> int:
    raw = os.getenv("GABION_PROJECTION_REGISTRY_GAS_LIMIT", "").strip()
    if not raw:
        return _PROJECTION_REGISTRY_GAS_LIMIT_DEFAULT
    try:
        value = int(raw)
    except ValueError:
        never(
            "invalid projection registry gas limit",
            value=raw,
        )
    if value <= 0:
        never(
            "invalid projection registry gas limit",
            value=value,
        )
    return value


NEVER_INVARIANTS_SPEC = ProjectionSpec(
    spec_version=1,
    name="never_invariants_section",
    domain="never_invariants",
    pipeline=(
        ProjectionOp("select", {"predicates": ["never_status_allowed"]}),
        ProjectionOp(
            "project",
            {
                "fields": [
                    "status",
                    "status_rank",
                    "site_path",
                    "site_function",
                    "span_line",
                    "span_col",
                    "span_end_line",
                    "span_end_col",
                    "never_id",
                    "reason",
                    "witness_ref",
                    "environment_ref",
                    "undecidable_reason",
                ]
            },
        ),
        ProjectionOp(
            "sort",
            {
                "by": [
                    "status_rank",
                    "site_path",
                    "site_function",
                    "span_line",
                    "span_col",
                    "never_id",
                ]
            },
        ),
    ),
    params={
        "ordered_statuses": ["VIOLATION", "OBLIGATION", "PROVEN_UNREACHABLE"],
        "max_entries": 50,
        "include_proven_unreachable": True,
    },
)


TEST_OBSOLESCENCE_SUMMARY_SPEC = ProjectionSpec(
    spec_version=1,
    name="test_obsolescence_summary",
    domain="test_obsolescence",
    pipeline=(
        ProjectionOp("project", {"fields": ["class", "class_rank"]}),
        ProjectionOp("count_by", {"fields": ["class_rank", "class"]}),
        ProjectionOp("sort", {"by": ["class_rank", "class"]}),
        ProjectionOp("project", {"fields": ["class", "count"]}),
    ),
)


TEST_OBSOLESCENCE_BASELINE_SPEC = ProjectionSpec(
    spec_version=1,
    name="test_obsolescence_baseline",
    domain="test_obsolescence_baseline",
)


TEST_OBSOLESCENCE_STATE_SPEC = ProjectionSpec(
    spec_version=1,
    name="test_obsolescence_state",
    domain="test_obsolescence_state",
)


TEST_OBSOLESCENCE_DELTA_SPEC = ProjectionSpec(
    spec_version=1,
    name="test_obsolescence_delta",
    domain="test_obsolescence_delta",
)


AMBIGUITY_SUMMARY_SPEC = ProjectionSpec(
    spec_version=1,
    name="ambiguity_summary",
    domain="ambiguity_witnesses",
    pipeline=(
        ProjectionOp(
            "project",
            {
                "fields": [
                    "kind",
                    "site_path",
                    "site_function",
                    "span_line",
                    "span_col",
                    "span_end_line",
                    "span_end_col",
                    "candidate_count",
                ]
            },
        ),
        ProjectionOp(
            "sort",
            {
                "by": [
                    "kind",
                    "site_path",
                    "site_function",
                    "span_line",
                    "span_col",
                    "candidate_count",
                ]
            },
        ),
    ),
    params={
        "max_entries": 20,
    },
)


AMBIGUITY_SUITE_AGG_SPEC = ProjectionSpec(
    spec_version=1,
    name="ambiguity_suite_agg",
    domain="ambiguity_suite",
    pipeline=(
        ProjectionOp(
            "project",
            {
                "fields": [
                    "suite_path",
                    "suite_qual",
                    "kind",
                    "phase",
                ]
            },
        ),
        ProjectionOp(
            "count_by",
            {
                "fields": [
                    "suite_path",
                    "suite_qual",
                    "kind",
                    "phase",
                ]
            },
        ),
        ProjectionOp(
            "sort",
            {
                "by": [
                    "suite_path",
                    "suite_qual",
                    "kind",
                    "phase",
                ]
            },
        ),
    ),
)


AMBIGUITY_VIRTUAL_SET_SPEC = ProjectionSpec(
    spec_version=1,
    name="ambiguity_virtual_set",
    domain="ambiguity_suite",
    pipeline=(
        ProjectionOp(
            "project",
            {
                "fields": [
                    "suite_path",
                    "suite_qual",
                    "suite_kind",
                    "span_line",
                    "span_col",
                    "span_end_line",
                    "span_end_col",
                ]
            },
        ),
        ProjectionOp(
            "count_by",
            {
                "fields": [
                    "suite_path",
                    "suite_qual",
                    "suite_kind",
                    "span_line",
                    "span_col",
                    "span_end_line",
                    "span_end_col",
                ]
            },
        ),
        ProjectionOp("select", {"predicates": ["count_gt_1"]}),
        ProjectionOp(
            "sort",
            {
                "by": [
                    "suite_path",
                    "suite_qual",
                    "span_line",
                    "span_col",
                ]
            },
        ),
    ),
)


DEADLINE_OBLIGATIONS_SUMMARY_SPEC = ProjectionSpec(
    spec_version=1,
    name="deadline_obligations_summary",
    domain="deadline_obligations",
    pipeline=(
        ProjectionOp(
            "project",
            {
                "fields": [
                    "status",
                    "kind",
                    "detail",
                    "site_path",
                    "site_function",
                    "span_line",
                    "span_col",
                    "span_end_line",
                    "span_end_col",
                    "deadline_id",
                ]
            },
        ),
        ProjectionOp(
            "sort",
            {
                "by": [
                    "status",
                    "kind",
                    "site_path",
                    "site_function",
                    "span_line",
                    "span_col",
                    "deadline_id",
                ]
            },
        ),
    ),
    params={
        "max_entries": 20,
    },
)


LINT_FINDINGS_SPEC = ProjectionSpec(
    spec_version=1,
    name="lint_findings",
    domain="lint_findings",
    pipeline=(
        ProjectionOp(
            "project",
            {
                "fields": [
                    "path",
                    "line",
                    "col",
                    "code",
                    "message",
                ]
            },
        ),
        ProjectionOp(
            "sort",
            {
                "by": [
                    "path",
                    "line",
                    "col",
                    "code",
                    "message",
                ]
            },
        ),
    ),
)


REPORT_SECTION_LINES_SPEC = ProjectionSpec(
    spec_version=1,
    name="report_section_lines",
    domain="report_section_lines",
    pipeline=(
        ProjectionOp(
            "project",
            {
                "fields": [
                    "section",
                    "line_index",
                    "text",
                ]
            },
        ),
        ProjectionOp(
            "sort",
            {
                "by": [
                    "section",
                    "line_index",
                ]
            },
        ),
    ),
)


SUITE_ORDER_SPEC = ProjectionSpec(
    spec_version=1,
    name="suite_order",
    domain="suite_order",
    pipeline=(
        ProjectionOp(
            "project",
            {
                "fields": [
                    "suite_path",
                    "suite_qual",
                    "suite_kind",
                    "span_line",
                    "span_col",
                    "span_end_line",
                    "span_end_col",
                    "depth",
                    "complexity",
                    "order_key",
                ]
            },
        ),
        ProjectionOp(
            "sort",
            {
                "by": [
                    "depth",
                    "complexity",
                    "suite_path",
                    "suite_qual",
                    "span_line",
                    "span_col",
                ]
            },
        ),
    ),
)


CALL_CLUSTER_SUMMARY_SPEC = ProjectionSpec(
    spec_version=1,
    name="call_cluster_summary",
    domain="call_clusters",
    pipeline=(
        ProjectionOp("project", {"fields": ["identity", "display", "count"]}),
        ProjectionOp(
            "sort",
            {
                "by": [
                    {"field": "count", "order": "desc"},
                    {"field": "display", "order": "asc"},
                ]
            },
        ),
    ),
)


CALL_CLUSTER_CONSOLIDATION_SPEC = ProjectionSpec(
    spec_version=1,
    name="call_cluster_consolidation",
    domain="call_cluster_consolidation",
    pipeline=(
        ProjectionOp(
            "project",
            {
                "fields": [
                    "cluster_identity",
                    "cluster_display",
                    "cluster_count",
                    "test_id",
                    "file",
                    "line",
                    "replace",
                    "with",
                ]
            },
        ),
        ProjectionOp(
            "sort",
            {
                "by": [
                    {"field": "cluster_count", "order": "desc"},
                    {"field": "cluster_display", "order": "asc"},
                    {"field": "test_id", "order": "asc"},
                ]
            },
        ),
    ),
)


TEST_ANNOTATION_DRIFT_SPEC = ProjectionSpec(
    spec_version=1,
    name="test_annotation_drift",
    domain="test_annotation_drift",
    pipeline=(
        ProjectionOp("project", {"fields": ["status", "test_id", "tag", "reason"]}),
        ProjectionOp(
            "sort",
            {
                "by": [
                    "status",
                    "test_id",
                    "tag",
                ]
            },
        ),
    ),
)


TEST_ANNOTATION_DRIFT_BASELINE_SPEC = ProjectionSpec(
    spec_version=1,
    name="test_annotation_drift_baseline",
    domain="test_annotation_drift_baseline",
)


TEST_ANNOTATION_DRIFT_DELTA_SPEC = ProjectionSpec(
    spec_version=1,
    name="test_annotation_drift_delta",
    domain="test_annotation_drift_delta",
)


AMBIGUITY_BASELINE_SPEC = ProjectionSpec(
    spec_version=1,
    name="ambiguity_baseline",
    domain="ambiguity_baseline",
)


AMBIGUITY_DELTA_SPEC = ProjectionSpec(
    spec_version=1,
    name="ambiguity_delta",
    domain="ambiguity_delta",
)


AMBIGUITY_STATE_SPEC = ProjectionSpec(
    spec_version=1,
    name="ambiguity_state",
    domain="ambiguity_state",
)


WL_REFINEMENT_SPEC = ProjectionSpec(
    spec_version=1,
    name="wl_refinement",
    domain="wl_refinement",
    params={
        "target_kind": "SuiteSite",
        "edge_alt_kinds": ["SuiteContains"],
        "direction": "undirected",
        "seed_fields": ["suite_kind"],
        "steps": 2,
        "stabilize_early": True,
        "emit_steps": "final",
        "label_namespace": "wl",
        "require_injective_on_scope": False,
    },
)


def spec_metadata_lines(spec: ProjectionSpec) -> list[str]:
    spec_json = spec_canonical_json(spec)
    spec_id = spec_json
    return [
        f"generated_by_spec_id: {spec_id}",
        f"generated_by_spec: {spec_json}",
    ]


def spec_metadata_lines_from_payload(payload: Mapping[str, JSONValue]) -> list[str]:
    spec_id = str(payload.get("generated_by_spec_id", "") or "")
    spec_payload = payload.get("generated_by_spec", {})
    if not isinstance(spec_payload, Mapping):
        spec_payload = {}
    spec_json = json.dumps(spec_payload, sort_keys=True, separators=(",", ":"))
    return [
        f"generated_by_spec_id: {spec_id}",
        f"generated_by_spec: {spec_json}",
    ]


def spec_metadata_payload(spec: ProjectionSpec) -> dict[str, JSONValue]:
    spec_id = spec_canonical_json(spec)
    return {
        "generated_by_spec_id": spec_id,
        "generated_by_spec": normalize_spec(spec),
    }


def iter_registered_specs() -> Iterable[ProjectionSpec]:
    return (
        NEVER_INVARIANTS_SPEC,
        TEST_OBSOLESCENCE_SUMMARY_SPEC,
        TEST_OBSOLESCENCE_BASELINE_SPEC,
        TEST_OBSOLESCENCE_STATE_SPEC,
        TEST_OBSOLESCENCE_DELTA_SPEC,
        AMBIGUITY_SUMMARY_SPEC,
        AMBIGUITY_SUITE_AGG_SPEC,
        AMBIGUITY_VIRTUAL_SET_SPEC,
        DEADLINE_OBLIGATIONS_SUMMARY_SPEC,
        LINT_FINDINGS_SPEC,
        REPORT_SECTION_LINES_SPEC,
        SUITE_ORDER_SPEC,
        CALL_CLUSTER_SUMMARY_SPEC,
        TEST_ANNOTATION_DRIFT_SPEC,
        TEST_ANNOTATION_DRIFT_BASELINE_SPEC,
        TEST_ANNOTATION_DRIFT_DELTA_SPEC,
        AMBIGUITY_BASELINE_SPEC,
        AMBIGUITY_DELTA_SPEC,
        AMBIGUITY_STATE_SPEC,
        WL_REFINEMENT_SPEC,
    )


with forest_scope(Forest()):
    with deadline_scope(Deadline.from_timeout_ms(1000)):
        with deadline_clock_scope(GasMeter(limit=_projection_registry_gas_limit())):
            REGISTERED_SPECS = {spec_hash(spec): spec for spec in iter_registered_specs()}
