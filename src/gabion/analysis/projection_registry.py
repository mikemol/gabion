from __future__ import annotations

from typing import Iterable

from gabion.analysis.projection_normalize import (
    normalize_spec,
    spec_canonical_json,
    spec_hash,
)
from gabion.analysis.projection_spec import ProjectionOp, ProjectionSpec
from gabion.json_types import JSONValue


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


def spec_metadata_lines(spec: ProjectionSpec) -> list[str]:
    spec_id = spec_hash(spec)
    spec_json = spec_canonical_json(spec)
    return [
        f"generated_by_spec_id: {spec_id}",
        f"generated_by_spec: {spec_json}",
    ]


def spec_metadata_payload(spec: ProjectionSpec) -> dict[str, JSONValue]:
    return {
        "generated_by_spec_id": spec_hash(spec),
        "generated_by_spec": normalize_spec(spec),
    }


def iter_registered_specs() -> Iterable[ProjectionSpec]:
    return (NEVER_INVARIANTS_SPEC, TEST_OBSOLESCENCE_SUMMARY_SPEC)


REGISTERED_SPECS = {spec_hash(spec): spec for spec in iter_registered_specs()}
