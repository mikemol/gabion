# gabion:decision_protocol_module
from __future__ import annotations

"""Projection-surface helpers used outside the legacy runtime module."""

from collections import Counter
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from typing import Literal

from gabion.analysis.json_types import JSONObject
from gabion.invariants import never
from gabion.order_contract import sort_once

ReportProjectionPhase = Literal["collection", "forest", "edge", "post"]


@dataclass(frozen=True)
class ReportProjectionSpec:
    section_id: str
    phase: ReportProjectionPhase
    deps: tuple[str, ...]
    has_preview: bool


_REPORT_PROJECTION_PHASE_RANKS: dict[ReportProjectionPhase, int] = {
    "collection": 0,
    "forest": 1,
    "edge": 2,
    "post": 3,
}

_REPORT_PROJECTION_DECLARED_SPECS: tuple[ReportProjectionSpec, ...] = (
    ReportProjectionSpec("intro", "collection", (), False),
    ReportProjectionSpec("components", "forest", ("intro",), True),
    ReportProjectionSpec("type_flow", "edge", ("components",), True),
    ReportProjectionSpec("violations", "post", ("components",), True),
    ReportProjectionSpec("deadline_summary", "post", ("components",), True),
    ReportProjectionSpec("resumability_obligations", "post", ("components",), True),
    ReportProjectionSpec(
        "incremental_report_obligations",
        "post",
        ("components",),
        True,
    ),
    ReportProjectionSpec("parse_failure_witnesses", "post", ("components",), True),
    ReportProjectionSpec(
        "execution_pattern_suggestions",
        "post",
        ("components",),
        True,
    ),
    ReportProjectionSpec("pattern_schema_residue", "post", ("components",), True),
    ReportProjectionSpec("decision_surfaces", "post", ("components",), True),
    ReportProjectionSpec("fingerprint_warnings", "post", ("components",), True),
    ReportProjectionSpec("fingerprint_matches", "post", ("components",), True),
    ReportProjectionSpec("fingerprint_synthesis", "post", ("components",), True),
    ReportProjectionSpec("schema_surfaces", "post", ("components",), True),
    ReportProjectionSpec("deprecated_substrate", "post", ("components",), True),
    ReportProjectionSpec("constant_smells", "edge", ("type_flow",), True),
    ReportProjectionSpec("unused_arg_smells", "edge", ("type_flow",), True),
    ReportProjectionSpec(
        "value_decision_surfaces",
        "post",
        ("decision_surfaces",),
        True,
    ),
    ReportProjectionSpec("context_suggestions", "post", ("decision_surfaces",), True),
)


def report_projection_phase_rank(phase: ReportProjectionPhase) -> int:
    return _REPORT_PROJECTION_PHASE_RANKS[phase]


def _topologically_order_report_projection_specs(
    specs: tuple[ReportProjectionSpec, ...],
) -> tuple[ReportProjectionSpec, ...]:
    by_id = {spec.section_id: spec for spec in specs}
    if len(by_id) != len(specs):
        duplicate_id = next(
            section_id
            for section_id, count in Counter(spec.section_id for spec in specs).items()
            if count > 1
        )
        never("duplicate report projection section_id", section_id=duplicate_id)
    declaration_index = {spec.section_id: idx for idx, spec in enumerate(specs)}
    dep_pairs = tuple((spec.section_id, dep) for spec in specs for dep in spec.deps)
    missing_dep = next(
        ((section_id, dep) for section_id, dep in dep_pairs if dep not in by_id),
        None,
    )
    if missing_dep is not None:
        section_id, dep = missing_dep
        never(
            "report projection dependency missing",
            section_id=section_id,
            missing_dep=dep,
        )
    self_dep_section = next(
        (section_id for section_id, dep in dep_pairs if dep == section_id),
        None,
    )
    if self_dep_section is not None:
        never("report projection self dependency", section_id=self_dep_section)

    def _order_key(section_id: str) -> tuple[int, int, str]:
        spec = by_id[section_id]
        return (
            report_projection_phase_rank(spec.phase),
            declaration_index[section_id],
            section_id,
        )

    prioritized_ids = tuple(sort_once(by_id, key=_order_key, source=__file__))
    predecessor_graph = {
        section_id: tuple(
            dict.fromkeys(sort_once(by_id[section_id].deps, key=_order_key, source=__file__))
        )
        for section_id in prioritized_ids
    }
    try:
        ordered_ids = tuple(TopologicalSorter(predecessor_graph).static_order())
    except CycleError as exc:
        cycle = exc.args[1] if len(exc.args) > 1 else ()
        unresolved = [str(item) for item in cycle]
        never("report projection dependency cycle", unresolved=unresolved)
    return tuple(by_id[section_id] for section_id in ordered_ids)


_REPORT_PROJECTION_SPECS = _topologically_order_report_projection_specs(
    _REPORT_PROJECTION_DECLARED_SPECS
)


def report_projection_specs() -> tuple[ReportProjectionSpec, ...]:
    return _REPORT_PROJECTION_SPECS


def report_projection_spec_rows() -> list[JSONObject]:
    return [
        {
            "section_id": spec.section_id,
            "phase": spec.phase,
            "deps": list(spec.deps),
            "has_preview": spec.has_preview,
        }
        for spec in _REPORT_PROJECTION_SPECS
    ]

__all__ = [
    "ReportProjectionPhase",
    "ReportProjectionSpec",
    "_topologically_order_report_projection_specs",
    "report_projection_phase_rank",
    "report_projection_spec_rows",
    "report_projection_specs",
]
