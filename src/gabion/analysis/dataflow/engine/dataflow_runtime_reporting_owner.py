# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

"""Canonical owner for report-section runtime helpers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Literal, TypeVar

from gabion.analysis.dataflow.engine.dataflow_contracts import ReportCarrier
from gabion.analysis.dataflow.io.dataflow_reporting import emit_report as _emit_report
from gabion.analysis.indexed_scan.scanners.report_sections import (
    extract_report_sections as _extract_report_sections,
)
from gabion.order_contract import sort_once

ReportProjectionPhase = Literal["collection", "forest", "edge", "post"]
_ReportSectionValue = TypeVar("_ReportSectionValue")


@dataclass(frozen=True)
class ReportProjectionSpec(Generic[_ReportSectionValue]):
    section_id: str
    phase: ReportProjectionPhase
    deps: tuple[str, ...]
    build: Callable[
        [ReportCarrier, dict[Path, dict[str, list[set[str]]]]],
        _ReportSectionValue,
    ]
    render: Callable[[_ReportSectionValue], list[str]]
    violation_extract: Callable[[_ReportSectionValue], list[str]]
    preview_build: object = None


def _report_section_identity_render(lines: list[str]) -> list[str]:
    return lines


def _report_section_no_violations(_lines: list[str]) -> list[str]:
    return []


def _report_section_text(
    report: ReportCarrier,
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    *,
    section_id: str,
) -> list[str]:
    rendered, _ = _emit_report(
        groups_by_path,
        max_components=10,
        report=report,
    )
    return _extract_report_sections(rendered).get(section_id, [])


def _report_section_spec(
    *,
    section_id: str,
    phase: ReportProjectionPhase,
    deps: tuple[str, ...] = (),
    preview_build: object = None,
) -> ReportProjectionSpec[list[str]]:
    return ReportProjectionSpec[list[str]](
        section_id=section_id,
        phase=phase,
        deps=deps,
        build=lambda report, groups_by_path, _section_id=section_id: _report_section_text(
            report,
            groups_by_path,
            section_id=_section_id,
        ),
        render=_report_section_identity_render,
        violation_extract=_report_section_no_violations,
        preview_build=preview_build,
    )


def _compute_violations(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    max_components: int,
    *,
    report: ReportCarrier,
) -> list[str]:
    _, violations = _emit_report(
        groups_by_path,
        max_components,
        report=report,
    )
    return sort_once(
        set(violations),
        source="_compute_violations.violations",
    )


__all__ = [
    "ReportProjectionSpec",
    "ReportProjectionPhase",
    "_compute_violations",
    "_report_section_identity_render",
    "_report_section_no_violations",
    "_report_section_spec",
    "_report_section_text",
]
