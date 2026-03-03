from __future__ import annotations

import pytest

from gabion.analysis.dataflow.io import dataflow_projection_helpers as helpers
from gabion.exceptions import NeverThrown


def test_topologically_order_report_projection_specs_happy_path() -> None:
    specs = (
        helpers.ReportProjectionSpec("intro", "collection", (), False),
        helpers.ReportProjectionSpec("components", "forest", ("intro",), True),
        helpers.ReportProjectionSpec("violations", "post", ("components",), True),
    )

    ordered = helpers._topologically_order_report_projection_specs(specs)

    assert [spec.section_id for spec in ordered] == ["intro", "components", "violations"]


def test_topologically_order_report_projection_specs_rejects_duplicate_section_ids() -> None:
    specs = (
        helpers.ReportProjectionSpec("intro", "collection", (), False),
        helpers.ReportProjectionSpec("intro", "post", (), True),
    )

    with pytest.raises(NeverThrown):
        helpers._topologically_order_report_projection_specs(specs)


def test_topologically_order_report_projection_specs_rejects_missing_dependency() -> None:
    specs = (
        helpers.ReportProjectionSpec("intro", "collection", (), False),
        helpers.ReportProjectionSpec("violations", "post", ("components",), True),
    )

    with pytest.raises(NeverThrown):
        helpers._topologically_order_report_projection_specs(specs)


def test_topologically_order_report_projection_specs_rejects_self_dependency() -> None:
    specs = (
        helpers.ReportProjectionSpec("intro", "collection", (), False),
        helpers.ReportProjectionSpec("components", "forest", ("components",), True),
    )

    with pytest.raises(NeverThrown):
        helpers._topologically_order_report_projection_specs(specs)


def test_topologically_order_report_projection_specs_rejects_dependency_cycle() -> None:
    specs = (
        helpers.ReportProjectionSpec("a", "forest", ("b",), True),
        helpers.ReportProjectionSpec("b", "post", ("a",), True),
    )

    with pytest.raises(NeverThrown):
        helpers._topologically_order_report_projection_specs(specs)


def test_report_projection_spec_rows_emit_serializable_payload() -> None:
    specs = helpers.report_projection_specs()
    assert specs

    rows = helpers.report_projection_spec_rows()

    assert rows
    first = rows[0]
    assert {"section_id", "phase", "deps", "has_preview"} <= set(first)
