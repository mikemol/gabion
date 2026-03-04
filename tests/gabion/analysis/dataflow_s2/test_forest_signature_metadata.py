from __future__ import annotations

import pytest

from gabion.analysis.dataflow.io import dataflow_snapshot_io, dataflow_structure_reuse
from gabion.analysis.dataflow.io.forest_signature_metadata import (
    apply_forest_signature_metadata,
)


@pytest.mark.parametrize(
    ("snapshot", "prefix", "expected"),
    [
        (
            {
                "forest_signature": "sig",
                "forest_signature_partial": False,
                "forest_signature_basis": "exact",
            },
            "",
            {
                "forest_signature": "sig",
                "forest_signature_partial": False,
                "forest_signature_basis": "exact",
            },
        ),
        (
            {
                "forest_signature_partial": False,
                "forest_signature_basis": "fallback",
            },
            "",
            {
                "forest_signature_partial": True,
                "forest_signature_basis": "fallback",
            },
        ),
        (
            {},
            "",
            {
                "forest_signature_partial": True,
                "forest_signature_basis": "missing",
            },
        ),
        (
            {"forest_signature": "sig", "forest_signature_basis": "exact"},
            "baseline_",
            {
                "baseline_forest_signature": "sig",
                "baseline_forest_signature_basis": "exact",
            },
        ),
    ],
)
def test_apply_forest_signature_metadata_cases(
    snapshot: dict[str, object],
    prefix: str,
    expected: dict[str, object],
) -> None:
    payload: dict[str, object] = {}
    apply_forest_signature_metadata(payload, snapshot, prefix=prefix)
    assert payload == expected


def test_snapshot_and_reuse_modules_share_helper_behavior() -> None:
    snapshot = {"forest_signature_basis": "observed"}

    from_snapshot_module: dict[str, object] = {}
    dataflow_snapshot_io.apply_forest_signature_metadata(
        from_snapshot_module,
        snapshot,
        prefix="current_",
    )

    from_reuse_module: dict[str, object] = {}
    dataflow_structure_reuse.apply_forest_signature_metadata(
        from_reuse_module,
        snapshot,
        prefix="current_",
    )

    expected = {
        "current_forest_signature_partial": True,
        "current_forest_signature_basis": "observed",
    }
    assert from_snapshot_module == expected
    assert from_reuse_module == expected
