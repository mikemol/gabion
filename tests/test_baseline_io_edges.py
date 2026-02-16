from __future__ import annotations

import pytest

from gabion.analysis.baseline_io import parse_spec_metadata, parse_version


def test_parse_version_rejects_empty_allowed_versions() -> None:
    with pytest.raises(ValueError):
        parse_version({}, expected=())


def test_parse_spec_metadata_ignores_non_mapping_spec_payload() -> None:
    spec_id, spec = parse_spec_metadata(
        {
            "generated_by_spec_id": "spec-id",
            "generated_by_spec": ["not", "a", "mapping"],
        }
    )
    assert spec_id == "spec-id"
    assert spec == {}
