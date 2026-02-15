from __future__ import annotations

import os

import pytest

from gabion.analysis.projection_registry import (
    WL_REFINEMENT_SPEC,
    _projection_registry_gas_limit,
    spec_metadata_lines,
    spec_metadata_lines_from_payload,
)
from gabion.exceptions import NeverThrown


def test_spec_metadata_lines_emit_canonical_id_and_json() -> None:
    lines = spec_metadata_lines(WL_REFINEMENT_SPEC)
    assert lines[0].startswith("generated_by_spec_id: ")
    assert lines[1].startswith("generated_by_spec: ")


def test_spec_metadata_lines_from_payload_defaults_non_mapping_spec_payload() -> None:
    lines = spec_metadata_lines_from_payload(
        {
            "generated_by_spec_id": "spec-id",
            "generated_by_spec": "invalid",
        }
    )
    assert lines == [
        "generated_by_spec_id: spec-id",
        "generated_by_spec: {}",
    ]


def test_projection_registry_gas_limit_default_and_env_override() -> None:
    previous = os.environ.get("GABION_PROJECTION_REGISTRY_GAS_LIMIT")
    try:
        os.environ.pop("GABION_PROJECTION_REGISTRY_GAS_LIMIT", None)
        assert _projection_registry_gas_limit() > 0
        os.environ["GABION_PROJECTION_REGISTRY_GAS_LIMIT"] = "12345"
        assert _projection_registry_gas_limit() == 12_345
    finally:
        if previous is None:
            os.environ.pop("GABION_PROJECTION_REGISTRY_GAS_LIMIT", None)
        else:
            os.environ["GABION_PROJECTION_REGISTRY_GAS_LIMIT"] = previous


def test_projection_registry_gas_limit_rejects_invalid_env() -> None:
    previous = os.environ.get("GABION_PROJECTION_REGISTRY_GAS_LIMIT")
    try:
        os.environ["GABION_PROJECTION_REGISTRY_GAS_LIMIT"] = "0"
        with pytest.raises(NeverThrown):
            _projection_registry_gas_limit()
        os.environ["GABION_PROJECTION_REGISTRY_GAS_LIMIT"] = "bad"
        with pytest.raises(NeverThrown):
            _projection_registry_gas_limit()
    finally:
        if previous is None:
            os.environ.pop("GABION_PROJECTION_REGISTRY_GAS_LIMIT", None)
        else:
            os.environ["GABION_PROJECTION_REGISTRY_GAS_LIMIT"] = previous
