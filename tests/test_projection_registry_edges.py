from __future__ import annotations

import pytest

from gabion.analysis.projection_registry import (
    WL_REFINEMENT_SPEC,
    _projection_registry_gas_limit,
    build_registered_specs,
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


def test_projection_registry_gas_limit_default_and_env_override(
    env_scope,
    restore_env,
) -> None:
    previous = env_scope({"GABION_PROJECTION_REGISTRY_GAS_LIMIT": None})
    try:
        assert _projection_registry_gas_limit() > 0
        nested_previous = env_scope({"GABION_PROJECTION_REGISTRY_GAS_LIMIT": "12345"})
        try:
            assert _projection_registry_gas_limit() == 12_345
        finally:
            restore_env(nested_previous)
    finally:
        restore_env(previous)


def test_projection_registry_gas_limit_rejects_invalid_env(
    env_scope,
    restore_env,
) -> None:
    previous = env_scope({"GABION_PROJECTION_REGISTRY_GAS_LIMIT": None})
    try:
        invalid_zero = env_scope({"GABION_PROJECTION_REGISTRY_GAS_LIMIT": "0"})
        try:
            with pytest.raises(NeverThrown):
                _projection_registry_gas_limit()
        finally:
            restore_env(invalid_zero)
        invalid_text = env_scope({"GABION_PROJECTION_REGISTRY_GAS_LIMIT": "bad"})
        try:
            with pytest.raises(NeverThrown):
                _projection_registry_gas_limit()
        finally:
            restore_env(invalid_text)
    finally:
        restore_env(previous)


def test_build_registered_specs_uses_configured_gas_limit(
    env_scope,
    restore_env,
) -> None:
    previous = env_scope({"GABION_PROJECTION_REGISTRY_GAS_LIMIT": "10000"})
    try:
        specs = build_registered_specs()
    finally:
        restore_env(previous)
    assert specs
