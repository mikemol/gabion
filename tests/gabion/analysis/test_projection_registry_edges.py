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
from gabion.runtime.policy_runtime import RuntimePolicyConfig, runtime_policy_scope


# gabion:evidence E:call_footprint::tests/test_projection_registry_edges.py::test_spec_metadata_lines_emit_canonical_id_and_json::projection_registry.py::gabion.analysis.projection_registry.spec_metadata_lines
def test_spec_metadata_lines_emit_canonical_id_and_json() -> None:
    lines = spec_metadata_lines(WL_REFINEMENT_SPEC)
    assert lines[0].startswith("generated_by_spec_id: ")
    assert lines[1].startswith("generated_by_spec: ")


# gabion:evidence E:call_footprint::tests/test_projection_registry_edges.py::test_spec_metadata_lines_from_payload_defaults_non_mapping_spec_payload::projection_registry.py::gabion.analysis.projection_registry.spec_metadata_lines_from_payload
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


# gabion:evidence E:call_footprint::tests/test_projection_registry_edges.py::test_projection_registry_gas_limit_default_and_env_override::projection_registry.py::gabion.analysis.projection_registry._projection_registry_gas_limit
def test_projection_registry_gas_limit_default_and_override() -> None:
    assert _projection_registry_gas_limit() > 0
    with runtime_policy_scope(RuntimePolicyConfig(projection_registry_gas_limit=12_345)):
        assert _projection_registry_gas_limit() == 12_345


# gabion:evidence E:call_footprint::tests/test_projection_registry_edges.py::test_projection_registry_gas_limit_rejects_invalid_env::projection_registry.py::gabion.analysis.projection_registry._projection_registry_gas_limit
def test_projection_registry_gas_limit_rejects_invalid_runtime_value() -> None:
    with runtime_policy_scope(RuntimePolicyConfig(projection_registry_gas_limit=1)):
        assert _projection_registry_gas_limit() == 1
    with pytest.raises(NeverThrown):
        with runtime_policy_scope(RuntimePolicyConfig(projection_registry_gas_limit=0)):
            _projection_registry_gas_limit()


# gabion:evidence E:call_footprint::tests/test_projection_registry_edges.py::test_build_registered_specs_uses_configured_gas_limit::projection_registry.py::gabion.analysis.projection_registry.build_registered_specs
def test_build_registered_specs_uses_configured_gas_limit() -> None:
    with runtime_policy_scope(RuntimePolicyConfig(projection_registry_gas_limit=10_000)):
        specs = build_registered_specs()
    assert specs
