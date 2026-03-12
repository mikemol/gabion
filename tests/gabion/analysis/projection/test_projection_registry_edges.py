from __future__ import annotations

import pytest

from gabion.analysis.projection.projection_registry import (
    PROJECTION_FIBER_FRONTIER_SPEC,
    PROJECTION_FIBER_REFLECTION_SPEC,
    PROJECTION_FIBER_REFLECTIVE_BOUNDARY_SPEC,
    WL_REFINEMENT_SPEC,
    _projection_registry_gas_limit,
    build_registered_specs,
    spec_metadata_lines,
    spec_metadata_lines_from_payload,
)
from gabion.exceptions import NeverThrown
from gabion.runtime.policy_runtime import RuntimePolicyConfig, runtime_policy_scope


# gabion:evidence E:call_footprint::tests/test_projection_registry_edges.py::test_spec_metadata_lines_emit_canonical_id_and_json::projection_registry.py::gabion.analysis.projection_registry.spec_metadata_lines
# gabion:behavior primary=verboten facets=edge
def test_spec_metadata_lines_emit_canonical_id_and_json() -> None:
    lines = spec_metadata_lines(WL_REFINEMENT_SPEC)
    assert lines[0].startswith("generated_by_spec_id: ")
    assert lines[1].startswith("generated_by_spec: ")


# gabion:evidence E:call_footprint::tests/test_projection_registry_edges.py::test_spec_metadata_lines_from_payload_defaults_non_mapping_spec_payload::projection_registry.py::gabion.analysis.projection_registry.spec_metadata_lines_from_payload
# gabion:behavior primary=verboten facets=edge
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
# gabion:behavior primary=verboten facets=edge
def test_projection_registry_gas_limit_default_and_override() -> None:
    assert _projection_registry_gas_limit() > 0
    with runtime_policy_scope(RuntimePolicyConfig(projection_registry_gas_limit=12_345)):
        assert _projection_registry_gas_limit() == 12_345


# gabion:evidence E:call_footprint::tests/test_projection_registry_edges.py::test_projection_registry_gas_limit_rejects_invalid_env::projection_registry.py::gabion.analysis.projection_registry._projection_registry_gas_limit
# gabion:behavior primary=verboten facets=edge,invalid
def test_projection_registry_gas_limit_rejects_invalid_runtime_value() -> None:
    with runtime_policy_scope(RuntimePolicyConfig(projection_registry_gas_limit=1)):
        assert _projection_registry_gas_limit() == 1
    with pytest.raises(NeverThrown):
        with runtime_policy_scope(RuntimePolicyConfig(projection_registry_gas_limit=0)):
            _projection_registry_gas_limit()


# gabion:evidence E:call_footprint::tests/test_projection_registry_edges.py::test_build_registered_specs_uses_configured_gas_limit::projection_registry.py::gabion.analysis.projection_registry.build_registered_specs
# gabion:behavior primary=verboten facets=edge
def test_build_registered_specs_uses_configured_gas_limit() -> None:
    with runtime_policy_scope(RuntimePolicyConfig(projection_registry_gas_limit=10_000)):
        specs = build_registered_specs()
    assert specs


def test_projection_fiber_frontier_spec_is_registered_with_declared_quotient_face() -> None:
    specs = build_registered_specs()
    assert PROJECTION_FIBER_FRONTIER_SPEC in specs.values()
    project_op = PROJECTION_FIBER_FRONTIER_SPEC.pipeline[0]
    assert project_op.op == "project"
    assert project_op.params["quotient_face"] == "projection_fiber.frontier"


def test_projection_fiber_reflective_boundary_spec_is_registered_with_declared_quotient_face() -> None:
    specs = build_registered_specs()
    assert PROJECTION_FIBER_REFLECTIVE_BOUNDARY_SPEC in specs.values()
    project_op = PROJECTION_FIBER_REFLECTIVE_BOUNDARY_SPEC.pipeline[0]
    assert project_op.op == "project"
    assert project_op.params["quotient_face"] == "projection_fiber.reflective_boundary"


def test_projection_fiber_reflection_spec_is_registered_with_declared_surface() -> None:
    specs = build_registered_specs()
    assert PROJECTION_FIBER_REFLECTION_SPEC in specs.values()
    reflect_op = PROJECTION_FIBER_REFLECTION_SPEC.pipeline[0]
    assert reflect_op.op == "reflect"
    assert reflect_op.params["surface"] == "projection_fiber"
