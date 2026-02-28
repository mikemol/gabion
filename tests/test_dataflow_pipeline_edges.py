from __future__ import annotations

from pathlib import Path

import pytest

from gabion.analysis import dataflow_pipeline
from gabion.exceptions import NeverThrown


def _bind() -> None:
    dataflow_pipeline._bind_audit_symbols()


# gabion:evidence E:call_footprint::tests/test_dataflow_pipeline_edges.py::test_normalized_dimension_payload_filters_invalid_entries::dataflow_pipeline.py::gabion.analysis.dataflow_pipeline._normalized_dimension_payload
def test_normalized_dimension_payload_filters_invalid_entries() -> None:
    _bind()
    payload = {
        "valid": {"done": 4, "total": 7},
        "invalid_done_type": {"done": "4", "total": 7},
        "invalid_done_bool": {"done": True, "total": 7},
        "invalid_payload": [],
        ("tuple-key",): {"done": 1, "total": 1},
        "clamped": {"done": -5, "total": -2},
    }
    normalized = dataflow_pipeline._normalized_dimension_payload(payload)
    assert normalized == {
        "valid": {"done": 4, "total": 7},
        "clamped": {"done": 0, "total": 0},
    }


# gabion:evidence E:call_footprint::tests/test_dataflow_pipeline_edges.py::test_apply_forest_progress_delta_non_mapping_is_noop::dataflow_pipeline.py::gabion.analysis.dataflow_pipeline._apply_forest_progress_delta
def test_apply_forest_progress_delta_non_mapping_is_noop() -> None:
    _bind()
    base_dimensions = {"x": {"done": 1, "total": 2}}
    assert dataflow_pipeline._apply_forest_progress_delta(
        0,
        forest_mutable_progress_done=1,
        forest_mutable_progress_total=2,
        forest_progress_marker="start",
        forest_dimensions=base_dimensions,
    ) == (1, 2, "start", base_dimensions, False)


# gabion:evidence E:call_footprint::tests/test_dataflow_pipeline_edges.py::test_apply_forest_progress_delta_handles_invalid_scalars_and_dimension_shape::dataflow_pipeline.py::gabion.analysis.dataflow_pipeline._apply_forest_progress_delta
def test_apply_forest_progress_delta_handles_invalid_scalars_and_dimension_shape() -> None:
    _bind()
    base_dimensions = {"x": {"done": 1, "total": 2}}
    done, total, marker, dimensions, changed = dataflow_pipeline._apply_forest_progress_delta(
        {
            "primary_done": True,
            "primary_total": "bad",
            "marker": "",
            "dimensions": [],
        },
        forest_mutable_progress_done=1,
        forest_mutable_progress_total=2,
        forest_progress_marker="start",
        forest_dimensions=base_dimensions,
    )
    assert changed is True
    assert done == 1
    assert total == 2
    assert marker == "start"
    assert dimensions == base_dimensions


# gabion:evidence E:decision_surface/direct::dataflow_pipeline.py::gabion.analysis.dataflow_pipeline.analyze_paths::on_collection_progress
def test_analyze_paths_rejects_invalid_collection_progress_callback() -> None:
    _bind()
    with pytest.raises(NeverThrown):
        dataflow_pipeline.analyze_paths(
            paths=[],
            forest=dataflow_pipeline.Forest(),
            recursive=False,
            type_audit=False,
            type_audit_report=False,
            type_audit_max=1,
            include_constant_smells=False,
            include_unused_arg_smells=False,
            on_collection_progress="not-callable",  # type: ignore[arg-type]
        )


# gabion:evidence E:decision_surface/direct::dataflow_pipeline.py::gabion.analysis.dataflow_pipeline.analyze_paths::on_phase_progress
def test_analyze_paths_rejects_invalid_phase_progress_callback() -> None:
    _bind()
    with pytest.raises(NeverThrown):
        dataflow_pipeline.analyze_paths(
            paths=[],
            forest=dataflow_pipeline.Forest(),
            recursive=False,
            type_audit=False,
            type_audit_report=False,
            type_audit_max=1,
            include_constant_smells=False,
            include_unused_arg_smells=False,
            on_phase_progress="not-callable",  # type: ignore[arg-type]
        )



# gabion:evidence E:call_footprint::tests/test_dataflow_pipeline_edges.py::test_dataflow_pipeline_collect_fingerprint_atoms_order_invariant::dataflow_pipeline.py::gabion.analysis.dataflow_pipeline._bind_audit_symbols
def test_dataflow_pipeline_collect_fingerprint_atoms_order_invariant() -> None:
    _bind()
    first = Path("pkg/a.py")
    second = Path("pkg/b.py")
    groups_a = {
        first: {"f": [{"alpha", "beta"}]},
        second: {"g": [{"payload"}]},
    }
    annotations_a = {
        first: {"f": {"alpha": "dict[str, int]", "beta": "list[int]"}},
        second: {"g": {"payload": "tuple[int, str]"}},
    }
    groups_b = {
        second: {"g": [{"payload"}]},
        first: {"f": [{"beta", "alpha"}]},
    }
    annotations_b = {
        second: {"g": {"payload": "tuple[int, str]"}},
        first: {"f": {"beta": "list[int]", "alpha": "dict[str, int]"}},
    }

    assert dataflow_pipeline._collect_fingerprint_atom_keys(
        groups_a,
        annotations_a,
    ) == dataflow_pipeline._collect_fingerprint_atom_keys(groups_b, annotations_b)


# gabion:evidence E:call_footprint::tests/test_dataflow_pipeline_edges.py::test_analyze_paths_primes_constructor_registry_from_collected_ctor_keys::dataflow_pipeline.py::gabion.analysis.dataflow_pipeline._collect_fingerprint_atom_keys::dataflow_pipeline.py::gabion.analysis.dataflow_pipeline.analyze_paths
def test_analyze_paths_primes_constructor_registry_from_collected_ctor_keys(
    tmp_path: Path,
) -> None:
    _bind()
    registry, index = dataflow_pipeline.build_fingerprint_registry(
        {"shape": ["list[int]"]},
    )
    ctor_registry = dataflow_pipeline.TypeConstructorRegistry(registry)
    original_collect = dataflow_pipeline._collect_fingerprint_atom_keys
    dataflow_pipeline._collect_fingerprint_atom_keys = (
        lambda _groups, _annots: ([], ["list"])
    )
    try:
        dataflow_pipeline.analyze_paths(
            paths=[],
            forest=dataflow_pipeline.Forest(),
            recursive=False,
            type_audit=False,
            type_audit_report=False,
            type_audit_max=0,
            include_constant_smells=False,
            include_unused_arg_smells=False,
            config=dataflow_pipeline.AuditConfig(
                project_root=tmp_path,
                fingerprint_registry=registry,
                fingerprint_index=index,
                constructor_registry=ctor_registry,
            ),
            file_paths_override=[],
        )
    finally:
        dataflow_pipeline._collect_fingerprint_atom_keys = original_collect

    assert registry.prime_for("ctor:list") is not None


def test_capability_enabled_treats_non_mapping_capabilities_as_enabled() -> None:
    _bind()
    assert dataflow_pipeline._capability_enabled(
        {"name": "adapter", "capabilities": []},
        "bundle_inference",
    ) is True


def test_unsupported_surface_diagnostic_uses_adapter_name_from_contract(tmp_path: Path) -> None:
    _bind()
    diagnostic = dataflow_pipeline._unsupported_surface_diagnostic(
        surface="bundle-inference",
        capability_name="bundle_inference",
        runtime_config=dataflow_pipeline.AuditConfig(
            project_root=tmp_path,
            adapter_contract={"name": "limited"},
        ),
    )
    assert diagnostic["adapter"] == "limited"
    assert diagnostic["required_by_policy"] is False


def test_unsupported_surface_diagnostic_defaults_native_for_non_mapping_contract(
    tmp_path: Path,
) -> None:
    _bind()
    diagnostic = dataflow_pipeline._unsupported_surface_diagnostic(
        surface="bundle-inference",
        capability_name="bundle_inference",
        runtime_config=dataflow_pipeline.AuditConfig(
            project_root=tmp_path,
            adapter_contract="legacy",
        ),
    )
    assert diagnostic["adapter"] == "native"


def test_analyze_paths_disables_unsupported_surfaces_by_adapter_contract(tmp_path: Path) -> None:
    _bind()
    result = dataflow_pipeline.analyze_paths(
        paths=[],
        forest=dataflow_pipeline.Forest(),
        recursive=False,
        type_audit=True,
        type_audit_report=True,
        type_audit_max=1,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_bundle_forest=True,
        include_decision_surfaces=True,
        include_value_decision_surfaces=True,
        include_exception_obligations=True,
        include_handledness_witnesses=True,
        include_rewrite_plans=True,
        config=dataflow_pipeline.AuditConfig(
            project_root=tmp_path,
            adapter_contract={
                "name": "limited",
                "capabilities": {
                    "bundle_inference": False,
                    "decision_surfaces": False,
                    "type_flow": False,
                    "exception_obligations": False,
                    "rewrite_plan_support": False,
                },
            },
        ),
        file_paths_override=[],
    )
    surfaces = {
        str(item.get("surface", ""))
        for item in result.unsupported_by_adapter
    }
    assert {
        "bundle-inference",
        "decision-surfaces",
        "type-flow",
        "exception-obligations",
        "rewrite-plan-support",
    }.issubset(surfaces)
