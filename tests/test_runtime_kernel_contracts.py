from __future__ import annotations

from pathlib import Path

import pytest

from gabion.exceptions import NeverThrown
from gabion.runtime import (
    deadline_policy,
    env_policy,
    json_io,
    path_policy,
    stable_encode,
)
from gabion.tooling import delta_gate, tool_specs
from tests.env_helpers import env_scope


# gabion:evidence E:call_footprint::tests/test_runtime_kernel_contracts.py::test_env_policy_truthy_helpers_explicit_values::env_policy.py::gabion.runtime.env_policy.env_enabled_flag::env_policy.py::gabion.runtime.env_policy.env_enabled_truthy_only
def test_env_policy_truthy_helpers_explicit_values() -> None:
    assert env_policy.env_enabled_truthy_only("UNUSED", value="yes") is True
    assert env_policy.env_enabled_truthy_only("UNUSED", value="no") is False
    assert env_policy.env_enabled_flag("UNUSED", value="on") is True
    assert env_policy.env_enabled_flag("UNUSED", value="off") is False


# gabion:evidence E:call_footprint::tests/test_runtime_kernel_contracts.py::test_env_policy_zero_seconds_rejected::env_policy.py::gabion.runtime.env_policy.timeout_ticks_from_env::env_helpers.py::tests.env_helpers.env_scope
def test_env_policy_zero_seconds_rejected() -> None:
    with env_scope(
        {
            "GABION_LSP_TIMEOUT_TICKS": None,
            "GABION_LSP_TIMEOUT_TICK_NS": None,
            "GABION_LSP_TIMEOUT_MS": None,
            "GABION_LSP_TIMEOUT_SECONDS": "0",
        }
    ):
        with pytest.raises(NeverThrown):
            env_policy.timeout_ticks_from_env()


# gabion:evidence E:call_footprint::tests/test_runtime_kernel_contracts.py::test_deadline_policy_budget_from_env_paths::deadline_policy.py::gabion.runtime.deadline_policy.timeout_budget_from_lsp_env::env_helpers.py::tests.env_helpers.env_scope
def test_deadline_policy_budget_from_env_paths() -> None:
    with env_scope(
        {
            "GABION_LSP_TIMEOUT_TICKS": None,
            "GABION_LSP_TIMEOUT_TICK_NS": None,
            "GABION_LSP_TIMEOUT_MS": None,
            "GABION_LSP_TIMEOUT_SECONDS": None,
        }
    ):
        budget = deadline_policy.timeout_budget_from_lsp_env(
            default_budget=deadline_policy.DeadlineBudget(ticks=7, tick_ns=9),
        )
    assert budget.ticks == 7
    assert budget.tick_ns == 9

    with env_scope(
        {
            "GABION_LSP_TIMEOUT_TICKS": "11",
            "GABION_LSP_TIMEOUT_TICK_NS": "13",
            "GABION_LSP_TIMEOUT_MS": None,
            "GABION_LSP_TIMEOUT_SECONDS": None,
        }
    ):
        budget = deadline_policy.timeout_budget_from_lsp_env(
            default_budget=deadline_policy.DeadlineBudget(ticks=1, tick_ns=1),
        )
    assert budget.ticks == 11
    assert budget.tick_ns == 13


# gabion:evidence E:call_footprint::tests/test_runtime_kernel_contracts.py::test_path_policy_resolve_report_path_branches::path_policy.py::gabion.runtime.path_policy.resolve_report_path
def test_path_policy_resolve_report_path_branches(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit.md"
    assert path_policy.resolve_report_path(explicit, root=tmp_path) == explicit
    assert path_policy.resolve_report_path(None, root=tmp_path) == (
        tmp_path / path_policy.DEFAULT_CHECK_REPORT_REL_PATH
    )


# gabion:evidence E:call_footprint::tests/test_runtime_kernel_contracts.py::test_json_io_load_text_invalid_returns_empty::json_io.py::gabion.runtime.json_io.load_json_object_text
def test_json_io_load_text_invalid_returns_empty() -> None:
    assert json_io.load_json_object_text("{bad") == {}


# gabion:evidence E:call_footprint::tests/test_runtime_kernel_contracts.py::test_json_io_canonicalizes_nested_mapping_order::json_io.py::gabion.runtime.json_io.load_json_object_text::json_io.py::gabion.runtime.json_io.dump_json_pretty
def test_json_io_canonicalizes_nested_mapping_order() -> None:
    payload = json_io.load_json_object_text(
        '{"z": {"b": 2, "a": 1}, "a": [{"y": 2, "x": 1}], "m": {"k": null}}'
    )
    assert list(payload.keys()) == ["a", "m", "z"]
    nested = payload.get("z")
    assert isinstance(nested, dict)
    assert list(nested.keys()) == ["a", "b"]
    rendered = json_io.dump_json_pretty(payload)
    assert rendered.index('"a"') < rendered.index('"m"') < rendered.index('"z"')


# gabion:evidence E:call_footprint::tests/test_runtime_kernel_contracts.py::test_json_io_dump_rejects_unsorted_mapping_payload::json_io.py::gabion.runtime.json_io.dump_json_pretty
def test_json_io_dump_rejects_unsorted_mapping_payload() -> None:
    with pytest.raises(NeverThrown):
        json_io.dump_json_pretty({"z": 1, "a": 2})


# gabion:evidence E:call_footprint::tests/test_runtime_kernel_contracts.py::test_stable_encode_compact_bytes_and_stringify_fallback::stable_encode.py::gabion.runtime.stable_encode.stable_compact_bytes::stable_encode.py::gabion.runtime.stable_encode.stable_json_value
def test_stable_encode_compact_bytes_and_stringify_fallback() -> None:
    class _Custom:
        def __str__(self) -> str:
            return "custom-value"

    payload = {"b": 2, "a": _Custom()}
    encoded = stable_encode.stable_compact_bytes(payload)
    assert isinstance(encoded, bytes)
    text = encoded.decode("utf-8")
    assert text == stable_encode.stable_compact_text(payload)
    assert '"a":"custom-value"' in text


# gabion:evidence E:call_footprint::tests/test_runtime_kernel_contracts.py::test_delta_gate_load_payload_handles_unicode_error::delta_gate.py::gabion.tooling.delta_gate._load_payload
def test_delta_gate_load_payload_handles_unicode_error(tmp_path: Path) -> None:
    invalid_utf = tmp_path / "invalid_utf.json"
    invalid_utf.write_bytes(b"\xff")
    payload, decode_error = delta_gate._load_payload(invalid_utf)
    assert payload is None
    assert decode_error is None


# gabion:evidence E:call_footprint::tests/test_runtime_kernel_contracts.py::test_tool_specs_triplet_map_sorted_and_filtered::tool_specs.py::gabion.tooling.tool_specs.triplet_specs_map::tool_specs.py::gabion.tooling.tool_specs.dataflow_stage_gate_specs
def test_tool_specs_triplet_map_sorted_and_filtered() -> None:
    triplets = tool_specs.triplet_specs_map()
    assert list(triplets.keys()) == sorted(triplets.keys())
    gates = tool_specs.dataflow_stage_gate_specs()
    assert all(spec.include_dataflow_stage_gate for spec in gates)
    assert all(spec.kind == "gate" for spec in gates)
