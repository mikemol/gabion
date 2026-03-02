from __future__ import annotations

import re
from pathlib import Path

import pytest

from decimal import ROUND_FLOOR

from gabion.exceptions import NeverThrown
from gabion.runtime import (
    deadline_policy,
    env_policy,
    json_io,
    path_policy,
    stable_encode,
)
from gabion.tooling import delta_gate, governance_rules, tool_specs
from tests.env_helpers import env_scope


def _default_controller_drift_policy() -> governance_rules.ControllerDriftPolicy:
    return governance_rules.ControllerDriftPolicy(
        severity_classes=("low", "medium", "high", "critical"),
        enforce_at_or_above="high",
        remediation_by_severity={"high": "override_or_fix"},
        consecutive_passes_required=3,
    )


# gabion:evidence E:call_footprint::tests/test_runtime_kernel_contracts.py::test_env_policy_truthy_helpers_explicit_values::env_policy.py::gabion.runtime.env_policy.env_enabled_flag::env_policy.py::gabion.runtime.env_policy.env_enabled_truthy_only
def test_env_policy_truthy_helpers_explicit_values() -> None:
    assert env_policy.env_enabled_truthy_only("UNUSED", value="yes") is True
    assert env_policy.env_enabled_truthy_only("UNUSED", value="no") is False
    assert env_policy.env_enabled_flag("UNUSED", value="on") is True
    assert env_policy.env_enabled_flag("UNUSED", value="off") is False


# gabion:evidence E:call_footprint::tests/test_runtime_kernel_contracts.py::test_env_policy_timeout_env_removed::env_policy.py::gabion.runtime.env_policy.timeout_ticks_from_env
def test_env_policy_timeout_env_removed() -> None:
    original_has_any_non_empty_env = env_policy.has_any_non_empty_env
    try:
        env_policy.has_any_non_empty_env = lambda _keys: True  # type: ignore[assignment]
        with pytest.raises(NeverThrown):
            env_policy.timeout_ticks_from_env()
        env_policy.has_any_non_empty_env = lambda _keys: False  # type: ignore[assignment]
        with pytest.raises(NeverThrown):
            env_policy.timeout_ticks_from_env()
    finally:
        env_policy.has_any_non_empty_env = original_has_any_non_empty_env


# gabion:evidence E:call_footprint::tests/test_runtime_kernel_contracts.py::test_env_policy_helpers_cover_non_env_timeout_paths::env_policy.py::gabion.runtime.env_policy.has_any_non_empty_env::env_policy.py::gabion.runtime.env_policy.parse_positive_int_text
def test_env_policy_helpers_cover_non_env_timeout_paths() -> None:
    assert env_policy.has_any_non_empty_env(("UNUSED_TIMEOUT_KEY",)) is False
    assert env_policy.parse_positive_int_text("7", field="ticks") == 7
    with pytest.raises(NeverThrown):
        env_policy.parse_positive_int_text("0", field="ticks")
    with pytest.raises(NeverThrown):
        env_policy.parse_positive_int_text("bad", field="ticks")


# gabion:evidence E:function_site::test_runtime_kernel_contracts.py::tests.test_runtime_kernel_contracts.test_env_policy_cli_timeout_overrides_and_scope_paths
def test_env_policy_cli_timeout_overrides_and_scope_paths() -> None:
    with pytest.raises(NeverThrown):
        env_policy.LspTimeoutConfig(ticks=0, tick_ns=1)
    with pytest.raises(NeverThrown):
        env_policy.LspTimeoutConfig(ticks=1, tick_ns=0)

    with pytest.raises(NeverThrown):
        env_policy.timeout_config_from_cli_flags()
    with pytest.raises(NeverThrown):
        env_policy.timeout_config_from_cli_flags(ticks=5, tick_ns=None)
    with pytest.raises(NeverThrown):
        env_policy.timeout_config_from_cli_flags(seconds="bad")

    timeout = env_policy.timeout_config_from_cli_flags(seconds="0.25")
    assert timeout.ticks == 250
    assert timeout.tick_ns == 1_000_000
    ms_timeout = env_policy.timeout_config_from_cli_flags(ms=15)
    assert ms_timeout.ticks == 15
    assert ms_timeout.tick_ns == 1_000_000
    with pytest.raises(NeverThrown):
        env_policy.timeout_config_from_cli_flags(seconds="0")
    with pytest.raises(NeverThrown):
        env_policy.timeout_config_from_cli_flags(seconds="0.0001")

    env_policy.apply_cli_timeout_flags(ticks=17, tick_ns=19)
    assert env_policy.lsp_timeout_env_present() is True
    assert env_policy.timeout_ticks_from_env() == (17, 19)
    env_policy.apply_cli_timeout_flags()
    assert env_policy.lsp_timeout_override() is None
    assert env_policy.lsp_timeout_env_present() is False

    assert env_policy.lsp_timeout_override() is None
    with env_policy.lsp_timeout_override_scope(
        env_policy.LspTimeoutConfig(ticks=3, tick_ns=5)
    ):
        assert env_policy.lsp_timeout_override() is not None
    assert env_policy.lsp_timeout_override() is None


# gabion:evidence E:function_site::tests/test_runtime_kernel_contracts.py::test_env_policy_duration_parsing_and_duration_text_edges
def test_env_policy_duration_parsing_and_duration_text_edges() -> None:
    assert env_policy.parse_duration_to_ns("750ms") == 750_000_000
    assert env_policy.parse_duration_to_ns("1m30s") == 90_000_000_000
    assert env_policy.timeout_config_from_duration("2s") == env_policy.LspTimeoutConfig(
        ticks=2_000,
        tick_ns=1_000_000,
    )
    assert env_policy.duration_text_from_ticks(ticks=2, tick_ns=5) == "10ns"

    with pytest.raises(NeverThrown):
        env_policy.parse_duration_to_ns("")
    with pytest.raises(NeverThrown):
        env_policy.parse_duration_to_ns("abc")
    with pytest.raises(NeverThrown):
        env_policy.parse_duration_to_ns("0s")
    with pytest.raises(NeverThrown):
        env_policy.duration_text_from_ticks(ticks=0, tick_ns=1)
    with pytest.raises(NeverThrown):
        env_policy.duration_text_from_ticks(ticks=1, tick_ns=0)


# gabion:evidence E:function_site::tests/test_runtime_kernel_contracts.py::test_env_policy_duration_parser_covers_unit_and_rounding_guardrails
def test_env_policy_duration_parser_covers_unit_and_rounding_guardrails() -> None:
    original_units = dict(env_policy._DURATION_UNIT_NS)
    original_rounding = env_policy.ROUND_CEILING
    try:
        env_policy._DURATION_UNIT_NS.pop("s", None)
        with pytest.raises(NeverThrown):
            env_policy.parse_duration_to_ns("1s")

        env_policy._DURATION_UNIT_NS["s"] = 0  # type: ignore[assignment]
        with pytest.raises(NeverThrown):
            env_policy.parse_duration_to_ns("1s")

        env_policy._DURATION_UNIT_NS["ns"] = 1  # type: ignore[assignment]
        env_policy.ROUND_CEILING = ROUND_FLOOR
        with pytest.raises(NeverThrown):
            env_policy.parse_duration_to_ns("0.1ns")
    finally:
        env_policy._DURATION_UNIT_NS.clear()
        env_policy._DURATION_UNIT_NS.update(original_units)
        env_policy.ROUND_CEILING = original_rounding


# gabion:evidence E:function_site::tests/test_runtime_kernel_contracts.py::test_env_policy_duration_parser_covers_decimal_parse_error_and_total_ns_guard
def test_env_policy_duration_parser_covers_decimal_parse_error_and_total_ns_guard() -> None:
    original_re = env_policy._DURATION_TOKEN_RE
    had_module_int = hasattr(env_policy, "int")
    original_int = getattr(env_policy, "int", int)
    try:
        env_policy._DURATION_TOKEN_RE = re.compile(r"(?P<value>x)(?P<unit>s)")
        with pytest.raises(NeverThrown):
            env_policy.parse_duration_to_ns("xs")

        class _WeirdInt(int):
            def __mul__(self, _other: object) -> int:
                return -1

        env_policy.int = lambda _value: _WeirdInt(1)  # type: ignore[assignment]
        with pytest.raises(NeverThrown):
            env_policy.duration_text_from_ticks(ticks=5, tick_ns=7)
    finally:
        env_policy._DURATION_TOKEN_RE = original_re
        if had_module_int:
            env_policy.int = original_int
        elif hasattr(env_policy, "int"):
            delattr(env_policy, "int")


# gabion:evidence E:call_footprint::tests/test_runtime_kernel_contracts.py::test_deadline_policy_budget_from_override_paths::deadline_policy.py::gabion.runtime.deadline_policy.timeout_budget_from_lsp_env
def test_deadline_policy_budget_from_override_paths() -> None:
    budget = deadline_policy.timeout_budget_from_lsp_env(
        default_budget=deadline_policy.DeadlineBudget(ticks=7, tick_ns=9),
    )
    assert budget.ticks == 7
    assert budget.tick_ns == 9

    with env_policy.lsp_timeout_override_scope(
        env_policy.LspTimeoutConfig(ticks=11, tick_ns=13)
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


# gabion:evidence E:call_footprint::tests/test_runtime_kernel_contracts.py::test_stable_encode_rejects_unsupported_objects::stable_encode.py::gabion.runtime.stable_encode.stable_compact_bytes::stable_encode.py::gabion.runtime.stable_encode.stable_json_value
def test_stable_encode_rejects_unsupported_objects() -> None:
    class _Custom:
        def __str__(self) -> str:
            return "custom-value"

    class _Unmapped:
        pass

    with pytest.raises(TypeError, match="stable_json_value does not support value type _Custom"):
        stable_encode.stable_compact_bytes({"b": 2, "a": _Custom()})
    # Regression: default object repr contains runtime memory identity (e.g. 0x...).
    # Unsupported objects must fail instead of leaking identity text into carriers.
    with pytest.raises(TypeError, match="stable_json_value does not support value type _Unmapped"):
        stable_encode.stable_compact_text({"a": _Unmapped()})


# gabion:evidence E:call_footprint::tests/test_runtime_kernel_contracts.py::test_delta_gate_load_payload_handles_unicode_error::delta_gate.py::gabion.tooling.delta_gate._load_payload
def test_delta_gate_load_payload_handles_unicode_error(tmp_path: Path) -> None:
    invalid_utf = tmp_path / "invalid_utf.json"
    invalid_utf.write_bytes(b"\xff")
    payload, decode_error = delta_gate._load_payload(invalid_utf)
    assert payload is None
    assert decode_error is None


# gabion:evidence E:call_footprint::tests/test_runtime_kernel_contracts.py::test_delta_gate_error_and_ok_branches::delta_gate.py::gabion.tooling.delta_gate._policy_spec::delta_gate.py::gabion.tooling.delta_gate._gate_id_for_env_flag::delta_gate.py::gabion.tooling.delta_gate._check_standard_gate
def test_delta_gate_error_and_ok_branches(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    original_loader = delta_gate.load_governance_rules
    try:
        delta_gate.load_governance_rules = lambda: governance_rules.GovernanceRules(
            override_token_env="TOKEN",
            gates={},
            command_policies={},
            controller_drift=_default_controller_drift_policy(),
        )
        with pytest.raises(ValueError):
            delta_gate._policy_spec("missing_gate")
    finally:
        delta_gate.load_governance_rules = original_loader

    with pytest.raises(ValueError):
        delta_gate._gate_id_for_env_flag("UNSUPPORTED_ENV")

    policy = governance_rules.GatePolicy(
        gate_id="obsolescence_opaque",
        env_flag=delta_gate.OBSOLESCENCE_OPAQUE_ENV_FLAG,
        enabled_mode="default_true",
        delta_keys=("summary", "opaque_evidence", "delta"),
        before_keys=("summary", "opaque_evidence", "baseline"),
        after_keys=("summary", "opaque_evidence", "current"),
        baseline_missing_key=None,
        severity=governance_rules.SeverityPolicy(warning_threshold=2, blocking_threshold=3),
        correction=governance_rules.CorrectionPolicy(
            mode="hard-fail",
            transitions=("advisory->ratchet",),
            bounded_steps=("baseline_write_requires_explicit_flag",),
        ),
        disabled_message="disabled",
        missing_message="missing",
        unreadable_message="unreadable",
        warning_prefix="warn",
        blocking_prefix="block",
        ok_prefix="ok-prefix",
    )
    path = tmp_path / "delta.json"
    path.write_text(
        '{"summary":{"opaque_evidence":{"delta":0,"baseline":1,"current":1}}}\n',
        encoding="utf-8",
    )
    spec = delta_gate._standard_spec_from_policy(policy)
    try:
        delta_gate.load_governance_rules = lambda: governance_rules.GovernanceRules(
            override_token_env="TOKEN",
            gates={"obsolescence_opaque": policy},
            command_policies={},
            controller_drift=_default_controller_drift_policy(),
        )
        assert delta_gate._check_standard_gate(spec, path, enabled=True) == 0
    finally:
        delta_gate.load_governance_rules = original_loader
    assert "ok-prefix" in capsys.readouterr().out


# gabion:evidence E:call_footprint::tests/test_runtime_kernel_contracts.py::test_tool_specs_triplet_map_sorted_and_filtered::tool_specs.py::gabion.tooling.tool_specs.triplet_specs_map::tool_specs.py::gabion.tooling.tool_specs.dataflow_stage_gate_specs
def test_tool_specs_triplet_map_sorted_and_filtered() -> None:
    triplets = tool_specs.triplet_specs_map()
    assert list(triplets.keys()) == sorted(triplets.keys())
    gates = tool_specs.dataflow_stage_gate_specs()
    assert all(spec.include_dataflow_stage_gate for spec in gates)
    assert all(spec.kind == "gate" for spec in gates)
