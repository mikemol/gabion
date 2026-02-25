from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

import pytest

from gabion.commands import command_ids
from gabion.commands import transport_policy
from gabion.exceptions import NeverThrown
from gabion.lsp_client import run_command, run_command_direct


@contextmanager
def _env_scope(updates: dict[str, str | None]):
    originals = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in originals.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# gabion:evidence E:function_site::test_transport_policy.py::tests.test_transport_policy.test_transport_policy_denies_direct_for_governed_without_override
def test_transport_policy_denies_direct_for_governed_without_override() -> None:
    with _env_scope({transport_policy.DIRECT_RUN_ENV: "1", transport_policy.DIRECT_RUN_OVERRIDE_EVIDENCE_ENV: None}):
        with pytest.raises(NeverThrown):
            transport_policy.resolve_command_transport(command=command_ids.CHECK_COMMAND, runner=run_command)


# gabion:evidence E:function_site::test_transport_policy.py::tests.test_transport_policy.test_transport_policy_allows_direct_for_governed_with_valid_override
def test_transport_policy_allows_direct_for_governed_with_valid_override() -> None:
    with _env_scope(
        {
            transport_policy.DIRECT_RUN_ENV: "1",
            transport_policy.DIRECT_RUN_OVERRIDE_EVIDENCE_ENV: "artifact://override/1",
            transport_policy.OVERRIDE_RECORD_JSON_ENV: json.dumps(
                {
                    "actor": "ci",
                    "rationale": "temporary",
                    "scope": "direct_transport",
                    "start": "2024-01-01T00:00:00Z",
                    "expiry": "2999-01-01T00:00:00Z",
                    "rollback_condition": "fix merged",
                    "evidence_links": ["artifact://override/1"],
                }
            ),
        }
    ):
        decision = transport_policy.resolve_command_transport(command=command_ids.CHECK_COMMAND, runner=run_command)
    assert decision.runner is run_command_direct
    assert decision.direct_override_telemetry is not None


# gabion:evidence E:function_site::test_transport_policy.py::tests.test_transport_policy.test_transport_policy_keeps_direct_for_non_governed_command
def test_transport_policy_keeps_direct_for_non_governed_command() -> None:
    with _env_scope({transport_policy.DIRECT_RUN_ENV: "1", transport_policy.DIRECT_RUN_OVERRIDE_EVIDENCE_ENV: None}):
        decision = transport_policy.resolve_command_transport(
            command=command_ids.LSP_PARITY_GATE_COMMAND,
            runner=run_command,
        )
    assert decision.runner is run_command_direct


# gabion:evidence E:function_site::test_transport_policy.py::tests.test_transport_policy.test_transport_policy_unknown_command_without_direct_sets_no_policy
def test_transport_policy_unknown_command_without_direct_sets_no_policy() -> None:
    with _env_scope(
        {
            transport_policy.DIRECT_RUN_ENV: None,
            transport_policy.DIRECT_RUN_OVERRIDE_EVIDENCE_ENV: None,
            transport_policy.OVERRIDE_RECORD_JSON_ENV: None,
        }
    ):
        decision = transport_policy.resolve_command_transport(
            command="gabion.unknown-command",
            runner=run_command,
        )
    assert decision.policy is None
    assert decision.direct_requested is False


# gabion:evidence E:function_site::test_transport_policy.py::tests.test_transport_policy.test_transport_override_scope_prefers_context_over_env
def test_transport_override_scope_prefers_context_over_env() -> None:
    with _env_scope(
        {
            transport_policy.DIRECT_RUN_ENV: "0",
            transport_policy.DIRECT_RUN_OVERRIDE_EVIDENCE_ENV: "env://evidence",
            transport_policy.OVERRIDE_RECORD_JSON_ENV: " {} ",
        }
    ):
        with transport_policy.transport_override_scope(
            transport_policy.TransportOverrideConfig(
                direct_requested=True,
                override_record_json="  {\"actor\":\"ci\"}  ",
            )
        ):
            assert transport_policy.transport_override_present() is True
            direct_requested, record_json = (
                transport_policy._resolve_transport_controls()
            )
            assert direct_requested is True
            assert record_json == "{\"actor\":\"ci\"}"


# gabion:evidence E:function_site::test_transport_policy.py::tests.test_transport_policy.test_apply_cli_transport_flags_normalizes_strings_and_clears_override
def test_apply_cli_transport_flags_normalizes_strings_and_clears_override() -> None:
    transport_policy.apply_cli_transport_flags(
        carrier="lsp",
        override_record_path="  /tmp/override_record.json  ",
    )
    override = transport_policy.transport_override()
    assert override is not None
    assert override.direct_requested is False
    assert override.override_record_path == "/tmp/override_record.json"
    assert override.override_record_json is None
    transport_policy.apply_cli_transport_flags()
    assert transport_policy.transport_override() is None


# gabion:evidence E:function_site::tests/test_transport_policy.py::test_apply_cli_transport_flags_supports_path_only_and_rejects_invalid_carrier
def test_apply_cli_transport_flags_supports_path_only_and_rejects_invalid_carrier() -> None:
    try:
        transport_policy.apply_cli_transport_flags(
            carrier=None,
            override_record_path="/tmp/record.json",
        )
        override = transport_policy.transport_override()
        assert override is not None
        assert override.direct_requested is None
        assert override.override_record_path == "/tmp/record.json"
        with pytest.raises(NeverThrown):
            transport_policy.apply_cli_transport_flags(
                carrier="invalid",
                override_record_path=None,
            )
    finally:
        transport_policy.apply_cli_transport_flags()


# gabion:evidence E:function_site::tests/test_transport_policy.py::test_resolve_transport_controls_reads_override_record_path_and_missing_path_errors
def test_resolve_transport_controls_reads_override_record_path_and_missing_path_errors(
    tmp_path: Path,
) -> None:
    record_path = tmp_path / "override_record.json"
    record_path.write_text(" {\"actor\":\"ci\"} ", encoding="utf-8")
    with transport_policy.transport_override_scope(
        transport_policy.TransportOverrideConfig(
            direct_requested=True,
            override_record_path=str(record_path),
            override_record_json=None,
        )
    ):
        direct_requested, record_json = transport_policy._resolve_transport_controls()
    assert direct_requested is True
    assert record_json == "{\"actor\":\"ci\"}"

    with pytest.raises(NeverThrown):
        transport_policy._load_override_record_json_from_path(
            str(tmp_path / "missing.json")
        )
