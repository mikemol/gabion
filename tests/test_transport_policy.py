from __future__ import annotations

import json
import os
from contextlib import contextmanager

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


def test_transport_policy_denies_direct_for_governed_without_override() -> None:
    with _env_scope({transport_policy.DIRECT_RUN_ENV: "1", transport_policy.DIRECT_RUN_OVERRIDE_EVIDENCE_ENV: None}):
        with pytest.raises(NeverThrown):
            transport_policy.resolve_command_transport(command=command_ids.CHECK_COMMAND, runner=run_command)


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


def test_transport_policy_keeps_direct_for_non_governed_command() -> None:
    with _env_scope({transport_policy.DIRECT_RUN_ENV: "1", transport_policy.DIRECT_RUN_OVERRIDE_EVIDENCE_ENV: None}):
        decision = transport_policy.resolve_command_transport(
            command=command_ids.LSP_PARITY_GATE_COMMAND,
            runner=run_command,
        )
    assert decision.runner is run_command_direct
