# gabion:decision_protocol_module
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
import os
from typing import Callable, Mapping, TypeAlias
import warnings

from gabion.invariants import never
from gabion.lsp_client import run_command, run_command_direct
from gabion.tooling.governance_rules import CommandPolicy, load_governance_rules
from gabion.tooling.override_record import validate_override_record_json

Runner: TypeAlias = Callable[..., Mapping[str, object]]

DIRECT_RUN_ENV = "GABION_DIRECT_RUN"
DIRECT_RUN_OVERRIDE_EVIDENCE_ENV = "GABION_DIRECT_RUN_OVERRIDE_EVIDENCE"
OVERRIDE_RECORD_JSON_ENV = "GABION_OVERRIDE_RECORD_JSON"
_LEGACY_TRANSPORT_ENV_WARNED = False


@dataclass(frozen=True)
class TransportOverrideConfig:
    direct_requested: bool | None = None
    direct_override_evidence: str | None = None
    override_record_json: str | None = None


_TRANSPORT_OVERRIDE: ContextVar[TransportOverrideConfig | None] = ContextVar(
    "gabion_transport_override",
    default=None,
)


@dataclass(frozen=True)
class CommandTransportDecision:
    runner: Runner
    direct_requested: bool
    direct_override_evidence: str | None
    direct_override_telemetry: Mapping[str, object] | None
    policy: CommandPolicy | None


def transport_override() -> TransportOverrideConfig | None:
    return _TRANSPORT_OVERRIDE.get()


def transport_override_present() -> bool:
    if transport_override() is not None:
        return True
    return (
        DIRECT_RUN_ENV in os.environ
        or DIRECT_RUN_OVERRIDE_EVIDENCE_ENV in os.environ
        or OVERRIDE_RECORD_JSON_ENV in os.environ
    )


def set_transport_override(
    override: TransportOverrideConfig | None,
) -> Token[TransportOverrideConfig | None]:
    return _TRANSPORT_OVERRIDE.set(override)


def reset_transport_override(
    token: Token[TransportOverrideConfig | None],
) -> None:
    _TRANSPORT_OVERRIDE.reset(token)


@contextmanager
def transport_override_scope(override: TransportOverrideConfig | None):
    token = set_transport_override(override)
    try:
        yield
    finally:
        reset_transport_override(token)


def apply_cli_transport_flags(
    *,
    direct_requested: bool | None = None,
    direct_override_evidence: str | None = None,
    override_record_json: str | None = None,
) -> None:
    if (
        direct_requested is None
        and direct_override_evidence is None
        and override_record_json is None
    ):
        set_transport_override(None)
        return
    set_transport_override(
        TransportOverrideConfig(
            direct_requested=direct_requested,
            direct_override_evidence=(
                direct_override_evidence.strip()
                if isinstance(direct_override_evidence, str)
                else None
            )
            or None,
            override_record_json=(
                override_record_json.strip()
                if isinstance(override_record_json, str)
                else None
            )
            or None,
        )
    )


def _warn_legacy_transport_env_usage() -> None:
    global _LEGACY_TRANSPORT_ENV_WARNED
    if _LEGACY_TRANSPORT_ENV_WARNED:
        return
    _LEGACY_TRANSPORT_ENV_WARNED = True
    warnings.warn(
        (
            "Legacy transport env overrides (GABION_DIRECT_RUN*, "
            "GABION_OVERRIDE_RECORD_JSON) are deprecated. "
            "Use CLI flags (--transport, --direct-run-override-evidence, "
            "--override-record-json) instead."
        ),
        DeprecationWarning,
        stacklevel=3,
    )


def _resolve_transport_controls() -> tuple[bool, str | None, str | None]:
    override = transport_override()
    if override is not None:
        direct_requested = bool(override.direct_requested)
        evidence = override.direct_override_evidence
        record_json = override.override_record_json
        return (
            direct_requested,
            evidence.strip() if isinstance(evidence, str) else None,
            record_json.strip() if isinstance(record_json, str) else None,
        )
    direct_flag = os.getenv(DIRECT_RUN_ENV, "").strip().lower()
    direct_requested = direct_flag in {"1", "true", "yes", "on"}
    override_evidence = os.getenv(DIRECT_RUN_OVERRIDE_EVIDENCE_ENV, "").strip() or None
    override_record_json = os.getenv(OVERRIDE_RECORD_JSON_ENV)
    if direct_flag or override_evidence or (
        isinstance(override_record_json, str) and override_record_json.strip()
    ):
        _warn_legacy_transport_env_usage()
    return (direct_requested, override_evidence, override_record_json)


# gabion:decision_protocol
def resolve_command_transport(*, command: str, runner: Runner) -> CommandTransportDecision:
    direct_requested, override_evidence, override_record_json = (
        _resolve_transport_controls()
    )
    if runner is not run_command:
        return CommandTransportDecision(
            runner=runner,
            direct_requested=direct_requested,
            direct_override_evidence=override_evidence,
            direct_override_telemetry=None,
            policy=None,
        )

    policy = load_governance_rules().command_policies.get(command)
    require_lsp_carrier = False
    if policy is not None:
        require_lsp_carrier = policy.require_lsp_carrier or policy.maturity in {
            "beta",
            "production",
        }

    override_record = validate_override_record_json(override_record_json)
    override_telemetry = None
    if direct_requested and require_lsp_carrier:
        if override_evidence is None:
            never(
                "direct transport forbidden by command maturity policy",
                command=command,
                maturity=(policy.maturity if policy is not None else "unknown"),
                override_evidence_env=DIRECT_RUN_OVERRIDE_EVIDENCE_ENV,
            )
        if not override_record.valid:
            never(
                "direct transport override record invalid",
                command=command,
                maturity=(policy.maturity if policy is not None else "unknown"),
                override_record_env=OVERRIDE_RECORD_JSON_ENV,
                **override_record.telemetry(source="command_transport"),
            )
        override_telemetry = override_record.telemetry(source="command_transport")
    resolved_runner = run_command_direct if direct_requested else run_command
    return CommandTransportDecision(
        runner=resolved_runner,
        direct_requested=direct_requested,
        direct_override_evidence=override_evidence,
        direct_override_telemetry=override_telemetry,
        policy=policy,
    )
