# gabion:decision_protocol_module
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable, Literal, Mapping, TypeAlias
import warnings

from gabion.invariants import never
from gabion.lsp_client import run_command, run_command_direct
from gabion.tooling.governance_rules import CommandPolicy, load_governance_rules
from gabion.schema import TransportSelectionDTO
from gabion.tooling.override_record import validate_override_record_json

Runner: TypeAlias = Callable[..., Mapping[str, object]]

DIRECT_RUN_ENV = "GABION_DIRECT_RUN"
DIRECT_RUN_OVERRIDE_EVIDENCE_ENV = "GABION_DIRECT_RUN_OVERRIDE_EVIDENCE"
OVERRIDE_RECORD_JSON_ENV = "GABION_OVERRIDE_RECORD_JSON"
_LEGACY_TRANSPORT_ENV_WARNED = False


@dataclass(frozen=True)
class TransportOverrideConfig:
    direct_requested: bool | None = None
    override_record_path: str | None = None
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


TransportCarrierMode = Literal["auto", "lsp", "direct"]


@dataclass(frozen=True)
class TransportCarrierDecision:
    mode: TransportCarrierMode

    @classmethod
    def from_carrier(cls, carrier: str | None) -> "TransportCarrierDecision":
        if carrier is None:
            return cls(mode="auto")
        carrier_text = carrier.strip().lower()
        if carrier_text not in {"lsp", "direct"}:
            never("invalid transport carrier", carrier=carrier)
        return cls(mode=carrier_text)

    def to_direct_requested(self) -> bool | None:
        if self.mode == "auto":
            return None
        return self.mode == "direct"


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
    carrier: str | None = None,
    override_record_path: str | None = None,
) -> None:
    if (
        carrier is None
        and override_record_path is None
    ):
        set_transport_override(None)
        return
    carrier_text = carrier.strip().lower() if isinstance(carrier, str) and carrier.strip() else None
    carrier_decision = TransportCarrierDecision.from_carrier(carrier_text)
    selection = TransportSelectionDTO(
        carrier=carrier_decision.mode,
        carrier_override_record=(
            override_record_path.strip()
            if isinstance(override_record_path, str) and override_record_path.strip()
            else None
        ),
    )
    set_transport_override(
        TransportOverrideConfig(
            direct_requested=carrier_decision.to_direct_requested(),
            override_record_path=selection.carrier_override_record,
            override_record_json=None,
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
            "Use CLI flags (--carrier, --carrier-override-record) instead."
        ),
        DeprecationWarning,
        stacklevel=3,
    )


def _load_override_record_json_from_path(path_text: str) -> str:
    override_path = Path(path_text)
    if not override_path.exists():
        never("transport override record path not found", path=str(override_path))
    return override_path.read_text(encoding="utf-8")


def _resolve_transport_controls() -> tuple[bool, str | None]:
    override = transport_override()
    if override is not None:
        direct_requested = bool(override.direct_requested)
        record_json = override.override_record_json
        if (
            (record_json is None or not record_json.strip())
            and isinstance(override.override_record_path, str)
            and override.override_record_path.strip()
        ):
            record_json = _load_override_record_json_from_path(
                override.override_record_path.strip()
            )
        return (
            direct_requested,
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
    return (direct_requested, override_record_json)


# gabion:decision_protocol
def resolve_command_transport(*, command: str, runner: Runner) -> CommandTransportDecision:
    direct_requested, override_record_json = (
        _resolve_transport_controls()
    )
    if runner is not run_command:
        return CommandTransportDecision(
            runner=runner,
            direct_requested=direct_requested,
            direct_override_evidence=None,
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
        direct_override_evidence=None,
        direct_override_telemetry=override_telemetry,
        policy=policy,
    )
