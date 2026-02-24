# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

try:
    import yaml
except ImportError:  # pragma: no cover - dependency is pinned in pyproject
    yaml = None


@dataclass(frozen=True)
class SeverityPolicy:
    warning_threshold: int
    blocking_threshold: int


@dataclass(frozen=True)
class CorrectionPolicy:
    mode: str
    transitions: tuple[str, ...]
    bounded_steps: tuple[str, ...]


@dataclass(frozen=True)
class GatePolicy:
    gate_id: str
    env_flag: str
    enabled_mode: str
    delta_keys: tuple[str, ...]
    before_keys: tuple[str, ...]
    after_keys: tuple[str, ...]
    baseline_missing_key: tuple[str, ...] | None
    severity: SeverityPolicy
    correction: CorrectionPolicy
    disabled_message: str
    missing_message: str
    unreadable_message: str
    warning_prefix: str
    blocking_prefix: str
    ok_prefix: str


@dataclass(frozen=True)
class GovernanceRules:
    override_token_env: str
    gates: Mapping[str, GatePolicy]
    command_policies: Mapping[str, "CommandPolicy"]


@dataclass(frozen=True)
class CommandPolicy:
    command_id: str
    maturity: str
    require_lsp_carrier: bool
    parity_required: bool
    probe_payload: Mapping[str, object] | None
    parity_ignore_keys: tuple[str, ...]


def _yaml_loader():
    if yaml is None:
        raise RuntimeError("PyYAML is required to load docs/governance_rules.yaml")

    class Loader(yaml.SafeLoader):
        pass

    for key, values in list(Loader.yaml_implicit_resolvers.items()):
        Loader.yaml_implicit_resolvers[key] = [
            (tag, regexp) for tag, regexp in values if tag != "tag:yaml.org,2002:bool"
        ]
    return Loader


def _tuple_path(raw: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        raise ValueError(f"governance_rules invalid {field_name}: expected list[str]")
    return tuple(raw)


def _as_int(raw: object, *, field_name: str) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"governance_rules invalid {field_name}: expected int") from exc


def _gate_from_mapping(gate_id: str, payload: Mapping[str, object]) -> GatePolicy:
    severity_raw = payload.get("severity")
    if not isinstance(severity_raw, Mapping):
        raise ValueError(f"governance_rules invalid gates.{gate_id}.severity")
    correction_raw = payload.get("correction")
    if not isinstance(correction_raw, Mapping):
        raise ValueError(f"governance_rules invalid gates.{gate_id}.correction")

    baseline_missing_value = payload.get("baseline_missing_key")
    baseline_missing_key = None
    if isinstance(baseline_missing_value, list):
        baseline_missing_key = _tuple_path(
            baseline_missing_value,
            field_name=f"gates.{gate_id}.baseline_missing_key",
        )

    mode = correction_raw.get("mode")
    if not isinstance(mode, str):
        raise ValueError(f"governance_rules invalid gates.{gate_id}.correction.mode")

    transitions = correction_raw.get("transitions")
    if not isinstance(transitions, list) or any(not isinstance(item, str) for item in transitions):
        raise ValueError(f"governance_rules invalid gates.{gate_id}.correction.transitions")

    bounded_steps = correction_raw.get("bounded_steps")
    if not isinstance(bounded_steps, list) or any(not isinstance(item, str) for item in bounded_steps):
        raise ValueError(f"governance_rules invalid gates.{gate_id}.correction.bounded_steps")

    return GatePolicy(
        gate_id=gate_id,
        env_flag=str(payload.get("env_flag", "")),
        enabled_mode=str(payload.get("enabled_mode", "default_true")),
        delta_keys=_tuple_path(payload.get("delta_keys"), field_name=f"gates.{gate_id}.delta_keys"),
        before_keys=_tuple_path(payload.get("before_keys"), field_name=f"gates.{gate_id}.before_keys"),
        after_keys=_tuple_path(payload.get("after_keys"), field_name=f"gates.{gate_id}.after_keys"),
        baseline_missing_key=baseline_missing_key,
        severity=SeverityPolicy(
            warning_threshold=_as_int(
                severity_raw.get("warning_threshold"),
                field_name=f"gates.{gate_id}.severity.warning_threshold",
            ),
            blocking_threshold=_as_int(
                severity_raw.get("blocking_threshold"),
                field_name=f"gates.{gate_id}.severity.blocking_threshold",
            ),
        ),
        correction=CorrectionPolicy(
            mode=mode,
            transitions=tuple(transitions),
            bounded_steps=tuple(bounded_steps),
        ),
        disabled_message=str(payload.get("disabled_message", "Gate disabled by policy override.")),
        missing_message=str(payload.get("missing_message", "Delta payload missing; gate failed.")),
        unreadable_message=str(payload.get("unreadable_message", "Delta payload unreadable; gate failed.")),
        warning_prefix=str(payload.get("warning_prefix", "Delta warning")),
        blocking_prefix=str(payload.get("blocking_prefix", "Delta blocking")),
        ok_prefix=str(payload.get("ok_prefix", "Delta OK")),
    )


@lru_cache(maxsize=1)
def load_governance_rules(path: Path | None = None) -> GovernanceRules:
    default_path = Path(__file__).resolve().parents[3] / "docs" / "governance_rules.yaml"
    rule_path = default_path if path is None else path
    loader = _yaml_loader()
    with rule_path.open("r", encoding="utf-8") as handle:
        raw = yaml.load(handle, Loader=loader) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("governance_rules root must be a mapping")

    gates_raw = raw.get("gates")
    if not isinstance(gates_raw, Mapping):
        raise ValueError("governance_rules must define gates")

    gates: dict[str, GatePolicy] = {}
    for gate_id, gate_payload in gates_raw.items():
        if isinstance(gate_id, str) and isinstance(gate_payload, Mapping):
            gates[gate_id] = _gate_from_mapping(gate_id, gate_payload)

    command_policies_raw = raw.get("command_policies")
    command_policies: dict[str, CommandPolicy] = {}
    if isinstance(command_policies_raw, Mapping):
        for command_id, command_payload in command_policies_raw.items():
            if not isinstance(command_id, str) or not isinstance(command_payload, Mapping):
                continue
            maturity = str(command_payload.get("maturity", "experimental"))
            require_lsp_carrier = bool(command_payload.get("require_lsp_carrier", False))
            parity_required = bool(command_payload.get("parity_required", False))
            probe_payload_raw = command_payload.get("probe_payload")
            probe_payload = (
                dict(probe_payload_raw)
                if isinstance(probe_payload_raw, Mapping)
                else None
            )
            parity_ignore_keys_raw = command_payload.get("parity_ignore_keys")
            parity_ignore_keys = (
                tuple(str(item) for item in parity_ignore_keys_raw)
                if isinstance(parity_ignore_keys_raw, list)
                else ()
            )
            command_policies[command_id] = CommandPolicy(
                command_id=command_id,
                maturity=maturity,
                require_lsp_carrier=require_lsp_carrier,
                parity_required=parity_required,
                probe_payload=probe_payload,
                parity_ignore_keys=parity_ignore_keys,
            )

    return GovernanceRules(
        override_token_env=str(raw.get("override_token_env", "GABION_POLICY_OVERRIDE_TOKEN")),
        gates=gates,
        command_policies=command_policies,
    )
