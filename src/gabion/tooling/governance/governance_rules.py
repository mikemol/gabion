from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, singledispatch
from importlib import import_module
from pathlib import Path
from typing import Mapping

from gabion.invariants import never


def _load_yaml_module(*, importer=import_module):
    try:
        module = importer("yaml")
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required by the pinned governance toolchain; run `mise install` to provision dependencies."
        ) from exc
    return module


yaml = _load_yaml_module()


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
    controller_drift: "ControllerDriftPolicy"


@dataclass(frozen=True)
class CommandPolicy:
    command_id: str
    maturity: str
    require_lsp_carrier: bool
    parity_required: bool
    probe_payload: Mapping[str, object] | None
    parity_ignore_keys: tuple[str, ...]


@dataclass(frozen=True)
class ControllerDriftPolicy:
    severity_classes: tuple[str, ...]
    enforce_at_or_above: str
    remediation_by_severity: Mapping[str, str]
    consecutive_passes_required: int


def _yaml_loader():
    class Loader(yaml.SafeLoader):
        pass

    for key, values in list(Loader.yaml_implicit_resolvers.items()):
        Loader.yaml_implicit_resolvers[key] = [
            (tag, regexp) for tag, regexp in values if tag != "tag:yaml.org,2002:bool"
        ]
    return Loader


@singledispatch
def _str_optional(raw: object) -> str | None:
    never("unregistered runtime type", value_type=type(raw).__name__)


@_str_optional.register(str)
def _sd_reg_1(raw: str) -> str | None:
    return raw


def _none_str(raw: object) -> str | None:
    _ = raw
    return None


for _str_none_type in (dict, list, tuple, set, int, float, bool, type(None)):
    _str_optional.register(_str_none_type)(_none_str)


@singledispatch
def _dict_optional(raw: object) -> dict[object, object] | None:
    never("unregistered runtime type", value_type=type(raw).__name__)


@_dict_optional.register(dict)
def _sd_reg_2(raw: dict[object, object]) -> dict[object, object] | None:
    return raw


def _none_dict(raw: object) -> dict[object, object] | None:
    _ = raw
    return None


for _dict_none_type in (list, tuple, set, str, int, float, bool, type(None)):
    _dict_optional.register(_dict_none_type)(_none_dict)


@singledispatch
def _list_optional(raw: object) -> list[object] | None:
    never("unregistered runtime type", value_type=type(raw).__name__)


@_list_optional.register(list)
def _sd_reg_3(raw: list[object]) -> list[object] | None:
    return raw


def _none_list(raw: object) -> list[object] | None:
    _ = raw
    return None


for _list_none_type in (dict, tuple, set, str, int, float, bool, type(None)):
    _list_optional.register(_list_none_type)(_none_list)


def _required_mapping(raw: object, *, error_message: str) -> dict[object, object]:
    mapping = _dict_optional(raw)
    if mapping is None:
        raise ValueError(error_message)
    return mapping


def _required_str(raw: object, *, error_message: str) -> str:
    value = _str_optional(raw)
    if value is None:
        raise ValueError(error_message)
    return value


def _tuple_path(raw: object, *, field_name: str) -> tuple[str, ...]:
    values = _list_optional(raw)
    if values is None:
        raise ValueError(f"governance_rules invalid {field_name}: expected list[str]")
    typed: list[str] = []
    for item in values:
        text = _str_optional(item)
        if text is None:
            raise ValueError(f"governance_rules invalid {field_name}: expected list[str]")
        typed.append(text)
    return tuple(typed)


def _as_int(raw: object, *, field_name: str) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"governance_rules invalid {field_name}: expected int") from exc


@singledispatch
def _as_bool(raw: object, *, field_name: str) -> bool:
    never("unregistered runtime type", value_type=type(raw).__name__)


@_as_bool.register(bool)
def _sd_reg_4(raw: bool, *, field_name: str) -> bool:
    _ = field_name
    return raw


@_as_bool.register(str)
def _sd_reg_5(raw: str, *, field_name: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"governance_rules invalid {field_name}: expected bool")


def _invalid_bool(raw: object, *, field_name: str) -> bool:
    _ = raw
    raise ValueError(f"governance_rules invalid {field_name}: expected bool")


for _bool_invalid_type in (dict, list, tuple, set, int, float, type(None)):
    _as_bool.register(_bool_invalid_type)(_invalid_bool)


@singledispatch
def _optional_tuple_path(raw: object, *, field_name: str) -> tuple[str, ...] | None:
    never("unregistered runtime type", value_type=type(raw).__name__)


@_optional_tuple_path.register(list)
def _sd_reg_6(raw: list[object], *, field_name: str) -> tuple[str, ...] | None:
    return _tuple_path(raw, field_name=field_name)


def _none_tuple_path(raw: object, *, field_name: str) -> tuple[str, ...] | None:
    _ = (raw, field_name)
    return None


for _tuple_path_none_type in (dict, tuple, set, str, int, float, bool, type(None)):
    _optional_tuple_path.register(_tuple_path_none_type)(_none_tuple_path)


@singledispatch
def _string_key_mapping(raw: object) -> dict[str, object]:
    never("unregistered runtime type", value_type=type(raw).__name__)


@_string_key_mapping.register(dict)
def _sd_reg_7(raw: dict[object, object]) -> dict[str, object]:
    mapped: dict[str, object] = {}
    for key, value in raw.items():
        key_text = _str_optional(key)
        if key_text is not None:
            mapped[key_text] = value
    return mapped


@singledispatch
def _string_tuple_default_empty(raw: object) -> tuple[str, ...]:
    never("unregistered runtime type", value_type=type(raw).__name__)


@_string_tuple_default_empty.register(list)
def _sd_reg_8(raw: list[object]) -> tuple[str, ...]:
    return tuple(str(item) for item in raw)


def _empty_string_tuple(raw: object) -> tuple[str, ...]:
    _ = raw
    return ()


for _string_tuple_empty_type in (dict, tuple, set, str, int, float, bool, type(None)):
    _string_tuple_default_empty.register(_string_tuple_empty_type)(_empty_string_tuple)


def _gate_from_mapping(gate_id: str, payload: Mapping[str, object]) -> GatePolicy:
    severity_raw = _required_mapping(
        payload.get("severity"),
        error_message=f"governance_rules invalid gates.{gate_id}.severity",
    )
    correction_raw = _required_mapping(
        payload.get("correction"),
        error_message=f"governance_rules invalid gates.{gate_id}.correction",
    )

    baseline_missing_value = payload.get("baseline_missing_key")
    baseline_missing_key = _optional_tuple_path(
        baseline_missing_value,
        field_name=f"gates.{gate_id}.baseline_missing_key",
    )

    mode = _required_str(
        correction_raw.get("mode"),
        error_message=f"governance_rules invalid gates.{gate_id}.correction.mode",
    )
    transitions = _tuple_path(
        correction_raw.get("transitions"),
        field_name=f"gates.{gate_id}.correction.transitions",
    )
    bounded_steps = _tuple_path(
        correction_raw.get("bounded_steps"),
        field_name=f"gates.{gate_id}.correction.bounded_steps",
    )

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
            transitions=transitions,
            bounded_steps=bounded_steps,
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
    default_path = Path(__file__).resolve().parents[4] / "docs" / "governance_rules.yaml"
    rule_path = default_path if path is None else path
    loader = _yaml_loader()
    with rule_path.open("r", encoding="utf-8") as handle:
        raw = yaml.load(handle, Loader=loader) or {}
    raw_mapping = _dict_optional(raw)
    if raw_mapping is None:
        raise ValueError("governance_rules root must be a mapping")

    gates_raw = _dict_optional(raw_mapping.get("gates"))
    if gates_raw is None:
        raise ValueError("governance_rules must define gates")

    gates: dict[str, GatePolicy] = {}
    for gate_id, gate_payload in gates_raw.items():
        gate_name = _str_optional(gate_id)
        gate_mapping = _dict_optional(gate_payload)
        if gate_name is not None and gate_mapping is not None:
            gates[gate_name] = _gate_from_mapping(gate_name, gate_mapping)

    command_policies_raw = _dict_optional(raw_mapping.get("command_policies"))
    command_policies: dict[str, CommandPolicy] = {}
    if command_policies_raw is not None:
        for command_id, command_payload in command_policies_raw.items():
            command_name = _str_optional(command_id)
            command_mapping = _dict_optional(command_payload)
            if command_name is not None and command_mapping is not None:
                maturity = str(command_mapping.get("maturity", "experimental"))
                require_lsp_carrier = _as_bool(
                    command_mapping.get("require_lsp_carrier", False),
                    field_name=f"command_policies.{command_name}.require_lsp_carrier",
                )
                parity_required = _as_bool(
                    command_mapping.get("parity_required", False),
                    field_name=f"command_policies.{command_name}.parity_required",
                )
                probe_payload = _dict_optional(command_mapping.get("probe_payload"))
                parity_ignore_keys = _string_tuple_default_empty(
                    command_mapping.get("parity_ignore_keys")
                )
                command_policies[command_name] = CommandPolicy(
                    command_id=command_name,
                    maturity=maturity,
                    require_lsp_carrier=require_lsp_carrier,
                    parity_required=parity_required,
                    probe_payload=probe_payload,
                    parity_ignore_keys=parity_ignore_keys,
                )

    controller_drift_raw = _dict_optional(raw_mapping.get("controller_drift"))
    if controller_drift_raw is None:
        raise ValueError("governance_rules must define controller_drift")
    severity_classes = _tuple_path(
        controller_drift_raw.get("severity_classes"),
        field_name="controller_drift.severity_classes",
    )
    remediation_raw = _dict_optional(controller_drift_raw.get("remediation_by_severity"))
    if remediation_raw is None:
        raise ValueError("governance_rules invalid controller_drift.remediation_by_severity")
    remediation_by_severity = _string_key_mapping(remediation_raw)
    controller_drift = ControllerDriftPolicy(
        severity_classes=severity_classes,
        enforce_at_or_above=str(controller_drift_raw.get("enforce_at_or_above", "high")),
        remediation_by_severity=remediation_by_severity,
        consecutive_passes_required=_as_int(
            controller_drift_raw.get("consecutive_passes_required", 3),
            field_name="controller_drift.consecutive_passes_required",
        ),
    )

    return GovernanceRules(
        override_token_env=str(raw_mapping.get("override_token_env", "GABION_POLICY_OVERRIDE_TOKEN")),
        gates=gates,
        command_policies=command_policies,
        controller_drift=controller_drift,
    )
