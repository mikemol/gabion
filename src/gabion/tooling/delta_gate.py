# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from gabion.runtime import env_policy, json_io
from gabion.tooling.governance_rules import GatePolicy, load_governance_rules

OBSOLESCENCE_OPAQUE_ENV_FLAG = "GABION_GATE_OPAQUE_DELTA"
OBSOLESCENCE_UNMAPPED_ENV_FLAG = "GABION_GATE_UNMAPPED_DELTA"
ANNOTATION_ORPHANED_ENV_FLAG = "GABION_GATE_ORPHANED_DELTA"
AMBIGUITY_DELTA_ENV_FLAG = "GABION_GATE_AMBIGUITY_DELTA"
DOCFLOW_DELTA_ENV_FLAG = "GABION_GATE_DOCFLOW_DELTA"

_DEFAULT_FALSE_VALUES = {"0", "false", "no", "off"}
_TRUTHY_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class StandardGateSpec:
    env_flag: str
    disabled_message: str
    missing_message: str
    unreadable_message: str
    delta_keys: tuple[str, ...]
    before_keys: tuple[str, ...]
    after_keys: tuple[str, ...]
    warning_prefix: str
    blocking_prefix: str
    ok_prefix: str


def _standard_spec_from_policy(policy: GatePolicy) -> StandardGateSpec:
    return StandardGateSpec(
        env_flag=policy.env_flag,
        disabled_message=policy.disabled_message,
        missing_message=policy.missing_message,
        unreadable_message=policy.unreadable_message,
        delta_keys=policy.delta_keys,
        before_keys=policy.before_keys,
        after_keys=policy.after_keys,
        warning_prefix=policy.warning_prefix,
        blocking_prefix=policy.blocking_prefix,
        ok_prefix=policy.ok_prefix,
    )


def _policy_spec(gate_id: str) -> StandardGateSpec:
    policy = load_governance_rules().gates.get(gate_id)
    if policy is None:
        raise ValueError(f"governance policy missing gate: {gate_id}")
    return _standard_spec_from_policy(policy)


def _enabled_default_true(env_flag: str, value: str | None = None) -> bool:
    if value is None:
        return env_policy.env_enabled_default_true(env_flag)
    return value.strip().lower() not in _DEFAULT_FALSE_VALUES


def _enabled_truthy_only(env_flag: str, value: str | None = None) -> bool:
    if value is None:
        return env_policy.env_enabled_truthy_only(env_flag)
    return value.strip().lower() in _TRUTHY_VALUES


def _nested_int(payload: Mapping[str, object], keys: tuple[str, ...]) -> int:
    node: object = payload
    for key in keys:
        if not isinstance(node, Mapping):
            return 0
        node = node.get(key)
    try:
        return int(node if node is not None else 0)
    except (TypeError, ValueError):
        return 0


def _load_payload(path: Path) -> tuple[Mapping[str, object] | None, str | None]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None, None
    payload = json_io.load_json_object_text(text)
    if payload:
        return payload, None
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, str(exc)
    if isinstance(raw, Mapping):
        return {str(key): raw[key] for key in raw}, None
    return None, None


def _gate_id_for_env_flag(env_flag: str) -> str:
    mapping = {
        OBSOLESCENCE_OPAQUE_ENV_FLAG: "obsolescence_opaque",
        OBSOLESCENCE_UNMAPPED_ENV_FLAG: "obsolescence_unmapped",
        ANNOTATION_ORPHANED_ENV_FLAG: "annotation_orphaned",
        AMBIGUITY_DELTA_ENV_FLAG: "ambiguity",
        DOCFLOW_DELTA_ENV_FLAG: "docflow",
    }
    gate_id = mapping.get(env_flag)
    if gate_id is None:
        raise ValueError(f"unsupported env flag for governance mapping: {env_flag}")
    return gate_id


def _check_standard_gate(
    spec: StandardGateSpec,
    path: Path,
    *,
    enabled: bool | None,
) -> int:
    gate_enabled = _enabled_default_true(spec.env_flag) if enabled is None else enabled
    if not gate_enabled:
        print(spec.disabled_message)
        return 0
    if not path.exists():
        print(spec.missing_message)
        return 2
    payload, decode_error = _load_payload(path)
    if payload is None:
        if isinstance(decode_error, str) and decode_error:
            print(f"{spec.unreadable_message}: {decode_error}")
        else:
            print(spec.unreadable_message)
        return 2
    gate_policy = load_governance_rules().gates[_gate_id_for_env_flag(spec.env_flag)]
    delta_value = _nested_int(payload, spec.delta_keys)
    if delta_value >= gate_policy.severity.blocking_threshold:
        before = _nested_int(payload, spec.before_keys)
        after = _nested_int(payload, spec.after_keys)
        print(f"{spec.blocking_prefix}: {before} -> {after} (+{delta_value}).")
        return 1
    if delta_value > gate_policy.severity.warning_threshold:
        before = _nested_int(payload, spec.before_keys)
        after = _nested_int(payload, spec.after_keys)
        print(f"{spec.warning_prefix}: {before} -> {after} (+{delta_value}).")
        return 0
    print(f"{spec.ok_prefix} ({delta_value}).")
    return 0


def obsolescence_opaque_enabled(value: str | None = None) -> bool:
    return _enabled_default_true(OBSOLESCENCE_OPAQUE_ENV_FLAG, value)


def obsolescence_unmapped_enabled(value: str | None = None) -> bool:
    return _enabled_default_true(OBSOLESCENCE_UNMAPPED_ENV_FLAG, value)


def annotation_orphaned_enabled(value: str | None = None) -> bool:
    return _enabled_default_true(ANNOTATION_ORPHANED_ENV_FLAG, value)


def ambiguity_enabled(value: str | None = None) -> bool:
    return _enabled_default_true(AMBIGUITY_DELTA_ENV_FLAG, value)


def docflow_enabled(value: str | None = None) -> bool:
    return _enabled_truthy_only(DOCFLOW_DELTA_ENV_FLAG, value)


def obsolescence_opaque_delta_value(payload: Mapping[str, object]) -> int:
    return _nested_int(payload, _policy_spec("obsolescence_opaque").delta_keys)


def obsolescence_unmapped_delta_value(payload: Mapping[str, object]) -> int:
    return _nested_int(payload, _policy_spec("obsolescence_unmapped").delta_keys)


def annotation_orphaned_delta_value(payload: Mapping[str, object]) -> int:
    return _nested_int(payload, _policy_spec("annotation_orphaned").delta_keys)


def ambiguity_delta_value(payload: Mapping[str, object]) -> int:
    return _nested_int(payload, _policy_spec("ambiguity").delta_keys)


def docflow_delta_value(payload: Mapping[str, object], key: str) -> int:
    return _nested_int(payload, ("summary", "delta", key))


def check_obsolescence_opaque_gate(path: Path, *, enabled: bool | None = None) -> int:
    return _check_standard_gate(_policy_spec("obsolescence_opaque"), path, enabled=enabled)


def check_obsolescence_unmapped_gate(path: Path, *, enabled: bool | None = None) -> int:
    return _check_standard_gate(_policy_spec("obsolescence_unmapped"), path, enabled=enabled)


def check_annotation_orphaned_gate(path: Path, *, enabled: bool | None = None) -> int:
    return _check_standard_gate(_policy_spec("annotation_orphaned"), path, enabled=enabled)


def check_ambiguity_gate(path: Path, *, enabled: bool | None = None) -> int:
    return _check_standard_gate(_policy_spec("ambiguity"), path, enabled=enabled)


def check_docflow_gate(path: Path, *, enabled: bool | None = None) -> int:
    policy = load_governance_rules().gates["docflow"]
    gate_enabled = docflow_enabled() if enabled is None else enabled
    if not gate_enabled:
        print(policy.disabled_message)
        return 0
    if not path.exists():
        print(policy.missing_message)
        return 0
    payload, decode_error = _load_payload(path)
    if payload is None:
        if isinstance(decode_error, str) and decode_error:
            print(f"{policy.unreadable_message}: {decode_error}")
        else:
            print(policy.unreadable_message)
        return 0
    baseline_missing = bool(payload.get("baseline_missing"))
    if baseline_missing:
        print("Docflow baseline missing; gate skipped.")
        return 0
    contradicts_delta = docflow_delta_value(payload, "contradicts")
    excess_delta = docflow_delta_value(payload, "excess")
    proposed_delta = docflow_delta_value(payload, "proposed")
    if contradicts_delta >= policy.severity.blocking_threshold:
        before = _nested_int(payload, ("summary", "baseline", "contradicts"))
        after = _nested_int(payload, ("summary", "current", "contradicts"))
        print(
            f"{policy.blocking_prefix}: "
            f"{before} -> {after} (+{contradicts_delta})."
        )
        return 1
    print(
        f"{policy.ok_prefix} "
        f"(contradicts {contradicts_delta}, excess {excess_delta}, proposed {proposed_delta})."
    )
    return 0


def obsolescence_opaque_main() -> int:
    return check_obsolescence_opaque_gate(Path("artifacts/out/test_obsolescence_delta.json"))


def obsolescence_unmapped_main() -> int:
    return check_obsolescence_unmapped_gate(Path("artifacts/out/test_obsolescence_delta.json"))


def annotation_orphaned_main() -> int:
    return check_annotation_orphaned_gate(Path("artifacts/out/test_annotation_drift_delta.json"))


def ambiguity_main() -> int:
    return check_ambiguity_gate(Path("artifacts/out/ambiguity_delta.json"))


def docflow_main() -> int:
    return check_docflow_gate(Path("artifacts/out/docflow_compliance_delta.json"))
