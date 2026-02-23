# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from gabion.runtime import env_policy, json_io

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
    increased_prefix: str
    ok_prefix: str


_OBSOLESCENCE_OPAQUE_SPEC = StandardGateSpec(
    env_flag=OBSOLESCENCE_OPAQUE_ENV_FLAG,
    disabled_message=(
        "Opaque obsolescence gate disabled by override; "
        f"set {OBSOLESCENCE_OPAQUE_ENV_FLAG}=1 to enforce."
    ),
    missing_message="Test obsolescence delta missing; gate failed.",
    unreadable_message="Test obsolescence delta unreadable; gate failed.",
    delta_keys=("summary", "opaque_evidence", "delta"),
    before_keys=("summary", "opaque_evidence", "baseline"),
    after_keys=("summary", "opaque_evidence", "current"),
    increased_prefix="Opaque evidence delta increased",
    ok_prefix="Opaque evidence delta OK",
)

_OBSOLESCENCE_UNMAPPED_SPEC = StandardGateSpec(
    env_flag=OBSOLESCENCE_UNMAPPED_ENV_FLAG,
    disabled_message=(
        "Unmapped delta gate disabled by override; "
        f"set {OBSOLESCENCE_UNMAPPED_ENV_FLAG}=1 to enforce."
    ),
    missing_message="Test obsolescence delta missing; gate failed.",
    unreadable_message="Test obsolescence delta unreadable; gate failed.",
    delta_keys=("summary", "counts", "delta", "unmapped"),
    before_keys=("summary", "counts", "baseline", "unmapped"),
    after_keys=("summary", "counts", "current", "unmapped"),
    increased_prefix="Unmapped evidence delta increased",
    ok_prefix="Unmapped evidence delta OK",
)

_ANNOTATION_ORPHANED_SPEC = StandardGateSpec(
    env_flag=ANNOTATION_ORPHANED_ENV_FLAG,
    disabled_message=(
        "Annotation drift gate disabled by override; "
        f"set {ANNOTATION_ORPHANED_ENV_FLAG}=1 to enforce."
    ),
    missing_message="Annotation drift delta missing; gate failed.",
    unreadable_message="Annotation drift delta unreadable; gate failed.",
    delta_keys=("summary", "delta", "orphaned"),
    before_keys=("summary", "baseline", "orphaned"),
    after_keys=("summary", "current", "orphaned"),
    increased_prefix="Orphaned annotation delta increased",
    ok_prefix="Orphaned annotation delta OK",
)

_AMBIGUITY_SPEC = StandardGateSpec(
    env_flag=AMBIGUITY_DELTA_ENV_FLAG,
    disabled_message=(
        "Ambiguity delta gate disabled by override; "
        f"set {AMBIGUITY_DELTA_ENV_FLAG}=1 to enforce."
    ),
    missing_message="Ambiguity delta missing; gate failed.",
    unreadable_message="Ambiguity delta unreadable; gate failed.",
    delta_keys=("summary", "total", "delta"),
    before_keys=("summary", "total", "baseline"),
    after_keys=("summary", "total", "current"),
    increased_prefix="Ambiguity delta increased",
    ok_prefix="Ambiguity delta OK",
)


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
    delta_value = _nested_int(payload, spec.delta_keys)
    if delta_value > 0:
        before = _nested_int(payload, spec.before_keys)
        after = _nested_int(payload, spec.after_keys)
        print(f"{spec.increased_prefix}: {before} -> {after} (+{delta_value}).")
        return 1
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
    return _nested_int(payload, _OBSOLESCENCE_OPAQUE_SPEC.delta_keys)


def obsolescence_unmapped_delta_value(payload: Mapping[str, object]) -> int:
    return _nested_int(payload, _OBSOLESCENCE_UNMAPPED_SPEC.delta_keys)


def annotation_orphaned_delta_value(payload: Mapping[str, object]) -> int:
    return _nested_int(payload, _ANNOTATION_ORPHANED_SPEC.delta_keys)


def ambiguity_delta_value(payload: Mapping[str, object]) -> int:
    return _nested_int(payload, _AMBIGUITY_SPEC.delta_keys)


def docflow_delta_value(payload: Mapping[str, object], key: str) -> int:
    return _nested_int(payload, ("summary", "delta", key))


def check_obsolescence_opaque_gate(path: Path, *, enabled: bool | None = None) -> int:
    return _check_standard_gate(_OBSOLESCENCE_OPAQUE_SPEC, path, enabled=enabled)


def check_obsolescence_unmapped_gate(path: Path, *, enabled: bool | None = None) -> int:
    return _check_standard_gate(_OBSOLESCENCE_UNMAPPED_SPEC, path, enabled=enabled)


def check_annotation_orphaned_gate(path: Path, *, enabled: bool | None = None) -> int:
    return _check_standard_gate(_ANNOTATION_ORPHANED_SPEC, path, enabled=enabled)


def check_ambiguity_gate(path: Path, *, enabled: bool | None = None) -> int:
    return _check_standard_gate(_AMBIGUITY_SPEC, path, enabled=enabled)


def check_docflow_gate(path: Path, *, enabled: bool | None = None) -> int:
    gate_enabled = docflow_enabled() if enabled is None else enabled
    if not gate_enabled:
        print(f"Docflow delta gate disabled; set {DOCFLOW_DELTA_ENV_FLAG}=1 to enable.")
        return 0
    if not path.exists():
        print("Docflow delta missing; gate skipped.")
        return 0
    payload, decode_error = _load_payload(path)
    if payload is None:
        if isinstance(decode_error, str) and decode_error:
            print(f"Docflow delta unreadable; gate skipped: {decode_error}")
        else:
            print("Docflow delta unreadable; gate skipped.")
        return 0
    baseline_missing = bool(payload.get("baseline_missing"))
    if baseline_missing:
        print("Docflow baseline missing; gate skipped.")
        return 0
    contradicts_delta = docflow_delta_value(payload, "contradicts")
    excess_delta = docflow_delta_value(payload, "excess")
    proposed_delta = docflow_delta_value(payload, "proposed")
    if contradicts_delta > 0:
        before = _nested_int(payload, ("summary", "baseline", "contradicts"))
        after = _nested_int(payload, ("summary", "current", "contradicts"))
        print(
            "Docflow contradictions increased: "
            f"{before} -> {after} (+{contradicts_delta})."
        )
        return 1
    print(
        "Docflow delta OK "
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
