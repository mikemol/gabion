from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping

REQUIRED_OVERRIDE_FIELDS: tuple[str, ...] = (
    "actor",
    "rationale",
    "scope",
    "start",
    "expiry",
    "rollback_condition",
    "evidence_links",
)


@dataclass(frozen=True)
class OverrideValidationResult:
    active: bool
    valid: bool
    reason: str | None
    record: Mapping[str, object] | None

    def telemetry(self, *, source: str) -> dict[str, object]:
        payload: dict[str, object] = {
            "override_active": self.active,
            "override_valid": self.valid,
            "override_source": source,
            "override_reason": self.reason,
        }
        if self.record is None:
            return payload
        payload["override_actor"] = self.record.get("actor")
        payload["override_scope"] = self.record.get("scope")
        payload["override_expiry"] = self.record.get("expiry")
        return payload


def _parse_time(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be a string")
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def validate_override_record(
    raw_record: object,
    *,
    now: datetime | None = None,
) -> OverrideValidationResult:
    if raw_record is None:
        return OverrideValidationResult(active=False, valid=False, reason="missing", record=None)
    if not isinstance(raw_record, Mapping):
        return OverrideValidationResult(active=True, valid=False, reason="non_mapping", record=None)
    record = dict(raw_record)
    missing = [field for field in REQUIRED_OVERRIDE_FIELDS if field not in record]
    if missing:
        return OverrideValidationResult(active=True, valid=False, reason="missing_fields", record=record)
    try:
        expiry = _parse_time(record["expiry"])
        _parse_time(record["start"])
    except ValueError:
        return OverrideValidationResult(active=True, valid=False, reason="invalid_timestamp", record=record)
    now_utc = now or datetime.now(timezone.utc)
    if expiry <= now_utc:
        return OverrideValidationResult(active=True, valid=False, reason="expired", record=record)
    return OverrideValidationResult(active=True, valid=True, reason=None, record=record)


def validate_override_record_json(raw_json: str | None) -> OverrideValidationResult:
    stripped = (raw_json or "").strip()
    if not stripped:
        return OverrideValidationResult(active=False, valid=False, reason="missing", record=None)
    try:
        decoded = json.loads(stripped)
    except json.JSONDecodeError:
        return OverrideValidationResult(active=True, valid=False, reason="invalid_json", record=None)
    return validate_override_record(decoded)


def validate_override_record_file(path: Path) -> OverrideValidationResult:
    if not path.exists():
        return OverrideValidationResult(active=False, valid=False, reason="missing", record=None)
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return OverrideValidationResult(active=True, valid=False, reason="invalid_json", record=None)
    return validate_override_record(decoded)

