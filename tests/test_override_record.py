from __future__ import annotations

from datetime import datetime, timezone

from gabion.tooling.override_record import validate_override_record


def _valid_record() -> dict[str, object]:
    return {
        "actor": "ci",
        "rationale": "temporary",
        "scope": "direct_transport",
        "start": "2024-01-01T00:00:00Z",
        "expiry": "2999-01-01T00:00:00Z",
        "rollback_condition": "fix merged",
        "evidence_links": ["artifact://override/1"],
    }


def test_validate_override_record_reason_codes() -> None:
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert validate_override_record(None, now=now).reason == "missing"
    assert validate_override_record([], now=now).reason == "non_mapping"
    assert validate_override_record({"actor": "ci"}, now=now).reason == "missing_fields"

    empty = _valid_record()
    empty["rationale"] = "   "
    assert validate_override_record(empty, now=now).reason == "empty_field"

    bad_links = _valid_record()
    bad_links["evidence_links"] = ["", "artifact://ok"]
    assert validate_override_record(bad_links, now=now).reason == "invalid_evidence_links"

    bad_ts = _valid_record()
    bad_ts["start"] = "not-a-time"
    assert validate_override_record(bad_ts, now=now).reason == "invalid_timestamp"

    bad_interval = _valid_record()
    bad_interval["start"] = "2025-02-01T00:00:00Z"
    bad_interval["expiry"] = "2025-01-01T00:00:00Z"
    assert validate_override_record(bad_interval, now=now).reason == "invalid_interval"

    expired = _valid_record()
    expired["expiry"] = "2024-12-31T00:00:00Z"
    assert validate_override_record(expired, now=now).reason == "expired"


def test_validate_override_record_valid_path() -> None:
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    result = validate_override_record(_valid_record(), now=now)
    assert result.valid is True
    assert result.reason is None
    telemetry = result.telemetry(source="test")
    assert telemetry["override_valid"] is True
    assert telemetry["override_source"] == "test"
