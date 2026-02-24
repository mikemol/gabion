from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from gabion.tooling.override_record import (
    _parse_time,
    validate_override_record,
    validate_override_record_file,
    validate_override_record_json,
)


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


def test_validate_override_record_json_and_file_invalid_json_paths(tmp_path: Path) -> None:
    json_result = validate_override_record_json("{not-json")
    assert json_result.valid is False
    assert json_result.reason == "invalid_json"

    bad_file = tmp_path / "override.json"
    bad_file.write_text("{not-json", encoding="utf-8")
    file_result = validate_override_record_file(bad_file)
    assert file_result.valid is False
    assert file_result.reason == "invalid_json"


def test_validate_override_record_additional_validation_edges() -> None:
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    links_empty = _valid_record()
    links_empty["evidence_links"] = []
    assert validate_override_record(links_empty, now=now).reason == "invalid_evidence_links"

    with pytest.raises(ValueError):
        _parse_time(123)
