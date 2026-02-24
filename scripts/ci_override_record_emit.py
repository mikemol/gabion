#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

REQUIRED = {"actor", "rationale", "scope", "start", "expiry", "rollback_condition", "evidence_links"}


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/out/governance_override_record.json"))
    args = parser.parse_args()

    override_channels = {
        "direct_transport": os.getenv("GABION_DIRECT_RUN_OVERRIDE_EVIDENCE", "").strip(),
        "controller_drift": os.getenv("CONTROLLER_DRIFT_OVERRIDE", "").strip(),
    }
    active = {k: v for k, v in override_channels.items() if v}
    raw_record = os.getenv("GABION_OVERRIDE_RECORD_JSON", "").strip()
    payload: dict[str, object] = {"active_channels": active, "override_record": None}
    if active:
        if not raw_record:
            raise SystemExit("override channels active but GABION_OVERRIDE_RECORD_JSON is missing")
        record = json.loads(raw_record)
        if not isinstance(record, dict) or not REQUIRED.issubset(set(record.keys())):
            raise SystemExit("override record missing required schema fields")
        if _parse_time(str(record["expiry"])) <= datetime.now(timezone.utc):
            raise SystemExit("override record expired")
        payload["override_record"] = record

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
