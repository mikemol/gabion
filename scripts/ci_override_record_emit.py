#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from gabion.tooling.override_record import validate_override_record_json


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
        validation = validate_override_record_json(raw_record)
        if not validation.valid:
            raise SystemExit(
                f"override channels active but override record is invalid: {validation.reason}"
            )
        payload["override_record"] = validation.record

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
