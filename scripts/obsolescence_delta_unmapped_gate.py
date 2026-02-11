from __future__ import annotations

import json
import os
from pathlib import Path


ENV_FLAG = "GABION_GATE_UNMAPPED_DELTA"


def _enabled() -> bool:
    value = os.getenv(ENV_FLAG, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def main() -> int:
    if not _enabled():
        print(f"Unmapped delta gate disabled; set {ENV_FLAG}=1 to enable.")
        return 0
    path = Path("artifacts/out/test_obsolescence_delta.json")
    if not path.exists():
        print("Test obsolescence delta missing; gate skipped.")
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Test obsolescence delta unreadable; gate skipped: {exc}")
        return 0
    summary = payload.get("summary", {})
    counts = summary.get("counts", {}) if isinstance(summary, dict) else {}
    delta = counts.get("delta", {}) if isinstance(counts, dict) else {}
    try:
        delta_value = int(delta.get("unmapped", 0))
    except (TypeError, ValueError):
        delta_value = 0
    if delta_value > 0:
        baseline = counts.get("baseline", {}).get("unmapped", 0)
        current = counts.get("current", {}).get("unmapped", 0)
        print(
            "Unmapped evidence delta increased: "
            f"{baseline} -> {current} (+{delta_value})."
        )
        return 1
    print(f"Unmapped evidence delta OK ({delta_value}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
