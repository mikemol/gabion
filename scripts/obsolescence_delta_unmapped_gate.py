from __future__ import annotations

import json
import os
from pathlib import Path


ENV_FLAG = "GABION_GATE_UNMAPPED_DELTA"


def _enabled(value: str | None = None) -> bool:
    if value is None:
        value = os.getenv(ENV_FLAG)
    if value is None or not value.strip():
        return True
    return value.strip().lower() in {"1", "true", "yes", "on"}


def check_gate(path: Path, *, enabled: bool | None = None) -> int:
    if enabled is None:
        enabled = _enabled()
    if not enabled:
        print(
            "Unmapped delta gate disabled via emergency override "
            f"({ENV_FLAG})."
        )
        return 0
    if not path.exists():
        print(f"Test obsolescence delta artifact missing: {path}")
        return 1
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Test obsolescence delta artifact malformed ({path}): {exc}")
        return 1
    if not isinstance(payload, dict):
        print(f"Test obsolescence delta artifact must be an object: {path}")
        return 1
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
            "Unmapped evidence delta increased "
            f"(before={baseline}, current={current}, delta=+{delta_value})."
        )
        return 1
    print(f"Unmapped evidence delta OK ({delta_value}).")
    return 0


def main() -> int:
    return check_gate(Path("artifacts/out/test_obsolescence_delta.json"))


if __name__ == "__main__":
    raise SystemExit(main())
