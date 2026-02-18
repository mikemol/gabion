from __future__ import annotations

import json
import os
from pathlib import Path


ENV_FLAG = "GABION_GATE_OBSOLESCENCE_DELTA"


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
            "Test obsolescence gate disabled via emergency override "
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
    opaque = summary.get("opaque_evidence", {}) if isinstance(summary, dict) else {}
    delta = opaque.get("delta", 0)
    try:
        delta_value = int(delta)
    except (TypeError, ValueError):
        delta_value = 0
    if delta_value > 0:
        baseline = opaque.get("baseline", 0)
        current = opaque.get("current", 0)
        print(
            "Opaque evidence delta increased "
            f"(before={baseline}, current={current}, delta=+{delta_value})."
        )
        return 1
    print(f"Opaque evidence delta OK ({delta_value}).")
    return 0


def main() -> int:
    return check_gate(Path("artifacts/out/test_obsolescence_delta.json"))


if __name__ == "__main__":
    raise SystemExit(main())
