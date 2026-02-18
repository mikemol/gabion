from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

ENV_FLAG = "GABION_GATE_ORPHANED_DELTA"


def _enabled(value: str | None = None) -> bool:
    if value is None:
        value = os.getenv(ENV_FLAG)
    if value is None or not value.strip():
        return True
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _delta_value(payload: Mapping[str, object]) -> int:
    summary = payload.get("summary", {})
    if not isinstance(summary, Mapping):
        return 0
    delta = summary.get("delta", {})
    if not isinstance(delta, Mapping):
        return 0
    try:
        return int(delta.get("orphaned", 0))
    except (TypeError, ValueError):
        return 0


def check_gate(path: Path, *, enabled: bool | None = None) -> int:
    if enabled is None:
        enabled = _enabled()
    if not enabled:
        print(
            "Annotation drift gate disabled via emergency override "
            f"({ENV_FLAG})."
        )
        return 0
    if not path.exists():
        print(f"Annotation drift delta artifact missing: {path}")
        return 1
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Annotation drift delta artifact malformed ({path}): {exc}")
        return 1
    if not isinstance(payload, Mapping):
        print(f"Annotation drift delta artifact must be an object: {path}")
        return 1
    delta_value = _delta_value(payload)
    if delta_value > 0:
        summary = payload.get("summary", {})
        baseline = (
            summary.get("baseline", {}) if isinstance(summary, Mapping) else {}
        )
        current = (
            summary.get("current", {}) if isinstance(summary, Mapping) else {}
        )
        before = (
            baseline.get("orphaned", 0) if isinstance(baseline, Mapping) else 0
        )
        after = (
            current.get("orphaned", 0) if isinstance(current, Mapping) else 0
        )
        print(
            "Orphaned annotation delta increased "
            f"(before={before}, current={after}, delta=+{delta_value})."
        )
        return 1
    print(f"Orphaned annotation delta OK ({delta_value}).")
    return 0


def main() -> int:
    return check_gate(Path("artifacts/out/test_annotation_drift_delta.json"))


if __name__ == "__main__":
    raise SystemExit(main())
