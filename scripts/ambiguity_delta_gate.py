from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

ENV_FLAG = "GABION_GATE_AMBIGUITY_DELTA"


def _enabled(value: str | None = None) -> bool:
    if value is None:
        value = os.getenv(ENV_FLAG, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _delta_value(payload: Mapping[str, object]) -> int:
    summary = payload.get("summary", {})
    if not isinstance(summary, Mapping):
        return 0
    total = summary.get("total", {})
    if not isinstance(total, Mapping):
        return 0
    try:
        return int(total.get("delta", 0))
    except (TypeError, ValueError):
        return 0


def check_gate(path: Path, *, enabled: bool | None = None) -> int:
    if enabled is None:
        enabled = _enabled()
    if not enabled:
        print(f"Ambiguity delta gate disabled; set {ENV_FLAG}=1 to enable.")
        return 0
    if not path.exists():
        print("Ambiguity delta missing; gate skipped.")
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Ambiguity delta unreadable; gate skipped: {exc}")
        return 0
    if not isinstance(payload, Mapping):
        print("Ambiguity delta unreadable; gate skipped.")
        return 0
    delta_value = _delta_value(payload)
    if delta_value > 0:
        summary = payload.get("summary", {})
        total = summary.get("total", {}) if isinstance(summary, Mapping) else {}
        before = total.get("baseline", 0) if isinstance(total, Mapping) else 0
        after = total.get("current", 0) if isinstance(total, Mapping) else 0
        print(
            "Ambiguity delta increased: "
            f"{before} -> {after} (+{delta_value})."
        )
        return 1
    print(f"Ambiguity delta OK ({delta_value}).")
    return 0


def main() -> int:
    return check_gate(Path("artifacts/out/ambiguity_delta.json"))


if __name__ == "__main__":
    raise SystemExit(main())
