from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

ENV_FLAG = "GABION_GATE_OPAQUE_DELTA"


def _enabled(value: str | None = None) -> bool:
    if value is None:
        value = os.getenv(ENV_FLAG)
    if value is None:
        return True
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _delta_value(payload: Mapping[str, object]) -> int:
    summary = payload.get("summary", {})
    if not isinstance(summary, Mapping):
        return 0
    opaque = summary.get("opaque_evidence", {})
    if not isinstance(opaque, Mapping):
        return 0
    try:
        return int(opaque.get("delta", 0))
    except (TypeError, ValueError):
        return 0


def check_gate(path: Path, *, enabled: bool | None = None) -> int:
    if enabled is None:
        enabled = _enabled()
    if not enabled:
        print(
            "Opaque obsolescence gate disabled by override; "
            f"set {ENV_FLAG}=1 to enforce."
        )
        return 0
    if not path.exists():
        print("Test obsolescence delta missing; gate failed.")
        return 2
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Test obsolescence delta unreadable; gate failed: {exc}")
        return 2
    if not isinstance(payload, Mapping):
        print("Test obsolescence delta unreadable; gate failed.")
        return 2
    delta_value = _delta_value(payload)
    if delta_value > 0:
        summary = payload.get("summary", {})
        opaque = summary.get("opaque_evidence", {}) if isinstance(summary, Mapping) else {}
        baseline = opaque.get("baseline", 0) if isinstance(opaque, Mapping) else 0
        current = opaque.get("current", 0) if isinstance(opaque, Mapping) else 0
        print(
            "Opaque evidence delta increased: "
            f"{baseline} -> {current} (+{delta_value})."
        )
        return 1
    print(f"Opaque evidence delta OK ({delta_value}).")
    return 0


def main() -> int:
    return check_gate(Path("artifacts/out/test_obsolescence_delta.json"))


