from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping


ENV_FLAG = "GABION_GATE_DOCFLOW_DELTA"


def _enabled(value: str | None = None) -> bool:
    if value is None:
        value = os.getenv(ENV_FLAG, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _delta_value(payload: Mapping[str, object], key: str) -> int:
    summary = payload.get("summary", {})
    if not isinstance(summary, Mapping):
        return 0
    delta = summary.get("delta", {})
    if not isinstance(delta, Mapping):
        return 0
    try:
        return int(delta.get(key, 0))
    except (TypeError, ValueError):
        return 0


def check_gate(path: Path, *, enabled: bool | None = None) -> int:
    if enabled is None:
        enabled = _enabled()
    if not enabled:
        print(f"Docflow delta gate disabled; set {ENV_FLAG}=1 to enable.")
        return 0
    if not path.exists():
        print("Docflow delta missing; gate skipped.")
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Docflow delta unreadable; gate skipped: {exc}")
        return 0
    if not isinstance(payload, Mapping):
        print("Docflow delta unreadable; gate skipped.")
        return 0
    if payload.get("baseline_missing"):
        print("Docflow baseline missing; gate skipped.")
        return 0
    contradicts_delta = _delta_value(payload, "contradicts")
    excess_delta = _delta_value(payload, "excess")
    proposed_delta = _delta_value(payload, "proposed")
    if contradicts_delta > 0:
        summary = payload.get("summary", {})
        delta = summary.get("delta", {}) if isinstance(summary, Mapping) else {}
        before = (summary.get("baseline", {}) if isinstance(summary, Mapping) else {}).get(
            "contradicts", 0
        )
        after = (summary.get("current", {}) if isinstance(summary, Mapping) else {}).get(
            "contradicts", 0
        )
        print(
            "Docflow contradictions increased: "
            f"{before} -> {after} (+{contradicts_delta})."
        )
        return 1
    print(
        "Docflow delta OK "
        f"(contradicts {contradicts_delta}, excess {excess_delta}, proposed {proposed_delta})."
    )
    return 0


def main() -> int:
    return check_gate(Path("artifacts/out/docflow_compliance_delta.json"))


if __name__ == "__main__":
    raise SystemExit(main())
