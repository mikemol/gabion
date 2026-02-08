from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    path = Path("out/test_obsolescence_delta.json")
    if not path.exists():
        print("Test obsolescence delta missing; gate skipped.")
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Test obsolescence delta unreadable; gate skipped: {exc}")
        return 0
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
            "Opaque evidence delta increased: "
            f"{baseline} -> {current} (+{delta_value})."
        )
        return 1
    print(f"Opaque evidence delta OK ({delta_value}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
