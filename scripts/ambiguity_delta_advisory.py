from __future__ import annotations

import json
from pathlib import Path


def _print_summary(delta_path: Path) -> None:
    if not delta_path.exists():
        print("Ambiguity delta missing (advisory).")
        return
    payload = json.loads(delta_path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    total = summary.get("total", {})
    by_kind = summary.get("by_kind", {})
    print("Ambiguity delta summary (advisory):")
    print(
        f"- total: {total.get('baseline', 0)} -> {total.get('current', 0)} ({total.get('delta', 0)})"
    )
    baseline = by_kind.get("baseline", {})
    current = by_kind.get("current", {})
    delta = by_kind.get("delta", {})
    keys = sorted({*baseline.keys(), *current.keys(), *delta.keys()})
    for key in keys:
        print(
            f"- {key}: {baseline.get(key, 0)} -> {current.get(key, 0)} ({delta.get(key, 0)})"
        )


def main() -> int:
    try:
        _print_summary(Path("out/ambiguity_delta.json"))
    except Exception as exc:  # advisory only; keep CI green
        print(f"Ambiguity delta advisory error: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
