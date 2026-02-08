from __future__ import annotations

import json
from pathlib import Path


def _print_summary(delta_path: Path) -> None:
    if not delta_path.exists():
        print("Test obsolescence delta missing (advisory).")
        return
    payload = json.loads(delta_path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    counts = summary.get("counts", {})
    delta = counts.get("delta", {})
    baseline = counts.get("baseline", {})
    current = counts.get("current", {})
    opaque = summary.get("opaque_evidence", {})
    keys = [
        "redundant_by_evidence",
        "equivalent_witness",
        "obsolete_candidate",
        "unmapped",
    ]
    print("Test obsolescence delta summary (advisory):")
    for key in keys:
        print(
            f"- {key}: {baseline.get(key, 0)} -> {current.get(key, 0)} ({delta.get(key, 0)})"
        )
    print(
        "- opaque_evidence_count: "
        f"{opaque.get('baseline', 0)} -> {opaque.get('current', 0)} ({opaque.get('delta', 0)})"
    )


def main() -> int:
    try:
        _print_summary(Path("out/test_obsolescence_delta.json"))
    except Exception as exc:  # advisory only; keep CI green
        print(f"Test obsolescence delta advisory error: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
