from __future__ import annotations

import json
import os
from pathlib import Path

from gabion.analysis.timeout_context import check_deadline


ENV_FLAG = "GABION_GATE_ORPHANED_DELTA"


def _enabled() -> bool:
    value = os.getenv(ENV_FLAG, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _print_summary(delta_path: Path) -> None:
    if not delta_path.exists():
        print("Annotation drift delta missing (advisory).")
        return
    payload = json.loads(delta_path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    baseline = summary.get("baseline", {})
    current = summary.get("current", {})
    delta = summary.get("delta", {})
    keys = sorted({*baseline.keys(), *current.keys(), *delta.keys()})
    print("Annotation drift delta summary (advisory):")
    for key in keys:
        check_deadline()
        print(
            f"- {key}: {baseline.get(key, 0)} -> {current.get(key, 0)} ({delta.get(key, 0)})"
        )


def main() -> int:
    try:
        if _enabled():
            print(
                "Annotation drift delta advisory skipped; "
                f"{ENV_FLAG}=1 enables the gate."
            )
            return 0
        _print_summary(Path("artifacts/out/test_annotation_drift_delta.json"))
    except Exception as exc:  # advisory only; keep CI green
        print(f"Annotation drift delta advisory error: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
