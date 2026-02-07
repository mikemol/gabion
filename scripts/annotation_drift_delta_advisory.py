from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run_delta() -> None:
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "gabion",
                "check",
                "--emit-test-annotation-drift-delta",
            ],
            check=True,
        )
    except subprocess.CalledProcessError:
        print("Annotation drift delta failed (advisory).")


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
        print(
            f"- {key}: {baseline.get(key, 0)} -> {current.get(key, 0)} ({delta.get(key, 0)})"
        )


def main() -> int:
    try:
        _run_delta()
        _print_summary(Path("out/test_annotation_drift_delta.json"))
    except Exception as exc:  # advisory only; keep CI green
        print(f"Annotation drift delta advisory error: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
