from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


BASELINE_PATH = Path("baselines/docflow_compliance_baseline.json")
CURRENT_PATH = Path("artifacts/out/docflow_compliance.json")
DELTA_PATH = Path("artifacts/out/docflow_compliance_delta.json")


def _run_docflow_audit() -> None:
    env = dict(os.environ)
    env.setdefault("GABION_DIRECT_RUN", "1")
    subprocess.run(
        [sys.executable, "-m", "gabion", "docflow-audit"],
        check=True,
        env=env,
    )


def _load_summary(path: Path) -> tuple[dict[str, int], bool]:
    if not path.exists():
        return {"compliant": 0, "contradicts": 0, "excess": 0, "proposed": 0}, True
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    counts: dict[str, int] = {}
    for key in ("compliant", "contradicts", "excess", "proposed"):
        value = 0
        if isinstance(summary, dict):
            try:
                value = int(summary.get(key, 0))
            except (TypeError, ValueError):
                value = 0
        counts[key] = value
    return counts, False


def _delta_counts(
    baseline: dict[str, int],
    current: dict[str, int],
) -> dict[str, int]:
    delta: dict[str, int] = {}
    for key in ("compliant", "contradicts", "excess", "proposed"):
        delta[key] = int(current.get(key, 0)) - int(baseline.get(key, 0))
    return delta


def main() -> int:
    try:
        _run_docflow_audit()
    except subprocess.CalledProcessError:
        print("Docflow delta emit failed: docflow audit did not succeed.")
        return 0
    if not CURRENT_PATH.exists():
        print("Docflow compliance output missing; delta emit skipped.")
        return 0
    baseline_counts, baseline_missing = _load_summary(BASELINE_PATH)
    current_counts, _ = _load_summary(CURRENT_PATH)
    payload = {
        "baseline": {"path": str(BASELINE_PATH)},
        "current": {"path": str(CURRENT_PATH)},
        "baseline_missing": baseline_missing,
        "summary": {
            "baseline": baseline_counts,
            "current": current_counts,
            "delta": _delta_counts(baseline_counts, current_counts),
        },
        "version": 1,
    }
    DELTA_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
