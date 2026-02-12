from __future__ import annotations

import json
from pathlib import Path

from gabion.analysis.timeout_context import Deadline, check_deadline, deadline_scope
from gabion.lsp_client import _env_timeout_ticks, _has_env_timeout

_DEFAULT_ADVISORY_TIMEOUT_TICKS = 120_000
_DEFAULT_ADVISORY_TIMEOUT_TICK_NS = 1_000_000


def _deadline_scope():
    if _has_env_timeout():
        ticks, tick_ns = _env_timeout_ticks()
    else:
        ticks, tick_ns = (
            _DEFAULT_ADVISORY_TIMEOUT_TICKS,
            _DEFAULT_ADVISORY_TIMEOUT_TICK_NS,
        )
    return deadline_scope(Deadline.from_timeout_ticks(ticks, tick_ns))


def _print_summary(delta_path: Path) -> None:
    if not delta_path.exists():
        print("Docflow compliance delta missing (advisory).")
        return
    payload = json.loads(delta_path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    baseline = summary.get("baseline", {}) if isinstance(summary, dict) else {}
    current = summary.get("current", {}) if isinstance(summary, dict) else {}
    delta = summary.get("delta", {}) if isinstance(summary, dict) else {}
    keys = ["compliant", "contradicts", "excess", "proposed"]
    print("Docflow compliance delta summary (advisory):")
    for key in keys:
        check_deadline()
        print(f"- {key}: {baseline.get(key, 0)} -> {current.get(key, 0)} ({delta.get(key, 0)})")


def main() -> int:
    with _deadline_scope():
        try:
            _print_summary(Path("artifacts/out/docflow_compliance_delta.json"))
        except Exception as exc:  # advisory only; keep CI green
            print(f"Docflow compliance delta advisory error: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
