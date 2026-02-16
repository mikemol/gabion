from __future__ import annotations

import json
import os
from pathlib import Path

from deadline_runtime import DeadlineBudget, deadline_scope_from_lsp_env
from gabion.analysis.timeout_context import check_deadline


ENV_FLAG = "GABION_GATE_UNMAPPED_DELTA"
_DEFAULT_ADVISORY_TIMEOUT_TICKS = 120_000
_DEFAULT_ADVISORY_TIMEOUT_TICK_NS = 1_000_000
_DEFAULT_ADVISORY_TIMEOUT_BUDGET = DeadlineBudget(
    ticks=_DEFAULT_ADVISORY_TIMEOUT_TICKS,
    tick_ns=_DEFAULT_ADVISORY_TIMEOUT_TICK_NS,
)


def _deadline_scope():
    return deadline_scope_from_lsp_env(
        default_budget=_DEFAULT_ADVISORY_TIMEOUT_BUDGET,
    )


def _enabled() -> bool:
    value = os.getenv(ENV_FLAG, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


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
        check_deadline()
        print(
            f"- {key}: {baseline.get(key, 0)} -> {current.get(key, 0)} ({delta.get(key, 0)})"
        )
    print(
        "- opaque_evidence_count: "
        f"{opaque.get('baseline', 0)} -> {opaque.get('current', 0)} ({opaque.get('delta', 0)})"
    )


def main() -> int:
    with _deadline_scope():
        try:
            if _enabled():
                print(
                    "Test obsolescence delta advisory skipped; "
                    f"{ENV_FLAG}=1 enables the gate."
                )
                return 0
            _print_summary(Path("artifacts/out/test_obsolescence_delta.json"))
        except Exception as exc:  # advisory only; keep CI green
            print(f"Test obsolescence delta advisory error: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
