from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


OBSOLESCENCE_DELTA_PATH = Path("out/test_obsolescence_delta.json")
ANNOTATION_DRIFT_DELTA_PATH = Path("out/test_annotation_drift_delta.json")
AMBIGUITY_DELTA_PATH = Path("out/ambiguity_delta.json")

ENV_GATE_UNMAPPED = "GABION_GATE_UNMAPPED_DELTA"
ENV_GATE_ORPHANED = "GABION_GATE_ORPHANED_DELTA"
ENV_GATE_AMBIGUITY = "GABION_GATE_AMBIGUITY_DELTA"


def _run_check(flag: str, timeout: int | None) -> None:
    subprocess.run(
        [sys.executable, "-m", "gabion", "check", flag],
        check=True,
        timeout=timeout,
    )


def _enabled(env_flag: str) -> bool:
    value = os.getenv(env_flag, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_nested(payload: object, keys: list[str], default: int = 0) -> int:
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    try:
        return int(current) if current is not None else default
    except (TypeError, ValueError):
        return default


def _ensure_delta(flag: str, path: Path, timeout: int | None) -> dict[str, object]:
    if not path.exists():
        _run_check(flag, timeout)
    if not path.exists():
        raise FileNotFoundError(f"Missing delta output at {path}")
    return _load_json(path)


def _guard_obsolescence_delta(timeout: int | None) -> None:
    payload = _ensure_delta(
        "--emit-test-obsolescence-delta", OBSOLESCENCE_DELTA_PATH, timeout
    )
    opaque_delta = _get_nested(payload, ["summary", "opaque_evidence", "delta"])
    if opaque_delta > 0:
        raise SystemExit(
            "Refusing to refresh obsolescence baseline: opaque evidence delta > 0."
        )
    if _enabled(ENV_GATE_UNMAPPED):
        unmapped_delta = _get_nested(payload, ["summary", "counts", "delta", "unmapped"])
        if unmapped_delta > 0:
            raise SystemExit(
                "Refusing to refresh obsolescence baseline: unmapped delta > 0."
            )


def _guard_annotation_drift_delta(timeout: int | None) -> None:
    if not _enabled(ENV_GATE_ORPHANED):
        return
    payload = _ensure_delta(
        "--emit-test-annotation-drift-delta",
        ANNOTATION_DRIFT_DELTA_PATH,
        timeout,
    )
    orphaned_delta = _get_nested(payload, ["summary", "delta", "orphaned"])
    if orphaned_delta > 0:
        raise SystemExit(
            "Refusing to refresh annotation drift baseline: orphaned delta > 0."
        )


def _guard_ambiguity_delta(timeout: int | None) -> None:
    if not _enabled(ENV_GATE_AMBIGUITY):
        return
    payload = _ensure_delta("--emit-ambiguity-delta", AMBIGUITY_DELTA_PATH, timeout)
    total_delta = _get_nested(payload, ["summary", "total", "delta"])
    if total_delta > 0:
        raise SystemExit(
            "Refusing to refresh ambiguity baseline: ambiguity delta > 0."
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh baseline carriers via gabion check.",
    )
    parser.add_argument(
        "--obsolescence",
        action="store_true",
        help="Refresh baselines/test_obsolescence_baseline.json",
    )
    parser.add_argument(
        "--annotation-drift",
        action="store_true",
        help="Refresh baselines/test_annotation_drift_baseline.json",
    )
    parser.add_argument(
        "--ambiguity",
        action="store_true",
        help="Refresh baselines/ambiguity_baseline.json",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Refresh all baselines (default when no flags provided).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Seconds to wait for each gabion check (default: no timeout).",
    )
    args = parser.parse_args()

    if not (args.obsolescence or args.annotation_drift or args.ambiguity or args.all):
        args.all = True

    if args.all or args.obsolescence:
        _guard_obsolescence_delta(args.timeout)
        _run_check("--write-test-obsolescence-baseline", args.timeout)
    if args.all or args.annotation_drift:
        _guard_annotation_drift_delta(args.timeout)
        _run_check("--write-test-annotation-drift-baseline", args.timeout)
    if args.all or args.ambiguity:
        _guard_ambiguity_delta(args.timeout)
        _run_check("--write-ambiguity-baseline", args.timeout)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
