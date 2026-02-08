from __future__ import annotations

import argparse
import subprocess
import sys


def _run_check(flag: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "gabion", "check", flag],
        check=True,
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
    args = parser.parse_args()

    if not (args.obsolescence or args.annotation_drift or args.ambiguity or args.all):
        args.all = True

    if args.all or args.obsolescence:
        _run_check("--write-test-obsolescence-baseline")
    if args.all or args.annotation_drift:
        _run_check("--write-test-annotation-drift-baseline")
    if args.all or args.ambiguity:
        _run_check("--write-ambiguity-baseline")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
