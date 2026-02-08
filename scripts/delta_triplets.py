from __future__ import annotations

import concurrent.futures
import os
import subprocess
import sys


TRIPLETS: dict[str, list[str]] = {
    "obsolescence": [
        "scripts/obsolescence_delta_emit.py",
        "scripts/obsolescence_delta_advisory.py",
        "scripts/obsolescence_delta_gate.py",
        "scripts/obsolescence_delta_unmapped_gate.py",
    ],
    "annotation_drift": [
        "scripts/annotation_drift_delta_emit.py",
        "scripts/annotation_drift_delta_advisory.py",
        "scripts/annotation_drift_orphaned_gate.py",
    ],
    "ambiguity": [
        "scripts/ambiguity_delta_emit.py",
        "scripts/ambiguity_delta_advisory.py",
        "scripts/ambiguity_delta_gate.py",
    ],
}


def _run_triplet(name: str, steps: list[str]) -> int:
    exit_code = 0
    env = dict(os.environ)
    env.setdefault("GABION_DIRECT_RUN", "1")
    for step in steps:
        result = subprocess.run([sys.executable, step], check=False, env=env)
        if result.returncode != 0:
            print(f"{name} step failed: {step} (exit {result.returncode})")
            exit_code = exit_code or result.returncode
    return exit_code


def main() -> int:
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(TRIPLETS)
    ) as executor:
        futures = {
            executor.submit(_run_triplet, name, steps): name
            for name, steps in TRIPLETS.items()
        }
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"{name} triplet crashed: {exc}")
                failures += 1
                continue
            if result != 0:
                failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
