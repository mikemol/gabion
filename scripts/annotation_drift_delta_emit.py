from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _build_command() -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "gabion",
        "check",
        "--emit-test-annotation-drift-delta",
    ]
    state_path = Path("out/test_annotation_drift.json")
    if state_path.exists():
        cmd.extend(["--test-annotation-drift-state", str(state_path)])
    return cmd


def main() -> int:
    try:
        env = dict(os.environ)
        env.setdefault("GABION_DIRECT_RUN", "1")
        subprocess.run(_build_command(), check=True, env=env)
    except subprocess.CalledProcessError:
        print("Annotation drift delta emit failed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
