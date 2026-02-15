from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _build_command() -> list[str]:
    cmd = [sys.executable, "-m", "gabion", "check", "--emit-ambiguity-delta"]
    state_path = Path("artifacts/out/ambiguity_state.json")
    if state_path.exists():
        cmd.extend(["--ambiguity-state", str(state_path)])
    return cmd


def main() -> int:
    try:
        env = dict(os.environ)
        env.setdefault("GABION_DIRECT_RUN", "1")
        subprocess.run(_build_command(), check=True, env=env)
    except subprocess.CalledProcessError:
        print("Ambiguity delta emit failed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
