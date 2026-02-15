from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    try:
        env = dict(os.environ)
        env.setdefault("GABION_DIRECT_RUN", "1")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "gabion",
                "check",
                "--emit-test-obsolescence-state",
                "--emit-test-annotation-drift",
                "--emit-ambiguity-state",
            ],
            env=env,
            check=True,
        )
    except subprocess.CalledProcessError:
        print("Delta state emit failed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
