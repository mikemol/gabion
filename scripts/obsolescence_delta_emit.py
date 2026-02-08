from __future__ import annotations

import subprocess
import sys


def main() -> int:
    try:
        subprocess.run(
            [sys.executable, "-m", "gabion", "check", "--emit-test-obsolescence-delta"],
            check=True,
        )
    except subprocess.CalledProcessError:
        print("Test obsolescence delta emit failed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
