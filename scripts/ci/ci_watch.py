from __future__ import annotations

import sys


def main() -> int:
    print(
        "Removed: scripts/ci/ci_watch.py is retired. "
        "Use 'gabion ci watch [args]' directly. "
        "See docs/user_workflows.md#user_workflows and "
        "docs/normative_clause_index.md#clause-command-maturity-parity.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
