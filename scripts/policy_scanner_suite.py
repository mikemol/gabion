#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from gabion.tooling import policy_scanner_suite


def run(*, root: Path, out: Path) -> int:
    result = policy_scanner_suite.load_or_scan_policy_suite(root=root, artifact_path=out)
    total = result.total_violations()
    print(f"policy-suite scan: cached={result.cached} total_violations={total} out={out}")
    if total == 0:
        return 0
    for rule in ("no_monkeypatch", "branchless", "defensive_fallback"):
        items = policy_scanner_suite.violations_for_rule(result, rule=rule)
        if not items:
            continue
        print(f"{rule} violations:")
        for item in items:
            print(f"  - {item.get('render', item)}")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="artifacts/out/policy_suite_results.json")
    args = parser.parse_args(argv)
    return run(root=Path(args.root).resolve(), out=Path(args.out).resolve())


if __name__ == "__main__":
    raise SystemExit(main())
