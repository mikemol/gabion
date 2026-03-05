#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from gabion.tooling.runtime import policy_scanner_suite


def run(*, root: Path, out: Path) -> int:
    result = policy_scanner_suite.load_or_scan_policy_suite(root=root, artifact_path=out)
    total = result.total_violations()
    type_debt_items = policy_scanner_suite.violations_for_rule(result, rule="type_contract_debt")
    type_debt_counts = {kind: 0 for kind in policy_scanner_suite._TYPE_DEBT_KINDS}
    ratchet_regressions = 0
    for item in type_debt_items:
        kind = str(item.get("kind", "") or "")
        if kind == "ratchet_regression":
            ratchet_regressions += 1
            continue
        if kind in type_debt_counts:
            type_debt_counts[kind] += 1
    debt_summary = ", ".join(f"{kind}={type_debt_counts[kind]}" for kind in sorted(type_debt_counts))
    print(
        f"policy-suite scan: cached={result.cached} total_violations={total} out={out} "
        f"type_contract_debt[{debt_summary}] ratchet_regressions={ratchet_regressions}"
    )
    if total == 0:
        return 0
    for rule in (
        "no_monkeypatch",
        "branchless",
        "defensive_fallback",
        "no_legacy_monolith_import",
        "orchestrator_primitive_barrel",
        "type_contract_debt",
    ):
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
