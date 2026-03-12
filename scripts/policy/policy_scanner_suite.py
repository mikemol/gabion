#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from gabion.tooling.runtime import policy_result_schema
from gabion.tooling.runtime import policy_scanner_suite as runtime_policy_scanner_suite
from scripts.policy import hotspot_neighborhood_queue


def _load_projection_fiber_semantics(
    *,
    out_dir: Path,
) -> dict[str, object] | None:
    artifact = out_dir / "policy_check_result.json"
    loaded = policy_result_schema.load_policy_result(artifact)
    if loaded is None or str(loaded.get("rule_id", "") or "").strip() != "policy_check":
        raise RuntimeError(
            "required child-owned policy result artifact missing before wrapper invocation: "
            f"rule_id=policy_check artifact={artifact}"
        )
    raw_semantics = loaded.get("projection_fiber_semantics")
    match raw_semantics:
        case dict() as semantics_mapping if semantics_mapping:
            return dict(semantics_mapping)
        case _:
            return None


def run(
    *,
    root: Path,
    out_dir: Path,
    base_sha: str | None = None,
    head_sha: str | None = None,
) -> int:
    projection_fiber_semantics = _load_projection_fiber_semantics(out_dir=out_dir)
    result = runtime_policy_scanner_suite.scan_policy_suite(
        root=root,
        base_sha=base_sha,
        head_sha=head_sha,
    )
    decision = result.decision()
    queue_json = out_dir / "hotspot_neighborhood_queue.json"
    queue_md = out_dir / "hotspot_neighborhood_queue.md"
    hotspot_neighborhood_queue.run_from_inputs(
        violations_by_rule=result.violations_by_rule,
        projection_fiber_semantics=projection_fiber_semantics,
        out_path=queue_json,
        markdown_out=queue_md,
    )
    total = sum(len(items) for items in result.violations_by_rule.values())
    print(f"policy-suite scan: total_violations={total} out_dir={out_dir}")
    print(
        "policy-suite decision: "
        f"rule_id={decision.rule_id} outcome={decision.outcome.value} "
        f"severity={decision.severity.value}"
    )
    print(f"hotspot-neighborhood queue: {queue_json}")
    if total == 0:
        return 0
    for rule, items in result.violations_by_rule.items():
        if not items:
            continue
        print(f"{rule} violations:")
        for item in items:
            print(f"  - {item.get('render', item)}")
    return 1 if decision.outcome.value == "block" else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--base-sha", default=None)
    parser.add_argument("--head-sha", default=None)
    args = parser.parse_args(argv)
    return run(
        root=Path(args.root).resolve(),
        out_dir=Path(args.out_dir).resolve(),
        base_sha=str(args.base_sha).strip() or None,
        head_sha=str(args.head_sha).strip() or None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
