#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
from typing import Iterable

from gabion.policy_dsl import PolicyDomain, evaluate_policy
from gabion.tooling.runtime import policy_scanner_suite as runtime_policy_scanner_suite
from scripts.policy import hotspot_neighborhood_queue


def _policy_suite_decision(
    violations_by_rule: dict[str, list[dict[str, object]]],
):
    counts = {
        rule: len(items)
        for rule, items in violations_by_rule.items()
    }
    return evaluate_policy(
        domain=PolicyDomain.POLICY_SCANNER,
        data={"counts": counts},
    )


def _changed_paths_from_git(
    *,
    root: Path,
    base_sha: str | None,
    head_sha: str | None,
) -> set[str] | None:
    if base_sha and head_sha:
        command = ["git", "diff", "--name-only", f"{base_sha}..{head_sha}"]
    else:
        command = ["git", "diff", "--name-only", "HEAD"]
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return None
    changed = set(_iter_nonempty_stripped_lines(completed.stdout))
    if base_sha and head_sha:
        return changed

    try:
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return changed
    changed.update(_iter_nonempty_stripped_lines(untracked.stdout))
    return changed


def _iter_nonempty_stripped_lines(payload: str) -> Iterable[str]:
    for line in payload.splitlines():
        stripped = line.strip()
        if stripped:
            yield stripped


def run(
    *,
    root: Path,
    out_dir: Path,
    base_sha: str | None = None,
    head_sha: str | None = None,
) -> int:
    changed_paths = _changed_paths_from_git(
        root=root,
        base_sha=base_sha,
        head_sha=head_sha,
    )
    violations_by_rule = runtime_policy_scanner_suite.scan_policy_suite(
        root=root,
        changed_paths=changed_paths,
    )
    decision = _policy_suite_decision(violations_by_rule)
    queue_json = out_dir / "hotspot_neighborhood_queue.json"
    queue_md = out_dir / "hotspot_neighborhood_queue.md"
    hotspot_neighborhood_queue.run_from_inputs(
        violations_by_rule=violations_by_rule,
        policy_check_result_path=out_dir / "policy_check_result.json",
        out_path=queue_json,
        markdown_out=queue_md,
    )
    total = sum(len(items) for items in violations_by_rule.values())
    print(f"policy-suite scan: total_violations={total} out_dir={out_dir}")
    print(
        "policy-suite decision: "
        f"rule_id={decision.rule_id} outcome={decision.outcome.value} "
        f"severity={decision.severity.value}"
    )
    print(f"hotspot-neighborhood queue: {queue_json}")
    if total == 0:
        return 0
    for rule, items in violations_by_rule.items():
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
