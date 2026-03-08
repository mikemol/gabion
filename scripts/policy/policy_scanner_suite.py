#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from gabion.tooling.runtime import policy_result_schema, policy_scanner_suite


def _run_external_policy_results(*, root: Path, out: Path) -> dict[str, dict[str, object]]:
    checks: tuple[tuple[str, list[str]], ...] = (
        (
            "policy_check",
            [
                "scripts/policy/policy_check.py",
                "--workflows",
                "--output",
                str(out.parent / "policy_check_result.json"),
            ],
        ),
        (
            "structural_hash",
            [
                "scripts/policy/structural_hash_policy_check.py",
                "--root",
                str(root),
                "--output",
                str(out.parent / "structural_hash_result.json"),
            ],
        ),
        (
            "deprecated_nonerasability",
            [
                "scripts/policy/deprecated_nonerasability_policy_check.py",
                "--baseline",
                str(root / "out" / "deprecated_fibers_baseline.json"),
                "--current",
                str(root / "out" / "deprecated_fibers_current.json"),
                "--output",
                str(out.parent / "deprecated_nonerasability_result.json"),
            ],
        ),
    )
    results: dict[str, dict[str, object]] = {}
    env = os.environ.copy()
    existing_pythonpath = str(env.get("PYTHONPATH", "") or "").strip()
    root_pythonpath = str(root)
    env["PYTHONPATH"] = (
        root_pythonpath
        if not existing_pythonpath
        else f"{root_pythonpath}:{existing_pythonpath}"
    )
    for rule_id, command in checks:
        if rule_id == "deprecated_nonerasability":
            baseline = Path(command[2])
            current = Path(command[4])
            if not baseline.exists() or not current.exists():
                results[rule_id] = policy_result_schema.make_policy_result(
                    rule_id=rule_id,
                    status="skip",
                    violations=[
                        {
                            "message": "baseline/current payload missing; rule skipped in suite aggregation",
                            "render": f"missing baseline={baseline.exists()} current={current.exists()}",
                        }
                    ],
                    baseline_mode="baseline_compare",
                    source_tool="scripts/policy/policy_scanner_suite.py",
                    input_scope={"baseline": str(baseline), "current": str(current)},
                )
                continue
        completed = subprocess.run(
            [sys.executable, *command],
            cwd=root,
            env=env,
            check=False,
        )
        artifact = Path(command[-1])
        loaded = policy_result_schema.load_policy_result(artifact)
        if loaded is not None:
            results[rule_id] = loaded
            continue
        status = "pass" if completed.returncode == 0 else "fail"
        results[rule_id] = policy_result_schema.make_policy_result(
            rule_id=rule_id,
            status=status,
            violations=[{"message": f"fallback result from return code {completed.returncode}", "render": f"returncode={completed.returncode}"}] if completed.returncode != 0 else [],
            baseline_mode="fallback",
            source_tool="scripts/policy/policy_scanner_suite.py",
            input_scope={"command": command[:-2]},
        )
    return results


def run(
    *,
    root: Path,
    out: Path,
    base_sha: str | None = None,
    head_sha: str | None = None,
) -> int:
    policy_results = _run_external_policy_results(root=root, out=out)
    result = policy_scanner_suite.load_or_scan_policy_suite(
        root=root,
        artifact_path=out,
        policy_results=policy_results,
        base_sha=base_sha,
        head_sha=head_sha,
    )
    total = result.total_violations()
    print(f"policy-suite scan: cached={result.cached} total_violations={total} out={out}")
    if total == 0:
        for rule_id in ("policy_check", "structural_hash", "deprecated_nonerasability"):
            status = str(result.policy_results.get(rule_id, {}).get("status", "unknown"))
            print(f"{rule_id} status: {status}")
        return 0
    for rule in (
        "no_monkeypatch",
        "branchless",
        "defensive_fallback",
        "fiber_scalar_sentinel_contract",
        "no_legacy_monolith_import",
        "orchestrator_primitive_barrel",
        "typing_surface",
        "runtime_narrowing_boundary",
        "aspf_normalization_idempotence",
        "boundary_core_contract",
        "fiber_normalization_contract",
        "test_subprocess_hygiene",
        "test_sleep_hygiene",
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
    parser.add_argument("--base-sha", default=None)
    parser.add_argument("--head-sha", default=None)
    args = parser.parse_args(argv)
    return run(
        root=Path(args.root).resolve(),
        out=Path(args.out).resolve(),
        base_sha=str(args.base_sha).strip() or None,
        head_sha=str(args.head_sha).strip() or None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
