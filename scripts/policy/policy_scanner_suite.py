#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from gabion.tooling.runtime import policy_result_schema
from gabion.tooling.runtime import policy_scanner_suite as runtime_policy_scanner_suite
from scripts.policy import hotspot_neighborhood_queue


def _policy_result_status(payload: dict[str, object]) -> str | None:
    status = str(payload.get("status", "") or "").strip()
    return status or None


def _policy_check_projection_fiber_semantics(
    payload: dict[str, object],
) -> dict[str, object] | None:
    if str(payload.get("rule_id", "") or "").strip() != "policy_check":
        return None
    raw_semantics = payload.get("projection_fiber_semantics")
    match raw_semantics:
        case dict() as semantics_mapping if semantics_mapping:
            return dict(semantics_mapping)
        case _:
            return None


def _load_preserved_policy_result(
    *, artifact: Path, expected_rule_id: str
) -> dict[str, object] | None:
    loaded = policy_result_schema.load_policy_result(artifact)
    if loaded is None:
        return None
    if str(loaded.get("rule_id", "") or "").strip() != expected_rule_id:
        return None
    return loaded


def _resolve_external_child_inputs(
    *, root: Path, out: Path
) -> runtime_policy_scanner_suite.PolicySuiteChildInputs:
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
    env = os.environ.copy()
    existing_pythonpath = str(env.get("PYTHONPATH", "") or "").strip()
    root_pythonpath = str(root)
    env["PYTHONPATH"] = (
        root_pythonpath
        if not existing_pythonpath
        else f"{root_pythonpath}:{existing_pythonpath}"
    )
    child_statuses: dict[str, str] = {}
    projection_fiber_semantics: dict[str, object] | None = None
    for rule_id, command in checks:
        artifact = Path(command[-1])
        preserved = _load_preserved_policy_result(
            artifact=artifact,
            expected_rule_id=rule_id,
        )
        if preserved is not None:
            status = _policy_result_status(preserved)
            if status is not None:
                child_statuses[rule_id] = status
            projection_fiber_semantics = (
                _policy_check_projection_fiber_semantics(preserved)
                or projection_fiber_semantics
            )
            continue
        completed = subprocess.run(
            [sys.executable, *command],
            cwd=root,
            env=env,
            check=False,
        )
        loaded = policy_result_schema.load_policy_result(artifact)
        if loaded is not None:
            status = _policy_result_status(loaded)
            if status is not None:
                child_statuses[rule_id] = status
            projection_fiber_semantics = (
                _policy_check_projection_fiber_semantics(loaded)
                or projection_fiber_semantics
            )
            continue
        raise RuntimeError(
            "external policy result artifact missing after wrapper invocation: "
            f"rule_id={rule_id} returncode={completed.returncode} artifact={artifact}"
        )
    return runtime_policy_scanner_suite.PolicySuiteChildInputs(
        child_statuses=child_statuses,
        projection_fiber_semantics=projection_fiber_semantics,
    )


def run(
    *,
    root: Path,
    out: Path,
    base_sha: str | None = None,
    head_sha: str | None = None,
) -> int:
    child_inputs = _resolve_external_child_inputs(root=root, out=out)
    result = runtime_policy_scanner_suite.load_or_scan_policy_suite(
        root=root,
        artifact_path=out,
        child_inputs=child_inputs,
        base_sha=base_sha,
        head_sha=head_sha,
    )
    decision = result.decision()
    queue_json = out.parent / "hotspot_neighborhood_queue.json"
    queue_md = out.parent / "hotspot_neighborhood_queue.md"
    hotspot_neighborhood_queue.run_from_payload(
        payload=result.to_payload(),
        out_path=queue_json,
        markdown_out=queue_md,
    )
    total = result.total_violations()
    print(f"policy-suite scan: cached={result.cached} total_violations={total} out={out}")
    print(
        "policy-suite decision: "
        f"rule_id={decision.rule_id} outcome={decision.outcome.value} "
        f"severity={decision.severity.value}"
    )
    print(f"hotspot-neighborhood queue: {queue_json}")
    semantic_queue_path = out.parent / "projection_semantic_fragment_queue.json"
    print(
        "projection-semantic-fragment queue: "
        f"{semantic_queue_path if semantic_queue_path.exists() else '<not emitted by wrapper>'}"
    )
    if total == 0:
        for rule_id in ("policy_check", "structural_hash", "deprecated_nonerasability"):
            status = str(child_inputs.child_statuses.get(rule_id, "unknown"))
            print(f"{rule_id} status: {status}")
        return 0
    for rule in (
        "no_monkeypatch",
        "branchless",
        "defensive_fallback",
        "fiber_loop_structure_contract",
        "fiber_filter_processor_contract",
        "fiber_return_shape_contract",
        "fiber_scalar_sentinel_contract",
        "fiber_type_dispatch_contract",
        "no_anonymous_tuple",
        "no_mutable_dict",
        "no_scalar_conversion_boundary",
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
        items = runtime_policy_scanner_suite.violations_for_rule(result, rule=rule)
        if not items:
            continue
        print(f"{rule} violations:")
        for item in items:
            print(f"  - {item.get('render', item)}")
    return 1 if decision.outcome.value == "block" else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", required=True)
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
