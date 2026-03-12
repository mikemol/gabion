#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from gabion.tooling.runtime import policy_result_schema
from gabion.tooling.runtime import policy_scanner_suite as runtime_policy_scanner_suite
from scripts.policy import hotspot_neighborhood_queue


@dataclass(frozen=True)
class ExternalChildInputs:
    child_statuses: dict[str, str]
    runtime_child_inputs: runtime_policy_scanner_suite.PolicySuiteChildInputs


@dataclass(frozen=True)
class ExternalChildArtifact:
    status: str | None
    projection_fiber_semantics: dict[str, object] | None


@dataclass(frozen=True)
class ExternalChildRequirement:
    rule_id: str
    artifact: Path


def _load_external_child_artifact(
    *, artifact: Path, expected_rule_id: str
) -> ExternalChildArtifact | None:
    loaded = policy_result_schema.load_policy_result(artifact)
    if loaded is None:
        return None
    if str(loaded.get("rule_id", "") or "").strip() != expected_rule_id:
        return None
    status_raw = str(loaded.get("status", "") or "").strip()
    projection_fiber_semantics: dict[str, object] | None = None
    if expected_rule_id == "policy_check":
        raw_semantics = loaded.get("projection_fiber_semantics")
        match raw_semantics:
            case dict() as semantics_mapping if semantics_mapping:
                projection_fiber_semantics = dict(semantics_mapping)
            case _:
                projection_fiber_semantics = None
    return ExternalChildArtifact(
        status=status_raw or None,
        projection_fiber_semantics=projection_fiber_semantics,
    )


def _resolve_external_child_inputs(
    *, root: Path, out: Path
) -> ExternalChildInputs:
    _ = root
    requirements: tuple[ExternalChildRequirement, ...] = (
        ExternalChildRequirement(
            rule_id="policy_check",
            artifact=out.parent / "policy_check_result.json",
        ),
        ExternalChildRequirement(
            rule_id="structural_hash",
            artifact=out.parent / "structural_hash_result.json",
        ),
        ExternalChildRequirement(
            rule_id="deprecated_nonerasability",
            artifact=out.parent / "deprecated_nonerasability_result.json",
        ),
    )
    child_statuses: dict[str, str] = {}
    projection_fiber_semantics: dict[str, object] | None = None
    for requirement in requirements:
        preserved = _load_external_child_artifact(
            artifact=requirement.artifact,
            expected_rule_id=requirement.rule_id,
        )
        if preserved is None:
            raise RuntimeError(
                "required child-owned policy result artifact missing before wrapper invocation: "
                f"rule_id={requirement.rule_id} artifact={requirement.artifact}"
            )
        if preserved.status is not None:
            child_statuses[requirement.rule_id] = preserved.status
        projection_fiber_semantics = (
            preserved.projection_fiber_semantics or projection_fiber_semantics
        )
    return ExternalChildInputs(
        child_statuses=child_statuses,
        runtime_child_inputs=runtime_policy_scanner_suite.PolicySuiteChildInputs(
            projection_fiber_semantics=projection_fiber_semantics,
        ),
    )


def run(
    *,
    root: Path,
    out: Path,
    base_sha: str | None = None,
    head_sha: str | None = None,
) -> int:
    child_inputs = _resolve_external_child_inputs(root=root, out=out)
    outcome = runtime_policy_scanner_suite.load_or_scan_policy_suite(
        root=root,
        artifact_path=out,
        child_inputs=child_inputs.runtime_child_inputs,
        base_sha=base_sha,
        head_sha=head_sha,
    )
    result = outcome.result
    decision = result.decision()
    queue_json = out.parent / "hotspot_neighborhood_queue.json"
    queue_md = out.parent / "hotspot_neighborhood_queue.md"
    payload: dict[str, object] = {
        "format_version": 1,
        "violations": result.violations_by_rule,
    }
    if result.projection_fiber_semantics is not None:
        payload["projection_fiber_semantics"] = result.projection_fiber_semantics
    hotspot_neighborhood_queue.run_from_payload(
        payload=payload,
        out_path=queue_json,
        markdown_out=queue_md,
    )
    total = sum(len(items) for items in result.violations_by_rule.values())
    print(f"policy-suite scan: cached={outcome.cached} total_violations={total} out={out}")
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
        items = list(result.violations_by_rule.get(rule, []))
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
