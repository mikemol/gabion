#!/usr/bin/env python3
import cProfile
import argparse
import ast
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from scripts.deadline_runtime import DeadlineBudget, deadline_scope_from_ticks
from gabion.analysis.timeout_context import Deadline, check_deadline, deadline_clock_scope, deadline_scope
from gabion.invariants import never
from gabion.deadline_clock import MonotonicClock
from gabion.order_contract import ordered_or_sorted
from gabion.config import dataflow_adapter_payload, dataflow_defaults, dataflow_required_surfaces
from gabion.tooling.runtime import policy_result_schema
from gabion.tooling.runtime.perf_artifact import write_cprofile_perf_artifact
from gabion.policy_dsl import PolicyDomain, evaluate_policy
from gabion.policy_dsl.compile import compile_document
from gabion.policy_dsl.registry import build_registry
from gabion.policy_dsl.schema import PolicyOutcomeKind
from gabion.policy_dsl.typecheck import typecheck
from gabion.tooling.policy_substrate.lattice_convergence_semantic import (
    iter_semantic_lattice_convergence,
    materialize_semantic_lattice_convergence,
)

try:
    import yaml
except ImportError:  # pragma: no cover - handled as a hard error at runtime.
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"

ALLOWED_ACTIONS_FILE = REPO_ROOT / "docs" / "allowed_actions.txt"
REQUIRED_RUNNER_LABELS = {"self-hosted", "gpu", "local"}
TRUSTED_BRANCHES = {"main", "stage", "next", "release"}
CONTENT_WRITE_WORKFLOWS = {
    "auto-test-tag.yml",
    "release-tag.yml",
    "mirror-next.yml",
    "promote-release.yml",
}

_WORKFLOW_POLICY_OUTPUT_ROOT = REPO_ROOT / "out"
_QUOTIENT_GOVERNANCE_REPORT = _WORKFLOW_POLICY_OUTPUT_ROOT / "quotient_governance_report.json"
_QUOTIENT_RATCHET_DELTA = _WORKFLOW_POLICY_OUTPUT_ROOT / "quotient_ratchet_delta.json"
_QUOTIENT_POLICY_VIOLATIONS = _WORKFLOW_POLICY_OUTPUT_ROOT / "quotient_policy_violations.json"
_QUOTIENT_PROTOCOL_READINESS = _WORKFLOW_POLICY_OUTPUT_ROOT / "quotient_protocol_readiness.json"
_QUOTIENT_PROMOTION_DECISION = _WORKFLOW_POLICY_OUTPUT_ROOT / "quotient_promotion_decision.json"
_QUOTIENT_DEMOTION_INCIDENTS = _WORKFLOW_POLICY_OUTPUT_ROOT / "quotient_demotion_incidents.json"
_LOCAL_CI_REPRO_CONTRACT = (
    REPO_ROOT / "artifacts" / "out" / "local_ci_repro_contract.json"
)

_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")

_DEFAULT_POLICY_TIMEOUT_TICKS = 120_000
_DEFAULT_POLICY_TIMEOUT_TICK_NS = 1_000_000
_DEFAULT_POLICY_TIMEOUT_BUDGET = DeadlineBudget(
    ticks=_DEFAULT_POLICY_TIMEOUT_TICKS,
    tick_ns=_DEFAULT_POLICY_TIMEOUT_TICK_NS,
)


NORMATIVE_ENFORCEMENT_MAP = REPO_ROOT / "docs" / "normative_enforcement_map.yaml"
TIER2_RESIDUE_BASELINE = REPO_ROOT / "out" / "tier2_pattern_residue_baseline.json"
_REQUIRED_NORMATIVE_CLAUSES = {
    "NCI-LSP-FIRST",
    "NCI-ACTIONS-PINNED",
    "NCI-ACTIONS-ALLOWLIST",
    "NCI-DATAFLOW-BUNDLE-TIERS",
    "NCI-SHIFT-AMBIGUITY-LEFT",
    "NCI-BASELINE-RATCHET",
    "NCI-DEADLINE-TIMEOUT-PROPAGATION",
    "NCI-CONTROLLER-ADAPTATION-LAW",
    "NCI-OVERRIDE-LIFECYCLE",
    "NCI-CONTROLLER-DRIFT-LIFECYCLE",
    "NCI-COMMAND-MATURITY-PARITY",
    "NCI-DUAL-SENSOR-CORRECTION-LOOP",
}

ASPF_TAINT_MAP = REPO_ROOT / "docs" / "aspf_taint_isomorphism_map.yaml"
ASPF_TAINT_NO_CHANGE = REPO_ROOT / "docs" / "aspf_taint_isomorphism_no_change.yaml"
_REQUIRED_ASPF_IN_STEPS = {"in-46", "in-47", "in-48", "in-49", "in-50", "in-51", "in-52", "in-53", "in-58"}
_EXPECTED_ASPF_IDENTIFIER_ANCHORS = {
    "AspfOneCell",
    "AspfTwoCellWitness",
    "DomainToAspfCofibration",
    "append_delta_record",
    "NeverInvariantSink",
    "NEVER_INVARIANTS_SPEC",
}
_ASPF_TAINT_TRIGGER_PATHS = {
    "src/gabion/invariants.py",
    "src/gabion/exceptions.py",
    "in/in-46.md",
    "in/in-47.md",
    "in/in-48.md",
    "in/in-49.md",
    "in/in-50.md",
    "in/in-51.md",
    "in/in-52.md",
    "in/in-53.md",
    "in/in-58.md",
}
_ASPF_TAINT_TRIGGER_PREFIXES = (
    "src/gabion/analysis/",
)


def _changed_repo_paths() -> set[str]:
    check_deadline()
    commands = (
        ["git", "diff", "--name-only", "HEAD"],
        ["git", "ls-files", "--others", "--exclude-standard"],
    )
    changed: set[str] = set()
    for command in commands:
        check_deadline()
        try:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        except (subprocess.CalledProcessError, OSError):
            continue
        for line in completed.stdout.splitlines():
            check_deadline()
            path = line.strip()
            if path:
                changed.add(path)
    if changed:
        return changed
    commit_range_commands = (
        ["git", "diff", "--name-only", "HEAD~1", "HEAD"],
        ["git", "show", "--name-only", "--pretty=format:", "HEAD"],
    )
    for command in commit_range_commands:
        check_deadline()
        try:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        except (subprocess.CalledProcessError, OSError):
            continue
        commit_changed = {
            line.strip()
            for line in completed.stdout.splitlines()
            if line.strip()
        }
        if commit_changed:
            return commit_changed
    return changed


def _is_aspf_taint_trigger(path: str) -> bool:
    if path in _ASPF_TAINT_TRIGGER_PATHS:
        return True
    return any(path.startswith(prefix) for prefix in _ASPF_TAINT_TRIGGER_PREFIXES)


def check_aspf_taint_crosswalk_ack() -> None:
    check_deadline()
    changed = _changed_repo_paths()
    triggered = sorted(path for path in changed if _is_aspf_taint_trigger(path))
    if not triggered:
        return
    if not ASPF_TAINT_MAP.exists():
        _fail([f"missing ASPF taint crosswalk map: {ASPF_TAINT_MAP}"])
    payload = _load_yaml(ASPF_TAINT_MAP)
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        _fail(["docs/aspf_taint_isomorphism_map.yaml: entries must be a list"])
    mapped_steps: set[str] = set()
    seen_identifiers: set[str] = set()
    for entry in entries:
        check_deadline()
        if not isinstance(entry, dict):
            continue
        in_step = str(entry.get("in_step", "")).strip()
        if in_step:
            mapped_steps.add(in_step)
        identifiers = entry.get("existing_identifiers")
        if isinstance(identifiers, list):
            for item in identifiers:
                check_deadline()
                value = str(item).strip()
                if value:
                    seen_identifiers.add(value)
    missing_steps = _REQUIRED_ASPF_IN_STEPS - mapped_steps
    missing_anchors = _EXPECTED_ASPF_IDENTIFIER_ANCHORS - seen_identifiers
    map_changed = "docs/aspf_taint_isomorphism_map.yaml" in changed
    no_change_payload = _load_yaml(ASPF_TAINT_NO_CHANGE) if ASPF_TAINT_NO_CHANGE.exists() else {}
    no_change_changed = "docs/aspf_taint_isomorphism_no_change.yaml" in changed
    no_change_valid = False
    if isinstance(no_change_payload, dict):
        justification = str(no_change_payload.get("justification", "")).strip()
        steps = no_change_payload.get("in_steps")
        if justification and isinstance(steps, list) and set(str(step).strip() for step in steps) >= _REQUIRED_ASPF_IN_STEPS:
            no_change_valid = True
    errors: list[str] = []
    if missing_steps:
        errors.append(f"ASPF taint crosswalk missing in_step mappings: {_sorted(missing_steps)}")
    if missing_anchors:
        errors.append(f"ASPF taint crosswalk missing identifier anchors: {_sorted(missing_anchors)}")
    if not map_changed and not (no_change_changed and no_change_valid):
        errors.append(
            "taint/marker/ASPF-relevant changes require either docs/aspf_taint_isomorphism_map.yaml updates "
            "or docs/aspf_taint_isomorphism_no_change.yaml with in_steps + justification"
        )
    if errors:
        _fail(["ASPF taint crosswalk acknowledgement check failed", f"triggered paths: {triggered}", *errors])



def _workflow_doc(path: Path):
    if not path.exists():
        return {}
    return _load_yaml(path)


def check_normative_enforcement_map() -> None:
    errors: list[str] = []
    check_deadline()
    if not NORMATIVE_ENFORCEMENT_MAP.exists():
        _fail([f"missing normative enforcement map: {NORMATIVE_ENFORCEMENT_MAP}"])
    payload = _load_yaml(NORMATIVE_ENFORCEMENT_MAP)
    clauses = payload.get("clauses")
    if not isinstance(clauses, dict):
        _fail(["docs/normative_enforcement_map.yaml: clauses must be a mapping"])
    clause_ids = set(clauses.keys())
    missing = _REQUIRED_NORMATIVE_CLAUSES - clause_ids
    extra = clause_ids - _REQUIRED_NORMATIVE_CLAUSES
    if missing:
        errors.append(f"normative map missing required clauses: {_sorted(missing)}")
    if extra:
        errors.append(f"normative map has unknown clause keys: {_sorted(extra)}")

    for clause_id, entry in clauses.items():
        check_deadline()
        if not isinstance(entry, dict):
            errors.append(f"{clause_id}: entry must be mapping")
            continue
        status = entry.get("status")
        if status not in {"enforced", "partial", "document-only"}:
            errors.append(f"{clause_id}: invalid status {status!r}")
        for module_path in entry.get("enforcing_modules") or []:
            check_deadline()
            module_ref = REPO_ROOT / str(module_path)
            if not module_ref.exists():
                errors.append(f"{clause_id}: missing enforcing module path {module_path}")
        ci_anchors = entry.get("ci_anchors") or []
        if not isinstance(ci_anchors, list):
            errors.append(f"{clause_id}: ci_anchors must be a list")
            continue
        for anchor in ci_anchors:
            check_deadline()
            if not isinstance(anchor, dict):
                errors.append(f"{clause_id}: ci anchor must be mapping")
                continue
            workflow = REPO_ROOT / str(anchor.get("workflow", ""))
            job = str(anchor.get("job", ""))
            step = str(anchor.get("step", ""))
            if not workflow.exists():
                errors.append(f"{clause_id}: workflow does not exist: {workflow}")
                continue
            workflow_doc = _workflow_doc(workflow)
            jobs = workflow_doc.get("jobs")
            if not isinstance(jobs, dict) or job not in jobs:
                errors.append(f"{clause_id}: missing workflow job anchor {workflow}:{job}")
                continue
            steps = jobs[job].get("steps") if isinstance(jobs[job], dict) else []
            step_names = {
                str(item.get("name", ""))
                for item in steps
                if isinstance(item, dict)
            }
            if step and step not in step_names:
                errors.append(f"{clause_id}: missing workflow step anchor {workflow}:{job}:{step}")
        artifacts = entry.get("expected_artifacts") or []
        if not isinstance(artifacts, list):
            errors.append(f"{clause_id}: expected_artifacts must be a list")
            continue
        for artifact in artifacts:
            check_deadline()
            if not isinstance(artifact, str) or not artifact.strip():
                errors.append(f"{clause_id}: invalid expected artifact reference {artifact!r}")
    if errors:
        _fail(errors)


def _policy_timeout_budget() -> DeadlineBudget:
    raw_ticks = os.getenv("GABION_POLICY_TIMEOUT_TICKS", "").strip()
    raw_tick_ns = os.getenv("GABION_POLICY_TIMEOUT_TICK_NS", "").strip()
    if raw_ticks or raw_tick_ns:
        if not raw_ticks:
            never("missing policy timeout ticks", tick_ns=raw_tick_ns)
        if not raw_tick_ns:
            never("missing policy timeout tick_ns", ticks=raw_ticks)
        try:
            ticks_value = int(raw_ticks)
        except ValueError:
            ticks_value = -1
        try:
            tick_ns_value = int(raw_tick_ns)
        except ValueError:
            tick_ns_value = -1
        if ticks_value <= 0:
            never("invalid policy timeout ticks", ticks=raw_ticks)
        if tick_ns_value <= 0:
            never("invalid policy timeout tick_ns", tick_ns=raw_tick_ns)
        return DeadlineBudget(
            ticks=ticks_value,
            tick_ns=tick_ns_value,
        )
    return DeadlineBudget(
        ticks=_DEFAULT_POLICY_TIMEOUT_BUDGET.ticks,
        tick_ns=_DEFAULT_POLICY_TIMEOUT_BUDGET.tick_ns,
    )


def check_policy_dsl() -> None:
    errors: list[str] = []
    docs = [
        REPO_ROOT / "docs" / "policy_rules.yaml",
        REPO_ROOT / "docs" / "aspf_opportunity_rules.yaml",
        REPO_ROOT / "docs" / "projection_fiber_rules.yaml",
    ]
    for path in docs:
        if not path.exists():
            continue
        program, issues = compile_document(path)
        for issue in issues:
            errors.append(f"{path}: compile {issue.code}: {issue.message} ({issue.rule_id})")
        if program is not None:
            for issue in typecheck(program):
                errors.append(f"{path}: typecheck {issue.code}: {issue.message} ({issue.rule_id})")
    try:
        _ = build_registry()
    except ValueError as exc:
        errors.append(f"registry build failed: {exc}")
    if errors:
        _fail(errors)


@dataclass(frozen=True)
class ProjectionFiberLatticeConvergenceResult:
    decision_rule_id: str
    decision_outcome: str
    decision_severity: str
    decision_message: str
    report_payload: dict[str, object]
    error_messages: tuple[str, ...]

    @property
    def blocking(self) -> bool:
        return self.decision_outcome == PolicyOutcomeKind.BLOCK.value

    def as_policy_output(self) -> dict[str, object]:
        return {
            "decision": {
                "rule_id": self.decision_rule_id,
                "outcome": self.decision_outcome,
                "severity": self.decision_severity,
                "message": self.decision_message,
            },
            "report": self.report_payload,
            "error_messages": list(self.error_messages),
        }


def collect_aspf_lattice_convergence_result() -> ProjectionFiberLatticeConvergenceResult:
    events = iter_semantic_lattice_convergence(repo_root=REPO_ROOT)
    report = materialize_semantic_lattice_convergence(
        events=events,
    )
    decision = evaluate_policy(
        domain=PolicyDomain.PROJECTION_FIBER,
        data=report.policy_data(),
    )
    return ProjectionFiberLatticeConvergenceResult(
        decision_rule_id=decision.rule_id,
        decision_outcome=decision.outcome.value,
        decision_severity=decision.severity.value,
        decision_message=decision.message,
        report_payload=report.policy_data(),
        error_messages=report.error_messages(),
    )


def _policy_deadline_scope():
    return deadline_scope_from_ticks(
        budget=_policy_timeout_budget(),
    )


def _sorted(values, *, key=None, reverse: bool = False):
    return ordered_or_sorted(
        values,
        source="scripts.policy_check",
        key=key,
        reverse=reverse,
    )


def _workflow_reason_code(error: str) -> str:
    normalized = error.lower()
    if "unable to read lock-in source" in normalized or "dense-core lock-in missing token" in normalized:
        return "WF57_DENSE_CORE_LOCKIN"
    if "must invoke" in normalized and "workflow" in normalized:
        return "WF57_ENTRYPOINT_MISSING"
    if "workflow must invoke" in normalized:
        return "WF57_ENTRYPOINT_MISSING"
    if "release tag workflow must use" in normalized or "release_tag.py missing" in normalized:
        return "WF53_RELEASE_TAG_ENTRYPOINT"
    if "must derive tag from pyproject.toml" in normalized:
        return "WF53_VERSION_DERIVATION"
    if "must verify tag equals main/next/release" in normalized:
        return "WF53_RELEASE_TAG_PROVENANCE"
    if "must verify tag equals main/next" in normalized:
        return "WF53_TEST_TAG_PROVENANCE"
    return "WF57_POLICY_GUARDRAIL"


@dataclass(frozen=True)
class _CiReproCapabilitySpec:
    capability_id: str
    summary: str
    source_alternatives: tuple[tuple[str, ...], ...] = ()
    command_alternatives: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class _CiReproSurfaceSpec:
    surface_id: str
    surface_kind: str
    title: str
    summary: str
    source_ref: str
    mode: str
    capabilities: tuple[_CiReproCapabilitySpec, ...]
    artifacts: tuple[str, ...] = ()
    commands: tuple[str, ...] = ()


@dataclass(frozen=True)
class _CiReproRelationSpec:
    relation_id: str
    relation_kind: str
    source_surface_id: str
    target_surface_id: str
    summary: str


def _read_text_if_exists(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _workflow_job_run_text(doc: object, *, job_name: str) -> str:
    if not isinstance(doc, dict):
        return ""
    jobs = doc.get("jobs", {})
    if not isinstance(jobs, dict):
        return ""
    job = jobs.get(job_name, {})
    if not isinstance(job, dict):
        return ""
    steps = job.get("steps", [])
    if not isinstance(steps, list):
        return ""
    lines: list[str] = []
    for step in steps:
        check_deadline()
        if not isinstance(step, dict):
            continue
        run = step.get("run")
        if isinstance(run, str):
            lines.append(run)
    return "\n".join(lines)


def _surface_payload(
    *,
    spec: _CiReproSurfaceSpec,
    source_text: str,
) -> dict[str, object]:
    command_text = "\n".join(spec.commands)
    capability_payloads: list[dict[str, object]] = []
    missing_capability_ids: list[str] = []
    required_token_groups: list[list[str]] = []
    missing_token_groups: list[list[str]] = []
    for capability in spec.capabilities:
        matched_source_alternative_index: int | None = None
        if capability.source_alternatives:
            for index, group in enumerate(capability.source_alternatives):
                if all(token in source_text for token in group):
                    matched_source_alternative_index = index
                    break
        matched_command_alternative_index: int | None = None
        if capability.command_alternatives:
            for index, group in enumerate(capability.command_alternatives):
                if all(token in command_text for token in group):
                    matched_command_alternative_index = index
                    break
        source_ok = (
            matched_source_alternative_index is not None
            if capability.source_alternatives
            else True
        )
        command_ok = (
            matched_command_alternative_index is not None
            if capability.command_alternatives
            else True
        )
        status = "pass" if source_ok and command_ok else "fail"
        if status != "pass":
            missing_capability_ids.append(capability.capability_id)
            missing_token_groups.extend(
                [list(group) for group in capability.source_alternatives]
            )
            missing_token_groups.extend(
                [list(group) for group in capability.command_alternatives]
            )
        required_token_groups.extend(
            [list(group) for group in capability.source_alternatives]
        )
        required_token_groups.extend(
            [list(group) for group in capability.command_alternatives]
        )
        capability_payloads.append(
            {
                "capability_id": capability.capability_id,
                "summary": capability.summary,
                "status": status,
                "source_alternative_token_groups": [
                    list(group) for group in capability.source_alternatives
                ],
                "command_alternative_token_groups": [
                    list(group) for group in capability.command_alternatives
                ],
                "matched_source_alternative_index": matched_source_alternative_index,
                "matched_command_alternative_index": matched_command_alternative_index,
            }
        )
    return {
        "surface_id": spec.surface_id,
        "surface_kind": spec.surface_kind,
        "title": spec.title,
        "summary": spec.summary,
        "source_ref": spec.source_ref,
        "mode": spec.mode,
        "status": "pass" if not missing_capability_ids else "fail",
        "required_capabilities": capability_payloads,
        "missing_capability_ids": missing_capability_ids,
        "required_token_groups": required_token_groups,
        "missing_token_groups": missing_token_groups,
        "commands": list(spec.commands),
        "artifacts": list(spec.artifacts),
    }


def _build_local_ci_repro_contract_payload(*, repo_root: Path) -> dict[str, object]:
    workflow_dir = repo_root / ".github" / "workflows"
    ci_doc = _load_yaml(workflow_dir / "ci.yml")
    pr_doc = _load_yaml(workflow_dir / "pr-dataflow-grammar.yml")
    checks_text = _read_text_if_exists(repo_root / "scripts" / "checks.sh")
    ci_local_repro_text = _read_text_if_exists(repo_root / "scripts" / "ci_local_repro.sh")
    ci_local_repro_runtime_text = _read_text_if_exists(
        repo_root / "src" / "gabion" / "tooling" / "runtime" / "ci_local_repro.py"
    )
    ci_cycle_text = _read_text_if_exists(repo_root / "scripts" / "ci" / "ci_cycle.py")
    workflow_checks_capabilities = (
        _CiReproCapabilitySpec(
            capability_id="policy_workflows_output",
            summary="Materialize the workflow policy artifact before downstream policy tooling.",
            source_alternatives=(
                (
                    "scripts/policy/policy_check.py",
                    "--workflows",
                    "artifacts/out/policy_check_result.json",
                ),
            ),
        ),
        _CiReproCapabilitySpec(
            capability_id="policy_ambiguity_contract",
            summary="Run the ambiguity gate in the checks lane.",
            source_alternatives=(("--ambiguity-contract",),),
        ),
        _CiReproCapabilitySpec(
            capability_id="policy_tier2_residue_contract",
            summary="Run the tier2 residue contract in the checks lane.",
            source_alternatives=(("--tier2-residue-contract",),),
        ),
        _CiReproCapabilitySpec(
            capability_id="strict_docflow",
            summary="Run strict docflow with required SPPF GH-reference mode.",
            source_alternatives=(
                ("gabion", "docflow", "--sppf-gh-ref-mode", "required"),
            ),
        ),
        _CiReproCapabilitySpec(
            capability_id="docflow_packet_loop",
            summary="Run docflow packetize plus packet enforce with proving tests.",
            source_alternatives=(
                (
                    "scripts/policy/docflow_packetize.py",
                    "scripts/policy/docflow_packet_enforce.py",
                    "--run-proving-tests",
                ),
            ),
        ),
        _CiReproCapabilitySpec(
            capability_id="sppf_status_audit",
            summary="Run the local SPPF status audit.",
            source_alternatives=(("scripts.sppf.sppf_status_audit",),),
        ),
        _CiReproCapabilitySpec(
            capability_id="sppf_issue_lifecycle_validation",
            summary="Validate SPPF issue lifecycle labels on the active rev range.",
            source_alternatives=(
                (
                    "scripts.sppf.sppf_sync",
                    "--validate",
                    "--require-label",
                    "done-on-stage",
                    "--require-label",
                    "status/pending-release",
                ),
            ),
        ),
        _CiReproCapabilitySpec(
            capability_id="test_evidence_index",
            summary="Refresh test evidence and fail on drift.",
            source_alternatives=(
                (
                    "scripts.misc.extract_test_evidence",
                    "git",
                    "diff",
                    "--exit-code",
                    "out/test_evidence.json",
                ),
            ),
        ),
        _CiReproCapabilitySpec(
            capability_id="test_behavior_index",
            summary="Refresh test behavior metadata and fail on drift.",
            source_alternatives=(
                (
                    "scripts.misc.extract_test_behavior",
                    "git",
                    "diff",
                    "--exit-code",
                    "out/test_behavior.json",
                ),
            ),
        ),
        _CiReproCapabilitySpec(
            capability_id="policy_scanner_suite",
            summary="Run the policy scanner suite after policy-check output exists.",
            source_alternatives=(("scripts/policy/policy_scanner_suite.py",),),
        ),
        _CiReproCapabilitySpec(
            capability_id="structural_hash_policy_check",
            summary="Run the structural hash policy check in the checks lane.",
            source_alternatives=(("scripts/policy/structural_hash_policy_check.py",),),
        ),
        _CiReproCapabilitySpec(
            capability_id="deprecated_nonerasability_policy_check",
            summary="Run the deprecated nonerasability policy check in the checks lane.",
            source_alternatives=(("scripts/policy/deprecated_nonerasability_policy_check.py",),),
        ),
        _CiReproCapabilitySpec(
            capability_id="controller_drift_audit",
            summary="Materialize the controller drift artifact.",
            source_alternatives=(("scripts/governance/governance_controller_audit.py",),),
        ),
        _CiReproCapabilitySpec(
            capability_id="controller_override_record",
            summary="Emit override lifecycle state for the controller drift gate.",
            source_alternatives=(("scripts/ci/ci_override_record_emit.py",),),
        ),
        _CiReproCapabilitySpec(
            capability_id="controller_drift_gate",
            summary="Evaluate controller drift against the override lifecycle record.",
            source_alternatives=(("scripts/ci/ci_controller_drift_gate.py",),),
        ),
        _CiReproCapabilitySpec(
            capability_id="lsp_parity_gate",
            summary="Run the LSP parity gate for gabion.check.",
            source_alternatives=(("gabion", "lsp-parity-gate"),),
        ),
        _CiReproCapabilitySpec(
            capability_id="pytest_cov_junit",
            summary="Run pytest with coverage and JUnit output.",
            source_alternatives=(
                ("--junitxml", "artifacts/test_runs/junit.xml", "--cov=src/gabion"),
            ),
        ),
        _CiReproCapabilitySpec(
            capability_id="delta_bundle",
            summary="Run the delta-bundle gate with direct carrier execution.",
            source_alternatives=(("check", "delta-bundle"),),
        ),
        _CiReproCapabilitySpec(
            capability_id="delta_gates",
            summary="Run the delta-gates followup gate.",
            source_alternatives=(("check", "delta-gates"),),
        ),
        _CiReproCapabilitySpec(
            capability_id="governance_telemetry_emit",
            summary="Emit governance telemetry from the checks lane timings.",
            source_alternatives=(("scripts/governance/governance_telemetry_emit.py",),),
        ),
    )
    workflow_dataflow_capabilities = (
        _CiReproCapabilitySpec(
            capability_id="run_dataflow_stage",
            summary="Run the direct-carrier dataflow stage.",
            source_alternatives=(("run-dataflow-stage",),),
        ),
        _CiReproCapabilitySpec(
            capability_id="terminal_outcome_finalize",
            summary="Project terminal outcome semantics for dataflow completion.",
            source_alternatives=(
                ("scripts/ci/ci_finalize_dataflow_outcome.py",),
                (
                    "terminal_status=\"${terminal_status:-unknown}\"",
                    "terminal_status=\"timeout_resume\"",
                    "terminal_status=\"hard_failure\"",
                ),
            ),
        ),
        _CiReproCapabilitySpec(
            capability_id="deadline_profile_summary",
            summary="Render the deadline profile CI summary if the artifact exists.",
            source_alternatives=(("scripts.deadline_profile_ci_summary",),),
        ),
    )
    workflow_pr_dataflow_capabilities = (
        _CiReproCapabilitySpec(
            capability_id="verify_stage_ci",
            summary="Verify the pushed SHA already passed stage CI before PR grammar rendering.",
            source_alternatives=(
                ("Stage CI", "workflow_runs", "head_sha"),
            ),
        ),
        _CiReproCapabilitySpec(
            capability_id="policy_workflows_output",
            summary="Materialize the workflow policy artifact before downstream policy tooling.",
            source_alternatives=(
                (
                    "scripts/policy/policy_check.py",
                    "--workflows",
                    "artifacts/out/policy_check_result.json",
                ),
            ),
        ),
        _CiReproCapabilitySpec(
            capability_id="policy_scanner_suite",
            summary="Run the policy scanner suite in the PR grammar lane.",
            source_alternatives=(("scripts/policy/policy_scanner_suite.py",),),
        ),
        _CiReproCapabilitySpec(
            capability_id="governance_pr_template",
            summary="Check governance PR template fields before rendering grammar output.",
            source_alternatives=(("scripts/audit/check_pr_governance_template.py",),),
        ),
        _CiReproCapabilitySpec(
            capability_id="controller_drift_audit",
            summary="Materialize the controller drift artifact in advisory mode.",
            source_alternatives=(("scripts/governance/governance_controller_audit.py",),),
        ),
        _CiReproCapabilitySpec(
            capability_id="impact_select_tests",
            summary="Select impacted tests before running pytest.",
            source_alternatives=(("gabion", "impact-select-tests"),),
        ),
        _CiReproCapabilitySpec(
            capability_id="pytest_cov_junit",
            summary="Run pytest with coverage and JUnit output.",
            source_alternatives=(
                ("--junitxml", "artifacts/test_runs/junit.xml", "--cov=src/gabion"),
            ),
        ),
        _CiReproCapabilitySpec(
            capability_id="render_dataflow_grammar_report",
            summary="Render the PR dataflow grammar report via gabion check raw.",
            source_alternatives=(
                ("check", "raw", "artifacts/dataflow_grammar/report.md"),
            ),
        ),
    )
    surfaces = (
        _CiReproSurfaceSpec(
            surface_id="workflow:ci.yml:checks",
            surface_kind="workflow_job",
            title="CI checks workflow job",
            summary="Strict governance, docflow, scanner, and test gates on stage pushes.",
            source_ref=".github/workflows/ci.yml",
            mode="checks",
            capabilities=workflow_checks_capabilities,
            artifacts=(
                "artifacts/out/docflow_compliance.json",
                "artifacts/out/docflow_packet_enforcement.json",
                "artifacts/out/controller_drift.json",
                "artifacts/out/policy_check_result.json",
                "artifacts/test_runs/junit.xml",
                "out/test_evidence.json",
            ),
            commands=(
                "python -m scripts.policy.policy_check --workflows --output artifacts/out/policy_check_result.json",
                "python -m gabion docflow --root . --fail-on-violations --sppf-gh-ref-mode required",
                "python scripts/policy/docflow_packet_enforce.py --check --run-proving-tests",
                "python -m pytest --junitxml artifacts/test_runs/junit.xml",
            ),
        ),
        _CiReproSurfaceSpec(
            surface_id="workflow:ci.yml:dataflow-grammar",
            surface_kind="workflow_job",
            title="CI dataflow grammar workflow job",
            summary="Stage dataflow execution lane with terminal outcome finalization and deadline summaries.",
            source_ref=".github/workflows/ci.yml",
            mode="dataflow-grammar",
            capabilities=workflow_dataflow_capabilities,
            artifacts=(
                "artifacts/out/deadline_profile.json",
                "artifacts/out/deadline_profile_ci_summary.json",
                "artifacts/out/aspf_handoff_manifest.json",
            ),
            commands=(
                "python -m gabion run-dataflow-stage",
                "python scripts/ci/ci_finalize_dataflow_outcome.py",
                "python -m scripts.deadline_profile_ci_summary",
            ),
        ),
        _CiReproSurfaceSpec(
            surface_id="workflow:pr-dataflow-grammar.yml:dataflow-grammar",
            surface_kind="workflow_job",
            title="PR dataflow grammar workflow job",
            summary="Pull-request diff, targeted-test, and dataflow report workflow lane.",
            source_ref=".github/workflows/pr-dataflow-grammar.yml",
            mode="dataflow-grammar",
            capabilities=workflow_pr_dataflow_capabilities,
            artifacts=(
                "artifacts/out/policy_check_result.json",
                "artifacts/out/controller_drift.json",
                "artifacts/test_runs/junit.xml",
                "artifacts/audit_reports/impact_selection.json",
                "artifacts/dataflow_grammar/report.md",
            ),
            commands=(
                "python scripts/policy/policy_check.py --workflows --output artifacts/out/policy_check_result.json",
                "python scripts/policy/policy_scanner_suite.py --root . --out-dir artifacts/out",
                "python -m gabion impact-select-tests",
                "python -m pytest --junitxml artifacts/test_runs/junit.xml",
                "python -m gabion check raw --report artifacts/dataflow_grammar/report.md",
            ),
        ),
        _CiReproSurfaceSpec(
            surface_id="tooling_command:gabion:ci-local-repro:checks",
            surface_kind="tooling_command",
            title="gabion ci-local-repro checks lane",
            summary="gabion command host for local reproduction of the ci.yml checks job.",
            source_ref="src/gabion/tooling/runtime/ci_local_repro.py",
            mode="checks-only",
            capabilities=workflow_checks_capabilities,
            artifacts=(
                "artifacts/out/docflow_compliance.json",
                "artifacts/out/docflow_packet_enforcement.json",
                "artifacts/out/controller_drift.json",
                "artifacts/out/policy_check_result.json",
                "artifacts/test_runs/junit.xml",
                "artifacts/test_runs/coverage.xml",
                "out/test_evidence.json",
            ),
            commands=("python -m gabion ci-local-repro --checks-only",),
        ),
        _CiReproSurfaceSpec(
            surface_id="tooling_command:gabion:ci-local-repro:dataflow",
            surface_kind="tooling_command",
            title="gabion ci-local-repro dataflow lane",
            summary="gabion command host for local reproduction of the ci.yml dataflow-grammar job.",
            source_ref="src/gabion/tooling/runtime/ci_local_repro.py",
            mode="dataflow-only",
            capabilities=workflow_dataflow_capabilities,
            artifacts=(
                "artifacts/out/deadline_profile.json",
                "artifacts/out/deadline_profile_ci_summary.json",
                "artifacts/out/aspf_handoff_manifest.json",
            ),
            commands=("python -m gabion ci-local-repro --dataflow-only",),
        ),
        _CiReproSurfaceSpec(
            surface_id="tooling_command:gabion:ci-local-repro:pr-dataflow",
            surface_kind="tooling_command",
            title="gabion ci-local-repro PR dataflow lane",
            summary="gabion command host for local reproduction of the PR dataflow workflow.",
            source_ref="src/gabion/tooling/runtime/ci_local_repro.py",
            mode="pr-dataflow-only",
            capabilities=workflow_pr_dataflow_capabilities,
            artifacts=(
                "artifacts/out/policy_check_result.json",
                "artifacts/out/controller_drift.json",
                "artifacts/test_runs/junit.xml",
                "artifacts/test_runs/coverage.xml",
                "artifacts/audit_reports/impact_selection.json",
            ),
            commands=("python -m gabion ci-local-repro --pr-dataflow-only",),
        ),
        _CiReproSurfaceSpec(
            surface_id="local_script:scripts/ci_local_repro.sh:wrapper",
            surface_kind="local_repro_wrapper",
            title="ci_local_repro shell wrapper",
            summary="Bootstrap wrapper that dispatches into gabion ci-local-repro.",
            source_ref="scripts/ci_local_repro.sh",
            mode="wrapper",
            capabilities=(
                _CiReproCapabilitySpec(
                    capability_id="dispatch_ci_local_repro",
                    summary="Bootstrap the repo venv and dispatch into gabion ci-local-repro.",
                    source_alternatives=(("-m gabion ci-local-repro",),),
                ),
            ),
            commands=("scripts/ci_local_repro.sh --checks-only",),
        ),
        _CiReproSurfaceSpec(
            surface_id="local_script:scripts/checks.sh:dataflow",
            surface_kind="local_verification_lane",
            title="Narrow local dataflow verification lane",
            summary="Fast local verification path for dataflow and optional status-watch.",
            source_ref="scripts/checks.sh",
            mode="dataflow-only",
            capabilities=(
                _CiReproCapabilitySpec(
                    capability_id="lsp_parity_gate",
                    summary="Run the LSP parity gate for gabion.check.",
                    source_alternatives=(("gabion lsp-parity-gate",),),
                ),
                _CiReproCapabilitySpec(
                    capability_id="gabion_check_run",
                    summary="Run gabion check in the fast local verification lane.",
                    source_alternatives=(("gabion check run",),),
                ),
            ),
            artifacts=("artifacts/out/aspf_handoff_manifest.json",),
            commands=("scripts/checks.sh --dataflow-only",),
        ),
        _CiReproSurfaceSpec(
            surface_id="local_script:scripts/checks.sh:docflow",
            surface_kind="local_verification_lane",
            title="Narrow local docflow verification lane",
            summary="Fast local verification path for strict docflow and packet enforcement.",
            source_ref="scripts/checks.sh",
            mode="docflow-only",
            capabilities=(
                _CiReproCapabilitySpec(
                    capability_id="strict_docflow",
                    summary="Run strict docflow with required GH-reference mode.",
                    source_alternatives=(("gabion docflow",),),
                ),
                _CiReproCapabilitySpec(
                    capability_id="docflow_packet_loop",
                    summary="Run docflow packetize plus packet enforce with proving tests.",
                    source_alternatives=(
                        (
                            "scripts/policy/docflow_packetize.py",
                            "scripts/policy/docflow_packet_enforce.py",
                        ),
                    ),
                ),
            ),
            artifacts=(
                "artifacts/out/docflow_compliance.json",
                "artifacts/out/docflow_packet_enforcement.json",
            ),
            commands=("scripts/checks.sh --docflow-only",),
        ),
        _CiReproSurfaceSpec(
            surface_id="local_script:scripts/checks.sh:tests",
            surface_kind="local_verification_lane",
            title="Narrow local test verification lane",
            summary="Fast local verification path for the pytest/JUnit gate.",
            source_ref="scripts/checks.sh",
            mode="tests-only",
            capabilities=(
                _CiReproCapabilitySpec(
                    capability_id="pytest_junit",
                    summary="Run pytest with JUnit output in the fast local verification lane.",
                    source_alternatives=(("pytest", "--junitxml"),),
                ),
            ),
            artifacts=("artifacts/test_runs/junit.xml",),
            commands=("scripts/checks.sh --tests-only",),
        ),
        _CiReproSurfaceSpec(
            surface_id="local_script:scripts/ci/ci_cycle.py:watch",
            surface_kind="remote_watch_lane",
            title="Push-and-watch CI cycle helper",
            summary="Local helper that creates a no-op correction unit and watches remote CI through gabion ci-watch.",
            source_ref="scripts/ci/ci_cycle.py",
            mode="watch",
            capabilities=(
                _CiReproCapabilitySpec(
                    capability_id="git_push",
                    summary="Push the local correction unit to the remote stage branch.",
                    source_alternatives=(("git", "push"),),
                    command_alternatives=(("--push",),),
                ),
                _CiReproCapabilitySpec(
                    capability_id="ci_watch",
                    summary="Watch the remote CI workflow after pushing the correction unit.",
                    source_alternatives=(("gabion", "ci-watch"),),
                    command_alternatives=(("--watch",),),
                ),
            ),
            artifacts=("artifacts/out/ci_watch",),
            commands=("python scripts/ci/ci_cycle.py --push --watch",),
        ),
    )
    source_text_by_surface = {
        "workflow:ci.yml:checks": _workflow_job_run_text(ci_doc, job_name="checks"),
        "workflow:ci.yml:dataflow-grammar": _workflow_job_run_text(
            ci_doc,
            job_name="dataflow-grammar",
        ),
        "workflow:pr-dataflow-grammar.yml:dataflow-grammar": _workflow_job_run_text(
            pr_doc,
            job_name="dataflow-grammar",
        ),
        "tooling_command:gabion:ci-local-repro:checks": ci_local_repro_runtime_text,
        "tooling_command:gabion:ci-local-repro:dataflow": ci_local_repro_runtime_text,
        "tooling_command:gabion:ci-local-repro:pr-dataflow": ci_local_repro_runtime_text,
        "local_script:scripts/ci_local_repro.sh:wrapper": ci_local_repro_text,
        "local_script:scripts/checks.sh:dataflow": checks_text,
        "local_script:scripts/checks.sh:docflow": checks_text,
        "local_script:scripts/checks.sh:tests": checks_text,
        "local_script:scripts/ci/ci_cycle.py:watch": ci_cycle_text,
    }
    surface_payloads = [
        _surface_payload(
            spec=spec,
            source_text=source_text_by_surface.get(spec.surface_id, ""),
        )
        for spec in surfaces
    ]
    surface_status = {
        str(item["surface_id"]): str(item["status"])
        for item in surface_payloads
    }
    relations = (
        _CiReproRelationSpec(
            relation_id="ci-repro:local-checks->workflow-checks",
            relation_kind="reproduces",
            source_surface_id="tooling_command:gabion:ci-local-repro:checks",
            target_surface_id="workflow:ci.yml:checks",
            summary="gabion ci-local-repro --checks-only reproduces the ci.yml checks job locally.",
        ),
        _CiReproRelationSpec(
            relation_id="ci-repro:local-dataflow->workflow-dataflow",
            relation_kind="reproduces",
            source_surface_id="tooling_command:gabion:ci-local-repro:dataflow",
            target_surface_id="workflow:ci.yml:dataflow-grammar",
            summary="gabion ci-local-repro --dataflow-only reproduces the ci.yml dataflow-grammar job locally.",
        ),
        _CiReproRelationSpec(
            relation_id="ci-repro:local-pr-dataflow->workflow-pr-dataflow",
            relation_kind="reproduces",
            source_surface_id="tooling_command:gabion:ci-local-repro:pr-dataflow",
            target_surface_id="workflow:pr-dataflow-grammar.yml:dataflow-grammar",
            summary="gabion ci-local-repro --pr-dataflow-only reproduces the PR dataflow workflow locally.",
        ),
        _CiReproRelationSpec(
            relation_id="ci-repro:script-wrapper->local-checks",
            relation_kind="dispatches",
            source_surface_id="local_script:scripts/ci_local_repro.sh:wrapper",
            target_surface_id="tooling_command:gabion:ci-local-repro:checks",
            summary="scripts/ci_local_repro.sh dispatches to gabion ci-local-repro for checks parity.",
        ),
        _CiReproRelationSpec(
            relation_id="ci-repro:script-wrapper->local-dataflow",
            relation_kind="dispatches",
            source_surface_id="local_script:scripts/ci_local_repro.sh:wrapper",
            target_surface_id="tooling_command:gabion:ci-local-repro:dataflow",
            summary="scripts/ci_local_repro.sh dispatches to gabion ci-local-repro for dataflow parity.",
        ),
        _CiReproRelationSpec(
            relation_id="ci-repro:script-wrapper->local-pr-dataflow",
            relation_kind="dispatches",
            source_surface_id="local_script:scripts/ci_local_repro.sh:wrapper",
            target_surface_id="tooling_command:gabion:ci-local-repro:pr-dataflow",
            summary="scripts/ci_local_repro.sh dispatches to gabion ci-local-repro for PR dataflow parity.",
        ),
        _CiReproRelationSpec(
            relation_id="ci-repro:checks-dataflow->local-checks",
            relation_kind="supports",
            source_surface_id="local_script:scripts/checks.sh:dataflow",
            target_surface_id="tooling_command:gabion:ci-local-repro:checks",
            summary="scripts/checks.sh --dataflow-only is a narrower local dataflow verification lane supporting the broader CI reproduction loop.",
        ),
        _CiReproRelationSpec(
            relation_id="ci-repro:checks-docflow->local-checks",
            relation_kind="supports",
            source_surface_id="local_script:scripts/checks.sh:docflow",
            target_surface_id="tooling_command:gabion:ci-local-repro:checks",
            summary="scripts/checks.sh --docflow-only is a narrower local docflow verification lane supporting the broader CI reproduction loop.",
        ),
        _CiReproRelationSpec(
            relation_id="ci-repro:checks-tests->local-checks",
            relation_kind="supports",
            source_surface_id="local_script:scripts/checks.sh:tests",
            target_surface_id="tooling_command:gabion:ci-local-repro:checks",
            summary="scripts/checks.sh --tests-only is a narrower local test verification lane supporting the broader CI reproduction loop.",
        ),
        _CiReproRelationSpec(
            relation_id="ci-repro:ci-cycle->workflow-ci",
            relation_kind="observes",
            source_surface_id="local_script:scripts/ci/ci_cycle.py:watch",
            target_surface_id="workflow:ci.yml:checks",
            summary="scripts/ci/ci_cycle.py --push --watch drives the remote status-watch half of the dual-sensor CI loop.",
        ),
    )
    relation_payloads = []
    for relation in relations:
        source_status = surface_status.get(relation.source_surface_id, "fail")
        target_status = surface_status.get(relation.target_surface_id, "fail")
        relation_payloads.append(
            {
                "relation_id": relation.relation_id,
                "relation_kind": relation.relation_kind,
                "source_surface_id": relation.source_surface_id,
                "target_surface_id": relation.target_surface_id,
                "source_missing_capability_ids": next(
                    (
                        list(item["missing_capability_ids"])
                        for item in surface_payloads
                        if item["surface_id"] == relation.source_surface_id
                    ),
                    [],
                ),
                "target_missing_capability_ids": next(
                    (
                        list(item["missing_capability_ids"])
                        for item in surface_payloads
                        if item["surface_id"] == relation.target_surface_id
                    ),
                    [],
                ),
                "status": "pass"
                if source_status == "pass" and target_status == "pass"
                else "fail",
                "summary": relation.summary,
            }
        )
    return {
        "schema_version": 2,
        "artifact_kind": "local_ci_repro_contract",
        "generated_by": "scripts/policy/policy_check.py --workflows",
        "summary": (
            "Declarative local/remote CI reproduction topology for workflow parity, "
            "bounded local verification lanes, and remote status-watch orchestration."
        ),
        "surfaces": surface_payloads,
        "relations": relation_payloads,
    }


def _write_local_ci_repro_contract_artifact(
    *,
    output_path: Path = _LOCAL_CI_REPRO_CONTRACT,
    repo_root: Path = REPO_ROOT,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _build_local_ci_repro_contract_payload(repo_root=repo_root)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_workflow_governance_artifacts(*, errors: list[str], output_root: Path = _WORKFLOW_POLICY_OUTPUT_ROOT) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    governance_report_path = output_root / _QUOTIENT_GOVERNANCE_REPORT.name
    ratchet_delta_path = output_root / _QUOTIENT_RATCHET_DELTA.name
    policy_violations_path = output_root / _QUOTIENT_POLICY_VIOLATIONS.name
    readiness_path = output_root / _QUOTIENT_PROTOCOL_READINESS.name
    promotion_path = output_root / _QUOTIENT_PROMOTION_DECISION.name
    demotion_path = output_root / _QUOTIENT_DEMOTION_INCIDENTS.name
    normalized_errors = _sorted(errors)
    violations = []
    reason_counts: dict[str, int] = {}
    for idx, message in enumerate(normalized_errors, start=1):
        check_deadline()
        reason_code = _workflow_reason_code(message)
        reason_counts[reason_code] = reason_counts.get(reason_code, 0) + 1
        violations.append(
            {
                "violation_id": f"WFV-{idx:04d}",
                "clause_id": "F57-4",
                "severity": "error",
                "reason_code": reason_code,
                "message": message,
            }
        )

    previous_violation_count = 0
    if governance_report_path.exists():
        try:
            previous_payload = json.loads(governance_report_path.read_text(encoding="utf-8"))
            metrics = previous_payload.get("metrics", {}) if isinstance(previous_payload, dict) else {}
            if isinstance(metrics, dict):
                previous_violation_count = int(metrics.get("violation_count", 0))
        except (OSError, ValueError, TypeError):
            previous_violation_count = 0

    violation_count = len(normalized_errors)
    readiness_pass = violation_count == 0
    sorted_reason_counts = {
        key: reason_counts[key]
        for key in _sorted(reason_counts)
    }
    governance_report = {
        "profile_id": "workflow_policy.enforce",
        "decision": "pass" if readiness_pass else "fail",
        "metrics": {
            "violation_count": violation_count,
            "reason_code_counts": sorted_reason_counts,
        },
        "sources": {
            "workflows": ".github/workflows/*.yml",
            "policy_check": "scripts/policy/policy_check.py --workflows",
        },
    }
    ratchet_delta = {
        "baseline_id": "workflow_policy.previous_report",
        "previous_violation_count": previous_violation_count,
        "current_violation_count": violation_count,
        "violation_count_delta": violation_count - previous_violation_count,
        "reason_code_counts": sorted_reason_counts,
    }
    policy_violations = {
        "profile_id": "workflow_policy.enforce",
        "violations": violations,
    }
    readiness_payload = {
        "profile": "enforce",
        "gate_predicates": [
            {
                "gate_id": "governance.workflow_policy_green",
                "pass": readiness_pass,
                "clause_id": "F58-1",
                "source_artifact": "out/quotient_governance_report.json",
                "reason_codes": _sorted(reason_counts) if not readiness_pass else [],
            }
        ],
        "pass": readiness_pass,
        "details": {
            "violation_count": violation_count,
        },
    }
    promotion_payload = {
        "decision": "promote" if readiness_pass else "hold",
        "target_profile": "strict-core",
        "rationale_codes": [] if readiness_pass else _sorted(reason_counts),
        "source_artifact": "out/quotient_protocol_readiness.json",
    }
    demotion_payload = {
        "incidents": []
        if readiness_pass
        else [
            {
                "trigger_id": "workflow_policy.nonzero_violation_count",
                "impacted_profile": "strict-core",
                "reason_codes": _sorted(reason_counts),
                "closure_status": "open",
            }
        ]
    }

    governance_report_path.write_text(json.dumps(governance_report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    ratchet_delta_path.write_text(json.dumps(ratchet_delta, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    policy_violations_path.write_text(json.dumps(policy_violations, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    readiness_path.write_text(json.dumps(readiness_payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    promotion_path.write_text(json.dumps(promotion_payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    demotion_path.write_text(json.dumps(demotion_payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class JobContext:
    job_name: str
    path: Path


def _fail(errors):
    global _LAST_FAIL_ERRORS
    _LAST_FAIL_ERRORS = list(errors)
    for err in errors:
        check_deadline()
        print(f"policy-check: {err}", file=sys.stderr)
    raise SystemExit(2)


def _load_allowed_actions(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        raw = path.read_text()
    except OSError:
        return set()
    allowed: set[str] = set()
    for line in raw.splitlines():
        check_deadline()
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        allowed.add(line)
    return allowed


def _yaml_loader():
    if yaml is None:
        return None

    class Loader(yaml.SafeLoader):
        pass

    # Treat "on"/"off"/"yes"/"no" as strings, not booleans (YAML 1.1 quirk).
    for key, values in list(Loader.yaml_implicit_resolvers.items()):
        check_deadline()
        Loader.yaml_implicit_resolvers[key] = [
            (tag, regexp) for tag, regexp in values if tag != "tag:yaml.org,2002:bool"
        ]
    return Loader


def _load_yaml(path):
    if yaml is None:
        _fail(["PyYAML is required for policy checks (pip install pyyaml)."])
    with path.open("r", encoding="utf-8") as handle:
        loader = _yaml_loader()
        return yaml.load(handle, Loader=loader) or {}


def _normalize_if(expr):
    if not expr:
        return ""
    return re.sub(r"\s+", "", str(expr))


def _runs_on_labels(runs_on):
    if runs_on is None:
        return []
    if isinstance(runs_on, str):
        return [runs_on]
    if isinstance(runs_on, list):
        return [str(item) for item in runs_on]
    if isinstance(runs_on, dict):
        labels = runs_on.get("labels")
        if isinstance(labels, list):
            return [str(item) for item in labels]
        if isinstance(labels, str):
            return [labels]
    return []


def _is_self_hosted(runs_on):
    labels = _runs_on_labels(runs_on)
    return "self-hosted" in labels


def _event_names(on_block):
    if isinstance(on_block, str):
        return {on_block}
    if isinstance(on_block, list):
        return {str(item) for item in on_block}
    if isinstance(on_block, dict):
        return {str(key) for key in on_block.keys()}
    return set()


def _has_tag_push(on_block) -> bool:
    if not isinstance(on_block, dict):
        return False
    push_block = on_block.get("push")
    if push_block is None:
        return False
    if isinstance(push_block, dict):
        tags = push_block.get("tags")
        if isinstance(tags, list):
            return len(tags) > 0
        if isinstance(tags, str):
            return True
    return False


def _workflow_run_names(on_block) -> set[str]:
    if not isinstance(on_block, dict):
        return set()
    workflow_run = on_block.get("workflow_run")
    if not isinstance(workflow_run, dict):
        return set()
    workflows = workflow_run.get("workflows")
    if isinstance(workflows, list):
        return {str(item) for item in workflows}
    if isinstance(workflows, str):
        return {workflows}
    return set()


def _push_branches(on_block) -> set[str]:
    if not isinstance(on_block, dict):
        return set()
    push_block = on_block.get("push")
    if not isinstance(push_block, dict):
        return set()
    branches = push_block.get("branches")
    if isinstance(branches, list):
        return {str(item) for item in branches}
    if isinstance(branches, str):
        return {branches}
    return set()


def _push_tags(on_block) -> set[str]:
    if not isinstance(on_block, dict):
        return set()
    push_block = on_block.get("push")
    if not isinstance(push_block, dict):
        return set()
    tags = push_block.get("tags")
    if isinstance(tags, list):
        return {str(item) for item in tags}
    if isinstance(tags, str):
        return {tags}
    return set()


def _is_tag_only_push(on_block) -> bool:
    if not isinstance(on_block, dict):
        return False
    if _event_names(on_block) != {"push"}:
        return False
    push_block = on_block.get("push")
    if not isinstance(push_block, dict):
        return False
    tags = push_block.get("tags")
    if not tags:
        return False
    branches = push_block.get("branches")
    if branches:
        return False
    return True


def _has_actor_guard(cond: str) -> bool:
    return (
        "github.actor==github.repository_owner" in cond
        or "github.actor=='" in cond
        or "github.actor==\"" in cond
    )


def _has_ref_guard(cond: str) -> bool:
    return (
        "github.ref=='refs/heads/" in cond
        or "github.ref==\"refs/heads/" in cond
        or "github.ref=='refs/tags/" in cond
        or "github.ref==\"refs/tags/" in cond
        or "startswith(github.ref,'refs/heads/')" in cond
        or "startsWith(github.ref,'refs/heads/')" in cond
        or "startswith(github.ref,'refs/tags/')" in cond
        or "startsWith(github.ref,'refs/tags/')" in cond
    )


def _check_workflow_dispatch_guards(doc, path, errors):
    events = _event_names(doc.get("on"))
    if "workflow_dispatch" not in events:
        return
    jobs = doc.get("jobs", {})
    if not isinstance(jobs, dict):
        return
    for name, job in jobs.items():
        check_deadline()
        cond = _normalize_if(job.get("if"))
        if not cond:
            errors.append(
                f"{path}:{name}: workflow_dispatch jobs must guard on actor and ref"
            )
            continue
        if not _has_actor_guard(cond):
            errors.append(
                f"{path}:{name}: workflow_dispatch jobs must guard on actor"
            )
        if not _has_ref_guard(cond):
            errors.append(
                f"{path}:{name}: workflow_dispatch jobs must guard on ref"
            )


def _check_permissions(
    doc,
    path,
    errors,
    *,
    allow_pr_write=False,
    allow_id_token=False,
    allow_contents_write=False,
    require_id_token=False,
    require_actions_read=False,
):
    # dataflow-bundle: doc, errors
    permissions = doc.get("permissions")
    if permissions is None:
        errors.append(f"{path}: missing top-level permissions")
        return
    if isinstance(permissions, str):
        errors.append(f"{path}: permissions must be a mapping, not {permissions!r}")
        return
    contents = permissions.get("contents")
    if allow_contents_write:
        if contents != "write":
            errors.append(f"{path}: permissions.contents must be 'write'")
    elif contents != "read":
        errors.append(f"{path}: permissions.contents must be 'read'")
    if require_id_token and permissions.get("id-token") != "write":
        errors.append(f"{path}: permissions.id-token must be 'write'")
    if require_actions_read and permissions.get("actions") != "read":
        errors.append(f"{path}: permissions.actions must be 'read'")
    for key, value in permissions.items():
        check_deadline()
        if key == "contents":
            continue
        if value in ("read", "none"):
            continue
        if allow_pr_write and key == "pull-requests" and value == "write":
            continue
        if allow_id_token and key == "id-token" and value == "write":
            continue
        if allow_contents_write and key == "contents" and value == "write":
            continue
        errors.append(f"{path}: permissions.{key} must be 'read' or 'none'")


def _check_job_permissions(
    job,
    job_ctx: JobContext,
    errors,
    *,
    allow_pr_write=False,
    allow_id_token=False,
    allow_contents_write=False,
    require_id_token=False,
    require_actions_read=False,
):
    # dataflow-bundle: errors, job, job_ctx
    permissions = job.get("permissions")
    if permissions is None:
        return
    if isinstance(permissions, str):
        errors.append(
            f"{job_ctx.path}:{job_ctx.job_name}: job permissions must be a mapping"
        )
        return
    contents = permissions.get("contents")
    if allow_contents_write:
        if contents != "write":
            errors.append(
                f"{job_ctx.path}:{job_ctx.job_name}: permissions.contents must be 'write'"
            )
    elif contents != "read":
        errors.append(
            f"{job_ctx.path}:{job_ctx.job_name}: permissions.contents must be 'read'"
        )
    if require_id_token and permissions.get("id-token") != "write":
        errors.append(
            f"{job_ctx.path}:{job_ctx.job_name}: permissions.id-token must be 'write'"
        )
    if require_actions_read and permissions.get("actions") != "read":
        errors.append(
            f"{job_ctx.path}:{job_ctx.job_name}: permissions.actions must be 'read'"
        )
    is_self_hosted = _is_self_hosted(job.get("runs-on"))
    for key, value in permissions.items():
        check_deadline()
        if key == "contents":
            continue
        if value in ("read", "none"):
            continue
        if (
            allow_pr_write
            and not is_self_hosted
            and key == "pull-requests"
            and value == "write"
        ):
            continue
        if allow_id_token and not is_self_hosted and key == "id-token" and value == "write":
            continue
        if allow_contents_write and key == "contents" and value == "write":
            continue
        errors.append(
            f"{job_ctx.path}:{job_ctx.job_name}: permissions.{key} must be 'read' or 'none'"
        )


def _check_release_tag_workflow(doc, path, errors):
    events = _event_names(doc.get("on"))
    if events != {"workflow_dispatch"}:
        errors.append(f"{path}: release tag workflow must use workflow_dispatch only")
    jobs = doc.get("jobs", {})
    if not isinstance(jobs, dict):
        return
    for name, job in jobs.items():
        check_deadline()
        if _is_self_hosted(job.get("runs-on")):
            errors.append(f"{path}:{name}: release tag workflow must not use self-hosted")
        cond = _normalize_if(job.get("if"))
        if (
            "github.ref=='refs/heads/release'" not in cond
            and "github.ref=='refs/heads/next'" not in cond
        ):
            errors.append(
                f"{path}:{name}: release tag workflow must guard on refs/heads/release or refs/heads/next"
            )
        if "github.actor==github.repository_owner" not in cond:
            errors.append(
                f"{path}:{name}: release tag workflow must guard on repository owner"
            )
        steps = job.get("steps", [])
        if not _step_run_contains_any(steps, {"scripts/release/release_tag.py"}):
            errors.append(
                f"{path}:{name}: release tag workflow must use scripts/release/release_tag.py"
            )
    script_path = REPO_ROOT / "scripts" / "release" / "release_tag.py"
    if script_path.exists():
        script_text = script_path.read_text(encoding="utf-8")
        required_tokens = [
            "Next branch must mirror main",
            "Release branch must mirror next",
            "Tag already exists",
            "Test tags must be created from next",
            "Release tags must be created from release",
            "tag_target",
        ]
        missing = [token for token in required_tokens if token not in script_text]
        if missing:
            errors.append(
                f"{path}: release_tag.py missing required guard tokens: {missing}"
            )
    else:
        errors.append(f"{path}: release_tag.py missing (required for release tagging)")


def _check_auto_test_tag_workflow(doc, path, errors):
    events = _event_names(doc.get("on"))
    if events != {"workflow_run"}:
        errors.append(f"{path}: auto test tag workflow must use workflow_run only")
    workflows = _workflow_run_names(doc.get("on"))
    if not workflows:
        errors.append(f"{path}: auto test tag workflow must specify workflows")
    elif "mirror-next" not in workflows:
        errors.append(f"{path}: auto test tag workflow must target mirror-next")
    jobs = doc.get("jobs", {})
    if not isinstance(jobs, dict):
        return
    for name, job in jobs.items():
        check_deadline()
        if _is_self_hosted(job.get("runs-on")):
            errors.append(f"{path}:{name}: auto test tag workflow must not use self-hosted")
        cond = _normalize_if(job.get("if"))
        if "github.event.workflow_run.conclusion=='success'" not in cond:
            errors.append(f"{path}:{name}: auto test tag workflow must guard on success")
        if "github.event.workflow_run.head_branch=='main'" not in cond:
            errors.append(f"{path}:{name}: auto test tag workflow must guard on main")
        if (
            "github.event.workflow_run.actor.login==github.repository_owner" not in cond
            and "github.event.workflow_run.actor.login=='github-actions[bot]'" not in cond
        ):
            errors.append(f"{path}:{name}: auto test tag workflow must guard on actor")
        steps = job.get("steps", [])
        if not _step_run_contains_tokens(
            steps, {"origin/main", "origin/next", "Next must mirror main"}
        ):
            errors.append(
                f"{path}:{name}: auto test tag workflow must verify next mirrors main"
            )
        if not _step_run_contains_any(steps, {"scripts/release/release_read_project_version.py"}):
            errors.append(
                f"{path}:{name}: auto test tag workflow must derive tag from pyproject.toml"
            )
        if not _step_run_contains_any(steps, {"steps.verify.outputs.target_sha"}):
            errors.append(
                f"{path}:{name}: auto test tag workflow must tag verified target_sha"
            )
        if not _step_run_contains_tokens(steps, {"rev-parse", "-q", "--verify", "refs/tags"}):
            errors.append(
                f"{path}:{name}: auto test tag workflow must guard against existing tags"
            )
        if not _step_run_contains_any(steps, {"test-v"}):
            errors.append(f"{path}:{name}: auto test tag workflow must create test-v tags")


def _check_mirror_next_workflow(doc, path, errors):
    events = _event_names(doc.get("on"))
    if events != {"push"}:
        errors.append(f"{path}: mirror workflow must use push only")
    branches = _push_branches(doc.get("on"))
    if branches != {"main"}:
        errors.append(f"{path}: mirror workflow must target main branch pushes")
    tags = _push_tags(doc.get("on"))
    if tags:
        errors.append(f"{path}: mirror workflow must not trigger on tags")
    jobs = doc.get("jobs", {})
    if not isinstance(jobs, dict):
        return
    for name, job in jobs.items():
        check_deadline()
        if _is_self_hosted(job.get("runs-on")):
            errors.append(f"{path}:{name}: mirror workflow must not use self-hosted")
        cond = _normalize_if(job.get("if"))
        if "github.ref=='refs/heads/main'" not in cond:
            errors.append(f"{path}:{name}: mirror workflow must guard on main")
        if "github.actor==github.repository_owner" not in cond:
            errors.append(
                f"{path}:{name}: mirror workflow must guard on repository owner"
            )
        steps = job.get("steps", [])
        if not _step_run_contains_tokens(
            steps, {"merge-base", "--is-ancestor", "origin/next", "origin/main"}
        ):
            errors.append(
                f"{path}:{name}: mirror workflow must verify next is ancestor of main"
            )
        if not _step_run_contains_tokens(
            steps, {"--force-with-lease=refs/heads/next:", "refs/heads/next"}
        ):
            errors.append(
                f"{path}:{name}: mirror workflow must use --force-with-lease for branch update"
            )
        if not _step_run_contains_any(steps, {"HEAD_SHA\":refs/heads/next", "HEAD_SHA:refs/heads/next"}):
            errors.append(
                f"{path}:{name}: mirror workflow must push explicit HEAD_SHA to next"
            )


def _check_promote_release_workflow(doc, path, errors):
    events = _event_names(doc.get("on"))
    if events != {"workflow_run"}:
        errors.append(f"{path}: promote workflow must use workflow_run only")
    workflows = _workflow_run_names(doc.get("on"))
    if not workflows:
        errors.append(f"{path}: promote workflow must specify workflows")
    elif "release-testpypi" not in workflows:
        errors.append(f"{path}: promote workflow must target release-testpypi")
    jobs = doc.get("jobs", {})
    if not isinstance(jobs, dict):
        return
    for name, job in jobs.items():
        check_deadline()
        if _is_self_hosted(job.get("runs-on")):
            errors.append(f"{path}:{name}: promote workflow must not use self-hosted")
        cond = _normalize_if(job.get("if"))
        if "github.event.workflow_run.conclusion=='success'" not in cond:
            errors.append(f"{path}:{name}: promote workflow must guard on success")
        steps = job.get("steps", [])
        has_tag_artifact = False
        if isinstance(steps, list):
            for step in steps:
                check_deadline()
                uses = step.get("uses", "")
                if uses.startswith("actions/download-artifact@"):
                    with_block = step.get("with", {}) or {}
                    if with_block.get("name") == "release-testpypi-tag":
                        has_tag_artifact = True
                        break
                run = step.get("run", "")
                if "release-testpypi/tag.txt" in run:
                    has_tag_artifact = True
                    break
        if (
            "startswith(github.event.workflow_run.head_branch,'test-v')" not in cond
            and "startsWith(github.event.workflow_run.head_branch,'test-v')" not in cond
            and not has_tag_artifact
        ):
            errors.append(f"{path}:{name}: promote workflow must guard on test-v tags")
        if (
            "github.event.workflow_run.actor.login==github.repository_owner" not in cond
            and "github.event.workflow_run.actor.login=='github-actions[bot]'" not in cond
        ):
            errors.append(
                f"{path}:{name}: promote workflow must guard on repository owner or github-actions[bot]"
            )
        steps = job.get("steps", [])
        if not _step_run_contains_tokens(
            steps, {"merge-base", "--is-ancestor", "origin/main"}
        ):
            errors.append(
                f"{path}:{name}: promote workflow must verify tested commit is ancestor of main"
            )
        if not _step_run_contains_tokens(
            steps, {"NEXT_SHA", "HEAD_SHA", "Next must mirror tested commit"}
        ):
            errors.append(
                f"{path}:{name}: promote workflow must verify tested commit matches next"
            )
        if not _step_run_contains_tokens(
            steps, {"--force-with-lease=refs/heads/release:", "refs/heads/release"}
        ):
            errors.append(
                f"{path}:{name}: promote workflow must use --force-with-lease for branch update"
            )
        if not _step_run_contains_any(steps, {"HEAD_SHA\":refs/heads/release", "HEAD_SHA:refs/heads/release"}):
            errors.append(
                f"{path}:{name}: promote workflow must push explicit HEAD_SHA to release"
            )


def _step_run_contains_tokens(steps, tokens: set[str]) -> bool:
    for step in steps:
        check_deadline()
        if not isinstance(step, dict):
            continue
        run = step.get("run")
        if not isinstance(run, str):
            continue
        if all(token in run for token in tokens):
            return True
    return False


def _step_run_contains_any(steps, tokens: set[str]) -> bool:
    for step in steps:
        check_deadline()
        if not isinstance(step, dict):
            continue
        run = step.get("run")
        if not isinstance(run, str):
            continue
        if any(token in run for token in tokens):
            return True
    return False




def _check_ci_script_entrypoints(doc, path, errors):
    check_deadline()
    if path.name != "ci.yml":
        return
    jobs = doc.get("jobs", {})
    if not isinstance(jobs, dict):
        return
    required_tokens = {
        "scripts/ci/ci_finalize_dataflow_outcome.py",
        "scripts/ci/ci_controller_drift_gate.py",
        "scripts/ci/ci_override_record_emit.py",
        "scripts/policy/policy_scanner_suite.py",
        "scripts/misc/aspf_handoff.py run",
        "check delta-bundle",
        "check delta-gates",
    }
    steps = []
    for job in jobs.values():
        check_deadline()
        if not isinstance(job, dict):
            continue
        raw_steps = job.get("steps", [])
        if isinstance(raw_steps, list):
            steps.extend(raw_steps)
    for token in sorted(required_tokens):
        check_deadline()
        if not _step_run_contains_any(steps, {token}):
            errors.append(f"{path}: ci workflow must invoke {token}")


def _check_policy_scanner_suite_entrypoints(doc, path, errors):
    check_deadline()
    if path.name not in {"ci.yml", "pr-dataflow-grammar.yml"}:
        return
    jobs = doc.get("jobs", {})
    if not isinstance(jobs, dict):
        return
    steps = []
    for job in jobs.values():
        check_deadline()
        if not isinstance(job, dict):
            continue
        raw_steps = job.get("steps", [])
        if isinstance(raw_steps, list):
            steps.extend(raw_steps)
    if not _step_run_contains_any(steps, {"scripts/policy/policy_scanner_suite.py"}):
        errors.append(f"{path}: workflow must invoke scripts/policy/policy_scanner_suite.py")
    required_policy_check_tokens = {
        "scripts/policy/policy_check.py",
        "--workflows",
        "artifacts/out/policy_check_result.json",
    }
    if not _step_run_contains_tokens(steps, required_policy_check_tokens):
        errors.append(
            f"{path}: workflow must invoke scripts/policy/policy_check.py --workflows "
            "--output artifacts/out/policy_check_result.json before "
            "scripts/policy/policy_scanner_suite.py"
        )
    deprecated_scanner_tokens = {
        "scripts/no_monkeypatch_policy_check.py",
        "scripts/branchless_policy_check.py",
        "scripts/defensive_fallback_policy_check.py",
    }
    for token in sorted(deprecated_scanner_tokens):
        check_deadline()
        if _step_run_contains_any(steps, {token}):
            errors.append(
                f"{path}: workflow must not invoke legacy scanner entrypoint {token}; "
                "use scripts/policy/policy_scanner_suite.py"
            )


def _check_pr_stage_ci_poll_cadence(doc, path, errors):
    check_deadline()
    if path.name != "pr-dataflow-grammar.yml":
        return
    jobs = doc.get("jobs", {})
    if not isinstance(jobs, dict):
        return
    steps = []
    for job in jobs.values():
        check_deadline()
        if not isinstance(job, dict):
            continue
        raw_steps = job.get("steps", [])
        if isinstance(raw_steps, list):
            steps.extend(raw_steps)
    if not _step_run_contains_any(steps, {"time.sleep(60)"}):
        errors.append(
            f"{path}: stage CI verification polling cadence must be at least one request per minute"
        )


def _check_dense_core_lock_in(errors):
    check_deadline()
    run_dataflow_stage_path = REPO_ROOT / "src/gabion/tooling/runtime/run_dataflow_stage.py"
    finalize_path = REPO_ROOT / "scripts/ci/ci_finalize_dataflow_outcome.py"
    try:
        run_dataflow_stage_source = run_dataflow_stage_path.read_text(encoding="utf-8")
    except OSError:
        errors.append(f"{run_dataflow_stage_path}: unable to read lock-in source")
        return
    try:
        finalize_source = finalize_path.read_text(encoding="utf-8")
    except OSError:
        errors.append(f"{finalize_path}: unable to read lock-in source")
        return

    required_stage_tokens = {
        "_parse_stage_command_envelope(command)",
        "invocation_runner.run_delta_bundle(envelope).exit_code",
    }
    for token in sorted(required_stage_tokens):
        check_deadline()
        if token not in run_dataflow_stage_source:
            errors.append(
                f"{run_dataflow_stage_path}: dense-core lock-in missing token {token!r}"
            )

    required_terminal_tokens = {
        "terminal_outcome_projector.read_terminal_outcome_artifact",
        "terminal_outcome_projector.project_terminal_outcome",
    }
    for token in sorted(required_terminal_tokens):
        check_deadline()
        if token not in finalize_source:
            errors.append(
                f"{finalize_path}: terminal projector lock-in missing token {token!r}"
            )

def _check_release_testpypi_workflow(doc, path, errors):
    on_block = doc.get("on")
    workflows = _workflow_run_names(on_block)
    if workflows and "auto-test-tag" not in workflows:
        errors.append(f"{path}: release-testpypi workflow_run must target auto-test-tag")
    tags = _push_tags(on_block)
    if "push" in _event_names(on_block) and not tags:
        errors.append(f"{path}: release-testpypi push trigger must include tag patterns")
    if tags and not any("test-v" in tag for tag in tags):
        errors.append(f"{path}: release-testpypi push tags must include test-v*")
    jobs = doc.get("jobs", {})
    if not isinstance(jobs, dict):
        return
    required_tokens = {
        "refs/tags/$TAG",
        "origin/main",
        "origin/next",
        "TAG_SHA",
    }
    required_scripts = {
        "scripts/release/release_verify_test_tag.py",
    }
    for name, job in jobs.items():
        check_deadline()
        cond = _normalize_if(job.get("if"))
        if "github.event.workflow_run.conclusion=='success'" not in cond:
            errors.append(
                f"{path}:{name}: release-testpypi workflow_run must guard on success"
            )
        if (
            "github.event.workflow_run.actor.login==github.repository_owner" not in cond
            and "github.event.workflow_run.actor.login=='github-actions[bot]'" not in cond
        ):
            errors.append(
                f"{path}:{name}: release-testpypi workflow_run must guard on actor"
            )
        steps = job.get("steps", [])
        if not (
            _step_run_contains_tokens(steps, required_tokens)
            or _step_run_contains_any(steps, required_scripts)
        ):
            errors.append(
                f"{path}:{name}: release-testpypi workflow must verify tag equals main/next"
            )


def _check_release_pypi_workflow(doc, path, errors):
    on_block = doc.get("on")
    workflows = _workflow_run_names(on_block)
    if workflows and "release-tag" not in workflows:
        errors.append(f"{path}: release-pypi workflow_run must target release-tag")
    tags = _push_tags(on_block)
    if "push" in _event_names(on_block) and not tags:
        errors.append(f"{path}: release-pypi push trigger must include tag patterns")
    if tags and not any(tag == "v*" or tag.startswith("v") for tag in tags):
        errors.append(f"{path}: release-pypi push tags must include v*")
    if tags and not any("!test-v" in tag for tag in tags):
        errors.append(f"{path}: release-pypi push tags must exclude test-v*")
    jobs = doc.get("jobs", {})
    if not isinstance(jobs, dict):
        return
    required_tokens = {
        "origin/main",
        "origin/next",
        "origin/release",
        "TAG_SHA",
    }
    required_scripts = {
        "scripts/release/release_verify_pypi_tag.py",
    }
    for name, job in jobs.items():
        check_deadline()
        cond = _normalize_if(job.get("if"))
        if "github.event.workflow_run.conclusion=='success'" not in cond:
            errors.append(
                f"{path}:{name}: release-pypi workflow_run must guard on success"
            )
        if (
            "github.event.workflow_run.actor.login==github.repository_owner" not in cond
            and "github.event.workflow_run.actor.login=='github-actions[bot]'" not in cond
        ):
            errors.append(
                f"{path}:{name}: release-pypi workflow_run must guard on actor"
            )
        steps = job.get("steps", [])
        if not (
            _step_run_contains_tokens(steps, required_tokens)
            or _step_run_contains_any(steps, required_scripts)
        ):
            errors.append(
                f"{path}:{name}: release-pypi workflow must verify tag equals main/next/release"
            )


def _job_uses_action(job, action_prefix: str) -> bool:
    steps = job.get("steps", [])
    for step in steps:
        check_deadline()
        if not isinstance(step, dict):
            continue
        uses = step.get("uses")
        if isinstance(uses, str) and uses.startswith(action_prefix):
            return True
    return False


def _job_has_same_repo_pr_guard(job) -> bool:
    guard_tokens = {
        "github.event.pull_request.head.repo.full_name==github.repository",
        "github.event.pull_request.head.repo.fork==false",
    }
    cond = _normalize_if(job.get("if"))
    if any(token in cond for token in guard_tokens):
        return True
    for step in job.get("steps", []):
        check_deadline()
        if not isinstance(step, dict):
            continue
        cond = _normalize_if(step.get("if"))
        if any(token in cond for token in guard_tokens):
            return True
    return False


def _check_pr_comment_guard(doc, path, errors):
    jobs = doc.get("jobs", {})
    if not isinstance(jobs, dict):
        return
    for name, job in jobs.items():
        check_deadline()
        permissions = job.get("permissions")
        if not isinstance(permissions, dict):
            continue
        if permissions.get("pull-requests") != "write":
            continue
        if not _job_has_same_repo_pr_guard(job):
            errors.append(
                f"{path}:{name}: pull-requests write requires same-repo PR guard"
            )


def _check_id_token_scoping(doc, path, errors):
    jobs = doc.get("jobs", {})
    if not isinstance(jobs, dict):
        return
    if len(jobs) <= 1:
        return
    top_perms = doc.get("permissions")
    if isinstance(top_perms, dict) and top_perms.get("id-token") == "write":
        errors.append(
            f"{path}: id-token write must be scoped to publishing jobs when multiple jobs are present"
        )
    for name, job in jobs.items():
        check_deadline()
        permissions = job.get("permissions")
        has_id_token = isinstance(permissions, dict) and permissions.get("id-token") == "write"
        uses_publish = _job_uses_action(job, "pypa/gh-action-pypi-publish@")
        if uses_publish and not has_id_token:
            errors.append(
                f"{path}:{name}: publishing job must request id-token: write"
            )
        if has_id_token and not uses_publish:
            errors.append(
                f"{path}:{name}: only publishing jobs may request id-token: write"
            )


def _check_actions(job, job_ctx: JobContext, errors, allowed_actions: set[str]):
    # dataflow-bundle: errors, job, job_ctx
    steps = job.get("steps", [])
    for idx, step in enumerate(steps):
        check_deadline()
        uses = step.get("uses")
        if not uses:
            continue
        if uses.startswith("./") or uses.startswith("docker://"):
            continue
        if "@" not in uses:
            errors.append(
                f"{job_ctx.path}:{job_ctx.job_name}: step {idx} uses unpinned action {uses!r}"
            )
            continue
        action_ref, ref = uses.split("@", 1)
        action_parts = action_ref.split("/")
        if len(action_parts) < 2:
            errors.append(
                f"{job_ctx.path}:{job_ctx.job_name}: step {idx} invalid action {uses!r}"
            )
            continue
        action_name = "/".join(action_parts[:2])
        if action_name not in allowed_actions:
            errors.append(
                f"{job_ctx.path}:{job_ctx.job_name}: step {idx} action not allow-listed ({action_name})"
            )
        if not _SHA_RE.match(ref):
            errors.append(
                f"{job_ctx.path}:{job_ctx.job_name}: step {idx} action not pinned to full SHA ({uses})"
            )


def _check_self_hosted_constraints(doc, path, errors):
    # dataflow-bundle: doc, errors
    jobs = doc.get("jobs", {})
    if not isinstance(jobs, dict):
        return
    self_hosted_jobs = []
    for name, job in jobs.items():
        check_deadline()
        if _is_self_hosted(job.get("runs-on")):
            self_hosted_jobs.append((name, job))

    if not self_hosted_jobs:
        return

    on_block = doc.get("on")
    events = _event_names(on_block)
    if events != {"push"}:
        errors.append(f"{path}: self-hosted workflow must trigger only on push")
    push_block = None
    if isinstance(on_block, dict):
        push_block = on_block.get("push")
    branches = None
    if isinstance(push_block, dict):
        branches = push_block.get("branches")
        tags = push_block.get("tags")
        if tags:
            errors.append(f"{path}: self-hosted workflow must not trigger on tags")
    if not branches:
        errors.append(f"{path}: push trigger must restrict branches")
    else:
        branch_set = {str(item) for item in branches}
        if not branch_set.issubset(TRUSTED_BRANCHES):
            errors.append(
                f"{path}: push branches must be subset of {_sorted(TRUSTED_BRANCHES)}"
            )

    for name, job in self_hosted_jobs:
        check_deadline()
        runs_on = job.get("runs-on")
        labels = set(_runs_on_labels(runs_on))
        if not labels:
            errors.append(f"{path}:{name}: runs-on must be explicit labels")
        if not REQUIRED_RUNNER_LABELS.issubset(labels):
            errors.append(
                f"{path}:{name}: runs-on must include {_sorted(REQUIRED_RUNNER_LABELS)}"
            )
        cond = _normalize_if(job.get("if"))
        if "github.actor==github.repository_owner" not in cond:
            errors.append(
                f"{path}:{name}: missing actor guard (github.actor == github.repository_owner)"
            )


def check_workflows():
    allowed_actions = _load_allowed_actions(ALLOWED_ACTIONS_FILE)
    if not allowed_actions:
        _fail([f"allowed actions list is empty or missing: {ALLOWED_ACTIONS_FILE}"])
    errors = []
    for path in _sorted(WORKFLOW_DIR.glob("*.yml")):
        check_deadline()
        doc = _load_yaml(path)
        jobs = doc.get("jobs", {})
        has_self_hosted = False
        if isinstance(jobs, dict):
            for name, job in jobs.items():
                check_deadline()
                if _is_self_hosted(job.get("runs-on")):
                    has_self_hosted = True
        events = _event_names(doc.get("on"))
        allow_id_token = _is_tag_only_push(doc.get("on")) and (not has_self_hosted)
        allow_pr_write = (("pull_request" in events) or ("pull_request_target" in events)) and (
            not has_self_hosted
        )
        allow_contents_write = path.name in CONTENT_WRITE_WORKFLOWS
        require_id_token = False
        require_actions_read = False
        if path.name in {"release-testpypi.yml", "release-pypi.yml"} and not has_self_hosted:
            allow_id_token = True
            require_id_token = True
            require_actions_read = True
        if allow_contents_write:
            if path.name == "release-tag.yml":
                _check_release_tag_workflow(doc, path, errors)
            if path.name == "auto-test-tag.yml":
                _check_auto_test_tag_workflow(doc, path, errors)
            if path.name == "mirror-next.yml":
                _check_mirror_next_workflow(doc, path, errors)
            if path.name == "promote-release.yml":
                _check_promote_release_workflow(doc, path, errors)
        if path.name == "release-testpypi.yml":
            _check_release_testpypi_workflow(doc, path, errors)
            _check_id_token_scoping(doc, path, errors)
        if path.name == "release-pypi.yml":
            _check_release_pypi_workflow(doc, path, errors)
            _check_id_token_scoping(doc, path, errors)
        _check_workflow_dispatch_guards(doc, path, errors)
        _check_ci_script_entrypoints(doc, path, errors)
        _check_policy_scanner_suite_entrypoints(doc, path, errors)
        _check_pr_stage_ci_poll_cadence(doc, path, errors)
        _check_permissions(
            doc,
            path,
            errors,
            allow_pr_write=allow_pr_write,
            allow_id_token=allow_id_token,
            allow_contents_write=allow_contents_write,
            require_id_token=require_id_token,
            require_actions_read=require_actions_read,
        )
        _check_self_hosted_constraints(doc, path, errors)
        if allow_pr_write:
            _check_pr_comment_guard(doc, path, errors)
        if isinstance(jobs, dict):
            for name, job in jobs.items():
                check_deadline()
                job_ctx = JobContext(job_name=str(name), path=path)
                _check_job_permissions(
                    job,
                    job_ctx,
                    errors,
                    allow_pr_write=allow_pr_write,
                    allow_id_token=allow_id_token,
                    allow_contents_write=allow_contents_write,
                    require_id_token=require_id_token,
                    require_actions_read=require_actions_read,
                )
                _check_actions(job, job_ctx, errors, allowed_actions)
    _check_dense_core_lock_in(errors)
    _write_workflow_governance_artifacts(errors=errors)
    _write_local_ci_repro_contract_artifact()
    if errors:
        _fail(errors)


def _api_json(url, token):
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "policy-check",
        "Authorization": f"Bearer {token}",
    }
    request = Request(url, headers=headers)
    with urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def _gh_auth_token() -> str | None:
    try:
        proc = subprocess.run(
            ["gh", "auth", "token"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    token = proc.stdout.strip()
    if not token:
        return None
    return token


def check_posture():
    policy_token = os.environ.get("POLICY_GITHUB_TOKEN")
    env_token = os.environ.get("GITHUB_TOKEN")
    gh_token = _gh_auth_token()
    token = policy_token or env_token or gh_token
    using_policy_token = policy_token is not None
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not token:
        _fail(
            [
                "missing GITHUB_TOKEN or POLICY_GITHUB_TOKEN for posture check "
                "(or ensure `gh auth token` is available)"
            ]
        )
    if policy_token is not None and len(policy_token) < 20:
        _fail(
            [
                "POLICY_GITHUB_TOKEN appears too short; reset the repo secret "
                "with scripts/reset_policy_secret.sh"
            ]
        )
    if not repo:
        _fail(["missing GITHUB_REPOSITORY for posture check"])

    base = f"https://api.github.com/repos/{repo}"
    errors = []
    try:
        perms = _api_json(f"{base}/actions/permissions", token)
    except (HTTPError, URLError) as exc:
        hint = ""
        if isinstance(exc, HTTPError):
            if exc.code == 401:
                hint = " (POLICY_GITHUB_TOKEN is invalid or missing required auth)"
            elif exc.code == 403 and not using_policy_token:
                hint = " (set POLICY_GITHUB_TOKEN with admin read access)"
        _fail([f"actions permissions query failed: {exc}{hint}"])
    allowed = perms.get("allowed_actions")
    if allowed != "selected":
        errors.append("allowed_actions must be 'selected'")

    try:
        workflow_perms = _api_json(f"{base}/actions/permissions/workflow", token)
    except (HTTPError, URLError) as exc:
        hint = ""
        if isinstance(exc, HTTPError):
            if exc.code == 401:
                hint = " (POLICY_GITHUB_TOKEN is invalid or missing required auth)"
            elif exc.code == 403 and not using_policy_token:
                hint = " (set POLICY_GITHUB_TOKEN with admin read access)"
        _fail([f"workflow permissions query failed: {exc}{hint}"])
    default_perms = workflow_perms.get("default_workflow_permissions")
    if default_perms != "read":
        errors.append("default_workflow_permissions must be 'read'")
    if workflow_perms.get("can_approve_pull_request_reviews"):
        errors.append("can_approve_pull_request_reviews must be false")

    try:
        selected = _api_json(
            f"{base}/actions/permissions/selected-actions", token
        )
    except (HTTPError, URLError) as exc:
        hint = ""
        if isinstance(exc, HTTPError):
            if exc.code == 401:
                hint = " (POLICY_GITHUB_TOKEN is invalid or missing required auth)"
            elif exc.code == 403 and not using_policy_token:
                hint = " (set POLICY_GITHUB_TOKEN with admin read access)"
        _fail([f"selected actions query failed: {exc}{hint}"])

    if selected.get("github_owned_allowed") or selected.get("verified_allowed"):
        errors.append("only allow-listed actions are permitted (disable broad allowances)")

    patterns = selected.get("patterns_allowed") or []
    normalized = []
    invalid_patterns = []
    for pattern in patterns:
        check_deadline()
        if "@" in pattern:
            name, ref = pattern.split("@", 1)
            if ref != "*":
                invalid_patterns.append(pattern)
                continue
            normalized.append(name)
        else:
            normalized.append(pattern)
    if invalid_patterns:
        errors.append(
            f"invalid action patterns (must end with @* or be bare): {_sorted(invalid_patterns)}"
        )
    allowed_actions = _load_allowed_actions(ALLOWED_ACTIONS_FILE)
    if set(normalized) != set(allowed_actions):
        errors.append(
            f"allowed action patterns must match {_sorted(allowed_actions)}"
        )

    if errors:
        _fail(errors)


def _load_tier2_residue_baseline(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    residues = payload.get("residues", []) if isinstance(payload, dict) else []
    if not isinstance(residues, list):
        return set()
    return {str(item) for item in residues if isinstance(item, str)}


def check_tier2_residue_contract() -> None:
    from gabion.analysis import dataflow_audit

    source_path = REPO_ROOT / "src" / "gabion" / "analysis" / "dataflow_audit.py"
    try:
        source = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        _fail([f"tier2 residue policy check failed to read source: {exc}"])
    with deadline_clock_scope(MonotonicClock()):
        with deadline_scope(Deadline.from_timeout_ms(10_000)):
            instances = dataflow_audit._pattern_schema_matches(
                groups_by_path={},
                source=source,
                source_path=source_path,
            )
            residues = dataflow_audit._tier2_unreified_residue_entries(
                dataflow_audit._pattern_schema_residue_entries(instances)
            )
    current = {
        f"{entry.reason}:{entry.payload.get('kind', '')}:{entry.schema_id}"
        for entry in residues
    }
    baseline = _load_tier2_residue_baseline(TIER2_RESIDUE_BASELINE)
    new_keys = current - baseline
    if new_keys:
        _fail([
            "tier2 residue policy check failed (new unreified Tier-2 residues)",
            *[f"new residue: {item}" for item in _sorted(new_keys)],
        ])

def check_ambiguity_contract() -> None:
    cmd = [
        sys.executable,
        "-m",
        "gabion",
        "ambiguity-contract-gate",
        "--root",
        str(REPO_ROOT),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        details = []
        if stdout:
            details.append(stdout)
        if stderr:
            details.append(stderr)
        _fail(["ambiguity contract policy check failed", *details])



_SEMANTIC_CORE_PAYLOAD_BRANCH_MODULES = (
    REPO_ROOT / "src" / "gabion" / "analysis" / "dataflow_audit.py",
    REPO_ROOT / "src" / "gabion" / "analysis" / "timeout_context.py",
)

_ALLOWED_PAYLOAD_BRANCH_FUNCTION_PREFIXES = (
    "_decode_",
)

_KNOWN_ADAPTER_SURFACES = {
    "bundle-inference": "bundle_inference",
    "decision-surfaces": "decision_surfaces",
    "type-flow": "type_flow",
    "exception-obligations": "exception_obligations",
    "rewrite-plan-support": "rewrite_plan_support",
}

_LAST_FAIL_ERRORS: list[str] = []


def _write_projection_semantic_fragment_queue_artifacts(
    *,
    output_path: Path,
    phase5_workstreams_projection: Mapping[str, object] | None = None,
) -> None:
    from scripts.policy import projection_semantic_fragment_queue

    projection_semantic_fragment_queue.run(
        source_artifact_path=output_path,
        out_path=output_path.parent / "projection_semantic_fragment_queue.json",
        markdown_out=output_path.parent / "projection_semantic_fragment_queue.md",
        phase5_workstreams_projection=phase5_workstreams_projection,
    )


def _write_ingress_merge_parity_artifact(*, output_path: Path) -> Path:
    from gabion.tooling.policy_substrate.structured_artifact_ingress import (
        StructuredArtifactIdentitySpace,
        build_ingress_merge_parity_artifact,
        write_ingress_merge_parity_artifact,
    )

    artifact_rel_path = "artifacts/out/ingress_merge_parity.json"
    artifact = build_ingress_merge_parity_artifact(
        root=REPO_ROOT,
        rel_path=artifact_rel_path,
        identities=StructuredArtifactIdentitySpace(),
    )
    return write_ingress_merge_parity_artifact(
        root=REPO_ROOT,
        rel_path=artifact_rel_path,
        artifact=artifact,
    )


def _write_git_state_artifact(*, output_path: Path) -> Path:
    from gabion.tooling.runtime.git_state_artifact import write_git_state_artifact

    return write_git_state_artifact(
        path=output_path.parent / "git_state.json",
        root=REPO_ROOT,
    )


def _write_cross_origin_witness_contract_artifact(*, output_path: Path) -> Path:
    from gabion.tooling.runtime.cross_origin_witness_artifact import (
        write_cross_origin_witness_contract_artifact,
    )

    return write_cross_origin_witness_contract_artifact(
        path=output_path.parent / "cross_origin_witness_contract.json",
        root=REPO_ROOT,
    )


def _write_kernel_vm_alignment_artifact(*, output_path: Path) -> Path:
    from gabion.tooling.runtime.kernel_vm_alignment_artifact import (
        write_kernel_vm_alignment_artifact,
    )

    return write_kernel_vm_alignment_artifact(
        path=output_path.parent / "kernel_vm_alignment.json",
        root=REPO_ROOT,
    )


def _write_invariant_graph_artifact(
    *,
    output_path: Path,
) -> Mapping[str, object]:
    from gabion.tooling.policy_substrate.invariant_graph import (
        build_invariant_graph,
        build_invariant_ledger_projections,
        build_invariant_workstreams,
        load_invariant_workstreams,
        write_invariant_graph,
        write_invariant_ledger_projections,
        write_invariant_workstreams,
    )

    graph = build_invariant_graph(REPO_ROOT)
    workstreams = build_invariant_workstreams(graph, root=REPO_ROOT)
    workstreams_path = output_path.parent / "invariant_workstreams.json"
    write_invariant_graph(output_path.parent / "invariant_graph.json", graph)
    write_invariant_workstreams(
        workstreams_path,
        workstreams,
    )
    write_invariant_ledger_projections(
        output_path.parent / "invariant_ledger_projections.json",
        build_invariant_ledger_projections(workstreams, root=REPO_ROOT),
    )
    return load_invariant_workstreams(workstreams_path)



def _raw_payload_branching_violations(path: Path) -> list[str]:
    check_deadline()
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"{path}: failed to read source ({exc})"]
    try:
        module = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [f"{path}: failed to parse source ({exc})"]

    function_ranges: list[tuple[int, int, str]] = []
    for node in ast.walk(module):
        check_deadline()
        if isinstance(node, ast.FunctionDef):
            start = int(getattr(node, "lineno", 0) or 0)
            end = int(getattr(node, "end_lineno", start) or start)
            function_ranges.append((start, end, node.name))

    def _function_name_for_line(line_no: int) -> str | None:
        check_deadline()
        best: tuple[int, str] | None = None
        for start, end, name in function_ranges:
            check_deadline()
            if start <= line_no <= end:
                width = end - start
                if best is None or width < best[0]:
                    best = (width, name)
        return best[1] if best is not None else None

    def _is_allowed(line_no: int) -> bool:
        name = _function_name_for_line(line_no)
        if name is None:
            return False
        return name.startswith(_ALLOWED_PAYLOAD_BRANCH_FUNCTION_PREFIXES)

    def _pattern_mentions_raw_payload(pattern: ast.pattern) -> bool:
        check_deadline()
        if isinstance(pattern, ast.MatchClass):
            cls = pattern.cls
            if isinstance(cls, ast.Name) and cls.id in {"Mapping", "list"}:
                return True
            if isinstance(cls, ast.Attribute) and cls.attr in {"Mapping", "list"}:
                return True
        for child in ast.iter_child_nodes(pattern):
            check_deadline()
            if isinstance(child, ast.pattern) and _pattern_mentions_raw_payload(child):
                return True
        return False

    def _isinstance_mentions_raw_payload(call: ast.Call) -> bool:
        check_deadline()
        if not (isinstance(call.func, ast.Name) and call.func.id == "isinstance"):
            return False
        if len(call.args) < 2:
            return False
        type_arg = call.args[1]
        targets: list[ast.expr] = []
        if isinstance(type_arg, ast.Tuple):
            targets.extend(type_arg.elts)
        else:
            targets.append(type_arg)
        for target in targets:
            check_deadline()
            if isinstance(target, ast.Name) and target.id in {"Mapping", "list"}:
                return True
            if isinstance(target, ast.Attribute) and target.attr in {"Mapping", "list"}:
                return True
        return False

    try:
        display_path = str(path.relative_to(REPO_ROOT))
    except ValueError:
        display_path = str(path)

    violations: list[str] = []
    for node in ast.walk(module):
        check_deadline()
        if isinstance(node, ast.match_case):
            lineno = int(getattr(node.pattern, "lineno", 0) or 0)
            if lineno > 0 and _pattern_mentions_raw_payload(node.pattern) and not _is_allowed(lineno):
                violations.append(f"{display_path}:{lineno}: raw Mapping/list match outside boundary decode")
        if isinstance(node, ast.Call):
            lineno = int(getattr(node, "lineno", 0) or 0)
            if lineno > 0 and _isinstance_mentions_raw_payload(node) and not _is_allowed(lineno):
                violations.append(f"{display_path}:{lineno}: isinstance Mapping/list branch outside boundary decode")
    return violations


def check_semantic_core_payload_branching() -> None:
    errors: list[str] = []
    for path in _SEMANTIC_CORE_PAYLOAD_BRANCH_MODULES:
        check_deadline()
        errors.extend(_raw_payload_branching_violations(path))
    if errors:
        _fail([
            "semantic-core payload branching policy check failed",
            *errors,
        ])

def check_adapter_surface_policy() -> None:
    errors: list[str] = []
    dataflow = dataflow_defaults(root=REPO_ROOT)
    required = set(dataflow_required_surfaces(dataflow))
    unknown = required - set(_KNOWN_ADAPTER_SURFACES)
    if unknown:
        errors.append(f"unknown required adapter surfaces: {_sorted(unknown)}")
    adapter_payload = dataflow_adapter_payload(dataflow)
    capabilities = adapter_payload.get("capabilities", {}) if isinstance(adapter_payload, dict) else {}
    if not isinstance(capabilities, dict):
        capabilities = {}
    for surface in required & set(_KNOWN_ADAPTER_SURFACES):
        check_deadline()
        capability_key = _KNOWN_ADAPTER_SURFACES[surface]
        if capabilities.get(capability_key) is False:
            errors.append(
                f"required adapter surface {surface!r} is disabled via capability {capability_key!r}"
            )
    if errors:
        _fail(errors)

def main(argv: list[str] | None = None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="POLICY_SEED guardrails")
    parser.add_argument("--workflows", action="store_true", help="lint workflows")
    parser.add_argument("--posture", action="store_true", help="check GitHub posture")
    parser.add_argument("--ambiguity-contract", action="store_true", help="run ambiguity contract policy checks")
    parser.add_argument("--normative-map", action="store_true", help="validate docs/normative_enforcement_map.yaml")
    parser.add_argument("--tier2-residue-contract", action="store_true", help="run tier-2 residue policy checks")
    parser.add_argument("--adapter-surfaces", action="store_true", help="validate configured adapter surface requirements")
    parser.add_argument("--semantic-core-payload-branching", action="store_true", help="forbid raw Mapping/list payload branching outside boundary decode functions")
    parser.add_argument("--aspf-taint-crosswalk", action="store_true", help="require ASPF/taint crosswalk acknowledgement when relevant files change")
    parser.add_argument("--policy-dsl", action="store_true", help="compile/typecheck policy DSL sources")
    parser.add_argument("--output", type=Path, help="optional policy-result artifact path")
    parser.add_argument(
        "--perf-artifact",
        type=Path,
        help="optional cProfile-based structural perf artifact path",
    )
    args = parser.parse_args(argv)

    if not args.workflows and not args.posture and not args.ambiguity_contract and not args.normative_map and not args.tier2_residue_contract and not args.adapter_surfaces and not args.semantic_core_payload_branching and not args.aspf_taint_crosswalk and not args.policy_dsl:
        args.workflows = True

    requested_checks = tuple(
        name
        for name, enabled in (
            ("workflows", args.workflows),
            ("posture", args.posture),
            ("ambiguity_contract", args.ambiguity_contract),
            ("normative_map", args.normative_map),
            ("tier2_residue_contract", args.tier2_residue_contract),
            ("adapter_surfaces", args.adapter_surfaces),
            ("semantic_core_payload_branching", args.semantic_core_payload_branching),
            ("aspf_taint_crosswalk", args.aspf_taint_crosswalk),
            ("policy_dsl", args.policy_dsl),
        )
        if enabled
    )

    returncode = 0
    lattice_convergence_result: ProjectionFiberLatticeConvergenceResult | None = None
    perf_profile = cProfile.Profile() if args.perf_artifact is not None else None
    result: dict[str, object] | None = None
    _LAST_FAIL_ERRORS.clear()
    try:
        if perf_profile is not None:
            perf_profile.enable()
        try:
            with _policy_deadline_scope():
                if args.workflows:
                    check_workflows()
                if args.posture:
                    check_posture()
                if args.ambiguity_contract:
                    check_ambiguity_contract()
                if args.normative_map:
                    check_normative_enforcement_map()
                if args.tier2_residue_contract:
                    check_tier2_residue_contract()
                if args.adapter_surfaces:
                    check_adapter_surface_policy()
                if args.semantic_core_payload_branching:
                    check_semantic_core_payload_branching()
                if args.aspf_taint_crosswalk or args.workflows:
                    check_aspf_taint_crosswalk_ack()
                if args.policy_dsl or args.workflows:
                    check_policy_dsl()
                    lattice_convergence_result = collect_aspf_lattice_convergence_result()
                    if lattice_convergence_result.blocking:
                        _fail(
                            [
                                lattice_convergence_result.decision_message,
                                *lattice_convergence_result.error_messages,
                            ]
                        )
        except SystemExit as exc:
            code = exc.code
            returncode = int(code) if isinstance(code, int) else 1

        if args.output is not None:
            violations: list[dict[str, str]] = []
            if returncode != 0:
                violations = [
                    {
                        "message": message,
                        "render": message,
                    }
                    for message in _LAST_FAIL_ERRORS
                ]
                if not violations:
                    violations.append(
                        {
                            "message": "policy checks failed",
                            "render": f"policy_check returncode={returncode}",
                        }
                    )
            result = policy_result_schema.make_policy_result(
                rule_id="policy_check",
                status="pass" if returncode == 0 else "fail",
                violations=violations,
                baseline_mode="none",
                source_tool="scripts/policy/policy_check.py",
                input_scope={"checks": requested_checks},
            )
            if lattice_convergence_result is not None:
                result["projection_fiber_semantics"] = (
                    lattice_convergence_result.as_policy_output()
                )
            policy_result_schema.write_policy_result(
                path=args.output,
                result=result,
            )
            if lattice_convergence_result is not None:
                _write_git_state_artifact(
                    output_path=args.output,
                )
                _write_cross_origin_witness_contract_artifact(
                    output_path=args.output,
                )
                _write_kernel_vm_alignment_artifact(
                    output_path=args.output,
                )
                _write_ingress_merge_parity_artifact(
                    output_path=args.output,
                )
                phase5_workstreams_projection = _write_invariant_graph_artifact(
                    output_path=args.output
                )
                _write_projection_semantic_fragment_queue_artifacts(
                    output_path=args.output,
                    phase5_workstreams_projection=phase5_workstreams_projection,
                )
    finally:
        if perf_profile is not None:
            perf_profile.disable()
    graph_artifact = (
        args.output.parent / "invariant_graph.json"
        if args.output is not None
        else None
    )
    if args.perf_artifact is not None and perf_profile is not None:
        try:
            write_cprofile_perf_artifact(
                path=args.perf_artifact,
                profile=perf_profile,
                root=REPO_ROOT,
                command=("scripts/policy/policy_check.py", *raw_argv),
                requested_checks=requested_checks,
                returncode=returncode,
                graph_artifact=graph_artifact,
            )
        except Exception as exc:
            message = f"perf artifact write failed: {exc}"
            _LAST_FAIL_ERRORS.append(message)
            returncode = 1
            if result is not None:
                result["status"] = "fail"
                result["violations"] = [
                    *result["violations"],
                    {
                        "message": message,
                        "render": message,
                    },
                ]
                policy_result_schema.write_policy_result(
                    path=args.output,
                    result=result,
                )

    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
