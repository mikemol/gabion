#!/usr/bin/env python3
"""Detect controller drift between normative governance docs and enforcing checks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO_ROOT / "POLICY_SEED.md"
DEFAULT_OUT = REPO_ROOT / "artifacts" / "out" / "controller_drift.json"
NORMATIVE_DOCS = (
    "POLICY_SEED.md",
    "CONTRIBUTING.md",
    "README.md",
    "AGENTS.md",
    "glossary.md",
    "docs/normative_clause_index.md",
    "docs/governance_control_loops.md",
    "docs/governance_loop_matrix.md",
)
NORMATIVE_DOC_RE = re.compile(r"controller-normative-doc:\s*(?P<doc>[^`\n]+)")

ANCHOR_RE = re.compile(
    r"controller-anchor:\s*(?P<id>CD-\d+)\s*\|\s*"
    r"doc:\s*(?P<doc>[^|]+)\|\s*"
    r"sensor:\s*(?P<sensor>[^|]+)\|\s*"
    r"check:\s*(?P<check>[^|]+)\|\s*"
    r"severity:\s*(?P<severity>[^`]+)"
)
COMMAND_RE = re.compile(r"controller-command:\s*(?P<command>[^`]+)")
SCRIPT_IN_WORKFLOW_RE = re.compile(r"scripts/[A-Za-z0-9_\-./]+\.py")

SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}

REQUIRED_ENFORCEMENT_CLAUSES = {
    "controller_drift": ("NCI-CONTROLLER-DRIFT-LIFECYCLE", "clause-controller-drift-lifecycle"),
    "command_policies": ("NCI-COMMAND-MATURITY-PARITY", "clause-command-maturity-parity"),
}


@dataclass(frozen=True)
class ControllerAnchor:
    anchor_id: str
    doc: str
    sensor: str
    check: str
    severity: str


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normative_docs(policy_text: str) -> tuple[str, ...]:
    declared = [match.group("doc").strip() for match in NORMATIVE_DOC_RE.finditer(policy_text)]
    if declared:
        return tuple(dict.fromkeys(declared))
    return NORMATIVE_DOCS


def _parse_policy(policy_text: str) -> tuple[list[ControllerAnchor], list[str]]:
    anchors: list[ControllerAnchor] = []
    commands: list[str] = []
    for match in ANCHOR_RE.finditer(policy_text):
        anchors.append(
            ControllerAnchor(
                anchor_id=match.group("id").strip(),
                doc=match.group("doc").strip(),
                sensor=match.group("sensor").strip(),
                check=match.group("check").strip(),
                severity=match.group("severity").strip(),
            )
        )
    for match in COMMAND_RE.finditer(policy_text):
        commands.append(match.group("command").strip())
    return anchors, commands


def _collect_normative_anchor_signatures(normative_docs: tuple[str, ...]) -> tuple[dict[str, set[str]], list[str]]:
    signatures: dict[str, set[str]] = {}
    missing_docs: list[str] = []
    for rel in normative_docs:
        path = REPO_ROOT / rel
        if not path.exists():
            missing_docs.append(rel)
            continue
        text = _load_text(path)
        for match in ANCHOR_RE.finditer(text):
            key = match.group("id").strip()
            signature = "|".join(
                (
                    match.group("doc").strip(),
                    match.group("sensor").strip(),
                    match.group("check").strip(),
                    match.group("severity").strip(),
                )
            )
            signatures.setdefault(key, set()).add(signature)
    return signatures, missing_docs


def _governance_checks_from_workflows() -> set[str]:
    discovered: set[str] = set()
    workflow_dir = REPO_ROOT / ".github" / "workflows"
    if not workflow_dir.exists():
        return discovered
    for workflow in workflow_dir.glob("*.yml"):
        text = _load_text(workflow)
        for script_path in SCRIPT_IN_WORKFLOW_RE.findall(text):
            if "governance" in script_path or "policy" in script_path or "template" in script_path:
                discovered.add(script_path)
    return discovered


def _classify_command_staleness(command: str) -> str | None:
    parts = command.split()
    if not parts:
        return "empty command reference"
    if "scripts/" in command:
        for token in parts:
            if token.startswith("scripts/") and token.endswith(".py"):
                if not (REPO_ROOT / token).exists():
                    return f"missing script path: {token}"
    return None


def _severity_at_least(value: str, threshold: str) -> bool:
    return SEVERITY_RANK.get(value.lower(), 0) >= SEVERITY_RANK.get(threshold.lower(), 99)


def _enforcement_clause_findings() -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    rules_path = REPO_ROOT / "docs" / "governance_rules.yaml"
    clause_path = REPO_ROOT / "docs" / "normative_clause_index.md"
    if not rules_path.exists():
        findings.append(
            {
                "sensor": "missing_normative_docs_in_repo",
                "severity": "high",
                "anchor": None,
                "detail": f"Expected governance rules file is missing: `{_relative(rules_path)}`.",
            }
        )
        return findings
    if not clause_path.exists():
        findings.append(
            {
                "sensor": "missing_normative_docs_in_repo",
                "severity": "high",
                "anchor": None,
                "detail": f"Expected normative clause index file is missing: `{_relative(clause_path)}`.",
            }
        )
        return findings
    rules_text = _load_text(rules_path)
    clause_text = _load_text(clause_path)
    for key, (clause_id, anchor_id) in REQUIRED_ENFORCEMENT_CLAUSES.items():
        annotation = f"{key}:  # {clause_id}"
        if annotation not in rules_text:
            findings.append(
                {
                    "sensor": "unindexed_enforcement_surfaces",
                    "severity": "high",
                    "anchor": None,
                    "detail": f"governance_rules key `{key}` is missing clause annotation `{clause_id}`.",
                }
            )
        if f'<a id="{anchor_id}"></a>' not in clause_text:
            findings.append(
                {
                    "sensor": "unindexed_enforcement_surfaces",
                    "severity": "high",
                    "anchor": None,
                    "detail": f"normative clause index missing anchor `{anchor_id}` for `{key}` enforcement.",
                }
            )
    return findings


def run(policy_path: Path, out_path: Path, fail_on_severity: str | None) -> int:
    policy_text = _load_text(policy_path)
    anchors, commands = _parse_policy(policy_text)
    normative_docs = _normative_docs(policy_text)

    findings: list[dict[str, object]] = []

    # Sensor 1: policy clauses with no enforcing check.
    for anchor in anchors:
        check_path = (REPO_ROOT / anchor.check).resolve()
        if anchor.check in {"", "none", "tbd"} or not check_path.exists():
            findings.append(
                {
                    "sensor": "policy_clauses_without_enforcing_check",
                    "severity": "high",
                    "anchor": anchor.anchor_id,
                    "detail": f"Anchor {anchor.anchor_id} references missing check `{anchor.check}`.",
                }
            )

    # Sensor 2: checks with no normative anchor.
    anchored_checks = {anchor.check for anchor in anchors}
    discovered_checks = _governance_checks_from_workflows()
    for check in sorted(discovered_checks - anchored_checks):
        findings.append(
            {
                "sensor": "checks_without_normative_anchor",
                "severity": "high",
                "anchor": None,
                "detail": f"Enforcement script `{check}` is referenced by workflow but missing a controller anchor.",
            }
        )

    # Sensor 3: contradictory anchors across normative docs.
    signatures, missing_docs = _collect_normative_anchor_signatures(normative_docs)
    for rel in missing_docs:
        findings.append(
            {
                "sensor": "missing_normative_docs_in_repo",
                "severity": "high",
                "anchor": None,
                "detail": f"Expected normative doc is missing from repository: `{rel}`.",
            }
        )
    for anchor_id, variants in sorted(signatures.items()):
        if len(variants) > 1:
            findings.append(
                {
                    "sensor": "contradictory_anchors_across_normative_docs",
                    "severity": "high",
                    "anchor": anchor_id,
                    "detail": f"Anchor {anchor_id} has contradictory signatures across normative docs.",
                    "variants": sorted(variants),
                }
            )

    # Sensor 4: stale command references.
    for command in commands:
        stale_reason = _classify_command_staleness(command)
        if stale_reason:
            findings.append(
                {
                    "sensor": "stale_command_references",
                    "severity": "medium",
                    "anchor": None,
                    "detail": f"Command `{command}` is stale: {stale_reason}.",
                }
            )

    # Sensor 5: enforcement surfaces without clause anchors.
    findings.extend(_enforcement_clause_findings())

    severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    for item in findings:
        sev = str(item.get("severity", "")).lower()
        if sev in severity_counts:
            severity_counts[sev] += 1
    highest = "none"
    for candidate in ("critical", "high", "medium", "low"):
        if severity_counts[candidate] > 0:
            highest = candidate
            break
    summary = {
        "total_findings": len(findings),
        "high_severity_findings": sum(1 for item in findings if str(item.get("severity")) == "high"),
        "sensors": sorted({str(item.get("sensor")) for item in findings}),
        "severity_counts": severity_counts,
        "highest_severity": highest,
    }

    payload = {
        "policy": _relative(policy_path),
        "anchors_scanned": len(anchors),
        "commands_scanned": len(commands),
        "normative_docs": list(normative_docs),
        "findings": findings,
        "summary": summary,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if fail_on_severity is not None:
        for finding in findings:
            sev = str(finding.get("severity", ""))
            if _severity_at_least(sev, fail_on_severity):
                print(
                    f"controller-drift: failing due to {sev} finding: {finding.get('detail')}",
                    file=sys.stderr,
                )
                return 2
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=POLICY_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--fail-on-severity",
        choices=["low", "medium", "high", "critical"],
        default=None,
        help="Optional severity threshold that turns findings into a failing exit code.",
    )
    args = parser.parse_args()
    return run(policy_path=args.policy, out_path=args.out, fail_on_severity=args.fail_on_severity)


if __name__ == "__main__":
    raise SystemExit(main())
