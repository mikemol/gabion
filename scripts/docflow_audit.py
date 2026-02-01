#!/usr/bin/env python3
"""Docflow audit for governance documents.

Treats frontmatter as the semantic signature and the body as presentation.
Validates required cross-references and commutation symmetry.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple

GOVERNANCE_DOCS = [
    "POLICY_SEED.md",
    "glossary.md",
    "README.md",
    "CONTRIBUTING.md",
    "AGENTS.md",
]

REQUIRED_FIELDS = [
    "doc_id",
    "doc_role",
    "doc_scope",
    "doc_authority",
    "doc_requires",
    "doc_change_protocol",
]


def _parse_frontmatter(text: str) -> Tuple[Dict[str, object], str]:
    if not text.startswith("---\n"):
        return {}, text
    lines = text.split("\n")
    if lines[0].strip() != "---":
        return {}, text
    fm_lines: List[str] = []
    idx = 1
    while idx < len(lines):
        line = lines[idx]
        if line.strip() == "---":
            idx += 1
            break
        fm_lines.append(line)
        idx += 1
    body = "\n".join(lines[idx:])
    return _parse_yaml_like(fm_lines), body


def _parse_yaml_like(lines: List[str]) -> Dict[str, object]:
    data: Dict[str, object] = {}
    current_list_key: str | None = None
    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            continue
        if line.lstrip().startswith("- "):
            if current_list_key is None:
                continue
            item = line.strip()[2:].strip()
            if isinstance(data.get(current_list_key), list):
                data[current_list_key].append(item)
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value == "":
                data[key] = []
                current_list_key = key
                continue
            if value.startswith("\"") and value.endswith("\""):
                value = value[1:-1]
            if value.isdigit():
                data[key] = int(value)
            else:
                data[key] = value
            current_list_key = None
    return data


def _lower_name(path: str) -> str:
    return Path(path).stem.lower()


def _audit(root: Path) -> Tuple[List[str], List[str]]:
    violations: List[str] = []
    warnings: List[str] = []

    docs: Dict[str, Dict[str, object]] = {}
    doc_ids: Dict[str, str] = {}

    for rel in GOVERNANCE_DOCS:
        path = root / rel
        if not path.exists():
            violations.append(f"missing governance doc: {rel}")
            continue
        text = path.read_text(encoding="utf-8")
        fm, body = _parse_frontmatter(text)
        docs[rel] = {"frontmatter": fm, "body": body}
        doc_id = fm.get("doc_id")
        if isinstance(doc_id, str):
            if doc_id in doc_ids:
                violations.append(
                    f"duplicate doc_id '{doc_id}' in {rel} and {doc_ids[doc_id]}"
                )
            else:
                doc_ids[doc_id] = rel

    governance_set = set(GOVERNANCE_DOCS)

    for rel, payload in docs.items():
        fm = payload["frontmatter"]
        body = payload["body"]
        # Required fields
        for field in REQUIRED_FIELDS:
            if field not in fm:
                violations.append(f"{rel}: missing frontmatter field '{field}'")
        # Required list for doc_scope/doc_requires
        for field in ("doc_scope", "doc_requires"):
            if field in fm and not isinstance(fm[field], list):
                violations.append(f"{rel}: frontmatter field '{field}' must be a list")
        # Normative docs must require the governance bundle.
        if fm.get("doc_authority") == "normative":
            requires = set(fm.get("doc_requires", []))
            expected = governance_set - {rel}
            missing = sorted(expected - requires)
            if missing:
                violations.append(
                    f"{rel}: missing required governance references: {', '.join(missing)}"
                )
        # Commutation symmetry
        commutes = fm.get("doc_commutes_with")
        if isinstance(commutes, list):
            for other in commutes:
                other_fm = docs.get(other, {}).get("frontmatter", {})
                other_commutes = other_fm.get("doc_commutes_with", [])
                if other not in docs:
                    violations.append(f"{rel}: doc_commutes_with target missing: {other}")
                    continue
                if not isinstance(other_commutes, list) or rel not in other_commutes:
                    violations.append(
                        f"{rel}: commutation with {other} not reciprocated"
                    )
        # Body references vs frontmatter requirements
        requires = fm.get("doc_requires", [])
        if isinstance(requires, list):
            body_lower = body.lower()
            for req in requires:
                if req in body:
                    continue
                req_name = _lower_name(req)
                if req_name and req_name in body_lower:
                    warnings.append(
                        f"{rel}: implicit reference to {req} (Tier-2); prefer explicit path"
                    )
                else:
                    violations.append(f"{rel}: missing explicit reference to {req}")

    warnings.extend(_tooling_warnings(root, docs))

    return violations, warnings


def _tooling_warnings(root: Path, docs: Dict[str, Dict[str, object]]) -> List[str]:
    warnings: List[str] = []
    makefile = root / "Makefile"
    if makefile.exists():
        for rel in ("README.md", "CONTRIBUTING.md"):
            body = docs.get(rel, {}).get("body", "")
            if "make " not in body and "Make targets" not in body:
                warnings.append(
                    f"{rel}: Makefile present but make targets are not documented"
                )
    checks_script = root / "scripts" / "checks.sh"
    if checks_script.exists():
        body = docs.get("CONTRIBUTING.md", {}).get("body", "")
        if "scripts/checks.sh" not in body:
            warnings.append(
                "CONTRIBUTING.md: scripts/checks.sh present but not documented"
            )
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root (default: current directory)",
    )
    parser.add_argument(
        "--fail-on-violations",
        action="store_true",
        help="Exit non-zero if violations are detected",
    )
    args = parser.parse_args()
    root = Path(args.root)

    violations, warnings = _audit(root)

    print("Docflow audit summary")
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"- {w}")
    if violations:
        print("Violations:")
        for v in violations:
            print(f"- {v}")
    if not warnings and not violations:
        print("No issues detected.")

    if violations and args.fail_on_violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
