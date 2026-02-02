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

CORE_GOVERNANCE_DOCS = [
    "POLICY_SEED.md",
    "glossary.md",
    "README.md",
    "CONTRIBUTING.md",
    "AGENTS.md",
]

GOVERNANCE_DOCS = CORE_GOVERNANCE_DOCS + [
    "docs/publishing_practices.md",
    "docs/influence_index.md",
]

REQUIRED_FIELDS = [
    "doc_id",
    "doc_role",
    "doc_scope",
    "doc_authority",
    "doc_requires",
    "doc_reviewed_as_of",
    "doc_change_protocol",
]
LIST_FIELDS = {
    "doc_scope",
    "doc_requires",
    "doc_commutes_with",
    "doc_invariants",
    "doc_erasure",
}
MAP_FIELDS = {
    "doc_reviewed_as_of",
}


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
    current_map_key: str | None = None
    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            continue
        if current_map_key is not None and line.startswith("  ") and ":" in line:
            key, value = line.strip().split(":", 1)
            key = key.strip()
            value = value.strip()
            if value.startswith("\"") and value.endswith("\""):
                value = value[1:-1]
            if value.isdigit():
                parsed: object = int(value)
            else:
                parsed = value
            mapping = data.get(current_map_key)
            if not isinstance(mapping, dict):
                mapping = {}
            mapping[key] = parsed
            data[current_map_key] = mapping
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
                if key in MAP_FIELDS:
                    data[key] = {}
                    current_map_key = key
                    current_list_key = None
                else:
                    data[key] = []
                    current_list_key = key
                    current_map_key = None
                continue
            if value.startswith("\"") and value.endswith("\""):
                value = value[1:-1]
            if value.isdigit():
                data[key] = int(value)
            else:
                data[key] = value
            current_list_key = None
            current_map_key = None
    return data


def _lower_name(path: Path) -> str:
    return path.stem.lower()


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
    core_set = set(CORE_GOVERNANCE_DOCS)

    revisions: Dict[str, int] = {}
    for rel, payload in docs.items():
        fm = payload["frontmatter"]
        if isinstance(fm.get("doc_revision"), int):
            revisions[rel] = fm["doc_revision"]

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
        if "doc_reviewed_as_of" in fm and not isinstance(fm["doc_reviewed_as_of"], dict):
            violations.append(f"{rel}: frontmatter field 'doc_reviewed_as_of' must be a map")
        # Normative docs must require the governance bundle.
        if fm.get("doc_authority") == "normative":
            requires = set(fm.get("doc_requires", []))
            expected = core_set - {rel}
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
                req_name = _lower_name(Path(req))
                if req_name and req_name in body_lower:
                    warnings.append(
                        f"{rel}: implicit reference to {req} (Tier-2); prefer explicit path"
                    )
                else:
                    violations.append(f"{rel}: missing explicit reference to {req}")
        # Convergence check (re-reviewed as of)
        reviewed = fm.get("doc_reviewed_as_of")
        if isinstance(requires, list) and requires:
            if not isinstance(reviewed, dict):
                violations.append(f"{rel}: doc_reviewed_as_of missing or invalid")
            else:
                for req in requires:
                    expected = revisions.get(req)
                    if expected is None:
                        violations.append(
                            f"{rel}: doc_reviewed_as_of cannot resolve {req}"
                        )
                        continue
                    seen = reviewed.get(req)
                    if not isinstance(seen, int):
                        violations.append(
                            f"{rel}: doc_reviewed_as_of[{req}] must be an integer"
                        )
                    elif seen != expected:
                        violations.append(
                            f"{rel}: doc_reviewed_as_of[{req}]={seen} does not match {expected}"
                        )

    warnings.extend(_tooling_warnings(root, docs))
    warnings.extend(_influence_warnings(root))

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


def _influence_warnings(root: Path) -> List[str]:
    warnings: List[str] = []
    inbox = root / "in"
    index_path = root / "docs" / "influence_index.md"
    if not inbox.exists():
        return warnings
    if not index_path.exists():
        warnings.append("docs/influence_index.md: missing influence index for in/")
        return warnings
    index_text = index_path.read_text(encoding="utf-8")
    for path in sorted(inbox.glob("in-*.md")):
        rel = path.as_posix()
        if rel not in index_text:
            warnings.append(f"docs/influence_index.md: missing {rel}")
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
