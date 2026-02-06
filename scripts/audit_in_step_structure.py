#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


REQUIRED_FRONTMATTER_FIELDS = {
    "doc_id",
    "doc_role",
    "doc_scope",
    "doc_authority",
    "doc_owner",
    "doc_requires",
    "doc_reviewed_as_of",
    "doc_review_notes",
    "doc_change_protocol",
    "doc_erasure",
}

REQUIRED_SECTIONS = [
    "purpose",
    "non-goals",
    "status",
    "definitions",
    "face / kit / solver",
    "artifact contract",
    "admissibility",
    "checks as lemmas",
    "success criteria",
    "failure modes and diagnostics",
    "appendix a: canonical template (authoritative)",
    "appendix b: structural self-audit",
]


@dataclass(frozen=True)
class Doc:
    frontmatter: dict[str, object]
    body_lines: list[str]


def _parse_frontmatter(text: str) -> tuple[dict[str, object], list[str]]:
    if not text.startswith("---\n"):
        return {}, text.splitlines()
    lines = text.splitlines()
    if lines[0].strip() != "---":
        return {}, lines
    fm_lines: list[str] = []
    idx = 1
    while idx < len(lines):
        line = lines[idx]
        if line.strip() == "---":
            idx += 1
            break
        fm_lines.append(line)
        idx += 1
    body_lines = lines[idx:]
    return _parse_yaml_like(fm_lines), body_lines


def _parse_yaml_like(lines: list[str]) -> dict[str, object]:
    data: dict[str, object] = {}
    current_list_key: str | None = None
    current_map_key: str | None = None
    for raw in lines:
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if current_map_key and line.startswith("  ") and ":" in line:
            key, value = line.strip().split(":", 1)
            data.setdefault(current_map_key, {})
            mapping = data[current_map_key]
            if isinstance(mapping, dict):
                mapping[key.strip()] = value.strip().strip("\"")
            continue
        if current_list_key and line.strip().startswith("-"):
            data.setdefault(current_list_key, [])
            seq = data[current_list_key]
            if isinstance(seq, list):
                seq.append(line.strip().lstrip("-").strip())
            continue
        current_list_key = None
        current_map_key = None
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value == "":
                data[key] = []
                current_list_key = key
            elif value == "{}":
                data[key] = {}
                current_map_key = key
            else:
                if value.startswith("\"") and value.endswith("\""):
                    value = value[1:-1]
                data[key] = value
        else:
            continue
    return data


def _normalize_header(header: str) -> str:
    return " ".join(header.strip().lower().split())


def _collect_headers(body_lines: list[str]) -> list[str]:
    headers: list[str] = []
    for line in body_lines:
        if not line.startswith("#"):
            continue
        stripped = line.lstrip("#").strip()
        if stripped:
            headers.append(_normalize_header(stripped))
    return headers


def _section_contains_obligations(body_lines: list[str]) -> bool:
    return any("F1" in line or "F2" in line or "F3" in line for line in body_lines)


def _audit_doc(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    frontmatter, body_lines = _parse_frontmatter(text)
    violations: list[str] = []

    if frontmatter.get("doc_role") != "in_step":
        return violations

    for field in REQUIRED_FRONTMATTER_FIELDS:
        if field not in frontmatter:
            violations.append(f"{path}: missing frontmatter field '{field}'")

    headers = set(_collect_headers(body_lines))
    for section in REQUIRED_SECTIONS:
        if section not in headers:
            violations.append(f"{path}: missing section '{section}'")

    if not _section_contains_obligations(body_lines):
        violations.append(f"{path}: Face obligations missing (no F1/F2/F3 markers found)")

    return violations


def _iter_paths(paths: list[str]) -> list[Path]:
    resolved: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            resolved.extend(sorted(path.rglob("*.md")))
        else:
            resolved.append(path)
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit in_step document structure.")
    parser.add_argument("paths", nargs="+", help="Markdown files or directories to audit.")
    args = parser.parse_args(argv)

    violations: list[str] = []
    for path in _iter_paths(args.paths):
        if not path.exists():
            violations.append(f"{path}: missing file")
            continue
        violations.extend(_audit_doc(path))

    if violations:
        for violation in violations:
            print(violation)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
