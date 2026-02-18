#!/usr/bin/env python3
"""Audit SPPF status consistency across in/, checklist, and influence index."""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


IN_FILE_RE = re.compile(r"in-(\d+)\.md$")
INFLUENCE_ROW_RE = re.compile(r"^-\s+in/in-(\d+)\.md\s+—\s+\*\*(\w+)\*\*", re.MULTILINE)
CHECKLIST_TAG_RE = re.compile(
    r"^(?P<line>.*?sppf\{[^\n]*doc=(?P<doc>\w+);\s*impl=(?P<impl>\w+);\s*doc_ref=(?P<refs>[^}]+)\}.*?)$",
    re.MULTILINE,
)
CHECKLIST_STATUS_NODE_RE = re.compile(r"<a\s+id=\"in-(\d+)")
DOC_REF_IN_RE = re.compile(r"in-(\d+)@")
STATUS_HEADING_RE = re.compile(r"^###\s+Status\s*$", re.MULTILINE)
OVERRIDE_RE = re.compile(r"sppf-status-override\s*:\s*(in-\d+|all)", re.IGNORECASE)


@dataclass(frozen=True)
class StatusRecord:
    category: str
    source: str
    detail: str


def _normalize_in_status(text: str) -> str | None:
    lowered = text.lower()
    if "implemented in part" in lowered or "partially implemented" in lowered:
        return "implemented-in-part"
    if "partial" in lowered or "in progress" in lowered:
        return "implemented-in-part"
    if any(token in lowered for token in ("planned", "plan", "queued", "todo")):
        return "planned"
    if any(token in lowered for token in ("done", "completed", "adopted", "implemented and active")):
        return "done"
    return None


def _normalize_influence_status(token: str) -> str | None:
    token = token.lower().strip()
    mapping = {
        "adopted": "done",
        "partial": "implemented-in-part",
        "queued": "planned",
        "untriaged": "planned",
        "rejected": "rejected",
    }
    return mapping.get(token)


def _normalize_checklist_pair(doc: str, impl: str) -> str:
    doc = doc.lower().strip()
    impl = impl.lower().strip()
    if doc == "done" and impl == "done":
        return "done"
    if impl == "planned":
        return "planned"
    return "implemented-in-part"


def _extract_in_status(path: Path) -> str | None:
    text = path.read_text(encoding="utf-8")
    match = STATUS_HEADING_RE.search(text)
    if not match:
        return None
    tail = text[match.end() :]
    for line in tail.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            return None
        return _normalize_in_status(line)
    return None


def _collect_overrides(*texts: str) -> set[str]:
    overrides: set[str] = set()
    for text in texts:
        for match in OVERRIDE_RE.finditer(text):
            overrides.add(match.group(1).lower())
    return overrides


def run_audit(root: Path) -> tuple[int, list[str]]:
    in_dir = root / "in"
    checklist_path = root / "docs" / "sppf_checklist.md"
    influence_path = root / "docs" / "influence_index.md"

    checklist_text = checklist_path.read_text(encoding="utf-8")
    influence_text = influence_path.read_text(encoding="utf-8")

    overrides = _collect_overrides(checklist_text, influence_text)

    records: dict[str, list[StatusRecord]] = {}

    for in_file in sorted(in_dir.glob("in-*.md")):
        if not IN_FILE_RE.search(in_file.name):
            continue
        in_id = in_file.stem
        in_text = in_file.read_text(encoding="utf-8")
        overrides.update(_collect_overrides(in_text))
        category = _extract_in_status(in_file)
        if category:
            records.setdefault(in_id, []).append(
                StatusRecord(category=category, source="in", detail=f"{in_file.as_posix()}")
            )

    for in_num, token in INFLUENCE_ROW_RE.findall(influence_text):
        in_id = f"in-{in_num}"
        category = _normalize_influence_status(token)
        if category:
            records.setdefault(in_id, []).append(
                StatusRecord(
                    category=category,
                    source="influence_index",
                    detail=f"{influence_path.as_posix()}:**{token}**",
                )
            )

    checklist_rows: dict[str, list[str]] = {}
    for match in CHECKLIST_TAG_RE.finditer(checklist_text):
        line = match.group("line")
        if "sppf-status-node" not in line and not CHECKLIST_STATUS_NODE_RE.search(line):
            continue
        category = _normalize_checklist_pair(match.group("doc"), match.group("impl"))
        refs = match.group("refs")
        for in_num in DOC_REF_IN_RE.findall(refs):
            checklist_rows.setdefault(f"in-{in_num}", []).append(category)

    for in_id, categories in checklist_rows.items():
        uniq = set(categories)
        if uniq == {"done"}:
            aggregate = "done"
        elif uniq == {"planned"}:
            aggregate = "planned"
        else:
            aggregate = "implemented-in-part"
        records.setdefault(in_id, []).append(
            StatusRecord(
                category=aggregate,
                source="sppf_checklist",
                detail=f"{checklist_path.as_posix()}:{sorted(uniq)}",
            )
        )

    errors: list[str] = []
    for in_id, row in sorted(records.items()):
        categories = {item.category for item in row}
        if len(categories) <= 1:
            continue
        if "all" in overrides or in_id in overrides:
            continue
        errors.append(f"{in_id}: status drift detected")
        for item in row:
            errors.append(f"  - {item.source}: {item.category} ({item.detail})")

    if errors:
        return 1, errors
    return 0, ["sppf-status-audit: no drift detected"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root")
    args = parser.parse_args(argv)
    code, lines = run_audit(args.root.resolve())
    stream = sys.stderr if code else sys.stdout
    for line in lines:
        print(line, file=stream)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
