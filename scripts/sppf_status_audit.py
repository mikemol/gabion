#!/usr/bin/env python3
"""Audit SPPF status consistency across in/, checklist, and influence index."""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from scripts.deadline_runtime import DeadlineBudget, deadline_scope_from_lsp_env
except ModuleNotFoundError:
    from deadline_runtime import DeadlineBudget, deadline_scope_from_lsp_env
from gabion.analysis.timeout_context import check_deadline, deadline_loop_iter
from gabion.order_contract import ordered_or_sorted


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

_DEFAULT_AUDIT_TIMEOUT_TICKS = 120_000
_DEFAULT_AUDIT_TIMEOUT_TICK_NS = 1_000_000
_DEFAULT_AUDIT_TIMEOUT_BUDGET = DeadlineBudget(
    ticks=_DEFAULT_AUDIT_TIMEOUT_TICKS,
    tick_ns=_DEFAULT_AUDIT_TIMEOUT_TICK_NS,
)


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
        check_deadline()
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            return None
        return _normalize_in_status(line)
    return None


def _collect_overrides(*texts: str) -> set[str]:
    overrides: set[str] = set()
    for text in deadline_loop_iter(texts):
        for match in deadline_loop_iter(OVERRIDE_RE.finditer(text)):
            overrides.add(match.group(1).lower())
    return overrides


def _deadline_scope():
    return deadline_scope_from_lsp_env(
        default_budget=_DEFAULT_AUDIT_TIMEOUT_BUDGET,
    )


def run_audit(root: Path) -> tuple[int, list[str]]:
    in_dir = root / "in"
    checklist_path = root / "docs" / "sppf_checklist.md"
    influence_path = root / "docs" / "influence_index.md"

    checklist_text = checklist_path.read_text(encoding="utf-8")
    influence_text = influence_path.read_text(encoding="utf-8")

    overrides = _collect_overrides(checklist_text, influence_text)

    records: dict[str, list[StatusRecord]] = {}

    for in_file in ordered_or_sorted(
        in_dir.glob("in-*.md"),
        source="scripts.sppf_status_audit.run_audit.in_files",
    ):
        check_deadline()
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

    for in_num, token in deadline_loop_iter(INFLUENCE_ROW_RE.findall(influence_text)):
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
    for match in deadline_loop_iter(CHECKLIST_TAG_RE.finditer(checklist_text)):
        line = match.group("line")
        if "sppf-status-node" not in line and not CHECKLIST_STATUS_NODE_RE.search(line):
            continue
        category = _normalize_checklist_pair(match.group("doc"), match.group("impl"))
        refs = match.group("refs")
        for in_num in deadline_loop_iter(DOC_REF_IN_RE.findall(refs)):
            checklist_rows.setdefault(f"in-{in_num}", []).append(category)

    for in_id, categories in deadline_loop_iter(checklist_rows.items()):
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
                detail=(
                    f"{checklist_path.as_posix()}:"
                    f"{ordered_or_sorted(uniq, source='scripts.sppf_status_audit.run_audit.uniq')}"
                ),
            )
        )

    errors: list[str] = []
    for in_id, row in ordered_or_sorted(
        records.items(),
        source="scripts.sppf_status_audit.run_audit.records",
    ):
        check_deadline()
        categories = {item.category for item in row}
        if len(categories) <= 1:
            continue
        if "all" in overrides or in_id in overrides:
            continue
        errors.append(f"{in_id}: status drift detected")
        for item in deadline_loop_iter(row):
            errors.append(f"  - {item.source}: {item.category} ({item.detail})")

    if errors:
        return 1, errors
    return 0, ["sppf-status-audit: no drift detected"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root")
    args = parser.parse_args(argv)
    with _deadline_scope():
        code, lines = run_audit(args.root.resolve())
        stream = sys.stderr if code else sys.stdout
        for line in deadline_loop_iter(lines):
            print(line, file=stream)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
