#!/usr/bin/env python3
"""Promote docs to anchor-level section metadata and (optionally) anchorize refs.

This tool adds:
  - doc_sections (single-anchor default)
  - doc_section_requires (per-anchor deps)
  - doc_section_reviews (per-anchor review ledger)
  - anchor tags in body (if missing)

Optionally, it rewrites doc_requires/review pins to anchor refs when the
dependency doc exposes exactly one anchor.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Iterable

from audit_tools import _parse_frontmatter  # type: ignore
from deadline_runtime import DeadlineBudget, deadline_scope_from_ticks
from gabion.analysis.timeout_context import check_deadline
from gabion.order_contract import ordered_or_sorted


AnchorMap = dict[str, tuple[str, int]]

_DEFAULT_PROMOTE_TIMEOUT_TICKS = 120_000
_DEFAULT_PROMOTE_TIMEOUT_TICK_NS = 1_000_000
_DEFAULT_PROMOTE_TIMEOUT_BUDGET = DeadlineBudget(
    ticks=_DEFAULT_PROMOTE_TIMEOUT_TICKS,
    tick_ns=_DEFAULT_PROMOTE_TIMEOUT_TICK_NS,
)


def _promote_deadline_scope():
    return deadline_scope_from_ticks(
        budget=_DEFAULT_PROMOTE_TIMEOUT_BUDGET,
    )


def _iter_docs(paths: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        check_deadline()
        if not raw:
            continue
        path = Path(raw)
        if path.is_dir():
            for doc in ordered_or_sorted(
                path.rglob("*.md"),
                source="scripts.docflow_promote_sections.iter_docs",
            ):
                check_deadline()
                if doc not in seen:
                    out.append(doc)
                    seen.add(doc)
        elif path.is_file():
            if path not in seen:
                out.append(path)
                seen.add(path)
    return out


def _format_scalar(value: Any) -> str:
    if value is None:
        return "\"\""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "":
        return "\"\""
    return text


def _dump_yaml_like(data: dict[str, Any], indent: int = 0) -> list[str]:
    lines: list[str] = []
    pad = " " * indent
    for key, value in data.items():
        check_deadline()
        if isinstance(value, dict):
            if value:
                lines.append(f"{pad}{key}:")
                lines.extend(_dump_yaml_like(value, indent + 2))
            else:
                lines.append(f"{pad}{key}: {{}}")
            continue
        if isinstance(value, list):
            if not value:
                lines.append(f"{pad}{key}: []")
                continue
            lines.append(f"{pad}{key}:")
            for item in value:
                check_deadline()
                if isinstance(item, dict):
                    lines.append(f"{pad}  -")
                    lines.extend(_dump_yaml_like(item, indent + 4))
                else:
                    lines.append(f"{pad}  - {_format_scalar(item)}")
            continue
        lines.append(f"{pad}{key}: {_format_scalar(value)}")
    return lines


def _normalize_frontmatter(
    path: Path,
    fm: dict[str, Any],
) -> dict[str, Any]:
    if "doc_requires" not in fm or not isinstance(fm.get("doc_requires"), list):
        fm["doc_requires"] = [] if fm.get("doc_requires") is None else list(fm.get("doc_requires") or [])
    if "doc_reviewed_as_of" not in fm or not isinstance(fm.get("doc_reviewed_as_of"), dict):
        fm["doc_reviewed_as_of"] = {} if fm.get("doc_reviewed_as_of") is None else dict(fm.get("doc_reviewed_as_of") or {})
    if "doc_review_notes" not in fm or not isinstance(fm.get("doc_review_notes"), dict):
        fm["doc_review_notes"] = {} if fm.get("doc_review_notes") is None else dict(fm.get("doc_review_notes") or {})
    if "doc_change_protocol" not in fm:
        fm["doc_change_protocol"] = "POLICY_SEED.md#change_protocol"
    if "doc_id" not in fm:
        stem = path.stem.lower().replace("-", "_")
        if path.parts and path.parts[0] in {"in", "out"}:
            fm["doc_id"] = f"{path.parts[0]}_{stem}"
        else:
            fm["doc_id"] = stem
    if "doc_role" not in fm:
        lower = path.stem.lower()
        if lower == "readme":
            fm["doc_role"] = "readme"
        elif lower == "contributing":
            fm["doc_role"] = "contributing"
        elif lower == "agents":
            fm["doc_role"] = "agent"
        elif path.parts and path.parts[0] == "out":
            fm["doc_role"] = "report"
        else:
            fm["doc_role"] = "inbox"
    if "doc_scope" not in fm:
        fm["doc_scope"] = ["repo", "documentation"]
    if "doc_authority" not in fm:
        fm["doc_authority"] = "informative"
    return fm


def _derive_anchor(path: Path, fm: dict[str, Any]) -> str:
    doc_id = fm.get("doc_id")
    if isinstance(doc_id, str) and doc_id:
        if re.fullmatch(r"in_\d+", doc_id):
            return f"in_{doc_id}"
        if re.fullmatch(r"out_\d+", doc_id):
            return f"out_{doc_id}"
        return doc_id
    stem = path.stem.lower().replace("-", "_")
    if path.parts and path.parts[0] in {"in", "out"}:
        return f"{path.parts[0]}_{stem}"
    return stem


def _ensure_anchor(body: str, anchor: str) -> str:
    marker = f'<a id="{anchor}"></a>'
    if marker in body:
        return body
    lines = body.splitlines()
    for idx, line in enumerate(lines):
        check_deadline()
        if line.strip().startswith("#"):
            lines.insert(idx, "")
            lines.insert(idx, marker)
            return "\n".join(lines)
    return f"{marker}\n\n{body}"


def _anchor_map(all_docs: dict[str, dict[str, Any]]) -> AnchorMap:
    anchors: AnchorMap = {}
    for rel, fm in all_docs.items():
        check_deadline()
        sections = fm.get("doc_sections")
        if isinstance(sections, dict) and len(sections) == 1:
            key, value = next(iter(sections.items()))
            if isinstance(key, str) and isinstance(value, int):
                anchors[rel] = (key, value)
    return anchors


def _anchorize_ref(ref: str, anchors: AnchorMap) -> tuple[str, int | None]:
    if "#" in ref:
        return ref, None
    if ref in anchors:
        anchor, version = anchors[ref]
        return f"{ref}#{anchor}", version
    return ref, None


def _rewrite_body_refs(body: str, replacements: dict[str, str]) -> str:
    for old, new in replacements.items():
        check_deadline()
        pattern = re.compile(rf"{re.escape(old)}(?!#)")
        body = pattern.sub(new, body)
    return body


def _update_section_reviews(
    reviews: dict[str, Any],
    anchors: AnchorMap,
) -> dict[str, Any]:
    updated: dict[str, Any] = {}
    for anchor, deps in reviews.items():
        check_deadline()
        if not isinstance(deps, dict):
            updated[anchor] = deps
            continue
        next_deps: dict[str, Any] = {}
        for dep, payload in deps.items():
            check_deadline()
            if not isinstance(dep, str):
                next_deps[dep] = payload
                continue
            new_dep, version = _anchorize_ref(dep, anchors)
            if isinstance(payload, dict) and version is not None:
                payload = dict(payload)
                payload["dep_version"] = version
            next_deps[new_dep] = payload
        updated[anchor] = next_deps
    return updated


def _update_doc(path: Path, anchors: AnchorMap, *, add_sections: bool, anchorize: bool) -> bool:
    text = path.read_text(encoding="utf-8")
    fm, body = _parse_frontmatter(text)
    if not fm:
        return False
    fm = _normalize_frontmatter(path, dict(fm))

    changed = False
    desired_anchor = _derive_anchor(path, fm)
    if add_sections and "doc_sections" not in fm:
        anchor = desired_anchor
        fm["doc_sections"] = {anchor: 1}
        fm["doc_section_requires"] = {anchor: list(fm.get("doc_requires") or [])}
        reviews: dict[str, Any] = {}
        for dep in fm.get("doc_requires") or []:
            check_deadline()
            if not isinstance(dep, str):
                continue
            dep_version = None
            if isinstance(fm.get("doc_reviewed_as_of"), dict):
                dep_version = fm["doc_reviewed_as_of"].get(dep)
            note = None
            if isinstance(fm.get("doc_review_notes"), dict):
                note = fm["doc_review_notes"].get(dep)
            if isinstance(dep_version, int) and isinstance(note, str):
                reviews[dep] = {
                    "dep_version": dep_version,
                    "self_version_at_review": 1,
                    "outcome": "no_change",
                    "note": note,
                }
        fm["doc_section_reviews"] = {anchor: reviews}
        body = _ensure_anchor(body, anchor)
        changed = True
    elif isinstance(fm.get("doc_sections"), dict):
        sections = fm["doc_sections"]
        if len(sections) == 1 and path.parts and path.parts[0] in {"in", "out"}:
            current_anchor = next(iter(sections))
            doc_id = fm.get("doc_id")
            if isinstance(doc_id, str) and current_anchor == doc_id and desired_anchor != current_anchor:
                value = sections.pop(current_anchor)
                sections[desired_anchor] = value
                if isinstance(fm.get("doc_section_requires"), dict):
                    reqs = fm["doc_section_requires"]
                    if current_anchor in reqs:
                        reqs[desired_anchor] = reqs.pop(current_anchor)
                if isinstance(fm.get("doc_section_reviews"), dict):
                    reviews = fm["doc_section_reviews"]
                    if current_anchor in reviews:
                        reviews[desired_anchor] = reviews.pop(current_anchor)
                body = body.replace(f'<a id="{current_anchor}"></a>', f'<a id="{desired_anchor}"></a>')
                body = _ensure_anchor(body, desired_anchor)
                changed = True

    if anchorize:
        replacements: dict[str, str] = {}
        new_requires: list[str] = []
        for dep in fm.get("doc_requires") or []:
            check_deadline()
            if not isinstance(dep, str):
                continue
            new_dep, version = _anchorize_ref(dep, anchors)
            if new_dep != dep:
                replacements[dep] = new_dep
            new_requires.append(new_dep)
            if version is not None:
                reviewed = fm.get("doc_reviewed_as_of")
                if isinstance(reviewed, dict):
                    reviewed[new_dep] = version
                    if dep in reviewed:
                        reviewed.pop(dep, None)
                notes = fm.get("doc_review_notes")
                if isinstance(notes, dict):
                    if dep in notes and new_dep not in notes:
                        notes[new_dep] = notes.pop(dep)
        if new_requires != fm.get("doc_requires"):
            fm["doc_requires"] = new_requires
            changed = True
        if replacements:
            body = _rewrite_body_refs(body, replacements)
            changed = True

        if isinstance(fm.get("doc_section_requires"), dict):
            updated = {}
            for anchor, deps in fm["doc_section_requires"].items():
                check_deadline()
                if not isinstance(deps, list):
                    updated[anchor] = deps
                    continue
                out_list = []
                for dep in deps:
                    check_deadline()
                    if not isinstance(dep, str):
                        continue
                    new_dep, version = _anchorize_ref(dep, anchors)
                    if new_dep != dep:
                        changed = True
                    out_list.append(new_dep)
                    if version is not None:
                        reviews = fm.get("doc_section_reviews")
                        if isinstance(reviews, dict):
                            anchor_reviews = reviews.get(anchor)
                            if isinstance(anchor_reviews, dict):
                                entry = anchor_reviews.get(dep)
                                if isinstance(entry, dict):
                                    entry = dict(entry)
                                    entry["dep_version"] = version
                                    anchor_reviews[new_dep] = entry
                                    anchor_reviews.pop(dep, None)
                updated[anchor] = out_list
            fm["doc_section_requires"] = updated

        if isinstance(fm.get("doc_section_reviews"), dict):
            fm["doc_section_reviews"] = _update_section_reviews(fm["doc_section_reviews"], anchors)

    if not changed:
        return False

    frontmatter = "\n".join(_dump_yaml_like(fm))
    path.write_text(f"---\n{frontmatter}\n---\n{body}", encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote docflow sections and anchorize refs.")
    parser.add_argument(
        "paths",
        nargs="+",
        help="Docs or directories to update.",
    )
    parser.add_argument(
        "--anchorize",
        action="store_true",
        help="Rewrite doc_requires/review pins to anchor refs when possible.",
    )
    parser.add_argument(
        "--add-sections",
        action="store_true",
        help="Add doc_sections/doc_section_requires/doc_section_reviews if missing.",
    )
    args = parser.parse_args()

    with _promote_deadline_scope():
        doc_paths = _iter_docs(args.paths)
        all_docs: dict[str, dict[str, Any]] = {}
        for path in doc_paths:
            check_deadline()
            fm, _ = _parse_frontmatter(path.read_text(encoding="utf-8"))
            if not fm:
                continue
            all_docs[path.as_posix()] = dict(fm)

        anchors = _anchor_map(all_docs)

        changed = 0
        for path in doc_paths:
            check_deadline()
            if _update_doc(path, anchors, add_sections=args.add_sections, anchorize=args.anchorize):
                changed += 1
                print(f"updated {path}")
        print(f"Updated {changed} document(s).")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
