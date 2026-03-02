"""Validate textual purge of retired monolith tokens in non-generated files."""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
from pathlib import Path
from typing import Iterable


_RUNTIME_TOKEN = "dataflow_" + "runtime"
_AUDIT_TOKEN = "dataflow_" + "audit"
_RUNTIME_TOKEN_CAP = "Dataflow" + "Runtime"
_AUDIT_TOKEN_CAP = "Dataflow" + "Audit"
TOKEN_SET = (_RUNTIME_TOKEN, _AUDIT_TOKEN, _RUNTIME_TOKEN_CAP, _AUDIT_TOKEN_CAP)
TOKEN_PATTERN = re.compile("|".join(re.escape(token) for token in TOKEN_SET))

IN_SCOPE_ROOTS = (
    "src",
    "tests",
    "scripts",
    "docs",
    "baselines",
    "in",
    "gabion.toml",
    "README.md",
    "CONTRIBUTING.md",
    "AGENTS.md",
    "POLICY_SEED.md",
    "glossary.md",
)

EXCLUDE_GLOBS = (
    "artifacts/out/**",
    "artifacts/test_runs/**",
    "out/**",
    "dist/**",
    "**/__pycache__/**",
    "**/*.pyc",
    "artifacts/audit_reports/*.json",
    "docs/audits/*retirement_ledger*.md",
)


def _iter_scope_files(root: Path) -> Iterable[Path]:
    for scope in IN_SCOPE_ROOTS:
        scope_path = root / scope
        if not scope_path.exists():
            continue
        if scope_path.is_file():
            yield scope_path
            continue
        for path in scope_path.rglob("*"):
            if path.is_file():
                yield path


def _is_excluded(root: Path, path: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    return any(fnmatch.fnmatch(rel, pattern) for pattern in EXCLUDE_GLOBS)


def _scan_file(path: Path) -> tuple[int, list[dict[str, object]]]:
    text = path.read_text(encoding="utf-8")
    line_hits: list[dict[str, object]] = []
    match_count = 0
    for lineno, line in enumerate(text.splitlines(), start=1):
        matches = TOKEN_PATTERN.findall(line)
        if not matches:
            continue
        match_count += len(matches)
        line_hits.append(
            {
                "line": lineno,
                "count": len(matches),
                "tokens": sorted(set(matches)),
            }
        )
    return match_count, line_hits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--json",
        dest="json_out",
        help="Optional path for deterministic JSON report output.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    files = sorted(set(_iter_scope_files(root)))
    findings: list[dict[str, object]] = []
    scanned_files = 0
    skipped_non_utf8 = 0

    for path in files:
        if _is_excluded(root, path):
            continue
        scanned_files += 1
        rel = path.relative_to(root).as_posix()
        path_matches = TOKEN_PATTERN.findall(rel)
        path_tokens = sorted(set(path_matches))
        line_hits: list[dict[str, object]] = []
        content_count = 0
        try:
            content_count, line_hits = _scan_file(path)
        except UnicodeDecodeError:
            skipped_non_utf8 += 1
            if not path_matches:
                continue
        total_count = len(path_matches) + content_count
        if total_count == 0:
            continue
        finding: dict[str, object] = {
            "path": rel,
            "count": total_count,
            "lines": line_hits,
        }
        if path_tokens:
            finding["path_token_count"] = len(path_matches)
            finding["path_tokens"] = path_tokens
        findings.append(finding)

    findings.sort(key=lambda item: item["path"])
    total_hits = sum(int(item["count"]) for item in findings)
    report = {
        "format_version": 1,
        "scanned_files": scanned_files,
        "skipped_non_utf8": skipped_non_utf8,
        "excluded_globs": list(EXCLUDE_GLOBS),
        "total_hits": total_hits,
        "files_with_hits": len(findings),
        "findings": findings,
    }

    if args.json_out:
        out = (root / args.json_out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8")
        print(f"wrote purge report JSON: {out}")

    if total_hits == 0:
        print("legacy monolith token purge check passed: no in-scope hits")
        return 0

    print(
        "legacy monolith token purge check failed:",
        f"{total_hits} hit(s) across {len(findings)} file(s)",
    )
    for item in findings:
        print(f"- {item['path']}: {item['count']} hit(s)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
