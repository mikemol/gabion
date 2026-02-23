#!/usr/bin/env python3
"""Require governance PR template fields when governance/tooling files change."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIELD_LABELS = ("controller impact", "loop updated?")
GOVERNANCE_PATH_PREFIXES = (
    "POLICY_SEED.md",
    "CONTRIBUTING.md",
    "README.md",
    "AGENTS.md",
    "glossary.md",
    ".github/workflows/",
    ".github/actions/",
    "scripts/",
)


def _changed_files(base: str, head: str) -> list[str]:
    cmd = ["git", "diff", "--name-only", f"{base}..{head}"]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, check=True, text=True, capture_output=True)
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _is_governance_change(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix) for prefix in GOVERNANCE_PATH_PREFIXES)


def _body_has_required_fields(body: str) -> tuple[bool, list[str]]:
    missing: list[str] = []
    lower = body.lower()
    for label in FIELD_LABELS:
        pattern = re.compile(rf"(^|\n)\s*[-*]?\s*{re.escape(label)}\s*:", re.IGNORECASE)
        if not pattern.search(lower):
            missing.append(label)
    return (len(missing) == 0), missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=os.getenv("GITHUB_BASE_SHA", ""))
    parser.add_argument("--head", default=os.getenv("GITHUB_HEAD_SHA", ""))
    parser.add_argument("--body-file", type=Path, default=None)
    args = parser.parse_args()

    if not args.base or not args.head:
        print("pr-template-check: missing base/head SHA; skipping.")
        return 0

    changed = _changed_files(args.base, args.head)
    if not any(_is_governance_change(path) for path in changed):
        print("pr-template-check: no governance/tooling changes detected; skipping.")
        return 0

    if args.body_file is not None and args.body_file.exists():
        body = args.body_file.read_text(encoding="utf-8")
    else:
        body = os.getenv("PR_BODY", "")

    ok, missing = _body_has_required_fields(body)
    if ok:
        print("pr-template-check: required governance fields present.")
        return 0

    print(
        "pr-template-check: governance/tooling changes require PR fields: "
        + ", ".join(missing),
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
