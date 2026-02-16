#!/usr/bin/env python3
"""Pin GitHub Actions to full commit SHAs in workflow files.

Uses the GitHub CLI to resolve tags to commit SHAs.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from deadline_runtime import DeadlineBudget, deadline_scope_from_lsp_env
from gabion.analysis.timeout_context import check_deadline


USES_RE = re.compile(r"^(\s*(?:-\s+)?uses:\s+)([^@\s]+)@([^\s]+)\s*$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_DEFAULT_TIMEOUT_TICKS = 120_000
_DEFAULT_TIMEOUT_TICK_NS = 1_000_000
_DEFAULT_TIMEOUT_BUDGET = DeadlineBudget(
    ticks=_DEFAULT_TIMEOUT_TICKS,
    tick_ns=_DEFAULT_TIMEOUT_TICK_NS,
)


def _deadline_scope():
    return deadline_scope_from_lsp_env(
        default_budget=_DEFAULT_TIMEOUT_BUDGET,
    )


@dataclass(frozen=True)
class ActionRef:
    owner: str
    repo: str
    ref: str

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"


def _run_gh(args: list[str]) -> dict:
    result = subprocess.run(
        ["gh", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _resolve_ref(action: ActionRef) -> str:
    if SHA_RE.fullmatch(action.ref):
        return action.ref
    ref = action.ref
    if ref.startswith("v"):
        ref = ref
    payload = _run_gh(
        [
            "api",
            f"repos/{action.owner}/{action.repo}/git/ref/tags/{ref}",
        ]
    )
    obj = payload.get("object", {})
    sha = obj.get("sha")
    if not sha:
        raise SystemExit(f"Unable to resolve tag {action.full_name}@{action.ref}")
    if obj.get("type") == "tag":
        tag_payload = _run_gh(
            [
                "api",
                f"repos/{action.owner}/{action.repo}/git/tags/{sha}",
            ]
        )
        sha = tag_payload.get("object", {}).get("sha", sha)
    if not sha or not SHA_RE.fullmatch(sha):
        raise SystemExit(f"Resolved SHA is invalid for {action.full_name}@{action.ref}")
    return sha


def _parse_action(ref: str) -> ActionRef | None:
    if "@" not in ref:
        return None
    name, version = ref.split("@", 1)
    if "/" not in name:
        return None
    if "<" in version or ">" in version:
        return None
    owner, repo = name.split("/", 1)
    return ActionRef(owner=owner, repo=repo, ref=version)


def _pin_file(path: Path) -> tuple[int, list[str]]:
    lines = path.read_text().splitlines()
    updated = 0
    pinned: list[str] = []
    for i, line in enumerate(lines):
        check_deadline()
        match = USES_RE.match(line)
        if not match:
            continue
        prefix, name, ref = match.groups()
        action = _parse_action(f"{name}@{ref}")
        if action is None:
            continue
        if SHA_RE.fullmatch(action.ref):
            continue
        sha = _resolve_ref(action)
        lines[i] = f"{prefix}{action.full_name}@{sha}"
        updated += 1
        pinned.append(f"{action.full_name}@{action.ref} -> {sha}")
    if updated:
        path.write_text("\n".join(lines) + "\n")
    return updated, pinned


def main() -> int:
    with _deadline_scope():
        parser = argparse.ArgumentParser(description="Pin GitHub Actions to SHAs.")
        parser.add_argument(
            "paths",
            nargs="+",
            help="Workflow files to update in-place.",
        )
        args = parser.parse_args()

        total = 0
        for raw in args.paths:
            check_deadline()
            path = Path(raw)
            if not path.exists():
                raise SystemExit(f"File not found: {path}")
            updated, pinned = _pin_file(path)
            total += updated
            for line in pinned:
                check_deadline()
                print(f"{path}: {line}")
        print(f"Updated {total} action reference(s).")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
