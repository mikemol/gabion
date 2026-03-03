# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time
from typing import Iterable, Mapping

from gabion.order_contract import sort_once

_HUNK_RE = re.compile(r"@@ -\d+(?:,\d+)? \+(?P<start>\d+)(?:,(?P<count>\d+))? @@")


@dataclass(frozen=True)
class ChangedLine:
    path: str
    line: int


@dataclass(frozen=True)
class DiffEvidenceIndexResult:
    changed_lines: list[ChangedLine]
    changed_paths: list[str]
    index_payload: Mapping[str, object] | None
    stale: bool
    refreshed: bool
    key: dict[str, str]


def parse_changed_lines(diff_text: str) -> list[ChangedLine]:
    changed: list[ChangedLine] = []
    current_path: str | None = None
    for raw_line in diff_text.splitlines():
        if raw_line.startswith("+++ "):
            path_token = raw_line[4:].strip()
            if path_token == "/dev/null":
                current_path = None
                continue
            if path_token.startswith("b/"):
                path_token = path_token[2:]
            current_path = path_token
            continue
        if current_path is None:
            continue
        match = _HUNK_RE.match(raw_line)
        if match is None:
            continue
        start = int(match.group("start"))
        count = int(match.group("count") or "1")
        if count <= 0:
            continue
        changed.extend(ChangedLine(path=current_path, line=start + offset) for offset in range(count))
    return changed


def git_diff_changed_lines(root: Path, *, base: str | None, head: str | None) -> list[ChangedLine]:
    if base and head:
        diff_range = f"{base}...{head}"
        cmd = ["git", "diff", "--unified=0", diff_range]
    elif base:
        cmd = ["git", "diff", "--unified=0", base]
    else:
        cmd = ["git", "diff", "--unified=0"]
    proc = subprocess.run(
        cmd,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        message = proc.stderr.strip() or proc.stdout.strip() or "git diff failed"
        raise RuntimeError(message)
    return parse_changed_lines(proc.stdout)


def load_json(path: Path) -> Mapping[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def refresh_test_evidence_index(root: Path, *, index_path: Path, tests_root: str) -> bool:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.extract_test_evidence",
            "--root",
            str(root),
            "--tests",
            tests_root,
            "--out",
            str(index_path),
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def diff_evidence_key(
    *,
    root: Path,
    base: str | None,
    head: str | None,
    index_path: Path,
) -> dict[str, str]:
    tree_hash = _git_tree_hash(root)
    return {
        "base_sha": str(base or ""),
        "head_sha": str(head or ""),
        "tree_hash": tree_hash,
        "index_path": str(index_path),
    }


def write_diff_evidence_artifacts(
    *,
    changed_lines_path: Path,
    meta_path: Path,
    changed_lines: list[ChangedLine],
    key: Mapping[str, str],
    stale: bool,
    refreshed: bool,
    index_path: Path,
) -> None:
    changed_lines_path.parent.mkdir(parents=True, exist_ok=True)
    changed_lines_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "changed_lines": [
                    {"path": item.path, "line": item.line}
                    for item in changed_lines
                ],
            },
            indent=2,
            sort_keys=False,
        )
        + "\n",
        encoding="utf-8",
    )
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "key": dict(key),
                "stale": bool(stale),
                "refreshed": bool(refreshed),
                "index_path": str(index_path),
                "changed_count": len(changed_lines),
                "changed_paths": sort_once(
                    {item.path for item in changed_lines},
                    source="write_diff_evidence_artifacts.changed_paths",
                ),
            },
            indent=2,
            sort_keys=False,
        )
        + "\n",
        encoding="utf-8",
    )


def build_diff_evidence_index(
    *,
    root: Path,
    base: str | None,
    head: str | None,
    index_path: Path,
    tests_root: str,
    stale_seconds: int,
    no_refresh: bool,
) -> DiffEvidenceIndexResult:
    changed_lines = git_diff_changed_lines(root, base=base, head=head)
    key = diff_evidence_key(root=root, base=base, head=head, index_path=index_path)
    index_payload = load_json(index_path)
    stale = False
    refreshed = False
    if index_payload is None:
        stale = True
    elif stale_seconds >= 0:
        age_seconds = time.time() - index_path.stat().st_mtime
        stale = age_seconds > stale_seconds

    if (index_payload is None or stale) and not no_refresh:
        refreshed = refresh_test_evidence_index(root, index_path=index_path, tests_root=tests_root)
        index_payload = load_json(index_path)
    changed_paths = sort_once({item.path for item in changed_lines}, source="build_diff_evidence_index.changed_paths")
    return DiffEvidenceIndexResult(
        changed_lines=changed_lines,
        changed_paths=changed_paths,
        index_payload=index_payload,
        stale=stale,
        refreshed=refreshed,
        key=key,
    )


def _git_tree_hash(root: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        fallback = hashlib.sha256(str(root).encode("utf-8")).hexdigest()
        return fallback
    return proc.stdout.strip()


__all__ = [
    "ChangedLine",
    "DiffEvidenceIndexResult",
    "build_diff_evidence_index",
    "diff_evidence_key",
    "git_diff_changed_lines",
    "load_json",
    "parse_changed_lines",
    "refresh_test_evidence_index",
    "write_diff_evidence_artifacts",
]
