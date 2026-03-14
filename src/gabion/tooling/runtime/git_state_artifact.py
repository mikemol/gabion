from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from typing import TypedDict

from gabion.order_contract import ordered_or_sorted


class GitStateLineSpan(TypedDict):
    start_line: int
    line_count: int


class GitStateEntry(TypedDict):
    state_class: str
    change_code: str
    path: str
    previous_path: str
    current_line_spans: list[GitStateLineSpan]


class GitStateSummary(TypedDict):
    committed_count: int
    staged_count: int
    unstaged_count: int
    untracked_count: int


class GitStatePayload(TypedDict):
    format_version: int
    schema_version: int
    artifact_kind: str
    generated_at_utc: str
    root: str
    head_sha: str
    branch: str
    upstream: str
    is_detached: bool
    summary: GitStateSummary
    entries: list[GitStateEntry]


_UNIFIED_DIFF_HUNK_RE = re.compile(
    r"^@@ -\d+(?:,\d+)? \+(?P<start>\d+)(?:,(?P<count>\d+))? @@"
)


def _sorted[T](values: list[T], *, key=None) -> list[T]:
    return ordered_or_sorted(
        values,
        source="tooling.runtime.git_state_artifact",
        key=key,
    )


def _git_output(*, root: Path, args: tuple[str, ...]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return ""
    return completed.stdout.strip()


def _parse_name_status_output(
    *,
    payload: str,
    state_class: str,
) -> list[GitStateEntry]:
    entries: list[GitStateEntry] = []
    for raw_line in payload.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        parts = raw_line.split("\t")
        if not parts:
            continue
        change_code = parts[0].strip()
        if not change_code:
            continue
        if len(parts) >= 3 and change_code[:1] in {"R", "C"}:
            previous_path = parts[1].strip()
            path = parts[2].strip()
        elif len(parts) >= 2:
            previous_path = ""
            path = parts[1].strip()
        else:
            previous_path = ""
            path = ""
        if not path:
            continue
        entries.append(
            GitStateEntry(
                state_class=state_class,
                change_code=change_code,
                path=path,
                previous_path=previous_path,
                current_line_spans=[],
            )
        )
    return entries


def _parse_untracked_output(payload: str) -> list[GitStateEntry]:
    entries: list[GitStateEntry] = []
    for raw_line in payload.splitlines():
        path = raw_line.strip()
        if not path:
            continue
        entries.append(
            GitStateEntry(
                state_class="untracked",
                change_code="??",
                path=path,
                previous_path="",
                current_line_spans=[],
            )
        )
    return entries


def _parse_current_line_spans_by_path(payload: str) -> dict[str, list[GitStateLineSpan]]:
    spans_by_path: dict[str, list[GitStateLineSpan]] = {}
    current_path = ""
    for raw_line in payload.splitlines():
        if raw_line.startswith("+++ "):
            if raw_line == "+++ /dev/null":
                current_path = ""
            elif raw_line.startswith("+++ b/"):
                current_path = raw_line[6:].strip()
            else:
                current_path = ""
            continue
        if not current_path:
            continue
        match = _UNIFIED_DIFF_HUNK_RE.match(raw_line)
        if match is None:
            continue
        start_line = int(match.group("start"))
        line_count = int(match.group("count") or "1")
        if line_count <= 0:
            continue
        spans_by_path.setdefault(current_path, []).append(
            GitStateLineSpan(
                start_line=start_line,
                line_count=line_count,
            )
        )
    return {
        path: _sorted(
            spans,
            key=lambda item: (item["start_line"], item["line_count"]),
        )
        for path, spans in spans_by_path.items()
    }


def _full_file_line_spans(path: Path) -> list[GitStateLineSpan]:
    try:
        line_count = len(path.read_text(encoding="utf-8").splitlines())
    except OSError:
        return []
    if line_count <= 0:
        return []
    return [GitStateLineSpan(start_line=1, line_count=line_count)]


def _attach_current_line_spans(
    *,
    entries: list[GitStateEntry],
    current_line_spans_by_path: dict[str, list[GitStateLineSpan]],
    root: Path,
) -> list[GitStateEntry]:
    resolved: list[GitStateEntry] = []
    for entry in entries:
        current_line_spans = current_line_spans_by_path.get(entry["path"], [])
        if entry["state_class"] == "untracked":
            current_line_spans = _full_file_line_spans(root / entry["path"])
        resolved.append(
            GitStateEntry(
                state_class=entry["state_class"],
                change_code=entry["change_code"],
                path=entry["path"],
                previous_path=entry["previous_path"],
                current_line_spans=current_line_spans,
            )
        )
    return resolved


def build_git_state_artifact_payload(*, root: Path) -> GitStatePayload:
    root = root.resolve()
    head_sha = _git_output(root=root, args=("rev-parse", "HEAD"))
    branch = _git_output(root=root, args=("rev-parse", "--abbrev-ref", "HEAD"))
    upstream = _git_output(
        root=root,
        args=("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"),
    )
    committed = _parse_name_status_output(
        payload=_git_output(
            root=root,
            args=("show", "--name-status", "--find-renames", "--format=", "HEAD"),
        ),
        state_class="committed",
    )
    staged = _parse_name_status_output(
        payload=_git_output(
            root=root,
            args=("diff", "--cached", "--name-status", "--find-renames"),
        ),
        state_class="staged",
    )
    unstaged = _parse_name_status_output(
        payload=_git_output(
            root=root,
            args=("diff", "--name-status", "--find-renames"),
        ),
        state_class="unstaged",
    )
    untracked = _parse_untracked_output(
        _git_output(
            root=root,
            args=("ls-files", "--others", "--exclude-standard"),
        )
    )
    current_line_spans_by_path = _parse_current_line_spans_by_path(
        _git_output(
            root=root,
            args=("diff", "HEAD", "--unified=0", "--find-renames", "--no-color"),
        )
    )
    committed = _attach_current_line_spans(
        entries=committed,
        current_line_spans_by_path={},
        root=root,
    )
    staged = _attach_current_line_spans(
        entries=staged,
        current_line_spans_by_path=current_line_spans_by_path,
        root=root,
    )
    unstaged = _attach_current_line_spans(
        entries=unstaged,
        current_line_spans_by_path=current_line_spans_by_path,
        root=root,
    )
    untracked = _attach_current_line_spans(
        entries=untracked,
        current_line_spans_by_path=current_line_spans_by_path,
        root=root,
    )
    entries = _sorted(
        [*committed, *staged, *unstaged, *untracked],
        key=lambda item: (
            item["state_class"],
            item["path"],
            item["previous_path"],
            item["change_code"],
        ),
    )
    return GitStatePayload(
        format_version=1,
        schema_version=1,
        artifact_kind="git_state",
        generated_at_utc=datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        root=str(root),
        head_sha=head_sha,
        branch=branch,
        upstream=upstream,
        is_detached=branch == "HEAD",
        summary=GitStateSummary(
            committed_count=len(committed),
            staged_count=len(staged),
            unstaged_count=len(unstaged),
            untracked_count=len(untracked),
        ),
        entries=entries,
    )


def write_git_state_artifact(*, path: Path, root: Path) -> Path:
    payload = build_git_state_artifact_payload(root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return path
