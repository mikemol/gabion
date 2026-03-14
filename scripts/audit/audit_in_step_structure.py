#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - fallback path
    yaml = None

from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.order_contract import ordered_or_sorted
from gabion.tooling.runtime.declarative_script_host import (
    DeclarativeScriptSpec,
    ScriptInvocation,
    ScriptOptionArity,
    ScriptOptionKind,
    ScriptOptionSpec,
    ScriptRuntimeMode,
    ScriptRuntimeSpec,
    invoke_script,
    script_runtime_scope,
)
from scripts.deadline.deadline_runtime import DeadlineBudget


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

_DEFAULT_TIMEOUT_TICKS = 120_000
_DEFAULT_TIMEOUT_TICK_NS = 1_000_000
_DEFAULT_TIMEOUT_BUDGET = DeadlineBudget(
    ticks=_DEFAULT_TIMEOUT_TICKS,
    tick_ns=_DEFAULT_TIMEOUT_TICK_NS,
)
_SCRIPT_RUNTIME = ScriptRuntimeSpec(
    mode=ScriptRuntimeMode.LSP_ENV,
    deadline_budget=_DEFAULT_TIMEOUT_BUDGET,
)


def _deadline_scope():
    return script_runtime_scope(
        runtime=_SCRIPT_RUNTIME,
    )


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
        check_deadline()
        line = lines[idx]
        if line.strip() == "---":
            idx += 1
            break
        fm_lines.append(line)
        idx += 1
    body_lines = lines[idx:]
    if yaml is not None:
        try:
            parsed = yaml.safe_load("\n".join(fm_lines))
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            normalized: dict[str, object] = {}
            for key, value in parsed.items():
                check_deadline()
                normalized[str(key)] = value
            return normalized, body_lines
    return {}, body_lines


def _normalize_header(header: str) -> str:
    return " ".join(header.strip().lower().split())


def _collect_headers(body_lines: list[str]) -> list[str]:
    headers: list[str] = []
    for line in body_lines:
        check_deadline()
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
        check_deadline()
        if field not in frontmatter:
            violations.append(f"{path}: missing frontmatter field '{field}'")

    headers = set(_collect_headers(body_lines))
    for section in REQUIRED_SECTIONS:
        check_deadline()
        if section not in headers:
            violations.append(f"{path}: missing section '{section}'")

    if not _section_contains_obligations(body_lines):
        violations.append(f"{path}: Face obligations missing (no F1/F2/F3 markers found)")

    return violations


def _iter_paths(paths: tuple[Path, ...]) -> list[Path]:
    resolved: list[Path] = []
    for path in paths:
        check_deadline()
        if path.is_dir():
            resolved.extend(
                ordered_or_sorted(
                    path.rglob("*.md"),
                    source="scripts.audit_in_step_structure.iter_paths",
                )
            )
        else:
            resolved.append(path)
    return resolved


def _run_invocation(invocation: ScriptInvocation) -> int:
    with _deadline_scope():
        violations: list[str] = []
        for path in _iter_paths(invocation.paths("paths")):
            check_deadline()
            if not path.exists():
                violations.append(f"{path}: missing file")
                continue
            violations.extend(_audit_doc(path))

        if violations:
            for violation in violations:
                check_deadline()
                print(violation)
            return 2
        return 0


_SCRIPT_SPEC = DeclarativeScriptSpec(
    script_id="audit_in_step_structure",
    description="Audit in_step document structure.",
    options=(
        ScriptOptionSpec(
            dest="paths",
            flags=("paths",),
            kind=ScriptOptionKind.PATH,
            positional=True,
            arity=ScriptOptionArity.ONE_OR_MORE,
            help="Markdown files or directories to audit.",
        ),
    ),
    handler=_run_invocation,
    runtime=_SCRIPT_RUNTIME,
)


def main(argv: list[str] | None = None) -> int:
    return invoke_script(_SCRIPT_SPEC, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
