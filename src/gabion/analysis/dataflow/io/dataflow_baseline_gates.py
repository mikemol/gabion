from __future__ import annotations

from pathlib import Path

from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.order_contract import sort_once


def _resolve_baseline_path(path: object, root: Path):
    if not path:
        return None
    baseline = Path(path)
    if not baseline.is_absolute():
        baseline = root / baseline
    return baseline


def _load_baseline(path: Path) -> set[str]:
    check_deadline()
    if not path.exists():
        return set()
    try:
        raw = path.read_text()
    except OSError:
        return set()
    entries: set[str] = set()
    for line in raw.splitlines():
        check_deadline()
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entries.add(line)
    return entries


def _write_baseline(path: Path, violations: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unique = sort_once(
        set(violations),
        source="src/gabion/analysis/dataflow_baseline_gates.py:_write_baseline",
    )
    header = [
        "# gabion baseline (ratchet)",
        "# Lines list known violations to allow; new ones should fail.",
        "",
    ]
    path.write_text("\n".join(header + unique) + "\n")


def _apply_baseline(
    violations: list[str],
    baseline_allowlist: set[str],
) -> tuple[list[str], list[str]]:
    if not baseline_allowlist:
        return violations, []
    new = [line for line in violations if line not in baseline_allowlist]
    suppressed = [line for line in violations if line in baseline_allowlist]
    return new, suppressed


def resolve_baseline_path(path: object, root: Path):
    return _resolve_baseline_path(path, root)


def load_baseline(path: Path) -> set[str]:
    return _load_baseline(path)


def write_baseline(path: Path, violations: list[str]) -> None:
    _write_baseline(path, violations)


def apply_baseline(
    violations: list[str],
    baseline_allowlist: set[str],
) -> tuple[list[str], list[str]]:
    return _apply_baseline(violations, baseline_allowlist)


__all__ = [
    "_apply_baseline",
    "_load_baseline",
    "_resolve_baseline_path",
    "_write_baseline",
    "apply_baseline",
    "load_baseline",
    "resolve_baseline_path",
    "write_baseline",
]
