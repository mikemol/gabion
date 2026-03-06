from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping


_POLICY_RESULT_REQUIRED_FIELDS = (
    "rule_id",
    "status",
    "violations",
    "baseline_mode",
    "source_tool",
    "timestamp_utc",
    "input_scope",
)


def make_policy_result(
    *,
    rule_id: str,
    status: str,
    violations: list[dict[str, Any]],
    baseline_mode: str,
    source_tool: str,
    input_scope: Mapping[str, Any],
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "status": status,
        "violations": violations,
        "baseline_mode": baseline_mode,
        "source_tool": source_tool,
        "timestamp_utc": timestamp_utc or datetime.now(timezone.utc).isoformat(),
        "input_scope": dict(input_scope),
    }


def write_policy_result(*, path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def load_policy_result(path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    for field in _POLICY_RESULT_REQUIRED_FIELDS:
        if field not in raw:
            return None
    if not isinstance(raw.get("violations"), list):
        return None
    return dict(raw)


__all__ = [
    "load_policy_result",
    "make_policy_result",
    "write_policy_result",
]
