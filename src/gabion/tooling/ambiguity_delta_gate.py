from __future__ import annotations

from pathlib import Path
from typing import Mapping

from gabion.tooling import delta_gate

ENV_FLAG = delta_gate.AMBIGUITY_DELTA_ENV_FLAG


def _enabled(value: str | None = None) -> bool:
    return delta_gate.ambiguity_enabled(value)


def _delta_value(payload: Mapping[str, object]) -> int:
    return delta_gate.ambiguity_delta_value(payload)


def check_gate(path: Path, *, enabled: bool | None = None) -> int:
    return delta_gate.check_ambiguity_gate(path, enabled=enabled)


def main() -> int:
    return delta_gate.ambiguity_main()
