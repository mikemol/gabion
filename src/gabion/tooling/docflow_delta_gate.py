from __future__ import annotations

from pathlib import Path
from typing import Mapping

from gabion.tooling import delta_gate

ENV_FLAG = delta_gate.DOCFLOW_DELTA_ENV_FLAG


def _enabled(value: str | None = None) -> bool:
    return delta_gate.docflow_enabled(value)


def _delta_value(payload: Mapping[str, object], key: str) -> int:
    return delta_gate.docflow_delta_value(payload, key)


def check_gate(path: Path, *, enabled: bool | None = None) -> int:
    return delta_gate.check_docflow_gate(path, enabled=enabled)


def main() -> int:
    return delta_gate.docflow_main()
