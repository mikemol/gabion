from __future__ import annotations

from pathlib import Path
from typing import Mapping

from gabion.tooling.delta import delta_gate

_GATE_ADAPTER = delta_gate.make_standard_gate_adapter(gate_id="obsolescence_unmapped")
ENV_FLAG = _GATE_ADAPTER.spec.env_flag


def _enabled(value: str | None = None) -> bool:
    return _GATE_ADAPTER.enabled(value)


def _delta_value(payload: Mapping[str, object]) -> int:
    return _GATE_ADAPTER.delta_value(payload)


def check_gate(path: Path, *, enabled: bool | None = None) -> int:
    return _GATE_ADAPTER.check_gate(path, enabled=enabled)


def main() -> int:
    return _GATE_ADAPTER.main()
