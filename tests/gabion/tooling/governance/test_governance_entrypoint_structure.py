from __future__ import annotations

from pathlib import Path


# gabion:behavior primary=desired
def test_governance_entrypoint_remains_thin() -> None:
    entrypoint = Path("src/gabion_governance/governance_entrypoint.py")
    lines = entrypoint.read_text(encoding="utf-8").splitlines()
    assert len(lines) <= 140
