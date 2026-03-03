# gabion:decision_protocol_module
from __future__ import annotations

"""Snapshot contract carriers for structure/decision surfaces."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DecisionSnapshotSurfaces:
    decision_surfaces: list[str]
    value_decision_surfaces: list[str]


@dataclass(frozen=True)
class StructureSnapshotDiffRequest:
    baseline_path: Path
    current_path: Path
