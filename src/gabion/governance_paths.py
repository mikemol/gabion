from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GovernancePathConfig:
    """Centralized governance path/linkage rules for repo-local tooling."""

    src_prefix: str = "src/"
    in_prefix: str = "in/"
    sppf_checklist_rel: str = "docs/sppf_checklist.md"
    influence_index_rel: str = "docs/influence_index.md"

    def in_dir(self, *, root: Path) -> Path:
        return root / self.in_prefix.rstrip("/")

    def sppf_checklist_path(self, *, root: Path) -> Path:
        return root / self.sppf_checklist_rel

    def influence_index_path(self, *, root: Path) -> Path:
        return root / self.influence_index_rel

    def is_sppf_relevant_path(self, path: str) -> bool:
        if path == self.sppf_checklist_rel:
            return True
        return path.startswith(self.src_prefix) or path.startswith(self.in_prefix)


GOVERNANCE_PATHS = GovernancePathConfig()
