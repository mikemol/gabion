from __future__ import annotations

from pathlib import Path

from gabion.governance_paths import GovernancePathConfig


# gabion:evidence E:function_site::governance_paths.py::gabion.governance_paths.GovernancePathConfig
def test_governance_paths_resolve_root_relative_paths(tmp_path: Path) -> None:
    cfg = GovernancePathConfig(
        src_prefix="src/",
        in_prefix="in/",
        sppf_checklist_rel="docs/sppf_checklist.md",
        influence_index_rel="docs/influence_index.md",
    )

    assert cfg.in_dir(root=tmp_path) == tmp_path / "in"
    assert cfg.sppf_checklist_path(root=tmp_path) == tmp_path / "docs" / "sppf_checklist.md"
    assert cfg.influence_index_path(root=tmp_path) == tmp_path / "docs" / "influence_index.md"
