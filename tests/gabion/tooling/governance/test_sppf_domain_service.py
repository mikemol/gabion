from __future__ import annotations

from pathlib import Path

from gabion_governance.sppf_audit import build_sppf_graph, run_status_consistency
from gabion_governance.docflow_audit import Doc


def test_build_sppf_graph_returns_typed_result() -> None:
    result = build_sppf_graph(
        root=Path("."),
        issues_json=None,
        build_graph=lambda *_args, **_kwargs: {"format_version": 1},
    )
    assert result.graph["format_version"] == 1


def test_run_status_consistency_merges_axis_and_sync_findings() -> None:
    result = run_status_consistency(
        root=Path("."),
        extra_paths=None,
        load_docs=lambda **_kwargs: {"README.md": Doc(frontmatter={}, body="")},
        axis_audit=lambda *_args, **_kwargs: (["axis-v"], ["axis-w"]),
        sync_check=lambda *_args, **_kwargs: (["sync-v"], ["sync-w"]),
    )
    assert result.violations == ["axis-v", "sync-v"]
    assert result.warnings == ["axis-w", "sync-w"]
    assert result.payload["summary"]["violation_count"] == 2
