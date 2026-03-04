from __future__ import annotations

import argparse
import json
from pathlib import Path

from gabion_governance import governance_audit_impl
from gabion_governance.sppf_audit.contracts import SppfStatusConsistencyResult


def test_status_consistency_command_preserves_markdown_json_shape(tmp_path: Path, monkeypatch) -> None:
    json_output = tmp_path / "status.json"
    md_output = tmp_path / "status.md"

    monkeypatch.setattr(
        governance_audit_impl,
        "run_status_consistency",
        lambda **_kwargs: SppfStatusConsistencyResult(violations=["v"], warnings=["w"]),
    )

    args = argparse.Namespace(
        root=str(tmp_path),
        extra_path=None,
        json_output=json_output,
        md_output=md_output,
        fail_on_violations=False,
    )

    rc = governance_audit_impl._status_consistency_command(args)
    assert rc == 0

    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert payload["summary"] == {"violation_count": 1, "warning_count": 1}
    assert payload["violations"] == ["v"]
    assert payload["warnings"] == ["w"]

    markdown = md_output.read_text(encoding="utf-8")
    assert "# SPPF Status Consistency" in markdown
    assert "## Violations" in markdown
    assert "## Warnings" in markdown
