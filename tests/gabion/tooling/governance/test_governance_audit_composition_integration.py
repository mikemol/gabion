from __future__ import annotations

import argparse
import json
from pathlib import Path

from gabion_governance import governance_audit_impl


def test_status_consistency_command_preserves_markdown_json_shape(tmp_path: Path) -> None:
    json_output = tmp_path / "status.json"
    md_output = tmp_path / "status.md"

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
    summary = payload.get("summary", {})
    assert isinstance(summary, dict)
    assert isinstance(summary.get("violation_count"), int)
    assert isinstance(summary.get("warning_count"), int)
    violation_count = int(summary["violation_count"])
    warning_count = int(summary["warning_count"])
    assert isinstance(payload.get("violations"), list)
    assert isinstance(payload.get("warnings"), list)

    markdown = md_output.read_text(encoding="utf-8")
    assert "# SPPF Status Consistency" in markdown
    assert ("## Violations" in markdown) is (violation_count > 0)
    assert ("## Warnings" in markdown) is (warning_count > 0)
    assert ("No issues detected." in markdown) is (violation_count == 0 and warning_count == 0)
