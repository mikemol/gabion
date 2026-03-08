from __future__ import annotations

from gabion_governance.compliance_render import render_status_consistency_markdown
from gabion_governance.sppf_audit.contracts import SppfStatusConsistencyResult


# gabion:behavior primary=desired
def test_render_status_consistency_markdown_includes_sections() -> None:
    rendered = render_status_consistency_markdown(
        SppfStatusConsistencyResult(violations=["v1"], warnings=["w1"])
    )
    assert "# SPPF Status Consistency" in rendered.markdown
    assert "## Violations" in rendered.markdown
    assert "- v1" in rendered.markdown
    assert "## Warnings" in rendered.markdown
    assert "- w1" in rendered.markdown
