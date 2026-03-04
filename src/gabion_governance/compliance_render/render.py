from __future__ import annotations

from gabion_governance.sppf_audit.contracts import SppfStatusConsistencyResult

from .contracts import ComplianceRenderResult, RenderedArtifact


def render_status_consistency_markdown(result: SppfStatusConsistencyResult) -> RenderedArtifact:
    lines = [
        "# SPPF Status Consistency",
        "",
        f"- violations: {len(result.violations)}",
        f"- warnings: {len(result.warnings)}",
        "",
    ]
    if result.violations:
        lines.extend(["## Violations", ""])
        lines.extend(f"- {item}" for item in result.violations)
        lines.append("")
    if result.warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {item}" for item in result.warnings)
        lines.append("")
    if not result.warnings and not result.violations:
        lines.extend(["No issues detected.", ""])
    return RenderedArtifact(markdown="\n".join(lines))


def render_compliance(result: SppfStatusConsistencyResult) -> ComplianceRenderResult:
    return ComplianceRenderResult(status_consistency=render_status_consistency_markdown(result))
