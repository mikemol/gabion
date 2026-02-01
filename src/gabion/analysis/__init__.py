"""Static analysis subpackage for Gabion."""

from .dataflow_audit import (
    AnalysisResult,
    AuditConfig,
    analyze_paths,
    build_synthesis_plan,
    compute_violations,
    render_dot,
    render_report,
    render_synthesis_section,
)

__all__ = [
    "AnalysisResult",
    "AuditConfig",
    "analyze_paths",
    "build_synthesis_plan",
    "compute_violations",
    "render_dot",
    "render_report",
    "render_synthesis_section",
]
