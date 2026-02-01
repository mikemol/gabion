"""Static analysis subpackage for Gabion."""

from .dataflow_audit import (
    AnalysisResult,
    AuditConfig,
    analyze_paths,
    compute_violations,
    render_dot,
    render_report,
)

__all__ = [
    "AnalysisResult",
    "AuditConfig",
    "analyze_paths",
    "compute_violations",
    "render_dot",
    "render_report",
]
