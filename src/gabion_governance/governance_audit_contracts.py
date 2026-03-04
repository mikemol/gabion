from __future__ import annotations

from dataclasses import dataclass

from gabion_governance.compliance_render.contracts import ComplianceRenderResult
from gabion_governance.docflow_audit.contracts import DocflowDomainResult
from gabion_governance.sppf_audit.contracts import SppfGraphResult, SppfStatusConsistencyResult


@dataclass(frozen=True)
class GovernanceAuditAggregateResult:
    docflow: DocflowDomainResult | None = None
    sppf_graph: SppfGraphResult | None = None
    sppf_status: SppfStatusConsistencyResult | None = None
    compliance_render: ComplianceRenderResult | None = None
