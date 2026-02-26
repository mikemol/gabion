from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel


class BundleDTO(BaseModel):
    fields: List[str]
    locations: List[str]
    suggested_name: str
    tier: int


class AnalysisRequest(BaseModel):
    root_path: str


class AnalysisResponse(BaseModel):
    bundles: List[BundleDTO]
    stats: Dict[str, int]


class SynthesisBundleDTO(BaseModel):
    bundle: List[str]
    tier: int


class SynthesisRequest(BaseModel):
    bundles: List[SynthesisBundleDTO]
    field_types: Dict[str, str] = {}
    existing_names: List[str] = []
    frequency: Dict[str, int] = {}
    fallback_prefix: str = "Bundle"
    max_tier: int = 2
    min_bundle_size: int = 2
    allow_singletons: bool = False
    merge_overlap_threshold: float = 0.75


class SynthesisFieldDTO(BaseModel):
    name: str
    type_hint: Optional[str] = None
    source_params: List[str] = []


class SynthesisProtocolDTO(BaseModel):
    name: str
    fields: List[SynthesisFieldDTO]
    bundle: List[str]
    tier: int
    rationale: Optional[str] = None
    evidence: List[str] = []


class SynthesisResponse(BaseModel):
    protocols: List[SynthesisProtocolDTO]
    warnings: List[str] = []
    errors: List[str] = []


class RefactorFieldDTO(BaseModel):
    name: str
    type_hint: Optional[str] = None


class RefactorCompatibilityShimDTO(BaseModel):
    enabled: bool = True
    emit_deprecation_warning: bool = True
    emit_overload_stubs: bool = True


class RefactorRequest(BaseModel):
    protocol_name: str
    bundle: List[str]
    fields: List[RefactorFieldDTO] = []
    target_path: str
    target_functions: List[str] = []
    compatibility_shim: bool | RefactorCompatibilityShimDTO = False
    ambient_rewrite: bool = False
    rationale: Optional[str] = None


class TextEditDTO(BaseModel):
    path: str
    start: Tuple[int, int]
    end: Tuple[int, int]
    replacement: str


class RewritePlanEntryDTO(BaseModel):
    kind: str
    status: str
    target: str
    summary: str
    non_rewrite_reasons: List[str] = []


class RefactorResponse(BaseModel):
    edits: List[TextEditDTO] = []
    rewrite_plans: List[RewritePlanEntryDTO] = []
    warnings: List[str] = []
    errors: List[str] = []


class LintEntryDTO(BaseModel):
    path: str
    line: int
    col: int
    code: str
    message: str
    severity: str = "warning"


class AspfOneCellDTO(BaseModel):
    source: str
    target: str
    representative: str
    basis_path: List[str]
    kind: Optional[str] = None
    surface: Optional[str] = None
    metadata: Dict[str, Any] = {}


class AspfTwoCellWitnessDTO(BaseModel):
    left: AspfOneCellDTO
    right: AspfOneCellDTO
    witness_id: str
    reason: str


class AspfTraceDTO(BaseModel):
    format_version: int = 1
    trace_id: str
    started_at_utc: str
    controls: Dict[str, Any] = {}
    one_cells: List[AspfOneCellDTO] = []
    two_cell_witnesses: List[AspfTwoCellWitnessDTO] = []
    cofibration_witnesses: List[Dict[str, Any]] = []
    surface_representatives: Dict[str, str] = {}
    imported_trace_count: int = 0


class AspfEquivalenceSurfaceDTO(BaseModel):
    surface: str
    classification: str
    baseline_representative: Optional[str] = None
    current_representative: Optional[str] = None
    witness_id: Optional[str] = None
    representative_selection: Optional[Dict[str, Any]] = None


class AspfEquivalenceDTO(BaseModel):
    format_version: int = 1
    trace_id: str
    verdict: str
    surface_table: List[AspfEquivalenceSurfaceDTO] = []


class AspfOpportunityDTO(BaseModel):
    opportunity_id: str
    kind: str
    confidence: float
    affected_surfaces: List[str] = []
    witness_ids: List[str] = []
    reason: str


class AspfOpportunitiesDTO(BaseModel):
    format_version: int = 1
    trace_id: str
    opportunities: List[AspfOpportunityDTO] = []


class AspfDeltaRecordDTO(BaseModel):
    seq: int
    ts_utc: str
    event_kind: str
    phase: str
    analysis_state: Optional[str] = None
    mutation_target: str
    mutation_value: Any = None
    one_cell_ref: Optional[str] = None


class AspfDeltaLedgerDTO(BaseModel):
    format_version: int = 1
    trace_id: str
    records: List[AspfDeltaRecordDTO] = []


class AspfResumeProjectionDTO(BaseModel):
    analysis_state: Optional[str] = None
    semantic_surfaces: Dict[str, Any] = {}
    exit_code: Optional[int] = None


class AspfStateDTO(BaseModel):
    format_version: int = 1
    state_id: str
    session_id: str
    step_id: str
    created_at_utc: str
    command_profile: str
    analysis_manifest_digest: Optional[str] = None
    resume_source: Optional[str] = None
    resume_compatibility_status: Optional[str] = None
    trace: Dict[str, Any] = {}
    equivalence: Dict[str, Any] = {}
    opportunities: Dict[str, Any] = {}
    semantic_surfaces: Dict[str, Any] = {}
    resume_projection: Dict[str, Any] = {}
    delta_ledger: Dict[str, Any] = {}
    exit_code: Optional[int] = None
    analysis_state: Optional[str] = None


class DataflowAuditResponseDTO(BaseModel):
    exit_code: int = 0
    timeout: bool = False
    analysis_state: Optional[str] = None
    classification: Optional[str] = None
    error_kind: Optional[str] = None
    errors: List[str] = []
    lint_lines: List[str] = []
    lint_entries: List[LintEntryDTO] = []
    aspf_trace: Optional[AspfTraceDTO] = None
    aspf_equivalence: Optional[AspfEquivalenceDTO] = None
    aspf_opportunities: Optional[AspfOpportunitiesDTO] = None
    aspf_delta_ledger: Optional[AspfDeltaLedgerDTO] = None
    aspf_state: Optional[AspfStateDTO] = None
    payload: Dict[str, Any] = {}


class SynthesisPlanResponseDTO(SynthesisResponse):
    pass


class RefactorProtocolResponseDTO(RefactorResponse):
    pass


class StructureDiffResponseDTO(BaseModel):
    exit_code: int = 0
    diff: Optional[Dict[str, Any]] = None
    errors: List[str] = []


class DecisionDiffResponseDTO(BaseModel):
    exit_code: int = 0
    diff: Optional[Dict[str, Any]] = None
    errors: List[str] = []


class StructureReuseResponseDTO(BaseModel):
    exit_code: int = 0
    reuse: Optional[Dict[str, Any]] = None
    lemma_stubs: Optional[str] = None
    errors: List[str] = []


class LspParityCommandResultDTO(BaseModel):
    command_id: str
    maturity: str
    require_lsp_carrier: bool
    parity_required: bool
    lsp_validated: bool
    parity_ok: bool
    error: Optional[str] = None


class LspParityGateResponseDTO(BaseModel):
    exit_code: int = 0
    checked_commands: List[LspParityCommandResultDTO] = []
    errors: List[str] = []
