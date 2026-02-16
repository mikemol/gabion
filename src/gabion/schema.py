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
    rationale: Optional[str] = None


class TextEditDTO(BaseModel):
    path: str
    start: Tuple[int, int]
    end: Tuple[int, int]
    replacement: str


class RefactorResponse(BaseModel):
    edits: List[TextEditDTO] = []
    warnings: List[str] = []
    errors: List[str] = []


class LintEntryDTO(BaseModel):
    path: str
    line: int
    col: int
    code: str
    message: str
    severity: str = "warning"


class DataflowAuditResponseDTO(BaseModel):
    exit_code: int = 0
    timeout: bool = False
    analysis_state: Optional[str] = None
    errors: List[str] = []
    lint_lines: List[str] = []
    lint_entries: List[LintEntryDTO] = []
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
