from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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


class SynthesisResponse(BaseModel):
    protocols: List[SynthesisProtocolDTO]
    warnings: List[str] = []
    errors: List[str] = []


class RefactorFieldDTO(BaseModel):
    name: str
    type_hint: Optional[str] = None


class RefactorRequest(BaseModel):
    protocol_name: str
    bundle: List[str]
    fields: List[RefactorFieldDTO] = []
    target_path: str
    target_functions: List[str] = []
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
