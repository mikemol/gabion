from __future__ import annotations

from typing import Dict, List

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
