from __future__ import annotations

from gabion.schema import AnalysisResponse


class GabionEngine:
    def analyze(self) -> AnalysisResponse:
        return AnalysisResponse(bundles=[], stats={})
