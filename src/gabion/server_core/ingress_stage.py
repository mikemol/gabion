from __future__ import annotations

from pathlib import Path
from typing import Mapping

from gabion.server_core.stage_contracts import (
    IngressModeSelector,
    PayloadNormalizer,
    PayloadOptionsParser,
    StageIngressResult,
)


def run_ingress_stage(
    *,
    payload: dict[str, object],
    root: Path,
    normalize_payload: PayloadNormalizer,
    parse_options: PayloadOptionsParser,
    select_mode: IngressModeSelector,
) -> StageIngressResult:
    normalized_payload = normalize_payload(payload=payload)
    options = parse_options(payload=normalized_payload, root=root)
    mode = select_mode(payload=normalized_payload, options=options)
    return StageIngressResult(payload=normalized_payload, options=options, mode=mode)


def default_mode_selector(*, payload: Mapping[str, object], options: object) -> str:
    if payload.get("aux_operation") is not None:
        return "aux_operation"
    return "analysis"
