from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, TypeVar

from gabion.analysis.foundation.json_types import JSONObject

_KernelResult = TypeVar("_KernelResult")


@dataclass(frozen=True)
class ScanKernelRequest:
    paths: list[Path]
    project_root: Path | None
    ignore_params: set[str]
    strictness: str
    external_filter: bool
    transparent_decorators: set[str] | None
    parse_failure_witnesses: list[JSONObject] | None
    analysis_index: object | None


@dataclass(frozen=True)
class ScanKernelPass(Generic[_KernelResult]):
    pass_id: str
    run: Callable[[object], _KernelResult]


@dataclass(frozen=True)
class ScanKernelDeps(Generic[_KernelResult]):
    run_indexed_pass_fn: Callable[..., _KernelResult]


def run_canonical_scan_ingress(
    *,
    request: ScanKernelRequest,
    spec: ScanKernelPass[_KernelResult],
    deps: ScanKernelDeps[_KernelResult],
) -> _KernelResult:
    return deps.run_indexed_pass_fn(
        request.paths,
        project_root=request.project_root,
        ignore_params=request.ignore_params,
        strictness=request.strictness,
        external_filter=request.external_filter,
        transparent_decorators=request.transparent_decorators,
        parse_failure_witnesses=request.parse_failure_witnesses,
        analysis_index=request.analysis_index,
        spec=spec,
    )
