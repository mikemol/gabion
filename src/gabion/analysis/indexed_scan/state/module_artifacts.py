# gabion:decision_protocol_module
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, cast

from gabion.analysis.foundation.json_types import ParseFailureWitnesses


@dataclass(frozen=True)
class BuildModuleArtifactsDeps:
    check_deadline_fn: Callable[[], None]
    parse_module_error_types: tuple[type[BaseException], ...]
    record_parse_failure_witness_fn: Callable[..., None]


def build_module_artifacts(
    paths: list[Path],
    *,
    specs: tuple[object, ...],
    parse_failure_witnesses: ParseFailureWitnesses,
    parse_module: Callable[[Path], ast.Module],
    deps: BuildModuleArtifactsDeps,
) -> tuple[object, ...]:
    deps.check_deadline_fn()
    if not specs:
        return ()
    parse_cache: dict[Path, object] = {}
    accumulators = [spec.init() for spec in specs]
    for path in paths:
        deps.check_deadline_fn()
        parsed = parse_cache.get(path)
        if parsed is None:
            try:
                parsed = parse_module(path)
            except deps.parse_module_error_types as exc:
                parsed = exc
            parse_cache[path] = parsed
        if type(parsed) is not ast.Module:
            parsed_error = cast(BaseException, parsed)
            for spec in specs:
                deps.check_deadline_fn()
                deps.record_parse_failure_witness_fn(
                    sink=parse_failure_witnesses,
                    path=path,
                    stage=spec.stage,
                    error=cast(Exception, parsed_error),
                )
            continue
        parsed_module = cast(ast.Module, parsed)
        for idx, spec in enumerate(specs):
            deps.check_deadline_fn()
            spec.fold(accumulators[idx], path, parsed_module)
    return tuple(
        spec.finish(accumulator) for spec, accumulator in zip(specs, accumulators)
    )
