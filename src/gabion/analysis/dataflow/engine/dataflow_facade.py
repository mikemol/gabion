# gabion:boundary_normalization_module
from __future__ import annotations

"""Facade compatibility module for legacy indexed-dataflow symbols."""

import importlib

_RUNTIME_MODULE = "gabion.analysis.dataflow.engine.dataflow_indexed_file_scan"
_runtime = importlib.import_module(_RUNTIME_MODULE)


def _parse_lint_location(*args, **kwargs):
    from gabion.analysis.dataflow.engine.dataflow_lint_helpers import (
        _parse_lint_location as _parse_lint_location_impl,
    )

    return _parse_lint_location_impl(*args, **kwargs)


def _resolve_method_in_hierarchy(*args, **kwargs):
    outcome = _runtime._resolve_method_in_hierarchy(*args, **kwargs)
    resolved = getattr(outcome, "resolved", None)
    if resolved is not None:
        return resolved
    return outcome


def _internal_broad_type_lint_lines(
    paths,
    *,
    project_root,
    ignore_params,
    strictness,
    external_filter,
    transparent_decorators=None,
    parse_failure_witnesses,
    analysis_index=None,
):
    if analysis_index is None:
        analysis_index = _runtime._build_analysis_index(
            list(paths),
            project_root=project_root,
            ignore_params=set(ignore_params),
            strictness=strictness,
            external_filter=external_filter,
            transparent_decorators=transparent_decorators,
            parse_failure_witnesses=list(parse_failure_witnesses),
        )
    return _runtime._internal_broad_type_lint_lines(
        list(paths),
        project_root=project_root,
        ignore_params=set(ignore_params),
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
        parse_failure_witnesses=list(parse_failure_witnesses),
        analysis_index=analysis_index,
    )


def __getattr__(name: str):
    return getattr(_runtime, name)


def __dir__() -> list[str]:
    return sorted(set(dir(_runtime)))
