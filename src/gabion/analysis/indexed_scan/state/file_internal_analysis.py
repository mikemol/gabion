# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from gabion.analysis.foundation.json_types import JSONObject


@dataclass(frozen=True)
class AnalyzeFileInternalDeps:
    check_deadline_fn: Callable[[], None]
    analyze_function_default_fn: Callable[..., object]
    audit_config_ctor: Callable[..., object]
    ingest_python_file_fn: Callable[..., object]
    parse_module_source_fn: Callable[..., object]
    collect_functions_fn: Callable[..., object]
    collect_return_aliases_fn: Callable[..., object]
    load_file_scan_resume_state_fn: Callable[..., object]
    serialize_file_scan_resume_state_fn: Callable[..., JSONObject]
    profiling_payload_fn: Callable[..., JSONObject]
    enclosing_class_fn: Callable[..., object]
    enclosing_scopes_fn: Callable[..., object]
    enclosing_function_scopes_fn: Callable[..., object]
    function_key_fn: Callable[..., str]
    decorators_transparent_fn: Callable[..., bool]
    param_names_fn: Callable[..., list[str]]
    param_spans_fn: Callable[..., object]
    collect_local_class_bases_fn: Callable[..., object]
    resolve_local_method_in_hierarchy_fn: Callable[..., object]
    is_test_path_fn: Callable[..., bool]
    parent_annotator_factory: object
    file_scan_progress_emit_interval: int
    progress_emit_min_interval_seconds: float
    analyze_ingested_file_fn: Callable[..., object]


def analyze_file_internal(
    path,
    recursive: bool = True,
    *,
    config=None,
    resume_state=None,
    on_progress=None,
    on_profile=None,
    analyze_function_fn=None,
    deps: AnalyzeFileInternalDeps,
):
    deps.check_deadline_fn()
    if analyze_function_fn is None:
        analyze_function_fn = deps.analyze_function_default_fn
    if config is None:
        config = deps.audit_config_ctor()
    ingest_carrier = deps.ingest_python_file_fn(
        path,
        config=config,
        recursive=recursive,
        parse_module=deps.parse_module_source_fn,
        collect_functions=deps.collect_functions_fn,
        collect_return_aliases=deps.collect_return_aliases_fn,
        load_resume_state=deps.load_file_scan_resume_state_fn,
        serialize_resume_state=deps.serialize_file_scan_resume_state_fn,
        profiling_payload=deps.profiling_payload_fn,
        analyze_function=analyze_function_fn,
        enclosing_class=deps.enclosing_class_fn,
        enclosing_scopes=deps.enclosing_scopes_fn,
        enclosing_function_scopes=deps.enclosing_function_scopes_fn,
        function_key=deps.function_key_fn,
        decorators_transparent=deps.decorators_transparent_fn,
        param_names=deps.param_names_fn,
        param_spans=deps.param_spans_fn,
        collect_local_class_bases=deps.collect_local_class_bases_fn,
        resolve_local_method_in_hierarchy=deps.resolve_local_method_in_hierarchy_fn,
        is_test_path=deps.is_test_path_fn,
        check_deadline=deps.check_deadline_fn,
        parent_annotator_factory=deps.parent_annotator_factory,
        progress_emit_interval=deps.file_scan_progress_emit_interval,
        progress_min_interval_seconds=deps.progress_emit_min_interval_seconds,
        on_progress=on_progress,
        on_profile=on_profile,
        resume_state=resume_state,
    )
    return deps.analyze_ingested_file_fn(
        ingest_carrier,
        recursive=recursive,
        config=config,
        on_profile=on_profile,
    )

