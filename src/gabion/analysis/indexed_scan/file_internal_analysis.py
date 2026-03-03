# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from gabion.analysis.json_types import JSONObject


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


def analyze_file_internal_from_runtime_module(
    path,
    recursive: bool = True,
    *,
    config=None,
    resume_state=None,
    on_progress=None,
    on_profile=None,
    analyze_function_fn=None,
    runtime_module,
    check_deadline_fn: Callable[[], None],
    analyze_function_default_fn: Callable[..., object],
    audit_config_ctor: Callable[..., object],
    ingest_python_file_fn: Callable[..., object],
    analyze_ingested_file_fn: Callable[..., object],
    file_scan_progress_emit_interval: int,
    progress_emit_min_interval_seconds: float,
):
    return analyze_file_internal(
        path,
        recursive=recursive,
        config=config,
        resume_state=resume_state,
        on_progress=on_progress,
        on_profile=on_profile,
        analyze_function_fn=analyze_function_fn,
        deps=AnalyzeFileInternalDeps(
            check_deadline_fn=check_deadline_fn,
            analyze_function_default_fn=analyze_function_default_fn,
            audit_config_ctor=audit_config_ctor,
            ingest_python_file_fn=ingest_python_file_fn,
            parse_module_source_fn=runtime_module._parse_module_source,
            collect_functions_fn=runtime_module._collect_functions,
            collect_return_aliases_fn=runtime_module._collect_return_aliases,
            load_file_scan_resume_state_fn=runtime_module._load_file_scan_resume_state,
            serialize_file_scan_resume_state_fn=runtime_module._serialize_file_scan_resume_state,
            profiling_payload_fn=runtime_module._profiling_v1_payload,
            enclosing_class_fn=runtime_module._enclosing_class,
            enclosing_scopes_fn=runtime_module._enclosing_scopes,
            enclosing_function_scopes_fn=runtime_module._enclosing_function_scopes,
            function_key_fn=runtime_module._function_key,
            decorators_transparent_fn=runtime_module._decorators_transparent,
            param_names_fn=runtime_module._param_names,
            param_spans_fn=runtime_module._param_spans,
            collect_local_class_bases_fn=runtime_module._collect_local_class_bases,
            resolve_local_method_in_hierarchy_fn=(
                runtime_module._resolve_local_method_in_hierarchy
            ),
            is_test_path_fn=runtime_module._is_test_path,
            parent_annotator_factory=runtime_module.ParentAnnotator,
            file_scan_progress_emit_interval=file_scan_progress_emit_interval,
            progress_emit_min_interval_seconds=progress_emit_min_interval_seconds,
            analyze_ingested_file_fn=analyze_ingested_file_fn,
        ),
    )


def analyze_file_internal_from_runtime_module_defaults(
    path,
    recursive: bool = True,
    *,
    config=None,
    resume_state=None,
    on_progress=None,
    on_profile=None,
    analyze_function_fn=None,
    runtime_module,
):
    return analyze_file_internal_from_runtime_module(
        path,
        recursive=recursive,
        config=config,
        resume_state=resume_state,
        on_progress=on_progress,
        on_profile=on_profile,
        analyze_function_fn=analyze_function_fn,
        runtime_module=runtime_module,
        check_deadline_fn=runtime_module.check_deadline,
        analyze_function_default_fn=runtime_module._analyze_function,
        audit_config_ctor=runtime_module.AuditConfig,
        ingest_python_file_fn=runtime_module.ingest_python_file,
        analyze_ingested_file_fn=runtime_module.analyze_ingested_file,
        file_scan_progress_emit_interval=runtime_module._FILE_SCAN_PROGRESS_EMIT_INTERVAL,
        progress_emit_min_interval_seconds=(
            runtime_module._PROGRESS_EMIT_MIN_INTERVAL_SECONDS
        ),
    )
