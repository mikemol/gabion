# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

"""Analysis-index owner surface during WS-5 migration."""

import ast
import hashlib
import json
from dataclasses import dataclass, field
from functools import partial
from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Generic, TypeVar, cast

from gabion.analysis.aspf.aspf import NodeId, structural_key_atom, structural_key_json
from gabion.analysis.core.type_fingerprints import fingerprint_stage_cache_identity
from gabion.analysis.dataflow.engine.dataflow_resume_serialization import (
    _CACHE_IDENTITY_DIGEST_HEX,
    _CACHE_IDENTITY_PREFIX,
    _CacheIdentity,
    _analysis_collection_resume_path_key as _analysis_collection_resume_path_key_resume,
    _build_analysis_collection_resume_payload,
    _load_file_scan_resume_state,
    _load_analysis_collection_resume_payload,
    _load_analysis_index_resume_payload as _load_analysis_index_resume_payload_owner,
    _serialize_file_scan_resume_state,
    _serialize_analysis_index_resume_payload as _serialize_analysis_index_resume_payload_owner,
)
from gabion.analysis.dataflow.engine.dataflow_parse_failures import (
    _PARSE_MODULE_ERROR_TYPES,
    _parse_failure_sink,
    _record_parse_failure_witness,
)
from gabion.analysis.dataflow.engine.dataflow_contracts import (
    AuditConfig,
    ClassInfo,
    FunctionInfo,
    SymbolTable,
)
from gabion.analysis.dataflow.io.dataflow_parse_helpers import _ParseModuleStage
from gabion.analysis.dataflow.engine.dataflow_evidence_helpers import (
    ImportVisitor,
    ParentAnnotator,
    _base_identifier,
    _collect_module_exports,
    _enclosing_class_scopes,
    _module_name,
    _resolve_callee,
)
from gabion.analysis.dataflow.engine.dataflow_function_index_helpers import (
    _collect_functions,
    _enclosing_class,
    _enclosing_function_scopes,
    _enclosing_scopes,
    _is_test_path,
    _node_span,
    _param_annotations,
    _param_defaults,
    _param_names,
    _param_spans,
)
from gabion.analysis.dataflow.engine.dataflow_function_semantics import (
    _analyze_function,
    _collect_return_aliases,
)
from gabion.analysis.dataflow.engine.dataflow_function_index_runtime_support import (
    _direct_lambda_callee_by_call_span,
    _materialize_direct_lambda_callees,
    _unused_params,
)
from gabion.analysis.dataflow.engine.dataflow_lambda_runtime_support import (
    _collect_lambda_bindings_by_caller,
    _collect_lambda_function_infos,
    _function_key,
)
from gabion.analysis.dataflow.engine.dataflow_function_index_decision_support import (
    _decision_surface_reason_map,
    _decorators_transparent,
    _value_encoded_decision_params,
)
from gabion.analysis.dataflow.engine.dataflow_local_class_hierarchy import (
    _collect_local_class_bases,
    _resolve_local_method_in_hierarchy,
)
from gabion.analysis.derivation.derivation_contract import DerivationOp
from gabion.analysis.derivation.derivation_cache import get_global_derivation_cache
from gabion.analysis.foundation.json_types import JSONObject
from gabion.analysis.foundation.timeout_context import TimeoutExceeded, check_deadline
from gabion.analysis.indexed_scan.index.analysis_index_builder import (
    AnalysisIndexBuildDeps as _AnalysisIndexBuildDeps,
    build_analysis_index as _build_analysis_index_impl,
)
from gabion.analysis.indexed_scan.index.analysis_index_module_trees import (
    AnalysisIndexModuleTreesDeps as _AnalysisIndexModuleTreesDeps,
    analysis_index_module_trees as _analysis_index_module_trees_impl,
)
from gabion.analysis.indexed_scan.index.analysis_index_stage_cache import (
    AnalysisIndexStageCacheDeps as _AnalysisIndexStageCacheDeps,
    analysis_index_stage_cache as _analysis_index_stage_cache_impl,
)
from gabion.analysis.indexed_scan.state.module_artifacts import (
    BuildModuleArtifactsDeps as _BuildModuleArtifactsDeps,
    build_module_artifacts as _build_module_artifacts_impl,
)
from gabion.analysis.indexed_scan.state.function_index_accumulator import (
    FunctionIndexAccumulatorDeps as _FunctionIndexAccumulatorDeps,
    accumulate_function_index_for_tree as _accumulate_function_index_for_tree_impl,
)
from gabion.analysis.indexed_scan.state.file_internal_analysis import (
    AnalyzeFileInternalDeps as _AnalyzeFileInternalDeps,
    analyze_file_internal as _analyze_file_internal_impl,
)
from gabion.analysis.indexed_scan.scanners.edge_param_events import (
    iter_resolved_edge_param_events as _iter_resolved_edge_param_events_impl,
)
from gabion.analysis.indexed_scan.scanners.key_aliases import (
    stage_cache_key_aliases as _stage_cache_key_aliases_impl,
)
from gabion.analysis.indexed_scan.scanners.class_index_accumulator import (
    AccumulateClassIndexForTreeDeps as _AccumulateClassIndexForTreeDeps,
    accumulate_class_index_for_tree as _accumulate_class_index_for_tree_impl,
)
from gabion.invariants import never
from gabion.ingest.python_ingest import ingest_python_file
from gabion.order_contract import sort_once


_IndexedPassResult = TypeVar("_IndexedPassResult")
_ModuleArtifactAcc = TypeVar("_ModuleArtifactAcc")
_ModuleArtifactOut = TypeVar("_ModuleArtifactOut")

OptionalProjectRoot = Path
OptionalDecorators = set[str]
OptionalParseFailures = list[JSONObject]
OptionalAnalysisIndex = object


@dataclass(frozen=True)
class _IndexedPassContext:
    paths: list[Path]
    project_root: OptionalProjectRoot
    ignore_params: set[str]
    strictness: str
    external_filter: bool
    transparent_decorators: OptionalDecorators
    parse_failure_witnesses: list[JSONObject]
    analysis_index: object


@dataclass(frozen=True)
class _IndexedPassSpec(Generic[_IndexedPassResult]):
    pass_id: str
    run: Callable[[_IndexedPassContext], _IndexedPassResult]


@dataclass(frozen=True)
class _ModuleArtifactSpec(Generic[_ModuleArtifactAcc, _ModuleArtifactOut]):
    artifact_id: str
    stage: object
    init: Callable[[], _ModuleArtifactAcc]
    fold: Callable[[_ModuleArtifactAcc, Path, ast.Module], None]
    finish: Callable[[_ModuleArtifactAcc], _ModuleArtifactOut]


@dataclass(frozen=True)
class _ResolvedCallEdge:
    caller: object
    call: object
    callee: object


@dataclass(frozen=True)
class _ResolvedEdgeParamEvent:
    kind: str
    param: str
    value: object
    countable: bool


@dataclass(frozen=True)
class _CacheSemanticContext:
    forest_spec_id: str = ""
    fingerprint_seed_revision: str = ""


@dataclass(frozen=True)
class _StageCacheIdentitySpec:
    stage: str
    forest_spec_id: str
    fingerprint_seed_revision: str
    normalized_config: object


@dataclass
class _AnalysisIndexCarrier:
    by_name: dict[str, list[object]]
    by_qual: dict[str, object]
    symbol_table: object
    class_index: dict[str, object]
    parsed_modules_by_path: dict[Path, ast.Module] = field(default_factory=dict)
    module_parse_errors_by_path: dict[Path, Exception] = field(default_factory=dict)
    stage_cache_by_key: dict[object, dict[Path, object]] = field(default_factory=dict)
    index_cache_identity: str = ""
    projection_cache_identity: str = ""
    transitive_callers: object = None
    resolved_call_edges: object = None
    resolved_transparent_call_edges: object = None
    resolved_transparent_edges_by_caller: object = None


@dataclass
class _FunctionIndexAccumulator:
    by_name: dict[str, list[object]] = field(default_factory=lambda: defaultdict(list))
    by_qual: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class _PhaseWorkProgress:
    work_done: int
    work_total: int


_EMPTY_CACHE_SEMANTIC_CONTEXT = _CacheSemanticContext()

_ANALYSIS_INDEX_STAGE_CACHE_OP = DerivationOp(
    name="analysis_index.stage_cache",
    version=1,
    scope="gabion.analysis.dataflow_indexed_file_scan",
)
_ANALYSIS_PROFILING_FORMAT_VERSION = 1
_PROGRESS_EMIT_MIN_INTERVAL_SECONDS = 1.0
_FILE_SCAN_PROGRESS_EMIT_INTERVAL = 1


def _phase_work_progress(*, work_done: int, work_total: int) -> _PhaseWorkProgress:
    check_deadline()
    normalized_total = max(int(work_total), 0)
    normalized_done = max(int(work_done), 0)
    if normalized_total:
        normalized_done = min(normalized_done, normalized_total)
    return _PhaseWorkProgress(work_done=normalized_done, work_total=normalized_total)


_phase_work_progress_owner = _phase_work_progress


def _default_parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


_analysis_index_ctor_runtime = _AnalysisIndexCarrier

_function_index_acc_ctor_runtime = _FunctionIndexAccumulator


_FUNCTION_INDEX_ACCUMULATOR_DEPS = _FunctionIndexAccumulatorDeps(
    check_deadline_fn=check_deadline,
    collect_functions_fn=_collect_functions,
    parent_annotator_ctor=ParentAnnotator,
    module_name_fn=_module_name,
    collect_lambda_function_infos_fn=_collect_lambda_function_infos,
    collect_lambda_bindings_by_caller_fn=_collect_lambda_bindings_by_caller,
    direct_lambda_callee_by_call_span_fn=_direct_lambda_callee_by_call_span,
    collect_return_aliases_fn=_collect_return_aliases,
    enclosing_class_fn=_enclosing_class,
    enclosing_scopes_fn=_enclosing_scopes,
    enclosing_function_scopes_fn=_enclosing_function_scopes,
    analyze_function_fn=_analyze_function,
    is_test_path_fn=_is_test_path,
    materialize_direct_lambda_callees_fn=_materialize_direct_lambda_callees,
    unused_params_fn=_unused_params,
    decision_surface_reason_map_fn=_decision_surface_reason_map,
    value_encoded_decision_params_fn=_value_encoded_decision_params,
    param_names_fn=_param_names,
    param_annotations_fn=_param_annotations,
    param_defaults_fn=_param_defaults,
    decorators_transparent_fn=_decorators_transparent,
    param_spans_fn=_param_spans,
    node_span_fn=_node_span,
    function_info_ctor=FunctionInfo,
)

_accumulate_function_index_for_tree_runtime = partial(
    _accumulate_function_index_for_tree_impl,
    deps=_FUNCTION_INDEX_ACCUMULATOR_DEPS,
)


def _function_index_module_artifact_spec_runtime(
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    transparent_decorators,
) -> _ModuleArtifactSpec[
    _FunctionIndexAccumulator,
    tuple[dict[str, list[FunctionInfo]], dict[str, FunctionInfo]],
]:
    return _ModuleArtifactSpec[
        _FunctionIndexAccumulator,
        tuple[dict[str, list[FunctionInfo]], dict[str, FunctionInfo]],
    ](
        artifact_id="function_index",
        stage=_ParseModuleStage.FUNCTION_INDEX,
        init=_FunctionIndexAccumulator,
        fold=lambda acc, path, tree: _accumulate_function_index_for_tree_runtime(
            acc,
            path,
            tree,
            project_root=project_root,
            ignore_params=ignore_params,
            strictness=strictness,
            transparent_decorators=transparent_decorators,
        ),
        finish=lambda acc: (
            cast(dict[str, list[FunctionInfo]], acc.by_name),
            cast(dict[str, FunctionInfo], acc.by_qual),
        ),
    )


def _build_single_module_artifact_runtime(
    paths: list[Path],
    *,
    spec: _ModuleArtifactSpec[object, object],
    parse_failure_witnesses: list[JSONObject],
) -> object:
    check_deadline()
    raw_artifact, = _build_module_artifacts(
        paths,
        specs=(spec,),
        parse_failure_witnesses=parse_failure_witnesses,
    )
    return raw_artifact


def _build_function_index_runtime(
    paths: list[Path],
    project_root,
    ignore_params: set[str],
    strictness: str,
    transparent_decorators=None,
    *,
    parse_failure_witnesses: list[JSONObject],
) -> tuple[dict[str, list[FunctionInfo]], dict[str, FunctionInfo]]:
    raw_index = _build_single_module_artifact_runtime(
        paths,
        spec=cast(
            _ModuleArtifactSpec[object, object],
            _function_index_module_artifact_spec_runtime(
                project_root=project_root,
                ignore_params=ignore_params,
                strictness=strictness,
                transparent_decorators=transparent_decorators,
            ),
        ),
        parse_failure_witnesses=parse_failure_witnesses,
    )
    return cast(
        tuple[dict[str, list[FunctionInfo]], dict[str, FunctionInfo]],
        raw_index,
    )


def _accumulate_symbol_table_for_tree_runtime(
    table,
    path: Path,
    tree: ast.Module,
    *,
    project_root,
) -> None:
    check_deadline()
    module = _module_name(path, project_root)
    table.internal_roots.add(module.split(".")[0])
    visitor = ImportVisitor(module, table)
    visitor.visit(tree)
    import_map = {
        local: fqn for (mod, local), fqn in table.imports.items() if mod == module
    }
    exports, export_map = _collect_module_exports(
        tree,
        module_name=module,
        import_map=import_map,
    )
    table.module_exports[module] = exports
    table.module_export_map[module] = export_map


def _symbol_table_module_artifact_spec_runtime(
    *,
    project_root,
    external_filter: bool,
) -> _ModuleArtifactSpec[SymbolTable, SymbolTable]:
    return _ModuleArtifactSpec[SymbolTable, SymbolTable](
        artifact_id="symbol_table",
        stage=_ParseModuleStage.SYMBOL_TABLE,
        init=lambda: SymbolTable(external_filter=external_filter),
        fold=lambda table, path, tree: _accumulate_symbol_table_for_tree_runtime(
            table,
            path,
            tree,
            project_root=project_root,
        ),
        finish=lambda table: table,
    )


def _build_symbol_table_runtime(
    paths: list[Path],
    project_root,
    *,
    external_filter: bool,
    parse_failure_witnesses: list[JSONObject],
) -> SymbolTable:
    raw_table = _build_single_module_artifact_runtime(
        paths,
        spec=cast(
            _ModuleArtifactSpec[object, object],
            _symbol_table_module_artifact_spec_runtime(
                project_root=project_root,
                external_filter=external_filter,
            ),
        ),
        parse_failure_witnesses=parse_failure_witnesses,
    )
    return cast(SymbolTable, raw_table)


_ACCUMULATE_CLASS_INDEX_FOR_TREE_DEPS = _AccumulateClassIndexForTreeDeps(
    check_deadline_fn=check_deadline,
    parent_annotator_ctor=ParentAnnotator,
    module_name_fn=_module_name,
    enclosing_class_scopes_fn=_enclosing_class_scopes,
    base_identifier_fn=_base_identifier,
    class_info_ctor=ClassInfo,
)

_accumulate_class_index_for_tree_runtime = partial(
    _accumulate_class_index_for_tree_impl,
    deps=_ACCUMULATE_CLASS_INDEX_FOR_TREE_DEPS,
)


def _iter_monotonic_paths(paths, *, source: str):
    ordered: list[Path] = []
    previous_path_key = ""
    has_previous_path_key = False
    for path in paths:
        check_deadline()
        path_key = _analysis_collection_resume_path_key_resume(path)
        if has_previous_path_key and previous_path_key > path_key:
            never(
                "path order regression",
                source=source,
                previous_path=previous_path_key,
                current_path=path_key,
            )
        previous_path_key = path_key
        has_previous_path_key = True
        ordered.append(path)
    return ordered


_iter_monotonic_paths_owner = _iter_monotonic_paths


def _profiling_v1_payload(*, stage_ns: Mapping[str, int], counters: Mapping[str, int]) -> JSONObject:
    return {
        "format_version": _ANALYSIS_PROFILING_FORMAT_VERSION,
        "stage_ns": {str(key): int(stage_ns[key]) for key in stage_ns},
        "counters": {str(key): int(counters[key]) for key in counters},
    }


_profiling_v1_payload_owner = _profiling_v1_payload


def _progress_emit_min_interval_seconds() -> float:
    return float(_PROGRESS_EMIT_MIN_INTERVAL_SECONDS)


_progress_emit_min_interval_seconds_owner = _progress_emit_min_interval_seconds


def _path_dependency_payload(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "mtime_ns": int(stat.st_mtime_ns),
        "size": int(stat.st_size),
    }


_BUILD_MODULE_ARTIFACTS_DEPS = _BuildModuleArtifactsDeps(
    check_deadline_fn=check_deadline,
    parse_module_error_types=cast(
        tuple[type[BaseException], ...],
        _PARSE_MODULE_ERROR_TYPES,
    ),
    record_parse_failure_witness_fn=_record_parse_failure_witness,
)

_build_module_artifacts = partial(
    _build_module_artifacts_impl,
    parse_module=_default_parse_module,
    deps=_BUILD_MODULE_ARTIFACTS_DEPS,
)


def _collect_transitive_callers(
    callers_by_qual: dict[str, set[str]],
    by_qual: dict[str, object],
) -> dict[str, set[str]]:
    check_deadline()
    transitive: dict[str, set[str]] = {}
    for qual in by_qual:
        check_deadline()
        seen: set[str] = set()
        stack = list(callers_by_qual.get(qual, set()))
        while stack:
            check_deadline()
            caller = stack.pop()
            if caller in seen:
                continue
            seen.add(caller)
            stack.extend(callers_by_qual.get(caller, set()))
        transitive[qual] = seen
    return transitive


def _analysis_index_transitive_callers(
    analysis_index,
    *,
    project_root,
) -> dict[str, set[str]]:
    check_deadline()
    if analysis_index.transitive_callers is not None:
        return analysis_index.transitive_callers
    callers_by_qual: dict[str, set[str]] = defaultdict(set)
    for edge in _analysis_index_resolved_call_edges(
        analysis_index,
        project_root=project_root,
        require_transparent=False,
    ):
        check_deadline()
        callers_by_qual[edge.callee.qual].add(edge.caller.qual)
    analysis_index.transitive_callers = _collect_transitive_callers(
        callers_by_qual,
        analysis_index.by_qual,
    )
    return analysis_index.transitive_callers


def _analysis_index_resolved_call_edges(
    analysis_index,
    *,
    project_root,
    require_transparent: bool,
) -> tuple[object, ...]:
    check_deadline()
    if require_transparent:
        cached_edges = analysis_index.resolved_transparent_call_edges
    else:
        cached_edges = analysis_index.resolved_call_edges
    if cached_edges is not None:
        return cached_edges
    edges: list[object] = []
    for infos in analysis_index.by_name.values():
        check_deadline()
        for info in infos:
            check_deadline()
            for call in info.calls:
                check_deadline()
                if not call.is_test:
                    callee = _resolve_callee(
                        call.callee,
                        info,
                        analysis_index.by_name,
                        analysis_index.by_qual,
                        analysis_index.symbol_table,
                        project_root,
                        analysis_index.class_index,
                    )
                    if callee is not None and (not require_transparent or callee.transparent):
                        edges.append(_ResolvedCallEdge(caller=info, call=call, callee=callee))
    frozen_edges = tuple(edges)
    if require_transparent:
        analysis_index.resolved_transparent_call_edges = frozen_edges
    else:
        analysis_index.resolved_call_edges = frozen_edges
    return frozen_edges


def _analysis_index_resolved_call_edges_by_caller(
    analysis_index,
    *,
    project_root,
    require_transparent: bool,
) -> dict[str, tuple[object, ...]]:
    check_deadline()
    if require_transparent and analysis_index.resolved_transparent_edges_by_caller is not None:
        return analysis_index.resolved_transparent_edges_by_caller
    grouped: dict[str, list[object]] = defaultdict(list)
    for edge in _analysis_index_resolved_call_edges(
        analysis_index,
        project_root=project_root,
        require_transparent=require_transparent,
    ):
        check_deadline()
        grouped[edge.caller.qual].append(edge)
    frozen_grouped = {qual: tuple(edges) for qual, edges in grouped.items()}
    if require_transparent:
        analysis_index.resolved_transparent_edges_by_caller = frozen_grouped
    return frozen_grouped


def _reduce_resolved_call_edges(
    analysis_index,
    *,
    project_root,
    require_transparent: bool,
    spec,
):
    check_deadline()
    acc = spec.init()
    for edge in _analysis_index_resolved_call_edges(
        analysis_index,
        project_root=project_root,
        require_transparent=require_transparent,
    ):
        check_deadline()
        spec.fold(acc, edge)
    return spec.finish(acc)


def _iter_resolved_edge_param_events(
    edge,
    *,
    strictness: str,
    include_variadics_in_low_star: bool,
):
    yield from _iter_resolved_edge_param_events_impl(
        edge=edge,
        strictness=strictness,
        include_variadics_in_low_star=include_variadics_in_low_star,
        check_deadline_fn=check_deadline,
        event_ctor=_ResolvedEdgeParamEvent,
    )


def _build_call_graph(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators=None,
    parse_failure_witnesses: list[dict[str, object]],
    analysis_index=None,
) -> tuple[dict[str, list[object]], dict[str, object], dict[str, set[str]]]:
    check_deadline()
    index = analysis_index
    if index is None:
        index = _build_analysis_index(
            list(paths),
            project_root=project_root,
            ignore_params=set(ignore_params),
            strictness=strictness,
            external_filter=external_filter,
            transparent_decorators=transparent_decorators,
            parse_failure_witnesses=list(parse_failure_witnesses),
        )
    transitive_callers = _analysis_index_transitive_callers(
        index,
        project_root=project_root,
    )
    return index.by_name, index.by_qual, transitive_callers


def _sorted_text(values=None) -> tuple[str, ...]:
    if values is None:
        return ()
    cleaned = {str(value).strip() for value in values if str(value).strip()}
    return tuple(sort_once(cleaned, source="gabion.analysis.dataflow_indexed_file_scan._sorted_text.site_1"))


def _normalize_cache_config(value):
    if type(value) is dict:
        mapping = cast(dict[object, object], value)
        normalized = {
            str(key): _normalize_cache_config(mapping[key])
            for key in sort_once(mapping, source="_normalize_cache_config.mapping")
        }
        return normalized
    if type(value) is list:
        return [_normalize_cache_config(item) for item in value]
    return value


def _canonical_stage_cache_detail(detail) -> str:
    structural_detail = structural_key_atom(
        detail,
        source="gabion.analysis.dataflow_indexed_file_scan._canonical_stage_cache_detail",
    )
    canonical_json = structural_key_json(structural_detail)
    return json.dumps(canonical_json, sort_keys=False, separators=(",", ":"))


def _build_stage_cache_identity_spec(
    *,
    stage: str,
    cache_context,
    config_subset: Mapping[str, object],
):
    normalized_config = _normalize_cache_config(config_subset)
    return _StageCacheIdentitySpec(
        stage=stage,
        forest_spec_id=str(cache_context.forest_spec_id or ""),
        fingerprint_seed_revision=fingerprint_stage_cache_identity(cache_context.fingerprint_seed_revision),
        normalized_config=normalized_config,
    )


def _canonical_stage_cache_identity(spec) -> str:
    payload: dict[str, object] = {
        "stage": spec.stage,
        "forest_spec_id": spec.forest_spec_id,
        "fingerprint_seed_revision": spec.fingerprint_seed_revision,
        "config_subset": spec.normalized_config,
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"aspf:sha1:{digest}"


def _canonical_cache_identity(
    *,
    stage: str,
    cache_context,
    config_subset: Mapping[str, object],
):
    spec = _build_stage_cache_identity_spec(
        stage=stage,
        cache_context=cache_context,
        config_subset=config_subset,
    )
    canonical = _CacheIdentity.from_boundary(_canonical_stage_cache_identity(spec))
    if canonical is None:
        never("failed to construct canonical cache identity", stage=stage)  # pragma: no cover - invariant sink
    return canonical


def _cache_identity_aliases(identity: str) -> tuple[str, ...]:
    canonical = _CacheIdentity.from_boundary(identity)
    if canonical is None:
        return ("",)
    return (canonical.value,)


def _resume_variant_for_identity(
    variants: Mapping[str, dict[str, object]],
    expected_identity,
):
    direct = variants.get(expected_identity.value)
    if direct is not None:
        return direct
    return None


def _parse_stage_cache_key(
    *,
    stage,
    cache_context,
    config_subset: Mapping[str, object],
    detail,
):
    identity = _canonical_cache_identity(
        stage="parse",
        cache_context=cache_context,
        config_subset=config_subset,
    )
    return NodeId(
        kind="ParseStageCacheIdentity",
        key=(
            stage.value,
            identity.value,
            _canonical_stage_cache_detail(detail),
        ),
    )


def _index_stage_cache_identity(
    *,
    cache_context,
    config_subset: Mapping[str, object],
):
    return _canonical_cache_identity(
        stage="index",
        cache_context=cache_context,
        config_subset=config_subset,
    )


def _projection_stage_cache_identity(
    *,
    cache_context,
    config_subset: Mapping[str, object],
):
    return _canonical_cache_identity(
        stage="projection",
        cache_context=cache_context,
        config_subset=config_subset,
    )


def _stage_cache_key_aliases(key) -> tuple[object, ...]:
    return _stage_cache_key_aliases_impl(
        key,
        cache_identity_aliases_fn=_cache_identity_aliases,
        cache_identity_prefix=_CACHE_IDENTITY_PREFIX,
        cache_identity_digest_hex=_CACHE_IDENTITY_DIGEST_HEX,
        node_id_type=NodeId,
    )


def _get_stage_cache_bucket(
    analysis_index,
    *,
    scoped_cache_key,
) -> dict[Path, object]:
    stage_cache_by_key = analysis_index.stage_cache_by_key
    bucket = stage_cache_by_key.get(scoped_cache_key)
    if bucket is not None:
        return bucket
    for candidate_key in _stage_cache_key_aliases(scoped_cache_key):
        check_deadline()
        if candidate_key == scoped_cache_key:
            continue
        legacy_bucket = stage_cache_by_key.get(candidate_key)
        if legacy_bucket is not None:
            stage_cache_by_key[scoped_cache_key] = legacy_bucket
            return legacy_bucket
    return stage_cache_by_key.setdefault(scoped_cache_key, {})


def _analyze_file_internal(
    path,
    recursive: bool = True,
    *,
    config=None,
    resume_state=None,
    on_progress=None,
    on_profile=None,
    analyze_function_fn=None,
):
    from gabion.analysis.dataflow.engine.dataflow_ingested_analysis_support import (
        analyze_ingested_file as _analyze_ingested_file_owner,
    )

    return cast(
        object,
        _analyze_file_internal_impl(
            path,
            recursive=recursive,
            config=config,
            resume_state=resume_state,
            on_progress=on_progress,
            on_profile=on_profile,
            analyze_function_fn=analyze_function_fn,
            deps=_AnalyzeFileInternalDeps(
                check_deadline_fn=check_deadline,
                analyze_function_default_fn=_analyze_function,
                audit_config_ctor=AuditConfig,
                ingest_python_file_fn=ingest_python_file,
                parse_module_source_fn=_default_parse_module,
                collect_functions_fn=_collect_functions,
                collect_return_aliases_fn=_collect_return_aliases,
                load_file_scan_resume_state_fn=_load_file_scan_resume_state,
                serialize_file_scan_resume_state_fn=_serialize_file_scan_resume_state,
                profiling_payload_fn=_profiling_v1_payload,
                enclosing_class_fn=_enclosing_class,
                enclosing_scopes_fn=_enclosing_scopes,
                enclosing_function_scopes_fn=_enclosing_function_scopes,
                function_key_fn=_function_key,
                decorators_transparent_fn=_decorators_transparent,
                param_names_fn=_param_names,
                param_spans_fn=_param_spans,
                collect_local_class_bases_fn=_collect_local_class_bases,
                resolve_local_method_in_hierarchy_fn=_resolve_local_method_in_hierarchy,
                is_test_path_fn=_is_test_path,
                parent_annotator_factory=ParentAnnotator,
                file_scan_progress_emit_interval=_FILE_SCAN_PROGRESS_EMIT_INTERVAL,
                progress_emit_min_interval_seconds=_progress_emit_min_interval_seconds(),
                analyze_ingested_file_fn=_analyze_ingested_file_owner,
            ),
        ),
    )


def analyze_file(path: Path, recursive: bool = True, *, config=None):
    groups, spans, _ = _analyze_file_internal(path, recursive=recursive, config=config)
    return groups, spans


def _build_analysis_index(
    paths,
    *,
    project_root,
    ignore_params,
    strictness,
    external_filter,
    transparent_decorators,
    parse_failure_witnesses,
    resume_payload=None,
    on_progress=None,
    forest_spec_id=None,
    fingerprint_seed_revision=None,
    decision_ignore_params=None,
    decision_require_tiers=False,
):
    return cast(
        object,
        _build_analysis_index_impl(
            paths,
            project_root=project_root,
            ignore_params=ignore_params,
            strictness=strictness,
            external_filter=external_filter,
            transparent_decorators=transparent_decorators,
            parse_failure_witnesses=parse_failure_witnesses,
            resume_payload=resume_payload,
            on_progress=on_progress,
            accumulate_function_index_for_tree_fn=None,
            forest_spec_id=forest_spec_id,
            fingerprint_seed_revision=fingerprint_seed_revision,
            decision_ignore_params=decision_ignore_params,
            decision_require_tiers=decision_require_tiers,
            deps=_AnalysisIndexBuildDeps(
                check_deadline_fn=check_deadline,
                accumulate_function_index_for_tree_default_fn=_accumulate_function_index_for_tree_runtime,
                sorted_text_fn=_sorted_text,
                cache_context_ctor=_CacheSemanticContext,
                index_stage_cache_identity_fn=_index_stage_cache_identity,
                projection_stage_cache_identity_fn=_projection_stage_cache_identity,
                iter_monotonic_paths_fn=_iter_monotonic_paths,
                load_analysis_index_resume_payload_fn=_load_analysis_index_resume_payload_owner,
                function_index_acc_ctor=_function_index_acc_ctor_runtime,
                sort_once_fn=sort_once,
                profiling_payload_fn=_profiling_v1_payload,
                serialize_resume_payload_fn=_serialize_analysis_index_resume_payload_owner,
                parse_module_source_fn=_default_parse_module,
                parse_module_error_types=cast(
                    tuple[type[BaseException], ...],
                    _PARSE_MODULE_ERROR_TYPES,
                ),
                record_parse_failure_witness_fn=_record_parse_failure_witness,
                parse_module_stage_function_index=_ParseModuleStage.FUNCTION_INDEX,
                parse_module_stage_symbol_table=_ParseModuleStage.SYMBOL_TABLE,
                parse_module_stage_class_index=_ParseModuleStage.CLASS_INDEX,
                accumulate_symbol_table_for_tree_fn=_accumulate_symbol_table_for_tree_runtime,
                accumulate_class_index_for_tree_fn=_accumulate_class_index_for_tree_runtime,
                timeout_exceeded_type=TimeoutExceeded,
                analysis_index_ctor=_analysis_index_ctor_runtime,
                progress_emit_min_interval_seconds=_progress_emit_min_interval_seconds(),
            ),
        ),
    )


_ANALYSIS_INDEX_MODULE_TREES_DEPS = _AnalysisIndexModuleTreesDeps(
    check_deadline_fn=check_deadline,
    parse_module_source_fn=_default_parse_module,
    parse_module_error_types=_PARSE_MODULE_ERROR_TYPES,
    record_parse_failure_witness_fn=_record_parse_failure_witness,
)

_analysis_index_module_trees = partial(
    _analysis_index_module_trees_impl,
    deps=_ANALYSIS_INDEX_MODULE_TREES_DEPS,
)

_ANALYSIS_INDEX_STAGE_CACHE_DEPS = _AnalysisIndexStageCacheDeps(
    check_deadline_fn=check_deadline,
    get_global_derivation_cache_fn=get_global_derivation_cache,
    analysis_index_module_trees_fn=_analysis_index_module_trees,
    get_stage_cache_bucket_fn=_get_stage_cache_bucket,
    path_dependency_payload_fn=_path_dependency_payload,
    analysis_index_stage_cache_op=_ANALYSIS_INDEX_STAGE_CACHE_OP,
)

_analysis_index_stage_cache = partial(
    _analysis_index_stage_cache_impl,
    deps=_ANALYSIS_INDEX_STAGE_CACHE_DEPS,
)


def _run_indexed_pass(
    paths,
    *,
    project_root,
    ignore_params,
    strictness,
    external_filter,
    transparent_decorators=None,
    parse_failure_witnesses=None,
    analysis_index=None,
    spec,
    build_index=_build_analysis_index,
):
    check_deadline()
    sink = _parse_failure_sink(parse_failure_witnesses)
    index = analysis_index
    if index is None:
        index = build_index(
            paths,
            project_root=project_root,
            ignore_params=ignore_params,
            strictness=strictness,
            external_filter=external_filter,
            transparent_decorators=transparent_decorators,
            parse_failure_witnesses=sink,
        )
    context = _IndexedPassContext(
        paths=paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
        parse_failure_witnesses=sink,
        analysis_index=index,
    )
    return spec.run(context)


__all__ = [
    "_ANALYSIS_INDEX_STAGE_CACHE_OP",
    "_analysis_index_module_trees",
    "_analysis_index_resolved_call_edges",
    "_analysis_index_resolved_call_edges_by_caller",
    "_analysis_index_stage_cache",
    "_analysis_index_transitive_callers",
    "_build_module_artifacts",
    "_analyze_file_internal",
    "_build_stage_cache_identity_spec",
    "_CacheSemanticContext",
    "_cache_identity_aliases",
    "_build_function_index_runtime",
    "_build_symbol_table_runtime",
    "_canonical_cache_identity",
    "_canonical_stage_cache_detail",
    "_canonical_stage_cache_identity",
    "_EMPTY_CACHE_SEMANTIC_CONTEXT",
    "_build_analysis_collection_resume_payload",
    "_build_analysis_index",
    "_build_call_graph",
    "_collect_transitive_callers",
    "_get_stage_cache_bucket",
    "_index_stage_cache_identity",
    "_function_index_module_artifact_spec_runtime",
    "_symbol_table_module_artifact_spec_runtime",
    "_iter_resolved_edge_param_events",
    "_load_analysis_collection_resume_payload",
    "_normalize_cache_config",
    "_parse_stage_cache_key",
    "_path_dependency_payload",
    "_profiling_v1_payload",
    "_projection_stage_cache_identity",
    "_progress_emit_min_interval_seconds",
    "_reduce_resolved_call_edges",
    "_resume_variant_for_identity",
    "_run_indexed_pass",
    "_iter_monotonic_paths",
    "_phase_work_progress",
    "_PhaseWorkProgress",
    "_phase_work_progress_owner",
    "_IndexedPassContext",
    "_IndexedPassSpec",
    "_ModuleArtifactSpec",
    "OptionalAnalysisIndex",
    "OptionalDecorators",
    "OptionalParseFailures",
    "OptionalProjectRoot",
    "_StageCacheIdentitySpec",
    "_sorted_text",
    "_stage_cache_key_aliases",
]
