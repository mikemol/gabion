from pathlib import Path
from types import SimpleNamespace

import pytest

from gabion.analysis.core.visitors import ParentAnnotator
from gabion.analysis.dataflow.engine import dataflow_analysis_index as index_owner
from gabion.analysis.dataflow.engine import dataflow_function_index_decision_support as decision_support
from gabion.analysis.dataflow.engine import dataflow_function_index_helpers as function_index_helpers
from gabion.analysis.dataflow.engine import dataflow_function_semantics as function_semantics
from gabion.analysis.dataflow.engine import dataflow_ingest_helpers as ingest_helpers
from gabion.analysis.dataflow.engine import dataflow_ingested_analysis_support as ingested_support
from gabion.analysis.dataflow.engine import dataflow_lambda_runtime_support as lambda_runtime
from gabion.analysis.dataflow.engine import dataflow_local_class_hierarchy as class_hierarchy
from gabion.analysis.dataflow.engine import dataflow_post_phase_analyses as post_phase
from gabion.analysis.dataflow.engine import dataflow_resume_serialization as resume_serialization
from gabion.analysis.dataflow.engine.dataflow_contracts import AuditConfig
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.ingest.python_ingest import _default_deadline, ingest_python_file, iter_python_paths
from gabion.order_contract import sort_once

da = SimpleNamespace(
    AuditConfig=AuditConfig,
    ParentAnnotator=ParentAnnotator,
    _FILE_SCAN_PROGRESS_EMIT_INTERVAL=index_owner._FILE_SCAN_PROGRESS_EMIT_INTERVAL,
    _PROGRESS_EMIT_MIN_INTERVAL_SECONDS=index_owner._PROGRESS_EMIT_MIN_INTERVAL_SECONDS,
    _analyze_file_internal=index_owner._analyze_file_internal,
    _analyze_function=function_semantics._analyze_function,
    _collect_functions=ingest_helpers._collect_functions,
    _collect_local_class_bases=class_hierarchy._collect_local_class_bases,
    _collect_return_aliases=function_semantics._collect_return_aliases,
    _decorators_transparent=decision_support._decorators_transparent,
    _enclosing_class=function_index_helpers._enclosing_class,
    _enclosing_function_scopes=function_index_helpers._enclosing_function_scopes,
    _enclosing_scopes=function_index_helpers._enclosing_scopes,
    _function_key=lambda_runtime._function_key,
    _is_test_path=function_index_helpers._is_test_path,
    _load_file_scan_resume_state=resume_serialization._load_file_scan_resume_state,
    _param_names=function_index_helpers._param_names,
    _param_spans=function_index_helpers._param_spans,
    _parse_module_source=post_phase._parse_module_source,
    _profiling_v1_payload=index_owner._profiling_v1_payload,
    _resolve_local_method_in_hierarchy=class_hierarchy._resolve_local_method_in_hierarchy,
    _serialize_file_scan_resume_state=resume_serialization._serialize_file_scan_resume_state,
    analyze_ingested_file=ingested_support.analyze_ingested_file,
    check_deadline=check_deadline,
    sort_once=sort_once,
)


# gabion:behavior primary=desired
def test_iter_python_paths_expands_and_filters(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    include = pkg / "mod.py"
    include.write_text("def f(x):\n    return x\n", encoding="utf-8")
    (pkg / "ignore.txt").write_text("x", encoding="utf-8")
    excluded = tmp_path / ".venv"
    excluded.mkdir()
    (excluded / "skip.py").write_text("def g():\n    return 1\n", encoding="utf-8")

    config = da.AuditConfig(project_root=tmp_path, exclude_dirs={".venv"})
    paths = iter_python_paths([str(tmp_path)], config=config, check_deadline=da.check_deadline, sort_once=da.sort_once)
    assert include in paths
    assert all(".venv" not in str(path) for path in paths)


# gabion:behavior primary=desired
def test_iter_python_paths_uses_default_deadline_callable(tmp_path: Path) -> None:
    source = tmp_path / "mod.py"
    source.write_text("def f(x):\n    return x\n", encoding="utf-8")
    config = da.AuditConfig(project_root=tmp_path)
    assert _default_deadline() is None
    paths = iter_python_paths([str(source)], config=config, sort_once=da.sort_once)
    assert paths == [source]


# gabion:behavior primary=desired
def test_ingest_contract_adapts_to_analysis(tmp_path: Path) -> None:
    source = tmp_path / "mod.py"
    source.write_text(
        "def child(a, b):\n"
        "    return a + b\n\n"
        "def parent(x, y):\n"
        "    return child(x, y)\n",
        encoding="utf-8",
    )
    config = da.AuditConfig(project_root=tmp_path)

    carrier = ingest_python_file(
        source,
        config=config,
        recursive=True,
        parse_module=da._parse_module_source,
        collect_functions=da._collect_functions,
        collect_return_aliases=da._collect_return_aliases,
        load_resume_state=da._load_file_scan_resume_state,
        serialize_resume_state=da._serialize_file_scan_resume_state,
        profiling_payload=da._profiling_v1_payload,
        analyze_function=da._analyze_function,
        enclosing_class=da._enclosing_class,
        enclosing_scopes=da._enclosing_scopes,
        enclosing_function_scopes=da._enclosing_function_scopes,
        function_key=da._function_key,
        decorators_transparent=da._decorators_transparent,
        param_names=da._param_names,
        param_spans=da._param_spans,
        collect_local_class_bases=da._collect_local_class_bases,
        resolve_local_method_in_hierarchy=da._resolve_local_method_in_hierarchy,
        is_test_path=da._is_test_path,
        check_deadline=da.check_deadline,
        parent_annotator_factory=da.ParentAnnotator,
        progress_emit_interval=da._FILE_SCAN_PROGRESS_EMIT_INTERVAL,
        progress_min_interval_seconds=da._PROGRESS_EMIT_MIN_INTERVAL_SECONDS,
    )
    assert carrier.functions
    function = carrier.functions[0]
    assert isinstance(function.params, tuple)
    assert function.callsites is not None
    assert function.param_spans is not None
    assert function.decision_evidence == {}

    direct = da._analyze_file_internal(source, recursive=True, config=config)
    adapted = da.analyze_ingested_file(carrier, recursive=True, config=config)
    assert adapted == direct


# gabion:behavior primary=verboten facets=exception
def test_ingest_python_file_emits_profile_on_exception(tmp_path: Path) -> None:
    source = tmp_path / "mod.py"
    source.write_text("def parent(x, y):\n    return x + y\n", encoding="utf-8")
    config = da.AuditConfig(project_root=tmp_path)
    profiles: list[dict[str, object]] = []
    with pytest.raises(RuntimeError):
        ingest_python_file(
            source,
            config=config,
            recursive=True,
            parse_module=da._parse_module_source,
            collect_functions=da._collect_functions,
            collect_return_aliases=da._collect_return_aliases,
            load_resume_state=da._load_file_scan_resume_state,
            serialize_resume_state=da._serialize_file_scan_resume_state,
            profiling_payload=da._profiling_v1_payload,
            analyze_function=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("boom")
            ),
            enclosing_class=da._enclosing_class,
            enclosing_scopes=da._enclosing_scopes,
            enclosing_function_scopes=da._enclosing_function_scopes,
            function_key=da._function_key,
            decorators_transparent=da._decorators_transparent,
            param_names=da._param_names,
            param_spans=da._param_spans,
            collect_local_class_bases=da._collect_local_class_bases,
            resolve_local_method_in_hierarchy=da._resolve_local_method_in_hierarchy,
            is_test_path=da._is_test_path,
            check_deadline=da.check_deadline,
            parent_annotator_factory=da.ParentAnnotator,
            progress_emit_interval=da._FILE_SCAN_PROGRESS_EMIT_INTERVAL,
            progress_min_interval_seconds=da._PROGRESS_EMIT_MIN_INTERVAL_SECONDS,
            on_profile=profiles.append,
        )
    assert profiles
