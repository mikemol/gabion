from pathlib import Path

import pytest

from gabion.analysis import dataflow_audit as da
from gabion.ingest.python_ingest import _default_deadline, ingest_python_file, iter_python_paths


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


def test_iter_python_paths_uses_default_deadline_callable(tmp_path: Path) -> None:
    source = tmp_path / "mod.py"
    source.write_text("def f(x):\n    return x\n", encoding="utf-8")
    config = da.AuditConfig(project_root=tmp_path)
    assert _default_deadline() is None
    paths = iter_python_paths([str(source)], config=config, sort_once=da.sort_once)
    assert paths == [source]


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
