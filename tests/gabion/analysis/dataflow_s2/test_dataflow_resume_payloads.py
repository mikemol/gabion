from __future__ import annotations

from pathlib import Path

from gabion.analysis.dataflow.engine import dataflow_facade as da
from gabion.analysis.dataflow.engine.dataflow_contracts import CallArgs, ClassInfo, FunctionInfo, InvariantProposition, SymbolTable


def _fn(path: Path) -> FunctionInfo:
    return FunctionInfo(
        name="worker",
        qual="pkg.mod.worker",
        path=path,
        params=["x", "y"],
        annots={"x": "int", "y": None},
        calls=[
            CallArgs(
                callee="callee",
                pos_map={"0": "x"},
                kw_map={"y": "y"},
                const_pos={},
                const_kw={},
                non_const_pos=set(),
                non_const_kw=set(),
                star_pos=[(1, "x")],
                star_kw=["y"],
                is_test=False,
                span=(1, 0, 1, 4),
                callable_kind="function",
                callable_source="symbol",
            )
        ],
        unused_params={"y"},
        unknown_key_carriers={"x"},
        defaults={"y"},
        transparent=True,
        class_name="Worker",
        scope=("pkg", "mod"),
        lexical_scope=("pkg",),
        decision_params={"x"},
        decision_surface_reasons={"x": {"guard"}},
        value_decision_params={"y"},
        value_decision_reasons={"value_guard"},
        positional_params=("x",),
        kwonly_params=("y",),
        vararg="args",
        kwarg="kwargs",
        param_spans={"x": (1, 0, 1, 1), "y": (1, 3, 1, 4)},
        function_span=(1, 0, 2, 0),
    )


# gabion:evidence E:function_site::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._serialize_analysis_index_resume_payload E:function_site::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._load_analysis_index_resume_payload E:function_site::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._analysis_index_resume_variants
def test_analysis_index_resume_payload_round_trip_and_variant_selection(tmp_path: Path) -> None:
    path = tmp_path / "pkg" / "mod.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("def worker(x, y):\n    return x + y\n", encoding="utf-8")

    symbol_table = SymbolTable()
    symbol_table.imports[("pkg.mod", "Base")] = "pkg.base.Base"
    symbol_table.internal_roots.add("pkg")
    symbol_table.star_imports["pkg.mod"] = {"pkg.base"}
    symbol_table.module_exports["pkg.base"] = {"Base"}
    symbol_table.module_export_map["pkg.base"] = {"Base": "pkg.base.Base"}

    by_qual = {"pkg.mod.worker": _fn(path)}
    class_index = {
        "pkg.base.Base": ClassInfo(
            qual="pkg.base.Base",
            module="pkg.base",
            bases=[],
            methods={"run"},
        )
    }

    payload_a = da._serialize_analysis_index_resume_payload(
        hydrated_paths={path},
        by_qual=by_qual,
        symbol_table=symbol_table,
        class_index=class_index,
        index_cache_identity="a" * 40,
        projection_cache_identity="b" * 40,
        profiling_v1={"stage": 1},
    )
    payload_b = da._serialize_analysis_index_resume_payload(
        hydrated_paths={path},
        by_qual=by_qual,
        symbol_table=symbol_table,
        class_index=class_index,
        index_cache_identity="c" * 40,
        projection_cache_identity="d" * 40,
        profiling_v1={"stage": 2},
        previous_payload=payload_a,
    )

    loaded_current = da._load_analysis_index_resume_payload(
        payload=payload_b,
        file_paths=[path],
        expected_index_cache_identity="c" * 40,
        expected_projection_cache_identity="d" * 40,
    )
    assert loaded_current[0] == {path}
    assert "pkg.mod.worker" in loaded_current[1]

    loaded_variant = da._load_analysis_index_resume_payload(
        payload=payload_b,
        file_paths=[path],
        expected_index_cache_identity="a" * 40,
        expected_projection_cache_identity="b" * 40,
    )
    assert loaded_variant[0] == {path}
    assert "pkg.mod.worker" in loaded_variant[1]

    loaded_mismatch = da._load_analysis_index_resume_payload(
        payload=payload_b,
        file_paths=[path],
        expected_index_cache_identity="c" * 40,
        expected_projection_cache_identity="e" * 40,
    )
    assert loaded_mismatch[0] == set()
    assert loaded_mismatch[1] == {}
    assert loaded_mismatch[3] == {}
    assert isinstance(loaded_mismatch[2], da.SymbolTable)
    assert loaded_mismatch[2].imports == {}


# gabion:evidence E:function_site::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._deserialize_function_info_for_resume
def test_deserialize_function_info_for_resume_filters_invalid_entries(tmp_path: Path) -> None:
    source = tmp_path / "mod.py"
    source.write_text("def f():\n    return 1\n", encoding="utf-8")
    allowed_paths = {"mod.py": source}

    payload = {
        "name": "f",
        "qual": "pkg.mod.f",
        "path": "mod.py",
        "params": ["x", "y"],
        "annots": {"x": "int", "bad": 1},
        "calls": [
            {"callee": "g", "pos_map": {"0": "x"}},
            {"callee": 1},
        ],
        "unused_params": ["y"],
        "unknown_key_carriers": ["x"],
        "defaults": ["y"],
        "transparent": False,
        "class_name": 1,
        "scope": ["pkg", "mod"],
        "lexical_scope": ["pkg"],
        "decision_params": ["x"],
        "decision_surface_reasons": {"x": ["guard"], "bad": 1},
        "value_decision_params": ["x"],
        "value_decision_reasons": ["value_guard"],
        "positional_params": ["x"],
        "kwonly_params": ["y"],
        "vararg": "args",
        "kwarg": "kwargs",
        "param_spans": {"x": [1, 0, 1, 1], "bad": [1, 2]},
        "function_span": [1, 0, 1, 2],
    }

    info = da._deserialize_function_info_for_resume(payload, allowed_paths=allowed_paths)
    assert info is not None
    assert info.annots == {"x": "int"}
    assert len(info.calls) == 1
    assert info.class_name is None
    assert info.decision_surface_reasons == {"x": {"guard"}}
    assert info.param_spans == {"x": (1, 0, 1, 1)}

    invalid = da._deserialize_function_info_for_resume(
        {"name": "x", "qual": "x", "path": "missing", "params": []},
        allowed_paths=allowed_paths,
    )
    assert invalid is None


# gabion:evidence E:function_site::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._deserialize_symbol_table_for_resume
def test_deserialize_symbol_table_for_resume_filters_shape_errors() -> None:
    payload = {
        "external_filter": False,
        "imports": [["pkg.mod", "Alias", "pkg.base.Alias"], ["bad", "shape"], 1],
        "internal_roots": ["pkg", 1],
        "star_imports": {"pkg.mod": ["pkg.base", 1]},
        "module_exports": {"pkg.base": ["Alias", 1]},
        "module_export_map": {"pkg.base": {"Alias": "pkg.base.Alias", "Bad": 1}},
    }
    table = da._deserialize_symbol_table_for_resume(payload)
    assert table.external_filter is False
    assert table.imports == {("pkg.mod", "Alias"): "pkg.base.Alias"}
    assert table.internal_roots == {"pkg"}
    assert table.star_imports["pkg.mod"] == {"pkg.base"}
    assert table.module_exports["pkg.base"] == {"Alias"}
    assert table.module_export_map["pkg.base"] == {"Alias": "pkg.base.Alias"}


# gabion:evidence E:function_site::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._serialize_file_scan_resume_state E:function_site::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._load_file_scan_resume_state
def test_load_file_scan_resume_state_round_trip_and_invalid_payload() -> None:
    fn_key = "pkg.mod.f"
    fn_use = {
        fn_key: {
            "x": da.ParamUse(
                direct_forward={("pkg.mod.g", "arg[0]")},
                non_forward=False,
                current_aliases={"x"},
                forward_sites={("pkg.mod.g", "arg[0]"): {(1, 0, 1, 1)}},
                unknown_key_carrier=True,
                unknown_key_sites={(2, 0, 2, 1)},
            )
        }
    }
    fn_calls = {
        fn_key: [
            CallArgs(
                callee="pkg.mod.g",
                pos_map={"0": "x"},
                kw_map={},
                const_pos={},
                const_kw={},
                non_const_pos=set(),
                non_const_kw=set(),
                star_pos=[],
                star_kw=[],
                is_test=False,
                span=(1, 0, 1, 1),
            )
        ]
    }
    payload = da._serialize_file_scan_resume_state(
        fn_use=fn_use,
        fn_calls=fn_calls,
        fn_param_orders={fn_key: ["x"]},
        fn_param_spans={fn_key: {"x": (1, 0, 1, 1)}},
        fn_names={fn_key: "f"},
        fn_lexical_scopes={fn_key: ("pkg", "mod")},
        fn_class_names={fn_key: None},
        opaque_callees={fn_key},
    )

    loaded = da._load_file_scan_resume_state(payload=payload, valid_fn_keys={fn_key})
    assert fn_key in loaded[0]
    assert fn_key in loaded[1]
    assert loaded[2][fn_key] == ["x"]
    assert loaded[4][fn_key] == "f"
    assert loaded[7] == {fn_key}

    empty = da._load_file_scan_resume_state(payload={"phase": "wrong"}, valid_fn_keys={fn_key})
    assert empty == ({}, {}, {}, {}, {}, {}, {}, set())


# gabion:evidence E:function_site::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._build_analysis_collection_resume_payload E:function_site::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._load_analysis_collection_resume_payload E:function_site::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._resolve_synth_registry_path
def test_analysis_collection_resume_round_trip_and_synth_registry_resolution(tmp_path: Path) -> None:
    complete_path = (tmp_path / "complete.py").resolve()
    pending_path = (tmp_path / "pending.py").resolve()
    complete_path.write_text("def a():\n    return 1\n", encoding="utf-8")
    pending_path.write_text("def b():\n    return 2\n", encoding="utf-8")

    payload = da._build_analysis_collection_resume_payload(
        groups_by_path={complete_path: {"f": [{"x"}]}},
        param_spans_by_path={complete_path: {"f": {"x": (1, 0, 1, 1)}}},
        bundle_sites_by_path={complete_path: {"f": [[{"path": "complete.py"}]]}},
        invariant_propositions=[
            InvariantProposition(
                form="equality",
                terms=("x", "y"),
                scope="pkg.mod.f",
                source="test",
                invariant_id="inv-1",
            )
        ],
        completed_paths={complete_path},
        in_progress_scan_by_path={pending_path: {"phase": "function_scan"}},
        analysis_index_resume={"k": "v"},
    )

    loaded = da._load_analysis_collection_resume_payload(
        payload=payload,
        file_paths=[complete_path, pending_path],
        include_invariant_propositions=True,
    )
    assert complete_path in loaded[4]
    assert pending_path in loaded[5]
    assert loaded[6] == {"k": "v"}
    assert loaded[3] and loaded[3][0].invariant_id == "inv-1"

    empty = da._load_analysis_collection_resume_payload(
        payload={},
        file_paths=[complete_path, pending_path],
        include_invariant_propositions=False,
    )
    assert empty == ({}, {}, {}, [], set(), {}, None)

    assert da._resolve_synth_registry_path(None, tmp_path) is None
    assert da._resolve_synth_registry_path("  ", tmp_path) is None
    resolved_relative = da._resolve_synth_registry_path("fingerprint.json", tmp_path)
    assert resolved_relative == (tmp_path / "fingerprint.json").resolve()

    latest_root = tmp_path / "artifacts"
    latest_root.mkdir(parents=True, exist_ok=True)
    marker = latest_root / "LATEST.txt"
    marker.write_text("2026-03-02", encoding="utf-8")
    expected = (latest_root / "2026-03-02" / "fingerprint_synth.json")
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.write_text("{}", encoding="utf-8")
    # Canonical marker path contract resolves through LATEST.txt indirection.
    resolved_latest = da._resolve_synth_registry_path(
        "artifacts/LATEST/fingerprint_synth.json",
        tmp_path,
    )
    assert resolved_latest == expected.resolve()
