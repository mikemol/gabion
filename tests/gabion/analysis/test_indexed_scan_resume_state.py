from __future__ import annotations

from pathlib import Path

import pytest

from gabion.analysis.dataflow_contracts import InvariantProposition
from gabion.analysis.indexed_scan import resume_state


# gabion:evidence E:function_site::indexed_scan/resume_state.py::gabion.analysis.indexed_scan.resume_state.deserialize_param_spans_for_resume
def test_deserialize_param_spans_skips_non_numeric_values() -> None:
    decoded = resume_state.deserialize_param_spans_for_resume(
        {
            "fn": {
                "ok": [1, 0, 1, 2],
                "bad": ["x", 0, 1, 2],
            }
        }
    )

    assert decoded == {"fn": {"ok": (1, 0, 1, 2)}}


# gabion:evidence E:function_site::indexed_scan/resume_state.py::gabion.analysis.indexed_scan.resume_state.build_analysis_collection_resume_payload
def test_build_payload_rejects_descending_in_progress_path_order() -> None:
    path_b = Path("b.py")
    path_a = Path("a.py")
    in_progress = {
        path_b: {"phase": "function_scan"},
        path_a: {"phase": "function_scan"},
    }

    with pytest.raises(RuntimeError, match="path-order"):
        resume_state.build_analysis_collection_resume_payload(
            groups_by_path={},
            param_spans_by_path={},
            bundle_sites_by_path={},
            invariant_propositions=[],
            completed_paths=set(),
            in_progress_scan_by_path=in_progress,
            format_version=2,
            never_fn=lambda _message, **_details: (_ for _ in ()).throw(RuntimeError("path-order")),
        )


# gabion:evidence E:function_site::indexed_scan/resume_state.py::gabion.analysis.indexed_scan.resume_state.load_analysis_collection_resume_payload
def test_load_payload_handles_missing_sections_and_empty_in_progress_payload() -> None:
    empty = resume_state.load_analysis_collection_resume_payload(
        payload={"format_version": 2},
        file_paths=[],
        include_invariant_propositions=False,
        format_version=2,
    )
    assert empty == ({}, {}, {}, [], set(), {}, None)

    payload = {
        "format_version": 2,
        "groups_by_path": {"a.py": {}},
        "param_spans_by_path": {"a.py": {}},
        "bundle_sites_by_path": {"a.py": {}},
        "completed_paths": ["a.py"],
        "invariant_propositions": [
            {
                "form": "eq",
                "terms": ["x", "y"],
                "scope": "scope",
                "source": "source",
            }
        ],
        "analysis_index_resume": {"k": "v"},
    }

    loaded = resume_state.load_analysis_collection_resume_payload(
        payload=payload,
        file_paths=[Path("a.py")],
        include_invariant_propositions=True,
        format_version=2,
        deserialize_invariants_for_resume_fn=lambda entries: [
            InvariantProposition(form="eq", terms=("x", "y"), scope="scope", source="source")
            for _ in entries
        ],
        mapping_payload_fn=lambda _value: None,
    )

    assert loaded[4] == {Path("a.py")}
    assert loaded[5] == {}
    assert loaded[6] == {"k": "v"}
    assert len(loaded[3]) == 1
