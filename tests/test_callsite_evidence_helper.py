from __future__ import annotations

from pathlib import Path

def _load():
    repo_root = Path(__file__).resolve().parents[1]
    from gabion.analysis.dataflow_audit import CallArgs, _callsite_evidence_for_bundle

    return CallArgs, _callsite_evidence_for_bundle

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._callsite_evidence_for_bundle::bundle
def test_callsite_evidence_skips_calls_without_span() -> None:
    CallArgs, _callsite_evidence_for_bundle = _load()
    call = CallArgs(
        callee="g",
        pos_map={"0": "a"},
        kw_map={},
        const_pos={},
        const_kw={},
        non_const_pos=set(),
        non_const_kw=set(),
        star_pos=[],
        star_kw=[],
        is_test=False,
        span=None,
    )
    assert _callsite_evidence_for_bundle([call], {"a", "b"}) == []

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._callsite_evidence_for_bundle::bundle
def test_callsite_evidence_records_star_args_and_star_kwargs() -> None:
    CallArgs, _callsite_evidence_for_bundle = _load()
    call = CallArgs(
        callee="g",
        pos_map={},
        kw_map={},
        const_pos={},
        const_kw={},
        non_const_pos=set(),
        non_const_kw=set(),
        star_pos=[(0, "a")],
        star_kw=["b"],
        is_test=False,
        span=(0, 0, 0, 1),
    )
    evidence = _callsite_evidence_for_bundle([call], {"a", "b"})
    assert evidence
    assert evidence[0]["slots"] == ["arg[0]*", "kw[**]"]

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._callsite_evidence_for_bundle::bundle
def test_callsite_evidence_dedupes_duplicate_calls() -> None:
    CallArgs, _callsite_evidence_for_bundle = _load()
    call = CallArgs(
        callee="g",
        pos_map={"0": "a"},
        kw_map={},
        const_pos={},
        const_kw={},
        non_const_pos=set(),
        non_const_kw=set(),
        star_pos=[],
        star_kw=[],
        is_test=False,
        span=(1, 2, 1, 3),
    )
    evidence = _callsite_evidence_for_bundle([call, call], {"a", "b"})
    assert len(evidence) == 1



# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._callsite_evidence_for_bundle::bundle
def test_callsite_evidence_includes_callable_context() -> None:
    CallArgs, _callsite_evidence_for_bundle = _load()
    evidence = _callsite_evidence_for_bundle(
        [
            CallArgs(
                callee="g",
                pos_map={"0": "a"},
                kw_map={},
                const_pos={},
                const_kw={},
                non_const_pos=set(),
                non_const_kw=set(),
                star_pos=[],
                star_kw=[],
                is_test=False,
                span=(1, 0, 1, 3),
                callable_kind="lambda",
                callable_source="inline",
            ),
            CallArgs(
                callee="make()",
                pos_map={"0": "a"},
                kw_map={},
                const_pos={},
                const_kw={},
                non_const_pos=set(),
                non_const_kw=set(),
                star_pos=[],
                star_kw=[],
                is_test=False,
                span=(2, 0, 2, 7),
                callable_kind="closure",
                callable_source="call_result",
            ),
        ],
        {"a", "b"},
    )
    contexts = {(row["callable_kind"], row["callable_source"]) for row in evidence}
    assert ("lambda", "inline") in contexts
    assert ("closure", "call_result") in contexts
