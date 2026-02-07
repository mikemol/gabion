from __future__ import annotations

from pathlib import Path

from gabion.analysis import dataflow_audit as da


def _make_function(path: Path, qual: str) -> da.FunctionInfo:
    return da.FunctionInfo(
        name=qual.split(".")[-1],
        qual=qual,
        path=path,
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
    )


# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_call_ambiguities
def test_collect_call_ambiguities_skips_test_calls(tmp_path: Path) -> None:
    source = tmp_path / "mod.py"
    source.write_text(
        "\n".join(
            [
                "def helper(x):",
                "    return x",
                "",
                "def test_call():",
                "    helper(1)",
            ]
        )
        + "\n"
    )
    ambiguities = da._collect_call_ambiguities(
        [source],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=False,
    )
    assert ambiguities == []


# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_call_ambiguities
def test_dedupe_emit_and_lint_call_ambiguities(tmp_path: Path) -> None:
    caller = _make_function(tmp_path / "mod.py", "mod.caller")
    candidate = _make_function(tmp_path / "mod.py", "mod.target")
    entry = da.CallAmbiguity(
        kind="local_resolution_ambiguous",
        caller=caller,
        call=None,
        callee_key="target",
        candidates=(candidate,),
        phase="resolve_local_callee",
    )
    deduped = da._dedupe_call_ambiguities([entry, entry])
    assert len(deduped) == 1

    emitted = da._emit_call_ambiguities(
        deduped,
        project_root=tmp_path,
        forest=None,
    )
    assert emitted[0]["candidate_count"] == 1

    lint_lines = da._lint_lines_from_call_ambiguities(
        [
            "bad",
            {"kind": "x", "site": "bad"},
            {"kind": "x", "site": {"path": "", "span": [1, 2, 3, 4]}},
            {
                "kind": "x",
                "site": {"path": "mod.py", "span": ["x", "y", 0, 0]},
                "candidate_count": "bad",
            },
        ]
    )
    assert any("GABION_AMBIGUITY" in line for line in lint_lines)

    summary = da._summarize_call_ambiguities(
        [
            "bad",
            {"kind": "x", "site": "bad"},
            emitted[0],
            dict(emitted[0]),
        ],
        max_entries=1,
    )
    assert any("Counts by witness kind" in line for line in summary)
    assert any("... " in line for line in summary)


# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit.render_report
def test_render_report_includes_ambiguities() -> None:
    report, _ = da.render_report(
        {},
        0,
        ambiguity_witnesses=[
            {
                "kind": "local_resolution_ambiguous",
                "site": {"path": "mod.py", "function": "f", "span": [1, 2, 3, 4]},
                "candidate_count": 2,
            }
        ],
    )
    assert "Ambiguities:" in report
