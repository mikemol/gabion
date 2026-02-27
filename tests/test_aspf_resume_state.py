from __future__ import annotations

import json
from pathlib import Path

from gabion.analysis import aspf_resume_state


def _state_payload(*, projection_value: int, seq: int) -> dict[str, object]:
    return {
        "resume_projection": {"projection_value": projection_value},
        "delta_ledger": {
            "format_version": 1,
            "trace_id": "aspf-trace:test",
            "records": [
                {
                    "seq": seq,
                    "event_kind": "resume",
                    "phase": "load",
                    "analysis_state": None,
                    "mutation_target": "semantic_surfaces.groups_by_path",
                    "mutation_value": {"v": projection_value},
                    "one_cell_ref": None,
                }
            ],
        },
    }


def test_iter_delta_records_streams_state_and_jsonl_inputs(tmp_path: Path) -> None:
    state_path = tmp_path / "state.snapshot.json"
    state_path.write_text(json.dumps(_state_payload(projection_value=3, seq=1)), encoding="utf-8")

    jsonl_path = tmp_path / "delta.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps({"seq": 2, "mutation_target": "x", "mutation_value": {"v": 5}}),
                "",
            ]
        ),
        encoding="utf-8",
    )

    records = list(
        aspf_resume_state.iter_delta_records(
            state_paths=(state_path,),
            jsonl_paths=(jsonl_path,),
        )
    )

    assert [record["seq"] for record in records] == [1, 2]
    assert records[0]["mutation_value"] == {"v": 3}
    assert records[1]["mutation_value"] == {"v": 5}


def test_append_delta_jsonl_record_appends_single_line_payload(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "out" / "delta.jsonl"

    aspf_resume_state.append_delta_jsonl_record(
        path=jsonl_path,
        record={"seq": 1, "mutation_target": "a", "mutation_value": {"k": "v"}},
    )
    aspf_resume_state.append_delta_jsonl_record(
        path=jsonl_path,
        record={"seq": 2, "mutation_target": "b", "mutation_value": {"k": "w"}},
    )

    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["seq"] == 1
    assert json.loads(lines[1])["seq"] == 2


def test_load_resume_projection_compatibility_wrapper_uses_streaming_internals(
    tmp_path: Path,
) -> None:
    first_state = tmp_path / "0001.snapshot.json"
    second_state = tmp_path / "0002.snapshot.json"
    first_state.write_text(json.dumps(_state_payload(projection_value=1, seq=1)), encoding="utf-8")
    second_state.write_text(json.dumps(_state_payload(projection_value=2, seq=2)), encoding="utf-8")

    projection, records = aspf_resume_state.load_resume_projection_from_state_files(
        state_paths=(first_state, second_state)
    )

    assert projection == {"projection_value": 2}
    assert [record["seq"] for record in records] == [1, 2]


def test_iter_delta_records_from_jsonl_paths_skips_blank_lines(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "delta.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps({"seq": 1, "mutation_target": "a", "mutation_value": {"v": 1}}),
                "",
                json.dumps({"seq": 2, "mutation_target": "b", "mutation_value": {"v": 2}}),
            ]
        ),
        encoding="utf-8",
    )

    records = list(aspf_resume_state.iter_delta_records_from_jsonl_paths(jsonl_paths=(jsonl_path,)))
    assert [record["seq"] for record in records] == [1, 2]


def test_iter_resume_mutations_prefers_jsonl_sidecar_over_snapshot_ledger(tmp_path: Path) -> None:
    state_path = tmp_path / "0001_stage.snapshot.json"
    state_path.write_text(
        json.dumps(_state_payload(projection_value=9, seq=99)),
        encoding="utf-8",
    )
    jsonl_path = tmp_path / "0001_stage.delta.jsonl"
    jsonl_path.write_text(
        json.dumps({"seq": 1, "mutation_target": "collection_resume.done", "mutation_value": 1}) + "\n",
        encoding="utf-8",
    )

    records = list(aspf_resume_state.iter_resume_mutations(state_paths=(state_path,)))

    assert [record["seq"] for record in records] == [1]


def test_fold_resume_mutations_applies_projection_and_tracks_tail() -> None:
    mutations = (
        {"mutation_target": "collection_resume.a", "mutation_value": 1},
        {"mutation_target": "collection_resume.b", "mutation_value": 2},
        {"mutation_target": "collection_resume.c", "mutation_value": 3},
    )

    projection, count, tail = aspf_resume_state.fold_resume_mutations(
        snapshot={"analysis_state": "timed_out"},
        mutations=mutations,
        tail_limit=2,
    )

    assert count == 3
    assert projection["analysis_state"] == "timed_out"
    assert projection["collection_resume"] == {"a": 1, "b": 2, "c": 3}
    assert [record["mutation_target"] for record in tail] == [
        "collection_resume.b",
        "collection_resume.c",
    ]
