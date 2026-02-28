from __future__ import annotations

import json
from pathlib import Path

from gabion.analysis import aspf_stream


def test_iter_jsonl_skips_blank_lines() -> None:
    path = Path("artifacts/out/test_aspf_stream_blank.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "",
                json.dumps({"kind": "one_cell", "payload": {"id": "a"}}),
                "",
            ]
        ),
        encoding="utf-8",
    )
    try:
        rows = list(aspf_stream._iter_jsonl(path=path))
    finally:
        path.unlink(missing_ok=True)
    assert rows == [{"kind": "one_cell", "payload": {"id": "a"}}]
