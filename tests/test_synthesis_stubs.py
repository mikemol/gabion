from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis.dataflow_audit import render_protocol_stubs

    return render_protocol_stubs


def test_render_protocol_stubs_emits_dataclass() -> None:
    render_protocol_stubs = _load()
    plan = {
        "protocols": [
            {
                "name": "ExampleBundle",
                "fields": [
                    {"name": "ctx", "type_hint": "Context"},
                    {"name": "config", "type_hint": None},
                ],
            }
        ]
    }
    stub = render_protocol_stubs(plan)
    assert "class ExampleBundle" in stub
    assert "ctx: Context" in stub
    assert "config: Any" in stub
