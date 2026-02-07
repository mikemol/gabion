from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis.dataflow_audit import render_protocol_stubs

    return render_protocol_stubs


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind
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
    assert "from typing import Any" in stub
    assert "from dataclasses import dataclass" in stub
    assert "class TODO_Name_Me" in stub
    assert "ctx: Context" in stub
    assert "config: Any" in stub


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind
def test_render_protocol_stubs_emits_protocol() -> None:
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
    stub = render_protocol_stubs(plan, kind="protocol")
    assert "from dataclasses import dataclass" not in stub
    assert "from typing import Any, Protocol" in stub
    assert "class TODO_Name_Me(Protocol)" in stub
    assert "ctx: Context" in stub
    assert "config: Any" in stub
