from __future__ import annotations

import json
from pathlib import Path

from gabion.analysis.impact_index import build_impact_index, emit_impact_index


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_impact_index_emits_expected_node_kinds(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/gabion/cli.py",
        'DATAFLOW_COMMAND = "gabion.dataflowAudit"\n@app.command("check")\ndef check():\n    helper()\n\n\ndef helper():\n    return 1\n',
    )
    _write(
        tmp_path / "src/gabion/server.py",
        'REFACTOR_COMMAND = "gabion.refactorProtocol"\n',
    )
    _write(
        tmp_path / "tests/test_cli.py",
        "from gabion.cli import helper\n\n\ndef test_helper():\n    assert helper() == 1\n",
    )
    _write(
        tmp_path / "README.md",
        '<a id="intro"></a>\nUse `helper` to perform checks.\n',
    )

    payload = build_impact_index(tmp_path)
    graph = payload["graph"]

    assert "span" in graph["nodes"]
    assert "symbol" in graph["nodes"]
    assert "callsite" in graph["nodes"]
    assert "test" in graph["nodes"]
    assert "doc_section" in graph["nodes"]
    assert "command" in graph["nodes"]
    assert "report_section" in graph["nodes"]


def test_emit_impact_index_writes_artifact_and_reverse_adjacency(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/gabion/cli.py",
        'DATAFLOW_COMMAND = "gabion.dataflowAudit"\n\n\ndef helper():\n    return 1\n',
    )
    _write(
        tmp_path / "src/gabion/server.py",
        'REFACTOR_COMMAND = "gabion.refactorProtocol"\n',
    )

    artifact = emit_impact_index(tmp_path)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    forward = payload["graph"]["adjacency"]["forward"]
    reverse = payload["graph"]["adjacency"]["reverse"]

    assert artifact == tmp_path / "artifacts/audit_reports/impact_index.json"
    assert artifact.exists()

    for source, edges in forward.items():
        for edge in edges:
            target = edge["target"]
            edge_type = edge["type"]
            reverse_edges = reverse.get(target, [])
            assert {
                "source": source,
                "type": edge_type,
            } in reverse_edges
