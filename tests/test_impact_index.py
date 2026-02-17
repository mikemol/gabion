from __future__ import annotations

import json
from pathlib import Path

from gabion.analysis.impact_index import build_impact_index, emit_impact_index


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_build_impact_index_emits_expected_node_kinds(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/gabion/cli.py",
        (
            'DATAFLOW_COMMAND = "gabion.dataflowAudit"\n'
            '@app.command("check")\n'
            "def check():\n"
            "    helper()\n\n\n"
            "def helper():\n"
            "    return 1\n"
        ),
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

    index = build_impact_index(root=tmp_path)
    assert index.graph is not None
    graph = index.graph

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

    artifact = emit_impact_index(root=tmp_path)
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
            assert {"source": source, "type": edge_type} in reverse_edges


def test_test_links_prefer_explicit_metadata(tmp_path: Path) -> None:
    _write(
        tmp_path / "tests" / "test_sample.py",
        """
@impact_target("gabion.mod.fn")
def test_example():
    pass
""".strip(),
    )
    _write(tmp_path / "src" / "gabion" / "mod.py", "def fn():\n    return 1\n")

    index = build_impact_index(root=tmp_path)

    assert ("tests/test_sample.py::test_example", "gabion.mod.fn", "explicit") in {
        (item.source, item.target, item.confidence) for item in index.links
    }


def test_test_links_infer_from_imports_when_metadata_missing(tmp_path: Path) -> None:
    _write(
        tmp_path / "tests" / "test_sample.py",
        """
from gabion.mod import fn

def test_example():
    fn()
""".strip(),
    )
    _write(tmp_path / "src" / "gabion" / "mod.py", "def fn():\n    return 1\n")

    index = build_impact_index(root=tmp_path)

    assert ("tests/test_sample.py::test_example", "gabion.mod.fn", "inferred") in {
        (item.source, item.target, item.confidence) for item in index.links
    }


def test_doc_links_read_doc_targets_and_fallback_to_mentions(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "gabion" / "mod.py", "def fn():\n    return 1\n")
    _write(
        tmp_path / "docs" / "a.md",
        """
---
doc_targets:
  - gabion.mod.fn
---
Notes.
""".strip(),
    )
    _write(tmp_path / "docs" / "b.md", "Touches `gabion.mod.fn` behavior.")

    index = build_impact_index(root=tmp_path)
    links = {(item.source, item.target, item.confidence) for item in index.links}

    assert ("docs/a.md", "gabion.mod.fn", "explicit") in links
    assert ("docs/b.md", "gabion.mod.fn", "inferred") in links


def test_doc_links_fallback_to_anchors_as_weak(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "gabion" / "mod.py", "def fn_name():\n    return 1\n")
    _write(
        tmp_path / "docs" / "a.md",
        """
See [details](#fn-name).
""".strip(),
    )

    index = build_impact_index(root=tmp_path)

    assert ("docs/a.md", "gabion.mod.fn_name", "weak") in {
        (item.source, item.target, item.confidence) for item in index.links
    }
