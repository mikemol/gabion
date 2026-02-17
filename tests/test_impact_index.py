from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import runpy
from types import SimpleNamespace

from gabion.analysis import impact_index as ii
from gabion.analysis.impact_index import (
    ImpactIndexGraph,
    ImpactLink,
    build_impact_index,
    emit_impact_index,
)


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


def test_links_from_test_returns_empty_for_read_and_parse_failures(tmp_path: Path) -> None:
    missing = ii._links_from_test(path=tmp_path / "missing.py", root=tmp_path)
    assert missing == []

    broken = tmp_path / "tests" / "test_broken.py"
    _write(broken, "def test_broken(:\n    pass\n")
    unparsable = ii._links_from_test(path=broken, root=tmp_path)
    assert unparsable == []


def test_links_from_test_collects_test_class_async_methods(tmp_path: Path) -> None:
    path = tmp_path / "tests" / "test_suite.py"
    _write(
        path,
        (
            "class TestSuite:\n"
            "    VALUE = 1\n"
            '    @impact_target("gabion.mod.fn")\n'
            "    async def test_async_case(self):\n"
            "        return None\n"
        ),
    )
    _write(tmp_path / "src" / "gabion" / "mod.py", "def fn():\n    return 1\n")

    links = ii._links_from_test(path=path, root=tmp_path)
    assert [
        (item.source, item.target, item.confidence)
        for item in links
    ] == [("tests/test_suite.py::test_async_case", "gabion.mod.fn", "explicit")]


def test_links_from_doc_returns_empty_when_unreadable(tmp_path: Path) -> None:
    assert ii._links_from_doc(
        path=tmp_path / "docs" / "missing.md",
        root=tmp_path,
        symbols={"gabion.mod.fn"},
    ) == []


def test_build_graph_payload_handles_async_and_unparseable_files(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "gabion" / "cli.py",
        (
            'DATAFLOW_COMMAND = "gabion.dataflowAudit"\n'
            "async def run_async(v):\n"
            "    return helper(v)\n\n"
            "def helper(v):\n"
            "    return v\n"
        ),
    )
    _write(
        tmp_path / "src" / "gabion" / "server.py",
        'REFACTOR_COMMAND = "gabion.refactorProtocol"\n',
    )
    _write(tmp_path / "src" / "gabion" / "broken.py", "def broken(:\n    pass\n")

    graph = ii._build_graph_payload(tmp_path)
    symbols = {entry["qualname"] for entry in graph["nodes"]["symbol"]}
    assert "run_async" in symbols
    assert "helper" in symbols


def test_module_and_parse_helpers_cover_error_edges(tmp_path: Path) -> None:
    assert ii._module_from_path(Path("src/gabion/sample.py")) == "gabion.sample"
    assert ii._module_from_path(Path("gabion/sample.py")) == "gabion.sample"

    assert ii._parse_python_file(tmp_path / "missing.py") is None
    broken = tmp_path / "broken.py"
    _write(broken, "def broken(:\n    pass\n")
    assert ii._parse_python_file(broken) is None

    assert ii._read_text(tmp_path / "none.txt") is None
    assert ii._parse_ast("def broken(:\n    pass\n") is None

    outside = tmp_path.parent / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    assert ii._relative(outside, tmp_path).endswith("outside.md")


def test_markdown_and_registry_helpers_cover_skip_edges(tmp_path: Path) -> None:
    (tmp_path / "fake.md").mkdir(parents=True)
    _write(tmp_path / "docs" / "ok.md", "# ok\n")
    files = list(ii._iter_markdown_files(tmp_path))
    assert tmp_path / "docs" / "ok.md" in files
    assert tmp_path / "fake.md" not in files

    graph = ImpactIndexGraph()
    sections = ii._emit_registry_sections(
        graph,
        rows=[
            {"section_id": "", "phase": "ignored", "deps": []},
            {"section_id": "summary", "phase": "post", "deps": ["intro"]},
        ],
        specs=[SimpleNamespace(name="projection.summary", domain="test")],
    )
    assert "report_section:summary" in sections
    assert "report_section:projection:projection.summary" in sections


def test_comment_decorator_frontmatter_and_target_helpers(tmp_path: Path) -> None:
    comments = ii._impact_comments(
        "# impact-target: gabion.mod.fn gabion.mod.alt\n"
        "def test_case():\n"
        "    return 1\n"
    )
    assert comments[1] == ["gabion.mod.alt", "gabion.mod.fn"]
    assert ii._nearest_comment_targets(2, comments) == ["gabion.mod.alt", "gabion.mod.fn"]
    assert ii._impact_comments("# impact-target:\n") == {}
    assert ii._impact_comments("# impact-target: , ,\n") == {}

    tree = ast.parse(
        "from gabion.mod import *\n"
        "from gabion.mod import fn as alias\n"
        "import gabion.other as other\n"
    )
    aliases = ii._import_aliases(tree)
    assert aliases["alias"] == "gabion.mod.fn"
    assert aliases["other"] == "gabion.other"

    decorated = ast.parse(
        "@plain\n"
        "@impact_targets(['gabion.mod.fn', 'gabion.mod.alt'])\n"
        "@impact_targets(('gabion.mod.more', 1))\n"
        "@impact_targets(VAR)\n"
        "def test_case():\n"
        "    return 1\n"
    )
    function = decorated.body[0]
    assert isinstance(function, ast.FunctionDef)
    targets = ii._decorator_targets(function.decorator_list)
    assert targets == ["gabion.mod.alt", "gabion.mod.fn", "gabion.mod.more"]

    lambda_expr = ast.parse("(lambda value: value)(1)").body[0]
    assert isinstance(lambda_expr, ast.Expr)
    assert isinstance(lambda_expr.value, ast.Call)
    assert ii._call_name(lambda_expr.value.func) == ""

    unclosed = "---\ndoc_targets:\n  - gabion.mod.fn\n"
    payload, body = ii._parse_frontmatter(unclosed)
    assert payload == {}
    assert body == unclosed

    with_list = (
        "---\n"
        "# ignored\n"
        "doc_targets:\n"
        "  - gabion.mod.fn\n"
        "other: value\n"
        "---\n"
        "body\n"
    )
    payload, body = ii._parse_frontmatter(with_list)
    assert payload["doc_targets"] == ["gabion.mod.fn"]
    assert payload["other"] == "value"
    assert body == "body"

    assert ii._coerce_target_list("[gabion.a.fn, gabion.b.fn]") == [
        "gabion.a.fn",
        "gabion.b.fn",
    ]
    assert ii._coerce_target_list("gabion.a.fn gabion.b.fn") == [
        "gabion.a.fn",
        "gabion.b.fn",
    ]


def test_collect_symbol_universe_and_dedupe_edges(tmp_path: Path) -> None:
    assert ii._collect_symbol_universe(tmp_path) == set()

    _write(tmp_path / "src" / "gabion" / "broken.py", "def broken(:\n    pass\n")
    assert ii._collect_symbol_universe(tmp_path) == set()

    deduped = ii._dedupe_links(
        [
            ImpactLink(
                source="tests/test_mod.py::test_value",
                source_kind="test",
                target="gabion.mod.fn",
                confidence="weak",
            ),
            ImpactLink(
                source="tests/test_mod.py::test_value",
                source_kind="test",
                target="gabion.mod.fn",
                confidence="explicit",
            ),
            ImpactLink(
                source="tests/test_mod.py::test_value",
                source_kind="test",
                target="gabion.mod.fn",
                confidence="weak",
            ),
        ]
    )
    assert len(deduped) == 1
    assert deduped[0].confidence == "explicit"


def test_impact_index_main_entrypoint_writes_artifact(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "gabion" / "cli.py",
        'DATAFLOW_COMMAND = "gabion.dataflowAudit"\n',
    )
    _write(
        tmp_path / "src" / "gabion" / "server.py",
        'REFACTOR_COMMAND = "gabion.refactorProtocol"\n',
    )

    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        runpy.run_module("gabion.analysis.impact_index", run_name="__main__")
    finally:
        os.chdir(cwd)

    artifact = tmp_path / "artifacts" / "audit_reports" / "impact_index.json"
    assert artifact.exists()
