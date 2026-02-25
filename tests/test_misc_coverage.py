from __future__ import annotations

import io
import json
import runpy
import sys
import time
from pathlib import Path

import pytest
from lsprotocol.types import (
    CodeActionParams,
    CodeActionContext,
    DidOpenTextDocumentParams,
    DidSaveTextDocumentParams,
    Position,
    Range,
    TextDocumentIdentifier,
    TextDocumentItem,
)

# gabion:evidence E:call_footprint::tests/test_misc_coverage.py::test_main_entrypoint_invokes_app::__main__.py::gabion.__main__
def test_main_entrypoint_invokes_app() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    old_argv = sys.argv[:]
    sys.argv = ["gabion", "--help"]
    try:
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("gabion.__main__", run_name="__main__")
        assert exc.value.code == 0
    finally:
        sys.argv = old_argv

# gabion:evidence E:call_footprint::tests/test_misc_coverage.py::test_main_module_import::__main__.py::gabion.__main__
def test_main_module_import() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = __import__("gabion.__main__", fromlist=["main"])
    assert hasattr(module, "main")

# gabion:evidence E:call_footprint::tests/test_misc_coverage.py::test_analysis_engine_and_model_defaults::engine.py::gabion.analysis.engine.GabionEngine::model.py::gabion.analysis.model.CallArgs::model.py::gabion.analysis.model.ClassInfo::model.py::gabion.analysis.model.DispatchTable::model.py::gabion.analysis.model.FunctionInfo::model.py::gabion.analysis.model.ParamUse::model.py::gabion.analysis.model.SymbolTable::schema.py::gabion.schema.AnalysisResponse
def test_analysis_engine_and_model_defaults() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    from gabion.analysis.engine import GabionEngine
    from gabion.analysis.model import CallArgs, ClassInfo, DispatchTable, FunctionInfo, ParamUse, SymbolTable
    from gabion.schema import AnalysisResponse

    engine = GabionEngine()
    result = engine.analyze()
    assert isinstance(result, AnalysisResponse)
    assert result.bundles == []
    assert result.stats == {}

    table = SymbolTable()
    assert table.imports == {}
    class_info = ClassInfo(qual="X", bases=["Base"], methods=set())
    assert class_info.qual == "X"
    dispatch = DispatchTable(name="call", targets=set())
    assert dispatch.targets == set()
    call_args = CallArgs(callee_expr="foo()", pos_map={}, kw_map={})
    assert call_args.resolved_targets == []
    func_info = FunctionInfo(qual="mod.fn", params=[], calls=[])
    assert func_info.calls == []
    param_use = ParamUse(direct_forward=set(), non_forward=False, current_aliases=set())
    assert param_use.current_aliases == set()

# gabion:evidence E:decision_surface/direct::lsp_client.py::gabion.lsp_client._read_response::request_id E:decision_surface/direct::lsp_client.py::gabion.lsp_client._read_response::stale_16b588918d64
def test_lsp_client_rpc_roundtrip() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    from gabion.lsp_client import LspClientError, _read_response, _read_rpc, _write_rpc

    payload = {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}
    data = json.dumps(payload).encode("utf-8")
    header = f"Content-Length: {len(data)}\r\n\r\n".encode("utf-8")
    stream = io.BytesIO(header + data)
    assert _read_rpc(stream, time.monotonic_ns() + 1_000_000_000)["id"] == 1

    bad = io.BytesIO(b"Content-Length: 0\r\n\r\n{}")
    try:
        _read_rpc(bad, time.monotonic_ns() + 1_000_000_000)
        assert False, "Expected LspClientError"
    except LspClientError:
        pass
    try:
        _read_rpc(io.BytesIO(b""), time.monotonic_ns() + 1_000_000_000)
        assert False, "Expected LspClientError"
    except LspClientError:
        pass

    payload2 = {"jsonrpc": "2.0", "id": 2, "result": {"ok": False}}
    data2 = json.dumps(payload2).encode("utf-8")
    stream2 = io.BytesIO(header + data + f"Content-Length: {len(data2)}\r\n\r\n".encode("utf-8") + data2)
    assert _read_response(stream2, 2, time.monotonic_ns() + 1_000_000_000)["result"] == {
        "ok": False
    }

    out = io.BytesIO()
    _write_rpc(out, payload)
    out_value = out.getvalue()
    assert out_value.startswith(b"Content-Length:")
    _, _, body = out_value.partition(b"\r\n\r\n")
    assert json.loads(body.decode("utf-8")) == payload

# gabion:evidence E:decision_surface/direct::server.py::gabion.server.did_open::ls E:decision_surface/direct::server.py::gabion.server.did_save::ls
def test_server_code_actions_and_diagnostics(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    from gabion import server

    sample = tmp_path / "sample.py"
    sample.write_text("def alpha(x):\n    return x\n")

    class _Doc:
        def __init__(self, path: str) -> None:
            self.path = path

    class _Workspace:
        def __init__(self, root: str, doc_path: str) -> None:
            self.root_path = root
            self._doc = _Doc(doc_path)

        def get_document(self, uri: str) -> _Doc:
            return self._doc

    class _Server:
        def __init__(self, root: str, doc_path: str) -> None:
            self.workspace = _Workspace(root, doc_path)
            self.published: list[tuple[str, list]] = []
            self._latest_uri: str | None = None

        def publish_diagnostics(self, uri: str, diagnostics: list) -> None:
            self._latest_uri = uri
            self.published.append((uri, list(diagnostics)))

    ls = _Server(str(tmp_path), str(sample))
    uri = TextDocumentIdentifier(uri=sample.as_uri())
    params = CodeActionParams(
        text_document=uri,
        range=Range(start=Position(line=0, character=0), end=Position(line=0, character=0)),
        context=CodeActionContext(diagnostics=[]),
    )
    actions = server.code_action(ls, params)
    assert actions and actions[0].command is not None

    open_params = DidOpenTextDocumentParams(
        text_document=TextDocumentItem(
            uri=sample.as_uri(),
            language_id="python",
            version=1,
            text=sample.read_text(),
        )
    )
    server.did_open(ls, open_params)
    assert ls.published

    save_params = DidSaveTextDocumentParams(text_document=uri)
    server.did_save(ls, save_params)
    assert len(ls.published) >= 2
