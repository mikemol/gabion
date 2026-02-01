from __future__ import annotations

from pathlib import Path

import pytest


def test_code_action_refactor_stub(tmp_path: Path) -> None:
    pygls = pytest.importorskip("pygls")
    lsprotocol = pytest.importorskip("lsprotocol")
    from gabion.server import REFACTOR_COMMAND, code_action
    from lsprotocol.types import CodeActionParams, Position, Range, TextDocumentIdentifier

    target = tmp_path / "sample.py"
    target.write_text("def f():\n    return 1\n")

    params = CodeActionParams(
        text_document=TextDocumentIdentifier(uri=target.as_uri()),
        range=Range(start=Position(line=0, character=0), end=Position(line=0, character=1)),
        context={"diagnostics": []},
    )

    actions = code_action(None, params)
    assert actions, "Expected at least one code action"
    action = actions[0]
    assert action.command is not None
    assert action.command.command == REFACTOR_COMMAND
    assert action.command.arguments
    payload = action.command.arguments[0]
    assert payload["protocol_name"] == "TODO_Bundle"
    assert payload["target_path"].endswith("sample.py")
