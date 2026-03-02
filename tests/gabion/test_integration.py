from __future__ import annotations

from gabion import cli, server


def test_root_integration_imports_cli_and_server() -> None:
    assert cli is not None
    assert server is not None
