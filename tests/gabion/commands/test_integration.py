from __future__ import annotations

from gabion.commands import boundary_order, check_contract, progress_transition


def test_commands_directory_integration_imports() -> None:
    assert boundary_order is not None
    assert check_contract is not None
    assert progress_transition is not None
