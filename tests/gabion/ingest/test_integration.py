from __future__ import annotations

from gabion.ingest import adapter_contract, python_ingest, registry


def test_ingest_directory_integration_imports() -> None:
    assert adapter_contract is not None
    assert python_ingest is not None
    assert registry is not None
