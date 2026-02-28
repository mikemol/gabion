from __future__ import annotations

from pathlib import Path

import pytest

from gabion.analysis.dataflow_audit import AuditConfig
from gabion.exceptions import NeverThrown
from gabion import ingest
from gabion.ingest.python_adapter import PythonAdapter
from gabion.ingest.registry import adapter_for_extension, resolve_adapter


# gabion:evidence E:function_site::registry.py::gabion.ingest.registry.resolve_adapter
def test_resolve_adapter_uses_explicit_language_id(tmp_path: Path) -> None:
    source_path = tmp_path / "module.py"
    source_path.write_text("def f() -> int:\n    return 1\n", encoding="utf-8")

    adapter = resolve_adapter(paths=[source_path], language_id="python")

    assert isinstance(adapter, PythonAdapter)


# gabion:evidence E:function_site::python_adapter.py::gabion.ingest.python_adapter.PythonAdapter.normalize
def test_python_adapter_normalize_discovers_and_parses_python_sources(tmp_path: Path) -> None:
    source_path = tmp_path / "module.py"
    source_path.write_text("def f() -> int:\n    return 1\n", encoding="utf-8")

    adapter = PythonAdapter()
    normalized = adapter.normalize([tmp_path], config=AuditConfig(project_root=tmp_path))

    assert normalized.language_id == "python"
    assert normalized.file_paths == (source_path,)
    assert len(normalized.parsed_units) == 1
    assert normalized.parsed_units[0].path == source_path
    assert normalized.parsed_units[0].function_count == 1


def test_resolve_adapter_prefers_extension_and_falls_back_to_default(tmp_path: Path) -> None:
    source_path = tmp_path / "module.PY"
    source_path.write_text("def f() -> int:\n    return 1\n", encoding="utf-8")
    assert isinstance(adapter_for_extension(".PY"), PythonAdapter)
    assert isinstance(resolve_adapter(paths=[source_path]), PythonAdapter)
    assert isinstance(resolve_adapter(paths=[tmp_path / "README"]), PythonAdapter)


def test_resolve_adapter_skips_unknown_extension_then_uses_known_extension(tmp_path: Path) -> None:
    unknown = tmp_path / "README.md"
    known = tmp_path / "module.py"
    adapter = resolve_adapter(paths=[unknown, known])
    assert isinstance(adapter, PythonAdapter)


def test_resolve_adapter_rejects_unknown_explicit_language() -> None:
    with pytest.raises(NeverThrown):
        resolve_adapter(paths=[], language_id="not-a-language")


def test_ingest_package_resolve_adapter_wrapper() -> None:
    adapter = ingest.resolve_adapter(paths=[], language_id="python")
    assert isinstance(adapter, PythonAdapter)
