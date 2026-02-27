from __future__ import annotations

from pathlib import Path

from gabion.analysis.dataflow_audit import AuditConfig
from gabion.ingest.python_adapter import PythonAdapter
from gabion.ingest.registry import resolve_adapter


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
