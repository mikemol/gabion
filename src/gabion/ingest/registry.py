# gabion:decision_protocol_module
from __future__ import annotations

from pathlib import Path

from gabion import never
from gabion.ingest.adapter_contract import LanguageAdapter
from gabion.ingest.python_adapter import PythonAdapter


_ADAPTERS_BY_LANGUAGE: dict[str, LanguageAdapter] = {}
_ADAPTERS_BY_EXTENSION: dict[str, LanguageAdapter] = {}


def register_adapter(adapter: LanguageAdapter) -> None:
    _ADAPTERS_BY_LANGUAGE[adapter.language_id] = adapter
    for extension in adapter.file_extensions:
        _ADAPTERS_BY_EXTENSION[extension.lower()] = adapter


def adapter_for_language(language_id: str) -> LanguageAdapter | None:
    return _ADAPTERS_BY_LANGUAGE.get(language_id.lower())


def adapter_for_extension(extension: str) -> LanguageAdapter | None:
    return _ADAPTERS_BY_EXTENSION.get(extension.lower())


def resolve_adapter(
    *,
    paths: list[Path],
    language_id: str | None = None,
    default_language_id: str = "python",
) -> LanguageAdapter:
    if language_id is not None:
        adapter = adapter_for_language(language_id)
        if adapter is None:
            never("unknown language adapter", language_id=language_id)
        return adapter
    for path in paths:
        suffix = path.suffix.lower()
        if not suffix:
            continue
        adapter = adapter_for_extension(suffix)
        if adapter is not None:
            return adapter
    # Import-time registration guarantees a canonical fallback adapter.
    return _ADAPTERS_BY_LANGUAGE[default_language_id.lower()]


register_adapter(PythonAdapter())
