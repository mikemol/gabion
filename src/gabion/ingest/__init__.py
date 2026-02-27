from gabion.ingest.adapter_contract import LanguageAdapter, NormalizedIngestBundle, ParsedFileUnit
from .python_ingest import (
    ParseFailureWitness,
    PythonFileIngestCarrier,
    PythonFunctionIngestCarrier,
    ingest_python_file,
    iter_python_paths,
)

def resolve_adapter(*, paths, language_id=None, default_language_id="python"):
    from gabion.ingest.registry import resolve_adapter as _resolve_adapter

    return _resolve_adapter(
        paths=paths,
        language_id=language_id,
        default_language_id=default_language_id,
    )


__all__ = [
    "LanguageAdapter",
    "NormalizedIngestBundle",
    "ParsedFileUnit",
    "ParseFailureWitness",
    "PythonFileIngestCarrier",
    "PythonFunctionIngestCarrier",
    "ingest_python_file",
    "iter_python_paths",
    "resolve_adapter",
]
