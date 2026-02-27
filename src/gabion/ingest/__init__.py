"""Normalized ingestion contracts for analysis pipelines."""

from .python_ingest import (
    ParseFailureWitness,
    PythonFileIngestCarrier,
    PythonFunctionIngestCarrier,
    ingest_python_file,
    iter_python_paths,
)

__all__ = [
    "ParseFailureWitness",
    "PythonFileIngestCarrier",
    "PythonFunctionIngestCarrier",
    "ingest_python_file",
    "iter_python_paths",
]
