from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis.dataflow_audit import (
        _build_symbol_table,
        _collect_dataclass_registry,
        _iter_dataclass_call_bundles,
    )

    return _iter_dataclass_call_bundles, _build_symbol_table, _collect_dataclass_registry


def test_dataclass_call_bundles_accepts_expression_values(tmp_path: Path) -> None:
    _iter_dataclass_call_bundles, _, _ = _load()
    source = tmp_path / "example.py"
    source.write_text(
        """
from dataclasses import dataclass

@dataclass
class Payload:
    alpha: int
    beta: int
    gamma: int


def build(alpha, beta, gamma):
    return Payload(alpha, beta + 1, gamma=make_gamma())
"""
    )
    bundles = _iter_dataclass_call_bundles(source)
    assert ("alpha", "beta", "gamma") in bundles


def test_dataclass_call_bundles_resolve_cross_file(tmp_path: Path) -> None:
    _iter_dataclass_call_bundles, _build_symbol_table, _collect_dataclass_registry = _load()
    root = tmp_path
    (root / "models.py").write_text(
        """
from dataclasses import dataclass

@dataclass
class Payload:
    alpha: int
    beta: int
    gamma: int
"""
    )
    caller = root / "consumer.py"
    caller.write_text(
        """
from models import Payload

def build(alpha, beta, gamma):
    return Payload(alpha, beta + 1, gamma=make_gamma())
"""
    )
    paths = [root / "models.py", caller]
    symbol_table = _build_symbol_table(paths, root, external_filter=True)
    registry = _collect_dataclass_registry(paths, project_root=root)
    bundles = _iter_dataclass_call_bundles(
        caller,
        project_root=root,
        symbol_table=symbol_table,
        dataclass_registry=registry,
    )
    assert ("alpha", "beta", "gamma") in bundles
