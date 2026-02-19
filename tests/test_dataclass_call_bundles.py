from __future__ import annotations

from pathlib import Path

def _load():
    repo_root = Path(__file__).resolve().parents[1]
    from gabion.analysis.dataflow_audit import (
        _build_symbol_table,
        _collect_dataclass_registry,
        _iter_dataclass_call_bundles,
    )

    return _iter_dataclass_call_bundles, _build_symbol_table, _collect_dataclass_registry

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_dataclass_call_bundles._resolve_fields::call E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_dataclass_call_bundles::dataclass_registry,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._module_name::project_root
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
    bundles = _iter_dataclass_call_bundles(source, parse_failure_witnesses=[])
    assert ("alpha", "beta", "gamma") in bundles

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_dataclass_call_bundles._resolve_fields::call E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_dataclass_call_bundles::dataclass_registry,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_module_exports::import_map,module_name E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._module_name::project_root E:decision_surface/value_encoded::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_module_exports::import_map
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
    parse_failure_witnesses = []
    symbol_table = _build_symbol_table(
        paths,
        root,
        external_filter=True,
        parse_failure_witnesses=parse_failure_witnesses,
    )
    registry = _collect_dataclass_registry(
        paths,
        project_root=root,
        parse_failure_witnesses=parse_failure_witnesses,
    )
    bundles = _iter_dataclass_call_bundles(
        caller,
        project_root=root,
        symbol_table=symbol_table,
        dataclass_registry=registry,
        parse_failure_witnesses=parse_failure_witnesses,
    )
    assert ("alpha", "beta", "gamma") in bundles


# gabion:evidence E:call_footprint::tests/test_dataclass_call_bundles.py::test_dataclass_call_bundles_support_literal_star_args::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_dataclass_call_bundles::test_dataclass_call_bundles.py::tests.test_dataclass_call_bundles._load
def test_dataclass_call_bundles_support_literal_star_args(tmp_path: Path) -> None:
    _iter_dataclass_call_bundles, _, _ = _load()
    source = tmp_path / "starred.py"
    source.write_text(
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Payload:\n"
        "    x: int\n"
        "    y: int\n"
        "def build():\n"
        "    Payload(*[1, 2])\n"
        "    Payload(**{'x': 1, 'y': 2})\n"
    )
    first_witnesses: list[dict[str, object]] = []
    first = _iter_dataclass_call_bundles(source, parse_failure_witnesses=first_witnesses)
    second_witnesses: list[dict[str, object]] = []
    second = _iter_dataclass_call_bundles(source, parse_failure_witnesses=second_witnesses)
    assert ("x", "y") in first
    assert first == second
    assert first_witnesses == second_witnesses == []


# gabion:evidence E:call_footprint::tests/test_dataclass_call_bundles.py::test_dataclass_call_bundles_emit_unresolved_starred_evidence::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_dataclass_call_bundles::test_dataclass_call_bundles.py::tests.test_dataclass_call_bundles._load
def test_dataclass_call_bundles_emit_unresolved_starred_evidence(tmp_path: Path) -> None:
    _iter_dataclass_call_bundles, _, _ = _load()
    source = tmp_path / "dynamic_starred.py"
    source.write_text(
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Payload:\n"
        "    x: int\n"
        "    y: int\n"
        "def build(values):\n"
        "    Payload(*values)\n"
    )
    witnesses: list[dict[str, object]] = []
    bundles = _iter_dataclass_call_bundles(source, parse_failure_witnesses=witnesses)
    assert bundles == set()
    assert any(w.get("reason") == "unresolved_starred_positional" for w in witnesses)
