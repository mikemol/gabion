from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from tests.path_helpers import REPO_ROOT

def _load():
    repo_root = REPO_ROOT
    from gabion.analysis.dataflow.engine import dataflow_analysis_index as index_owner
    from gabion.analysis.dataflow.engine import dataflow_documented_bundles as documented
    from gabion.analysis.dataflow.engine import dataflow_post_phase_analyses as post_phase

    return SimpleNamespace(
        _iter_config_fields=post_phase._iter_config_fields,
        _collect_config_bundles=post_phase._collect_config_bundles,
        _iter_documented_bundles=documented._iter_documented_bundles,
        _collect_dataclass_registry=post_phase._collect_dataclass_registry,
        _build_symbol_table=index_owner._build_symbol_table,
        _iter_dataclass_call_bundles=post_phase._iter_dataclass_call_bundles,
    )

# gabion:evidence E:function_site::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._collect_config_bundles E:function_site::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._iter_config_fields E:function_site::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._iter_documented_bundles
# gabion:behavior primary=desired
def test_config_bundles_and_documented_markers(tmp_path: Path) -> None:
    da = _load()
    config_path = tmp_path / "config.py"
    config_path.write_text(
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass\n"
        "class AppConfig:\n"
        "    host: str\n"
        "    port: int\n"
        "\n"
        "@dataclass\n"
        "class Helper:\n"
        "    alpha_fn: str\n"
        "    beta: int\n"
        "\n"
        "class PlainConfig:\n"
        "    enabled: bool\n"
        "    mode_fn: str\n"
    )
    bundles = da._iter_config_fields(config_path, parse_failure_witnesses=[])
    assert bundles["AppConfig"] == {"host", "port"}
    assert bundles["Helper"] == {"alpha_fn"}
    assert bundles["PlainConfig"] == {"enabled", "mode_fn"}

    other_path = tmp_path / "other.py"
    other_path.write_text("class Nope:\n    pass\n")
    collected = da._collect_config_bundles(
        [config_path, other_path],
        parse_failure_witnesses=[],
    )
    assert config_path in collected
    assert other_path not in collected

    doc_path = tmp_path / "doc.py"
    doc_path.write_text(
        "# dataflow-bundle: a, b\n"
        "# dataflow-bundle:\n"
        "# dataflow-bundle: single\n"
        "# dataflow-bundle: c d\n"
    )
    documented = da._iter_documented_bundles(doc_path)
    assert ("a", "b") in documented
    assert ("c", "d") in documented
    assert ("single",) not in documented

# gabion:evidence E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._iter_dataclass_call_bundles._resolve_fields::call E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._iter_dataclass_call_bundles::dataclass_registry,symbol_table E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._collect_module_exports::import_map,module_name E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._module_name::project_root E:decision_surface/value_encoded::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._collect_module_exports::import_map E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._collect_module_exports::stale_2e24fe6088de
# gabion:behavior primary=desired
def test_dataclass_registry_and_call_bundles(tmp_path: Path) -> None:
    da = _load()
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "from .types import ExternalConfig\n"
        "__all__ = [\"ExternalConfig\"]\n"
    )
    (pkg / "types.py").write_text(
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass\n"
        "class ExternalConfig:\n"
        "    x: int\n"
        "    y: int\n"
        "\n"
        "@dataclass\n"
        "class LocalOnly:\n"
        "    alpha: str\n"
        "    beta: str\n"
    )
    consumer = pkg / "consumer.py"
    consumer.write_text(
        "from dataclasses import dataclass\n"
        "from pkg.types import ExternalConfig\n"
        "from pkg.types import *\n"
        "import pkg.types as types\n"
        "\n"
        "@dataclass\n"
        "class LocalConfig:\n"
        "    a: int\n"
        "    b: int\n"
        "\n"
        "def build():\n"
        "    ExternalConfig(1, 2)\n"
        "    ExternalConfig(x=1, y=2)\n"
        "    LocalConfig(3, 4)\n"
        "    LocalOnly('a', 'b')\n"
        "    types.ExternalConfig(5, 6)\n"
        "    ExternalConfig(*[1, 2])\n"
        "    ExternalConfig(**{\"x\": 1, \"y\": 2})\n"
    )

    paths = [pkg / "__init__.py", pkg / "types.py", consumer]
    parse_failure_witnesses = []
    registry = da._collect_dataclass_registry(
        paths,
        project_root=tmp_path,
        parse_failure_witnesses=parse_failure_witnesses,
    )
    assert "pkg.types.ExternalConfig" in registry

    symbol_table = da._build_symbol_table(
        paths,
        tmp_path,
        external_filter=False,
        parse_failure_witnesses=parse_failure_witnesses,
    )
    bundles = da._iter_dataclass_call_bundles(
        consumer,
        project_root=tmp_path,
        symbol_table=symbol_table,
        dataclass_registry=registry,
        parse_failure_witnesses=parse_failure_witnesses,
    )
    assert ("x", "y") in bundles
    assert ("a", "b") in bundles
    assert ("alpha", "beta") in bundles

# gabion:evidence E:function_site::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._iter_config_fields E:function_site::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._iter_documented_bundles E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._iter_config_fields::stale_9c7c641d466a
# gabion:behavior primary=verboten facets=error
def test_config_and_documented_bundles_error_paths(tmp_path: Path) -> None:
    da = _load()
    missing = tmp_path / "missing.py"
    assert da._iter_documented_bundles(missing) == set()

    bad_config = tmp_path / "bad.py"
    bad_config.write_text("def broken(:\n")
    assert da._iter_config_fields(bad_config, parse_failure_witnesses=[]) == {}

# gabion:evidence E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._module_name::project_root E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._module_name::stale_6d71a13c7fb3
# gabion:behavior primary=desired
def test_collect_dataclass_registry_with_explicit_project_root(tmp_path: Path) -> None:
    da = _load()
    good = tmp_path / "good.py"
    good.write_text(
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass\n"
        "class Config:\n"
        "    a: int\n"
        "    b: int\n"
    )
    bad = tmp_path / "bad.py"
    bad.write_text("def broken(:\n")
    registry = da._collect_dataclass_registry(
        [good, bad],
        project_root=tmp_path,
        parse_failure_witnesses=[],
    )
    assert any(key.endswith(".Config") for key in registry)

# gabion:evidence E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._iter_dataclass_call_bundles._resolve_fields::call E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._iter_dataclass_call_bundles::dataclass_registry,symbol_table E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._module_name::project_root E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._iter_dataclass_call_bundles::stale_3248d400b81b
# gabion:behavior primary=verboten facets=invalid
def test_iter_dataclass_call_bundles_invalid_file(tmp_path: Path) -> None:
    da = _load()
    bad = tmp_path / "bad.py"
    bad.write_text("def broken(:\n")
    bundles = da._iter_dataclass_call_bundles(
        bad,
        project_root=tmp_path,
        parse_failure_witnesses=[],
    )
    assert bundles == set()
