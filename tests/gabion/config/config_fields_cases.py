from __future__ import annotations

from pathlib import Path
from tests.path_helpers import REPO_ROOT

def _load():
    repo_root = REPO_ROOT
    from gabion.analysis.dataflow.engine.dataflow_post_phase_analyses import (
        _iter_config_fields,
    )

    return _iter_config_fields

# gabion:evidence E:function_site::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._iter_config_fields E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._iter_config_fields::stale_3662200066f5
# gabion:behavior primary=desired
def test_iter_config_fields_expands_config_dataclass(tmp_path: Path) -> None:
    _iter_config_fields = _load()
    config_path = tmp_path / "config.py"
    config_path.write_text(
        """
from dataclasses import dataclass

@dataclass
class AppConfig:
    foo: int
    bar_fn: str
    baz: str = "ok"

@dataclass
class Other:
    alpha_fn: int
    beta: int
"""
    )
    bundles = _iter_config_fields(config_path, parse_failure_witnesses=[])
    assert bundles["AppConfig"] == {"foo", "bar_fn", "baz"}
    assert bundles["Other"] == {"alpha_fn"}
