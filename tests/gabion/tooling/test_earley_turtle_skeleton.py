from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "misc" / "earley_turtle_skeleton.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "earley_turtle_skeleton",
        _SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("failed to load earley_turtle_skeleton module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_kernel_turtle_earley_skeleton_builds_prime_backed_aspf_items() -> None:
    module = _load_module()

    result = module.run_kernel_turtle_earley_skeleton(max_tokens=96)

    assert result.source_paths == (
        "in/lg_kernel_ontology_cut_elim-1.ttl",
        "in/lg_kernel_shapes_cut_elim-1.ttl",
        "in/lg_kernel_example_cut_elim-1.ttl",
    )
    assert len(result.tokens) == 96
    assert len(result.chart) == len(result.tokens) + 1
    assert result.item_count > 0

    first_item = result.chart[0].items()[0]
    assert first_item.carrier.item_id
    assert first_item.carrier.projection.basis_path.atoms
    assert first_item.carrier.projection.prime_product > 0
    assert first_item.carrier.one_cell.representative
    assert first_item.carrier.one_cell.basis_path
