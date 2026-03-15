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
    assert len(result.lexemes) == 96
    assert len(result.tokens) == 96
    assert len(result.chart) == len(result.tokens) + 1
    assert result.item_count > 0
    assert all(len(rule.rhs) in (1, 2) for rule in result.grammar)
    assert all(not hasattr(rule, "rule_id") for rule in result.grammar)
    assert result.states[0].rank == 0
    assert result.final_state.rank == len(result.states) - 1
    assert isinstance(result.lineage.lineage_id, module.PrimeFactor)
    assert isinstance(result.lineage.lexeme_stream_id, module.PrimeFactor)
    assert isinstance(result.final_state.stamp.stamp_id, module.PrimeFactor)

    first_token = result.tokens[0]
    assert isinstance(first_token.carrier_id, module.PrimeFactor)
    assert first_token.lexeme == result.lexemes[0]
    assert tuple(first_token.window.as_islice(result.lexemes)) == (result.lexemes[0],)
    assert first_token.rule.head.token == f"terminal:{result.lexemes[0].terminal_name}"
    assert list(first_token.iter_frontier_generators()) == []
    assert first_token.rule.is_lexical

    first_item = result.chart[0].items()[0]
    assert isinstance(first_item, type(first_token))
    assert isinstance(first_item.carrier_id, module.PrimeFactor)
    assert first_item.projection.basis_path.atoms
    assert first_item.projection.prime_product > 0
    assert first_item.one_cell.representative
    assert first_item.one_cell.basis_path
    assert [factor.prime for factor in first_item.prime_factor.iter_chain()]
    assert first_item.stamp.state_rank == first_item.state_rank
    assert isinstance(result.chart[0].column_id, module.PrimeFactor)
