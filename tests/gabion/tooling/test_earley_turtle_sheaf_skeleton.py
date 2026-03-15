from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "misc" / "earley_turtle_sheaf_skeleton.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "earley_turtle_sheaf_skeleton",
        _SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("failed to load earley_turtle_sheaf_skeleton module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_turtle_sheaf_earley_skeleton_scans_presheaf_into_parsed_sections() -> None:
    module = _load_module()

    result = module.run_turtle_sheaf_earley_skeleton(max_tokens=96)

    assert result.source_paths == (
        "in/lg_kernel_ontology_cut_elim-1.ttl",
        "in/lg_kernel_shapes_cut_elim-1.ttl",
        "in/lg_kernel_example_cut_elim-1.ttl",
    )
    assert len(result.lexemes) == 96
    assert len(result.arena_presheaf) == 96
    assert len(result.scanned_sheaves) == len(result.arena_presheaf)
    assert len(result.section_presheaf) == len(result.scanned_sheaves)
    assert result.predictions
    assert result.completions
    assert all(len(rule.rhs) in (1, 2) for rule in result.grammar)
    assert all(not hasattr(rule, "rule_id") for rule in result.grammar)

    first_patch = result.arena_presheaf[0]
    assert isinstance(first_patch.patch_id, module.PrimeFactor)
    assert isinstance(first_patch.site.site_id, module.PrimeFactor)
    assert isinstance(first_patch.symbol, module.PrimeFactor)

    first_sheaf = result.scanned_sheaves[0]
    assert isinstance(first_sheaf.sheaf_id, module.PrimeFactor)
    assert isinstance(first_sheaf.prime_factor, module.PrimeFactor)
    assert first_sheaf.rule.head == first_patch.symbol
    assert first_sheaf.rule.rhs == (first_patch.symbol,)
    assert first_sheaf.cover_site_ids == (first_patch.site.site_id,)
    assert first_sheaf.local_section.member_ids == (first_patch.patch_id,)

    completion_symbols = {completion.symbol.token for completion in result.completions}
    assert {"iri_dot", "directive", "directive_tail"}.issubset(completion_symbols)
    assert any(
        prediction.rule.head.token == "document'"
        for prediction in result.predictions
    )
    assert all(
        isinstance(completion.completion_id, module.PrimeFactor)
        and isinstance(completion.left_section_id, module.PrimeFactor)
        and isinstance(completion.right_section_id, module.PrimeFactor)
        for completion in result.completions
    )
