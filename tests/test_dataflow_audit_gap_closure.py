from __future__ import annotations

import ast
from pathlib import Path

import pytest

from gabion.exceptions import NeverThrown


def _load():
    from gabion.analysis import dataflow_audit as da

    return da


def test_execution_pattern_predicate_call_shape_continue_paths() -> None:
    da = _load()
    predicate = da._ExecutionPatternPredicate(
        kind=da._ExecutionPatternPredicateKind.CALL_SHAPE,
        callee_names=frozenset({"target"}),
        required_keywords=frozenset({"kw"}),
        min_positional_args=2,
    )
    fact = da._ExecutionFunctionFact(
        function_name="f",
        param_names=frozenset(),
        called_names=frozenset({"target"}),
        call_shapes={
            "target": (
                da._ExecutionCallShape(
                    positional_args=1,
                    keyword_names=frozenset({"kw"}),
                ),
                da._ExecutionCallShape(
                    positional_args=2,
                    keyword_names=frozenset(),
                ),
            )
        },
    )
    assert predicate.matches(fact=fact) is False


def test_normalize_adapter_contract_defaults_for_non_mapping() -> None:
    da = _load()
    normalized = da.normalize_adapter_contract([])
    assert normalized["name"] == "native"
    capabilities = normalized["capabilities"]
    assert isinstance(capabilities, dict)
    assert all(bool(value) for value in capabilities.values())


def test_execution_pattern_instances_handles_missing_and_parse_error_sources(
    tmp_path: Path,
) -> None:
    da = _load()
    assert da._execution_pattern_instances(source_path=tmp_path / "missing.py") == []
    assert da._execution_pattern_instances(source="def broken(") == []


def test_witness_obligation_predicate_flags_identity_mismatch_and_missing_post_identity() -> None:
    da = _load()
    context = da._RewritePredicateContext(
        expected_base=[],
        expected_ctor=[],
        expected_remainder={},
        expected_strata="",
        expected_candidates=[],
        post_base=[],
        post_ctor=[],
        post_remainder={},
        post_matches=(),
        post_strata="",
        post_exception_obligations=[],
        pre={},
        plan_evidence={
            "witness_obligations": [
                {
                    "required": True,
                    "witness_ref": "witness:1",
                    "kind": "aspf_structure_class_equivalence",
                    "canonical_identity_contract": {"cache_identity": "aspf:sha1:" + ("1" * 40)},
                    "aspf_structure_class": {"label": "stable"},
                }
            ]
        },
        post_entry={},
        site=da.Site(path="mod.py", function="f", bundle=()),
    )
    evaluation = da._evaluate_witness_obligation_non_regression_predicate(
        {"kind": "witness_obligation_non_regression"},
        context,
    )
    assert evaluation["passed"] is False
    observed = evaluation["observed"]
    assert isinstance(observed, dict)
    mismatches = observed["aspf_identity_mismatches"]
    assert isinstance(mismatches, list)
    assert "canonical_identity_contract" in mismatches


def test_witness_obligation_predicate_handles_post_identity_without_pre_identity() -> None:
    da = _load()
    context = da._RewritePredicateContext(
        expected_base=[],
        expected_ctor=[],
        expected_remainder={},
        expected_strata="",
        expected_candidates=[],
        post_base=[],
        post_ctor=[],
        post_remainder={},
        post_matches=(),
        post_strata="",
        post_exception_obligations=[],
        pre={},
        plan_evidence={"witness_obligations": []},
        post_entry={"canonical_identity_contract": {"cache_identity": "aspf:sha1:" + ("2" * 40)}},
        site=da.Site(path="mod.py", function="f", bundle=()),
    )
    evaluation = da._evaluate_witness_obligation_non_regression_predicate(
        {"kind": "witness_obligation_non_regression"},
        context,
    )
    assert evaluation["passed"] is True


def test_deadline_and_never_summary_respect_existing_suite_kind() -> None:
    da = _load()
    forest = da.Forest()
    original_apply_spec = da.apply_spec
    da.apply_spec = lambda _spec, _relation: [  # type: ignore[assignment]
        {
            "deadline_id": "deadline:mod.py:f:missing",
            "site_path": "mod.py",
            "site_function": "f",
            "site_suite_kind": "loop",
            "span_line": 1,
            "span_col": 0,
            "span_end_line": 1,
            "span_end_col": 1,
            "status": "VIOLATION",
            "kind": "missing",
            "detail": "deadline omitted",
        }
    ]
    try:
        summary = da._summarize_deadline_obligations(
            [
                {
                    "deadline_id": "deadline:mod.py:f:missing",
                    "site": {"path": "mod.py", "function": "f", "bundle": []},
                    "status": "VIOLATION",
                    "kind": "missing",
                    "detail": "deadline omitted",
                    "span": [1, 0, 1, 1],
                }
            ],
            max_entries=5,
            forest=forest,
        )
    finally:
        da.apply_spec = original_apply_spec
    assert summary
    assert any(
        node.kind == "SuiteSite" and node.meta.get("suite_kind") == "loop"
        for node in forest.nodes.values()
    )

    original_apply_spec = da.apply_spec
    def _never_apply_spec(_spec, _relation, **_kwargs):
        return [
            {
                "status": "OBLIGATION",
                "status_rank": 2,
                "site_path": "mod.py",
                "site_function": "f",
                "site_suite_kind": "call",
                "span_line": 1,
                "span_col": 0,
                "span_end_line": 1,
                "span_end_col": 1,
                "never_id": "never:mod.py:f:1:0",
                "reason": "",
                "witness_ref": "",
                "environment_ref": None,
                "undecidable_reason": "",
            }
        ]
    da.apply_spec = _never_apply_spec  # type: ignore[assignment]
    try:
        never_lines = da._summarize_never_invariants(
            [
                {
                    "never_id": "never:mod.py:f:1:0",
                    "site": {
                        "path": "mod.py",
                        "function": "f",
                        "bundle": [],
                        "suite_kind": "call",
                    },
                    "status": "OBLIGATION",
                    "reason": "",
                    "span": [1, 0, 1, 1],
                }
            ],
            include_proven_unreachable=True,
        )
    finally:
        da.apply_spec = original_apply_spec
    assert any("[call]" in line for line in never_lines)


def test_cache_identity_matches_requires_canonical_identity() -> None:
    da = _load()
    with pytest.raises(NeverThrown):
        da._cache_identity_matches("bad", "also_bad")
    canonical = "aspf:sha1:" + ("3" * 40)
    assert da._cache_identity_matches(canonical, canonical) is True


def test_stage_cache_aliases_and_bucket_migration_paths(tmp_path: Path) -> None:
    da = _load()
    digest = "0123456789abcdef0123456789abcdef01234567"
    prefixed = f"aspf:sha1:{digest}"
    parse_key = ("parse", da._ParseModuleStage.CONFIG_FIELDS.value, prefixed, "config_fields")
    parse_aliases = da._stage_cache_key_aliases(parse_key)
    assert parse_key in parse_aliases
    assert ("parse", da._ParseModuleStage.CONFIG_FIELDS.value, digest, "config_fields") in parse_aliases

    invalid_prefixed_key = (
        "parse",
        da._ParseModuleStage.CONFIG_FIELDS.value,
        "aspf:sha1:not-a-hex-digest",
        "config_fields",
    )
    assert da._stage_cache_key_aliases(invalid_prefixed_key) == (invalid_prefixed_key,)
    not_prefixed_key = (
        "parse",
        da._ParseModuleStage.CONFIG_FIELDS.value,
        "not-prefixed",
        "config_fields",
    )
    assert da._stage_cache_key_aliases(not_prefixed_key) == (not_prefixed_key,)

    bad_node_key = da.NodeId("ParseStageCacheIdentity", (1, 2, "detail"))
    assert da._stage_cache_key_aliases(bad_node_key) == (bad_node_key,)

    analysis_index = da.AnalysisIndex(
        by_name={},
        by_qual={},
        symbol_table=da.SymbolTable(),
        class_index={},
        index_cache_identity=prefixed,
    )
    scoped_key = (prefixed, parse_key)
    legacy_scoped_key = (
        prefixed,
        ("parse", da._ParseModuleStage.CONFIG_FIELDS.value, digest, "config_fields"),
    )
    legacy_bucket = {tmp_path / "mod.py": {"value": 1}}
    analysis_index.stage_cache_by_key[legacy_scoped_key] = legacy_bucket
    resolved_bucket = da._get_stage_cache_bucket(
        analysis_index,
        scoped_cache_key=scoped_key,
    )
    assert resolved_bucket is legacy_bucket
    assert analysis_index.stage_cache_by_key[scoped_key] is legacy_bucket

    missing_scoped = (
        prefixed,
        ("parse", da._ParseModuleStage.FUNCTION_INDEX.value, prefixed, "function_index"),
    )
    created_bucket = da._get_stage_cache_bucket(
        analysis_index,
        scoped_cache_key=missing_scoped,
    )
    assert created_bucket == {}
    assert analysis_index.stage_cache_by_key[missing_scoped] is created_bucket


def test_dataclass_registry_supports_blank_module_projection_root() -> None:
    da = _load()
    tree = ast.parse(
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Local:\n"
        "    value: int\n"
    )
    registry = da._dataclass_registry_for_tree(
        Path("mod.py"),
        tree,
        project_root=Path("mod"),
    )
    assert registry == {"Local": ["value"]}


def test_analysis_index_resume_variants_skips_invalid_variant_payload_format() -> None:
    da = _load()
    canonical_id = "aspf:sha1:" + ("a" * 40)
    variants = da._analysis_index_resume_variants(
        {
            da._ANALYSIS_INDEX_RESUME_VARIANTS_KEY: {
                canonical_id: {"format_version": 2, "value": "legacy"},
            }
        }
    )
    assert variants == {}
