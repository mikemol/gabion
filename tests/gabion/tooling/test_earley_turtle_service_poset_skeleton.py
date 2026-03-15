from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = (
    _REPO_ROOT / "scripts" / "misc" / "earley_turtle_service_poset_skeleton.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "earley_turtle_service_poset_skeleton",
        _SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("failed to load earley_turtle_service_poset_skeleton module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _is_monotone(values: list[int]) -> bool:
    return values == sorted(values)


def _request_monotone_counts(trace, attr: str) -> bool:
    counts_by_request: dict[int, list[int]] = {}
    for emission in trace:
        counts_by_request.setdefault(emission.request_id.prime, []).append(
            len(getattr(emission.stage, attr))
        )
    return all(_is_monotone(counts) for counts in counts_by_request.values())


def test_run_turtle_service_poset_skeleton_finds_first_directive_witness() -> None:
    module = _load_module()

    result = module.run_turtle_service_poset_skeleton(symbol="directive", max_tokens=96)

    assert result.source_paths == (
        "in/lg_kernel_ontology_cut_elim-1.ttl",
        "in/lg_kernel_shapes_cut_elim-1.ttl",
        "in/lg_kernel_example_cut_elim-1.ttl",
    )
    assert len(result.lexemes) == 96
    assert result.first_completed_witness is not None
    assert result.exhaustion is None
    assert result.first_completed_witness.symbol.token == "directive"
    assert (result.first_completed_witness.start, result.first_completed_witness.stop) == (
        0,
        4,
    )
    assert result.scanner_trace
    assert result.predictor_trace
    assert result.completer_trace
    assert all(len(rule.rhs) in (1, 2) for rule in result.grammar)
    assert all(not hasattr(rule, "rule_id") for rule in result.grammar)
    assert isinstance(result.root_query.query_id, module.PrimeFactor)
    assert isinstance(result.first_completed_witness.witness_id, module.PrimeFactor)
    assert _request_monotone_counts(result.scanner_trace, "truth_ids")
    assert _request_monotone_counts(result.predictor_trace, "candidate_ids")
    assert _request_monotone_counts(result.completer_trace, "completed_ids")


def test_service_poset_router_packs_scanner_ambiguity_and_dedupes_repeated_queries() -> None:
    module = _load_module()

    router = module.ServicePosetRouter.create(max_tokens=256)
    scan_request = router.make_scan_request()
    responses = router.stream_scan(scan_request)

    assert any(isinstance(response, module.ScannerDelta) for response in responses)
    assert all(
        isinstance(response, (module.ScannerDelta, module.Exhausted))
        for response in responses
    )
    assert all(
        emission.delta is None or emission.yielded == (emission.delta,)
        for emission in router.scanner_trace
        if emission.service_name == "scanner"
    )

    ambiguity_samples: list[tuple[str, set[str]]] = []
    for emission in router.scanner_trace:
        delta = emission.delta
        if not isinstance(delta, module.ScannerDelta):
            continue
        for ambiguity in delta.new_ambiguity_classes:
            truth_ids = set(ambiguity.truth_ids)
            truths = [
                truth for truth in emission.stage.truths if truth.truth_id in truth_ids
            ]
            if len(truths) > 1:
                ambiguity_samples.append(
                    (
                        ambiguity.lexeme_text,
                        {truth.symbol.token for truth in truths},
                    )
                )

    assert any(
        lexeme_text == "a" and {"terminal:A", "terminal:NAME"} <= symbols
        for lexeme_text, symbols in ambiguity_samples
    )

    query = router.make_root_query(symbol_name="directive")
    first = router.resolve_query(query)
    second = router.resolve_query(query)

    assert first is not None
    assert second is not None
    assert first == second
    complete_total, complete_unique = router.request_metrics["complete"]
    assert complete_total > complete_unique


def test_service_poset_router_predictor_advances_by_deltas() -> None:
    module = _load_module()

    router = module.ServicePosetRouter.create(max_tokens=96)
    prediction_request = router._make_prediction_request(
        symbol=router.symbol_factor("directive"),
        start=None,
        end=None,
    )
    responses = tuple(router._stream_predictions(prediction_request))

    assert any(isinstance(response, module.PredictorDelta) for response in responses)
    assert all(
        isinstance(response, (module.PredictorDelta, module.Exhausted))
        for response in responses
    )
    assert all(
        emission.delta is None or emission.yielded == (emission.delta,)
        for emission in router.predictor_trace
        if emission.service_name == "predictor"
    )


def test_run_turtle_service_poset_skeleton_exhausts_cleanly_for_missing_symbol() -> None:
    module = _load_module()

    result = module.run_turtle_service_poset_skeleton(
        symbol="missing_symbol",
        max_tokens=96,
    )

    assert result.first_completed_witness is None
    assert result.exhaustion is not None
    assert result.exhaustion.service_name == "completer"
    assert result.scanner_trace
    assert result.predictor_trace
    assert result.completer_trace
    assert _request_monotone_counts(result.scanner_trace, "truth_ids")
    assert _request_monotone_counts(result.predictor_trace, "candidate_ids")
    assert _request_monotone_counts(result.completer_trace, "pending_ids")
