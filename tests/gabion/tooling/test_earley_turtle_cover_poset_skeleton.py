from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = (
    _REPO_ROOT / "scripts" / "misc" / "earley_turtle_cover_poset_skeleton.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "earley_turtle_cover_poset_skeleton",
        _SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("failed to load earley_turtle_cover_poset_skeleton module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _seed_root_runtime(module, *, symbol: str = "directive", max_tokens: int = 96):
    coordinator = module.FixedPointCoordinator.create(max_tokens=max_tokens)
    root_demand = coordinator.make_completion_demand(symbol_name=symbol)
    with module.deadline_scope(module.Deadline.from_timeout_ms(30_000)):
        with module.deadline_clock_scope(module.MonotonicClock()):
            coordinator._reset_runtime_state(root_demand_id=root_demand.demand_id)
            root_derivation = coordinator._new_completer_derivation(
                demand=root_demand,
                parent_derivation_id=None,
                parent_candidate_id=None,
            )
            seed_delta = module.CompleterDelta(
                delta_id=coordinator._id_factor(
                    "ttl_cover_completer_delta",
                    f"seed|{root_demand.demand_id.prime}",
                ),
                new_completion_demands=(root_demand,),
                new_discharged_completion_demand_ids=(),
                new_suspended_derivations=(root_derivation,),
                new_completed_witnesses=(),
                new_supports=(),
            )
            coordinator._enqueue_delta("completer", None, seed_delta)
            coordinator._drain_pending_deltas()
    return coordinator, root_demand, root_derivation


# gabion:behavior primary=desired
def test_run_turtle_cover_poset_skeleton_finds_first_directive_witness() -> None:
    module = _load_module()

    result = module.run_turtle_cover_poset_skeleton(symbol="directive", max_tokens=96)

    assert result.source_paths == (
        "in/lg_kernel_ontology_cut_elim-1.ttl",
        "in/lg_kernel_shapes_cut_elim-1.ttl",
        "in/lg_kernel_example_cut_elim-1.ttl",
    )
    assert len(result.lexemes) == 96
    assert result.selected_witness is not None
    assert result.closure_state is None
    assert result.selected_witness.symbol.token == "directive"
    assert (result.selected_witness.start, result.selected_witness.stop) == (0, 4)
    assert result.selected_witness in result.final_global_stage.completer.completed_witnesses
    assert any(
        demand.demand_id == result.root_completion_demand.demand_id
        for demand in result.coverable_completion_demands()
    )
    assert result.final_global_stage.scanner.truths
    assert result.final_global_stage.scanner.closed_scan_demand_ids
    assert result.final_global_stage.predictor.candidates
    assert result.final_global_stage.predictor.suspended_derivations
    assert result.final_global_stage.predictor.closed_prediction_demand_ids
    assert any(
        derivation.service_name == "predictor"
        and derivation.status is module.DerivationStatus.CLOSED
        for derivation in result.final_global_stage.predictor.suspended_derivations
    )
    assert result.final_global_stage.completer.unsatisfied_completion_demand_ids == frozenset()
    assert result.final_global_stage.completer.supports
    assert isinstance(result.root_completion_demand.demand_id, module.PrimeFactor)
    assert isinstance(result.consumer_policy.policy_id, module.PrimeFactor)
    assert isinstance(result.final_global_stage.stage_id, module.PrimeFactor)
    assert isinstance(result.selected_witness.witness_id, module.PrimeFactor)
    assert all(len(rule.rhs) in (1, 2) for rule in result.grammar)
    assert all(not hasattr(rule, "rule_id") for rule in result.grammar)
    assert isinstance(result.last_joined_delta, module.CompleterDelta)


# gabion:behavior primary=desired
def test_fixed_point_coordinator_observe_scan_demand_packs_real_ambiguity() -> None:
    module = _load_module()

    coordinator = module.FixedPointCoordinator.create(max_tokens=256)
    demand = coordinator.make_scan_demand()
    stage = coordinator.observe_scan_demand(demand)

    assert stage.scan_demands
    assert stage.truths
    assert stage.ambiguity_classes
    assert demand.demand_id in stage.closed_scan_demand_ids
    assert isinstance(demand.demand_id, module.PrimeFactor)
    assert isinstance(stage.stage_id, module.PrimeFactor)

    ambiguity_samples: list[tuple[str, set[str]]] = []
    for ambiguity in stage.ambiguity_classes:
        truth_ids = set(ambiguity.truth_ids)
        truths = [truth for truth in stage.truths if truth.truth_id in truth_ids]
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


# gabion:behavior primary=desired
def test_live_suspended_derivations_use_cover_objects() -> None:
    module = _load_module()
    coordinator, _, root_derivation = _seed_root_runtime(module)

    with module.deadline_scope(module.Deadline.from_timeout_ms(30_000)):
        with module.deadline_clock_scope(module.MonotonicClock()):
            proposal = coordinator._step_completer_derivation(root_derivation.derivation_id.prime)
            assert isinstance(proposal, module.PredictionWait)
            coordinator._handle_completer_proposal(root_derivation.derivation_id.prime, proposal)

            root_live = next(
                derivation
                for derivation in coordinator.completer_stage.suspended_derivations
                if derivation.derivation_id == root_derivation.derivation_id
            )
            assert isinstance(root_live.status, module.DerivationStatus)
            assert isinstance(root_live.wait_condition, module.PredictionCover)
            coordinator._assert_derivation_invariant(root_derivation.derivation_id.prime)
            assert (
                coordinator._cover_by_waiting_derivation_id[root_derivation.derivation_id.prime]
                == root_live.wait_condition
            )

            coordinator._drain_pending_deltas()
            predictor_derivation = coordinator.predictor_stage.suspended_derivations[0]
            predictor_proposal = coordinator._step_predictor_derivation(
                predictor_derivation.derivation_id.prime
            )
            assert isinstance(predictor_proposal, module.ScanWait)
            coordinator._handle_predictor_proposal(
                predictor_derivation.derivation_id.prime,
                predictor_proposal,
            )

            predictor_live = next(
                derivation
                for derivation in coordinator.predictor_stage.suspended_derivations
                if derivation.derivation_id == predictor_derivation.derivation_id
            )
            assert isinstance(predictor_live.wait_condition, module.ScannerCover)
            coordinator._assert_derivation_invariant(predictor_derivation.derivation_id.prime)
            assert (
                coordinator._cover_by_waiting_derivation_id[
                    predictor_derivation.derivation_id.prime
                ]
                == predictor_live.wait_condition
            )

            coordinator._drain_pending_deltas()
            predictor_delta = coordinator._step_predictor_derivation(
                predictor_derivation.derivation_id.prime
            )
            assert isinstance(predictor_delta, module.PredictorDelta)
            coordinator._handle_predictor_proposal(
                predictor_derivation.derivation_id.prime,
                predictor_delta,
            )
            coordinator._drain_pending_deltas()

            continuation = coordinator._step_completer_derivation(root_derivation.derivation_id.prime)
            assert isinstance(continuation, module.CompleterDelta)
            coordinator._handle_completer_proposal(root_derivation.derivation_id.prime, continuation)
            coordinator._drain_pending_deltas()

            child_wait = coordinator._step_completer_derivation(root_derivation.derivation_id.prime)
            assert isinstance(child_wait, module.CompletionWait)
            coordinator._handle_completer_proposal(root_derivation.derivation_id.prime, child_wait)

            root_waiting_live = next(
                derivation
                for derivation in coordinator.completer_stage.suspended_derivations
                if derivation.derivation_id == root_derivation.derivation_id
            )
            assert isinstance(root_waiting_live.wait_condition, module.CompletionResolved)
            assert not isinstance(root_waiting_live.wait_condition, module.CompletionSatisfied)
            coordinator._assert_derivation_invariant(root_derivation.derivation_id.prime)
            assert (
                coordinator._cover_by_waiting_derivation_id[root_derivation.derivation_id.prime]
                == root_waiting_live.wait_condition
            )

    assert any(
        trace.service_name == "scanner"
        and predictor_derivation.derivation_id in trace.awakened_derivation_ids
        for trace in coordinator.cover_traces
    )
    assert any(
        trace.service_name == "predictor"
        and root_derivation.derivation_id in trace.awakened_derivation_ids
        for trace in coordinator.cover_traces
    )


# gabion:behavior primary=desired
def test_cover_kernel_uses_trace_only_waits_and_cover_traces() -> None:
    module = _load_module()

    result = module.run_turtle_cover_poset_skeleton(symbol="directive", max_tokens=96)

    assert not hasattr(module, "ScanWitnessBatch")
    assert not hasattr(module, "PredictionWitnessBatch")
    assert not hasattr(module, "ServiceEmission")
    assert hasattr(module, "CompletionResolved")
    assert hasattr(module, "CompletionSatisfied")
    assert not hasattr(module, "CompletionCover")
    assert "status" in module.SuspendedDerivation.__annotations__
    assert "wait_condition" in module.SuspendedDerivation.__annotations__
    assert "waiting_on_service" not in module.SuspendedDerivation.__annotations__
    assert "waiting_on_demand_id" not in module.SuspendedDerivation.__annotations__
    assert module.RUNTIME_CLASSIFICATION["ScanWait"] == "provenance only"
    assert module.RUNTIME_CLASSIFICATION["PredictionWait"] == "provenance only"
    assert module.RUNTIME_CLASSIFICATION["CompletionWait"] == "provenance only"
    assert module.RUNTIME_CLASSIFICATION["CompletionSatisfied"] == "cover predicate"
    assert module.RUNTIME_CLASSIFICATION["LocallyQuiescent"] == "provenance only"
    assert module.RUNTIME_CLASSIFICATION["Unsatisfied"] == "closure truth"
    assert not hasattr(module.FixedPointCoordinator, "_queue_predictor_growth")

    scan_wait = next(
        trace.proposal
        for trace in result.proposal_traces
        if isinstance(trace.proposal, module.ScanWait)
    )
    prediction_wait = next(
        trace.proposal
        for trace in result.proposal_traces
        if isinstance(trace.proposal, module.PredictionWait)
    )
    completion_wait = next(
        trace.proposal
        for trace in result.proposal_traces
        if isinstance(trace.proposal, module.CompletionWait)
    )

    assert any(
        trace.service_name == "scanner"
        and trace.demand_id == scan_wait.demand_id
        for trace in result.cover_traces
    )
    assert any(
        trace.service_name == "predictor"
        and trace.demand_id == prediction_wait.demand_id
        and prediction_wait.requester_derivation_id in trace.awakened_derivation_ids
        for trace in result.cover_traces
    )
    assert any(
        trace.service_name == "completer"
        and trace.demand_id == completion_wait.demand_id
        and completion_wait.requester_derivation_id in trace.awakened_derivation_ids
        and trace.reason == "completion_demand_satisfied"
        for trace in result.cover_traces
    )
    assert any(
        isinstance(trace.proposal, module.LocallyQuiescent)
        for trace in result.proposal_traces
    )


# gabion:behavior primary=desired
def test_completion_resolution_wakes_on_unsatisfied_path() -> None:
    module = _load_module()
    coordinator, _, root_derivation = _seed_root_runtime(module)

    with module.deadline_scope(module.Deadline.from_timeout_ms(30_000)):
        with module.deadline_clock_scope(module.MonotonicClock()):
            prediction_wait = coordinator._step_completer_derivation(root_derivation.derivation_id.prime)
            coordinator._handle_completer_proposal(root_derivation.derivation_id.prime, prediction_wait)
            coordinator._drain_pending_deltas()

            predictor_derivation = coordinator.predictor_stage.suspended_derivations[0]
            predictor_wait = coordinator._step_predictor_derivation(
                predictor_derivation.derivation_id.prime
            )
            coordinator._handle_predictor_proposal(
                predictor_derivation.derivation_id.prime,
                predictor_wait,
            )
            coordinator._drain_pending_deltas()

            predictor_delta = coordinator._step_predictor_derivation(
                predictor_derivation.derivation_id.prime
            )
            coordinator._handle_predictor_proposal(
                predictor_derivation.derivation_id.prime,
                predictor_delta,
            )
            coordinator._drain_pending_deltas()

            continuation = coordinator._step_completer_derivation(root_derivation.derivation_id.prime)
            coordinator._handle_completer_proposal(root_derivation.derivation_id.prime, continuation)
            coordinator._drain_pending_deltas()

            child_wait = coordinator._step_completer_derivation(root_derivation.derivation_id.prime)
            assert isinstance(child_wait, module.CompletionWait)
            coordinator._handle_completer_proposal(root_derivation.derivation_id.prime, child_wait)
            coordinator._mark_completion_demand_unsatisfied(
                child_wait.demand_id,
                "synthetic_child_unsatisfied",
            )
            coordinator._assert_derivation_invariant(root_derivation.derivation_id.prime)

    assert any(
        trace.service_name == "completer"
        and trace.demand_id == child_wait.demand_id
        and trace.reason == "completion_demand_unsatisfied"
        and root_derivation.derivation_id in trace.awakened_derivation_ids
        for trace in coordinator.cover_traces
    )


# gabion:behavior primary=desired
def test_derivation_invariants_hold_for_completed_runtime() -> None:
    module = _load_module()
    coordinator = module.FixedPointCoordinator.create(max_tokens=96)
    root_demand = coordinator.make_completion_demand(symbol_name="directive")
    policy = coordinator.make_consumer_policy(mode="first")

    coordinator.run_root_demand(root_demand, policy)

    terminal_statuses = {
        module.DerivationStatus.DONE,
        module.DerivationStatus.UNSATISFIED,
        module.DerivationStatus.CLOSED,
    }
    for derivation_id, state in coordinator._predictor_derivation_states_by_id.items():
        coordinator._assert_derivation_invariant(derivation_id)
        shell = coordinator._suspended_derivations_by_id[derivation_id]
        if state.done:
            assert shell.status in terminal_statuses
    for derivation_id, state in coordinator._completer_derivation_states_by_id.items():
        coordinator._assert_derivation_invariant(derivation_id)
        shell = coordinator._suspended_derivations_by_id[derivation_id]
        if state.done:
            assert shell.status in terminal_statuses


# gabion:behavior primary=desired
def test_requeue_derivation_rejects_non_runnable_shell_status() -> None:
    module = _load_module()
    coordinator, _, root_derivation = _seed_root_runtime(module)

    with module.deadline_scope(module.Deadline.from_timeout_ms(30_000)):
        with module.deadline_clock_scope(module.MonotonicClock()):
            proposal = coordinator._step_completer_derivation(root_derivation.derivation_id.prime)
            assert isinstance(proposal, module.PredictionWait)
            coordinator._handle_completer_proposal(root_derivation.derivation_id.prime, proposal)

    with pytest.raises(ValueError, match="requires runnable shell status before queueing") as exc_info:
        coordinator._requeue_derivation(root_derivation.derivation_id)

    assert str(root_derivation.derivation_id.prime) in str(exc_info.value)
    assert module.DerivationStatus.WAITING_PREDICTION_COVER in str(exc_info.value)


# gabion:behavior primary=desired
def test_derivation_invariant_rejects_multiple_wait_set_registrations() -> None:
    module = _load_module()
    coordinator, _, root_derivation = _seed_root_runtime(module)

    with module.deadline_scope(module.Deadline.from_timeout_ms(30_000)):
        with module.deadline_clock_scope(module.MonotonicClock()):
            proposal = coordinator._step_completer_derivation(root_derivation.derivation_id.prime)
            assert isinstance(proposal, module.PredictionWait)
            coordinator._handle_completer_proposal(root_derivation.derivation_id.prime, proposal)

    root_live = next(
        derivation
        for derivation in coordinator.completer_stage.suspended_derivations
        if derivation.derivation_id == root_derivation.derivation_id
    )
    assert isinstance(root_live.wait_condition, module.PredictionCover)
    coordinator._waiting_derivation_ids_by_cover.setdefault(
        module.ScannerCover(proposal.demand_id),
        set(),
    ).add(root_derivation.derivation_id.prime)

    with pytest.raises(ValueError, match="wait set membership") as exc_info:
        coordinator._assert_derivation_invariant(root_derivation.derivation_id.prime)

    assert str(root_live.wait_condition) in str(exc_info.value)


# gabion:behavior primary=desired
def test_run_turtle_cover_poset_skeleton_unsatisfied_for_missing_symbol() -> None:
    module = _load_module()

    result = module.run_turtle_cover_poset_skeleton(
        symbol="missing_symbol",
        max_tokens=96,
    )

    assert result.selected_witness is None
    assert isinstance(result.closure_state, module.Unsatisfied)
    assert result.final_global_stage.completer.completed_witnesses == ()
    assert result.root_completion_demand.demand_id in (
        result.final_global_stage.completer.unsatisfied_completion_demand_ids
    )
    assert result.final_global_stage.predictor.closed_prediction_demand_ids
    assert result.final_global_stage.scanner.closed_scan_demand_ids
    assert any(
        isinstance(trace.proposal, module.ScanWait)
        for trace in result.proposal_traces
    )
    assert any(
        isinstance(trace.proposal, module.PredictionWait)
        for trace in result.proposal_traces
    )
    assert any(
        isinstance(trace.proposal, module.LocallyQuiescent)
        for trace in result.proposal_traces
    )
    assert any(
        trace.service_name == "scanner" and trace.reason == "scanner_demand_closed"
        for trace in result.cover_traces
    )
    assert any(
        trace.service_name == "predictor" and trace.reason == "prediction_demand_closed"
        for trace in result.cover_traces
    )
