from __future__ import annotations

"""Cover-predicate Earley kernel over the repo's kernel TTL files.

This experiment keeps scanner eager and stage-authoritative while predictor and
completer run as derivational fixed-point participants:

- scanner owns live observational truth, ambiguity classes, and scan closure
- predictor owns live candidate truth, predictor derivations, and prediction closure
- completer owns live completion demands, completion closure, suspended
  derivations, witnesses, and supports

Suspended derivations wait on explicit cover predicates. Wait and quiescence
objects remain observable trace projections rather than coordination inputs.
`GlobalStage` is the derived product surface over the primitive service stages.
"""

from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
import argparse
import json

from gabion.analysis.core.prime_identity_adapter import PrimeIdentityAdapter
from gabion.analysis.core.type_fingerprints import PrimeAssignmentEvent, PrimeRegistry
from gabion.analysis.foundation.identity_space import GlobalIdentitySpace
from gabion.analysis.foundation.timeout_context import (
    Deadline,
    deadline_clock_scope,
    deadline_scope,
)
from gabion.deadline_clock import MonotonicClock

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_TTL_PATHS = (
    Path("in/lg_kernel_ontology_cut_elim-1.ttl"),
    Path("in/lg_kernel_shapes_cut_elim-1.ttl"),
    Path("in/lg_kernel_example_cut_elim-1.ttl"),
)


@dataclass(frozen=True)
class TurtleLexeme:
    rel_path: str
    offset: int
    kind: str
    text: str

    @property
    def terminal_name(self) -> str:
        if self.text == "@prefix":
            return "PREFIX"
        if self.text == "a":
            return "A"
        if self.text == ".":
            return "DOT"
        if self.text == ";":
            return "SEMICOLON"
        if self.text == ",":
            return "COMMA"
        if self.text == "[":
            return "LBRACK"
        if self.text == "]":
            return "RBRACK"
        return self.kind


@dataclass(frozen=True)
class PrimeFactor:
    prime: int
    previous: PrimeFactor | int = 1
    namespace: str = ""
    token: str = ""

    def __post_init__(self) -> None:
        if self.prime < 2:
            raise ValueError("PrimeFactor.prime must be >= 2.")
        previous = self.previous
        if isinstance(previous, int):
            if previous != 1:
                raise ValueError("PrimeFactor.previous must be 1 or another PrimeFactor.")
            return
        if previous.prime >= self.prime:
            raise ValueError("PrimeFactor.previous must point to a lower prime.")

    def __hash__(self) -> int:
        return hash(self.prime)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PrimeFactor) and self.prime == other.prime

    def iter_chain(self) -> Iterator[PrimeFactor]:
        current: PrimeFactor | int = self
        while current != 1:
            if isinstance(current, PrimeFactor):
                yield current
                current = current.previous
                continue
            raise ValueError("PrimeFactor chain must terminate at 1.")


@dataclass(frozen=True)
class EarleyRule:
    head: PrimeFactor
    rhs: tuple[PrimeFactor, ...]

    def __post_init__(self) -> None:
        if len(self.rhs) not in (1, 2):
            raise ValueError("EarleyRule must be strict CNF-2: rhs length must be 1 or 2.")

    @property
    def is_binary(self) -> bool:
        return len(self.rhs) == 2

    @property
    def is_unary(self) -> bool:
        return len(self.rhs) == 1


@dataclass(frozen=True)
class ConsumerPolicy:
    policy_id: PrimeFactor
    mode: str = "first"

    def __post_init__(self) -> None:
        if self.mode != "first":
            raise ValueError("cover sibling kernel currently supports only ConsumerPolicy(mode='first').")


@dataclass(frozen=True)
class CompletionDemand:
    demand_id: PrimeFactor
    symbol: PrimeFactor
    start: int | None
    end: int | None
    require_complete: bool = True


@dataclass(frozen=True)
class PredictionDemand:
    demand_id: PrimeFactor
    symbol: PrimeFactor
    start: int | None
    end: int | None


@dataclass(frozen=True)
class ScanDemand:
    demand_id: PrimeFactor
    start: int | None
    end: int | None
    expected_symbols: tuple[PrimeFactor, ...]


@dataclass(frozen=True)
class ScannerCover:
    scan_demand_id: PrimeFactor


@dataclass(frozen=True)
class PredictionCover:
    prediction_demand_id: PrimeFactor


@dataclass(frozen=True)
class CompletionResolved:
    completion_demand_id: PrimeFactor


@dataclass(frozen=True)
class CompletionSatisfied:
    completion_demand_id: PrimeFactor


CoverCondition = ScannerCover | PredictionCover | CompletionResolved | CompletionSatisfied


class DerivationStatus(StrEnum):
    SEEDED = "seeded"
    WAITING_SCANNER_COVER = "waiting_scanner_cover"
    PROCESSING_SCANNER_TRUTHS = "processing_scanner_truths"
    AWAKENED = "awakened"
    CLOSED = "closed"
    WAITING_PREDICTION_COVER = "waiting_prediction_cover"
    EXAMINING_CANDIDATES = "examining_candidates"
    SPAWNED_CHILD_DEMAND = "spawned_child_demand"
    WAITING_COMPLETION_COVER = "waiting_completion_cover"
    DONE = "done"
    UNSATISFIED = "unsatisfied"


_TERMINAL_DERIVATION_STATUSES = frozenset(
    {
        DerivationStatus.DONE,
        DerivationStatus.UNSATISFIED,
        DerivationStatus.CLOSED,
    }
)
_WAITING_DERIVATION_STATUSES = frozenset(
    {
        DerivationStatus.WAITING_SCANNER_COVER,
        DerivationStatus.WAITING_PREDICTION_COVER,
        DerivationStatus.WAITING_COMPLETION_COVER,
    }
)
_RUNNABLE_DERIVATION_STATUSES = frozenset(
    {
        DerivationStatus.SEEDED,
        DerivationStatus.PROCESSING_SCANNER_TRUTHS,
        DerivationStatus.AWAKENED,
        DerivationStatus.EXAMINING_CANDIDATES,
        DerivationStatus.SPAWNED_CHILD_DEMAND,
    }
)


@dataclass(frozen=True)
class ScanWait:
    wait_id: PrimeFactor
    requester_derivation_id: PrimeFactor | None
    demand_id: PrimeFactor
    parent_prediction_demand_id: PrimeFactor


@dataclass(frozen=True)
class PredictionWait:
    wait_id: PrimeFactor
    requester_derivation_id: PrimeFactor
    demand_id: PrimeFactor
    parent_completion_demand_id: PrimeFactor


@dataclass(frozen=True)
class CompletionWait:
    wait_id: PrimeFactor
    requester_derivation_id: PrimeFactor
    demand_id: PrimeFactor
    parent_completion_demand_id: PrimeFactor
    parent_candidate_id: PrimeFactor


@dataclass(frozen=True)
class LocallyQuiescent:
    quiescence_id: PrimeFactor
    service_name: str
    derivation_id: PrimeFactor
    waiting_on_demand_id: PrimeFactor | None
    reason: str


@dataclass(frozen=True)
class GloballyClosed:
    closure_id: PrimeFactor
    reason: str


@dataclass(frozen=True)
class Unsatisfied:
    unsatisfied_id: PrimeFactor
    service_name: str
    demand_id: PrimeFactor
    reason: str


@dataclass(frozen=True)
class ScannerTruth:
    truth_id: PrimeFactor
    site_id: PrimeFactor
    ambiguity_class_id: PrimeFactor
    symbol: PrimeFactor
    rel_path_id: PrimeFactor
    start: int
    stop: int
    lexeme: TurtleLexeme


@dataclass(frozen=True)
class ScannerAmbiguityClass:
    ambiguity_class_id: PrimeFactor
    site_id: PrimeFactor
    truth_ids: tuple[PrimeFactor, ...]
    start: int
    stop: int
    lexeme_text: str


@dataclass(frozen=True)
class PredictedCandidate:
    candidate_id: PrimeFactor
    demand_id: PrimeFactor
    rule: EarleyRule
    anchor_truth_id: PrimeFactor
    anchor_symbol: PrimeFactor
    start: int
    stop: int
    residual_symbol: PrimeFactor | None


@dataclass(frozen=True)
class CompletedWitness:
    witness_id: PrimeFactor
    demand_id: PrimeFactor
    symbol: PrimeFactor
    start: int
    stop: int
    rule: EarleyRule
    candidate_id: PrimeFactor
    anchor_truth_id: PrimeFactor
    right_witness_id: PrimeFactor | None


@dataclass(frozen=True)
class CompletionSupport:
    support_id: PrimeFactor
    demand_id: PrimeFactor
    candidate_id: PrimeFactor
    child_completion_demand_id: PrimeFactor | None
    child_witness_id: PrimeFactor | None
    kind: str


@dataclass(frozen=True)
class SuspendedDerivation:
    derivation_id: PrimeFactor
    service_name: str
    demand_id: PrimeFactor
    status: DerivationStatus
    wait_condition: CoverCondition | None
    parent_derivation_id: PrimeFactor | None
    parent_candidate_id: PrimeFactor | None


@dataclass(frozen=True)
class ScannerStage:
    stage_id: PrimeFactor
    horizon: int
    scan_demand_ids: frozenset[PrimeFactor]
    closed_scan_demand_ids: frozenset[PrimeFactor]
    truth_ids: frozenset[PrimeFactor]
    ambiguity_class_ids: frozenset[PrimeFactor]
    scan_demands: tuple[ScanDemand, ...]
    truths: tuple[ScannerTruth, ...]
    ambiguity_classes: tuple[ScannerAmbiguityClass, ...]

    def extends(self, other: ScannerStage) -> bool:
        return (
            self.horizon >= other.horizon
            and other.scan_demand_ids <= self.scan_demand_ids
            and other.closed_scan_demand_ids <= self.closed_scan_demand_ids
            and other.truth_ids <= self.truth_ids
            and other.ambiguity_class_ids <= self.ambiguity_class_ids
        )


@dataclass(frozen=True)
class PredictorStage:
    stage_id: PrimeFactor
    prediction_demand_ids: frozenset[PrimeFactor]
    closed_prediction_demand_ids: frozenset[PrimeFactor]
    derivation_ids: frozenset[PrimeFactor]
    candidate_ids: frozenset[PrimeFactor]
    prediction_demands: tuple[PredictionDemand, ...]
    suspended_derivations: tuple[SuspendedDerivation, ...]
    candidates: tuple[PredictedCandidate, ...]

    def extends(self, other: PredictorStage) -> bool:
        return (
            other.prediction_demand_ids <= self.prediction_demand_ids
            and other.closed_prediction_demand_ids <= self.closed_prediction_demand_ids
            and other.derivation_ids <= self.derivation_ids
            and other.candidate_ids <= self.candidate_ids
        )


@dataclass(frozen=True)
class CompleterStage:
    stage_id: PrimeFactor
    completion_demand_ids: frozenset[PrimeFactor]
    discharged_completion_demand_ids: frozenset[PrimeFactor]
    unsatisfied_completion_demand_ids: frozenset[PrimeFactor]
    derivation_ids: frozenset[PrimeFactor]
    witness_ids: frozenset[PrimeFactor]
    support_ids: frozenset[PrimeFactor]
    completion_demands: tuple[CompletionDemand, ...]
    discharged_completion_demands: tuple[CompletionDemand, ...]
    unsatisfied_completion_demands: tuple[CompletionDemand, ...]
    suspended_derivations: tuple[SuspendedDerivation, ...]
    completed_witnesses: tuple[CompletedWitness, ...]
    supports: tuple[CompletionSupport, ...]

    def extends(self, other: CompleterStage) -> bool:
        return (
            other.completion_demand_ids <= self.completion_demand_ids
            and other.discharged_completion_demand_ids <= self.discharged_completion_demand_ids
            and other.unsatisfied_completion_demand_ids <= self.unsatisfied_completion_demand_ids
            and other.derivation_ids <= self.derivation_ids
            and other.witness_ids <= self.witness_ids
            and other.support_ids <= self.support_ids
        )


@dataclass(frozen=True)
class GlobalStage:
    stage_id: PrimeFactor
    root_demand_id: PrimeFactor
    scanner: ScannerStage
    predictor: PredictorStage
    completer: CompleterStage

    def extends(self, other: GlobalStage) -> bool:
        return (
            self.scanner.extends(other.scanner)
            and self.predictor.extends(other.predictor)
            and self.completer.extends(other.completer)
        )


@dataclass(frozen=True)
class ScannerDelta:
    delta_id: PrimeFactor
    new_scan_demands: tuple[ScanDemand, ...]
    new_truths: tuple[ScannerTruth, ...]
    new_ambiguity_classes: tuple[ScannerAmbiguityClass, ...]
    new_closed_scan_demand_ids: tuple[PrimeFactor, ...]
    new_horizon: int | None


@dataclass(frozen=True)
class PredictorDelta:
    delta_id: PrimeFactor
    new_prediction_demands: tuple[PredictionDemand, ...]
    new_suspended_derivations: tuple[SuspendedDerivation, ...]
    new_candidates: tuple[PredictedCandidate, ...]
    new_closed_prediction_demand_ids: tuple[PrimeFactor, ...]


@dataclass(frozen=True)
class CompleterDelta:
    delta_id: PrimeFactor
    new_completion_demands: tuple[CompletionDemand, ...]
    new_discharged_completion_demand_ids: tuple[PrimeFactor, ...]
    new_suspended_derivations: tuple[SuspendedDerivation, ...]
    new_completed_witnesses: tuple[CompletedWitness, ...]
    new_supports: tuple[CompletionSupport, ...]


@dataclass(frozen=True)
class ProposalTrace:
    """Observability surface for derivation waits, quiescence, conclusions, and deltas."""

    trace_id: PrimeFactor
    service_name: str
    derivation_id: PrimeFactor | None
    stage: ScannerStage | PredictorStage | CompleterStage
    proposal: (
        ScanWait
        | PredictionWait
        | CompletionWait
        | ScannerDelta
        | PredictorDelta
        | CompleterDelta
        | LocallyQuiescent
        | Unsatisfied
    )


@dataclass(frozen=True)
class CoverTransitionTrace:
    trace_id: PrimeFactor
    service_name: str
    demand_id: PrimeFactor
    awakened_derivation_ids: tuple[PrimeFactor, ...]
    triggering_delta_id: PrimeFactor | None
    stage_id: PrimeFactor
    reason: str


@dataclass(frozen=True)
class CoverPosetResult:
    source_paths: tuple[str, ...]
    lexemes: tuple[TurtleLexeme, ...]
    grammar: tuple[EarleyRule, ...]
    root_completion_demand: CompletionDemand
    consumer_policy: ConsumerPolicy
    final_global_stage: GlobalStage
    selected_witness: CompletedWitness | None
    closure_state: GloballyClosed | Unsatisfied | None
    proposal_traces: tuple[ProposalTrace, ...]
    cover_traces: tuple[CoverTransitionTrace, ...]
    last_joined_delta: ScannerDelta | PredictorDelta | CompleterDelta | None

    def coverable_completion_demands(self) -> tuple[CompletionDemand, ...]:
        return self.final_global_stage.completer.discharged_completion_demands

    def as_summary(self) -> dict[str, object]:
        ambiguity_samples: list[dict[str, object]] = []
        for ambiguity in self.final_global_stage.scanner.ambiguity_classes:
            if len(ambiguity.truth_ids) < 2:
                continue
            truths = [
                truth
                for truth in self.final_global_stage.scanner.truths
                if truth.truth_id in set(ambiguity.truth_ids)
            ]
            ambiguity_samples.append(
                {
                    "ambiguity_class_prime": ambiguity.ambiguity_class_id.prime,
                    "site": [ambiguity.start, ambiguity.stop],
                    "lexeme_text": ambiguity.lexeme_text,
                    "symbols": [truth.symbol.token for truth in truths],
                }
            )
            if len(ambiguity_samples) == 5:
                break

        predicted_samples = [
            {
                "candidate_prime": candidate.candidate_id.prime,
                "head": candidate.rule.head.token,
                "start": candidate.start,
                "stop": candidate.stop,
                "residual": None
                if candidate.residual_symbol is None
                else candidate.residual_symbol.token,
            }
            for candidate in self.final_global_stage.predictor.candidates[:5]
        ]
        completed_samples = [
            {
                "witness_prime": witness.witness_id.prime,
                "symbol": witness.symbol.token,
                "start": witness.start,
                "stop": witness.stop,
            }
            for witness in self.final_global_stage.completer.completed_witnesses[:5]
        ]
        return {
            "source_paths": list(self.source_paths),
            "lexeme_count": len(self.lexemes),
            "grammar_rule_count": len(self.grammar),
            "query_symbol": self.root_completion_demand.symbol.token,
            "query_start": self.root_completion_demand.start,
            "query_end": self.root_completion_demand.end,
            "consumer_policy": self.consumer_policy.mode,
            "found_witness": self.selected_witness is not None,
            "witness_span": None
            if self.selected_witness is None
            else [self.selected_witness.start, self.selected_witness.stop],
            "closure_state": None
            if self.closure_state is None
            else self.closure_state.__class__.__name__,
            "proposal_trace_count": len(self.proposal_traces),
            "cover_trace_count": len(self.cover_traces),
            "last_joined_delta_type": None
            if self.last_joined_delta is None
            else self.last_joined_delta.__class__.__name__,
            "sample_ambiguity_classes": ambiguity_samples,
            "sample_predicted_candidates": predicted_samples,
            "sample_completed_witnesses": completed_samples,
        }


RUNTIME_CLASSIFICATION = {
    "CompletionDemand": "semantic demand",
    "PredictionDemand": "semantic demand",
    "ScanDemand": "semantic demand",
    "ScannerCover": "cover predicate",
    "PredictionCover": "cover predicate",
    "CompletionResolved": "cover predicate",
    "CompletionSatisfied": "cover predicate",
    "DerivationStatus": "suspended derivation status",
    "ConsumerPolicy": "consumer policy",
    "ScannerTruth": "live stage truth",
    "ScannerAmbiguityClass": "live stage truth",
    "PredictedCandidate": "live stage truth",
    "CompletedWitness": "live stage truth",
    "CompletionSupport": "live stage truth",
    "ScannerStage": "live stage truth",
    "PredictorStage": "live stage truth",
    "CompleterStage": "live stage truth",
    "GlobalStage": "derived live product state",
    "SuspendedDerivation": "suspended derivation",
    "ScannerDelta": "delta proposal",
    "PredictorDelta": "delta proposal",
    "CompleterDelta": "delta proposal",
    "ScanWait": "provenance only",
    "PredictionWait": "provenance only",
    "CompletionWait": "provenance only",
    "LocallyQuiescent": "provenance only",
    "GloballyClosed": "coordinator artifact",
    "Unsatisfied": "closure truth",
    "FixedPointCoordinator": "coordinator artifact",
    "ProposalTrace": "provenance only",
    "CoverTransitionTrace": "provenance only",
}


@dataclass
class _PrimeFactorSpace:
    registry: PrimeRegistry
    _factors_by_prime: dict[int, PrimeFactor]
    _last_factor: PrimeFactor | int

    def __init__(self, registry: PrimeRegistry) -> None:
        self.registry = registry
        self._factors_by_prime = {}
        self._last_factor = 1

    def attach(self) -> int:
        return self.registry.register_assignment_observer(self._observe)

    def _observe(self, event: PrimeAssignmentEvent) -> None:
        factor = PrimeFactor(
            prime=event.atom_id,
            previous=self._last_factor,
            namespace=event.namespace,
            token=event.token,
        )
        self._factors_by_prime[event.atom_id] = factor
        self._last_factor = factor

    def factor_for(self, atom_id: int) -> PrimeFactor:
        factor = self._factors_by_prime.get(int(atom_id))
        if factor is None:
            raise ValueError(f"missing PrimeFactor for atom {atom_id}")
        return factor

    def intern_factor(
        self,
        *,
        identity_space: GlobalIdentitySpace,
        namespace: str,
        token: str,
    ) -> PrimeFactor:
        atom_id = identity_space.intern_atom(namespace=namespace, token=token)
        return self.factor_for(atom_id)


def _tokenize_turtle_text(*, rel_path: str, text: str) -> tuple[TurtleLexeme, ...]:
    tokens: list[TurtleLexeme] = []
    index = 0
    length = len(text)
    while index < length:
        if text[index].isspace():
            index += 1
            continue
        if text[index] == "#":
            while index < length and text[index] != "\n":
                index += 1
            continue
        if text.startswith('"""', index):
            end = text.find('"""', index + 3)
            if end < 0:
                raise ValueError(f"unterminated triple-quoted string in {rel_path}")
            tokens.append(
                TurtleLexeme(
                    rel_path=rel_path,
                    offset=index,
                    kind="STRING",
                    text=text[index : end + 3],
                )
            )
            index = end + 3
            continue
        if text.startswith("@prefix", index):
            tail = index + len("@prefix")
            if tail == length or text[tail].isspace():
                tokens.append(
                    TurtleLexeme(
                        rel_path=rel_path,
                        offset=index,
                        kind="DIRECTIVE",
                        text="@prefix",
                    )
                )
                index = tail
                continue
        char = text[index]
        if char in ".;,[]":
            tokens.append(
                TurtleLexeme(
                    rel_path=rel_path,
                    offset=index,
                    kind="PUNCT",
                    text=char,
                )
            )
            index += 1
            continue
        if char == "<":
            end = text.find(">", index + 1)
            if end < 0:
                raise ValueError(f"unterminated IRI in {rel_path}")
            tokens.append(
                TurtleLexeme(
                    rel_path=rel_path,
                    offset=index,
                    kind="IRI",
                    text=text[index : end + 1],
                )
            )
            index = end + 1
            continue
        if char == '"':
            cursor = index + 1
            escaped = False
            while cursor < length:
                current = text[cursor]
                if current == '"' and not escaped:
                    break
                escaped = current == "\\" and not escaped
                if current != "\\":
                    escaped = False
                cursor += 1
            if cursor >= length:
                raise ValueError(f"unterminated string in {rel_path}")
            tokens.append(
                TurtleLexeme(
                    rel_path=rel_path,
                    offset=index,
                    kind="STRING",
                    text=text[index : cursor + 1],
                )
            )
            index = cursor + 1
            continue
        if char.isdigit() or (char == "-" and index + 1 < length and text[index + 1].isdigit()):
            cursor = index + 1
            while cursor < length and text[cursor].isdigit():
                cursor += 1
            tokens.append(
                TurtleLexeme(
                    rel_path=rel_path,
                    offset=index,
                    kind="NUMBER",
                    text=text[index:cursor],
                )
            )
            index = cursor
            continue
        cursor = index
        while cursor < length and text[cursor] not in " \t\r\n#<\".;,[]":
            cursor += 1
        tokens.append(
            TurtleLexeme(
                rel_path=rel_path,
                offset=index,
                kind="NAME",
                text=text[index:cursor],
            )
        )
        index = cursor
    return tuple(tokens)


def load_kernel_ttl_lexemes(
    *,
    root: Path = _REPO_ROOT,
    rel_paths: tuple[Path, ...] = _DEFAULT_TTL_PATHS,
) -> tuple[TurtleLexeme, ...]:
    lexemes: list[TurtleLexeme] = []
    for rel_path in rel_paths:
        text = (root / rel_path).read_text(encoding="utf-8")
        lexemes.extend(_tokenize_turtle_text(rel_path=rel_path.as_posix(), text=text))
    return tuple(lexemes)


@dataclass
class CompleterDerivationState:
    derivation_id: PrimeFactor
    demand_id: PrimeFactor
    parent_derivation_id: PrimeFactor | None
    parent_candidate_id: PrimeFactor | None
    prediction_demand_id: PrimeFactor | None = None
    active_candidate_id: PrimeFactor | None = None
    child_completion_demand_id: PrimeFactor | None = None
    processed_candidate_ids: set[int] = field(default_factory=set)
    done: bool = False


@dataclass
class PredictorDerivationState:
    derivation_id: PrimeFactor
    demand_id: PrimeFactor
    scan_demand_id: PrimeFactor | None = None
    processed_truth_ids: set[int] = field(default_factory=set)
    done: bool = False


class FixedPointCoordinator:
    def __init__(
        self,
        *,
        source_paths: tuple[str, ...],
        lexemes: tuple[TurtleLexeme, ...],
        grammar: tuple[EarleyRule, ...],
        identity_space: GlobalIdentitySpace,
        prime_space: _PrimeFactorSpace,
        symbol_table: dict[str, PrimeFactor],
        max_steps: int,
    ) -> None:
        self.source_paths = source_paths
        self.lexemes = lexemes
        self.grammar = grammar
        self.identity_space = identity_space
        self.prime_space = prime_space
        self.symbol_table = symbol_table
        self.max_steps = max_steps

        self.rules_by_head: dict[PrimeFactor, tuple[EarleyRule, ...]] = {}
        for rule in grammar:
            self.rules_by_head[rule.head] = (*self.rules_by_head.get(rule.head, ()), rule)

        self._reset_runtime_state(root_demand_id=self._id_factor("ttl_cover_dummy_root", "0"))

    @classmethod
    def create(
        cls,
        *,
        root: Path = _REPO_ROOT,
        rel_paths: tuple[Path, ...] = _DEFAULT_TTL_PATHS,
        max_tokens: int | None = 256,
        max_steps: int = 10_000,
    ) -> FixedPointCoordinator:
        with deadline_scope(Deadline.from_timeout_ms(30_000)):
            with deadline_clock_scope(MonotonicClock()):
                lexemes = load_kernel_ttl_lexemes(root=root, rel_paths=rel_paths)
                if max_tokens is not None:
                    lexemes = lexemes[:max_tokens]

                registry = PrimeRegistry()
                prime_space = _PrimeFactorSpace(registry=registry)
                prime_space.attach()
                identity_space = GlobalIdentitySpace(allocator=PrimeIdentityAdapter(registry=registry))

                symbol_table: dict[str, PrimeFactor] = {}

                def symbol_factor(name: str) -> PrimeFactor:
                    factor = symbol_table.get(name)
                    if factor is None:
                        factor = prime_space.intern_factor(
                            identity_space=identity_space,
                            namespace="ttl_cover_symbol",
                            token=name,
                        )
                        symbol_table[name] = factor
                    return factor

                def terminal_factor(name: str) -> PrimeFactor:
                    return symbol_factor(f"terminal:{name}")

                document = symbol_factor("document'")
                directive = symbol_factor("directive")
                directive_tail = symbol_factor("directive_tail")
                iri_dot = symbol_factor("iri_dot")
                pred_name_name = symbol_factor("pred_name_name")
                pred_name_iri = symbol_factor("pred_name_iri")
                pred_name_string = symbol_factor("pred_name_string")
                pred_name_number = symbol_factor("pred_name_number")
                object_dot_name = symbol_factor("object_dot_name")
                object_dot_iri = symbol_factor("object_dot_iri")
                object_dot_string = symbol_factor("object_dot_string")
                object_dot_number = symbol_factor("object_dot_number")

                prefix = terminal_factor("PREFIX")
                name = terminal_factor("NAME")
                iri = terminal_factor("IRI")
                dot = terminal_factor("DOT")
                string = terminal_factor("STRING")
                number = terminal_factor("NUMBER")
                terminal_factor("A")

                grammar = (
                    EarleyRule(head=document, rhs=(directive, directive)),
                    EarleyRule(head=directive, rhs=(prefix, directive_tail)),
                    EarleyRule(head=directive_tail, rhs=(name, iri_dot)),
                    EarleyRule(head=iri_dot, rhs=(iri, dot)),
                    EarleyRule(head=pred_name_name, rhs=(name, object_dot_name)),
                    EarleyRule(head=pred_name_iri, rhs=(name, object_dot_iri)),
                    EarleyRule(head=pred_name_string, rhs=(name, object_dot_string)),
                    EarleyRule(head=pred_name_number, rhs=(name, object_dot_number)),
                    EarleyRule(head=object_dot_name, rhs=(name, dot)),
                    EarleyRule(head=object_dot_iri, rhs=(iri, dot)),
                    EarleyRule(head=object_dot_string, rhs=(string, dot)),
                    EarleyRule(head=object_dot_number, rhs=(number, dot)),
                )
                return cls(
                    source_paths=tuple(path.as_posix() for path in rel_paths),
                    lexemes=lexemes,
                    grammar=grammar,
                    identity_space=identity_space,
                    prime_space=prime_space,
                    symbol_table=symbol_table,
                    max_steps=max_steps,
                )

    def _reset_runtime_state(self, *, root_demand_id: PrimeFactor) -> None:
        self._scanner_demands_by_id: dict[int, ScanDemand] = {}
        self._scanner_truths_by_id: dict[int, ScannerTruth] = {}
        self._scanner_ambiguities_by_id: dict[int, ScannerAmbiguityClass] = {}
        self._scanner_truth_ids_by_demand_id: dict[int, list[int]] = {}
        self._closed_scan_demand_ids: set[int] = set()
        self._scanner_horizon = 0

        self._prediction_demands_by_id: dict[int, PredictionDemand] = {}
        self._candidates_by_id: dict[int, PredictedCandidate] = {}
        self._candidate_ids_by_demand_id: dict[int, list[int]] = {}
        self._closed_prediction_demand_ids: set[int] = set()
        self._scan_demand_id_by_prediction_demand_id: dict[int, int] = {}
        self._prediction_demand_id_to_derivation_id: dict[int, int] = {}

        self._completion_demands_by_id: dict[int, CompletionDemand] = {}
        self._discharged_completion_demand_ids: set[int] = set()
        self._completed_witnesses_by_id: dict[int, CompletedWitness] = {}
        self._witness_ids_by_demand_id: dict[int, list[int]] = {}
        self._supports_by_id: dict[int, CompletionSupport] = {}
        self._completion_demand_id_to_derivation_id: dict[int, int] = {}
        self._unsatisfied_completion_demand_ids: set[int] = set()

        self._suspended_derivations_by_id: dict[int, SuspendedDerivation] = {}
        self._predictor_derivation_states_by_id: dict[int, PredictorDerivationState] = {}
        self._completer_derivation_states_by_id: dict[int, CompleterDerivationState] = {}

        self._runnable_derivation_ids: deque[int] = deque()
        self._pending_deltas: deque[
            tuple[str, PrimeFactor | None, ScannerDelta | PredictorDelta | CompleterDelta]
        ] = deque()
        self._waiting_derivation_ids_by_cover: dict[CoverCondition, set[int]] = {}
        self._cover_by_waiting_derivation_id: dict[int, CoverCondition] = {}

        self.proposal_traces: list[ProposalTrace] = []
        self.cover_traces: list[CoverTransitionTrace] = []
        self.last_joined_delta: ScannerDelta | PredictorDelta | CompleterDelta | None = None

        self._scanner_revision = 0
        self._predictor_revision = 0
        self._completer_revision = 0
        self._global_revision = 0
        self._step_count = 0

        self.scanner_stage = ScannerStage(
            stage_id=self._id_factor("ttl_cover_scanner_stage", f"{root_demand_id.prime}|0"),
            horizon=0,
            scan_demand_ids=frozenset(),
            closed_scan_demand_ids=frozenset(),
            truth_ids=frozenset(),
            ambiguity_class_ids=frozenset(),
            scan_demands=(),
            truths=(),
            ambiguity_classes=(),
        )
        self.predictor_stage = PredictorStage(
            stage_id=self._id_factor("ttl_cover_predictor_stage", f"{root_demand_id.prime}|0"),
            prediction_demand_ids=frozenset(),
            closed_prediction_demand_ids=frozenset(),
            derivation_ids=frozenset(),
            candidate_ids=frozenset(),
            prediction_demands=(),
            suspended_derivations=(),
            candidates=(),
        )
        self.completer_stage = CompleterStage(
            stage_id=self._id_factor("ttl_cover_completer_stage", f"{root_demand_id.prime}|0"),
            completion_demand_ids=frozenset(),
            discharged_completion_demand_ids=frozenset(),
            unsatisfied_completion_demand_ids=frozenset(),
            derivation_ids=frozenset(),
            witness_ids=frozenset(),
            support_ids=frozenset(),
            completion_demands=(),
            discharged_completion_demands=(),
            unsatisfied_completion_demands=(),
            suspended_derivations=(),
            completed_witnesses=(),
            supports=(),
        )
        self.global_stage = GlobalStage(
            stage_id=self._id_factor("ttl_cover_global_stage", f"{root_demand_id.prime}|0"),
            root_demand_id=root_demand_id,
            scanner=self.scanner_stage,
            predictor=self.predictor_stage,
            completer=self.completer_stage,
        )

    def symbol_factor(self, name: str) -> PrimeFactor:
        factor = self.symbol_table.get(name)
        if factor is None:
            factor = self.prime_space.intern_factor(
                identity_space=self.identity_space,
                namespace="ttl_cover_symbol",
                token=name,
            )
            self.symbol_table[name] = factor
        return factor

    def make_consumer_policy(self, *, mode: str = "first") -> ConsumerPolicy:
        with deadline_scope(Deadline.from_timeout_ms(30_000)):
            with deadline_clock_scope(MonotonicClock()):
                return ConsumerPolicy(
                    policy_id=self._id_factor("ttl_cover_consumer_policy", mode),
                    mode=mode,
                )

    def make_completion_demand(
        self,
        *,
        symbol_name: str,
        start: int | None = None,
        end: int | None = None,
        require_complete: bool = True,
    ) -> CompletionDemand:
        with deadline_scope(Deadline.from_timeout_ms(30_000)):
            with deadline_clock_scope(MonotonicClock()):
                return self._make_completion_demand(
                    symbol=self.symbol_factor(symbol_name),
                    start=start,
                    end=end,
                    require_complete=require_complete,
                )

    def make_scan_demand(
        self,
        *,
        start: int | None = None,
        end: int | None = None,
        expected_symbol_names: tuple[str, ...] = (),
    ) -> ScanDemand:
        with deadline_scope(Deadline.from_timeout_ms(30_000)):
            with deadline_clock_scope(MonotonicClock()):
                expected_symbols = tuple(
                    sorted(
                        (self.symbol_factor(f"terminal:{name}") for name in expected_symbol_names),
                        key=lambda factor: factor.prime,
                    )
                )
                return self._make_scan_demand(
                    start=start,
                    end=end,
                    expected_symbols=expected_symbols,
                )

    def observe_scan_demand(self, demand: ScanDemand) -> ScannerStage:
        with deadline_scope(Deadline.from_timeout_ms(30_000)):
            with deadline_clock_scope(MonotonicClock()):
                self._reset_runtime_state(root_demand_id=demand.demand_id)
                self._enqueue_delta("scanner", None, self._scanner_delta_for_demand(demand))
                self._drain_pending_deltas()
                return self.global_stage.scanner

    def run_root_demand(
        self,
        root_demand: CompletionDemand,
        policy: ConsumerPolicy,
    ) -> CoverPosetResult:
        with deadline_scope(Deadline.from_timeout_ms(30_000)):
            with deadline_clock_scope(MonotonicClock()):
                self._reset_runtime_state(root_demand_id=root_demand.demand_id)
                root_derivation = self._new_completer_derivation(
                    demand=root_demand,
                    parent_derivation_id=None,
                    parent_candidate_id=None,
                )
                seed_delta = CompleterDelta(
                    delta_id=self._id_factor(
                        "ttl_cover_completer_delta",
                        f"seed|{root_demand.demand_id.prime}",
                    ),
                    new_completion_demands=(root_demand,),
                    new_discharged_completion_demand_ids=(),
                    new_suspended_derivations=(root_derivation,),
                    new_completed_witnesses=(),
                    new_supports=(),
                )
                self._enqueue_delta("completer", None, seed_delta)

                selected_witness: CompletedWitness | None = None
                closure_state: GloballyClosed | Unsatisfied | None = None
                while True:
                    self._drain_pending_deltas()
                    selected_witness = self._select_witness_for_policy(
                        demand=root_demand,
                        policy=policy,
                    )
                    if selected_witness is not None:
                        break
                    if self._runnable_derivation_ids:
                        derivation_id = self._runnable_derivation_ids.popleft()
                        if self._derivation_is_done(derivation_id):
                            continue
                        self._step_count += 1
                        if self._step_count > self.max_steps:
                            raise RuntimeError(
                                f"cover-poset skeleton exceeded max steps {self.max_steps}"
                            )
                        derivation = self._suspended_derivations_by_id[derivation_id]
                        if derivation.service_name == "predictor":
                            proposal = self._step_predictor_derivation(derivation_id)
                            if proposal is None:
                                continue
                            self._handle_predictor_proposal(derivation_id, proposal)
                            continue
                        if derivation.service_name == "completer":
                            proposal = self._step_completer_derivation(derivation_id)
                            if proposal is None:
                                continue
                            self._handle_completer_proposal(derivation_id, proposal)
                            continue
                        raise ValueError(f"unsupported derivation service {derivation.service_name}")
                    if self._pending_deltas:
                        continue
                    if self._completion_demand_is_unsatisfied(root_demand.demand_id):
                        closure_state = Unsatisfied(
                            unsatisfied_id=self._id_factor(
                                "ttl_cover_unsatisfied",
                                f"root|{root_demand.demand_id.prime}",
                            ),
                            service_name="completer",
                            demand_id=root_demand.demand_id,
                            reason="root_demand_unsatisfied",
                        )
                    else:
                        closure_state = GloballyClosed(
                            closure_id=self._id_factor(
                                "ttl_cover_globally_closed",
                                f"{root_demand.demand_id.prime}|{len(self.proposal_traces)}|{len(self.cover_traces)}",
                            ),
                            reason="no_pending_deltas_or_runnable_derivations",
                        )
                    break

                return CoverPosetResult(
                    source_paths=self.source_paths,
                    lexemes=self.lexemes,
                    grammar=self.grammar,
                    root_completion_demand=root_demand,
                    consumer_policy=policy,
                    final_global_stage=self.global_stage,
                    selected_witness=selected_witness,
                    closure_state=closure_state,
                    proposal_traces=tuple(self.proposal_traces),
                    cover_traces=tuple(self.cover_traces),
                    last_joined_delta=self.last_joined_delta,
                )

    def _drain_pending_deltas(self) -> None:
        while self._pending_deltas:
            service_name, proposer_id, delta = self._pending_deltas.popleft()
            if isinstance(delta, ScannerDelta):
                self._apply_scanner_delta(proposer_id, delta)
                self._wake_predictor_waiters_from_scanner_delta(delta)
            elif isinstance(delta, PredictorDelta):
                self._apply_predictor_delta(proposer_id, delta)
                self._wake_completer_waiters_from_predictor_delta(delta)
            elif isinstance(delta, CompleterDelta):
                self._apply_completer_delta(proposer_id, delta)
                self._wake_completer_waiters_from_completion_delta(delta)
            else:
                raise TypeError("unexpected delta type")
            self.last_joined_delta = delta

    def _enqueue_delta(
        self,
        service_name: str,
        proposer_id: PrimeFactor | None,
        delta: ScannerDelta | PredictorDelta | CompleterDelta,
    ) -> None:
        self._pending_deltas.append((service_name, proposer_id, delta))

    def _derivation_is_done(self, derivation_id: int) -> bool:
        derivation = self._suspended_derivations_by_id.get(derivation_id)
        if derivation is None:
            return True
        if derivation.service_name == "predictor":
            state = self._predictor_derivation_states_by_id[derivation_id]
            return state.done
        if derivation.service_name == "completer":
            state = self._completer_derivation_states_by_id[derivation_id]
            return state.done
        raise ValueError(f"unsupported derivation service {derivation.service_name}")

    def _private_derivation_state(
        self,
        derivation_id: int,
    ) -> PredictorDerivationState | CompleterDerivationState:
        derivation = self._suspended_derivations_by_id[derivation_id]
        if derivation.service_name == "predictor":
            return self._predictor_derivation_states_by_id[derivation_id]
        if derivation.service_name == "completer":
            return self._completer_derivation_states_by_id[derivation_id]
        raise ValueError(f"unsupported derivation service {derivation.service_name}")

    def _assert_derivation_invariant(self, derivation_id: int) -> None:
        derivation = self._suspended_derivations_by_id[derivation_id]
        private_state = self._private_derivation_state(derivation_id)
        status = derivation.status
        registered_cover = self._cover_by_waiting_derivation_id.get(derivation_id)
        wait_set_covers = [
            cover
            for cover, derivation_ids in self._waiting_derivation_ids_by_cover.items()
            if derivation_id in derivation_ids
        ]
        wait_set_cover_set = set(wait_set_covers)
        in_runnable_queue = derivation_id in self._runnable_derivation_ids

        if private_state.done and status not in _TERMINAL_DERIVATION_STATUSES:
            raise ValueError(
                f"derivation {derivation_id} done=True requires terminal status, got {status}"
            )
        if status in _TERMINAL_DERIVATION_STATUSES and not private_state.done:
            raise ValueError(
                f"derivation {derivation_id} terminal status {status} requires done=True"
            )
        if status in _WAITING_DERIVATION_STATUSES and derivation.wait_condition is None:
            raise ValueError(
                f"derivation {derivation_id} waiting status {status} requires wait_condition"
            )
        if status not in _WAITING_DERIVATION_STATUSES and derivation.wait_condition is not None:
            raise ValueError(
                f"derivation {derivation_id} non-waiting status {status} must not carry wait_condition"
            )
        if derivation.wait_condition is None:
            if registered_cover is not None or wait_set_covers:
                raise ValueError(
                    f"derivation {derivation_id} has stale wait registration without wait_condition"
                )
        else:
            if status not in _WAITING_DERIVATION_STATUSES:
                raise ValueError(
                    f"derivation {derivation_id} wait registration requires waiting status, got {status}"
                )
            if registered_cover != derivation.wait_condition:
                raise ValueError(
                    f"derivation {derivation_id} wait registration {registered_cover} disagrees with shell {derivation.wait_condition}"
                )
            if wait_set_cover_set != {derivation.wait_condition}:
                raise ValueError(
                    f"derivation {derivation_id} wait set membership {wait_set_covers} disagrees with shell {derivation.wait_condition}"
                )
        if in_runnable_queue and status not in _RUNNABLE_DERIVATION_STATUSES:
            raise ValueError(
                f"derivation {derivation_id} queued with non-runnable status {status}"
            )
        if status in _TERMINAL_DERIVATION_STATUSES and in_runnable_queue:
            raise ValueError(f"derivation {derivation_id} terminal status {status} cannot be queued")

    def _apply_scanner_delta(
        self,
        proposer_id: PrimeFactor | None,
        delta: ScannerDelta,
    ) -> None:
        changed = False
        owning_demand_id = (
            delta.new_scan_demands[0].demand_id.prime if delta.new_scan_demands else None
        )
        for demand in delta.new_scan_demands:
            if demand.demand_id.prime not in self._scanner_demands_by_id:
                self._scanner_demands_by_id[demand.demand_id.prime] = demand
                changed = True
        for truth in delta.new_truths:
            if truth.truth_id.prime in self._scanner_truths_by_id:
                continue
            self._scanner_truths_by_id[truth.truth_id.prime] = truth
            if owning_demand_id is not None:
                self._scanner_truth_ids_by_demand_id.setdefault(owning_demand_id, []).append(
                    truth.truth_id.prime
                )
            changed = True
        for ambiguity in delta.new_ambiguity_classes:
            if ambiguity.ambiguity_class_id.prime not in self._scanner_ambiguities_by_id:
                self._scanner_ambiguities_by_id[ambiguity.ambiguity_class_id.prime] = ambiguity
                changed = True
        for demand_id in delta.new_closed_scan_demand_ids:
            if demand_id.prime not in self._closed_scan_demand_ids:
                self._closed_scan_demand_ids.add(demand_id.prime)
                changed = True
        if delta.new_horizon is not None and delta.new_horizon > self._scanner_horizon:
            self._scanner_horizon = delta.new_horizon
            changed = True
        if not changed:
            return
        self._rebuild_scanner_stage()
        self._record_proposal_trace(
            service_name="scanner",
            derivation_id=proposer_id,
            stage=self.scanner_stage,
            proposal=delta,
        )

    def _apply_predictor_delta(
        self,
        proposer_id: PrimeFactor | None,
        delta: PredictorDelta,
    ) -> None:
        changed = False
        for demand in delta.new_prediction_demands:
            if demand.demand_id.prime not in self._prediction_demands_by_id:
                self._prediction_demands_by_id[demand.demand_id.prime] = demand
                changed = True
        for derivation in delta.new_suspended_derivations:
            if derivation.derivation_id.prime in self._suspended_derivations_by_id:
                continue
            self._suspended_derivations_by_id[derivation.derivation_id.prime] = derivation
            self._predictor_derivation_states_by_id[derivation.derivation_id.prime] = (
                PredictorDerivationState(
                    derivation_id=derivation.derivation_id,
                    demand_id=derivation.demand_id,
                )
            )
            self._prediction_demand_id_to_derivation_id.setdefault(
                derivation.demand_id.prime,
                derivation.derivation_id.prime,
            )
            self._requeue_derivation(derivation.derivation_id)
            changed = True
        for candidate in delta.new_candidates:
            if candidate.candidate_id.prime in self._candidates_by_id:
                continue
            self._candidates_by_id[candidate.candidate_id.prime] = candidate
            self._candidate_ids_by_demand_id.setdefault(candidate.demand_id.prime, []).append(
                candidate.candidate_id.prime
            )
            changed = True
        for demand_id in delta.new_closed_prediction_demand_ids:
            if demand_id.prime not in self._closed_prediction_demand_ids:
                self._closed_prediction_demand_ids.add(demand_id.prime)
                changed = True
        if not changed:
            return
        self._rebuild_predictor_stage()
        self._record_proposal_trace(
            service_name="predictor",
            derivation_id=proposer_id,
            stage=self.predictor_stage,
            proposal=delta,
        )

    def _apply_completer_delta(
        self,
        proposer_id: PrimeFactor | None,
        delta: CompleterDelta,
    ) -> None:
        changed = False
        for demand in delta.new_completion_demands:
            if demand.demand_id.prime not in self._completion_demands_by_id:
                self._completion_demands_by_id[demand.demand_id.prime] = demand
                changed = True
        for demand_id in delta.new_discharged_completion_demand_ids:
            if demand_id.prime not in self._discharged_completion_demand_ids:
                self._discharged_completion_demand_ids.add(demand_id.prime)
                changed = True
        for derivation in delta.new_suspended_derivations:
            if derivation.derivation_id.prime in self._suspended_derivations_by_id:
                continue
            self._suspended_derivations_by_id[derivation.derivation_id.prime] = derivation
            self._completer_derivation_states_by_id[derivation.derivation_id.prime] = (
                CompleterDerivationState(
                    derivation_id=derivation.derivation_id,
                    demand_id=derivation.demand_id,
                    parent_derivation_id=derivation.parent_derivation_id,
                    parent_candidate_id=derivation.parent_candidate_id,
                )
            )
            self._completion_demand_id_to_derivation_id.setdefault(
                derivation.demand_id.prime,
                derivation.derivation_id.prime,
            )
            self._requeue_derivation(derivation.derivation_id)
            changed = True
        for witness in delta.new_completed_witnesses:
            if witness.witness_id.prime in self._completed_witnesses_by_id:
                continue
            self._completed_witnesses_by_id[witness.witness_id.prime] = witness
            self._witness_ids_by_demand_id.setdefault(witness.demand_id.prime, []).append(
                witness.witness_id.prime
            )
            changed = True
        for support in delta.new_supports:
            if support.support_id.prime not in self._supports_by_id:
                self._supports_by_id[support.support_id.prime] = support
                changed = True
        if not changed:
            return
        self._rebuild_completer_stage()
        self._record_proposal_trace(
            service_name="completer",
            derivation_id=proposer_id,
            stage=self.completer_stage,
            proposal=delta,
        )

    def _rebuild_scanner_stage(self) -> None:
        self._scanner_revision += 1
        self.scanner_stage = ScannerStage(
            stage_id=self._id_factor(
                "ttl_cover_scanner_stage",
                f"{self.global_stage.root_demand_id.prime}|{self._scanner_revision}",
            ),
            horizon=self._scanner_horizon,
            scan_demand_ids=frozenset(
                demand.demand_id for demand in self._scanner_demands_by_id.values()
            ),
            closed_scan_demand_ids=frozenset(
                self._scanner_demands_by_id[demand_id].demand_id
                for demand_id in self._closed_scan_demand_ids
                if demand_id in self._scanner_demands_by_id
            ),
            truth_ids=frozenset(truth.truth_id for truth in self._scanner_truths_by_id.values()),
            ambiguity_class_ids=frozenset(
                ambiguity.ambiguity_class_id
                for ambiguity in self._scanner_ambiguities_by_id.values()
            ),
            scan_demands=tuple(self._scanner_demands_by_id.values()),
            truths=tuple(self._scanner_truths_by_id.values()),
            ambiguity_classes=tuple(self._scanner_ambiguities_by_id.values()),
        )
        self._refresh_global_stage()

    def _rebuild_predictor_stage(self) -> None:
        self._predictor_revision += 1
        suspended_derivations = tuple(
            derivation
            for derivation in self._suspended_derivations_by_id.values()
            if derivation.service_name == "predictor"
        )
        self.predictor_stage = PredictorStage(
            stage_id=self._id_factor(
                "ttl_cover_predictor_stage",
                f"{self.global_stage.root_demand_id.prime}|{self._predictor_revision}",
            ),
            prediction_demand_ids=frozenset(
                demand.demand_id for demand in self._prediction_demands_by_id.values()
            ),
            closed_prediction_demand_ids=frozenset(
                self._prediction_demands_by_id[demand_id].demand_id
                for demand_id in self._closed_prediction_demand_ids
                if demand_id in self._prediction_demands_by_id
            ),
            derivation_ids=frozenset(
                derivation.derivation_id for derivation in suspended_derivations
            ),
            candidate_ids=frozenset(
                candidate.candidate_id for candidate in self._candidates_by_id.values()
            ),
            prediction_demands=tuple(self._prediction_demands_by_id.values()),
            suspended_derivations=suspended_derivations,
            candidates=tuple(self._candidates_by_id.values()),
        )
        self._refresh_global_stage()

    def _rebuild_completer_stage(self) -> None:
        self._completer_revision += 1
        suspended_derivations = tuple(
            derivation
            for derivation in self._suspended_derivations_by_id.values()
            if derivation.service_name == "completer"
        )
        discharged_demands = tuple(
            self._completion_demands_by_id[demand_id]
            for demand_id in self._discharged_completion_demand_ids
            if demand_id in self._completion_demands_by_id
        )
        unsatisfied_demands = tuple(
            self._completion_demands_by_id[demand_id]
            for demand_id in self._unsatisfied_completion_demand_ids
            if demand_id in self._completion_demands_by_id
        )
        self.completer_stage = CompleterStage(
            stage_id=self._id_factor(
                "ttl_cover_completer_stage",
                f"{self.global_stage.root_demand_id.prime}|{self._completer_revision}",
            ),
            completion_demand_ids=frozenset(
                demand.demand_id for demand in self._completion_demands_by_id.values()
            ),
            discharged_completion_demand_ids=frozenset(
                demand.demand_id for demand in discharged_demands
            ),
            unsatisfied_completion_demand_ids=frozenset(
                demand.demand_id for demand in unsatisfied_demands
            ),
            derivation_ids=frozenset(
                derivation.derivation_id for derivation in suspended_derivations
            ),
            witness_ids=frozenset(
                witness.witness_id for witness in self._completed_witnesses_by_id.values()
            ),
            support_ids=frozenset(
                support.support_id for support in self._supports_by_id.values()
            ),
            completion_demands=tuple(self._completion_demands_by_id.values()),
            discharged_completion_demands=discharged_demands,
            unsatisfied_completion_demands=unsatisfied_demands,
            suspended_derivations=suspended_derivations,
            completed_witnesses=tuple(self._completed_witnesses_by_id.values()),
            supports=tuple(self._supports_by_id.values()),
        )
        self._refresh_global_stage()

    def _refresh_global_stage(self) -> None:
        self._global_revision += 1
        self.global_stage = GlobalStage(
            stage_id=self._id_factor(
                "ttl_cover_global_stage",
                f"{self.global_stage.root_demand_id.prime}|{self._global_revision}",
            ),
            root_demand_id=self.global_stage.root_demand_id,
            scanner=self.scanner_stage,
            predictor=self.predictor_stage,
            completer=self.completer_stage,
        )

    def _step_predictor_derivation(
        self,
        derivation_id: int,
    ) -> ScanWait | PredictorDelta | None:
        local_state = self._predictor_derivation_states_by_id[derivation_id]
        if local_state.done:
            return None
        demand = self._prediction_demands_by_id[local_state.demand_id.prime]

        scan_demand = self._scan_demand_for_prediction_demand(demand)
        local_state.scan_demand_id = scan_demand.demand_id
        if scan_demand.demand_id.prime not in self._scanner_demands_by_id:
            self._enqueue_delta(
                "scanner",
                local_state.derivation_id,
                self._scanner_delta_for_demand(scan_demand),
            )

        new_truths = [
            truth
            for truth in self._scanner_truths_for_scan_demand(scan_demand.demand_id)
            if truth.truth_id.prime not in local_state.processed_truth_ids
        ]
        if not new_truths:
            if scan_demand.demand_id.prime in self._closed_scan_demand_ids:
                local_state.done = True
                self._set_derivation_runtime_state(
                    derivation_id=derivation_id,
                    status=DerivationStatus.CLOSED,
                    wait_condition=None,
                )
                if demand.demand_id.prime in self._closed_prediction_demand_ids:
                    return None
                return PredictorDelta(
                    delta_id=self._id_factor(
                        "ttl_cover_predictor_delta",
                        f"closed|{demand.demand_id.prime}",
                    ),
                    new_prediction_demands=(),
                    new_suspended_derivations=(),
                    new_candidates=(),
                    new_closed_prediction_demand_ids=(demand.demand_id,),
                )
            self._set_derivation_runtime_state(
                derivation_id=derivation_id,
                status=DerivationStatus.WAITING_SCANNER_COVER,
                wait_condition=ScannerCover(scan_demand.demand_id),
            )
            return ScanWait(
                wait_id=self._id_factor(
                    "ttl_cover_scan_wait",
                    f"{local_state.derivation_id.prime}|{scan_demand.demand_id.prime}",
                ),
                requester_derivation_id=local_state.derivation_id,
                demand_id=scan_demand.demand_id,
                parent_prediction_demand_id=demand.demand_id,
            )

        new_candidates: list[PredictedCandidate] = []
        for truth in new_truths:
            local_state.processed_truth_ids.add(truth.truth_id.prime)
            for candidate in self._predictor_candidates_for_truth(demand=demand, truth=truth):
                if candidate.candidate_id.prime in self._candidates_by_id:
                    continue
                new_candidates.append(candidate)

        close_ids: tuple[PrimeFactor, ...] = ()
        if scan_demand.demand_id.prime in self._closed_scan_demand_ids:
            local_state.done = True
            self._set_derivation_runtime_state(
                derivation_id=derivation_id,
                status=DerivationStatus.CLOSED,
                wait_condition=None,
            )
            if demand.demand_id.prime not in self._closed_prediction_demand_ids:
                close_ids = (demand.demand_id,)
        else:
            self._set_derivation_runtime_state(
                derivation_id=derivation_id,
                status=DerivationStatus.PROCESSING_SCANNER_TRUTHS,
                wait_condition=None,
            )

        if not new_candidates and not close_ids:
            self._set_derivation_runtime_state(
                derivation_id=derivation_id,
                status=DerivationStatus.WAITING_SCANNER_COVER,
                wait_condition=ScannerCover(scan_demand.demand_id),
            )
            return ScanWait(
                wait_id=self._id_factor(
                    "ttl_cover_scan_wait",
                    f"{local_state.derivation_id.prime}|{scan_demand.demand_id.prime}|recheck",
                ),
                requester_derivation_id=local_state.derivation_id,
                demand_id=scan_demand.demand_id,
                parent_prediction_demand_id=demand.demand_id,
            )

        return PredictorDelta(
            delta_id=self._id_factor(
                "ttl_cover_predictor_delta",
                f"{demand.demand_id.prime}|{len(local_state.processed_truth_ids)}|{len(new_candidates)}",
            ),
            new_prediction_demands=(),
            new_suspended_derivations=(),
            new_candidates=tuple(new_candidates),
            new_closed_prediction_demand_ids=close_ids,
        )

    def _handle_predictor_proposal(
        self,
        derivation_id: int,
        proposal: ScanWait | PredictorDelta | LocallyQuiescent,
    ) -> None:
        derivation = self._suspended_derivations_by_id[derivation_id]
        local_state = self._predictor_derivation_states_by_id[derivation_id]
        if isinstance(proposal, PredictorDelta):
            self._record_proposal_trace(
                service_name="predictor",
                derivation_id=derivation.derivation_id,
                stage=self.predictor_stage,
                proposal=proposal,
            )
            self._enqueue_delta("predictor", derivation.derivation_id, proposal)
            if not local_state.done:
                self._requeue_derivation(local_state.derivation_id)
            return

        if isinstance(proposal, ScanWait):
            self._record_proposal_trace(
                service_name="predictor",
                derivation_id=derivation.derivation_id,
                stage=self.predictor_stage,
                proposal=proposal,
            )
            cover = self._suspended_derivations_by_id[derivation_id].wait_condition
            if not isinstance(cover, ScannerCover):
                raise ValueError("predictor wait must carry ScannerCover")
            if self._scan_demand_has_truths(cover.scan_demand_id):
                self._wake_waiters(
                    cover=cover,
                    triggering_delta_id=None,
                    reason="scanner_demand_already_covered",
                )
            elif cover.scan_demand_id.prime in self._closed_scan_demand_ids:
                self._wake_waiters(
                    cover=cover,
                    triggering_delta_id=None,
                    reason="scanner_demand_already_closed",
                )
            else:
                self._record_locally_quiescent_trace(
                    service_name="predictor",
                    derivation_id=derivation.derivation_id,
                    waiting_on_demand_id=proposal.demand_id,
                    stage=self.predictor_stage,
                    reason="waiting_on_scanner_cover",
                )
            return

        self._record_proposal_trace(
            service_name="predictor",
            derivation_id=derivation.derivation_id,
            stage=self.predictor_stage,
            proposal=proposal,
        )

    def _step_completer_derivation(
        self,
        derivation_id: int,
    ) -> PredictionWait | CompletionWait | CompleterDelta | Unsatisfied | None:
        local_state = self._completer_derivation_states_by_id[derivation_id]
        if local_state.done:
            return None
        demand = self._completion_demands_by_id[local_state.demand_id.prime]

        if local_state.child_completion_demand_id is not None and local_state.active_candidate_id is not None:
            child_demand_id = local_state.child_completion_demand_id
            if self._completion_demand_is_satisfied(child_demand_id):
                child_witness = self._first_completed_witness_for_demand(child_demand_id)
                if child_witness is None:
                    raise ValueError("completion cover must project to a witness")
                candidate = self._candidates_by_id[local_state.active_candidate_id.prime]
                local_state.child_completion_demand_id = None
                local_state.active_candidate_id = None
                if self._completion_satisfies(demand, start=candidate.start, stop=child_witness.stop):
                    witness = self._make_completed_witness(
                        demand=demand,
                        candidate=candidate,
                        right_witness=child_witness,
                    )
                    support = CompletionSupport(
                        support_id=self._id_factor(
                            "ttl_cover_completion_support",
                            f"{demand.demand_id.prime}|{candidate.candidate_id.prime}|{child_witness.witness_id.prime}|binary",
                        ),
                        demand_id=demand.demand_id,
                        candidate_id=candidate.candidate_id,
                        child_completion_demand_id=child_demand_id,
                        child_witness_id=child_witness.witness_id,
                        kind="binary_join",
                    )
                    local_state.done = True
                    self._set_derivation_runtime_state(
                        derivation_id=derivation_id,
                        status=DerivationStatus.DONE,
                        wait_condition=None,
                    )
                    return CompleterDelta(
                        delta_id=self._id_factor(
                            "ttl_cover_completer_delta",
                            f"witness|{witness.witness_id.prime}",
                        ),
                        new_completion_demands=(),
                        new_discharged_completion_demand_ids=(demand.demand_id,),
                        new_suspended_derivations=(),
                        new_completed_witnesses=(witness,),
                        new_supports=(support,),
                    )
            if self._completion_demand_is_unsatisfied(child_demand_id):
                local_state.child_completion_demand_id = None
                local_state.active_candidate_id = None
            else:
                self._set_derivation_runtime_state(
                    derivation_id=derivation_id,
                    status=DerivationStatus.WAITING_COMPLETION_COVER,
                    wait_condition=CompletionResolved(child_demand_id),
                )
                return CompletionWait(
                    wait_id=self._id_factor(
                        "ttl_cover_completion_wait",
                        f"{local_state.derivation_id.prime}|{child_demand_id.prime}",
                    ),
                    requester_derivation_id=local_state.derivation_id,
                    demand_id=child_demand_id,
                    parent_completion_demand_id=demand.demand_id,
                    parent_candidate_id=self._candidates_by_id[
                        local_state.active_candidate_id.prime
                    ].candidate_id,
                )

        prediction_demand = self._prediction_demand_for_completion_demand(demand)
        local_state.prediction_demand_id = prediction_demand.demand_id
        if not self._prediction_demand_has_candidates(prediction_demand.demand_id):
            if prediction_demand.demand_id.prime in self._closed_prediction_demand_ids:
                local_state.done = True
                self._set_derivation_runtime_state(
                    derivation_id=derivation_id,
                    status=DerivationStatus.UNSATISFIED,
                    wait_condition=None,
                )
                return Unsatisfied(
                    unsatisfied_id=self._id_factor(
                        "ttl_cover_unsatisfied",
                        f"{demand.demand_id.prime}|prediction_closed",
                    ),
                    service_name="completer",
                    demand_id=demand.demand_id,
                    reason="prediction_demand_closed_without_cover",
                )
            self._set_derivation_runtime_state(
                derivation_id=derivation_id,
                status=DerivationStatus.WAITING_PREDICTION_COVER,
                wait_condition=PredictionCover(prediction_demand.demand_id),
            )
            return PredictionWait(
                wait_id=self._id_factor(
                    "ttl_cover_prediction_wait",
                    f"{local_state.derivation_id.prime}|{prediction_demand.demand_id.prime}",
                ),
                requester_derivation_id=local_state.derivation_id,
                demand_id=prediction_demand.demand_id,
                parent_completion_demand_id=demand.demand_id,
            )

        self._set_derivation_runtime_state(
            derivation_id=derivation_id,
            status=DerivationStatus.EXAMINING_CANDIDATES,
            wait_condition=None,
        )
        for candidate in self._candidates_for_prediction_demand(prediction_demand.demand_id):
            if candidate.candidate_id.prime in local_state.processed_candidate_ids:
                continue
            local_state.processed_candidate_ids.add(candidate.candidate_id.prime)
            if candidate.residual_symbol is None:
                if not self._completion_satisfies(demand, start=candidate.start, stop=candidate.stop):
                    continue
                witness = self._make_completed_witness(
                    demand=demand,
                    candidate=candidate,
                    right_witness=None,
                )
                support = CompletionSupport(
                    support_id=self._id_factor(
                        "ttl_cover_completion_support",
                        f"{demand.demand_id.prime}|{candidate.candidate_id.prime}|unary",
                    ),
                    demand_id=demand.demand_id,
                    candidate_id=candidate.candidate_id,
                    child_completion_demand_id=None,
                    child_witness_id=None,
                    kind="unary",
                )
                local_state.done = True
                self._set_derivation_runtime_state(
                    derivation_id=derivation_id,
                    status=DerivationStatus.DONE,
                    wait_condition=None,
                )
                return CompleterDelta(
                    delta_id=self._id_factor(
                        "ttl_cover_completer_delta",
                        f"witness|{witness.witness_id.prime}",
                    ),
                    new_completion_demands=(),
                    new_discharged_completion_demand_ids=(demand.demand_id,),
                    new_suspended_derivations=(),
                    new_completed_witnesses=(witness,),
                    new_supports=(support,),
                )

            child_demand = self._make_completion_demand(
                symbol=candidate.residual_symbol,
                start=candidate.stop,
                end=demand.end,
                require_complete=True,
            )
            child_derivation = self._new_completer_derivation(
                demand=child_demand,
                parent_derivation_id=local_state.derivation_id,
                parent_candidate_id=candidate.candidate_id,
            )
            continuation_support = CompletionSupport(
                support_id=self._id_factor(
                    "ttl_cover_completion_support",
                    f"{demand.demand_id.prime}|{candidate.candidate_id.prime}|{child_demand.demand_id.prime}|continuation",
                ),
                demand_id=demand.demand_id,
                candidate_id=candidate.candidate_id,
                child_completion_demand_id=child_demand.demand_id,
                child_witness_id=None,
                kind="continuation",
            )
            local_state.active_candidate_id = candidate.candidate_id
            local_state.child_completion_demand_id = child_demand.demand_id
            self._set_derivation_runtime_state(
                derivation_id=derivation_id,
                status=DerivationStatus.SPAWNED_CHILD_DEMAND,
                wait_condition=None,
            )
            return CompleterDelta(
                delta_id=self._id_factor(
                    "ttl_cover_completer_delta",
                    f"continuation|{demand.demand_id.prime}|{candidate.candidate_id.prime}",
                ),
                new_completion_demands=()
                if child_demand.demand_id.prime in self._completion_demands_by_id
                else (child_demand,),
                new_discharged_completion_demand_ids=(),
                new_suspended_derivations=()
                if child_derivation.derivation_id.prime in self._suspended_derivations_by_id
                else (child_derivation,),
                new_completed_witnesses=(),
                new_supports=()
                if continuation_support.support_id.prime in self._supports_by_id
                else (continuation_support,),
            )

        local_state.done = True
        self._set_derivation_runtime_state(
            derivation_id=derivation_id,
            status=DerivationStatus.UNSATISFIED,
            wait_condition=None,
        )
        return Unsatisfied(
            unsatisfied_id=self._id_factor(
                "ttl_cover_unsatisfied",
                f"{demand.demand_id.prime}|no_covering_candidates",
            ),
            service_name="completer",
            demand_id=demand.demand_id,
            reason="no_covering_candidates",
        )

    def _handle_completer_proposal(
        self,
        derivation_id: int,
        proposal: PredictionWait | CompletionWait | CompleterDelta | LocallyQuiescent | Unsatisfied,
    ) -> None:
        derivation = self._suspended_derivations_by_id[derivation_id]
        local_state = self._completer_derivation_states_by_id[derivation_id]
        if isinstance(proposal, CompleterDelta):
            self._record_proposal_trace(
                service_name="completer",
                derivation_id=derivation.derivation_id,
                stage=self.completer_stage,
                proposal=proposal,
            )
            self._enqueue_delta("completer", derivation.derivation_id, proposal)
            if not local_state.done:
                self._requeue_derivation(local_state.derivation_id)
            return

        if isinstance(proposal, PredictionWait):
            self._record_proposal_trace(
                service_name="completer",
                derivation_id=derivation.derivation_id,
                stage=self.completer_stage,
                proposal=proposal,
            )
            completion_demand = self._completion_demands_by_id[local_state.demand_id.prime]
            prediction_demand = self._prediction_demand_for_completion_demand(completion_demand)
            seed_delta = self._predictor_seed_delta_for_demand(prediction_demand)
            if seed_delta is not None:
                self._enqueue_delta("predictor", derivation.derivation_id, seed_delta)
            cover = self._suspended_derivations_by_id[derivation_id].wait_condition
            if not isinstance(cover, PredictionCover):
                raise ValueError("completer prediction wait must carry PredictionCover")
            if self._prediction_demand_has_candidates(cover.prediction_demand_id):
                self._wake_waiters(
                    cover=cover,
                    triggering_delta_id=None,
                    reason="prediction_demand_already_covered",
                )
            elif cover.prediction_demand_id.prime in self._closed_prediction_demand_ids:
                self._wake_waiters(
                    cover=cover,
                    triggering_delta_id=None,
                    reason="prediction_demand_already_closed",
                )
            else:
                self._record_locally_quiescent_trace(
                    service_name="completer",
                    derivation_id=derivation.derivation_id,
                    waiting_on_demand_id=proposal.demand_id,
                    stage=self.completer_stage,
                    reason="waiting_on_prediction_cover",
                )
            return

        if isinstance(proposal, CompletionWait):
            self._record_proposal_trace(
                service_name="completer",
                derivation_id=derivation.derivation_id,
                stage=self.completer_stage,
                proposal=proposal,
            )
            cover = self._suspended_derivations_by_id[derivation_id].wait_condition
            if not isinstance(cover, CompletionResolved):
                raise ValueError("completer completion wait must carry CompletionResolved")
            if self._completion_demand_is_resolved(cover.completion_demand_id):
                resolution_reason = (
                    "completion_demand_already_satisfied"
                    if self._completion_demand_is_satisfied(cover.completion_demand_id)
                    else "completion_demand_already_unsatisfied"
                )
                self._wake_waiters(
                    cover=cover,
                    triggering_delta_id=None,
                    reason=resolution_reason,
                )
            else:
                self._record_locally_quiescent_trace(
                    service_name="completer",
                    derivation_id=derivation.derivation_id,
                    waiting_on_demand_id=proposal.demand_id,
                    stage=self.completer_stage,
                    reason="waiting_on_completion_cover",
                )
            return

        if isinstance(proposal, LocallyQuiescent):
            self._record_proposal_trace(
                service_name="completer",
                derivation_id=derivation.derivation_id,
                stage=self.completer_stage,
                proposal=proposal,
            )
            return

        self._record_proposal_trace(
            service_name="completer",
            derivation_id=derivation.derivation_id,
            stage=self.completer_stage,
            proposal=proposal,
        )
        self._mark_completion_demand_unsatisfied(proposal.demand_id, proposal.reason)
        local_state.done = True
        self._set_derivation_runtime_state(
            derivation_id=derivation_id,
            status=DerivationStatus.UNSATISFIED,
            wait_condition=None,
        )

    def _predictor_seed_delta_for_demand(
        self,
        demand: PredictionDemand,
    ) -> PredictorDelta | None:
        prediction_items = (
            ()
            if demand.demand_id.prime in self._prediction_demands_by_id
            else (demand,)
        )
        derivation = self._new_predictor_derivation(demand)
        derivation_items = (
            ()
            if derivation.derivation_id.prime in self._suspended_derivations_by_id
            else (derivation,)
        )
        if not prediction_items and not derivation_items:
            return None
        return PredictorDelta(
            delta_id=self._id_factor(
                "ttl_cover_predictor_delta",
                f"seed|{demand.demand_id.prime}",
            ),
            new_prediction_demands=prediction_items,
            new_suspended_derivations=derivation_items,
            new_candidates=(),
            new_closed_prediction_demand_ids=(),
        )

    def _new_predictor_derivation(self, demand: PredictionDemand) -> SuspendedDerivation:
        existing_id = self._prediction_demand_id_to_derivation_id.get(demand.demand_id.prime)
        if existing_id is not None:
            return self._suspended_derivations_by_id[existing_id]
        derivation_id = self._id_factor(
            "ttl_cover_suspended_derivation",
            f"predictor|{demand.demand_id.prime}",
        )
        return SuspendedDerivation(
            derivation_id=derivation_id,
            service_name="predictor",
            demand_id=demand.demand_id,
            status=DerivationStatus.SEEDED,
            wait_condition=None,
            parent_derivation_id=None,
            parent_candidate_id=None,
        )

    def _new_completer_derivation(
        self,
        *,
        demand: CompletionDemand,
        parent_derivation_id: PrimeFactor | None,
        parent_candidate_id: PrimeFactor | None,
    ) -> SuspendedDerivation:
        existing_id = self._completion_demand_id_to_derivation_id.get(demand.demand_id.prime)
        if existing_id is not None:
            return self._suspended_derivations_by_id[existing_id]
        derivation_id = self._id_factor(
            "ttl_cover_suspended_derivation",
            f"completer|{demand.demand_id.prime}|{None if parent_derivation_id is None else parent_derivation_id.prime}|{None if parent_candidate_id is None else parent_candidate_id.prime}",
        )
        return SuspendedDerivation(
            derivation_id=derivation_id,
            service_name="completer",
            demand_id=demand.demand_id,
            status=DerivationStatus.SEEDED,
            wait_condition=None,
            parent_derivation_id=parent_derivation_id,
            parent_candidate_id=parent_candidate_id,
        )

    def _set_derivation_runtime_state(
        self,
        *,
        derivation_id: int,
        status: DerivationStatus,
        wait_condition: CoverCondition | None,
    ) -> None:
        derivation = self._suspended_derivations_by_id[derivation_id]
        updated = SuspendedDerivation(
            derivation_id=derivation.derivation_id,
            service_name=derivation.service_name,
            demand_id=derivation.demand_id,
            status=status,
            wait_condition=wait_condition,
            parent_derivation_id=derivation.parent_derivation_id,
            parent_candidate_id=derivation.parent_candidate_id,
        )
        self._suspended_derivations_by_id[derivation_id] = updated
        self._update_wait_registration(derivation_id, wait_condition)
        if status not in _RUNNABLE_DERIVATION_STATUSES:
            self._drop_derivation_from_runnable_queue(derivation.derivation_id)
        self._assert_derivation_invariant(derivation_id)
        if derivation.service_name == "predictor":
            self._rebuild_predictor_stage()
            return
        if derivation.service_name == "completer":
            self._rebuild_completer_stage()
            return
        raise ValueError(f"unsupported derivation service {derivation.service_name}")

    def _update_wait_registration(
        self,
        derivation_id: int,
        wait_condition: CoverCondition | None,
    ) -> None:
        previous_cover = self._cover_by_waiting_derivation_id.pop(derivation_id, None)
        if previous_cover is not None:
            waiting_derivations = self._waiting_derivation_ids_by_cover.get(previous_cover)
            if waiting_derivations is not None:
                waiting_derivations.discard(derivation_id)
                if not waiting_derivations:
                    self._waiting_derivation_ids_by_cover.pop(previous_cover, None)
        if wait_condition is None:
            return
        self._cover_by_waiting_derivation_id[derivation_id] = wait_condition
        self._waiting_derivation_ids_by_cover.setdefault(wait_condition, set()).add(derivation_id)

    def _wake_predictor_waiters_from_scanner_delta(self, delta: ScannerDelta) -> None:
        demand_ids = {demand.demand_id for demand in delta.new_scan_demands}
        demand_ids.update(delta.new_closed_scan_demand_ids)
        for demand_id in demand_ids:
            cover = ScannerCover(demand_id)
            if self._scan_demand_has_truths(demand_id):
                self._wake_waiters(
                    cover=cover,
                    triggering_delta_id=delta.delta_id,
                    reason="scanner_demand_covered",
                )
            elif demand_id.prime in self._closed_scan_demand_ids:
                self._wake_waiters(
                    cover=cover,
                    triggering_delta_id=delta.delta_id,
                    reason="scanner_demand_closed",
                )

    def _wake_completer_waiters_from_predictor_delta(self, delta: PredictorDelta) -> None:
        demand_ids = {demand.demand_id for demand in delta.new_prediction_demands}
        demand_ids.update(candidate.demand_id for candidate in delta.new_candidates)
        demand_ids.update(delta.new_closed_prediction_demand_ids)
        for demand_id in demand_ids:
            cover = PredictionCover(demand_id)
            if self._prediction_demand_has_candidates(demand_id):
                self._wake_waiters(
                    cover=cover,
                    triggering_delta_id=delta.delta_id,
                    reason="prediction_demand_covered",
                )
            elif demand_id.prime in self._closed_prediction_demand_ids:
                self._wake_waiters(
                    cover=cover,
                    triggering_delta_id=delta.delta_id,
                    reason="prediction_demand_closed",
                )

    def _wake_completer_waiters_from_completion_delta(self, delta: CompleterDelta) -> None:
        demand_ids = set(delta.new_discharged_completion_demand_ids)
        demand_ids.update(witness.demand_id for witness in delta.new_completed_witnesses)
        for demand_id in demand_ids:
            if not self._completion_demand_is_satisfied(demand_id):
                continue
            self._wake_waiters(
                cover=CompletionResolved(demand_id),
                triggering_delta_id=delta.delta_id,
                reason="completion_demand_satisfied",
            )
            # Positive completion cover is emitted now for observability and future
            # consumers, but no derivation waits on CompletionSatisfied in this unit.
            self._wake_waiters(
                cover=CompletionSatisfied(demand_id),
                triggering_delta_id=delta.delta_id,
                reason="completion_demand_satisfied",
            )

    def _wake_waiters(
        self,
        *,
        cover: CoverCondition,
        triggering_delta_id: PrimeFactor | None,
        reason: str,
    ) -> None:
        derivation_ids = sorted(self._waiting_derivation_ids_by_cover.pop(cover, set()))
        if not derivation_ids:
            return
        for derivation_id in derivation_ids:
            self._cover_by_waiting_derivation_id.pop(derivation_id, None)
            self._set_derivation_runtime_state(
                derivation_id=derivation_id,
                status=DerivationStatus.AWAKENED,
                wait_condition=None,
            )
            self._requeue_derivation(self._suspended_derivations_by_id[derivation_id].derivation_id)
        self._record_cover_trace(
            service_name=self._cover_service_name(cover),
            demand_id=self._cover_demand_id(cover),
            awakened_derivation_ids=tuple(
                self._suspended_derivations_by_id[derivation_id].derivation_id
                for derivation_id in derivation_ids
            ),
            triggering_delta_id=triggering_delta_id,
            stage_id=self._cover_stage_id(cover),
            reason=reason,
        )

    def _cover_service_name(self, cover: CoverCondition) -> str:
        if isinstance(cover, ScannerCover):
            return "scanner"
        if isinstance(cover, PredictionCover):
            return "predictor"
        return "completer"

    def _cover_demand_id(self, cover: CoverCondition) -> PrimeFactor:
        if isinstance(cover, ScannerCover):
            return cover.scan_demand_id
        if isinstance(cover, PredictionCover):
            return cover.prediction_demand_id
        return cover.completion_demand_id

    def _cover_stage_id(self, cover: CoverCondition) -> PrimeFactor:
        if isinstance(cover, ScannerCover):
            return self.scanner_stage.stage_id
        if isinstance(cover, PredictionCover):
            return self.predictor_stage.stage_id
        return self.completer_stage.stage_id

    def _mark_completion_demand_unsatisfied(self, demand_id: PrimeFactor, reason: str) -> None:
        if demand_id.prime not in self._unsatisfied_completion_demand_ids:
            self._unsatisfied_completion_demand_ids.add(demand_id.prime)
            self._rebuild_completer_stage()
        self._wake_waiters(
            cover=CompletionResolved(demand_id),
            triggering_delta_id=None,
            reason="completion_demand_unsatisfied",
        )

    def _record_proposal_trace(
        self,
        *,
        service_name: str,
        derivation_id: PrimeFactor | None,
        stage: ScannerStage | PredictorStage | CompleterStage,
        proposal: (
            ScanWait
            | PredictionWait
            | CompletionWait
            | ScannerDelta
            | PredictorDelta
            | CompleterDelta
            | LocallyQuiescent
            | Unsatisfied
        ),
    ) -> None:
        self.proposal_traces.append(
            ProposalTrace(
                trace_id=self._id_factor(
                    "ttl_cover_proposal_trace",
                    f"{service_name}|{len(self.proposal_traces) + 1}|{stage.stage_id.prime}",
                ),
                service_name=service_name,
                derivation_id=derivation_id,
                stage=stage,
                proposal=proposal,
            )
        )

    def _record_cover_trace(
        self,
        *,
        service_name: str,
        demand_id: PrimeFactor,
        awakened_derivation_ids: tuple[PrimeFactor, ...],
        triggering_delta_id: PrimeFactor | None,
        stage_id: PrimeFactor,
        reason: str,
    ) -> None:
        self.cover_traces.append(
            CoverTransitionTrace(
                trace_id=self._id_factor(
                    "ttl_cover_cover_trace",
                    f"{service_name}|{len(self.cover_traces) + 1}|{demand_id.prime}",
                ),
                service_name=service_name,
                demand_id=demand_id,
                awakened_derivation_ids=awakened_derivation_ids,
                triggering_delta_id=triggering_delta_id,
                stage_id=stage_id,
                reason=reason,
            )
        )

    def _record_locally_quiescent_trace(
        self,
        *,
        service_name: str,
        derivation_id: PrimeFactor,
        waiting_on_demand_id: PrimeFactor | None,
        stage: ScannerStage | PredictorStage | CompleterStage,
        reason: str,
    ) -> None:
        self._record_proposal_trace(
            service_name=service_name,
            derivation_id=derivation_id,
            stage=stage,
            proposal=LocallyQuiescent(
                quiescence_id=self._id_factor(
                    "ttl_cover_locally_quiescent",
                    f"{derivation_id.prime}|{None if waiting_on_demand_id is None else waiting_on_demand_id.prime}|{reason}",
                ),
                service_name=service_name,
                derivation_id=derivation_id,
                waiting_on_demand_id=waiting_on_demand_id,
                reason=reason,
            ),
        )

    def _requeue_derivation(self, derivation_id: PrimeFactor) -> None:
        """Queue a derivation only after its observable shell is already runnable."""
        derivation = self._suspended_derivations_by_id[derivation_id.prime]
        if derivation.status not in _RUNNABLE_DERIVATION_STATUSES:
            raise ValueError(
                f"derivation {derivation_id.prime} requires runnable shell status before queueing, got {derivation.status}"
            )
        self._assert_derivation_invariant(derivation_id.prime)
        if derivation_id.prime not in self._runnable_derivation_ids:
            self._runnable_derivation_ids.append(derivation_id.prime)
        self._assert_derivation_invariant(derivation_id.prime)

    def _drop_derivation_from_runnable_queue(self, derivation_id: PrimeFactor) -> None:
        if derivation_id.prime not in self._runnable_derivation_ids:
            return
        self._runnable_derivation_ids = deque(
            queued_id
            for queued_id in self._runnable_derivation_ids
            if queued_id != derivation_id.prime
        )

    def _scanner_delta_for_demand(self, demand: ScanDemand) -> ScannerDelta:
        start = 0 if demand.start is None else demand.start
        stop = len(self.lexemes) if demand.end is None else min(demand.end, len(self.lexemes))
        truths: list[ScannerTruth] = []
        ambiguities: list[ScannerAmbiguityClass] = []
        horizon = self._scanner_horizon
        expected = set(demand.expected_symbols)
        for index in range(start, stop):
            lexeme = self.lexemes[index]
            ambiguity_symbols = self._scanner_symbols_for_lexeme(lexeme)
            if expected:
                ambiguity_symbols = tuple(
                    symbol_name
                    for symbol_name in ambiguity_symbols
                    if self.symbol_factor(f"terminal:{symbol_name}") in expected
                )
                if not ambiguity_symbols:
                    continue
            site_id = self._id_factor("ttl_cover_site", f"{index}|{index + 1}")
            ambiguity_class_id = self._id_factor(
                "ttl_cover_scanner_ambiguity_class",
                f"{demand.demand_id.prime}|{index}|{'|'.join(ambiguity_symbols)}",
            )
            rel_path_id = self._id_factor("ttl_cover_rel_path", lexeme.rel_path)
            site_truths: list[ScannerTruth] = []
            for symbol_name in ambiguity_symbols:
                symbol = self.symbol_factor(f"terminal:{symbol_name}")
                truth = ScannerTruth(
                    truth_id=self._id_factor(
                        "ttl_cover_scanner_truth",
                        f"{ambiguity_class_id.prime}|{symbol.prime}",
                    ),
                    site_id=site_id,
                    ambiguity_class_id=ambiguity_class_id,
                    symbol=symbol,
                    rel_path_id=rel_path_id,
                    start=index,
                    stop=index + 1,
                    lexeme=lexeme,
                )
                site_truths.append(truth)
                if truth.truth_id.prime not in self._scanner_truths_by_id:
                    truths.append(truth)
            ambiguity = ScannerAmbiguityClass(
                ambiguity_class_id=ambiguity_class_id,
                site_id=site_id,
                truth_ids=tuple(truth.truth_id for truth in site_truths),
                start=index,
                stop=index + 1,
                lexeme_text=lexeme.text,
            )
            if ambiguity.ambiguity_class_id.prime not in self._scanner_ambiguities_by_id:
                ambiguities.append(ambiguity)
            horizon = max(horizon, index + 1)
        return ScannerDelta(
            delta_id=self._id_factor(
                "ttl_cover_scanner_delta",
                f"{demand.demand_id.prime}|{start}|{stop}",
            ),
            new_scan_demands=()
            if demand.demand_id.prime in self._scanner_demands_by_id
            else (demand,),
            new_truths=tuple(truths),
            new_ambiguity_classes=tuple(ambiguities),
            new_closed_scan_demand_ids=()
            if demand.demand_id.prime in self._closed_scan_demand_ids
            else (demand.demand_id,),
            new_horizon=horizon,
        )

    def _predictor_candidates_for_truth(
        self,
        *,
        demand: PredictionDemand,
        truth: ScannerTruth,
    ) -> tuple[PredictedCandidate, ...]:
        if demand.start is not None and truth.start < demand.start:
            return ()
        if demand.end is not None and truth.stop > demand.end:
            return ()
        candidates: list[PredictedCandidate] = []
        if truth.symbol == demand.symbol:
            lexical_rule = EarleyRule(head=demand.symbol, rhs=(demand.symbol,))
            candidates.append(
                PredictedCandidate(
                    candidate_id=self._id_factor(
                        "ttl_cover_predicted_candidate",
                        f"{demand.demand_id.prime}|terminal|{truth.truth_id.prime}",
                    ),
                    demand_id=demand.demand_id,
                    rule=lexical_rule,
                    anchor_truth_id=truth.truth_id,
                    anchor_symbol=truth.symbol,
                    start=truth.start,
                    stop=truth.stop,
                    residual_symbol=None,
                )
            )
        for rule in self.rules_by_head.get(demand.symbol, ()):
            if rule.rhs[0] != truth.symbol:
                continue
            candidates.append(
                PredictedCandidate(
                    candidate_id=self._id_factor(
                        "ttl_cover_predicted_candidate",
                        f"{demand.demand_id.prime}|rule|{rule.head.prime}|{truth.truth_id.prime}",
                    ),
                    demand_id=demand.demand_id,
                    rule=rule,
                    anchor_truth_id=truth.truth_id,
                    anchor_symbol=truth.symbol,
                    start=truth.start,
                    stop=truth.stop,
                    residual_symbol=None if rule.is_unary else rule.rhs[1],
                )
            )
        return tuple(candidates)

    def _predictor_expected_symbols(self, symbol: PrimeFactor) -> tuple[PrimeFactor, ...]:
        expected: dict[int, PrimeFactor] = {symbol.prime: symbol}
        for rule in self.rules_by_head.get(symbol, ()):
            expected[rule.rhs[0].prime] = rule.rhs[0]
        return tuple(sorted(expected.values(), key=lambda factor: factor.prime))

    def _scanner_truths_for_scan_demand(self, scan_demand_id: PrimeFactor) -> tuple[ScannerTruth, ...]:
        return tuple(
            self._scanner_truths_by_id[truth_id]
            for truth_id in self._scanner_truth_ids_by_demand_id.get(scan_demand_id.prime, [])
        )

    def _scan_demand_for_prediction_demand(self, demand: PredictionDemand) -> ScanDemand:
        existing_scan_id = self._scan_demand_id_by_prediction_demand_id.get(demand.demand_id.prime)
        if existing_scan_id is not None and existing_scan_id in self._scanner_demands_by_id:
            return self._scanner_demands_by_id[existing_scan_id]
        scan_demand = self._make_scan_demand(
            start=demand.start,
            end=demand.end,
            expected_symbols=self._predictor_expected_symbols(demand.symbol),
        )
        self._scan_demand_id_by_prediction_demand_id[demand.demand_id.prime] = scan_demand.demand_id.prime
        return scan_demand

    def _prediction_demand_for_completion_demand(
        self,
        demand: CompletionDemand,
    ) -> PredictionDemand:
        prediction_demand = self._make_prediction_demand(
            symbol=demand.symbol,
            start=demand.start,
            end=demand.end,
        )
        return self._prediction_demands_by_id.get(
            prediction_demand.demand_id.prime,
            prediction_demand,
        )

    def _make_scan_demand(
        self,
        *,
        start: int | None,
        end: int | None,
        expected_symbols: tuple[PrimeFactor, ...],
    ) -> ScanDemand:
        token = "|".join(str(symbol.prime) for symbol in expected_symbols)
        return ScanDemand(
            demand_id=self._id_factor(
                "ttl_cover_scan_demand",
                f"{start}|{end}|{token}",
            ),
            start=start,
            end=end,
            expected_symbols=expected_symbols,
        )

    def _make_prediction_demand(
        self,
        *,
        symbol: PrimeFactor,
        start: int | None,
        end: int | None,
    ) -> PredictionDemand:
        return PredictionDemand(
            demand_id=self._id_factor(
                "ttl_cover_prediction_demand",
                f"{symbol.prime}|{start}|{end}",
            ),
            symbol=symbol,
            start=start,
            end=end,
        )

    def _make_completion_demand(
        self,
        *,
        symbol: PrimeFactor,
        start: int | None,
        end: int | None,
        require_complete: bool,
    ) -> CompletionDemand:
        return CompletionDemand(
            demand_id=self._id_factor(
                "ttl_cover_completion_demand",
                f"{symbol.prime}|{start}|{end}|{int(require_complete)}",
            ),
            symbol=symbol,
            start=start,
            end=end,
            require_complete=require_complete,
        )

    def _make_completed_witness(
        self,
        *,
        demand: CompletionDemand,
        candidate: PredictedCandidate,
        right_witness: CompletedWitness | None,
    ) -> CompletedWitness:
        stop = candidate.stop if right_witness is None else right_witness.stop
        return CompletedWitness(
            witness_id=self._id_factor(
                "ttl_cover_completed_witness",
                f"{demand.demand_id.prime}|{candidate.candidate_id.prime}|{None if right_witness is None else right_witness.witness_id.prime}",
            ),
            demand_id=demand.demand_id,
            symbol=demand.symbol,
            start=candidate.start,
            stop=stop,
            rule=candidate.rule,
            candidate_id=candidate.candidate_id,
            anchor_truth_id=candidate.anchor_truth_id,
            right_witness_id=None if right_witness is None else right_witness.witness_id,
        )

    def _completion_satisfies(
        self,
        demand: CompletionDemand,
        *,
        start: int,
        stop: int,
    ) -> bool:
        if demand.start is not None and start != demand.start:
            return False
        if demand.end is not None and stop != demand.end:
            return False
        return True

    def _scanner_symbols_for_lexeme(self, lexeme: TurtleLexeme) -> tuple[str, ...]:
        if lexeme.text == "a":
            return ("A", "NAME")
        return (lexeme.terminal_name,)

    def _scan_demand_has_truths(self, demand_id: PrimeFactor) -> bool:
        return bool(self._scanner_truth_ids_by_demand_id.get(demand_id.prime))

    def _prediction_demand_has_candidates(self, demand_id: PrimeFactor) -> bool:
        # Prediction cover remains coarse in this correction unit: any candidate counts.
        return bool(self._candidate_ids_by_demand_id.get(demand_id.prime))

    def _completion_demand_is_satisfied(self, demand_id: PrimeFactor) -> bool:
        return bool(self._witness_ids_by_demand_id.get(demand_id.prime))

    def _completion_demand_is_unsatisfied(self, demand_id: PrimeFactor) -> bool:
        return demand_id.prime in self._unsatisfied_completion_demand_ids

    def _completion_demand_is_resolved(self, demand_id: PrimeFactor) -> bool:
        return self._completion_demand_is_satisfied(demand_id) or self._completion_demand_is_unsatisfied(
            demand_id
        )

    def _candidates_for_prediction_demand(
        self,
        demand_id: PrimeFactor,
    ) -> tuple[PredictedCandidate, ...]:
        return tuple(
            self._candidates_by_id[candidate_id]
            for candidate_id in self._candidate_ids_by_demand_id.get(demand_id.prime, [])
        )

    def _first_completed_witness_for_demand(
        self,
        demand_id: PrimeFactor,
    ) -> CompletedWitness | None:
        witness_ids = self._witness_ids_by_demand_id.get(demand_id.prime, [])
        if not witness_ids:
            return None
        return self._completed_witnesses_by_id[witness_ids[0]]

    def _select_witness_for_policy(
        self,
        *,
        demand: CompletionDemand,
        policy: ConsumerPolicy,
    ) -> CompletedWitness | None:
        if policy.mode != "first":
            raise ValueError(f"unsupported consumer policy mode {policy.mode}")
        return self._first_completed_witness_for_demand(demand.demand_id)

    def _id_factor(self, namespace: str, token: str) -> PrimeFactor:
        return self.prime_space.intern_factor(
            identity_space=self.identity_space,
            namespace=namespace,
            token=token,
        )


def run_turtle_cover_poset_skeleton(
    *,
    symbol: str = "directive",
    start: int | None = None,
    end: int | None = None,
    root: Path = _REPO_ROOT,
    rel_paths: tuple[Path, ...] = _DEFAULT_TTL_PATHS,
    max_tokens: int | None = 256,
    max_steps: int = 10_000,
) -> CoverPosetResult:
    coordinator = FixedPointCoordinator.create(
        root=root,
        rel_paths=rel_paths,
        max_tokens=max_tokens,
        max_steps=max_steps,
    )
    root_demand = coordinator.make_completion_demand(
        symbol_name=symbol,
        start=start,
        end=end,
        require_complete=True,
    )
    policy = coordinator.make_consumer_policy(mode="first")
    return coordinator.run_root_demand(root_demand, policy)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cover-predicate Earley kernel over the repo's kernel TTL files.",
    )
    parser.add_argument(
        "--symbol",
        default="directive",
        help="Grammar symbol to satisfy with the configured consumer policy.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Optional exact start position for the requested witness.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Optional exact end position for the requested witness.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Limit scanned TTL lexemes before running the cover-poset kernel.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10_000,
        help="Maximum coordinator steps before the experiment aborts.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the cover-poset summary as JSON.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_turtle_cover_poset_skeleton(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        max_tokens=args.max_tokens,
        max_steps=args.max_steps,
    )
    print(json.dumps(result.as_summary(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
