from __future__ import annotations

"""Demand-routed continuation-first Earley skeleton over the repo's kernel TTL files.

This experiment treats parsing as a routed network of suspended continuations:

- completer continuations yield `NeedPredictions` or `NeedCompleted`
- predictor continuations yield `NeedScan`
- scanner continuations yield observational witness batches

The semantic core is continuation-first. Service stages are derived views over
the routed morphisms rather than the primary execution substrate.
"""

from collections import deque
from collections.abc import Generator, Iterable, Iterator
from dataclasses import dataclass
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
_START = object()


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
class RootQuery:
    query_id: PrimeFactor
    symbol: PrimeFactor
    start: int | None
    end: int | None
    require_complete: bool = True


@dataclass(frozen=True)
class CompletionKey:
    key_id: PrimeFactor
    symbol: PrimeFactor
    start: int | None
    end: int | None
    require_complete: bool = True


@dataclass(frozen=True)
class PredictionKey:
    key_id: PrimeFactor
    symbol: PrimeFactor
    start: int | None
    end: int | None


@dataclass(frozen=True)
class ScanKey:
    key_id: PrimeFactor
    start: int | None
    end: int | None
    expected_symbols: tuple[PrimeFactor, ...]


@dataclass(frozen=True)
class NeedScan:
    edge_id: PrimeFactor
    parent_continuation_id: PrimeFactor
    key: ScanKey
    cursor: int
    requested_by_prediction_key_id: PrimeFactor


@dataclass(frozen=True)
class NeedPredictions:
    edge_id: PrimeFactor
    parent_continuation_id: PrimeFactor
    key: PredictionKey
    cursor: int
    requested_by_completion_key_id: PrimeFactor


@dataclass(frozen=True)
class NeedCompleted:
    edge_id: PrimeFactor
    parent_continuation_id: PrimeFactor
    key: CompletionKey
    cursor: int
    parent_completion_key_id: PrimeFactor | None
    parent_candidate_id: PrimeFactor | None


@dataclass(frozen=True)
class Exhausted:
    exhaustion_id: PrimeFactor
    service_name: str
    key_id: PrimeFactor
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
class ScanWitnessBatch:
    batch_id: PrimeFactor
    key_id: PrimeFactor
    horizon: int
    truths: tuple[ScannerTruth, ...]
    ambiguity_classes: tuple[ScannerAmbiguityClass, ...]


@dataclass(frozen=True)
class PredictedCandidate:
    candidate_id: PrimeFactor
    key_id: PrimeFactor
    rule: EarleyRule
    anchor_truth_id: PrimeFactor
    anchor_symbol: PrimeFactor
    start: int
    stop: int
    residual_symbol: PrimeFactor | None


@dataclass(frozen=True)
class PredictionWitnessBatch:
    batch_id: PrimeFactor
    key_id: PrimeFactor
    candidates: tuple[PredictedCandidate, ...]


@dataclass(frozen=True)
class CompletedWitness:
    witness_id: PrimeFactor
    key_id: PrimeFactor
    symbol: PrimeFactor
    start: int
    stop: int
    rule: EarleyRule
    candidate_id: PrimeFactor
    anchor_truth_id: PrimeFactor
    right_witness_id: PrimeFactor | None


@dataclass(frozen=True)
class CompletionContinuation:
    continuation_id: PrimeFactor
    parent_completion_key_id: PrimeFactor
    child_completion_key_id: PrimeFactor
    parent_candidate_id: PrimeFactor
    parent_continuation_id: PrimeFactor


@dataclass(frozen=True)
class CompletionSupport:
    support_id: PrimeFactor
    completion_key_id: PrimeFactor
    candidate_id: PrimeFactor
    child_completion_key_id: PrimeFactor | None
    child_witness_id: PrimeFactor | None
    kind: str


@dataclass(frozen=True)
class ScannerStage:
    stage_id: PrimeFactor
    request_id: PrimeFactor
    scan_key_ids: frozenset[PrimeFactor]
    truth_ids: frozenset[PrimeFactor]
    ambiguity_class_ids: frozenset[PrimeFactor]
    scan_keys: tuple[ScanKey, ...]
    truths: tuple[ScannerTruth, ...]
    ambiguity_classes: tuple[ScannerAmbiguityClass, ...]

    def extends(self, other: ScannerStage) -> bool:
        return (
            other.scan_key_ids <= self.scan_key_ids
            and other.truth_ids <= self.truth_ids
            and other.ambiguity_class_ids <= self.ambiguity_class_ids
        )


@dataclass(frozen=True)
class ScannerDelta:
    delta_id: PrimeFactor
    request_id: PrimeFactor
    new_scan_key_ids: tuple[PrimeFactor, ...]
    new_truth_ids: tuple[PrimeFactor, ...]
    new_ambiguity_class_ids: tuple[PrimeFactor, ...]
    new_scan_keys: tuple[ScanKey, ...]
    new_truths: tuple[ScannerTruth, ...]
    new_ambiguity_classes: tuple[ScannerAmbiguityClass, ...]


@dataclass(frozen=True)
class PredictorStage:
    stage_id: PrimeFactor
    request_id: PrimeFactor
    prediction_key_ids: frozenset[PrimeFactor]
    candidate_ids: frozenset[PrimeFactor]
    prediction_keys: tuple[PredictionKey, ...]
    candidates: tuple[PredictedCandidate, ...]

    def extends(self, other: PredictorStage) -> bool:
        return (
            other.prediction_key_ids <= self.prediction_key_ids
            and other.candidate_ids <= self.candidate_ids
        )


@dataclass(frozen=True)
class PredictorDelta:
    delta_id: PrimeFactor
    request_id: PrimeFactor
    new_prediction_key_ids: tuple[PrimeFactor, ...]
    new_candidate_ids: tuple[PrimeFactor, ...]
    new_prediction_keys: tuple[PredictionKey, ...]
    new_candidates: tuple[PredictedCandidate, ...]


@dataclass(frozen=True)
class CompleterStage:
    stage_id: PrimeFactor
    request_id: PrimeFactor
    completion_key_ids: frozenset[PrimeFactor]
    discharged_completion_key_ids: frozenset[PrimeFactor]
    continuation_ids: frozenset[PrimeFactor]
    witness_ids: frozenset[PrimeFactor]
    support_ids: frozenset[PrimeFactor]
    completion_keys: tuple[CompletionKey, ...]
    discharged_completion_keys: tuple[CompletionKey, ...]
    continuations: tuple[CompletionContinuation, ...]
    completed_witnesses: tuple[CompletedWitness, ...]
    supports: tuple[CompletionSupport, ...]

    def extends(self, other: CompleterStage) -> bool:
        return (
            other.completion_key_ids <= self.completion_key_ids
            and other.discharged_completion_key_ids <= self.discharged_completion_key_ids
            and other.continuation_ids <= self.continuation_ids
            and other.witness_ids <= self.witness_ids
            and other.support_ids <= self.support_ids
        )


@dataclass(frozen=True)
class CompleterDelta:
    delta_id: PrimeFactor
    request_id: PrimeFactor
    new_completion_key_ids: tuple[PrimeFactor, ...]
    new_discharged_completion_key_ids: tuple[PrimeFactor, ...]
    new_continuation_ids: tuple[PrimeFactor, ...]
    new_witness_ids: tuple[PrimeFactor, ...]
    new_support_ids: tuple[PrimeFactor, ...]
    new_completion_keys: tuple[CompletionKey, ...]
    new_discharged_completion_keys: tuple[CompletionKey, ...]
    new_continuations: tuple[CompletionContinuation, ...]
    new_completed_witnesses: tuple[CompletedWitness, ...]
    new_supports: tuple[CompletionSupport, ...]


@dataclass(frozen=True)
class GlobalStage:
    stage_id: PrimeFactor
    query_id: PrimeFactor
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
class ServiceEmission:
    emission_id: PrimeFactor
    service_name: str
    request_id: PrimeFactor
    stage: ScannerStage | PredictorStage | CompleterStage
    delta: ScannerDelta | PredictorDelta | CompleterDelta | None
    blocked_on: tuple[NeedScan | NeedPredictions | NeedCompleted, ...]
    yielded: tuple[
        NeedScan
        | NeedPredictions
        | NeedCompleted
        | ScanWitnessBatch
        | PredictionWitnessBatch
        | CompletedWitness
        | Exhausted,
        ...,
    ]


@dataclass(frozen=True)
class ServicePosetResult:
    source_paths: tuple[str, ...]
    lexemes: tuple[TurtleLexeme, ...]
    grammar: tuple[EarleyRule, ...]
    root_query: RootQuery
    scanner_trace: tuple[ServiceEmission, ...]
    predictor_trace: tuple[ServiceEmission, ...]
    completer_trace: tuple[ServiceEmission, ...]
    final_global_stage: GlobalStage
    first_completed_witness: CompletedWitness | None
    exhaustion: Exhausted | None
    request_metrics: dict[str, tuple[int, int]]

    def as_summary(self) -> dict[str, object]:
        ambiguity_samples: list[dict[str, object]] = []
        for emission in self.scanner_trace:
            delta = emission.delta
            if not isinstance(delta, ScannerDelta):
                continue
            for ambiguity in delta.new_ambiguity_classes:
                if len(ambiguity.truth_ids) < 2:
                    continue
                truths = [
                    truth
                    for truth in emission.stage.truths
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
            if len(ambiguity_samples) == 5:
                break

        predicted_samples: list[dict[str, object]] = []
        for emission in self.predictor_trace:
            delta = emission.delta
            if not isinstance(delta, PredictorDelta):
                continue
            predicted_samples.extend(
                {
                    "candidate_prime": candidate.candidate_id.prime,
                    "head": candidate.rule.head.token,
                    "start": candidate.start,
                    "stop": candidate.stop,
                    "residual": None
                    if candidate.residual_symbol is None
                    else candidate.residual_symbol.token,
                }
                for candidate in delta.new_candidates[:3]
            )
            if len(predicted_samples) >= 5:
                predicted_samples = predicted_samples[:5]
                break

        completed_samples: list[dict[str, object]] = []
        for emission in self.completer_trace:
            delta = emission.delta
            if not isinstance(delta, CompleterDelta):
                continue
            completed_samples.extend(
                {
                    "witness_prime": witness.witness_id.prime,
                    "symbol": witness.symbol.token,
                    "start": witness.start,
                    "stop": witness.stop,
                }
                for witness in delta.new_completed_witnesses[:3]
            )
            if len(completed_samples) >= 5:
                completed_samples = completed_samples[:5]
                break

        return {
            "source_paths": list(self.source_paths),
            "lexeme_count": len(self.lexemes),
            "grammar_rule_count": len(self.grammar),
            "query_symbol": self.root_query.symbol.token,
            "query_start": self.root_query.start,
            "query_end": self.root_query.end,
            "found_witness": self.first_completed_witness is not None,
            "witness_span": None
            if self.first_completed_witness is None
            else [self.first_completed_witness.start, self.first_completed_witness.stop],
            "scanner_stage_count": len(self.scanner_trace),
            "predictor_stage_count": len(self.predictor_trace),
            "completer_stage_count": len(self.completer_trace),
            "request_metrics": {
                name: {"total": total, "unique": unique}
                for name, (total, unique) in self.request_metrics.items()
            },
            "sample_ambiguity_classes": ambiguity_samples,
            "sample_predicted_candidates": predicted_samples,
            "sample_completed_witnesses": completed_samples,
            "exhaustion_reason": None if self.exhaustion is None else self.exhaustion.reason,
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
        raw = text[index:cursor]
        tokens.append(
            TurtleLexeme(
                rel_path=rel_path,
                offset=index,
                kind="NAME",
                text=raw,
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
class _ContinuationRecord:
    continuation_id: PrimeFactor
    service_name: str
    key_id: PrimeFactor
    generator: Generator[
        NeedScan
        | NeedPredictions
        | NeedCompleted
        | ScanWitnessBatch
        | PredictionWitnessBatch
        | CompletedWitness
        | Exhausted,
        ScanWitnessBatch | PredictionWitnessBatch | CompletedWitness | Exhausted | object,
        None,
    ]
    started: bool = False
    done: bool = False


class ServicePosetRouter:
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

        self._reset_runtime_state()

    @classmethod
    def create(
        cls,
        *,
        root: Path = _REPO_ROOT,
        rel_paths: tuple[Path, ...] = _DEFAULT_TTL_PATHS,
        max_tokens: int | None = 256,
        max_steps: int = 10_000,
    ) -> ServicePosetRouter:
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
                            namespace="ttl_service_symbol",
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

    def _reset_runtime_state(self) -> None:
        self.scanner_trace: list[ServiceEmission] = []
        self.predictor_trace: list[ServiceEmission] = []
        self.completer_trace: list[ServiceEmission] = []
        self._metrics = {
            "scan": [0, 0],
            "predict": [0, 0],
            "complete": [0, 0],
        }
        self._step_count = 0
        self._continuations: dict[int, _ContinuationRecord] = {}
        self._producer_by_key: dict[tuple[str, int], PrimeFactor] = {}
        self._waiting_edges: dict[
            tuple[str, int],
            list[NeedScan | NeedPredictions | NeedCompleted],
        ] = {}
        self._service_history: dict[tuple[str, int], list[object]] = {}
        self._service_exhaustion: dict[tuple[str, int], Exhausted] = {}
        self._ready: deque[int] = deque()
        self._scheduled_messages: dict[int, object] = {}
        self._scan_keys_by_id: dict[int, ScanKey] = {}
        self._prediction_keys_by_id: dict[int, PredictionKey] = {}
        self._completion_keys_by_id: dict[int, CompletionKey] = {}
        self._completion_continuations_by_id: dict[int, CompletionContinuation] = {}
        self._completion_supports_by_id: dict[int, CompletionSupport] = {}

    def symbol_factor(self, name: str) -> PrimeFactor:
        factor = self.symbol_table.get(name)
        if factor is None:
            factor = self.prime_space.intern_factor(
                identity_space=self.identity_space,
                namespace="ttl_service_symbol",
                token=name,
            )
            self.symbol_table[name] = factor
        return factor

    @property
    def request_metrics(self) -> dict[str, tuple[int, int]]:
        return {
            name: (total, unique)
            for name, (total, unique) in self._metrics.items()
        }

    def make_root_query(
        self,
        *,
        symbol_name: str,
        start: int | None = None,
        end: int | None = None,
        require_complete: bool = True,
    ) -> RootQuery:
        with deadline_scope(Deadline.from_timeout_ms(30_000)):
            with deadline_clock_scope(MonotonicClock()):
                symbol = self.symbol_factor(symbol_name)
                query_id = self._id_factor(
                    "ttl_service_root_query",
                    f"{symbol.token}|{start}|{end}|{int(require_complete)}",
                )
                return RootQuery(
                    query_id=query_id,
                    symbol=symbol,
                    start=start,
                    end=end,
                    require_complete=require_complete,
                )

    def make_scan_request(
        self,
        *,
        start: int | None = None,
        end: int | None = None,
    ) -> NeedScan:
        with deadline_scope(Deadline.from_timeout_ms(30_000)):
            with deadline_clock_scope(MonotonicClock()):
                observer_id = self._id_factor(
                    "ttl_service_observer",
                    f"scan|{start}|{end}",
                )
                key = self._make_scan_key(
                    start=start,
                    end=end,
                    expected_symbols=(),
                )
                return NeedScan(
                    edge_id=self._id_factor(
                        "ttl_service_need_scan",
                        f"{observer_id.prime}|{key.key_id.prime}|0",
                    ),
                    parent_continuation_id=observer_id,
                    key=key,
                    cursor=0,
                    requested_by_prediction_key_id=key.key_id,
                )

    def make_completion_request(
        self,
        *,
        symbol_name: str,
        start: int | None = None,
        end: int | None = None,
        require_complete: bool = True,
    ) -> NeedCompleted:
        with deadline_scope(Deadline.from_timeout_ms(30_000)):
            with deadline_clock_scope(MonotonicClock()):
                observer_id = self._id_factor(
                    "ttl_service_observer",
                    f"complete|{symbol_name}|{start}|{end}|{int(require_complete)}",
                )
                key = self._make_completion_key(
                    symbol=self.symbol_factor(symbol_name),
                    start=start,
                    end=end,
                    require_complete=require_complete,
                )
                return NeedCompleted(
                    edge_id=self._id_factor(
                        "ttl_service_need_completed",
                        f"{observer_id.prime}|{key.key_id.prime}|0|root",
                    ),
                    parent_continuation_id=observer_id,
                    key=key,
                    cursor=0,
                    parent_completion_key_id=None,
                    parent_candidate_id=None,
                )

    def _make_prediction_request(
        self,
        *,
        symbol: PrimeFactor,
        start: int | None,
        end: int | None,
    ) -> NeedPredictions:
        observer_id = self._id_factor(
            "ttl_service_observer",
            f"predict|{symbol.prime}|{start}|{end}",
        )
        key = self._make_prediction_key(symbol=symbol, start=start, end=end)
        return NeedPredictions(
            edge_id=self._id_factor(
                "ttl_service_need_predictions",
                f"{observer_id.prime}|{key.key_id.prime}|0|observer",
            ),
            parent_continuation_id=observer_id,
            key=key,
            cursor=0,
            requested_by_completion_key_id=key.key_id,
        )

    def resolve_query(self, query: RootQuery) -> CompletedWitness | None:
        with deadline_scope(Deadline.from_timeout_ms(30_000)):
            with deadline_clock_scope(MonotonicClock()):
                return self.run_root_query(query).first_completed_witness

    def stream_scan(self, request: NeedScan) -> tuple[ScanWitnessBatch | Exhausted, ...]:
        with deadline_scope(Deadline.from_timeout_ms(30_000)):
            with deadline_clock_scope(MonotonicClock()):
                self._reset_runtime_state()
                outputs: list[ScanWitnessBatch | Exhausted] = []
                continuation_id = self._spawn_producer("scanner", request.key)
                while True:
                    output = self._drive_direct(continuation_id)
                    if output is None:
                        break
                    outputs.append(output)
                    if isinstance(output, Exhausted):
                        break
                return tuple(outputs)

    def run_root_query(self, query: RootQuery) -> ServicePosetResult:
        with deadline_scope(Deadline.from_timeout_ms(30_000)):
            with deadline_clock_scope(MonotonicClock()):
                self._reset_runtime_state()
                run = _ContinuationRun(router=self, query=query)
                return run.run()

    def _stream_predictions(
        self,
        request: NeedPredictions,
    ) -> Iterator[PredictionWitnessBatch | Exhausted]:
        self._reset_runtime_state()
        continuation_id = self._spawn_producer("predictor", request.key)
        while True:
            output = self._drive_direct(continuation_id)
            if output is None:
                return
            yield output
            if isinstance(output, Exhausted):
                return

    def _drive_direct(
        self,
        continuation_id: PrimeFactor,
    ) -> ScanWitnessBatch | PredictionWitnessBatch | Exhausted | None:
        record = self._continuations[continuation_id.prime]
        while True:
            try:
                if not record.started:
                    yielded = next(record.generator)
                    record.started = True
                else:
                    yielded = record.generator.send(_START)
            except StopIteration:
                record.done = True
                return None
            if isinstance(yielded, NeedScan):
                need_emission = self._make_service_emission(
                    record=record,
                    yielded=yielded,
                    delta=None,
                    blocked_on=(yielded,),
                )
                self._record_emission(need_emission)
                scanner_id = self._spawn_producer("scanner", yielded.key)
                response = self._drive_direct(scanner_id)
                if response is None:
                    response = self._make_exhausted(
                        "scanner",
                        yielded.key.key_id,
                        "direct_stream_end",
                    )
                try:
                    yielded = record.generator.send(response)
                except StopIteration:
                    record.done = True
                    return response if isinstance(response, Exhausted) else None
                if isinstance(yielded, (PredictionWitnessBatch, Exhausted)):
                    self._record_direct_emission(record, yielded)
                    return yielded
                continue
            if isinstance(yielded, (ScanWitnessBatch, PredictionWitnessBatch, Exhausted)):
                self._record_direct_emission(record, yielded)
                return yielded
            raise TypeError("unexpected direct-drive yield")

    def _record_direct_emission(
        self,
        record: _ContinuationRecord,
        yielded: ScanWitnessBatch | PredictionWitnessBatch | Exhausted,
    ) -> None:
        if record.service_name == "scanner":
            if isinstance(yielded, ScanWitnessBatch):
                self._scan_keys_by_id.setdefault(yielded.key_id.prime, self._scan_key_for_id(yielded.key_id))
                emission = self._record_service_output(record, yielded)
            else:
                emission = self._record_service_output(record, yielded)
            self._record_emission(emission)
            return
        if record.service_name == "predictor":
            emission = self._record_service_output(record, yielded)
            self._record_emission(emission)
            return
        raise ValueError("direct-drive only supports scanner or predictor")

    def _run_router(self, root_key: CompletionKey) -> tuple[GlobalStage, CompletedWitness | None, Exhausted | None]:
        self._spawn_producer("complete", root_key)
        while self._ready:
            continuation_prime = self._ready.popleft()
            message = self._scheduled_messages.pop(continuation_prime)
            record = self._continuations[continuation_prime]
            if record.done:
                continue
            yielded = self._resume_continuation(record, message)
            if yielded is None:
                continue
            if isinstance(yielded, (NeedScan, NeedPredictions, NeedCompleted)):
                emission = self._register_dependency(record, yielded)
            else:
                emission = self._record_service_output(record, yielded)
            self._record_emission(emission)
            root_witness = self._first_completed_for_key(root_key.key_id)
            if root_witness is not None:
                return self._build_global_stage(query_id=root_key.key_id), root_witness, None
            root_exhausted = self._service_exhaustion.get(("complete", root_key.key_id.prime))
            if root_exhausted is not None:
                return self._build_global_stage(query_id=root_key.key_id), None, root_exhausted
        exhaustion = self._make_exhausted("completer", root_key.key_id, "router_quiesced")
        return self._build_global_stage(query_id=root_key.key_id), None, exhaustion

    def _resume_continuation(
        self,
        record: _ContinuationRecord,
        message: object,
    ) -> (
        NeedScan
        | NeedPredictions
        | NeedCompleted
        | ScanWitnessBatch
        | PredictionWitnessBatch
        | CompletedWitness
        | Exhausted
        | None
    ):
        try:
            if not record.started:
                yielded = next(record.generator)
                record.started = True
            else:
                yielded = record.generator.send(message)
            return yielded
        except StopIteration:
            record.done = True
            return None

    def _register_dependency(
        self,
        record: _ContinuationRecord,
        edge: NeedScan | NeedPredictions | NeedCompleted,
    ) -> ServiceEmission:
        if isinstance(edge, NeedScan):
            service_name = "scanner"
            self._metrics["scan"][0] += 1
            self._spawn_producer("scanner", edge.key)
        elif isinstance(edge, NeedPredictions):
            service_name = "predictor"
            self._metrics["predict"][0] += 1
            self._spawn_producer("predictor", edge.key)
        else:
            service_name = "complete"
            self._metrics["complete"][0] += 1
            self._spawn_producer("complete", edge.key)
            self._register_completion_continuation(edge)

        history, exhausted = self._service_results_for_edge(edge)
        if edge.cursor < len(history):
            self._enqueue(edge.parent_continuation_id, history[edge.cursor])
        elif exhausted is not None:
            self._enqueue(edge.parent_continuation_id, exhausted)
        else:
            dep_key = self._dependency_key(service_name, edge.key.key_id)
            self._waiting_edges.setdefault(dep_key, []).append(edge)
            producer = self._producer_by_key[dep_key]
            self._enqueue(producer, _START)

        return self._make_service_emission(
            record=record,
            yielded=edge,
            delta=self._delta_for_need_edge(record.service_name, edge),
            blocked_on=(edge,),
        )

    def _record_service_output(
        self,
        record: _ContinuationRecord,
        yielded: ScanWitnessBatch | PredictionWitnessBatch | CompletedWitness | Exhausted,
    ) -> ServiceEmission:
        dep_key = self._dependency_key(record.service_name, record.key_id)
        delta = self._delta_for_output(record.service_name, yielded)
        if isinstance(yielded, Exhausted):
            self._service_exhaustion[dep_key] = yielded
            record.done = True
            waiters = self._waiting_edges.pop(dep_key, [])
            for edge in waiters:
                self._enqueue(edge.parent_continuation_id, yielded)
        else:
            self._service_history.setdefault(dep_key, []).append(yielded)
            waiters = self._waiting_edges.pop(dep_key, [])
            for edge in waiters:
                self._enqueue(edge.parent_continuation_id, yielded)
        return self._make_service_emission(
            record=record,
            yielded=yielded,
            delta=delta,
            blocked_on=(),
        )

    def _spawn_producer(
        self,
        service_name: str,
        key: ScanKey | PredictionKey | CompletionKey,
    ) -> PrimeFactor:
        dep_key = self._dependency_key(service_name, key.key_id)
        existing = self._producer_by_key.get(dep_key)
        if existing is not None:
            return existing
        continuation_id = self._id_factor(
            "ttl_service_continuation",
            f"{service_name}|{key.key_id.prime}",
        )
        if service_name == "scanner":
            generator = self._scanner_continuation(continuation_id, key)
            self._scan_keys_by_id[key.key_id.prime] = key
            self._metrics["scan"][1] += 1
        elif service_name == "predictor":
            generator = self._predictor_continuation(continuation_id, key)
            self._prediction_keys_by_id[key.key_id.prime] = key
            self._metrics["predict"][1] += 1
        else:
            generator = self._completer_continuation(continuation_id, key)
            self._completion_keys_by_id[key.key_id.prime] = key
            self._metrics["complete"][1] += 1
        self._continuations[continuation_id.prime] = _ContinuationRecord(
            continuation_id=continuation_id,
            service_name=service_name,
            key_id=key.key_id,
            generator=generator,
        )
        self._producer_by_key[dep_key] = continuation_id
        self._enqueue(continuation_id, _START)
        return continuation_id

    def _enqueue(self, continuation_id: PrimeFactor, message: object) -> None:
        self._scheduled_messages[continuation_id.prime] = message
        if continuation_id.prime not in self._ready:
            self._ready.append(continuation_id.prime)

    def _service_results_for_edge(
        self,
        edge: NeedScan | NeedPredictions | NeedCompleted,
    ) -> tuple[list[object], Exhausted | None]:
        if isinstance(edge, NeedScan):
            dep_key = self._dependency_key("scanner", edge.key.key_id)
        elif isinstance(edge, NeedPredictions):
            dep_key = self._dependency_key("predictor", edge.key.key_id)
        else:
            dep_key = self._dependency_key("complete", edge.key.key_id)
        return (
            self._service_history.get(dep_key, []),
            self._service_exhaustion.get(dep_key),
        )

    def _delta_for_need_edge(
        self,
        service_name: str,
        edge: NeedScan | NeedPredictions | NeedCompleted,
    ) -> CompleterDelta | PredictorDelta | ScannerDelta | None:
        if service_name != "complete" or not isinstance(edge, NeedCompleted):
            return None
        new_keys: list[CompletionKey] = []
        if edge.key.key_id.prime in self._completion_keys_by_id:
            key = self._completion_keys_by_id[edge.key.key_id.prime]
            new_keys = [key]
        continuation = None
        support = None
        if edge.parent_completion_key_id is not None and edge.parent_candidate_id is not None:
            continuation = self._completion_continuations_by_id.get(
                self._completion_continuation_id(edge).prime
            )
            support = self._completion_supports_by_id.get(
                self._continuation_support_id(edge).prime
            )
        return CompleterDelta(
            delta_id=self._id_factor(
                "ttl_service_completer_delta",
                f"need|{edge.edge_id.prime}",
            ),
            request_id=edge.key.key_id,
            new_completion_key_ids=tuple(key.key_id for key in new_keys),
            new_discharged_completion_key_ids=(),
            new_continuation_ids=()
            if continuation is None
            else (continuation.continuation_id,),
            new_witness_ids=(),
            new_support_ids=() if support is None else (support.support_id,),
            new_completion_keys=tuple(new_keys),
            new_discharged_completion_keys=(),
            new_continuations=() if continuation is None else (continuation,),
            new_completed_witnesses=(),
            new_supports=() if support is None else (support,),
        )

    def _delta_for_output(
        self,
        service_name: str,
        yielded: ScanWitnessBatch | PredictionWitnessBatch | CompletedWitness | Exhausted,
    ) -> ScannerDelta | PredictorDelta | CompleterDelta | None:
        if isinstance(yielded, ScanWitnessBatch):
            key = self._scan_keys_by_id[yielded.key_id.prime]
            return ScannerDelta(
                delta_id=self._id_factor(
                    "ttl_service_scanner_delta",
                    f"{yielded.batch_id.prime}",
                ),
                request_id=yielded.key_id,
                new_scan_key_ids=(),
                new_truth_ids=tuple(truth.truth_id for truth in yielded.truths),
                new_ambiguity_class_ids=tuple(
                    ambiguity.ambiguity_class_id for ambiguity in yielded.ambiguity_classes
                ),
                new_scan_keys=(),
                new_truths=yielded.truths,
                new_ambiguity_classes=yielded.ambiguity_classes,
            )
        if isinstance(yielded, PredictionWitnessBatch):
            return PredictorDelta(
                delta_id=self._id_factor(
                    "ttl_service_predictor_delta",
                    f"{yielded.batch_id.prime}",
                ),
                request_id=yielded.key_id,
                new_prediction_key_ids=(),
                new_candidate_ids=tuple(candidate.candidate_id for candidate in yielded.candidates),
                new_prediction_keys=(),
                new_candidates=yielded.candidates,
            )
        if isinstance(yielded, CompletedWitness):
            key = self._completion_keys_by_id[yielded.key_id.prime]
            discharged = (key,)
            support = self._support_for_completed_witness(yielded)
            return CompleterDelta(
                delta_id=self._id_factor(
                    "ttl_service_completer_delta",
                    f"{yielded.witness_id.prime}",
                ),
                request_id=yielded.key_id,
                new_completion_key_ids=(),
                new_discharged_completion_key_ids=(key.key_id,),
                new_continuation_ids=(),
                new_witness_ids=(yielded.witness_id,),
                new_support_ids=(support.support_id,),
                new_completion_keys=(),
                new_discharged_completion_keys=discharged,
                new_continuations=(),
                new_completed_witnesses=(yielded,),
                new_supports=(support,),
            )
        return None

    def _make_service_emission(
        self,
        *,
        record: _ContinuationRecord,
        yielded: (
            NeedScan
            | NeedPredictions
            | NeedCompleted
            | ScanWitnessBatch
            | PredictionWitnessBatch
            | CompletedWitness
            | Exhausted
        ),
        delta: ScannerDelta | PredictorDelta | CompleterDelta | None,
        blocked_on: tuple[NeedScan | NeedPredictions | NeedCompleted, ...],
    ) -> ServiceEmission:
        if record.service_name == "scanner":
            stage = self._build_scanner_stage(record.key_id)
        elif record.service_name == "predictor":
            stage = self._build_predictor_stage(record.key_id)
        else:
            stage = self._build_completer_stage(record.key_id)
        return ServiceEmission(
            emission_id=self._id_factor(
                "ttl_service_emission",
                f"{record.service_name}|{record.continuation_id.prime}|{self._step_count + 1}",
            ),
            service_name=record.service_name,
            request_id=record.key_id,
            stage=stage,
            delta=delta,
            blocked_on=blocked_on,
            yielded=(yielded,),
        )

    def _build_scanner_stage(self, request_id: PrimeFactor) -> ScannerStage:
        truths = tuple(
            truth
            for history in self._service_history.values()
            for batch in history
            if isinstance(batch, ScanWitnessBatch)
            for truth in batch.truths
        )
        ambiguities = tuple(
            ambiguity
            for history in self._service_history.values()
            for batch in history
            if isinstance(batch, ScanWitnessBatch)
            for ambiguity in batch.ambiguity_classes
        )
        return ScannerStage(
            stage_id=self._id_factor(
                "ttl_service_scanner_stage",
                f"{request_id.prime}|{len(self._scan_keys_by_id)}|{len(truths)}|{len(ambiguities)}",
            ),
            request_id=request_id,
            scan_key_ids=frozenset(key.key_id for key in self._scan_keys_by_id.values()),
            truth_ids=frozenset(truth.truth_id for truth in truths),
            ambiguity_class_ids=frozenset(
                ambiguity.ambiguity_class_id for ambiguity in ambiguities
            ),
            scan_keys=tuple(self._scan_keys_by_id.values()),
            truths=truths,
            ambiguity_classes=ambiguities,
        )

    def _build_predictor_stage(self, request_id: PrimeFactor) -> PredictorStage:
        candidates = tuple(
            candidate
            for history in self._service_history.values()
            for batch in history
            if isinstance(batch, PredictionWitnessBatch)
            for candidate in batch.candidates
        )
        return PredictorStage(
            stage_id=self._id_factor(
                "ttl_service_predictor_stage",
                f"{request_id.prime}|{len(self._prediction_keys_by_id)}|{len(candidates)}",
            ),
            request_id=request_id,
            prediction_key_ids=frozenset(
                key.key_id for key in self._prediction_keys_by_id.values()
            ),
            candidate_ids=frozenset(candidate.candidate_id for candidate in candidates),
            prediction_keys=tuple(self._prediction_keys_by_id.values()),
            candidates=candidates,
        )

    def _build_completer_stage(self, request_id: PrimeFactor) -> CompleterStage:
        witnesses = tuple(
            witness
            for history in self._service_history.values()
            for witness in history
            if isinstance(witness, CompletedWitness)
        )
        discharged_keys = tuple(
            self._completion_keys_by_id[witness.key_id.prime]
            for witness in witnesses
            if witness.key_id.prime in self._completion_keys_by_id
        )
        return CompleterStage(
            stage_id=self._id_factor(
                "ttl_service_completer_stage",
                f"{request_id.prime}|{len(self._completion_keys_by_id)}|{len(discharged_keys)}|{len(self._completion_continuations_by_id)}|{len(witnesses)}|{len(self._completion_supports_by_id)}",
            ),
            request_id=request_id,
            completion_key_ids=frozenset(
                key.key_id for key in self._completion_keys_by_id.values()
            ),
            discharged_completion_key_ids=frozenset(
                key.key_id for key in discharged_keys
            ),
            continuation_ids=frozenset(
                continuation.continuation_id
                for continuation in self._completion_continuations_by_id.values()
            ),
            witness_ids=frozenset(witness.witness_id for witness in witnesses),
            support_ids=frozenset(
                support.support_id for support in self._completion_supports_by_id.values()
            ),
            completion_keys=tuple(self._completion_keys_by_id.values()),
            discharged_completion_keys=discharged_keys,
            continuations=tuple(self._completion_continuations_by_id.values()),
            completed_witnesses=witnesses,
            supports=tuple(self._completion_supports_by_id.values()),
        )

    def _build_global_stage(self, *, query_id: PrimeFactor) -> GlobalStage:
        return GlobalStage(
            stage_id=self._id_factor(
                "ttl_service_global_stage",
                f"{query_id.prime}|{len(self.scanner_trace)}|{len(self.predictor_trace)}|{len(self.completer_trace)}",
            ),
            query_id=query_id,
            scanner=self._build_scanner_stage(query_id),
            predictor=self._build_predictor_stage(query_id),
            completer=self._build_completer_stage(query_id),
        )

    def _record_emission(self, emission: ServiceEmission) -> None:
        self._step_count += 1
        if self._step_count > self.max_steps:
            raise RuntimeError(f"service-poset skeleton exceeded max steps {self.max_steps}")
        if emission.service_name == "scanner":
            self.scanner_trace.append(emission)
        elif emission.service_name == "predictor":
            self.predictor_trace.append(emission)
        elif emission.service_name == "complete":
            self.completer_trace.append(emission)
        else:
            raise ValueError(f"unknown service emission {emission.service_name}")

    def _scanner_continuation(
        self,
        continuation_id: PrimeFactor,
        key: ScanKey,
    ) -> Generator[ScanWitnessBatch | Exhausted, object, None]:
        start = 0 if key.start is None else key.start
        stop = len(self.lexemes) if key.end is None else min(key.end, len(self.lexemes))
        for index in range(start, stop):
            batch = self._scan_batch_for_site(key=key, index=index)
            if batch is None:
                continue
            yield batch
        yield self._make_exhausted("scanner", key.key_id, "scan_complete")

    def _predictor_continuation(
        self,
        continuation_id: PrimeFactor,
        key: PredictionKey,
    ) -> Generator[NeedScan | PredictionWitnessBatch | Exhausted, object, None]:
        scan_key = self._make_scan_key(
            start=key.start,
            end=key.end,
            expected_symbols=self._predictor_expected_symbols(key.symbol),
        )
        cursor = 0
        seen_candidate_ids: set[int] = set()
        while True:
            response = yield NeedScan(
                edge_id=self._id_factor(
                    "ttl_service_need_scan",
                    f"{continuation_id.prime}|{scan_key.key_id.prime}|{cursor}",
                ),
                parent_continuation_id=continuation_id,
                key=scan_key,
                cursor=cursor,
                requested_by_prediction_key_id=key.key_id,
            )
            if isinstance(response, Exhausted):
                yield self._make_exhausted("predictor", key.key_id, "prediction_complete")
                return
            if not isinstance(response, ScanWitnessBatch):
                raise TypeError("predictor continuation expected ScanWitnessBatch or Exhausted")
            cursor += 1
            new_candidates = [
                candidate
                for candidate in self._candidates_from_batch(key=key, batch=response)
                if candidate.candidate_id.prime not in seen_candidate_ids
            ]
            for candidate in new_candidates:
                seen_candidate_ids.add(candidate.candidate_id.prime)
            if not new_candidates:
                continue
            yield PredictionWitnessBatch(
                batch_id=self._id_factor(
                    "ttl_service_prediction_batch",
                    f"{key.key_id.prime}|{response.batch_id.prime}",
                ),
                key_id=key.key_id,
                candidates=tuple(new_candidates),
            )

    def _completer_continuation(
        self,
        continuation_id: PrimeFactor,
        key: CompletionKey,
    ) -> Generator[NeedPredictions | NeedCompleted | CompletedWitness | Exhausted, object, None]:
        prediction_key = self._make_prediction_key(
            symbol=key.symbol,
            start=key.start,
            end=key.end,
        )
        prediction_cursor = 0
        seen_candidate_ids: set[int] = set()
        while True:
            response = yield NeedPredictions(
                edge_id=self._id_factor(
                    "ttl_service_need_predictions",
                    f"{continuation_id.prime}|{prediction_key.key_id.prime}|{prediction_cursor}",
                ),
                parent_continuation_id=continuation_id,
                key=prediction_key,
                cursor=prediction_cursor,
                requested_by_completion_key_id=key.key_id,
            )
            if isinstance(response, Exhausted):
                yield self._make_exhausted("completer", key.key_id, "completion_complete")
                return
            if not isinstance(response, PredictionWitnessBatch):
                raise TypeError(
                    "completer continuation expected PredictionWitnessBatch or Exhausted"
                )
            prediction_cursor += 1
            for candidate in response.candidates:
                if candidate.candidate_id.prime in seen_candidate_ids:
                    continue
                seen_candidate_ids.add(candidate.candidate_id.prime)
                if candidate.residual_symbol is None:
                    if self._completion_satisfies(
                        key,
                        start=candidate.start,
                        stop=candidate.stop,
                    ):
                        yield self._make_completed_witness(
                            key=key,
                            candidate=candidate,
                            right_witness=None,
                        )
                        return
                    continue
                child_key = self._make_completion_key(
                    symbol=candidate.residual_symbol,
                    start=candidate.stop,
                    end=key.end,
                    require_complete=True,
                )
                child_response = yield NeedCompleted(
                    edge_id=self._id_factor(
                        "ttl_service_need_completed",
                        f"{continuation_id.prime}|{child_key.key_id.prime}|0|{candidate.candidate_id.prime}",
                    ),
                    parent_continuation_id=continuation_id,
                    key=child_key,
                    cursor=0,
                    parent_completion_key_id=key.key_id,
                    parent_candidate_id=candidate.candidate_id,
                )
                if not isinstance(child_response, CompletedWitness):
                    continue
                if self._completion_satisfies(
                    key,
                    start=candidate.start,
                    stop=child_response.stop,
                ):
                    yield self._make_completed_witness(
                        key=key,
                        candidate=candidate,
                        right_witness=child_response,
                    )
                    return

    def _scan_batch_for_site(
        self,
        *,
        key: ScanKey,
        index: int,
    ) -> ScanWitnessBatch | None:
        lexeme = self.lexemes[index]
        ambiguity_symbols = self._scanner_symbols_for_lexeme(lexeme)
        if key.expected_symbols:
            expected = set(key.expected_symbols)
            ambiguity_symbols = tuple(
                symbol_name
                for symbol_name in ambiguity_symbols
                if self.symbol_factor(f"terminal:{symbol_name}") in expected
            )
            if not ambiguity_symbols:
                return None
        site_id = self._id_factor("ttl_service_site", f"{index}|{index + 1}")
        ambiguity_class_id = self._id_factor(
            "ttl_service_scanner_ambiguity_class",
            f"{key.key_id.prime}|{index}|{'|'.join(ambiguity_symbols)}",
        )
        rel_path_id = self._id_factor("ttl_service_rel_path", lexeme.rel_path)
        truths: list[ScannerTruth] = []
        for symbol_name in ambiguity_symbols:
            symbol = self.symbol_factor(f"terminal:{symbol_name}")
            truths.append(
                ScannerTruth(
                    truth_id=self._id_factor(
                        "ttl_service_scanner_truth",
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
            )
        ambiguity = ScannerAmbiguityClass(
            ambiguity_class_id=ambiguity_class_id,
            site_id=site_id,
            truth_ids=tuple(truth.truth_id for truth in truths),
            start=index,
            stop=index + 1,
            lexeme_text=lexeme.text,
        )
        return ScanWitnessBatch(
            batch_id=self._id_factor(
                "ttl_service_scan_batch",
                f"{key.key_id.prime}|{index}",
            ),
            key_id=key.key_id,
            horizon=index + 1,
            truths=tuple(truths),
            ambiguity_classes=(ambiguity,),
        )

    def _candidates_from_batch(
        self,
        *,
        key: PredictionKey,
        batch: ScanWitnessBatch,
    ) -> tuple[PredictedCandidate, ...]:
        candidates: list[PredictedCandidate] = []
        for truth in batch.truths:
            if truth.symbol == key.symbol:
                lexical_rule = EarleyRule(head=key.symbol, rhs=(key.symbol,))
                candidates.append(
                    PredictedCandidate(
                        candidate_id=self._id_factor(
                            "ttl_service_predicted_candidate",
                            f"{key.key_id.prime}|terminal|{truth.truth_id.prime}",
                        ),
                        key_id=key.key_id,
                        rule=lexical_rule,
                        anchor_truth_id=truth.truth_id,
                        anchor_symbol=truth.symbol,
                        start=truth.start,
                        stop=truth.stop,
                        residual_symbol=None,
                    )
                )
            for rule in self.rules_by_head.get(key.symbol, ()):
                if rule.rhs[0] != truth.symbol:
                    continue
                candidates.append(
                    PredictedCandidate(
                        candidate_id=self._id_factor(
                            "ttl_service_predicted_candidate",
                            f"{key.key_id.prime}|rule|{rule.head.prime}|{truth.truth_id.prime}",
                        ),
                        key_id=key.key_id,
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
        expected: dict[int, PrimeFactor] = {}
        expected[symbol.prime] = symbol
        for rule in self.rules_by_head.get(symbol, ()):
            expected[rule.rhs[0].prime] = rule.rhs[0]
        return tuple(expected.values())

    def _make_completed_witness(
        self,
        *,
        key: CompletionKey,
        candidate: PredictedCandidate,
        right_witness: CompletedWitness | None,
    ) -> CompletedWitness:
        stop = candidate.stop if right_witness is None else right_witness.stop
        return CompletedWitness(
            witness_id=self._id_factor(
                "ttl_service_completed_witness",
                f"{key.key_id.prime}|{candidate.candidate_id.prime}|{None if right_witness is None else right_witness.witness_id.prime}",
            ),
            key_id=key.key_id,
            symbol=key.symbol,
            start=candidate.start,
            stop=stop,
            rule=candidate.rule,
            candidate_id=candidate.candidate_id,
            anchor_truth_id=candidate.anchor_truth_id,
            right_witness_id=None if right_witness is None else right_witness.witness_id,
        )

    def _register_completion_continuation(self, edge: NeedCompleted) -> None:
        self._completion_keys_by_id.setdefault(edge.key.key_id.prime, edge.key)
        if edge.parent_completion_key_id is None or edge.parent_candidate_id is None:
            return
        continuation_id = self._completion_continuation_id(edge)
        if continuation_id.prime not in self._completion_continuations_by_id:
            self._completion_continuations_by_id[continuation_id.prime] = CompletionContinuation(
                continuation_id=continuation_id,
                parent_completion_key_id=edge.parent_completion_key_id,
                child_completion_key_id=edge.key.key_id,
                parent_candidate_id=edge.parent_candidate_id,
                parent_continuation_id=edge.parent_continuation_id,
            )
        support_id = self._continuation_support_id(edge)
        if support_id.prime not in self._completion_supports_by_id:
            self._completion_supports_by_id[support_id.prime] = CompletionSupport(
                support_id=support_id,
                completion_key_id=edge.parent_completion_key_id,
                candidate_id=edge.parent_candidate_id,
                child_completion_key_id=edge.key.key_id,
                child_witness_id=None,
                kind="continuation",
            )

    def _support_for_completed_witness(self, witness: CompletedWitness) -> CompletionSupport:
        support = CompletionSupport(
            support_id=self._id_factor(
                "ttl_service_completion_support",
                f"{witness.key_id.prime}|{witness.candidate_id.prime}|{None if witness.right_witness_id is None else witness.right_witness_id.prime}|{witness.witness_id.prime}",
            ),
            completion_key_id=witness.key_id,
            candidate_id=witness.candidate_id,
            child_completion_key_id=None,
            child_witness_id=witness.right_witness_id,
            kind="unary" if witness.right_witness_id is None else "binary_join",
        )
        self._completion_supports_by_id.setdefault(support.support_id.prime, support)
        return support

    def _completion_continuation_id(self, edge: NeedCompleted) -> PrimeFactor:
        return self._id_factor(
            "ttl_service_completion_continuation",
            f"{edge.parent_completion_key_id.prime}|{edge.parent_candidate_id.prime}|{edge.key.key_id.prime}",
        )

    def _continuation_support_id(self, edge: NeedCompleted) -> PrimeFactor:
        return self._id_factor(
            "ttl_service_completion_support",
            f"{edge.parent_completion_key_id.prime}|{edge.parent_candidate_id.prime}|{edge.key.key_id.prime}|continuation",
        )

    def _completion_satisfies(
        self,
        key: CompletionKey,
        *,
        start: int,
        stop: int,
    ) -> bool:
        if key.start is not None and start != key.start:
            return False
        if key.end is not None and stop != key.end:
            return False
        return True

    def _scanner_symbols_for_lexeme(self, lexeme: TurtleLexeme) -> tuple[str, ...]:
        if lexeme.text == "a":
            return ("A", "NAME")
        return (lexeme.terminal_name,)

    def _make_scan_key(
        self,
        *,
        start: int | None,
        end: int | None,
        expected_symbols: tuple[PrimeFactor, ...],
    ) -> ScanKey:
        token = "|".join(str(symbol.prime) for symbol in expected_symbols)
        return ScanKey(
            key_id=self._id_factor(
                "ttl_service_scan_key",
                f"{start}|{end}|{token}",
            ),
            start=start,
            end=end,
            expected_symbols=expected_symbols,
        )

    def _make_prediction_key(
        self,
        *,
        symbol: PrimeFactor,
        start: int | None,
        end: int | None,
    ) -> PredictionKey:
        return PredictionKey(
            key_id=self._id_factor(
                "ttl_service_prediction_key",
                f"{symbol.prime}|{start}|{end}",
            ),
            symbol=symbol,
            start=start,
            end=end,
        )

    def _make_completion_key(
        self,
        *,
        symbol: PrimeFactor,
        start: int | None,
        end: int | None,
        require_complete: bool,
    ) -> CompletionKey:
        return CompletionKey(
            key_id=self._id_factor(
                "ttl_service_completion_key",
                f"{symbol.prime}|{start}|{end}|{int(require_complete)}",
            ),
            symbol=symbol,
            start=start,
            end=end,
            require_complete=require_complete,
        )

    def _make_exhausted(
        self,
        service_name: str,
        key_id: PrimeFactor,
        reason: str,
    ) -> Exhausted:
        return Exhausted(
            exhaustion_id=self._id_factor(
                "ttl_service_exhausted",
                f"{service_name}|{key_id.prime}|{reason}",
            ),
            service_name=service_name,
            key_id=key_id,
            reason=reason,
        )

    def _first_completed_for_key(self, key_id: PrimeFactor) -> CompletedWitness | None:
        history = self._service_history.get(self._dependency_key("complete", key_id), [])
        for item in history:
            if isinstance(item, CompletedWitness):
                return item
        return None

    def _dependency_key(self, service_name: str, key_id: PrimeFactor) -> tuple[str, int]:
        return service_name, key_id.prime

    def _scan_key_for_id(self, key_id: PrimeFactor) -> ScanKey:
        key = self._scan_keys_by_id.get(key_id.prime)
        if key is None:
            raise ValueError(f"missing scan key for {key_id.prime}")
        return key

    def _id_factor(self, namespace: str, token: str) -> PrimeFactor:
        return self.prime_space.intern_factor(
            identity_space=self.identity_space,
            namespace=namespace,
            token=token,
        )


@dataclass
class _ContinuationRun:
    router: ServicePosetRouter
    query: RootQuery

    def run(self) -> ServicePosetResult:
        root_key = self.router._make_completion_key(
            symbol=self.query.symbol,
            start=self.query.start,
            end=self.query.end,
            require_complete=self.query.require_complete,
        )
        final_stage, witness, exhaustion = self.router._run_router(root_key)
        return ServicePosetResult(
            source_paths=self.router.source_paths,
            lexemes=self.router.lexemes,
            grammar=self.router.grammar,
            root_query=self.query,
            scanner_trace=tuple(self.router.scanner_trace),
            predictor_trace=tuple(self.router.predictor_trace),
            completer_trace=tuple(self.router.completer_trace),
            final_global_stage=final_stage,
            first_completed_witness=witness,
            exhaustion=exhaustion,
            request_metrics=self.router.request_metrics,
        )


def run_turtle_service_poset_skeleton(
    *,
    symbol: str = "directive",
    start: int | None = None,
    end: int | None = None,
    root: Path = _REPO_ROOT,
    rel_paths: tuple[Path, ...] = _DEFAULT_TTL_PATHS,
    max_tokens: int | None = 256,
    max_steps: int = 10_000,
) -> ServicePosetResult:
    router = ServicePosetRouter.create(
        root=root,
        rel_paths=rel_paths,
        max_tokens=max_tokens,
        max_steps=max_steps,
    )
    query = router.make_root_query(symbol_name=symbol, start=start, end=end)
    return router.run_root_query(query)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demand-routed continuation-first Earley skeleton over the repo's kernel TTL files.",
    )
    parser.add_argument(
        "--symbol",
        default="directive",
        help="Grammar symbol to satisfy with a first completed witness.",
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
        help="Limit scanned TTL lexemes before running the service-poset skeleton.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10_000,
        help="Maximum emitted service steps before the experiment aborts.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the service-poset summary as JSON.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_turtle_service_poset_skeleton(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        max_tokens=args.max_tokens,
        max_steps=args.max_steps,
    )
    summary = result.as_summary()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
