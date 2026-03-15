from __future__ import annotations

"""Demand-driven service-poset Earley skeleton over the repo's kernel TTL files.

This experiment treats parsing as a small network of memoized services:

- scanner: monotone observational stages over packed lexical truths
- predictor: monotone candidate stages anchored on scanner truths
- completer: monotone fulfillment stages over pending obligations and witnesses

The root query is generic and returns the first completed witness that satisfies
the requested symbol and optional span constraints.
"""

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
class NeedScan:
    request_id: PrimeFactor
    start: int | None
    end: int | None


@dataclass(frozen=True)
class NeedPredictions:
    request_id: PrimeFactor
    symbol: PrimeFactor
    start: int | None
    end: int | None


@dataclass(frozen=True)
class NeedCompleted:
    request_id: PrimeFactor
    symbol: PrimeFactor
    start: int | None
    end: int | None
    require_complete: bool = True


@dataclass(frozen=True)
class Exhausted:
    exhaustion_id: PrimeFactor
    service_name: str
    request_id: PrimeFactor
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
    request_id: PrimeFactor
    rule: EarleyRule
    anchor_truth_id: PrimeFactor
    anchor_symbol: PrimeFactor
    start: int
    stop: int
    residual_symbol: PrimeFactor | None


@dataclass(frozen=True)
class CompletedWitness:
    witness_id: PrimeFactor
    request_id: PrimeFactor
    symbol: PrimeFactor
    start: int
    stop: int
    rule: EarleyRule
    candidate_id: PrimeFactor
    anchor_truth_id: PrimeFactor
    right_witness_id: PrimeFactor | None


@dataclass(frozen=True)
class ScannerStage:
    stage_id: PrimeFactor
    request_id: PrimeFactor
    horizon: int
    truth_ids: frozenset[PrimeFactor]
    ambiguity_class_ids: frozenset[PrimeFactor]
    truths: tuple[ScannerTruth, ...]
    ambiguity_classes: tuple[ScannerAmbiguityClass, ...]


@dataclass(frozen=True)
class ScannerDelta:
    delta_id: PrimeFactor
    request_id: PrimeFactor
    horizon: int
    new_truth_ids: tuple[PrimeFactor, ...]
    new_ambiguity_class_ids: tuple[PrimeFactor, ...]
    new_truths: tuple[ScannerTruth, ...]
    new_ambiguity_classes: tuple[ScannerAmbiguityClass, ...]


@dataclass(frozen=True)
class PredictorStage:
    stage_id: PrimeFactor
    request_id: PrimeFactor
    candidate_ids: frozenset[PrimeFactor]
    candidates: tuple[PredictedCandidate, ...]


@dataclass(frozen=True)
class PredictorDelta:
    delta_id: PrimeFactor
    request_id: PrimeFactor
    new_candidate_ids: tuple[PrimeFactor, ...]
    new_candidates: tuple[PredictedCandidate, ...]


@dataclass(frozen=True)
class CompleterStage:
    stage_id: PrimeFactor
    request_id: PrimeFactor
    completed_ids: frozenset[PrimeFactor]
    pending_ids: frozenset[PrimeFactor]
    completed_witnesses: tuple[CompletedWitness, ...]
    pending_obligations: tuple[NeedCompleted, ...]


@dataclass(frozen=True)
class CompleterDelta:
    delta_id: PrimeFactor
    request_id: PrimeFactor
    new_completed_ids: tuple[PrimeFactor, ...]
    new_pending_ids: tuple[PrimeFactor, ...]
    new_completed_witnesses: tuple[CompletedWitness, ...]
    new_pending_obligations: tuple[NeedCompleted, ...]


@dataclass(frozen=True)
class ServiceEmission:
    emission_id: PrimeFactor
    service_name: str
    request_id: PrimeFactor
    stage: ScannerStage | PredictorStage | CompleterStage
    delta: ScannerDelta | PredictorDelta | CompleterDelta | None
    blocked_on: NeedScan | NeedPredictions | NeedCompleted | None
    yielded: tuple[ScannerDelta | PredictorDelta | CompleterDelta | CompletedWitness | Exhausted, ...]


@dataclass(frozen=True)
class ServicePosetResult:
    source_paths: tuple[str, ...]
    lexemes: tuple[TurtleLexeme, ...]
    grammar: tuple[EarleyRule, ...]
    root_query: RootQuery
    scanner_trace: tuple[ServiceEmission, ...]
    predictor_trace: tuple[ServiceEmission, ...]
    completer_trace: tuple[ServiceEmission, ...]
    first_completed_witness: CompletedWitness | None
    exhaustion: Exhausted | None
    request_metrics: dict[str, tuple[int, int]]

    def as_summary(self) -> dict[str, object]:
        ambiguity_samples: list[dict[str, object]] = []
        for emission in self.scanner_trace:
            if emission.delta is None:
                continue
            for ambiguity in emission.delta.new_ambiguity_classes:
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

        predicted_samples = []
        for emission in self.predictor_trace:
            if emission.delta is None:
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
                for candidate in emission.delta.new_candidates[:3]
            )
            if len(predicted_samples) >= 5:
                predicted_samples = predicted_samples[:5]
                break

        completed_samples = []
        for emission in self.completer_trace:
            if emission.delta is None:
                continue
            completed_samples.extend(
                {
                    "witness_prime": witness.witness_id.prime,
                    "symbol": witness.symbol.token,
                    "start": witness.start,
                    "stop": witness.stop,
                }
                for witness in emission.delta.new_completed_witnesses[:3]
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

        self.scanner_trace: list[ServiceEmission] = []
        self.predictor_trace: list[ServiceEmission] = []
        self.completer_trace: list[ServiceEmission] = []

        self._scan_cache: dict[int, tuple[ScannerDelta | Exhausted, ...]] = {}
        self._predict_cache: dict[int, tuple[PredictorDelta | Exhausted, ...]] = {}
        self._complete_cache: dict[int, CompletedWitness | Exhausted] = {}
        self._scan_stage_cache: dict[int, ScannerStage] = {}
        self._predict_stage_cache: dict[int, PredictorStage] = {}
        self._metrics = {
            "scan": [0, 0],
            "predict": [0, 0],
            "complete": [0, 0],
        }
        self._step_count = 0

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
                return NeedScan(
                    request_id=self._id_factor("ttl_service_need_scan", f"{start}|{end}"),
                    start=start,
                    end=end,
                )

    def _make_prediction_request(
        self,
        *,
        symbol: PrimeFactor,
        start: int | None,
        end: int | None,
    ) -> NeedPredictions:
        return NeedPredictions(
            request_id=self._id_factor(
                "ttl_service_need_predictions",
                f"{symbol.prime}|{start}|{end}",
            ),
            symbol=symbol,
            start=start,
            end=end,
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
                return self._make_completion_request(
                    symbol=self.symbol_factor(symbol_name),
                    start=start,
                    end=end,
                    require_complete=require_complete,
                )

    def _make_completion_request(
        self,
        *,
        symbol: PrimeFactor,
        start: int | None,
        end: int | None,
        require_complete: bool,
    ) -> NeedCompleted:
        return NeedCompleted(
            request_id=self._id_factor(
                "ttl_service_need_completed",
                f"{symbol.prime}|{start}|{end}|{int(require_complete)}",
            ),
            symbol=symbol,
            start=start,
            end=end,
            require_complete=require_complete,
        )

    def stream_scan(self, request: NeedScan) -> tuple[ScannerDelta | Exhausted, ...]:
        with deadline_scope(Deadline.from_timeout_ms(30_000)):
            with deadline_clock_scope(MonotonicClock()):
                return tuple(self._stream_scan(request))

    def resolve_query(self, query: RootQuery) -> CompletedWitness | None:
        with deadline_scope(Deadline.from_timeout_ms(30_000)):
            with deadline_clock_scope(MonotonicClock()):
                request = self._make_completion_request(
                    symbol=query.symbol,
                    start=query.start,
                    end=query.end,
                    require_complete=query.require_complete,
                )
                result = self._resolve_completed(request)
                if isinstance(result, CompletedWitness):
                    return result
                return None

    def run_root_query(self, query: RootQuery) -> ServicePosetResult:
        with deadline_scope(Deadline.from_timeout_ms(30_000)):
            with deadline_clock_scope(MonotonicClock()):
                result = self._resolve_completed(
                    self._make_completion_request(
                        symbol=query.symbol,
                        start=query.start,
                        end=query.end,
                        require_complete=query.require_complete,
                    )
                )
                return ServicePosetResult(
                    source_paths=self.source_paths,
                    lexemes=self.lexemes,
                    grammar=self.grammar,
                    root_query=query,
                    scanner_trace=tuple(self.scanner_trace),
                    predictor_trace=tuple(self.predictor_trace),
                    completer_trace=tuple(self.completer_trace),
                    first_completed_witness=result if isinstance(result, CompletedWitness) else None,
                    exhaustion=result if isinstance(result, Exhausted) else None,
                    request_metrics=self.request_metrics,
                )

    def _stream_scan(self, request: NeedScan) -> Iterator[ScannerDelta | Exhausted]:
        self._metrics["scan"][0] += 1
        cached = self._scan_cache.get(request.request_id.prime)
        if cached is not None:
            yield from cached
            return
        self._metrics["scan"][1] += 1
        responses: list[ScannerDelta | Exhausted] = []
        for emission in self._scanner_service(request):
            self._record_emission(emission)
            self._scan_stage_cache[request.request_id.prime] = emission.stage
            if isinstance(emission.delta, ScannerDelta):
                responses.append(emission.delta)
                yield emission.delta
            for obj in emission.yielded:
                if isinstance(obj, Exhausted):
                    responses.append(obj)
                    yield obj
        self._scan_cache[request.request_id.prime] = tuple(responses)

    def _stream_predictions(
        self,
        request: NeedPredictions,
    ) -> Iterator[PredictorDelta | Exhausted]:
        self._metrics["predict"][0] += 1
        cached = self._predict_cache.get(request.request_id.prime)
        if cached is not None:
            yield from cached
            return
        self._metrics["predict"][1] += 1
        responses: list[PredictorDelta | Exhausted] = []
        generator = self._predictor_service(request)
        emission = next(generator)
        scan_stream: Iterator[ScannerDelta | Exhausted] | None = None
        while True:
            self._record_emission(emission)
            self._predict_stage_cache[request.request_id.prime] = emission.stage
            if isinstance(emission.delta, PredictorDelta):
                responses.append(emission.delta)
                yield emission.delta
            for obj in emission.yielded:
                if isinstance(obj, Exhausted):
                    responses.append(obj)
                    yield obj
                    self._predict_cache[request.request_id.prime] = tuple(responses)
                    return
            blocked = emission.blocked_on
            if blocked is None:
                self._predict_cache[request.request_id.prime] = tuple(responses)
                return
            if scan_stream is None:
                scan_stream = iter(self._stream_scan(blocked))
            try:
                response = next(scan_stream)
            except StopIteration:
                response = self._make_exhausted("scanner", blocked.request_id, "scanner_stream_ended")
            try:
                emission = generator.send(response)
            except StopIteration:
                self._predict_cache[request.request_id.prime] = tuple(responses)
                return

    def _resolve_completed(
        self,
        request: NeedCompleted,
    ) -> CompletedWitness | Exhausted:
        self._metrics["complete"][0] += 1
        cached = self._complete_cache.get(request.request_id.prime)
        if cached is not None:
            return cached
        self._metrics["complete"][1] += 1
        generator = self._completer_service(request)
        emission = next(generator)
        prediction_stream: Iterator[PredictorDelta | Exhausted] | None = None
        while True:
            self._record_emission(emission)
            for obj in emission.yielded:
                if isinstance(obj, (CompletedWitness, Exhausted)):
                    self._complete_cache[request.request_id.prime] = obj
                    return obj
            blocked = emission.blocked_on
            if blocked is None:
                exhausted = self._make_exhausted("completer", request.request_id, "no_witness")
                self._complete_cache[request.request_id.prime] = exhausted
                return exhausted
            if isinstance(blocked, NeedPredictions):
                if prediction_stream is None:
                    prediction_stream = iter(self._stream_predictions(blocked))
                try:
                    response = next(prediction_stream)
                except StopIteration:
                    response = self._make_exhausted(
                        "predictor",
                        blocked.request_id,
                        "prediction_stream_ended",
                    )
            else:
                response = self._resolve_completed(blocked)
            try:
                emission = generator.send(response)
            except StopIteration as stop:
                result = stop.value
                if isinstance(result, (CompletedWitness, Exhausted)):
                    self._complete_cache[request.request_id.prime] = result
                    return result
                exhausted = self._make_exhausted("completer", request.request_id, "generator_stopped")
                self._complete_cache[request.request_id.prime] = exhausted
                return exhausted

    def _scanner_service(self, request: NeedScan) -> Iterator[ServiceEmission]:
        truths_by_id: dict[int, ScannerTruth] = {}
        classes_by_id: dict[int, ScannerAmbiguityClass] = {}
        start = 0 if request.start is None else request.start
        stop = len(self.lexemes) if request.end is None else min(request.end, len(self.lexemes))
        stage = ScannerStage(
            stage_id=self._id_factor(
                "ttl_service_scanner_stage",
                f"{request.request_id.prime}|0|0|0",
            ),
            request_id=request.request_id,
            horizon=start,
            truth_ids=frozenset(),
            ambiguity_class_ids=frozenset(),
            truths=(),
            ambiguity_classes=(),
        )
        for index in range(start, stop):
            lexeme = self.lexemes[index]
            truth_ids: list[PrimeFactor] = []
            new_truths: list[ScannerTruth] = []
            ambiguity_symbols = self._scanner_symbols_for_lexeme(lexeme)
            site_id = self._id_factor("ttl_service_site", f"{index}|{index + 1}")
            ambiguity_class_id = self._id_factor(
                "ttl_service_scanner_ambiguity_class",
                f"{request.request_id.prime}|{index}|{'|'.join(ambiguity_symbols)}",
            )
            rel_path_id = self._id_factor("ttl_service_rel_path", lexeme.rel_path)
            for symbol_name in ambiguity_symbols:
                symbol = self.symbol_factor(f"terminal:{symbol_name}")
                truth = ScannerTruth(
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
                truths_by_id[truth.truth_id.prime] = truth
                truth_ids.append(truth.truth_id)
                new_truths.append(truth)
            ambiguity_class = ScannerAmbiguityClass(
                ambiguity_class_id=ambiguity_class_id,
                site_id=site_id,
                truth_ids=tuple(truth_ids),
                start=index,
                stop=index + 1,
                lexeme_text=lexeme.text,
            )
            classes_by_id[ambiguity_class_id.prime] = ambiguity_class
            stage = ScannerStage(
                stage_id=self._id_factor(
                    "ttl_service_scanner_stage",
                    f"{request.request_id.prime}|{index + 1}|{len(truths_by_id)}|{len(classes_by_id)}",
                ),
                request_id=request.request_id,
                horizon=index + 1,
                truth_ids=frozenset(truth.truth_id for truth in truths_by_id.values()),
                ambiguity_class_ids=frozenset(
                    ambiguity.ambiguity_class_id for ambiguity in classes_by_id.values()
                ),
                truths=tuple(truths_by_id.values()),
                ambiguity_classes=tuple(classes_by_id.values()),
            )
            delta = ScannerDelta(
                delta_id=self._id_factor(
                    "ttl_service_scanner_delta",
                    f"{request.request_id.prime}|{index + 1}",
                ),
                request_id=request.request_id,
                horizon=index + 1,
                new_truth_ids=tuple(truth.truth_id for truth in new_truths),
                new_ambiguity_class_ids=(ambiguity_class.ambiguity_class_id,),
                new_truths=tuple(new_truths),
                new_ambiguity_classes=(ambiguity_class,),
            )
            yield ServiceEmission(
                emission_id=self._id_factor(
                    "ttl_service_scanner_emission",
                    f"{request.request_id.prime}|{index + 1}",
                ),
                service_name="scanner",
                request_id=request.request_id,
                stage=stage,
                delta=delta,
                blocked_on=None,
                yielded=(delta,),
            )
        final_stage = stage
        yield ServiceEmission(
            emission_id=self._id_factor(
                "ttl_service_scanner_emission",
                f"{request.request_id.prime}|exhausted",
            ),
            service_name="scanner",
            request_id=request.request_id,
            stage=final_stage,
            delta=None,
            blocked_on=None,
            yielded=(self._make_exhausted("scanner", request.request_id, "scan_complete"),),
        )

    def _predictor_service(
        self,
        request: NeedPredictions,
    ) -> Generator[ServiceEmission, ScannerDelta | Exhausted, None]:
        truths_by_id: dict[int, ScannerTruth] = {}
        candidates_by_id: dict[int, PredictedCandidate] = {}
        scan_request = self.make_scan_request(start=request.start, end=request.end)
        stage = PredictorStage(
            stage_id=self._id_factor("ttl_service_predictor_stage", f"{request.request_id.prime}|0"),
            request_id=request.request_id,
            candidate_ids=frozenset(),
            candidates=(),
        )
        response = yield ServiceEmission(
            emission_id=self._id_factor(
                "ttl_service_predictor_emission",
                f"{request.request_id.prime}|initial",
            ),
            service_name="predictor",
            request_id=request.request_id,
            stage=stage,
            delta=None,
            blocked_on=scan_request,
            yielded=(),
        )
        while True:
            if isinstance(response, Exhausted):
                yield ServiceEmission(
                    emission_id=self._id_factor(
                        "ttl_service_predictor_emission",
                        f"{request.request_id.prime}|exhausted",
                    ),
                    service_name="predictor",
                    request_id=request.request_id,
                    stage=stage,
                    delta=None,
                    blocked_on=None,
                    yielded=(self._make_exhausted("predictor", request.request_id, "prediction_complete"),),
                )
                return
            if not isinstance(response, ScannerDelta):
                raise TypeError("predictor_service expected ScannerDelta or Exhausted")

            for truth in response.new_truths:
                truths_by_id[truth.truth_id.prime] = truth

            new_candidates = self._candidates_from_stage(
                request=request,
                truths=tuple(truths_by_id.values()),
            )
            delta_candidates: list[PredictedCandidate] = []
            for candidate in new_candidates:
                if candidate.candidate_id.prime in candidates_by_id:
                    continue
                candidates_by_id[candidate.candidate_id.prime] = candidate
                delta_candidates.append(candidate)
            if delta_candidates:
                stage = PredictorStage(
                    stage_id=self._id_factor(
                        "ttl_service_predictor_stage",
                        f"{request.request_id.prime}|{len(candidates_by_id)}",
                    ),
                    request_id=request.request_id,
                    candidate_ids=frozenset(candidate.candidate_id for candidate in candidates_by_id.values()),
                    candidates=tuple(candidates_by_id.values()),
                )
                delta: PredictorDelta | None = PredictorDelta(
                    delta_id=self._id_factor(
                        "ttl_service_predictor_delta",
                        f"{request.request_id.prime}|{len(candidates_by_id)}",
                    ),
                    request_id=request.request_id,
                    new_candidate_ids=tuple(candidate.candidate_id for candidate in delta_candidates),
                    new_candidates=tuple(delta_candidates),
                )
            else:
                delta = None
            response = yield ServiceEmission(
                emission_id=self._id_factor(
                    "ttl_service_predictor_emission",
                    f"{request.request_id.prime}|{response.delta_id.prime}",
                ),
                service_name="predictor",
                request_id=request.request_id,
                stage=stage,
                delta=delta,
                blocked_on=scan_request,
                yielded=() if delta is None else (delta,),
            )

    def _completer_service(
        self,
        request: NeedCompleted,
    ) -> Generator[ServiceEmission, PredictorDelta | CompletedWitness | Exhausted, CompletedWitness | Exhausted]:
        completed_by_id: dict[int, CompletedWitness] = {}
        pending_by_id: dict[int, NeedCompleted] = {}
        stage = CompleterStage(
            stage_id=self._id_factor("ttl_service_completer_stage", f"{request.request_id.prime}|0|0"),
            request_id=request.request_id,
            completed_ids=frozenset(),
            pending_ids=frozenset(),
            completed_witnesses=(),
            pending_obligations=(),
        )
        prediction_request = self._make_prediction_request(
            symbol=request.symbol,
            start=request.start,
            end=request.end,
        )
        response = yield ServiceEmission(
            emission_id=self._id_factor(
                "ttl_service_completer_emission",
                f"{request.request_id.prime}|initial",
            ),
            service_name="completer",
            request_id=request.request_id,
            stage=stage,
            delta=None,
            blocked_on=prediction_request,
            yielded=(),
        )
        while True:
            if isinstance(response, Exhausted):
                exhausted = self._make_exhausted("completer", request.request_id, "completion_complete")
                yield ServiceEmission(
                    emission_id=self._id_factor(
                        "ttl_service_completer_emission",
                        f"{request.request_id.prime}|exhausted",
                    ),
                    service_name="completer",
                    request_id=request.request_id,
                    stage=stage,
                    delta=None,
                    blocked_on=None,
                    yielded=(exhausted,),
                )
                return exhausted
            if not isinstance(response, PredictorDelta):
                raise TypeError("completer_service expected PredictorDelta or Exhausted")

            for candidate in response.new_candidates:
                if candidate.residual_symbol is None:
                    if self._completion_satisfies(request, candidate.start, candidate.stop):
                        witness = self._make_completed_witness(
                            request=request,
                            candidate=candidate,
                            right_witness=None,
                        )
                        completed_by_id[witness.witness_id.prime] = witness
                        stage = CompleterStage(
                            stage_id=self._id_factor(
                                "ttl_service_completer_stage",
                                f"{request.request_id.prime}|{len(completed_by_id)}|{len(pending_by_id)}",
                            ),
                            request_id=request.request_id,
                            completed_ids=frozenset(
                                witness.witness_id for witness in completed_by_id.values()
                            ),
                            pending_ids=frozenset(
                                obligation.request_id for obligation in pending_by_id.values()
                            ),
                            completed_witnesses=tuple(completed_by_id.values()),
                            pending_obligations=tuple(pending_by_id.values()),
                        )
                        delta = CompleterDelta(
                            delta_id=self._id_factor(
                                "ttl_service_completer_delta",
                                f"{request.request_id.prime}|complete|{witness.witness_id.prime}",
                            ),
                            request_id=request.request_id,
                            new_completed_ids=(witness.witness_id,),
                            new_pending_ids=(),
                            new_completed_witnesses=(witness,),
                            new_pending_obligations=(),
                        )
                        yield ServiceEmission(
                            emission_id=self._id_factor(
                                "ttl_service_completer_emission",
                                f"{request.request_id.prime}|witness|{witness.witness_id.prime}",
                            ),
                            service_name="completer",
                            request_id=request.request_id,
                            stage=stage,
                            delta=delta,
                            blocked_on=None,
                            yielded=(witness,),
                        )
                        return witness
                    continue

                subrequest = self._make_completion_request(
                    symbol=candidate.residual_symbol,
                    start=candidate.stop,
                    end=request.end,
                    require_complete=True,
                )
                delta: CompleterDelta | None
                if subrequest.request_id.prime not in pending_by_id:
                    pending_by_id[subrequest.request_id.prime] = subrequest
                    stage = CompleterStage(
                        stage_id=self._id_factor(
                            "ttl_service_completer_stage",
                            f"{request.request_id.prime}|{len(completed_by_id)}|{len(pending_by_id)}",
                        ),
                        request_id=request.request_id,
                        completed_ids=frozenset(
                            witness.witness_id for witness in completed_by_id.values()
                        ),
                        pending_ids=frozenset(
                            obligation.request_id for obligation in pending_by_id.values()
                        ),
                        completed_witnesses=tuple(completed_by_id.values()),
                        pending_obligations=tuple(pending_by_id.values()),
                    )
                    delta = CompleterDelta(
                        delta_id=self._id_factor(
                            "ttl_service_completer_delta",
                            f"{request.request_id.prime}|pending|{subrequest.request_id.prime}",
                        ),
                        request_id=request.request_id,
                        new_completed_ids=(),
                        new_pending_ids=(subrequest.request_id,),
                        new_completed_witnesses=(),
                        new_pending_obligations=(subrequest,),
                    )
                else:
                    delta = None
                subresponse = yield ServiceEmission(
                    emission_id=self._id_factor(
                        "ttl_service_completer_emission",
                        f"{request.request_id.prime}|pending|{subrequest.request_id.prime}",
                    ),
                    service_name="completer",
                    request_id=request.request_id,
                    stage=stage,
                    delta=delta,
                    blocked_on=subrequest,
                    yielded=() if delta is None else (delta,),
                )
                if isinstance(subresponse, CompletedWitness):
                    stop = subresponse.stop
                    if self._completion_satisfies(request, candidate.start, stop):
                        witness = self._make_completed_witness(
                            request=request,
                            candidate=candidate,
                            right_witness=subresponse,
                        )
                        if witness.witness_id.prime not in completed_by_id:
                            completed_by_id[witness.witness_id.prime] = witness
                        stage = CompleterStage(
                            stage_id=self._id_factor(
                                "ttl_service_completer_stage",
                                f"{request.request_id.prime}|{len(completed_by_id)}|{len(pending_by_id)}",
                            ),
                            request_id=request.request_id,
                            completed_ids=frozenset(
                                witness.witness_id for witness in completed_by_id.values()
                            ),
                            pending_ids=frozenset(
                                obligation.request_id for obligation in pending_by_id.values()
                            ),
                            completed_witnesses=tuple(completed_by_id.values()),
                            pending_obligations=tuple(pending_by_id.values()),
                        )
                        delta = CompleterDelta(
                            delta_id=self._id_factor(
                                "ttl_service_completer_delta",
                                f"{request.request_id.prime}|complete|{witness.witness_id.prime}",
                            ),
                            request_id=request.request_id,
                            new_completed_ids=(witness.witness_id,),
                            new_pending_ids=(),
                            new_completed_witnesses=(witness,),
                            new_pending_obligations=(),
                        )
                        yield ServiceEmission(
                            emission_id=self._id_factor(
                                "ttl_service_completer_emission",
                                f"{request.request_id.prime}|witness|{witness.witness_id.prime}",
                            ),
                            service_name="completer",
                            request_id=request.request_id,
                            stage=stage,
                            delta=delta,
                            blocked_on=None,
                            yielded=(witness,),
                        )
                        return witness
            response = yield ServiceEmission(
                emission_id=self._id_factor(
                    "ttl_service_completer_emission",
                    f"{request.request_id.prime}|continue|{response.delta_id.prime}",
                ),
                service_name="completer",
                request_id=request.request_id,
                stage=stage,
                delta=None,
                blocked_on=prediction_request,
                yielded=(),
            )

    def _candidates_from_stage(
        self,
        *,
        request: NeedPredictions,
        truths: tuple[ScannerTruth, ...],
    ) -> tuple[PredictedCandidate, ...]:
        candidates: list[PredictedCandidate] = []
        for truth in truths:
            if truth.symbol == request.symbol:
                lexical_rule = EarleyRule(head=request.symbol, rhs=(request.symbol,))
                candidates.append(
                    PredictedCandidate(
                        candidate_id=self._id_factor(
                            "ttl_service_predicted_candidate",
                            f"{request.request_id.prime}|terminal|{truth.truth_id.prime}",
                        ),
                        request_id=request.request_id,
                        rule=lexical_rule,
                        anchor_truth_id=truth.truth_id,
                        anchor_symbol=truth.symbol,
                        start=truth.start,
                        stop=truth.stop,
                        residual_symbol=None,
                    )
                )
            for rule in self.rules_by_head.get(request.symbol, ()):
                if rule.rhs[0] != truth.symbol:
                    continue
                candidates.append(
                    PredictedCandidate(
                        candidate_id=self._id_factor(
                            "ttl_service_predicted_candidate",
                            f"{request.request_id.prime}|rule|{rule.head.prime}|{truth.truth_id.prime}",
                        ),
                        request_id=request.request_id,
                        rule=rule,
                        anchor_truth_id=truth.truth_id,
                        anchor_symbol=truth.symbol,
                        start=truth.start,
                        stop=truth.stop,
                        residual_symbol=None if rule.is_unary else rule.rhs[1],
                    )
                )
        return tuple(candidates)

    def _make_completed_witness(
        self,
        *,
        request: NeedCompleted,
        candidate: PredictedCandidate,
        right_witness: CompletedWitness | None,
    ) -> CompletedWitness:
        stop = candidate.stop if right_witness is None else right_witness.stop
        return CompletedWitness(
            witness_id=self._id_factor(
                "ttl_service_completed_witness",
                f"{request.request_id.prime}|{candidate.candidate_id.prime}|{None if right_witness is None else right_witness.witness_id.prime}",
            ),
            request_id=request.request_id,
            symbol=request.symbol,
            start=candidate.start,
            stop=stop,
            rule=candidate.rule,
            candidate_id=candidate.candidate_id,
            anchor_truth_id=candidate.anchor_truth_id,
            right_witness_id=None if right_witness is None else right_witness.witness_id,
        )

    def _completion_satisfies(
        self,
        request: NeedCompleted,
        start: int,
        stop: int,
    ) -> bool:
        if request.start is not None and start != request.start:
            return False
        if request.end is not None and stop != request.end:
            return False
        return True

    def _scanner_symbols_for_lexeme(self, lexeme: TurtleLexeme) -> tuple[str, ...]:
        if lexeme.text == "a":
            return ("A", "NAME")
        return (lexeme.terminal_name,)

    def _make_exhausted(
        self,
        service_name: str,
        request_id: PrimeFactor,
        reason: str,
    ) -> Exhausted:
        return Exhausted(
            exhaustion_id=self._id_factor(
                "ttl_service_exhausted",
                f"{service_name}|{request_id.prime}|{reason}",
            ),
            service_name=service_name,
            request_id=request_id,
            reason=reason,
        )

    def _record_emission(self, emission: ServiceEmission) -> None:
        self._step_count += 1
        if self._step_count > self.max_steps:
            raise RuntimeError(f"service-poset skeleton exceeded max steps {self.max_steps}")
        if emission.service_name == "scanner":
            self.scanner_trace.append(emission)
        elif emission.service_name == "predictor":
            self.predictor_trace.append(emission)
        elif emission.service_name == "completer":
            self.completer_trace.append(emission)
        else:
            raise ValueError(f"unknown service emission {emission.service_name}")

    def _id_factor(self, namespace: str, token: str) -> PrimeFactor:
        return self.prime_space.intern_factor(
            identity_space=self.identity_space,
            namespace=namespace,
            token=token,
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
        description="Demand-driven service-poset Earley skeleton over the repo's kernel TTL files.",
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
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
