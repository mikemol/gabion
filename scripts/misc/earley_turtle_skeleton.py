from __future__ import annotations

"""Experimental history-bound Earley skeleton over the repo's kernel TTL files.

This script keeps the parser experiment outside analysis core, but rewrites the
native carrier around the TTL history pattern:

- `HistoryLineage` is the append-only parse evolution.
- `HistoryState` is a frozen chart snapshot at rank `k`.
- `HistoryExtension` is one constructive append from rank `k` to `k + 1`.
- `HistoryStamp` names the finite closed universe at a fixed state.

Scanner tokens and chart items are the same immutable carrier species. Each
carrier keeps:

- a canonical identity projection,
- a prime-factor witness for the constructive operator that introduced it,
- an ASPF 1-cell witness over the same object,
- a history lineage / state / stamp boundary,
- lazy generator frontiers over frozen left/right structure.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
import argparse
import itertools
import json

from gabion.analysis.aspf.aspf_core import AspfOneCell, BasisZeroCell
from gabion.analysis.core.prime_identity_adapter import PrimeIdentityAdapter
from gabion.analysis.core.type_fingerprints import PrimeAssignmentEvent, PrimeRegistry
from gabion.analysis.foundation.identity_space import (
    GlobalIdentitySpace,
    IdentityProjection,
)
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
_EPSILON = "<eps>"


@dataclass(frozen=True)
class TurtleLexeme:
    rel_path: str
    offset: int
    kind: str
    text: str

    @property
    def terminal(self) -> str:
        if self.text in {"@prefix", "a", ".", ";", ",", "[", "]"}:
            return self.text
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
    lhs: str
    rhs: tuple[str, ...]
    rule_id: str


@dataclass(frozen=True)
class HistoryLineage:
    lineage_id: str
    source_paths: tuple[str, ...]
    lexeme_stream_id: str
    projection: IdentityProjection


@dataclass(frozen=True)
class HistoryStamp:
    stamp_id: str
    lineage_id: str
    state_rank: int
    digest_alias: str
    projection: IdentityProjection


@dataclass(frozen=True)
class HistoryExtension:
    extension_id: str
    lineage_id: str
    source_rank: int
    target_rank: int
    action_kind: str
    object_id: str
    target_stamp_id: str


@dataclass(frozen=True)
class HistoryBoundEarleyWindow:
    stream_id: str
    start: int
    stop: int
    cursor: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop < self.start or self.cursor < 0:
            raise ValueError("HistoryBoundEarleyWindow bounds must be non-negative and ordered.")

    def as_islice(self, lexemes: Iterable[TurtleLexeme]) -> Iterator[TurtleLexeme]:
        return itertools.islice(lexemes, self.start, self.stop)


@dataclass(frozen=True)
class HistoryBoundEarleyObject:
    carrier_id: str
    lineage_id: str
    state_rank: int
    stamp: HistoryStamp
    identified: str
    rule: EarleyRule
    window: HistoryBoundEarleyWindow
    prime_factor: PrimeFactor
    projection: IdentityProjection
    one_cell: AspfOneCell
    left: HistoryBoundEarleyObject | None = field(default=None, repr=False)
    right: HistoryBoundEarleyObject | None = field(default=None, repr=False)
    lexeme: TurtleLexeme | None = field(default=None, repr=False)

    @property
    def dot(self) -> int:
        return self.window.cursor

    @property
    def origin(self) -> int:
        return self.window.start

    @property
    def end(self) -> int:
        return self.window.stop

    @property
    def next_symbol(self) -> str | None:
        if self.dot >= len(self.rule.rhs):
            return None
        return self.rule.rhs[self.dot]

    @property
    def is_complete(self) -> bool:
        return self.dot >= len(self.rule.rhs)

    def iter_frontier_generators(self) -> Iterator[Iterator[HistoryBoundEarleyObject]]:
        if self.left is not None:
            yield self.left.iter_history()
        if self.right is not None:
            yield self.right.iter_history()

    def iter_history(self) -> Iterator[HistoryBoundEarleyObject]:
        stack = [self]
        while stack:
            current = stack.pop()
            yield current
            if current.right is not None:
                stack.append(current.right)
            if current.left is not None:
                stack.append(current.left)


@dataclass(frozen=True)
class EarleyChartColumn:
    index: int
    column_id: str
    _items: tuple[HistoryBoundEarleyObject, ...]
    item_ids: frozenset[str]

    def items(self) -> tuple[HistoryBoundEarleyObject, ...]:
        return self._items


@dataclass(frozen=True)
class HistoryState:
    rank: int
    lineage: HistoryLineage
    stamp: HistoryStamp
    chart: tuple[EarleyChartColumn, ...]
    last_extension: HistoryExtension | None = None

    @property
    def item_count(self) -> int:
        return sum(len(column.item_ids) for column in self.chart)


@dataclass(frozen=True)
class EarleyParseResult:
    source_paths: tuple[str, ...]
    lexemes: tuple[TurtleLexeme, ...]
    tokens: tuple[HistoryBoundEarleyObject, ...]
    grammar: tuple[EarleyRule, ...]
    lineage: HistoryLineage
    states: tuple[HistoryState, ...]

    @property
    def final_state(self) -> HistoryState:
        return self.states[-1]

    @property
    def chart(self) -> tuple[EarleyChartColumn, ...]:
        return self.final_state.chart

    @property
    def item_count(self) -> int:
        return self.final_state.item_count

    def as_summary(self) -> dict[str, object]:
        return {
            "source_paths": list(self.source_paths),
            "lexeme_count": len(self.lexemes),
            "token_count": len(self.tokens),
            "chart_column_count": len(self.chart),
            "item_count": self.item_count,
            "lineage_id": self.lineage.lineage_id,
            "state_count": len(self.states),
            "final_state_rank": self.final_state.rank,
            "final_stamp_id": self.final_state.stamp.stamp_id,
            "completed_start_items": [
                item.carrier_id
                for item in self.chart[-1].items()
                if item.rule.lhs == "document'" and item.is_complete and item.origin == 0
            ],
            "columns": [
                {
                    "index": column.index,
                    "item_count": len(column.item_ids),
                    "sample_items": [
                        {
                            "carrier_id": item.carrier_id,
                            "rule_id": item.rule.rule_id,
                            "lhs": item.rule.lhs,
                            "rhs": list(item.rule.rhs),
                            "dot": item.dot,
                            "origin": item.origin,
                            "end": item.end,
                            "state_rank": item.state_rank,
                            "stamp_id": item.stamp.stamp_id,
                            "prime_chain": [factor.prime for factor in item.prime_factor.iter_chain()],
                            "prime_product": item.projection.prime_product,
                            "basis_path_atoms": list(item.projection.basis_path.atoms),
                            "aspf_basis_path": list(item.one_cell.basis_path),
                        }
                        for item in list(column.items())[:5]
                    ],
                }
                for column in self.chart
            ],
        }


@dataclass(frozen=True)
class _EarleyBlueprint:
    carrier_id: str
    identified: str
    rule: EarleyRule
    window: HistoryBoundEarleyWindow
    prime_factor: PrimeFactor
    projection: IdentityProjection
    one_cell: AspfOneCell
    left: HistoryBoundEarleyObject | None
    right: HistoryBoundEarleyObject | None
    lexeme: TurtleLexeme | None

    def materialize(
        self,
        *,
        lineage_id: str,
        state_rank: int,
        stamp: HistoryStamp,
    ) -> HistoryBoundEarleyObject:
        return HistoryBoundEarleyObject(
            carrier_id=self.carrier_id,
            lineage_id=lineage_id,
            state_rank=state_rank,
            stamp=stamp,
            identified=self.identified,
            rule=self.rule,
            window=self.window,
            prime_factor=self.prime_factor,
            projection=self.projection,
            one_cell=self.one_cell,
            left=self.left,
            right=self.right,
            lexeme=self.lexeme,
        )


@dataclass
class _PrimeFactorSpace:
    registry: PrimeRegistry
    _factors_by_prime: dict[int, PrimeFactor] = field(default_factory=dict)
    _last_factor: PrimeFactor | int = 1

    def attach(self) -> int:
        return self.registry.register_assignment_observer(self._observe)

    def _observe(self, event: PrimeAssignmentEvent) -> None:
        previous = self._last_factor
        factor = PrimeFactor(
            prime=event.atom_id,
            previous=previous,
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


@dataclass
class _HistoryBoundEarleyBuilder:
    identity_space: GlobalIdentitySpace
    prime_space: _PrimeFactorSpace
    source_paths: tuple[str, ...]
    lexemes: tuple[TurtleLexeme, ...]
    grammar: tuple[EarleyRule, ...]
    lineage: HistoryLineage
    columns: list[list[HistoryBoundEarleyObject]]
    column_ids: list[set[str]]
    states: list[HistoryState]

    @classmethod
    def create(
        cls,
        *,
        identity_space: GlobalIdentitySpace,
        prime_space: _PrimeFactorSpace,
        source_paths: tuple[str, ...],
        lexemes: tuple[TurtleLexeme, ...],
        grammar: tuple[EarleyRule, ...],
    ) -> _HistoryBoundEarleyBuilder:
        lineage = _build_history_lineage(
            identity_space=identity_space,
            source_paths=source_paths,
            lexemes=lexemes,
        )
        columns = [[] for _ in range(len(lexemes) + 1)]
        column_ids = [set() for _ in range(len(lexemes) + 1)]
        builder = cls(
            identity_space=identity_space,
            prime_space=prime_space,
            source_paths=source_paths,
            lexemes=lexemes,
            grammar=grammar,
            lineage=lineage,
            columns=columns,
            column_ids=column_ids,
            states=[],
        )
        builder.states.append(builder._root_state())
        return builder

    def token_objects(self) -> tuple[HistoryBoundEarleyObject, ...]:
        root_stamp = self.states[0].stamp
        out: list[HistoryBoundEarleyObject] = []
        for index, lexeme in enumerate(self.lexemes):
            token_rule = EarleyRule(
                lhs=lexeme.terminal,
                rhs=(),
                rule_id=f"token.{lexeme.terminal}",
            )
            window = HistoryBoundEarleyWindow(
                stream_id=self.lineage.lexeme_stream_id,
                start=index,
                stop=index + 1,
                cursor=0,
            )
            blueprint = self._make_blueprint(
                identified=f"{lexeme.rel_path}:{lexeme.offset}:{lexeme.terminal}",
                rule=token_rule,
                window=window,
                operator_token=f"token:{lexeme.terminal}:{index}",
                left=None,
                right=None,
                lexeme=lexeme,
            )
            out.append(
                blueprint.materialize(
                    lineage_id=self.lineage.lineage_id,
                    state_rank=0,
                    stamp=root_stamp,
                )
            )
        return tuple(out)

    def add_blueprint(
        self,
        *,
        column_index: int,
        blueprint: _EarleyBlueprint,
        action_kind: str,
    ) -> HistoryBoundEarleyObject | None:
        if blueprint.carrier_id in self.column_ids[column_index]:
            return None
        next_rank = len(self.states)
        stamp = self._prospective_stamp(
            rank=next_rank,
            column_index=column_index,
            carrier_id=blueprint.carrier_id,
        )
        item = blueprint.materialize(
            lineage_id=self.lineage.lineage_id,
            state_rank=next_rank,
            stamp=stamp,
        )
        self.columns[column_index].append(item)
        self.column_ids[column_index].add(item.carrier_id)
        extension_projection = self.identity_space.project(
            path=self.identity_space.intern_path(
                namespace="ttl_history_extension",
                tokens=(
                    self.lineage.lineage_id,
                    f"source:{self.states[-1].rank}",
                    f"target:{next_rank}",
                    f"action:{action_kind}",
                    blueprint.carrier_id,
                ),
            )
        )
        extension = HistoryExtension(
            extension_id=extension_projection.digest_alias,
            lineage_id=self.lineage.lineage_id,
            source_rank=self.states[-1].rank,
            target_rank=next_rank,
            action_kind=action_kind,
            object_id=blueprint.carrier_id,
            target_stamp_id=stamp.stamp_id,
        )
        self.states.append(
            HistoryState(
                rank=next_rank,
                lineage=self.lineage,
                stamp=stamp,
                chart=self._freeze_chart(),
                last_extension=extension,
            )
        )
        return item

    def _root_state(self) -> HistoryState:
        stamp = self._stamp_from_snapshot(rank=0, snapshot=self._snapshot_item_ids())
        return HistoryState(
            rank=0,
            lineage=self.lineage,
            stamp=stamp,
            chart=self._freeze_chart(),
            last_extension=None,
        )

    def _make_blueprint(
        self,
        *,
        identified: str,
        rule: EarleyRule,
        window: HistoryBoundEarleyWindow,
        operator_token: str,
        left: HistoryBoundEarleyObject | None,
        right: HistoryBoundEarleyObject | None,
        lexeme: TurtleLexeme | None,
    ) -> _EarleyBlueprint:
        operator_atom = self.identity_space.intern_atom(
            namespace="ttl_earley_operator",
            token=operator_token,
        )
        prime_factor = self.prime_space.factor_for(operator_atom)
        projection = self._canonical_projection(
            identified=identified,
            rule=rule,
            window=window,
            lexeme=lexeme,
        )
        path_tokens = list(self._canonical_tokens(identified=identified, rule=rule, window=window, lexeme=lexeme))
        if left is not None:
            path_tokens.append(f"lhs:{left.carrier_id}")
        if right is not None:
            path_tokens.append(f"rhs:{right.carrier_id}")
        path_tokens.append(f"operator:{prime_factor.prime}")
        one_cell = AspfOneCell(
            source=BasisZeroCell(f"chart:{window.start}"),
            target=BasisZeroCell(f"chart:{window.stop}"),
            representative=projection.digest_alias,
            basis_path=tuple(path_tokens),
        )
        return _EarleyBlueprint(
            carrier_id=projection.digest_alias,
            identified=identified,
            rule=rule,
            window=window,
            prime_factor=prime_factor,
            projection=projection,
            one_cell=one_cell,
            left=left,
            right=right,
            lexeme=lexeme,
        )

    def _canonical_projection(
        self,
        *,
        identified: str,
        rule: EarleyRule,
        window: HistoryBoundEarleyWindow,
        lexeme: TurtleLexeme | None,
    ) -> IdentityProjection:
        return self.identity_space.project(
            path=self.identity_space.intern_path(
                namespace="ttl_history_bound_earley_object",
                tokens=self._canonical_tokens(
                    identified=identified,
                    rule=rule,
                    window=window,
                    lexeme=lexeme,
                ),
            )
        )

    def _canonical_tokens(
        self,
        *,
        identified: str,
        rule: EarleyRule,
        window: HistoryBoundEarleyWindow,
        lexeme: TurtleLexeme | None,
    ) -> tuple[str, ...]:
        lexeme_tokens: tuple[str, ...]
        if lexeme is None:
            lexeme_tokens = ()
        else:
            lexeme_tokens = (
                f"lexeme_path:{lexeme.rel_path}",
                f"lexeme_offset:{lexeme.offset}",
                f"lexeme_kind:{lexeme.kind}",
                f"lexeme_text:{lexeme.text}",
            )
        return (
            f"lineage:{self.lineage.lineage_id}",
            f"rule:{rule.rule_id}",
            f"lhs:{rule.lhs}",
            f"window:{window.start}:{window.stop}:{window.cursor}",
            f"identified:{identified}",
            *lexeme_tokens,
        )

    def _prospective_stamp(
        self,
        *,
        rank: int,
        column_index: int,
        carrier_id: str,
    ) -> HistoryStamp:
        snapshot = list(map(list, self._snapshot_item_ids()))
        snapshot[column_index].append(carrier_id)
        return self._stamp_from_snapshot(
            rank=rank,
            snapshot=tuple(tuple(column) for column in snapshot),
        )

    def _stamp_from_snapshot(
        self,
        *,
        rank: int,
        snapshot: tuple[tuple[str, ...], ...],
    ) -> HistoryStamp:
        stamp_projection = self.identity_space.project(
            path=self.identity_space.intern_path(
                namespace="ttl_history_stamp",
                tokens=(
                    self.lineage.lineage_id,
                    f"rank:{rank}",
                    *(
                        f"column:{index}:{'|'.join(column_ids) if column_ids else 'empty'}"
                        for index, column_ids in enumerate(snapshot)
                    ),
                ),
            )
        )
        return HistoryStamp(
            stamp_id=stamp_projection.digest_alias,
            lineage_id=self.lineage.lineage_id,
            state_rank=rank,
            digest_alias=stamp_projection.digest_alias,
            projection=stamp_projection,
        )

    def _snapshot_item_ids(self) -> tuple[tuple[str, ...], ...]:
        return tuple(
            tuple(item.carrier_id for item in column)
            for column in self.columns
        )

    def _freeze_chart(self) -> tuple[EarleyChartColumn, ...]:
        out: list[EarleyChartColumn] = []
        for index, column in enumerate(self.columns):
            column_ids = tuple(item.carrier_id for item in column)
            column_projection = self.identity_space.project(
                path=self.identity_space.intern_path(
                    namespace="ttl_history_chart_column",
                    tokens=(
                        self.lineage.lineage_id,
                        f"index:{index}",
                        *column_ids,
                    ),
                )
            )
            out.append(
                EarleyChartColumn(
                    index=index,
                    column_id=column_projection.digest_alias,
                    _items=tuple(column),
                    item_ids=frozenset(column_ids),
                )
            )
        return tuple(out)


def _build_history_lineage(
    *,
    identity_space: GlobalIdentitySpace,
    source_paths: tuple[str, ...],
    lexemes: tuple[TurtleLexeme, ...],
) -> HistoryLineage:
    lineage_projection = identity_space.project(
        path=identity_space.intern_path(
            namespace="ttl_history_lineage",
            tokens=("kernel_ttl", *source_paths),
        )
    )
    lexeme_stream_projection = identity_space.project(
        path=identity_space.intern_path(
            namespace="ttl_lexeme_stream",
            tokens=tuple(
                f"{lexeme.rel_path}:{lexeme.offset}:{lexeme.terminal}:{lexeme.text}"
                for lexeme in lexemes
            ),
        )
    )
    return HistoryLineage(
        lineage_id=lineage_projection.digest_alias,
        source_paths=source_paths,
        lexeme_stream_id=lexeme_stream_projection.digest_alias,
        projection=lineage_projection,
    )


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


def build_turtle_skeleton_grammar() -> tuple[EarleyRule, ...]:
    return (
        EarleyRule("document'", ("document",), "document'.0"),
        EarleyRule("document", ("statement_list",), "document.0"),
        EarleyRule("statement_list", ("statement", "statement_list"), "statement_list.0"),
        EarleyRule("statement_list", ("statement",), "statement_list.1"),
        EarleyRule("statement", ("directive",), "statement.0"),
        EarleyRule("statement", ("triple_stmt",), "statement.1"),
        EarleyRule("directive", ("@prefix", "NAME", "IRI", "."), "directive.0"),
        EarleyRule("triple_stmt", ("subject", "predicate_tail", "."), "triple_stmt.0"),
        EarleyRule("subject", ("NAME",), "subject.0"),
        EarleyRule("subject", ("IRI",), "subject.1"),
        EarleyRule("subject", ("blank_node",), "subject.2"),
        EarleyRule("predicate_tail", ("verb", "object_list", "predicate_more"), "predicate_tail.0"),
        EarleyRule("predicate_more", (";", "verb", "object_list", "predicate_more"), "predicate_more.0"),
        EarleyRule("predicate_more", (_EPSILON,), "predicate_more.1"),
        EarleyRule("verb", ("NAME",), "verb.0"),
        EarleyRule("verb", ("a",), "verb.1"),
        EarleyRule("object_list", ("object", "object_more"), "object_list.0"),
        EarleyRule("object_more", (",", "object", "object_more"), "object_more.0"),
        EarleyRule("object_more", (_EPSILON,), "object_more.1"),
        EarleyRule("object", ("NAME",), "object.0"),
        EarleyRule("object", ("IRI",), "object.1"),
        EarleyRule("object", ("STRING",), "object.2"),
        EarleyRule("object", ("NUMBER",), "object.3"),
        EarleyRule("object", ("blank_node",), "object.4"),
        EarleyRule("blank_node", ("[", "]"), "blank_node.0"),
        EarleyRule("blank_node", ("[", "predicate_tail", "]"), "blank_node.1"),
    )


def _is_nonterminal(symbol: str, grammar_index: dict[str, tuple[EarleyRule, ...]]) -> bool:
    return symbol in grammar_index


def run_kernel_turtle_earley_skeleton(
    *,
    root: Path = _REPO_ROOT,
    rel_paths: tuple[Path, ...] = _DEFAULT_TTL_PATHS,
    max_tokens: int | None = 256,
) -> EarleyParseResult:
    lexemes = load_kernel_ttl_lexemes(root=root, rel_paths=rel_paths)
    if max_tokens is not None:
        lexemes = lexemes[:max_tokens]
    grammar = build_turtle_skeleton_grammar()
    grammar_index: dict[str, tuple[EarleyRule, ...]] = {}
    for rule in grammar:
        grammar_index[rule.lhs] = (*grammar_index.get(rule.lhs, ()), rule)

    with deadline_scope(Deadline.from_timeout_ms(30_000)):
        with deadline_clock_scope(MonotonicClock()):
            registry = PrimeRegistry()
            prime_space = _PrimeFactorSpace(registry=registry)
            observer_id = prime_space.attach()
            try:
                identity_space = GlobalIdentitySpace(
                    allocator=PrimeIdentityAdapter(registry=registry)
                )
                builder = _HistoryBoundEarleyBuilder.create(
                    identity_space=identity_space,
                    prime_space=prime_space,
                    source_paths=tuple(path.as_posix() for path in rel_paths),
                    lexemes=lexemes,
                    grammar=grammar,
                )
                token_objects = builder.token_objects()
                start_blueprint = builder._make_blueprint(
                    identified="document':0:0:0",
                    rule=grammar[0],
                    window=HistoryBoundEarleyWindow(
                        stream_id=builder.lineage.lexeme_stream_id,
                        start=0,
                        stop=0,
                        cursor=0,
                    ),
                    operator_token="seed:document'.0",
                    left=None,
                    right=None,
                    lexeme=None,
                )
                builder.add_blueprint(
                    column_index=0,
                    blueprint=start_blueprint,
                    action_kind="seed",
                )

                for column_index in range(len(builder.columns)):
                    agenda_index = 0
                    while agenda_index < len(builder.columns[column_index]):
                        current = builder.columns[column_index][agenda_index]
                        agenda_index += 1
                        next_symbol = current.next_symbol
                        if next_symbol is None:
                            for candidate in tuple(builder.columns[current.origin]):
                                if candidate.next_symbol != current.rule.lhs:
                                    continue
                                builder.add_blueprint(
                                    column_index=column_index,
                                    blueprint=builder._make_blueprint(
                                        identified=(
                                            f"{candidate.rule.rule_id}:"
                                            f"{candidate.origin}:{column_index}:{candidate.dot + 1}"
                                        ),
                                        rule=candidate.rule,
                                        window=HistoryBoundEarleyWindow(
                                            stream_id=builder.lineage.lexeme_stream_id,
                                            start=candidate.origin,
                                            stop=column_index,
                                            cursor=candidate.dot + 1,
                                        ),
                                        operator_token=(
                                            f"complete:{candidate.rule.rule_id}:"
                                            f"{candidate.dot + 1}:{column_index}"
                                        ),
                                        left=candidate,
                                        right=current,
                                        lexeme=None,
                                    ),
                                    action_kind="complete",
                                )
                            continue
                        if next_symbol == _EPSILON:
                            builder.add_blueprint(
                                column_index=column_index,
                                blueprint=builder._make_blueprint(
                                    identified=(
                                        f"{current.rule.rule_id}:"
                                        f"{current.origin}:{column_index}:{current.dot + 1}"
                                    ),
                                    rule=current.rule,
                                    window=HistoryBoundEarleyWindow(
                                        stream_id=builder.lineage.lexeme_stream_id,
                                        start=current.origin,
                                        stop=column_index,
                                        cursor=current.dot + 1,
                                    ),
                                    operator_token=(
                                        f"epsilon:{current.rule.rule_id}:{current.dot + 1}"
                                    ),
                                    left=current,
                                    right=None,
                                    lexeme=None,
                                ),
                                action_kind="epsilon",
                            )
                            continue
                        if _is_nonterminal(next_symbol, grammar_index):
                            for predicted in grammar_index[next_symbol]:
                                builder.add_blueprint(
                                    column_index=column_index,
                                    blueprint=builder._make_blueprint(
                                        identified=f"{predicted.rule_id}:{column_index}:{column_index}:0",
                                        rule=predicted,
                                        window=HistoryBoundEarleyWindow(
                                            stream_id=builder.lineage.lexeme_stream_id,
                                            start=column_index,
                                            stop=column_index,
                                            cursor=0,
                                        ),
                                        operator_token=f"predict:{predicted.rule_id}:{column_index}",
                                        left=current,
                                        right=None,
                                        lexeme=None,
                                    ),
                                    action_kind="predict",
                                )
                            continue
                        if column_index >= len(token_objects):
                            continue
                        token_object = token_objects[column_index]
                        if token_object.rule.lhs != next_symbol:
                            continue
                        builder.add_blueprint(
                            column_index=column_index + 1,
                            blueprint=builder._make_blueprint(
                                identified=(
                                    f"{current.rule.rule_id}:"
                                    f"{current.origin}:{column_index + 1}:{current.dot + 1}"
                                ),
                                rule=current.rule,
                                window=HistoryBoundEarleyWindow(
                                    stream_id=builder.lineage.lexeme_stream_id,
                                    start=current.origin,
                                    stop=column_index + 1,
                                    cursor=current.dot + 1,
                                ),
                                operator_token=(
                                    f"scan:{current.rule.rule_id}:{next_symbol}:{column_index}"
                                ),
                                left=current,
                                right=token_object,
                                lexeme=None,
                            ),
                            action_kind="scan",
                        )
            finally:
                registry.unregister_assignment_observer(observer_id)

    return EarleyParseResult(
        source_paths=tuple(path.as_posix() for path in rel_paths),
        lexemes=lexemes,
        tokens=token_objects,
        grammar=grammar,
        lineage=builder.lineage,
        states=tuple(builder.states),
    )


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="History-bound Earley skeleton over the repo's kernel TTL files.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Limit scanned TTL lexemes before running the Earley chart.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the parse summary as JSON.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_kernel_turtle_earley_skeleton(max_tokens=args.max_tokens)
    summary = result.as_summary()
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
