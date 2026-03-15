from __future__ import annotations

"""Experimental history-bound Earley skeleton over the repo's kernel TTL files.

This experiment treats:

- history snapshots as immutable closed universes,
- scanner tokens and chart items as the same carrier species,
- all `_id` fields as prime-backed identity objects,
- grammar rules as self-identifying objects with CNF-2 right-hand sides.
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
from gabion.analysis.foundation.identity_space import GlobalIdentitySpace, IdentityProjection
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
    def is_lexical(self) -> bool:
        return len(self.rhs) == 1


@dataclass(frozen=True)
class HistoryLineage:
    lineage_id: PrimeFactor
    source_paths: tuple[str, ...]
    lexeme_stream_id: PrimeFactor
    projection: IdentityProjection


@dataclass(frozen=True)
class HistoryStamp:
    stamp_id: PrimeFactor
    lineage: HistoryLineage
    state_rank: int
    projection: IdentityProjection


@dataclass(frozen=True)
class HistoryExtension:
    extension_id: PrimeFactor
    lineage: HistoryLineage
    source_rank: int
    target_rank: int
    action: PrimeFactor
    object_id: PrimeFactor
    target_stamp: HistoryStamp


@dataclass(frozen=True)
class HistoryBoundEarleyWindow:
    stream_id: PrimeFactor
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
    carrier_id: PrimeFactor
    lineage: HistoryLineage
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
    def next_symbol(self) -> PrimeFactor | None:
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
    column_id: PrimeFactor
    _items: tuple[HistoryBoundEarleyObject, ...]
    item_ids: frozenset[PrimeFactor]

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
            "lineage_prime": self.lineage.lineage_id.prime,
            "state_count": len(self.states),
            "final_state_rank": self.final_state.rank,
            "final_stamp_prime": self.final_state.stamp.stamp_id.prime,
            "completed_start_items": [
                item.carrier_id.prime
                for item in self.chart[-1].items()
                if item.rule.head.token == "document'" and item.is_complete and item.origin == 0
            ],
            "columns": [
                {
                    "index": column.index,
                    "column_prime": column.column_id.prime,
                    "item_count": len(column.item_ids),
                    "sample_items": [
                        {
                            "carrier_prime": item.carrier_id.prime,
                            "head": item.rule.head.token,
                            "rhs": [symbol.token for symbol in item.rule.rhs],
                            "dot": item.dot,
                            "origin": item.origin,
                            "end": item.end,
                            "state_rank": item.state_rank,
                            "stamp_prime": item.stamp.stamp_id.prime,
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
    carrier_id: PrimeFactor
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
        lineage: HistoryLineage,
        state_rank: int,
        stamp: HistoryStamp,
    ) -> HistoryBoundEarleyObject:
        return HistoryBoundEarleyObject(
            carrier_id=self.carrier_id,
            lineage=lineage,
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

    def intern_factor(
        self,
        *,
        identity_space: GlobalIdentitySpace,
        namespace: str,
        token: str,
    ) -> PrimeFactor:
        atom_id = identity_space.intern_atom(namespace=namespace, token=token)
        return self.factor_for(atom_id)


@dataclass
class _HistoryBoundEarleyBuilder:
    identity_space: GlobalIdentitySpace
    prime_space: _PrimeFactorSpace
    source_paths: tuple[str, ...]
    lexemes: tuple[TurtleLexeme, ...]
    grammar: tuple[EarleyRule, ...]
    lineage: HistoryLineage
    columns: list[list[HistoryBoundEarleyObject]]
    column_ids: list[set[PrimeFactor]]
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
            prime_space=prime_space,
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
            terminal_factor = _terminal_factor(
                identity_space=self.identity_space,
                prime_space=self.prime_space,
                terminal_name=lexeme.terminal_name,
            )
            token_rule = EarleyRule(
                head=terminal_factor,
                rhs=(terminal_factor,),
            )
            window = HistoryBoundEarleyWindow(
                stream_id=self.lineage.lexeme_stream_id,
                start=index,
                stop=index + 1,
                cursor=1,
            )
            blueprint = self._make_blueprint(
                identified=f"{lexeme.rel_path}:{lexeme.offset}:{lexeme.terminal_name}",
                rule=token_rule,
                window=window,
                operator_token=f"token:{lexeme.terminal_name}:{index}",
                left=None,
                right=None,
                lexeme=lexeme,
            )
            out.append(
                blueprint.materialize(
                    lineage=self.lineage,
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
        action_token: str,
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
            lineage=self.lineage,
            state_rank=next_rank,
            stamp=stamp,
        )
        self.columns[column_index].append(item)
        self.column_ids[column_index].add(item.carrier_id)
        action_factor = self.prime_space.intern_factor(
            identity_space=self.identity_space,
            namespace="ttl_history_action",
            token=action_token,
        )
        extension_projection = self.identity_space.project(
            path=self.identity_space.intern_path(
                namespace="ttl_history_extension",
                tokens=(
                    f"lineage:{self.lineage.lineage_id.prime}",
                    f"source:{self.states[-1].rank}",
                    f"target:{next_rank}",
                    f"action:{action_factor.prime}",
                    f"object:{blueprint.carrier_id.prime}",
                ),
            )
        )
        extension_id = self.prime_space.intern_factor(
            identity_space=self.identity_space,
            namespace="ttl_history_extension_id",
            token=extension_projection.digest_alias,
        )
        extension = HistoryExtension(
            extension_id=extension_id,
            lineage=self.lineage,
            source_rank=self.states[-1].rank,
            target_rank=next_rank,
            action=action_factor,
            object_id=blueprint.carrier_id,
            target_stamp=stamp,
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
        operator_factor = self.prime_space.intern_factor(
            identity_space=self.identity_space,
            namespace="ttl_earley_operator",
            token=operator_token,
        )
        projection = self._canonical_projection(
            identified=identified,
            rule=rule,
            window=window,
            lexeme=lexeme,
        )
        carrier_id = self.prime_space.intern_factor(
            identity_space=self.identity_space,
            namespace="ttl_history_bound_earley_object_id",
            token=projection.digest_alias,
        )
        path_tokens = list(
            self._canonical_tokens(
                identified=identified,
                rule=rule,
                window=window,
                lexeme=lexeme,
            )
        )
        if left is not None:
            path_tokens.append(f"lhs:{left.carrier_id.prime}")
        if right is not None:
            path_tokens.append(f"rhs:{right.carrier_id.prime}")
        path_tokens.append(f"operator:{operator_factor.prime}")
        one_cell = AspfOneCell(
            source=BasisZeroCell(f"chart:{window.start}"),
            target=BasisZeroCell(f"chart:{window.stop}"),
            representative=projection.digest_alias,
            basis_path=tuple(path_tokens),
        )
        return _EarleyBlueprint(
            carrier_id=carrier_id,
            identified=identified,
            rule=rule,
            window=window,
            prime_factor=operator_factor,
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
        rhs_tokens = tuple(f"rhs:{index}:{factor.prime}" for index, factor in enumerate(rule.rhs))
        return (
            f"lineage:{self.lineage.lineage_id.prime}",
            f"head:{rule.head.prime}",
            *rhs_tokens,
            f"window:{window.start}:{window.stop}:{window.cursor}",
            f"stream:{window.stream_id.prime}",
            f"identified:{identified}",
            *lexeme_tokens,
        )

    def _prospective_stamp(
        self,
        *,
        rank: int,
        column_index: int,
        carrier_id: PrimeFactor,
    ) -> HistoryStamp:
        snapshot = [list(column) for column in self._snapshot_item_ids()]
        snapshot[column_index].append(carrier_id)
        return self._stamp_from_snapshot(
            rank=rank,
            snapshot=tuple(tuple(column) for column in snapshot),
        )

    def _stamp_from_snapshot(
        self,
        *,
        rank: int,
        snapshot: tuple[tuple[PrimeFactor, ...], ...],
    ) -> HistoryStamp:
        stamp_projection = self.identity_space.project(
            path=self.identity_space.intern_path(
                namespace="ttl_history_stamp",
                tokens=(
                    f"lineage:{self.lineage.lineage_id.prime}",
                    f"rank:{rank}",
                    *(
                        f"column:{index}:{'|'.join(str(item_id.prime) for item_id in column_ids) if column_ids else 'empty'}"
                        for index, column_ids in enumerate(snapshot)
                    ),
                ),
            )
        )
        stamp_id = self.prime_space.intern_factor(
            identity_space=self.identity_space,
            namespace="ttl_history_stamp_id",
            token=stamp_projection.digest_alias,
        )
        return HistoryStamp(
            stamp_id=stamp_id,
            lineage=self.lineage,
            state_rank=rank,
            projection=stamp_projection,
        )

    def _snapshot_item_ids(self) -> tuple[tuple[PrimeFactor, ...], ...]:
        return tuple(tuple(item.carrier_id for item in column) for column in self.columns)

    def _freeze_chart(self) -> tuple[EarleyChartColumn, ...]:
        out: list[EarleyChartColumn] = []
        for index, column in enumerate(self.columns):
            column_ids = tuple(item.carrier_id for item in column)
            column_projection = self.identity_space.project(
                path=self.identity_space.intern_path(
                    namespace="ttl_history_chart_column",
                    tokens=(
                        f"lineage:{self.lineage.lineage_id.prime}",
                        f"index:{index}",
                        *(f"item:{item_id.prime}" for item_id in column_ids),
                    ),
                )
            )
            column_id = self.prime_space.intern_factor(
                identity_space=self.identity_space,
                namespace="ttl_history_chart_column_id",
                token=column_projection.digest_alias,
            )
            out.append(
                EarleyChartColumn(
                    index=index,
                    column_id=column_id,
                    _items=tuple(column),
                    item_ids=frozenset(column_ids),
                )
            )
        return tuple(out)


def _build_history_lineage(
    *,
    identity_space: GlobalIdentitySpace,
    prime_space: _PrimeFactorSpace,
    source_paths: tuple[str, ...],
    lexemes: tuple[TurtleLexeme, ...],
) -> HistoryLineage:
    lineage_projection = identity_space.project(
        path=identity_space.intern_path(
            namespace="ttl_history_lineage",
            tokens=("kernel_ttl", *source_paths),
        )
    )
    lineage_id = prime_space.intern_factor(
        identity_space=identity_space,
        namespace="ttl_history_lineage_id",
        token=lineage_projection.digest_alias,
    )
    lexeme_stream_projection = identity_space.project(
        path=identity_space.intern_path(
            namespace="ttl_lexeme_stream",
            tokens=tuple(
                f"{lexeme.rel_path}:{lexeme.offset}:{lexeme.terminal_name}:{lexeme.text}"
                for lexeme in lexemes
            ),
        )
    )
    lexeme_stream_id = prime_space.intern_factor(
        identity_space=identity_space,
        namespace="ttl_lexeme_stream_id",
        token=lexeme_stream_projection.digest_alias,
    )
    return HistoryLineage(
        lineage_id=lineage_id,
        source_paths=source_paths,
        lexeme_stream_id=lexeme_stream_id,
        projection=lineage_projection,
    )


def _symbol_factor(
    *,
    identity_space: GlobalIdentitySpace,
    prime_space: _PrimeFactorSpace,
    name: str,
) -> PrimeFactor:
    return prime_space.intern_factor(
        identity_space=identity_space,
        namespace="ttl_earley_symbol",
        token=name,
    )


def _terminal_factor(
    *,
    identity_space: GlobalIdentitySpace,
    prime_space: _PrimeFactorSpace,
    terminal_name: str,
) -> PrimeFactor:
    return _symbol_factor(
        identity_space=identity_space,
        prime_space=prime_space,
        name=f"terminal:{terminal_name}",
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


def build_turtle_skeleton_grammar(
    *,
    identity_space: GlobalIdentitySpace,
    prime_space: _PrimeFactorSpace,
) -> tuple[EarleyRule, ...]:
    document = _symbol_factor(identity_space=identity_space, prime_space=prime_space, name="document'")
    directive = _symbol_factor(identity_space=identity_space, prime_space=prime_space, name="directive")
    directive_tail = _symbol_factor(identity_space=identity_space, prime_space=prime_space, name="directive_tail")
    iri_dot = _symbol_factor(identity_space=identity_space, prime_space=prime_space, name="iri_dot")
    triple_name_name = _symbol_factor(identity_space=identity_space, prime_space=prime_space, name="triple_name_name")
    triple_name_iri = _symbol_factor(identity_space=identity_space, prime_space=prime_space, name="triple_name_iri")
    triple_name_string = _symbol_factor(identity_space=identity_space, prime_space=prime_space, name="triple_name_string")
    triple_name_number = _symbol_factor(identity_space=identity_space, prime_space=prime_space, name="triple_name_number")
    triple_iri_name = _symbol_factor(identity_space=identity_space, prime_space=prime_space, name="triple_iri_name")
    triple_iri_iri = _symbol_factor(identity_space=identity_space, prime_space=prime_space, name="triple_iri_iri")
    triple_iri_string = _symbol_factor(identity_space=identity_space, prime_space=prime_space, name="triple_iri_string")
    triple_iri_number = _symbol_factor(identity_space=identity_space, prime_space=prime_space, name="triple_iri_number")
    pred_name_name = _symbol_factor(identity_space=identity_space, prime_space=prime_space, name="pred_name_name")
    pred_name_iri = _symbol_factor(identity_space=identity_space, prime_space=prime_space, name="pred_name_iri")
    pred_name_string = _symbol_factor(identity_space=identity_space, prime_space=prime_space, name="pred_name_string")
    pred_name_number = _symbol_factor(identity_space=identity_space, prime_space=prime_space, name="pred_name_number")
    object_dot_name = _symbol_factor(identity_space=identity_space, prime_space=prime_space, name="object_dot_name")
    object_dot_iri = _symbol_factor(identity_space=identity_space, prime_space=prime_space, name="object_dot_iri")
    object_dot_string = _symbol_factor(identity_space=identity_space, prime_space=prime_space, name="object_dot_string")
    object_dot_number = _symbol_factor(identity_space=identity_space, prime_space=prime_space, name="object_dot_number")

    prefix = _terminal_factor(identity_space=identity_space, prime_space=prime_space, terminal_name="PREFIX")
    name = _terminal_factor(identity_space=identity_space, prime_space=prime_space, terminal_name="NAME")
    iri = _terminal_factor(identity_space=identity_space, prime_space=prime_space, terminal_name="IRI")
    dot = _terminal_factor(identity_space=identity_space, prime_space=prime_space, terminal_name="DOT")
    string = _terminal_factor(identity_space=identity_space, prime_space=prime_space, terminal_name="STRING")
    number = _terminal_factor(identity_space=identity_space, prime_space=prime_space, terminal_name="NUMBER")

    return (
        EarleyRule(head=document, rhs=(directive, directive)),
        EarleyRule(head=directive, rhs=(prefix, directive_tail)),
        EarleyRule(head=directive_tail, rhs=(name, iri_dot)),
        EarleyRule(head=iri_dot, rhs=(iri, dot)),
        EarleyRule(head=triple_name_name, rhs=(name, pred_name_name)),
        EarleyRule(head=triple_name_iri, rhs=(name, pred_name_iri)),
        EarleyRule(head=triple_name_string, rhs=(name, pred_name_string)),
        EarleyRule(head=triple_name_number, rhs=(name, pred_name_number)),
        EarleyRule(head=triple_iri_name, rhs=(iri, pred_name_name)),
        EarleyRule(head=triple_iri_iri, rhs=(iri, pred_name_iri)),
        EarleyRule(head=triple_iri_string, rhs=(iri, pred_name_string)),
        EarleyRule(head=triple_iri_number, rhs=(iri, pred_name_number)),
        EarleyRule(head=pred_name_name, rhs=(name, object_dot_name)),
        EarleyRule(head=pred_name_iri, rhs=(name, object_dot_iri)),
        EarleyRule(head=pred_name_string, rhs=(name, object_dot_string)),
        EarleyRule(head=pred_name_number, rhs=(name, object_dot_number)),
        EarleyRule(head=object_dot_name, rhs=(name, dot)),
        EarleyRule(head=object_dot_iri, rhs=(iri, dot)),
        EarleyRule(head=object_dot_string, rhs=(string, dot)),
        EarleyRule(head=object_dot_number, rhs=(number, dot)),
    )


def run_kernel_turtle_earley_skeleton(
    *,
    root: Path = _REPO_ROOT,
    rel_paths: tuple[Path, ...] = _DEFAULT_TTL_PATHS,
    max_tokens: int | None = 256,
) -> EarleyParseResult:
    lexemes = load_kernel_ttl_lexemes(root=root, rel_paths=rel_paths)
    if max_tokens is not None:
        lexemes = lexemes[:max_tokens]

    with deadline_scope(Deadline.from_timeout_ms(30_000)):
        with deadline_clock_scope(MonotonicClock()):
            registry = PrimeRegistry()
            prime_space = _PrimeFactorSpace(registry=registry)
            observer_id = prime_space.attach()
            try:
                identity_space = GlobalIdentitySpace(
                    allocator=PrimeIdentityAdapter(registry=registry)
                )
                grammar = build_turtle_skeleton_grammar(
                    identity_space=identity_space,
                    prime_space=prime_space,
                )
                grammar_index: dict[PrimeFactor, tuple[EarleyRule, ...]] = {}
                for rule in grammar:
                    grammar_index[rule.head] = (*grammar_index.get(rule.head, ()), rule)

                builder = _HistoryBoundEarleyBuilder.create(
                    identity_space=identity_space,
                    prime_space=prime_space,
                    source_paths=tuple(path.as_posix() for path in rel_paths),
                    lexemes=lexemes,
                    grammar=grammar,
                )
                token_objects = builder.token_objects()
                start_blueprint = builder._make_blueprint(
                    identified=f"{grammar[0].head.token}:0:0:0",
                    rule=grammar[0],
                    window=HistoryBoundEarleyWindow(
                        stream_id=builder.lineage.lexeme_stream_id,
                        start=0,
                        stop=0,
                        cursor=0,
                    ),
                    operator_token=f"seed:{grammar[0].head.token}",
                    left=None,
                    right=None,
                    lexeme=None,
                )
                builder.add_blueprint(
                    column_index=0,
                    blueprint=start_blueprint,
                    action_token="seed",
                )

                for column_index in range(len(builder.columns)):
                    agenda_index = 0
                    while agenda_index < len(builder.columns[column_index]):
                        current = builder.columns[column_index][agenda_index]
                        agenda_index += 1
                        next_symbol = current.next_symbol
                        if next_symbol is None:
                            for candidate in tuple(builder.columns[current.origin]):
                                if candidate.next_symbol != current.rule.head:
                                    continue
                                builder.add_blueprint(
                                    column_index=column_index,
                                    blueprint=builder._make_blueprint(
                                        identified=(
                                            f"{candidate.rule.head.token}:"
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
                                            f"complete:{candidate.rule.head.token}:"
                                            f"{candidate.dot + 1}:{column_index}"
                                        ),
                                        left=candidate,
                                        right=current,
                                        lexeme=None,
                                    ),
                                    action_token="complete",
                                )
                            continue
                        if next_symbol in grammar_index:
                            for predicted in grammar_index[next_symbol]:
                                builder.add_blueprint(
                                    column_index=column_index,
                                    blueprint=builder._make_blueprint(
                                        identified=(
                                            f"{predicted.head.token}:"
                                            f"{column_index}:{column_index}:0"
                                        ),
                                        rule=predicted,
                                        window=HistoryBoundEarleyWindow(
                                            stream_id=builder.lineage.lexeme_stream_id,
                                            start=column_index,
                                            stop=column_index,
                                            cursor=0,
                                        ),
                                        operator_token=f"predict:{predicted.head.token}:{column_index}",
                                        left=current,
                                        right=None,
                                        lexeme=None,
                                    ),
                                    action_token="predict",
                                )
                            continue
                        if column_index >= len(token_objects):
                            continue
                        token_object = token_objects[column_index]
                        if token_object.rule.head != next_symbol:
                            continue
                        builder.add_blueprint(
                            column_index=column_index + 1,
                            blueprint=builder._make_blueprint(
                                identified=(
                                    f"{current.rule.head.token}:"
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
                                    f"scan:{current.rule.head.token}:"
                                    f"{next_symbol.prime}:{column_index}"
                                ),
                                left=current,
                                right=token_object,
                                lexeme=None,
                            ),
                            action_token="scan",
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
