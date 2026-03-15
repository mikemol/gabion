from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import argparse
import json
import re
from typing import Iterable

from gabion.analysis.aspf.aspf_core import AspfOneCell, BasisZeroCell
from gabion.analysis.core.prime_identity_adapter import PrimeIdentityAdapter
from gabion.analysis.core.type_fingerprints import PrimeRegistry
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
class EarleyRule:
    lhs: str
    rhs: tuple[str, ...]
    rule_id: str


@dataclass(frozen=True)
class EarleyItemCarrier:
    item_id: str
    projection: IdentityProjection
    one_cell: AspfOneCell


@dataclass(frozen=True)
class EarleyItem:
    rule: EarleyRule
    dot: int
    origin: int
    end: int
    carrier: EarleyItemCarrier

    @property
    def next_symbol(self) -> str | None:
        if self.dot >= len(self.rule.rhs):
            return None
        return self.rule.rhs[self.dot]

    @property
    def is_complete(self) -> bool:
        return self.dot >= len(self.rule.rhs)


@dataclass
class EarleyChartColumn:
    index: int
    items_by_id: dict[str, EarleyItem] = field(default_factory=dict)

    def add(self, item: EarleyItem) -> bool:
        if item.carrier.item_id in self.items_by_id:
            return False
        self.items_by_id[item.carrier.item_id] = item
        return True

    def items(self) -> tuple[EarleyItem, ...]:
        return tuple(self.items_by_id.values())


@dataclass(frozen=True)
class EarleyParseResult:
    source_paths: tuple[str, ...]
    tokens: tuple[TurtleLexeme, ...]
    chart: tuple[EarleyChartColumn, ...]
    grammar: tuple[EarleyRule, ...]

    @property
    def item_count(self) -> int:
        return sum(len(column.items_by_id) for column in self.chart)

    def as_summary(self) -> dict[str, object]:
        return {
            "source_paths": list(self.source_paths),
            "token_count": len(self.tokens),
            "chart_column_count": len(self.chart),
            "item_count": self.item_count,
            "completed_start_items": [
                item.carrier.item_id
                for item in self.chart[-1].items()
                if item.rule.lhs == "document'" and item.is_complete and item.origin == 0
            ],
            "columns": [
                {
                    "index": column.index,
                    "item_count": len(column.items_by_id),
                    "sample_items": [
                        {
                            "item_id": item.carrier.item_id,
                            "lhs": item.rule.lhs,
                            "rhs": list(item.rule.rhs),
                            "dot": item.dot,
                            "origin": item.origin,
                            "end": item.end,
                            "prime_product": item.carrier.projection.prime_product,
                            "basis_path_atoms": list(item.carrier.projection.basis_path.atoms),
                            "aspf_basis_path": list(item.carrier.one_cell.basis_path),
                        }
                        for item in list(column.items())[:5]
                    ],
                }
                for column in self.chart
            ],
        }


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


def _item_key_tokens(item: EarleyItem) -> tuple[str, ...]:
    return (
        "ttl_earley_item",
        item.rule.rule_id,
        str(item.dot),
        str(item.origin),
        str(item.end),
    )


def _item_id_from_projection(projection: IdentityProjection) -> str:
    atoms = ".".join(str(atom) for atom in projection.basis_path.atoms)
    return f"{projection.basis_path.namespace}:{atoms}"


def _make_item(
    *,
    rule: EarleyRule,
    dot: int,
    origin: int,
    end: int,
    identity_space: GlobalIdentitySpace,
) -> EarleyItem:
    provisional = EarleyItem(
        rule=rule,
        dot=dot,
        origin=origin,
        end=end,
        carrier=EarleyItemCarrier(
            item_id="",
            projection=identity_space.project(
                path=identity_space.intern_path(namespace="ttl_earley_item", tokens=("bootstrap",))
            ),
            one_cell=AspfOneCell(
                source=BasisZeroCell("chart:0"),
                target=BasisZeroCell("chart:0"),
                representative="bootstrap",
                basis_path=("bootstrap",),
            ),
        ),
    )
    path = identity_space.intern_path(
        namespace="ttl_earley_item",
        tokens=_item_key_tokens(provisional),
    )
    projection = identity_space.project(path=path)
    one_cell = AspfOneCell(
        source=BasisZeroCell(f"chart:{origin}"),
        target=BasisZeroCell(f"chart:{end}"),
        representative=projection.digest_alias,
        basis_path=_item_key_tokens(provisional),
    )
    return EarleyItem(
        rule=rule,
        dot=dot,
        origin=origin,
        end=end,
        carrier=EarleyItemCarrier(
            item_id=_item_id_from_projection(projection),
            projection=projection,
            one_cell=one_cell,
        ),
    )


def run_kernel_turtle_earley_skeleton(
    *,
    root: Path = _REPO_ROOT,
    rel_paths: tuple[Path, ...] = _DEFAULT_TTL_PATHS,
    max_tokens: int | None = 256,
) -> EarleyParseResult:
    tokens = load_kernel_ttl_lexemes(root=root, rel_paths=rel_paths)
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    grammar = build_turtle_skeleton_grammar()
    grammar_index: dict[str, tuple[EarleyRule, ...]] = {}
    for rule in grammar:
        grammar_index[rule.lhs] = (*grammar_index.get(rule.lhs, ()), rule)

    with deadline_scope(Deadline.from_timeout_ms(30_000)):
        with deadline_clock_scope(MonotonicClock()):
            identity_space = GlobalIdentitySpace(
                allocator=PrimeIdentityAdapter(registry=PrimeRegistry())
            )
            chart = tuple(EarleyChartColumn(index=i) for i in range(len(tokens) + 1))
            start_item = _make_item(
                rule=grammar[0],
                dot=0,
                origin=0,
                end=0,
                identity_space=identity_space,
            )
            chart[0].add(start_item)

            for column_index, column in enumerate(chart):
                agenda_index = 0
                while agenda_index < len(column.items_by_id):
                    current = list(column.items_by_id.values())[agenda_index]
                    agenda_index += 1
                    next_symbol = current.next_symbol
                    if next_symbol is None:
                        for candidate in chart[current.origin].items():
                            if candidate.next_symbol == current.rule.lhs:
                                chart[column_index].add(
                                    _make_item(
                                        rule=candidate.rule,
                                        dot=candidate.dot + 1,
                                        origin=candidate.origin,
                                        end=column_index,
                                        identity_space=identity_space,
                                    )
                                )
                        continue
                    if next_symbol == _EPSILON:
                        chart[column_index].add(
                            _make_item(
                                rule=current.rule,
                                dot=current.dot + 1,
                                origin=current.origin,
                                end=column_index,
                                identity_space=identity_space,
                            )
                        )
                        continue
                    if _is_nonterminal(next_symbol, grammar_index):
                        for predicted in grammar_index[next_symbol]:
                            chart[column_index].add(
                                _make_item(
                                    rule=predicted,
                                    dot=0,
                                    origin=column_index,
                                    end=column_index,
                                    identity_space=identity_space,
                                )
                            )
                        continue
                    if column_index >= len(tokens):
                        continue
                    if tokens[column_index].terminal == next_symbol:
                        chart[column_index + 1].add(
                            _make_item(
                                rule=current.rule,
                                dot=current.dot + 1,
                                origin=current.origin,
                                end=column_index + 1,
                                identity_space=identity_space,
                            )
                        )

    return EarleyParseResult(
        source_paths=tuple(path.as_posix() for path in rel_paths),
        tokens=tokens,
        chart=chart,
        grammar=grammar,
    )


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Skeleton Earley parser over the repo's kernel TTL files.",
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
