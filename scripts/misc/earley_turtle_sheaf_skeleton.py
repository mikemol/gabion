from __future__ import annotations

"""Sheaf-theoretic Earley skeleton over the repo's kernel TTL files.

The construction is intentionally split into three layers:

- Arena presheaf: local staged patches over lexeme sites.
- Scanner/sheafification: glue compatible local covers into scanned sheaves.
- Predict/complete: treat scanned sheaf sections as a new presheaf and build
  larger parsed sections by binary gluing.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
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
class PresheafSite:
    site_id: PrimeFactor
    start: int
    stop: int
    projection: IdentityProjection

    def as_islice(self, lexemes: Iterable[TurtleLexeme]) -> Iterator[TurtleLexeme]:
        return itertools.islice(lexemes, self.start, self.stop)


@dataclass(frozen=True)
class ArenaPresheafPatch:
    patch_id: PrimeFactor
    site: PresheafSite
    symbol: PrimeFactor
    lexeme: TurtleLexeme
    projection: IdentityProjection


@dataclass(frozen=True)
class LocalSection:
    section_id: PrimeFactor
    site: PresheafSite
    rule: EarleyRule
    member_ids: tuple[PrimeFactor, ...]
    projection: IdentityProjection


@dataclass(frozen=True)
class ScannedSheaf:
    sheaf_id: PrimeFactor
    glued_site: PresheafSite
    cover_site_ids: tuple[PrimeFactor, ...]
    rule: EarleyRule
    local_section: LocalSection
    prime_factor: PrimeFactor
    one_cell: AspfOneCell


@dataclass(frozen=True)
class ScannedPresheafSection:
    section_id: PrimeFactor
    site: PresheafSite
    symbol: PrimeFactor
    sheaf: ScannedSheaf
    projection: IdentityProjection


@dataclass(frozen=True)
class PredictedSection:
    prediction_id: PrimeFactor
    site: PresheafSite
    rule: EarleyRule
    left_section_id: PrimeFactor
    expected_symbol: PrimeFactor
    projection: IdentityProjection


@dataclass(frozen=True)
class CompletedSection:
    completion_id: PrimeFactor
    site: PresheafSite
    symbol: PrimeFactor
    rule: EarleyRule
    left_section_id: PrimeFactor
    right_section_id: PrimeFactor
    projection: IdentityProjection


@dataclass(frozen=True)
class SheafEarleyResult:
    source_paths: tuple[str, ...]
    lexemes: tuple[TurtleLexeme, ...]
    arena_presheaf: tuple[ArenaPresheafPatch, ...]
    scanned_sheaves: tuple[ScannedSheaf, ...]
    section_presheaf: tuple[ScannedPresheafSection, ...]
    predictions: tuple[PredictedSection, ...]
    completions: tuple[CompletedSection, ...]
    grammar: tuple[EarleyRule, ...]

    def as_summary(self) -> dict[str, object]:
        return {
            "source_paths": list(self.source_paths),
            "lexeme_count": len(self.lexemes),
            "arena_patch_count": len(self.arena_presheaf),
            "scanned_sheaf_count": len(self.scanned_sheaves),
            "section_presheaf_count": len(self.section_presheaf),
            "prediction_count": len(self.predictions),
            "completion_count": len(self.completions),
            "sample_sheaves": [
                {
                    "sheaf_prime": sheaf.sheaf_id.prime,
                    "head": sheaf.rule.head.token,
                    "rhs": [symbol.token for symbol in sheaf.rule.rhs],
                    "site": [sheaf.glued_site.start, sheaf.glued_site.stop],
                }
                for sheaf in self.scanned_sheaves[:5]
            ],
            "sample_completions": [
                {
                    "completion_prime": completion.completion_id.prime,
                    "symbol": completion.symbol.token,
                    "site": [completion.site.start, completion.site.stop],
                }
                for completion in self.completions[:5]
            ],
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


def _symbol_factor(
    *,
    identity_space: GlobalIdentitySpace,
    prime_space: _PrimeFactorSpace,
    name: str,
) -> PrimeFactor:
    return prime_space.intern_factor(
        identity_space=identity_space,
        namespace="ttl_sheaf_symbol",
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


def _id_factor(
    *,
    identity_space: GlobalIdentitySpace,
    prime_space: _PrimeFactorSpace,
    namespace: str,
    token: str,
) -> PrimeFactor:
    return prime_space.intern_factor(
        identity_space=identity_space,
        namespace=namespace,
        token=token,
    )


def _projection(
    *,
    identity_space: GlobalIdentitySpace,
    namespace: str,
    tokens: tuple[str, ...],
) -> IdentityProjection:
    return identity_space.project(
        path=identity_space.intern_path(namespace=namespace, tokens=tokens)
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


def build_sheaf_grammar(
    *,
    identity_space: GlobalIdentitySpace,
    prime_space: _PrimeFactorSpace,
) -> tuple[EarleyRule, ...]:
    document = _symbol_factor(identity_space=identity_space, prime_space=prime_space, name="document'")
    directive = _symbol_factor(identity_space=identity_space, prime_space=prime_space, name="directive")
    directive_tail = _symbol_factor(identity_space=identity_space, prime_space=prime_space, name="directive_tail")
    iri_dot = _symbol_factor(identity_space=identity_space, prime_space=prime_space, name="iri_dot")
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
        EarleyRule(head=pred_name_name, rhs=(name, object_dot_name)),
        EarleyRule(head=pred_name_iri, rhs=(name, object_dot_iri)),
        EarleyRule(head=pred_name_string, rhs=(name, object_dot_string)),
        EarleyRule(head=pred_name_number, rhs=(name, object_dot_number)),
        EarleyRule(head=object_dot_name, rhs=(name, dot)),
        EarleyRule(head=object_dot_iri, rhs=(iri, dot)),
        EarleyRule(head=object_dot_string, rhs=(string, dot)),
        EarleyRule(head=object_dot_number, rhs=(number, dot)),
    )


def _make_site(
    *,
    identity_space: GlobalIdentitySpace,
    prime_space: _PrimeFactorSpace,
    start: int,
    stop: int,
) -> PresheafSite:
    projection = _projection(
        identity_space=identity_space,
        namespace="ttl_sheaf_site",
        tokens=(f"start:{start}", f"stop:{stop}"),
    )
    site_id = _id_factor(
        identity_space=identity_space,
        prime_space=prime_space,
        namespace="ttl_sheaf_site_id",
        token=projection.digest_alias,
    )
    return PresheafSite(site_id=site_id, start=start, stop=stop, projection=projection)


def _glued_site(
    *,
    identity_space: GlobalIdentitySpace,
    prime_space: _PrimeFactorSpace,
    left: PresheafSite,
    right: PresheafSite,
) -> PresheafSite:
    return _make_site(
        identity_space=identity_space,
        prime_space=prime_space,
        start=left.start,
        stop=right.stop,
    )


def _section_symbol(section: ScannedPresheafSection | CompletedSection) -> PrimeFactor:
    return section.symbol


def _section_site(section: ScannedPresheafSection | CompletedSection) -> PresheafSite:
    return section.site


def _section_id(section: ScannedPresheafSection | CompletedSection) -> PrimeFactor:
    if isinstance(section, ScannedPresheafSection):
        return section.section_id
    return section.completion_id


def run_turtle_sheaf_earley_skeleton(
    *,
    root: Path = _REPO_ROOT,
    rel_paths: tuple[Path, ...] = _DEFAULT_TTL_PATHS,
    max_tokens: int | None = 256,
) -> SheafEarleyResult:
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
                grammar = build_sheaf_grammar(
                    identity_space=identity_space,
                    prime_space=prime_space,
                )
                binary_rules_by_left: dict[PrimeFactor, tuple[EarleyRule, ...]] = {}
                for rule in grammar:
                    left = rule.rhs[0]
                    binary_rules_by_left[left] = (*binary_rules_by_left.get(left, ()), rule)

                arena_patches: list[ArenaPresheafPatch] = []
                for index, lexeme in enumerate(lexemes):
                    site = _make_site(
                        identity_space=identity_space,
                        prime_space=prime_space,
                        start=index,
                        stop=index + 1,
                    )
                    symbol = _terminal_factor(
                        identity_space=identity_space,
                        prime_space=prime_space,
                        terminal_name=lexeme.terminal_name,
                    )
                    projection = _projection(
                        identity_space=identity_space,
                        namespace="ttl_arena_patch",
                        tokens=(
                            f"site:{site.site_id.prime}",
                            f"symbol:{symbol.prime}",
                            f"path:{lexeme.rel_path}",
                            f"offset:{lexeme.offset}",
                        ),
                    )
                    patch_id = _id_factor(
                        identity_space=identity_space,
                        prime_space=prime_space,
                        namespace="ttl_arena_patch_id",
                        token=projection.digest_alias,
                    )
                    arena_patches.append(
                        ArenaPresheafPatch(
                            patch_id=patch_id,
                            site=site,
                            symbol=symbol,
                            lexeme=lexeme,
                            projection=projection,
                        )
                    )

                scanned_sheaves: list[ScannedSheaf] = []
                section_presheaf: list[ScannedPresheafSection] = []
                section_keys: set[tuple[int, int, int]] = set()

                def _intern_scanned_section(
                    *,
                    site: PresheafSite,
                    cover_site_ids: tuple[PrimeFactor, ...],
                    rule: EarleyRule,
                    member_ids: tuple[PrimeFactor, ...],
                    prime_factor: PrimeFactor,
                ) -> None:
                    key = (rule.head.prime, site.start, site.stop)
                    if key in section_keys:
                        return
                    section_keys.add(key)
                    local_projection = _projection(
                        identity_space=identity_space,
                        namespace="ttl_scanned_local_section",
                        tokens=(
                            f"site:{site.site_id.prime}",
                            f"head:{rule.head.prime}",
                            *(f"member:{member_id.prime}" for member_id in member_ids),
                        ),
                    )
                    local_section = LocalSection(
                        section_id=_id_factor(
                            identity_space=identity_space,
                            prime_space=prime_space,
                            namespace="ttl_scanned_local_section_id",
                            token=local_projection.digest_alias,
                        ),
                        site=site,
                        rule=rule,
                        member_ids=member_ids,
                        projection=local_projection,
                    )
                    one_cell = AspfOneCell(
                        source=BasisZeroCell(f"site:{site.start}"),
                        target=BasisZeroCell(f"site:{site.stop}"),
                        representative=local_projection.digest_alias,
                        basis_path=(
                            f"site:{site.site_id.prime}",
                            f"head:{rule.head.prime}",
                            *(f"member:{member_id.prime}" for member_id in member_ids),
                        ),
                    )
                    sheaf_projection = _projection(
                        identity_space=identity_space,
                        namespace="ttl_scanned_sheaf",
                        tokens=(
                            f"section:{local_section.section_id.prime}",
                            *(f"cover:{member_id.prime}" for member_id in member_ids),
                        ),
                    )
                    sheaf = ScannedSheaf(
                        sheaf_id=_id_factor(
                            identity_space=identity_space,
                            prime_space=prime_space,
                            namespace="ttl_scanned_sheaf_id",
                            token=sheaf_projection.digest_alias,
                        ),
                        glued_site=site,
                        cover_site_ids=cover_site_ids,
                        rule=rule,
                        local_section=local_section,
                        prime_factor=prime_factor,
                        one_cell=one_cell,
                    )
                    scanned_sheaves.append(sheaf)
                    section_projection = _projection(
                        identity_space=identity_space,
                        namespace="ttl_scanned_section_presheaf",
                        tokens=(
                            f"sheaf:{sheaf.sheaf_id.prime}",
                            f"symbol:{rule.head.prime}",
                        ),
                    )
                    section_presheaf.append(
                        ScannedPresheafSection(
                            section_id=_id_factor(
                                identity_space=identity_space,
                                prime_space=prime_space,
                                namespace="ttl_scanned_section_presheaf_id",
                                token=section_projection.digest_alias,
                            ),
                            site=site,
                            symbol=rule.head,
                            sheaf=sheaf,
                            projection=section_projection,
                        )
                    )

                for patch in arena_patches:
                    terminal_rule = EarleyRule(head=patch.symbol, rhs=(patch.symbol,))
                    _intern_scanned_section(
                        site=patch.site,
                        cover_site_ids=(patch.site.site_id,),
                        rule=terminal_rule,
                        member_ids=(patch.patch_id,),
                        prime_factor=_id_factor(
                            identity_space=identity_space,
                            prime_space=prime_space,
                            namespace="ttl_scanner_operator",
                            token=f"scan:{patch.symbol.prime}:{patch.patch_id.prime}",
                        ),
                    )

                predictions: list[PredictedSection] = []
                completions: list[CompletedSection] = []
                prediction_keys: set[tuple[int, int, int, int]] = set()
                completion_keys: set[tuple[int, int, int]] = set()
                available_sections: list[ScannedPresheafSection | CompletedSection] = list(section_presheaf)
                growth = True
                while growth:
                    growth = False
                    current_sections = tuple(available_sections)
                    for left in current_sections:
                        left_symbol = _section_symbol(left)
                        left_site = _section_site(left)
                        for rule in binary_rules_by_left.get(left_symbol, ()):
                            prediction_key = (
                                rule.head.prime,
                                left_site.start,
                                left_site.stop,
                                rule.rhs[1].prime,
                            )
                            if prediction_key not in prediction_keys:
                                prediction_keys.add(prediction_key)
                                prediction_projection = _projection(
                                    identity_space=identity_space,
                                    namespace="ttl_predicted_section",
                                    tokens=(
                                        f"head:{rule.head.prime}",
                                        f"start:{left_site.start}",
                                        f"stop:{left_site.stop}",
                                        f"expect:{rule.rhs[1].prime}",
                                        f"left:{_section_id(left).prime}",
                                    ),
                                )
                                predictions.append(
                                    PredictedSection(
                                        prediction_id=_id_factor(
                                            identity_space=identity_space,
                                            prime_space=prime_space,
                                            namespace="ttl_predicted_section_id",
                                            token=prediction_projection.digest_alias,
                                        ),
                                        site=left_site,
                                        rule=rule,
                                        left_section_id=_section_id(left),
                                        expected_symbol=rule.rhs[1],
                                        projection=prediction_projection,
                                    )
                                )
                            for right in current_sections:
                                right_site = _section_site(right)
                                if right_site.start != left_site.stop:
                                    continue
                                if _section_symbol(right) != rule.rhs[1]:
                                    continue
                                completion_site = _glued_site(
                                    identity_space=identity_space,
                                    prime_space=prime_space,
                                    left=left_site,
                                    right=right_site,
                                )
                                completion_key = (
                                    rule.head.prime,
                                    completion_site.start,
                                    completion_site.stop,
                                )
                                if completion_key in completion_keys:
                                    continue
                                completion_keys.add(completion_key)
                                completion_projection = _projection(
                                    identity_space=identity_space,
                                    namespace="ttl_completed_section",
                                    tokens=(
                                        f"head:{rule.head.prime}",
                                        f"start:{completion_site.start}",
                                        f"stop:{completion_site.stop}",
                                        f"left:{_section_id(left).prime}",
                                        f"right:{_section_id(right).prime}",
                                    ),
                                )
                                completion = CompletedSection(
                                    completion_id=_id_factor(
                                        identity_space=identity_space,
                                        prime_space=prime_space,
                                        namespace="ttl_completed_section_id",
                                        token=completion_projection.digest_alias,
                                    ),
                                    site=completion_site,
                                    symbol=rule.head,
                                    rule=rule,
                                    left_section_id=_section_id(left),
                                    right_section_id=_section_id(right),
                                    projection=completion_projection,
                                )
                                completions.append(completion)
                                available_sections.append(completion)
                                growth = True
            finally:
                registry.unregister_assignment_observer(observer_id)

    return SheafEarleyResult(
        source_paths=tuple(path.as_posix() for path in rel_paths),
        lexemes=lexemes,
        arena_presheaf=tuple(arena_patches),
        scanned_sheaves=tuple(scanned_sheaves),
        section_presheaf=tuple(section_presheaf),
        predictions=tuple(predictions),
        completions=tuple(completions),
        grammar=grammar,
    )


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sheaf-theoretic Earley skeleton over the repo's kernel TTL files.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Limit scanned TTL lexemes before running the sheaf skeleton.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the parse summary as JSON.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_turtle_sheaf_earley_skeleton(max_tokens=args.max_tokens)
    summary = result.as_summary()
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
