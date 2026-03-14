from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from enum import StrEnum
from importlib import import_module
import re
from typing import cast

from gabion.analysis.core.prime_identity_adapter import PrimeIdentityAdapter
from gabion.analysis.core.type_fingerprints import PrimeRegistry
from gabion.analysis.foundation.timeout_context import (
    Deadline,
    deadline_clock_scope,
    deadline_scope,
)
from gabion.deadline_clock import MonotonicClock
from gabion.json_types import JSONValue


class FrontmatterParseMode(StrEnum):
    ABSENT = "absent"
    YAML = "yaml"
    YAML_PARSE_FAILED = "yaml_parse_failed"


class FrontmatterIdentityNamespace(StrEnum):
    DOCUMENT = "frontmatter_ingress.document"
    FIELD = "frontmatter_ingress.field"
    DECOMPOSITION = "frontmatter_ingress.decomposition"


class FrontmatterDecompositionKind(StrEnum):
    CANONICAL = "canonical"
    SOURCE_PATH = "source_path"
    SOURCE_PATH_SEGMENT = "source_path_segment"
    ITEM_KIND = "item_kind"
    ITEM_KEY = "item_key"
    ITEM_KEY_SEGMENT = "item_key_segment"


class FrontmatterDecompositionRelationKind(StrEnum):
    CANONICAL_OF = "canonical_of"
    ALTERNATE_OF = "alternate_of"
    EQUIVALENT_UNDER = "equivalent_under"
    DERIVED_FROM = "derived_from"


@dataclass(frozen=True, order=True)
class _PrimeBackedIdentity:
    atom_id: int
    namespace: FrontmatterIdentityNamespace = field(compare=False)
    token: str = field(compare=False)

    def wire(self) -> str:
        return self.token

    def __str__(self) -> str:
        return self.token


@dataclass(frozen=True, order=True)
class FrontmatterDecompositionIdentity:
    canonical: _PrimeBackedIdentity
    decomposition_kind: FrontmatterDecompositionKind = field(compare=False)
    origin: _PrimeBackedIdentity = field(compare=False)
    label: str = field(compare=False, default="")
    part_index: int = field(compare=False, default=-1)

    def wire(self) -> str:
        return self.canonical.token

    def __str__(self) -> str:
        return self.label or self.canonical.token


@dataclass(frozen=True)
class FrontmatterDecompositionRelation:
    relation_kind: FrontmatterDecompositionRelationKind
    source: FrontmatterDecompositionIdentity
    target: FrontmatterDecompositionIdentity
    rationale: str = ""


@dataclass(frozen=True, order=True)
class FrontmatterIdentity:
    canonical: _PrimeBackedIdentity
    source_path: str = field(compare=False)
    item_kind: str = field(compare=False, default="")
    item_key: str = field(compare=False, default="")
    label: str = field(compare=False, default="")
    decompositions: tuple[FrontmatterDecompositionIdentity, ...] = field(
        default=(),
        compare=False,
    )
    relations: tuple[FrontmatterDecompositionRelation, ...] = field(
        default=(),
        compare=False,
    )

    def wire(self) -> str:
        return self.canonical.token

    def __str__(self) -> str:
        return self.label or self.canonical.token


@dataclass
class FrontmatterIdentitySpace:
    registry: PrimeRegistry = field(default_factory=PrimeRegistry)
    _adapter: PrimeIdentityAdapter = field(init=False, repr=False)
    _identity_cache: dict[
        tuple[FrontmatterIdentityNamespace, str],
        _PrimeBackedIdentity,
    ] = field(init=False, repr=False, default_factory=dict)
    _decomposition_cache: dict[
        _PrimeBackedIdentity,
        tuple[
            tuple[FrontmatterDecompositionIdentity, ...],
            tuple[FrontmatterDecompositionRelation, ...],
        ],
    ] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._adapter = PrimeIdentityAdapter(registry=self.registry)

    @staticmethod
    def _structural_segments(value: str) -> tuple[str, ...]:
        parts = [part for part in re.split(r"[:/._-]+", value.strip()) if part]
        seen: set[str] = set()
        ordered: list[str] = []
        for part in parts:
            if part in seen:
                continue
            seen.add(part)
            ordered.append(part)
        return tuple(ordered)

    def _identity(
        self,
        *,
        namespace: FrontmatterIdentityNamespace,
        token: str,
    ) -> _PrimeBackedIdentity:
        normalized = str(token).strip()
        cache_key = (namespace, normalized)
        cached = self._identity_cache.get(cache_key)
        if cached is not None:
            return cached
        with ExitStack() as scope:
            scope.enter_context(deadline_clock_scope(MonotonicClock()))
            scope.enter_context(deadline_scope(Deadline.from_timeout_ms(60_000)))
            atom_id = self._adapter.get_or_assign(
                namespace=namespace.value,
                token=normalized,
            )
        created = _PrimeBackedIdentity(
            atom_id=atom_id,
            namespace=namespace,
            token=normalized,
        )
        self._identity_cache[cache_key] = created
        return created

    def _decomposition_identity(
        self,
        *,
        origin: _PrimeBackedIdentity,
        decomposition_kind: FrontmatterDecompositionKind,
        label: str,
        part_index: int = -1,
    ) -> FrontmatterDecompositionIdentity:
        token = f"{origin.token}|{decomposition_kind.value}|{part_index}|{label.strip()}"
        canonical = self._identity(
            namespace=FrontmatterIdentityNamespace.DECOMPOSITION,
            token=token,
        )
        return FrontmatterDecompositionIdentity(
            canonical=canonical,
            decomposition_kind=decomposition_kind,
            origin=origin,
            label=label.strip(),
            part_index=part_index,
        )

    def _decomposition_bundle(
        self,
        *,
        origin: _PrimeBackedIdentity,
        source_path: str,
        item_kind: str,
        item_key: str,
    ) -> tuple[
        tuple[FrontmatterDecompositionIdentity, ...],
        tuple[FrontmatterDecompositionRelation, ...],
    ]:
        cached = self._decomposition_cache.get(origin)
        if cached is not None:
            return cached
        canonical = self._decomposition_identity(
            origin=origin,
            decomposition_kind=FrontmatterDecompositionKind.CANONICAL,
            label=origin.token,
        )
        source_path_view = self._decomposition_identity(
            origin=origin,
            decomposition_kind=FrontmatterDecompositionKind.SOURCE_PATH,
            label=source_path,
        )
        source_path_segments = tuple(
            self._decomposition_identity(
                origin=origin,
                decomposition_kind=FrontmatterDecompositionKind.SOURCE_PATH_SEGMENT,
                label=segment,
                part_index=index,
            )
            for index, segment in enumerate(self._structural_segments(source_path))
        )
        item_kind_view = self._decomposition_identity(
            origin=origin,
            decomposition_kind=FrontmatterDecompositionKind.ITEM_KIND,
            label=item_kind,
        )
        item_key_view = self._decomposition_identity(
            origin=origin,
            decomposition_kind=FrontmatterDecompositionKind.ITEM_KEY,
            label=item_key,
        )
        item_key_segments = tuple(
            self._decomposition_identity(
                origin=origin,
                decomposition_kind=FrontmatterDecompositionKind.ITEM_KEY_SEGMENT,
                label=segment,
                part_index=index,
            )
            for index, segment in enumerate(self._structural_segments(item_key))
        )
        decompositions = (
            canonical,
            source_path_view,
            item_kind_view,
            item_key_view,
            *source_path_segments,
            *item_key_segments,
        )
        relations: list[FrontmatterDecompositionRelation] = [
            FrontmatterDecompositionRelation(
                relation_kind=FrontmatterDecompositionRelationKind.CANONICAL_OF,
                source=canonical,
                target=source_path_view,
                rationale="source_path_view",
            ),
            FrontmatterDecompositionRelation(
                relation_kind=FrontmatterDecompositionRelationKind.ALTERNATE_OF,
                source=source_path_view,
                target=canonical,
                rationale="source_path_view",
            ),
            FrontmatterDecompositionRelation(
                relation_kind=FrontmatterDecompositionRelationKind.EQUIVALENT_UNDER,
                source=source_path_view,
                target=canonical,
                rationale="source_path",
            ),
            FrontmatterDecompositionRelation(
                relation_kind=FrontmatterDecompositionRelationKind.CANONICAL_OF,
                source=canonical,
                target=item_kind_view,
                rationale="item_kind_view",
            ),
            FrontmatterDecompositionRelation(
                relation_kind=FrontmatterDecompositionRelationKind.ALTERNATE_OF,
                source=item_kind_view,
                target=canonical,
                rationale="item_kind_view",
            ),
            FrontmatterDecompositionRelation(
                relation_kind=FrontmatterDecompositionRelationKind.EQUIVALENT_UNDER,
                source=item_kind_view,
                target=canonical,
                rationale="item_kind",
            ),
            FrontmatterDecompositionRelation(
                relation_kind=FrontmatterDecompositionRelationKind.CANONICAL_OF,
                source=canonical,
                target=item_key_view,
                rationale="item_key_view",
            ),
            FrontmatterDecompositionRelation(
                relation_kind=FrontmatterDecompositionRelationKind.ALTERNATE_OF,
                source=item_key_view,
                target=canonical,
                rationale="item_key_view",
            ),
            FrontmatterDecompositionRelation(
                relation_kind=FrontmatterDecompositionRelationKind.EQUIVALENT_UNDER,
                source=item_key_view,
                target=canonical,
                rationale="item_key",
            ),
        ]
        relations.extend(
            FrontmatterDecompositionRelation(
                relation_kind=FrontmatterDecompositionRelationKind.DERIVED_FROM,
                source=item,
                target=source_path_view,
                rationale="source_path_segment",
            )
            for item in source_path_segments
        )
        relations.extend(
            FrontmatterDecompositionRelation(
                relation_kind=FrontmatterDecompositionRelationKind.DERIVED_FROM,
                source=item,
                target=item_key_view,
                rationale="item_key_segment",
            )
            for item in item_key_segments
        )
        bundle = (decompositions, tuple(relations))
        self._decomposition_cache[origin] = bundle
        return bundle

    def document_id(
        self,
        *,
        source_path: str,
        mode: FrontmatterParseMode,
        label: str = "",
    ) -> FrontmatterIdentity:
        canonical = self._identity(
            namespace=FrontmatterIdentityNamespace.DOCUMENT,
            token=f"frontmatter:{source_path}:document:{mode.value}",
        )
        decompositions, relations = self._decomposition_bundle(
            origin=canonical,
            source_path=source_path,
            item_kind="document",
            item_key=mode.value,
        )
        return FrontmatterIdentity(
            canonical=canonical,
            source_path=source_path,
            item_kind="document",
            item_key=mode.value,
            label=label,
            decompositions=decompositions,
            relations=relations,
        )

    def field_id(
        self,
        *,
        source_path: str,
        field_name: str,
        label: str = "",
    ) -> FrontmatterIdentity:
        canonical = self._identity(
            namespace=FrontmatterIdentityNamespace.FIELD,
            token=f"frontmatter:{source_path}:field:{field_name}",
        )
        decompositions, relations = self._decomposition_bundle(
            origin=canonical,
            source_path=source_path,
            item_kind="field",
            item_key=field_name,
        )
        return FrontmatterIdentity(
            canonical=canonical,
            source_path=source_path,
            item_kind="field",
            item_key=field_name,
            label=label,
            decompositions=decompositions,
            relations=relations,
        )


@dataclass(frozen=True)
class FrontmatterFieldCarrier:
    identity: FrontmatterIdentity
    field_name: str
    value: JSONValue

    def __str__(self) -> str:
        return self.field_name


@dataclass(frozen=True)
class FrontmatterDocumentCarrier:
    identity: FrontmatterIdentity
    source_path: str
    mode: FrontmatterParseMode
    detail: str | None
    payload: dict[str, JSONValue]
    body: str
    raw_frontmatter_lines: tuple[str, ...]
    parser_available: bool
    fields: tuple[FrontmatterFieldCarrier, ...]

    @property
    def has_closed_block(self) -> bool:
        return self.mode is not FrontmatterParseMode.ABSENT or bool(
            self.raw_frontmatter_lines
        )

    @property
    def is_unterminated(self) -> bool:
        return self.mode is FrontmatterParseMode.ABSENT and (
            self.detail == "unterminated YAML frontmatter"
        )

    def payload_mapping(self) -> dict[str, JSONValue]:
        return dict(self.payload)

    def __str__(self) -> str:
        return self.source_path or self.mode.value


def _yaml_module():
    return import_module("yaml")


def _normalized_source_path(source_path: str) -> str:
    text = str(source_path).strip()
    return text or "<anonymous_frontmatter>"


def _error_summary(exc: Exception) -> str:
    message = str(exc).strip()
    return message.splitlines()[0] if message else type(exc).__name__


def scan_frontmatter_lines(lines: Sequence[str]) -> tuple[tuple[str, ...], int] | None:
    if not lines or str(lines[0]).strip() != "---":
        return None
    for index in range(1, len(lines)):
        if str(lines[index]).strip() == "---":
            return tuple(str(line) for line in lines[1:index]), index
    return None


def parse_frontmatter_document(
    text: str,
    *,
    source_path: str = "",
    identities: FrontmatterIdentitySpace | None = None,
) -> FrontmatterDocumentCarrier:
    identity_space = identities or FrontmatterIdentitySpace()
    normalized_source_path = _normalized_source_path(source_path)
    lines = text.splitlines()
    block = scan_frontmatter_lines(lines)
    if block is None:
        detail = "unterminated YAML frontmatter" if text.startswith("---\n") else None
        mode = FrontmatterParseMode.ABSENT
        return FrontmatterDocumentCarrier(
            identity=identity_space.document_id(
                source_path=normalized_source_path,
                mode=mode,
                label=source_path.strip() or mode.value,
            ),
            source_path=source_path.strip(),
            mode=mode,
            detail=detail,
            payload={},
            body=text,
            raw_frontmatter_lines=(),
            parser_available=True,
            fields=(),
        )
    raw_frontmatter_lines, end = block
    body = "\n".join(lines[end + 1 :])
    try:
        yaml = _yaml_module()
    except ImportError:
        mode = FrontmatterParseMode.YAML_PARSE_FAILED
        return FrontmatterDocumentCarrier(
            identity=identity_space.document_id(
                source_path=normalized_source_path,
                mode=mode,
                label=source_path.strip() or mode.value,
            ),
            source_path=source_path.strip(),
            mode=mode,
            detail="pyyaml unavailable",
            payload={},
            body=body,
            raw_frontmatter_lines=raw_frontmatter_lines,
            parser_available=False,
            fields=(),
        )
    try:
        parsed = yaml.safe_load("\n".join(raw_frontmatter_lines))
    except Exception as exc:
        mode = FrontmatterParseMode.YAML_PARSE_FAILED
        return FrontmatterDocumentCarrier(
            identity=identity_space.document_id(
                source_path=normalized_source_path,
                mode=mode,
                label=source_path.strip() or mode.value,
            ),
            source_path=source_path.strip(),
            mode=mode,
            detail=_error_summary(exc),
            payload={},
            body=body,
            raw_frontmatter_lines=raw_frontmatter_lines,
            parser_available=True,
            fields=(),
        )
    if parsed is None:
        payload: dict[str, JSONValue] = {}
    elif not isinstance(parsed, Mapping):
        mode = FrontmatterParseMode.YAML_PARSE_FAILED
        return FrontmatterDocumentCarrier(
            identity=identity_space.document_id(
                source_path=normalized_source_path,
                mode=mode,
                label=source_path.strip() or mode.value,
            ),
            source_path=source_path.strip(),
            mode=mode,
            detail=f"frontmatter root must be a mapping (got {type(parsed).__name__})",
            payload={},
            body=body,
            raw_frontmatter_lines=raw_frontmatter_lines,
            parser_available=True,
            fields=(),
        )
    else:
        payload = {
            str(key): cast(JSONValue, value)
            for key, value in cast(Mapping[object, object], parsed).items()
        }
    mode = FrontmatterParseMode.YAML
    fields = tuple(
        FrontmatterFieldCarrier(
            identity=identity_space.field_id(
                source_path=normalized_source_path,
                field_name=field_name,
                label=field_name,
            ),
            field_name=field_name,
            value=value,
        )
        for field_name, value in payload.items()
    )
    return FrontmatterDocumentCarrier(
        identity=identity_space.document_id(
            source_path=normalized_source_path,
            mode=mode,
            label=source_path.strip() or mode.value,
        ),
        source_path=source_path.strip(),
        mode=mode,
        detail=None,
        payload=payload,
        body=body,
        raw_frontmatter_lines=raw_frontmatter_lines,
        parser_available=True,
        fields=fields,
    )
