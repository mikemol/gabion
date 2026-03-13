from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
import json
from pathlib import Path
from typing import Any, cast

from gabion.analysis.aspf.aspf_lattice_algebra import ReplayableStream
from gabion.analysis.foundation.json_types import JSONValue

from gabion.tooling.policy_substrate.policy_queue_identity import (
    encode_policy_queue_identity,
)


class ArtifactUnitKind(StrEnum):
    DOCUMENT = "document"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    BULLET_LIST = "bullet_list"
    LIST_ITEM = "list_item"
    TABLE = "table"
    ROW = "row"
    CELL = "cell"
    SCALAR = "scalar"
    LAZY = "lazy"


@dataclass(frozen=True)
class ArtifactSourceRef:
    rel_path: str
    qualname: str = ""
    line: int = 0
    column: int = 0


@dataclass(frozen=True)
class ArtifactColumn:
    key: str
    title: str


def _empty_stream() -> ReplayableStream[ArtifactUnit]:
    return ReplayableStream(factory=lambda: iter(()))


@dataclass(frozen=True)
class ArtifactUnit:
    kind: ArtifactUnitKind
    identity: object
    key: str = ""
    title: str = ""
    value: JSONValue | str | int | float | bool | None = None
    columns: tuple[ArtifactColumn, ...] = ()
    children: ReplayableStream["ArtifactUnit"] = field(default_factory=_empty_stream)

    def __iter__(self) -> Iterator["ArtifactUnit"]:
        return iter(self.children)


def document(
    *,
    identity: object,
    title: str = "",
    children: Callable[[], Iterator[ArtifactUnit]],
) -> ArtifactUnit:
    return ArtifactUnit(
        kind=ArtifactUnitKind.DOCUMENT,
        identity=identity,
        title=title,
        children=ReplayableStream(factory=children),
    )


def section(
    *,
    identity: object,
    key: str,
    title: str,
    children: Callable[[], Iterator[ArtifactUnit]],
) -> ArtifactUnit:
    return ArtifactUnit(
        kind=ArtifactUnitKind.SECTION,
        identity=identity,
        key=key,
        title=title,
        children=ReplayableStream(factory=children),
    )


def paragraph(
    *,
    identity: object,
    key: str = "",
    title: str = "",
    value: str,
) -> ArtifactUnit:
    return ArtifactUnit(
        kind=ArtifactUnitKind.PARAGRAPH,
        identity=identity,
        key=key,
        title=title,
        value=value,
    )


def scalar(
    *,
    identity: object,
    key: str,
    title: str,
    value: JSONValue | str | int | float | bool | None,
) -> ArtifactUnit:
    return ArtifactUnit(
        kind=ArtifactUnitKind.SCALAR,
        identity=identity,
        key=key,
        title=title,
        value=value,
    )


def bullet_list(
    *,
    identity: object,
    key: str,
    title: str = "",
    children: Callable[[], Iterator[ArtifactUnit]],
) -> ArtifactUnit:
    return ArtifactUnit(
        kind=ArtifactUnitKind.BULLET_LIST,
        identity=identity,
        key=key,
        title=title,
        children=ReplayableStream(factory=children),
    )


def list_item(
    *,
    identity: object,
    title: str = "",
    value: JSONValue | str | int | float | bool | None = None,
    children: Callable[[], Iterator[ArtifactUnit]] | None = None,
) -> ArtifactUnit:
    return ArtifactUnit(
        kind=ArtifactUnitKind.LIST_ITEM,
        identity=identity,
        title=title,
        value=value,
        children=ReplayableStream(factory=children) if children is not None else _empty_stream(),
    )


def table(
    *,
    identity: object,
    key: str,
    title: str,
    columns: Iterable[ArtifactColumn],
    children: Callable[[], Iterator[ArtifactUnit]],
) -> ArtifactUnit:
    return ArtifactUnit(
        kind=ArtifactUnitKind.TABLE,
        identity=identity,
        key=key,
        title=title,
        columns=tuple(columns),
        children=ReplayableStream(factory=children),
    )


def row(
    *,
    identity: object,
    children: Callable[[], Iterator[ArtifactUnit]],
) -> ArtifactUnit:
    return ArtifactUnit(
        kind=ArtifactUnitKind.ROW,
        identity=identity,
        children=ReplayableStream(factory=children),
    )


def cell(
    *,
    identity: object,
    key: str,
    title: str,
    value: JSONValue | str | int | float | bool | None,
) -> ArtifactUnit:
    return ArtifactUnit(
        kind=ArtifactUnitKind.CELL,
        identity=identity,
        key=key,
        title=title,
        value=value,
    )


def lazy(
    *,
    identity: object,
    children: Callable[[], Iterator[ArtifactUnit]],
) -> ArtifactUnit:
    return ArtifactUnit(
        kind=ArtifactUnitKind.LAZY,
        identity=identity,
        children=ReplayableStream(factory=children),
    )


def _path_ref(path: tuple[str, ...]) -> ArtifactSourceRef:
    qualname = ".".join(path) if path else "<root>"
    return ArtifactSourceRef(rel_path="<synthetic>", qualname=qualname)


def _item_title(*, index: int, value: object) -> str:
    match value:
        case Mapping() as mapping:
            for candidate in (
                "object_id",
                "queue_id",
                "subqueue_id",
                "touchpoint_id",
                "touchsite_id",
                "ring_1_scope",
                "seed_path",
                "spec_name",
                "path",
                "rule_id",
                "title",
            ):
                candidate_value = mapping.get(candidate)
                if isinstance(candidate_value, str) and candidate_value.strip():
                    return candidate_value.strip()
        case _:
            pass
    return str(index)


def _mapping_units(
    value: Mapping[str, object],
    *,
    path: tuple[str, ...],
) -> Iterator[ArtifactUnit]:
    for key, child in value.items():
        key_text = str(key)
        child_path = (*path, key_text)
        match child:
            case Mapping() as mapping:
                yield section(
                    identity=_path_ref(child_path),
                    key=key_text,
                    title=key_text,
                    children=lambda mapping=mapping, child_path=child_path: _mapping_units(
                        mapping,
                        path=child_path,
                    ),
                )
            case Sequence() as sequence if not isinstance(child, (str, bytes, bytearray)):
                yield bullet_list(
                    identity=_path_ref(child_path),
                    key=key_text,
                    title=key_text,
                    children=lambda sequence=tuple(sequence), child_path=child_path: (
                        _sequence_item_unit(
                            item,
                            index=index,
                            path=(*child_path, str(index)),
                        )
                        for index, item in enumerate(sequence)
                    ),
                )
            case _:
                yield scalar(
                    identity=_path_ref(child_path),
                    key=key_text,
                    title=key_text,
                    value=child,
                )


def _sequence_item_unit(
    value: object,
    *,
    index: int,
    path: tuple[str, ...],
) -> ArtifactUnit:
    match value:
        case Mapping() as mapping:
            title = _item_title(index=index, value=value)
            return list_item(
                identity=_path_ref(path),
                title=title,
                children=lambda mapping=mapping, path=path: _mapping_units(
                    mapping,
                    path=path,
                ),
            )
        case Sequence() as sequence if not isinstance(value, (str, bytes, bytearray)):
            return list_item(
                identity=_path_ref(path),
                title=str(index),
                children=lambda sequence=tuple(sequence), path=path: (
                    _sequence_item_unit(
                        item,
                        index=child_index,
                        path=(*path, str(child_index)),
                    )
                    for child_index, item in enumerate(sequence)
                ),
            )
        case _:
            return list_item(
                identity=_path_ref(path),
                title=str(index),
                value=cast(JSONValue | str | int | float | bool | None, value),
            )


def mapping_document(
    *,
    identity: object,
    title: str,
    payload: Mapping[str, object],
) -> ArtifactUnit:
    return document(
        identity=identity,
        title=title,
        children=lambda payload=payload: _mapping_units(payload, path=()),
    )


def _stringify(value: object) -> str:
    match value:
        case bool() as flag:
            return "true" if flag else "false"
        case None:
            return "<none>"
        case str() as text:
            return text
        case _:
            return str(value)


def _markdown_scalar_line(unit: ArtifactUnit) -> str:
    return f"- {unit.title}: `{_stringify(unit.value)}`"


def _table_row_mapping(unit: ArtifactUnit) -> Mapping[str, object]:
    return {
        child.key: child.value
        for child in unit
        if child.kind is ArtifactUnitKind.CELL and child.key
    }


def iter_markdown_lines(
    unit: ArtifactUnit,
    *,
    heading_level: int = 1,
) -> Iterator[str]:
    match unit.kind:
        case ArtifactUnitKind.DOCUMENT:
            if unit.title:
                yield f"{'#' * heading_level} {unit.title}"
                yield ""
            first = True
            for child in unit:
                if not first:
                    pass
                yield from iter_markdown_lines(child, heading_level=heading_level + 1)
                first = False
        case ArtifactUnitKind.SECTION:
            yield f"{'#' * heading_level} {unit.title}"
            yield ""
            for child in unit:
                yield from iter_markdown_lines(child, heading_level=heading_level + 1)
        case ArtifactUnitKind.PARAGRAPH:
            yield _stringify(unit.value)
            yield ""
        case ArtifactUnitKind.SCALAR:
            yield _markdown_scalar_line(unit)
        case ArtifactUnitKind.BULLET_LIST:
            if unit.title:
                yield f"{'#' * heading_level} {unit.title}"
                yield ""
            for child in unit:
                match child.kind:
                    case ArtifactUnitKind.LIST_ITEM if child.value is not None:
                        yield f"- `{_stringify(child.value)}`"
                    case ArtifactUnitKind.LIST_ITEM:
                        bullet = child.title or encode_policy_queue_identity(child.identity)
                        yield f"- {bullet}"
                        for grandchild in child:
                            for line in iter_markdown_lines(
                                grandchild,
                                heading_level=heading_level + 1,
                            ):
                                if line:
                                    yield f"  {line}"
                                else:
                                    yield ""
                    case _:
                        continue
        case ArtifactUnitKind.TABLE:
            if unit.title:
                yield f"{'#' * heading_level} {unit.title}"
                yield ""
            yield "| " + " | ".join(column.title for column in unit.columns) + " |"
            yield "| " + " | ".join("---" for _ in unit.columns) + " |"
            for child in unit:
                row_mapping = _table_row_mapping(child)
                yield (
                    "| "
                    + " | ".join(
                        _stringify(row_mapping.get(column.key, "")) for column in unit.columns
                    )
                    + " |"
                )
        case ArtifactUnitKind.LAZY:
            for child in unit:
                yield from iter_markdown_lines(child, heading_level=heading_level)
        case _:
            return


def _json_value(unit: ArtifactUnit) -> object:
    match unit.kind:
        case ArtifactUnitKind.DOCUMENT | ArtifactUnitKind.SECTION:
            return {
                child.key: _json_value(child)
                for child in unit
                if child.key
            }
        case ArtifactUnitKind.PARAGRAPH | ArtifactUnitKind.SCALAR | ArtifactUnitKind.CELL:
            return unit.value
        case ArtifactUnitKind.BULLET_LIST:
            return [_json_value(child) for child in unit]
        case ArtifactUnitKind.LIST_ITEM:
            children = tuple(iter(unit))
            if children:
                return {
                    child.key: _json_value(child)
                    for child in children
                    if child.key
                }
            return unit.value
        case ArtifactUnitKind.TABLE:
            return [_json_value(child) for child in unit]
        case ArtifactUnitKind.ROW:
            return {
                child.key: _json_value(child)
                for child in unit
                if child.key
            }
        case ArtifactUnitKind.LAZY:
            children = tuple(iter(unit))
            if not children:
                return None
            if len(children) == 1:
                return _json_value(children[0])
            return [_json_value(child) for child in children]
        case _:
            return None


def render_markdown(unit: ArtifactUnit) -> str:
    lines = tuple(iter_markdown_lines(unit))
    return "\n".join(lines).rstrip() + "\n"


def render_json_value(unit: ArtifactUnit) -> object:
    return _json_value(unit)


def write_markdown(path: Path, unit: ArtifactUnit) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for line in iter_markdown_lines(unit):
            handle.write(line)
            handle.write("\n")
    temp_path.replace(path)


def write_json(path: Path, unit: ArtifactUnit) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(render_json_value(unit), handle, indent=2)
        handle.write("\n")
    temp_path.replace(path)


__all__ = [
    "ArtifactColumn",
    "ArtifactSourceRef",
    "ArtifactUnit",
    "ArtifactUnitKind",
    "bullet_list",
    "cell",
    "document",
    "iter_markdown_lines",
    "lazy",
    "list_item",
    "mapping_document",
    "paragraph",
    "render_json_value",
    "render_markdown",
    "row",
    "scalar",
    "section",
    "table",
    "write_json",
    "write_markdown",
]
